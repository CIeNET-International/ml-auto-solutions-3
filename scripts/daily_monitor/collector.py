# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builds the daily digest from the Airflow REST API and Cloud Logging.

Collects the same data as the Jetski skill's collect_local.py. The digest
itself comes from dag_monitor.build_digest, so both have the same schema.
"""

import datetime
import logging
from typing import Any, Optional
from urllib import parse

from dags.common import vm_resource
from scripts.daily_monitor import clients
from scripts.daily_monitor import config
from xlml.utils import dag_monitor

LOG_SLACK = datetime.timedelta(minutes=2)


def _parse(value: str) -> datetime.datetime:
  return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> datetime.datetime:
  return datetime.datetime.now(datetime.timezone.utc)


class Collector:
  """Collects one environment's digest for one UTC day."""

  def __init__(
      self,
      env: config.Environment,
      airflow: clients.AirflowClient,
      logs: clients.LogClient,
  ):
    self.env = env
    self.airflow = airflow
    self.logs = logs
    self.errors = []

  def collect(self, report_date: str) -> dict[str, Any]:
    window = dag_monitor.resolve_window(report_date=report_date)
    metas = sorted(
        (
            d
            for d in self.airflow.get("/dags", limit=500)["dags"]
            if dag_monitor.in_folder(d["fileloc"], config.MONITORED_FOLDER)
            and d["dag_id"] != config.SELF_DAG_ID
        ),
        key=lambda d: d["dag_id"],
    )
    clusters, dags = {}, []
    for meta in metas:
      dag_id = meta["dag_id"]
      try:
        dags.append(self._collect_dag(meta, window, clusters))
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Collecting %s failed", dag_id)
        self.errors.append({"dag_id": dag_id, "error": repr(e)[:500]})
        dags.append({"dag_id": dag_id, "runs": [], "schedule": None})
    logging.info(
        "Collected %d DAGs: %d Airflow API calls, %s",
        len(dags),
        self.airflow.calls,
        self.logs.stats,
    )
    return dag_monitor.build_digest(
        self.env.composer,
        config.MONITORED_FOLDER,
        window,
        sorted(clusters.values(), key=lambda c: c["key"]),
        dags,
        report_date=report_date,
        collector_errors=self.errors,
    )

  def _collect_dag(
      self,
      meta: dict[str, Any],
      window: tuple[datetime.datetime, datetime.datetime],
      clusters: dict[str, Any],
  ) -> dict[str, Any]:
    dag_id = meta["dag_id"]
    start, end = window
    token = meta["file_token"]
    source = self.airflow.get(f"/dagSources/{token}")["content"]
    cluster_keys = dag_monitor.find_cluster_refs(source)
    cluster = None
    for key in cluster_keys:
      gke = getattr(vm_resource.GkeClusters, key, None)
      if gke is None:
        continue
      entry = clusters.setdefault(
          key,
          {
              "key": key,
              "name": gke.name,
              "project": gke.project,
              "location": dag_monitor.zone_to_location(gke.zone),
              "dags": 0,
          },
      )
      entry["dags"] += 1
      cluster = cluster or entry
    schedule = meta.get("schedule_interval")
    if isinstance(schedule, dict):
      schedule = schedule.get("value")
    fileloc = meta["fileloc"].replace("\\", "/")
    folder = config.MONITORED_FOLDER
    tags = sorted(t["name"] for t in meta.get("tags") or [])
    history = self.airflow.get(
        f"/dags/{dag_id}/dagRuns",
        order_by="-start_date",
        start_date_lte=dag_monitor.rfc3339(end),
        limit=config.HISTORY_RUNS,
    )["dag_runs"]
    in_window = self.airflow.get(
        f"/dags/{dag_id}/dagRuns",
        order_by="start_date",
        start_date_gte=dag_monitor.rfc3339(start),
        start_date_lte=dag_monitor.rfc3339(end),
        limit=config.MAX_RUNS_IN_WINDOW,
    )["dag_runs"]
    runs = [
        {
            "run_id": r["dag_run_id"],
            "state": r["state"],
            "start_date": r["start_date"],
            "end_date": r.get("end_date"),
            "airflow_url": dag_monitor.airflow_grid_url(
                self.env.airflow_url, dag_id, r["dag_run_id"]
            ),
        }
        for r in in_window
        if r.get("start_date") and _parse(r["start_date"]) < end
    ]
    latest = dag_monitor.latest_failed_run(runs)
    if latest is not None:
      try:
        self._collect_failure(
            dag_id, latest, cluster, dag_monitor.wants_probes(tags)
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Collecting logs for %s failed", dag_id)
        self.errors.append({"dag_id": dag_id, "error": f"logs: {e!r}"[:500]})
    return {
        "dag_id": dag_id,
        "file": f"{folder}/" + fileloc.split(f"/{folder}/", 1)[-1],
        "schedule": schedule,
        "owners": [
            o.strip()
            for o in meta.get("owners") or []
            if o.strip() and o.strip() != "airflow"
        ],
        "tags": tags,
        "clusters": cluster_keys,
        "history": dag_monitor.classify_history([r["state"] for r in history]),
        "runs": runs,
    }

  def _read(self, project: str, log_filter: str, limit: int) -> list[dict]:
    return self.logs.read(project, log_filter, limit)

  def _collect_failure(
      self,
      dag_id: str,
      run: dict[str, Any],
      cluster: Optional[dict[str, Any]],
      probe: bool,
  ) -> None:
    """Adds failed_tasks, log_scope, workloads, pod_errors, pod_tail to `run`.

    With `probe` (dag_monitor.PROBE_TAG DAGs), also each failed task's
    log_tail and the run's checkpoint events in `probes`.
    """
    env = self.env
    run_q = parse.quote(run["run_id"], safe="")
    tis = self.airflow.get(
        f"/dags/{dag_id}/dagRuns/{run_q}/taskInstances", limit=500
    )["task_instances"]
    run_start = _parse(run["start_date"]) - LOG_SLACK
    run_end = (
        _parse(run["end_date"]) if run.get("end_date") else _now()
    ) + LOG_SLACK
    workloads = {}  # generate_workload_id task -> workload, oldest first.
    for ti in sorted(tis, key=lambda t: t.get("start_date") or ""):
      if (
          ti["task_id"].endswith("generate_workload_id")
          and ti["state"] == "success"
      ):
        task_q = parse.quote(ti["task_id"], safe="")
        xcom = self.airflow.get(
            f"/dags/{dag_id}/dagRuns/{run_q}/taskInstances/{task_q}"
            "/xcomEntries/return_value"
        )
        value = str(xcom.get("value") or "").strip("'\"")
        if value:
          workloads[ti["task_id"]] = value
    workload_ids = list(workloads.values())
    failed_tasks, targets = [], []
    for ti in sorted(
        (t for t in tis if t["state"] == "failed"),
        key=lambda t: t.get("start_date") or "",
    ):
      task_id = ti["task_id"]
      t_start = (
          _parse(ti["start_date"]) - LOG_SLACK
          if ti.get("start_date")
          else run_start
      )
      t_end = (
          _parse(ti["end_date"]) + LOG_SLACK if ti.get("end_date") else run_end
      )
      worker_filter = dag_monitor.airflow_worker_filter(
          env.project,
          env.composer,
          dag_id,
          task_id,
          t_start,
          t_end,
          try_number=ti.get("try_number") or None,
      )
      lines = [
          clients.entry_text(e)
          for e in self._read(
              env.project,
              worker_filter + " AND severity>=ERROR",
              config.AIRFLOW_ERROR_LINES_PER_TASK,
          )
      ]
      exception = dag_monitor.extract_exception(lines)
      workload = dag_monitor.workload_for_task(task_id, workloads)
      targets.append(workload)
      task = {
          "task_id": task_id,
          "try_number": ti.get("try_number"),
          "exception": exception,
          "signature": dag_monitor.failure_signature(
              task_id, exception, workload_ids
          ),
          "workload": workload,
      }
      if probe:
        task["log_tail"] = [
            clients.entry_text(e)
            for e in self._read(
                env.project, worker_filter, dag_monitor.TASK_LOG_LINES
            )
        ]
      failed_tasks.append(task)
    if failed_tasks:
      run["airflow_url"] = dag_monitor.airflow_grid_url(
          env.airflow_url, dag_id, run["run_id"], failed_tasks[0]["task_id"]
      )
    run["failed_tasks"] = failed_tasks
    run["log_scope"] = dag_monitor.log_scope(
        env.project,
        env.composer,
        dag_id,
        [t["task_id"] for t in failed_tasks],
        cluster,
        workload_ids,
        run_start,
        run_end,
    )
    run["workloads"] = []
    if cluster is None or not workload_ids:
      run["pod_errors"] = []
      return

    def pod_filter(workload: str, extra: Optional[str] = None) -> str:
      return dag_monitor.pod_log_filter(
          cluster["project"],
          cluster["location"],
          cluster["name"],
          workload,
          run_start,
          run_end,
          extra=extra,
      )

    for workload in workload_ids:
      run["workloads"].append(
          {
              "id": workload,
              "logs_url": dag_monitor.logs_explorer_url(
                  cluster["project"], pod_filter(workload), run_start, run_end
              ),
          }
      )
    wanted = [w for w in targets if w]
    if not wanted or len(wanted) < len(targets):
      wanted = workload_ids  # Ambiguous: query every workload of the run.
    wanted = list(dict.fromkeys(wanted))
    errors_extra = dag_monitor.message_matches(dag_monitor.POD_ERROR_PATTERN)
    entries = []
    for workload in wanted:
      for e in self._read(
          cluster["project"],
          pod_filter(workload, errors_extra),
          config.POD_LINES_PER_WORKLOAD,
      ):
        entries.append(
            {
                "text": clients.entry_text(e),
                "pod": clients.entry_pod(e),
                "ts": e.get("timestamp", ""),
            }
        )
    run["pod_errors"] = dag_monitor.dedupe(entries)
    # The most recently started workload is the one that ran last.
    tail = self._read(
        cluster["project"],
        pod_filter(
            wanted[-1], f'resource.labels.pod_name=~"{config.TAIL_POD_PATTERN}"'
        ),
        config.POD_TAIL_LINES,
    )
    if tail:
      run["pod_tail"] = {
          "pod": clients.entry_pod(tail[-1]),
          "lines": [clients.entry_text(e) for e in tail],
      }
    if probe:
      events = []
      for workload in workload_ids:
        found = self._read(
            cluster["project"],
            pod_filter(
                workload,
                dag_monitor.checkpoint_events_extra(config.TAIL_POD_PATTERN),
            ),
            dag_monitor.CHECKPOINT_EVENT_LINES,
        )
        events.append(
            {
                "workload": workload,
                "pod": clients.entry_pod(found[-1]) if found else "",
                "lines": [clients.entry_text(e) for e in found],
            }
        )
      run["probes"] = {"checkpoint_events": events}
