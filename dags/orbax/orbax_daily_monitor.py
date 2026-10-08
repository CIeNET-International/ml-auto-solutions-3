"""Daily digest of Orbax DAG health.

Reads one UTC day of Orbax DAG runs from the Airflow DB, pulls the error lines
and pod logs of each DAG's latest failed run from Cloud Logging, and writes one
JSON digest per day to the Composer bucket. The digest is the input for the
daily report.

In an AUTO_POST_ENVS environment only (erniechang-test), three more tasks
triage the digest with Gemini on Vertex AI, render the report, and can post it
to the test Chat space through an incoming webhook. Posting is off by default:
new Chat webhooks at Google need a policy exception, so the Jetski skill posts
the DAG's draft after review instead. Prod and dev never create these tasks.

Window: a scheduled run covers the UTC day before it. A manual run takes
{"report_date": "YYYY-MM-DD"} or {"start": "...Z", "end": "...Z"} from its
conf; with no conf it covers yesterday (UTC).
"""

import datetime
import json
import logging
import os
import time
from typing import Any, Optional

from airflow import models
from airflow.decorators import task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

from dags import composer_env
from xlml.utils import dag_monitor

SELF_DAG_ID = "orbax_daily_monitor"
MONITORED_FOLDER = "dags/orbax"
SCHEDULE = "30 0 * * *" if composer_env.is_prod_env() else None
# Composer mounts gs://<bucket>/data at /home/airflow/gcs/data.
DIGEST_DIR = "/home/airflow/gcs/data/dag_monitor/orbax"

HISTORY_RUNS = 7
MAX_RUNS_IN_WINDOW = 100
AIRFLOW_ERROR_LINES_PER_TASK = 50
POD_LINES_PER_WORKLOAD = 300
POD_TAIL_LINES = 100
# GKE logs every stderr line as ERROR, so match error text, not severity.
POD_ERROR_EXTRA = (
    'textPayload=~"Traceback|Fatal Python error|Error:|Exception:|'
    'FAILED|Segmentation fault|core dumped|Killed|OOM"'
)
TAIL_POD_PATTERN = "slice-job-0-0"
LOG_SLACK = datetime.timedelta(minutes=2)
MIN_QUERY_GAP_S = 1.1  # Stay under 60 Logging read requests/min.

# Decision 7, AUTO_POST_ENVS only: Gemini triage and the webhook post.
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
GEMINI_MODEL = "gemini-2.5-pro"
# Pro thinks before it answers, so it needs more time than Flash.
GEMINI_TIMEOUT_S = 120
GEMINI_ATTEMPTS = 3  # The first call plus 2 retries.
WEBHOOK_SECRET = "orbax-daily-monitor-chat-webhook-{environment}"
WEBHOOK_TIMEOUT_S = 30
REPORT_TITLE = "[TEST {environment}] Testing Orbax Daily Monitoring"
ENVIRONMENT = os.environ.get(composer_env.COMPOSER_ENVIRONMENT, "")
AUTO_POST = ENVIRONMENT in dag_monitor.AUTO_POST_ENVS


def _state(value: Any) -> Optional[str]:
  """Airflow state enum or string -> plain string."""
  if value is None:
    return None
  return str(getattr(value, "value", value))


def _iso(value: Optional[datetime.datetime]) -> Optional[str]:
  """Aware datetime -> ISO 8601 UTC string, as the Airflow REST API returns."""
  if value is None:
    return None
  return value.astimezone(datetime.timezone.utc).isoformat()


def _parse(value: str) -> datetime.datetime:
  return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


class _LogReader:
  """Cloud Logging reads, one at a time, with retries on quota errors."""

  def __init__(self):
    self._clients = {}
    self._last_query = 0.0
    self.stats = {"log_queries": 0, "log_lines": 0}

  def read(self, project: str, log_filter: str, limit: int) -> list[Any]:
    """Returns up to `limit` of the newest matching entries, oldest first."""
    # pylint: disable=import-outside-toplevel
    from google.api_core import exceptions
    from google.cloud import logging as logging_api

    if project not in self._clients:
      self._clients[project] = logging_api.Client(project=project)
    client = self._clients[project]
    for attempt in range(4):
      wait = MIN_QUERY_GAP_S - (time.monotonic() - self._last_query)
      if wait > 0:
        time.sleep(wait)
      self._last_query = time.monotonic()
      self.stats["log_queries"] += 1
      try:
        entries = list(
            client.list_entries(
                filter_=log_filter,
                order_by=logging_api.DESCENDING,
                max_results=limit,
                page_size=limit,
            )
        )
      except exceptions.ResourceExhausted:
        time.sleep(15 * (attempt + 1))
        continue
      self.stats["log_lines"] += len(entries)
      return entries[::-1]
    raise RuntimeError("Cloud Logging quota: retries exhausted")


def _entry_text(entry: Any) -> str:
  payload = entry.payload
  if isinstance(payload, str):
    return payload.rstrip()
  if isinstance(payload, dict):
    if payload.get("message"):
      return str(payload["message"]).rstrip()
    return json.dumps(payload, default=str)[:500]
  return "" if payload is None else str(payload)[:500]


def _entry_pod(entry: Any) -> str:
  resource = getattr(entry, "resource", None)
  labels = getattr(resource, "labels", None) or {}
  return labels.get("pod_name", "")


def _airflow_base_url() -> str:
  """The Airflow web UI URL of this Composer environment."""
  # pylint: disable=import-outside-toplevel
  import google.auth
  from google.auth.transport.requests import AuthorizedSession

  credentials, project = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  url = (
      "https://composer.googleapis.com/v1/"
      f"projects/{project}/locations/"
      f"{os.environ[composer_env.COMPOSER_LOCATION]}/environments/"
      f"{os.environ[composer_env.COMPOSER_ENVIRONMENT]}"
  )
  response = AuthorizedSession(credentials).get(url, timeout=30)
  response.raise_for_status()
  return response.json()["config"]["airflowUri"]


def _composer_project() -> str:
  import google.auth  # pylint: disable=import-outside-toplevel

  _, project = google.auth.default()
  return project


@task
def get_window() -> dict[str, str]:
  """Resolves the UTC [start, end) window and the report date."""
  # pylint: disable=import-outside-toplevel
  from airflow.utils.types import DagRunType

  context = get_current_context()
  dag_run = context["dag_run"]
  # The trigger form puts params in conf; scheduled runs get the defaults.
  conf = {**context["params"], **(dag_run.conf or {})}
  if conf.get("start") or conf.get("end"):
    start, end = dag_monitor.resolve_window(
        start=conf.get("start"), end=conf.get("end")
    )
    report_date = start.strftime("%Y-%m-%d")
  else:
    if conf.get("report_date"):
      report_date = conf["report_date"]
    elif dag_run.run_type == DagRunType.SCHEDULED:
      report_date = context["data_interval_start"].strftime("%Y-%m-%d")
    else:
      yesterday = datetime.datetime.now(
          datetime.timezone.utc
      ) - datetime.timedelta(days=1)
      report_date = yesterday.strftime("%Y-%m-%d")
    start, end = dag_monitor.resolve_window(report_date=report_date)
  window = {
      "start": dag_monitor.rfc3339(start),
      "end": dag_monitor.rfc3339(end),
      "report_date": report_date,
  }
  logging.info("Window: %s", window)
  return window


@task
def build_registry() -> list[dict[str, Any]]:
  """Finds the active DAGs in dags/orbax and the GKE cluster each one uses."""
  # pylint: disable=import-outside-toplevel
  from airflow.models import DagModel
  from airflow.utils.session import create_session
  from dags.common.vm_resource import GkeClusters

  with create_session() as session:
    models_ = (
        session.query(DagModel)
        .filter(DagModel.is_active.is_(True))
        .filter(DagModel.fileloc.like(f"%/{MONITORED_FOLDER}/%"))
        .order_by(DagModel.dag_id)
        .all()
    )
    rows = [
        {
            "dag_id": m.dag_id,
            "fileloc": m.fileloc,
            "schedule": str(m.schedule_interval)
            if m.schedule_interval
            else None,
            "owners": m.owners or "",
        }
        for m in models_
    ]
  registry = []
  for row in rows:
    dag_id = row["dag_id"]
    if dag_id == SELF_DAG_ID or not dag_monitor.in_folder(
        row["fileloc"], MONITORED_FOLDER
    ):
      continue
    fileloc = row["fileloc"].replace("\\", "/")
    entry = {
        "dag_id": dag_id,
        "file": f"{MONITORED_FOLDER}/"
        + fileloc.split(f"/{MONITORED_FOLDER}/", 1)[-1],
        "schedule": row["schedule"],
        "owners": [
            o.strip()
            for o in row["owners"].split(",")
            if o.strip() and o.strip() != "airflow"
        ],
        "clusters": [],
        "cluster": None,
    }
    try:
      with open(row["fileloc"], encoding="utf-8") as f:
        keys = dag_monitor.find_cluster_refs(f.read())
    except (OSError, SyntaxError) as e:
      entry["error"] = f"cluster lookup: {e!r}"[:500]
      keys = []
    entry["clusters"] = keys
    for key in keys:
      config = getattr(GkeClusters, key, None)
      if config is not None:
        entry["cluster"] = {
            "key": key,
            "name": config.name,
            "project": config.project,
            "location": dag_monitor.zone_to_location(config.zone),
        }
        break
    registry.append(entry)
  logging.info(
      "Monitoring %d DAGs: %s", len(registry), [r["dag_id"] for r in registry]
  )
  return registry


@task
def collect_runs(
    window: dict[str, str], registry: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  """Reads runs, history and the latest failure's task metadata per DAG."""
  # pylint: disable=import-outside-toplevel
  from airflow.models import DagRun, TaskInstance, XCom
  from airflow.models.xcom import XCOM_RETURN_KEY
  from airflow.utils.session import create_session

  start, end = _parse(window["start"]), _parse(window["end"])
  collected = []
  with create_session() as session:
    for entry in registry:
      dag_id = entry["dag_id"]
      item = {"dag_id": dag_id, "history": [], "runs": [], "failure": None}
      try:
        history = (
            session.query(DagRun.state)
            .filter(DagRun.dag_id == dag_id)
            .filter(DagRun.start_date.isnot(None))
            .filter(DagRun.start_date < end)
            .order_by(DagRun.start_date.desc())
            .limit(HISTORY_RUNS)
            .all()
        )
        item["history"] = [_state(h.state) for h in history]
        in_window = (
            session.query(DagRun)
            .filter(DagRun.dag_id == dag_id)
            .filter(DagRun.start_date >= start)
            .filter(DagRun.start_date < end)
            .order_by(DagRun.start_date)
            .limit(MAX_RUNS_IN_WINDOW)
            .all()
        )
        item["runs"] = [
            {
                "run_id": r.run_id,
                "state": _state(r.state),
                "start_date": _iso(r.start_date),
                "end_date": _iso(r.end_date),
            }
            for r in in_window
        ]
        latest = dag_monitor.latest_failed_run(item["runs"])
        if latest is None:
          collected.append(item)
          continue
        tis = (
            session.query(TaskInstance)
            .filter(TaskInstance.dag_id == dag_id)
            .filter(TaskInstance.run_id == latest["run_id"])
            .all()
        )
        workloads = {}  # generate_workload_id task -> workload, oldest first.
        failed = []
        for ti in sorted(tis, key=lambda t: _iso(t.start_date) or ""):
          state = _state(ti.state)
          if ti.task_id.endswith("generate_workload_id") and state == "success":
            value = XCom.get_one(
                run_id=latest["run_id"],
                dag_id=dag_id,
                task_id=ti.task_id,
                key=XCOM_RETURN_KEY,
                session=session,
            )
            value = str(value or "").strip("'\"")
            if value:
              workloads[ti.task_id] = value
          elif state == "failed":
            failed.append(
                {
                    "task_id": ti.task_id,
                    "try_number": ti.try_number,
                    "start_date": _iso(ti.start_date),
                    "end_date": _iso(ti.end_date),
                }
            )
        item["failure"] = {
            "run_id": latest["run_id"],
            "workloads": workloads,
            "failed_tasks": failed,
        }
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Collecting runs for %s failed", dag_id)
        item["error"] = f"runs: {e!r}"[:500]
      collected.append(item)
  return collected


def _collect_failure(
    run: dict[str, Any],
    failure: dict[str, Any],
    dag_id: str,
    cluster: Optional[dict[str, Any]],
    reader: _LogReader,
    airflow_url: str,
    composer_project: str,
) -> None:
  """Adds failed_tasks, workloads, pod_errors and pod_tail to `run`."""
  environment = os.environ.get(composer_env.COMPOSER_ENVIRONMENT, "")
  run_start = _parse(run["start_date"]) - LOG_SLACK
  run_end = (
      _parse(run["end_date"])
      if run.get("end_date")
      else datetime.datetime.now(datetime.timezone.utc)
  ) + LOG_SLACK
  workloads = failure["workloads"]
  workload_ids = list(workloads.values())
  failed_tasks = []
  targets = []
  for ti in failure["failed_tasks"]:
    task_id = ti["task_id"]
    t_start = (
        _parse(ti["start_date"]) - LOG_SLACK if ti["start_date"] else run_start
    )
    t_end = _parse(ti["end_date"]) + LOG_SLACK if ti["end_date"] else run_end
    worker_filter = dag_monitor.airflow_worker_filter(
        composer_project,
        environment,
        dag_id,
        task_id,
        t_start,
        t_end,
        try_number=ti["try_number"] or None,
    )
    lines = [
        _entry_text(e)
        for e in reader.read(
            composer_project,
            worker_filter + " AND severity>=ERROR",
            AIRFLOW_ERROR_LINES_PER_TASK,
        )
    ]
    exception = dag_monitor.extract_exception(lines)
    workload = dag_monitor.workload_for_task(task_id, workloads)
    targets.append(workload)
    failed_tasks.append(
        {
            "task_id": task_id,
            "try_number": ti["try_number"],
            "exception": exception,
            "signature": dag_monitor.failure_signature(
                task_id, exception, workload_ids
            ),
            "workload": workload,
        }
    )
  if failed_tasks:
    run["airflow_url"] = dag_monitor.airflow_grid_url(
        airflow_url, dag_id, run["run_id"], failed_tasks[0]["task_id"]
    )
  run["failed_tasks"] = failed_tasks
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
  entries = []
  for workload in wanted:
    for e in reader.read(
        cluster["project"],
        pod_filter(workload, POD_ERROR_EXTRA),
        POD_LINES_PER_WORKLOAD,
    ):
      entries.append(
          {
              "text": _entry_text(e),
              "pod": _entry_pod(e),
              "ts": dag_monitor.rfc3339(e.timestamp) if e.timestamp else "",
          }
      )
  run["pod_errors"] = dag_monitor.dedupe(entries)
  # The most recently started workload is the one that ran last.
  tail = reader.read(
      cluster["project"],
      pod_filter(wanted[-1], f'resource.labels.pod_name=~"{TAIL_POD_PATTERN}"'),
      POD_TAIL_LINES,
  )
  if tail:
    run["pod_tail"] = {
        "pod": _entry_pod(tail[-1]),
        "lines": [_entry_text(e) for e in tail],
    }


@task
def write_digest(
    window: dict[str, str],
    registry: list[dict[str, Any]],
    collected: list[dict[str, Any]],
) -> str:
  """Pulls logs for each latest failure, builds the digest, writes it."""
  errors = []
  try:
    airflow_url = _airflow_base_url()
  except Exception as e:  # pylint: disable=broad-except
    logging.exception("Airflow URL lookup failed")
    errors.append({"dag_id": None, "error": f"airflow url: {e!r}"[:500]})
    airflow_url = ""
  composer_project = _composer_project()
  reader = _LogReader()
  by_id = {c["dag_id"]: c for c in collected}
  clusters = {}
  dags = []
  for entry in registry:
    dag_id = entry["dag_id"]
    item = by_id.get(dag_id, {})
    for err in (entry.get("error"), item.get("error")):
      if err:
        errors.append({"dag_id": dag_id, "error": err})
    cluster = entry["cluster"]
    if cluster:
      summary = clusters.setdefault(cluster["key"], {**cluster, "dags": 0})
      summary["dags"] += 1
    runs = [
        {
            **r,
            "airflow_url": dag_monitor.airflow_grid_url(
                airflow_url, dag_id, r["run_id"]
            ),
        }
        for r in item.get("runs", [])
    ]
    failure = item.get("failure")
    if failure:
      run = next(r for r in runs if r["run_id"] == failure["run_id"])
      try:
        _collect_failure(
            run,
            failure,
            dag_id,
            cluster,
            reader,
            airflow_url,
            composer_project,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Collecting logs for %s failed", dag_id)
        errors.append({"dag_id": dag_id, "error": f"logs: {e!r}"[:500]})
    dags.append(
        {
            "dag_id": dag_id,
            "file": entry["file"],
            "schedule": entry["schedule"],
            "owners": entry["owners"],
            "clusters": entry["clusters"],
            "history": dag_monitor.classify_history(item.get("history", [])),
            "runs": runs,
        }
    )
  digest = dag_monitor.build_digest(
      os.environ.get(composer_env.COMPOSER_ENVIRONMENT, ""),
      MONITORED_FOLDER,
      (_parse(window["start"]), _parse(window["end"])),
      sorted(clusters.values(), key=lambda c: c["key"]),
      dags,
      report_date=window["report_date"],
      collector_errors=errors,
  )
  os.makedirs(DIGEST_DIR, exist_ok=True)
  path = os.path.join(DIGEST_DIR, f"{window['report_date']}.json")
  with open(path, "w", encoding="utf-8") as f:
    json.dump(digest, f, indent=2)
    f.write("\n")
  logging.info(
      "Wrote %s: summary=%s, errors=%d, %s",
      path,
      digest["summary"],
      len(errors),
      reader.stats,
  )
  return path


def _sibling(digest_path: str, suffix: str) -> str:
  """<dir>/<date>.json -> <dir>/<date>.<suffix>."""
  return digest_path[: -len(".json")] + f".{suffix}"


def _write_json(path: str, value: Any) -> None:
  with open(path, "w", encoding="utf-8") as f:
    json.dump(value, f, indent=2)
    f.write("\n")


def _gemini_generate(prompt: str) -> str:
  """Calls Vertex AI generateContent and returns the reply text."""
  # pylint: disable=import-outside-toplevel
  import google.auth
  from google.auth.transport.requests import AuthorizedSession

  credentials, project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
  location = os.environ[composer_env.COMPOSER_LOCATION]
  url = (
      f"https://{location}-aiplatform.googleapis.com/v1/"
      f"projects/{project}/locations/{location}/"
      f"publishers/google/models/{GEMINI_MODEL}:generateContent"
  )
  body = {
      "contents": [{"role": "user", "parts": [{"text": prompt}]}],
      "generationConfig": {
          "temperature": 0,
          "responseMimeType": "application/json",
          "responseSchema": dag_monitor.TRIAGE_SCHEMA,
      },
  }
  session = AuthorizedSession(credentials)
  error = None
  for attempt in range(GEMINI_ATTEMPTS):
    if attempt:
      time.sleep(5 * attempt)
    try:
      response = session.post(url, json=body, timeout=GEMINI_TIMEOUT_S)
    except Exception as e:  # pylint: disable=broad-except
      error = repr(e)[:300]
      continue
    if response.status_code != 200:
      error = f"HTTP {response.status_code}: {response.text[:300]}"
      continue
    candidates = response.json().get("candidates") or []
    parts = (
        (candidates[0].get("content") or {}).get("parts") if candidates else []
    )
    return "".join(
        p.get("text", "") for p in parts or [] if not p.get("thought")
    )
  raise RuntimeError(f"Vertex AI {GEMINI_MODEL} failed: {error}")


@task
def triage_with_gemini(digest_path: str) -> str:
  """Triages each latest failure with Gemini; falls back to the digest."""
  with open(digest_path, encoding="utf-8") as f:
    digest = json.load(f)
  prompt = dag_monitor.build_triage_prompt(digest)
  error = None
  if prompt is None:
    triage = dag_monitor.fallback_triage(digest)  # No failure to triage.
  else:
    try:
      triage = dag_monitor.validate_triage(_gemini_generate(prompt), digest)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Gemini triage failed; using the digest only")
      error = repr(e)[:500]
      triage = dag_monitor.fallback_triage(digest)
  triage.update({"model": GEMINI_MODEL, "error": error})
  path = _sibling(digest_path, "triage.json")
  _write_json(path, triage)
  logging.info(
      "Wrote %s: source=%s, items=%d, dropped=%d",
      path,
      triage["source"],
      len(triage["items"]),
      triage["dropped"],
  )
  return path


@task
def write_draft(digest_path: str, triage_path: str) -> str:
  """Renders the Chat report from the digest and the validated triage."""
  with open(digest_path, encoding="utf-8") as f:
    digest = json.load(f)
  with open(triage_path, encoding="utf-8") as f:
    triage = json.load(f)
  messages = dag_monitor.render_report(
      digest, triage, REPORT_TITLE.format(environment=ENVIRONMENT)
  )
  path = _sibling(digest_path, "draft.json")
  _write_json(path, {"messages": messages})
  for i, message in enumerate(messages, 1):
    logging.info("Draft message %d (%d chars):\n%s", i, len(message), message)
  return path


def _webhook_url() -> str:
  """The Chat incoming-webhook URL from Secret Manager. Never log it."""
  # pylint: disable=import-outside-toplevel
  import google.auth
  from google.cloud import secretmanager

  _, project = google.auth.default()
  name = (
      f"projects/{project}/secrets/"
      f"{WEBHOOK_SECRET.format(environment=ENVIRONMENT)}/versions/latest"
  )
  client = secretmanager.SecretManagerServiceClient()
  response = client.access_secret_version(request={"name": name})
  return response.payload.data.decode("utf-8").strip()


@task(retries=0)
def post_to_chat(draft_path: str) -> Optional[str]:
  """Posts the draft to the test Chat space, once per report date."""
  # pylint: disable=import-outside-toplevel
  import requests
  from airflow.exceptions import AirflowSkipException

  context = get_current_context()
  conf = {**context["params"], **(context["dag_run"].conf or {})}
  if not dag_monitor.auto_post_enabled(ENVIRONMENT, conf):
    raise AirflowSkipException("Posting is off for this run (post=false).")
  marker = draft_path[: -len(".draft.json")] + ".posted.json"
  if os.path.exists(marker) and not conf.get("force_post"):
    raise AirflowSkipException(f"Already posted: {marker}")
  with open(draft_path, encoding="utf-8") as f:
    messages = json.load(f)["messages"]
  report_date = os.path.basename(marker).split(".", 1)[0]
  url = _webhook_url()
  posted = []
  for message in messages:
    try:
      response = requests.post(
          url,
          params={
              "threadKey": f"orbax-daily-monitor-{report_date}",
              "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
          },
          json={"text": message},
          timeout=WEBHOOK_TIMEOUT_S,
      )
    except requests.RequestException as e:
      # The exception text contains the URL, which holds the webhook key.
      raise RuntimeError(
          f"Chat webhook call failed: {type(e).__name__}"
      ) from None
    if response.status_code != 200:
      raise RuntimeError(
          f"Chat webhook returned HTTP {response.status_code} after "
          f"{len(posted)} of {len(messages)} messages"
      )
    reply = response.json()
    posted.append(
        {
            "name": reply.get("name"),
            "thread": (reply.get("thread") or {}).get("name"),
        }
    )
  _write_json(
      marker,
      {
          "messages": posted,
          "posted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "draft": draft_path,
      },
  )
  logging.info("Posted %d message(s): %s", len(posted), posted)
  return marker


with models.DAG(
    dag_id=SELF_DAG_ID,
    start_date=datetime.datetime(2026, 10, 1),
    schedule_interval=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=datetime.timedelta(minutes=45),
    tags=["multipod_team", "orbax", "monitoring"],
    description="Daily digest of Orbax DAG runs, failures and logs.",
    params={
        "report_date": Param(
            None,
            type=["null", "string"],
            format="date",
            title="Report date (UTC)",
            description="Day to collect. Leave empty for yesterday (UTC).",
        ),
        **(
            {
                "post": Param(
                    False,
                    type="boolean",
                    title="Post to Chat",
                    description=(
                        "Post through the Chat webhook. Needs the webhook"
                        " secret; leave off and post from Jetski instead."
                    ),
                ),
                "force_post": Param(
                    False,
                    type="boolean",
                    title="Post again",
                    description="Post even if this date was already posted.",
                ),
            }
            if AUTO_POST
            else {}
        ),
    },
    doc_md="""
      # Orbax daily monitor

      Writes `data/dag_monitor/orbax/<report_date>.json` in the Composer
      bucket: every Orbax DAG run in one UTC day, its 7-run history, and, for
      each DAG's latest failed run, the failed tasks' errors and the
      workload's pod logs. Read-only towards the monitored DAGs.

      Manual trigger: pick **Report date (UTC)** in the trigger form, or
      leave it empty for yesterday (UTC). From the CLI, conf can also be
      `{"start": "YYYY-MM-DDTHH:MM:SSZ", "end": "YYYY-MM-DDTHH:MM:SSZ"}`.
    """,
) as dag:
  window = get_window()
  registry = build_registry()
  collected = collect_runs(window, registry)
  digest_path = write_digest(window, registry, collected)
  if AUTO_POST:
    # Decision 7: erniechang-test only. Prod and dev never create these tasks.
    triage_path = triage_with_gemini(digest_path)
    draft_path = write_draft(digest_path, triage_path)
    post_to_chat(draft_path)
