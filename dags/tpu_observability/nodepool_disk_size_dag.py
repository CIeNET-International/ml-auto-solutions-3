import time
from datetime import datetime, timezone
import logging

from google.cloud.container_v1 import ClusterManagerClient
from google.cloud import container_v1
from airflow.models import Variable
from airflow.decorators import task, dag
from airflow.utils.trigger_rule import TriggerRule
from airflow import models
from airflow.utils.task_group import TaskGroup


from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.gcp_util import query_time_series, query_log_entries

QUERY_WINDOW_DURATION_SECONDS = 3600


def _nodepool_name(info):
  return (
      f"projects/{info.project_id}/locations/{info.location}/"
      f"clusters/{info.cluster_name}/nodePools/{info.node_pool_name}"
  )


def _operation_name(project_id, location, op_id_or_name):
  return (
      op_id_or_name
      if str(op_id_or_name).startswith("projects/")
      else f"projects/{project_id}/locations/{location}/operations/{op_id_or_name}"
  )

# latest version! : widening the query window


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_nodepool_metrics_event(
    project_id: str,
    filter_query: str,
    start_time,
    end_time,
) -> bool:
  """Poll Cloud Monitoring for node-pool recovery metrics in [start_time, end_time]."""

  # widen the window a bit
  start_tu = TimeUtil.from_unix_seconds(start_time)
  end_tu = TimeUtil.from_unix_seconds(
      end_time) + QUERY_WINDOW_DURATION_SECONDS  # +1h after op end

  # Let query_time_series raise if API errors → Airflow will handle
  series = query_time_series(
      project_id=project_id,
      filter_str=filter_query,
      start_time=start_tu,
      end_time=end_tu,
      log_enable=False,
  )

  if not series:
    logging.info(
        "[metrics] No matching time series; repoking "
        "window=%s → %s",
        start_tu.to_iso_string(),
        end_tu.to_iso_string(),
    )
    return False

  first = series[0]
  logging.info(
      "[metrics] Metric detected. series_count=%d, metric.type=%s, resource.labels=%s",
      len(series),
      first.metric.type,
      dict(first.resource.labels),
  )
  if first.points:
    p = first.points[0]
    logging.info(
        "[metrics] first point: value=%s @ %s",
        p.value,
        p.interval.end_time,
    )

  return True


# latest! version3: remove printout and try/except bubble
@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_nodepool_logs_event(
    project_id: str,
    base_filter: str,
    op_id: str,
    start_time,
    end_time,
) -> bool:
  """Poll Cloud Logging for node-pool related log entries in [start_time, end_time]."""

  start_tu = TimeUtil.from_unix_seconds(start_time)
  end_tu = TimeUtil.from_unix_seconds(end_time)

  # Combine base filter with operation.id constraint --> will need to revise the filter!
  full_filter = f'{base_filter} AND operation.id="{op_id}"'

  entries = query_log_entries(
      project_id=project_id,
      filter_str=full_filter,
      start_time=start_tu,
      end_time=end_tu,
      log_enable=False,
  )

  if not entries:
    logging.info(
        "[logs] No matching log entries; repoking window=%s → %s",
        start_tu.to_iso_string(),
        end_tu.to_iso_string(),
    )
    return False

  first = entries[0]
  logging.info(
      "[logs] Log event detected. count=%d, timestamp=%s, log_name=%s",
      len(entries),
      first.timestamp,
      first.log_name,
  )

  payload = getattr(first, "text_payload", None) or getattr(
      first, "json_payload", None
  )
  if payload:
    snippet = str(payload)
    if len(snippet) > 200:
      snippet = snippet[:200] + "…"
    logging.info("[logs] first entry payload snippet: %s", snippet)

  return True


# latest! version5: turn the task.sensor into just task
@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def wait_for_update_to_complete(op_full: str) -> bool:
  """
  Sensor that waits for a GKE node-pool update operation to reach DONE.
  Returns True when DONE (or raises on error), False to repoke.
  """
  cm = ClusterManagerClient()
  op = cm.get_operation(name=op_full)
  status = getattr(op.status, "name", str(op.status))
  logging.info("[wait] op=%s status=%s", op_full, status)

  if status != "DONE":
    # not DONE yet → repoke
    return False

  # DONE → surface any error
  if op.error and (op.error.message or op.error.details):
    raise RuntimeError(f"GKE operation finished with error: {op.error}")

  logging.info("[wait] operation DONE")
  return True

# latest! add on getting meta data


@task
def get_update_operation_meta(op_full: str) -> dict:
  """
  Fetch GKE operation metadata (start/end/duration/op_id) once the operation is DONE.
  Returns:
    {
      "op_id": str,         # e.g. "operation-1763540237994-..."
      "start_sec": int,
      "end_sec": int,
      "duration_s": int,
      "anchor_seconds": int,
    }
  """
  cm = ClusterManagerClient()
  op = cm.get_operation(name=op_full)

  # Use op.name as operation.id (matches Cloud Logging)
  op_id = op.name

  if getattr(op, "start_time", None):
    # op.start_time is RFC3339 string (from your logs)
    start_tu = TimeUtil.from_iso_string(str(op.start_time))
  else:
    raise RuntimeError("Operation is DONE but op.start_time is missing")

  if getattr(op, "end_time", None):
    end_tu = TimeUtil.from_iso_string(str(op.end_time))
  else:
    # fallback: end = start
    end_tu = start_tu

  start_sec = start_tu.to_unix_seconds()
  end_sec = end_tu.to_unix_seconds()
  duration_s = max(0, end_sec - start_sec)
  anchor_seconds = (start_sec + end_sec) // 2

  logging.info(
      "[op_meta] op_id=%s start=%s end=%s duration=%ss",
      op_id,
      start_tu.to_iso_string(),
      end_tu.to_iso_string(),
      duration_s,
  )

  if duration_s < 150:
    logging.info(
        "Restart shorter than 150 seconds. This may cause a false negative"
    )
  else:
    logging.info(
        "Restart longer than 150 seconds. False negative should not occur"
    )

  return {
      "op_id": op_id,
      "start_sec": start_sec,
      "end_sec": end_sec,
      "duration_s": duration_s,
      "anchor_seconds": anchor_seconds,
  }


# latest! ver1: remove try/except and change the printout into using logging
@task
def update_nodepool_disksize(info: node_pool.Info, new_size_gb: int) -> dict:
  """
  Calls GKE UpdateNodePool via API and returns:
    {"op_full": <projects/.../operations/operation-...>, "start_ts": <ISO>}
  """
  cm = container_v1.ClusterManagerClient()
  req = container_v1.UpdateNodePoolRequest(name=_nodepool_name(info))
  req.disk_size_gb = int(new_size_gb)

  op = cm.update_node_pool(request=req)
  op_full = _operation_name(info.project_id, info.location, op.name)

  if getattr(op, "start_time", None):
    # start_time is an RFC3339 string
    start_tu = TimeUtil.from_iso_string(op.start_time)
  else:
    start_tu = TimeUtil.from_unix_seconds(int(time.time()))

  start_iso = start_tu.to_iso_string()

  logging.info("[update] op_id=%s", op.name)
  logging.info("[update] op_full=%s", op_full)
  logging.info("[update] start_ts=%s", start_iso)

  return {"op_full": op_full, "start_ts": start_iso}


# latest! version 1: using logging
@task
def summarize_test(metrics_ok: bool, logs_ok: bool) -> None:
  """Final verdict task."""
  logging.info("[summary] metrics_ok=%s, logs_ok=%s", metrics_ok, logs_ok)
  if not metrics_ok or not logs_ok:
    raise RuntimeError(
        f"Expected both metrics and logs sensors to succeed, got: "
        f"metrics_ok={metrics_ok}, logs_ok={logs_ok}"
    )

  logging.info(
      "[summary] Disk-resize node-pool test PASSED: "
      "GKE operation finished, metric emitted, and log event detected."
  )


with models.DAG(
    dag_id="nodepool_disk_size_ttr",
    start_date=datetime(2025, 6, 26),
    schedule="00 02 * * *",
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "nodepool_disk_size_ttr",
        "tpu_obervability",
        "time_to_recover",
    ],
    description=(
        "Tests GKE node-pool recovery by resizing disks, waiting ≥150s, "
        "checking node readiness, and polling Cloud Monitoring for recovery events. "
        "Cleans up the node pool afterward."
    ),
    doc_md="""
    # Node-Pool Availability Test (Disk Resize)

    ### Purpose
    This DAG tests whether a GKE node pool remains observable and recovers
    as expected when a disk resize operation forces node restarts.

    ### Expected Outcome
    - Nodes restart and return to Ready state.
    - A recovery event is recorded in Cloud Monitoring.

    """,
) as dag:
  node_pool_info = node_pool.Info(
      project_id="cienet-cmcs",
      cluster_name=Variable.get(
          "CLUSTER_NAME", default_var="athielee-auto-test-4"
      ),
      node_pool_name=Variable.get(
          "NODE_POOL_NAME", default_var="athie-nodepool-auto"
      ),
      location=Variable.get("LOCATION", default_var="europe-west4"),
      node_locations=Variable.get(
          "NODE_LOCATIONS", default_var="europe-west4-a"
      ),
      num_nodes=Variable.get("NUM_NODES", default_var=2),
      machine_type=Variable.get(
          "MACHINE_TYPE", default_var="ct6e-standard-4t"),
      tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="2x4"),
  )

  with TaskGroup(group_id="v6e") as v6e_group:
    UPTIME_FILTER_QRY = (
        'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
        f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
    )

    create_nodepool = node_pool.create(node_pool=node_pool_info)

    update_result = update_nodepool_disksize(node_pool_info, new_size_gb=150)

    wait_done = wait_for_update_to_complete(update_result["op_full"])

    op_meta = get_update_operation_meta(update_result["op_full"])

    poll_metrics = wait_for_nodepool_metrics_event(
        project_id=node_pool_info.project_id,
        filter_query=UPTIME_FILTER_QRY,
        start_time=op_meta["start_sec"],
        end_time=op_meta["end_sec"],
    )

    LOG_FILTER_BASE = (
        'resource.type="k8s_cluster" '
        f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}" '
        f'AND resource.labels.location="{node_pool_info.location}"'
    )

    poll_logs = wait_for_nodepool_logs_event(
        project_id=node_pool_info.project_id,
        base_filter=LOG_FILTER_BASE,
        op_id=op_meta["op_id"],
        start_time=op_meta["start_sec"],
        end_time=op_meta["end_sec"],
    )

    summary = summarize_test(poll_metrics, poll_logs)

    cleanup_node_pool = node_pool.delete.override(trigger_rule="all_done")(
        node_pool=node_pool_info
    ).as_teardown(
        setups=create_nodepool,
    )

    (
        create_nodepool
        >> update_result
        >> wait_done         # sensor: just waits until DONE
        >> op_meta           # task: fetches timestamps & op_id
        >> [poll_metrics, poll_logs]
        >> summary
        >> cleanup_node_pool
    )

  v6e_group
