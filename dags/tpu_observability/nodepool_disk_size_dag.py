import time
from datetime import datetime, timezone
import os
import subprocess

from kubernetes import client as k8s_client, config as k8s_config
from google.cloud.container_v1 import ClusterManagerClient
from google.cloud import container_v1
from google.cloud import monitoring_v3
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.decorators import task, dag
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow import models

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


# metrics print out version without anchor_timestamp but with TimeUtil wrapped: wait_for_metrics_event()
@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_nodepool_metrics_event(
    project_id: str,
    filter_query: str,
    start_time,
    end_time,
) -> bool:
  """Poll Cloud Monitoring for node-pool recovery metrics in [start_time, end_time].

  The sensor finishes (returns True) once at least one matching time series is found.
  It uses the shared query_time_series utility and TimeUtil for time handling.
  """
  try:
    start_tu = TimeUtil.from_unix_seconds(start_time)
    end_tu = TimeUtil.from_unix_seconds(end_time)
  except Exception as e:
    print(f"[metrics] invalid time input: {e}")
    # this is a hard configuration error → fail the task
    raise

  try:
    series = query_time_series(
        project_id=project_id,
        filter_str=filter_query,
        start_time=start_tu,
        end_time=end_tu,
        # use defaults: view=FULL, page_size=500, no aggregation
        log_enable=False,
    )
  except Exception as e:
    print(f"[metrics] ERROR querying time series: {e}")
    # treat as transient; sensor will repoke
    return False

  if not series:
    print(
        "[metrics] No matching time series yet; repoking… "
        f"window={start_tu.to_iso_string()} → {end_tu.to_iso_string()}"
    )
    return False

  # --- Output some information for validation ---
  first = series[0]
  print(f"[metrics] Metric detected. series_count={len(series)}")
  print(f"[metrics] metric.type={first.metric.type}")
  print(f"[metrics] resource.labels={dict(first.resource.labels)}")

  if first.points:
    p = first.points[0]
    print(
        f"[metrics] first point: value={p.value} "
        f"@ {p.interval.end_time}"
    )

  return True


# metrics print out version without anchor_timestamp: wait_for_logs_event()
@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_nodepool_logs_event(
    project_id: str,
    filter_query: str,
    start_time,
    end_time,
) -> bool:
  """Poll Cloud Logging for node-pool related log entries in [start_time, end_time].

  Finishes once at least one matching log entry is found.
  """
  try:
    start_tu = TimeUtil.from_unix_seconds(start_time)
    end_tu = TimeUtil.from_unix_seconds(end_time)
  except Exception as e:
    print(f"[logs] invalid time input: {e}")
    raise

  try:
    entries = query_log_entries(
        project_id=project_id,
        filter_str=filter_query,
        start_time=start_tu,
        end_time=end_tu,
        log_enable=False,
    )
  except Exception as e:
    print(f"[logs] ERROR querying log entries: {e}")
    # transient → repoke
    return False

  if not entries:
    print(
        "[logs] No matching log entries yet; repoking… "
        f"window={start_tu.to_iso_string()} → {end_tu.to_iso_string()}"
    )
    return False

  # --- Output some information about the first log entry ---
  first = entries[0]
  print(
      f"[logs] Log event detected. count={len(entries)}, "
      f"timestamp={first.timestamp}, log_name={first.log_name}"
  )

  payload = getattr(first, "text_payload", None) or getattr(
      first, "json_payload", None
  )
  if payload:
    snippet = str(payload)
    if len(snippet) > 200:
      snippet = snippet[:200] + "…"
    print(f"[logs] first entry payload snippet: {snippet}")

  return True


# ver3: wait_for_update_to_complete() output timestamp
@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def wait_for_update_to_complete(op_full: str) -> dict:
  """
  Sensor that waits for a GKE node-pool update operation to reach DONE.

  Returns:
    {
      "start_sec": int,
      "end_sec": int,
      "duration_s": int,
      "anchor_seconds": int
    }
  """
  cm = ClusterManagerClient()
  op = cm.get_operation(name=op_full)
  status = getattr(op.status, "name", str(op.status))
  print(f"[wait] op={op_full} status={status}")

  if status == "DONE":
    # If DONE, surface any error right away
    if op.error and (op.error.message or op.error.details):
      raise RuntimeError(f"GKE operation finished with error: {op.error}")

    # --- Extract start & end times ---
    if getattr(op, "start_time", None):
      start_dt = op.start_time.ToDatetime()
      start_tu = TimeUtil.from_datetime(start_dt)
      start_sec = start_tu.to_unix_seconds()
      start_iso = start_tu.to_iso_string()
    else:
      raise RuntimeError("Operation is DONE but op.start_time is missing")

    if getattr(op, "end_time", None):
      end_dt = op.end_time.ToDatetime()
      end_tu = TimeUtil.from_datetime(end_dt)
      end_sec = end_tu.to_unix_seconds()
      end_iso = end_tu.to_iso_string()
    else:
      # Rare case — but safer to anchor end_time = start_time
      end_sec = start_sec
      end_iso = start_iso
  else:
     # not DONE yet → repoke
    print(f"[wait] status={status} …repoking")
    return False

  # --- Duration & anchor ---
  duration_s = max(0, end_sec - start_sec)
  anchor_seconds = (start_sec + end_sec) // 2

  # --- Logging only ---
  print(f"[wait] DONE start={start_iso} end={end_iso} duration={duration_s}s")

  # 150s informational note (kept here, so no separate task needed)
  if duration_s < 150:
    print("Restart shorter than 150 seconds. This may cause a false negative")
  else:
    print("Restart longer than 150 seconds. False negative should not occur")

  # --- Return raw timestamps ---
  return {
      "op_id": op_full.split("operation-")[1],
      "start_sec": start_sec,
      "end_sec": end_sec,
      "duration_s": duration_s,
      "anchor_seconds": anchor_seconds,
  }


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
    start_tu = TimeUtil.from_timestamp_pb2(op.start_time)
  else:
    start_tu = TimeUtil.from_unix_seconds(int(time.time()))

  start_iso = start_tu.to_iso_string()

  print(f"[update] op_id={op.name}")
  print(f"[update] op_full={op_full}")
  print(f"[update] start_ts={start_iso}")
  return {"op_full": op_full, "start_ts": start_iso}


@task
def summarize_test(metrics_ok: bool, logs_ok: bool) -> None:
  """
  Final verdict task.
  This runs only if both sensors have succeeded.
  """
  print(f"[summary] metrics_ok={metrics_ok}, logs_ok={logs_ok}")
  if not metrics_ok or not logs_ok:
    # This *shouldn't* happen if sensors are wired correctly,
    # but we keep it as a guardrail.
    raise RuntimeError(
        f"Expected both metrics and logs sensors to succeed, got: "
        f"metrics_ok={metrics_ok}, logs_ok={logs_ok}"
    )

  print("[summary] Disk-resize node-pool test PASSED: "
        "GKE operation finished, metric emitted, and log event detected.")


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

  UPTIME_FILTER_QRY = (
      'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
      f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
  )

  create_nodepool = node_pool.create(node_pool=node_pool_info)

  update_result = update_nodepool_disksize(node_pool_info, new_size_gb=150)

  op_meta = wait_for_update_to_complete(update_result["op_full"])

  poll_metrics = wait_for_nodepool_metrics_event(
      project_id=node_pool_info.project_id,
      filter_query=UPTIME_FILTER_QRY,
      start_time=op_meta["start_sec"],
      end_time=op_meta["end_sec"],
  )

  LOG_FILTER_QRY = (
      'resource.type="k8s_cluster" '
      f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
      f'AND resource.labels.location="{node_pool_info.location}"'
      f'AND operation.id="{op_meta.op_id}"'
  )

  poll_logs = wait_for_nodepool_logs_event(
      project_id=node_pool_info.project_id,
      filter_query=LOG_FILTER_QRY,   # your log filter
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
    >> op_meta
    >> [poll_metrics, poll_logs]
    >> summary
    >> cleanup_node_pool
)
