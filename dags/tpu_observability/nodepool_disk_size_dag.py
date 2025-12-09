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


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_nodepool_metrics_event(
    project_id: str,
    filter_query: str,
    start_time,
    end_time,
) -> bool:
  """Poll Cloud Monitoring for node-pool recovery metrics in [start_time, end_time]."""

  start_sec = int(start_time)
  end_sec = int(end_time) + QUERY_WINDOW_DURATION_SECONDS

  start_tu = TimeUtil.from_unix_seconds(start_sec)
  end_tu = TimeUtil.from_unix_seconds(end_sec)

  series = query_time_series(
      project_id=project_id,
      filter_str=filter_query,
      start_time=start_tu,
      end_time=end_tu,
      log_enable=False,
  )

  if not series:
    logging.info(
        "[metrics] No matching time series; repoking " "window=%s → %s",
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


@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def wait_for_update_to_complete(operation_name: str) -> bool:
  """Sensor that waits for the GKE node-pool update operation to reach DONE."""
  cluster_mgr_client = ClusterManagerClient()
  operation_response = cluster_mgr_client.get_operation(name=operation_name)

  status = getattr(
      operation_response.status, "name", str(operation_response.status)
  )
  logging.info("[wait] operation_name=%s status=%s", operation_name, status)

  if status != "DONE":
    return False

  if operation_response.error and (
      operation_response.error.message or operation_response.error.details
  ):
    raise RuntimeError(
        f"GKE operation finished with error: {operation_response.error}"
    )

  logging.info("[wait] operation_name=%s reached DONE", operation_name)
  return True


@task
def get_update_operation_meta(operation_name: str) -> dict:
  """Fetches operation metadata and computes timing info.

  Returns:
    {
      "operation_name": str,
      "operation_id": str,       # e.g. "operation-17636..."
      "start_time_sec": int,    # unix seconds
      "end_time_sec": int,      # unix seconds
      "duration_s": int,        # end - start
    }
  """
  cluster_mgr_client = ClusterManagerClient()
  operation_response = cluster_mgr_client.get_operation(name=operation_name)

  if operation_name.startswith("projects/"):
    operation_id = operation_name.rsplit("/", 1)[-1]
  else:
    operation_id = operation_response.name or operation_name

  if operation_response.start_time:
    start_tu = TimeUtil.from_iso_string(operation_response.start_time)
  else:
    raise RuntimeError("Operation is DONE but start_time is missing")

  if operation_response.end_time:
    end_tu = TimeUtil.from_iso_string(operation_response.end_time)
  else:
    end_tu = start_tu

  start_time_sec = start_tu.to_unix_seconds()
  end_time_sec = end_tu.to_unix_seconds()
  duration_s = max(0, end_time_sec - start_time_sec)

  logging.info(
      "[op_meta] operation_name=%s operation_id=%s start=%s end=%s duration=%ss",
      operation_name,
      operation_id,
      start_tu.to_iso_string(),
      end_tu.to_iso_string(),
      duration_s,
  )

  if duration_s < 150:
    logging.info(
        "Restart shorter than 150 seconds. This may cause a false negative."
    )
  else:
    logging.info(
        "Restart longer than 150 seconds. False negative should not occur."
    )

  return {
      "operation_name": operation_name,
      "operation_id": operation_id,
      "start_time_sec": start_time_sec,
      "end_time_sec": end_time_sec,
      "duration_s": duration_s,
  }


@task
def get_nodepool_disk_size(node_pool: node_pool.Info) -> int:
  """Return the current boot disk size (GB) of the node pool."""
  client = container_v1.ClusterManagerClient()

  nodepool_name = (
      f"projects/{node_pool.project_id}/locations/{node_pool.location}/"
      f"clusters/{node_pool.cluster_name}/nodePools/{node_pool.node_pool_name}"
  )

  response = client.get_node_pool(name=nodepool_name)

  disk_size = getattr(response.config, "disk_size_gb", None)

  if disk_size is None:
    raise RuntimeError(f"Nodepool returned no disk_size_gb: {response}")

  return int(disk_size)


@task
def validate_disk_resize(
    original: int, updated: int, expected_new_size: int
) -> bool:
  if original == expected_new_size:
    raise RuntimeError(
        f"Disk resize test invalid: original size already = {expected_new_size} GB."
    )

  if updated != expected_new_size:
    raise RuntimeError(
        f"Disk resize failed: expected {expected_new_size}, got {updated}"
    )

  logging.info(
      "[validate_disk_resize] OK: original=%d updated=%d expected=%d",
      original,
      updated,
      expected_new_size,
  )
  return True


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
        "checking node readiness, and verifying the recovery event through both Cloud Monitoring metrics and Cloud Logging entries. "
        "Cleans up the node pool afterward."
    ),
    doc_md="""
    # Node-Pool Availability Test (Disk Resize)

    ### Purpose
    This DAG tests whether a GKE node pool remains observable and recovers
    as expected when a disk resize operation forces node restarts.

    ### Expected Outcome
    - Nodes restart and return to Ready state.
    - A recovery event is recorded in both Cloud Monitoring and Cloud Logging.

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
      machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t"),
      tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="2x4"),
  )

  with TaskGroup(group_id="v6e") as v6e_group:
    create_nodepool = node_pool.create(node_pool=node_pool_info)

    get_original_disk_size = get_nodepool_disk_size(node_pool_info)

    operation_name = node_pool.update(
        node_pool_info,
        new_size_gb=150,
        return_operation_name=True,
    )

    wait_done = wait_for_update_to_complete(operation_name)

    op_meta = get_update_operation_meta(operation_name)

    get_updated_disk_size = get_nodepool_disk_size(node_pool_info)

    disk_resize_check = validate_disk_resize(
        original=get_original_disk_size,
        updated=get_updated_disk_size,
        expected_new_size=150,
    )

    poll_metrics = wait_for_nodepool_metrics_event(
        project_id=node_pool_info.project_id,
        filter_query=(
            'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
            f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
        ),
        start_time=op_meta["start_time_sec"],
        end_time=op_meta["end_time_sec"],
    )

    poll_logs = wait_for_nodepool_logs_event(
        project_id=node_pool_info.project_id,
        base_filter=(
            'resource.type="gke_nodepool" '
            f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}" '
            f'AND resource.labels.location="{node_pool_info.location}" '
            f'AND resource.labels.nodepool_name="{node_pool_info.node_pool_name}"'
        ),
        op_id=op_meta["operation_id"],
        start_time=op_meta["start_time_sec"],
        end_time=op_meta["end_time_sec"],
    )

    cleanup_node_pool = node_pool.delete.override(
        trigger_rule=TriggerRule.ALL_DONE
    )(node_pool=node_pool_info).as_teardown(
        setups=create_nodepool,
    )

    (
        create_nodepool
        >> get_original_disk_size
        >> operation_name
        >> wait_done
        >> op_meta
        >> get_updated_disk_size
        >> disk_resize_check
        >> [poll_metrics, poll_logs]
        >> cleanup_node_pool
    )
