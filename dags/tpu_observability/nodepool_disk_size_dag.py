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
from dags.common.vm_resource import Region, Zone
from dags.tpu_observability.configs.common import MachineConfigMap

QUERY_WINDOW_DURATION_SECONDS = 3600


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_nodepool_metrics_event(
    project_id: str,
    filter_query: str,
    start_time: int,
    end_time: int,
) -> bool:
  """ "
  Polls Cloud Monitoring for node pool recovery metric events.

  This sensor queries the Cloud Monitoring API for metric time series that
  match the provided filter and fall within the specified time window. It
  is used to detect whether a successful node pool recovery event was
  emitted after the update operation completed.

  Args:
    project_id: The Google Cloud project to query.
    filter_query: A Monitoring filter expression specifying the metric type.
    start_time_unixseconds: Start of the search window (Unix seconds).
    end_time_unixseconds: End of the search window (Unix seconds).

  Returns:
    True if at least one matching metric time series is found; otherwise False.
  """

  start_sec = int(start_time)
  end_sec = int(end_time) + QUERY_WINDOW_DURATION_SECONDS

  start_timeutil = TimeUtil.from_unix_seconds(start_sec)
  end_timeutil = TimeUtil.from_unix_seconds(end_sec)

  series = query_time_series(
      project_id=project_id,
      filter_str=filter_query,
      start_time=start_timeutil,
      end_time=end_timeutil,
  )

  if not series:
    logging.info(
        "[metrics] No matching time series; repoking " "window=%s → %s",
        start_timeutil.to_iso_string(),
        end_timeutil.to_iso_string(),
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
    start_time: int,
    end_time: int,
) -> bool:
  """
  Polls Cloud Logging for node pool update or recovery log entries.

  This sensor searches GKE logs within a specific time window to determine
  whether the node pool update operation generated the expected log events.
  It is typically used together with the Monitoring sensor to confirm both
  the control-plane and data-plane signals of a successful recovery.

  Args:
    project_id: The Google Cloud project to query.
    filter_query: A Log Explorer filter expression.
    start_time_unixseconds: Start of the search window (Unix seconds).
    end_time_unixseconds: End of the search window (Unix seconds).

  Returns:
    True if at least one matching log entry is found; otherwise False.
  """

  start_timeutil = TimeUtil.from_unix_seconds(start_time)
  end_timeutil = TimeUtil.from_unix_seconds(end_time)

  full_filter = f'{base_filter} AND operation.id="{op_id}"'

  entries = query_log_entries(
      project_id=project_id,
      filter_str=full_filter,
      start_time=start_timeutil,
      end_time=end_timeutil,
  )

  if not entries:
    logging.info(
        "[logs] No matching log entries; repoking window=%s → %s",
        start_timeutil.to_iso_string(),
        end_timeutil.to_iso_string(),
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

  operation_response_error = operation_response.error
  if operation_response_error and (
      operation_response_error.message or operation_response_error.details
  ):
    raise RuntimeError(
        f"GKE operation finished with error: {operation_response_error}"
    )

  logging.info("[wait] operation_name=%s reached DONE", operation_name)
  return True


@task
def get_update_operation_meta(operation_name: str) -> dict:
  """Fetches operation metadata and computes timing info.

  Returns:
      operation_id: str,       # e.g. "operation-17636..."
      start_time_unixseconds: int,    # unix seconds
      end_time_unixseconds: int,      # unix seconds
  """
  cluster_manager_client = ClusterManagerClient()
  operation_response = cluster_manager_client.get_operation(name=operation_name)

  # Extract operation_id for use in Logging filters
  # If operation_name is full path, the short name is usually the last segment.
  if operation_name.startswith("projects/"):
    operation_id = operation_name.rsplit("/", 1)[-1]
  else:
    operation_id = operation_response.name or operation_name

  # GKE returns RFC3339 strings for start_time / end_time in this environment
  if operation_response.start_time:
    start_timeutil = TimeUtil.from_iso_string(operation_response.start_time)
  else:
    raise RuntimeError("Operation is DONE but start_time is missing")

  if operation_response.end_time:
    end_timeutil = TimeUtil.from_iso_string(operation_response.end_time)
  else:
    # Very rare; safest is to anchor end at start
    end_timeutil = start_timeutil

  start_time_unixseconds = start_timeutil.to_unix_seconds()
  end_time_unixseconds = end_timeutil.to_unix_seconds()
  duration_s = max(0, end_time_unixseconds - start_time_unixseconds)

  logging.info(
      "[op_meta] operation_name=%s operation_id=%s start=%s end=%s duration=%ss",
      operation_name,
      operation_id,
      start_time_unixseconds.to_iso_string(),
      end_time_unixseconds.to_iso_string(),
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

  return (
      operation_id,
      start_time_unixseconds,
      end_time_unixseconds,
  )


@task
def get_nodepool_disk_size(node_pool: node_pool.Info) -> int:
  """
  Retrieves the current boot disk size (GB) of a GKE node pool.

  This task reads the node pool configuration via the GKE API and extracts
  the boot disk size from the instance group configuration. It is used to
  verify both the original size before an update and the updated size after
  a disk resize operation.

  Args:
    node_pool: An Info object identifying the GKE node pool.

  Returns:
    The boot disk size (in GB) of the node pool.

  Raises:
    RuntimeError: If the disk size cannot be determined from the API response.
  """

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
def validate_disk_resize(current_size_gb: int, expected_size_gb: int) -> bool:
  """
  Validates whether the node pool disk size matches the expected value.

  This function compares an expected boot disk size against the actual
  size retrieved from the GKE API and returns True if they match.

  Args:
    current_size_gb: The actual disk size retrieved from the node pool.
    expected_size_gb: The desired disk size for this test.

  Returns:
    True if the actual disk size equals the expected size.

  Raises:
    RuntimeError: If the disk size was already equal to the expected value
                  before the update, or if the resize operation did not succeed.
  """

  # If the disk was already equal to the target before update → invalid test
  if current_size_gb == expected_size_gb:
    raise RuntimeError(
        f"Disk resize test invalid: disk size already equals {expected_size_gb} GB."
    )

  # If resize was performed but size does not match expected → failure
  if current_size_gb != expected_size_gb:
    raise RuntimeError(
        f"Disk resize failed: expected {expected_size_gb} GB, got {current_size_gb} GB."
    )

  logging.info(
      "[validate_disk_resize] Disk size validation succeeded: current=%d expected=%d",
      current_size_gb,
      expected_size_gb,
  )
  return True


for machine in MachineConfigMap:
  config = machine.value

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
            "CLUSTER_NAME", default_var="tpu-observability-automation"
        ),
        node_pool_name=Variable.get(
            "NODE_POOL_NAME",
            default_var="update-node-pool-disksize-v6e-autotest",
        ),
        location=Variable.get("LOCATION", default_var=Region.US_CENTRAL1.value),
        node_locations=Variable.get(
            "NODE_LOCATIONS", default_var=Zone.US_CENTRAL1_B.value
        ),
        num_nodes=Variable.get("NUM_NODES", default_var=2),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    with TaskGroup(group_id=f"v{config.tpu_version.value}") as v6e_group:
      create_nodepool = node_pool.create(node_pool=node_pool_info)

      original_disk_size = get_nodepool_disk_size(node_pool_info)

      operation_name = node_pool.update(
          node_pool_info,
          new_size_gb=(original_disk_size + 50),
          return_operation_name=True,
      )

      wait_done = wait_for_update_to_complete(operation_name)

      (
          operation_id,
          start_time_unixseconds,
          end_time_unixseconds,
      ) = get_update_operation_meta(operation_name)

      updated_disk_size = get_nodepool_disk_size(node_pool_info)

      disk_resize_check = validate_disk_resize(
          current_size_gb=updated_disk_size,
          expected_size_gb=(original_disk_size + 50),
      )

      poll_metrics = wait_for_nodepool_metrics_event(
          project_id=node_pool_info.project_id,
          filter_query=(
              'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
              f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
          ),
          start_time=start_time_unixseconds,
          end_time=end_time_unixseconds,
      )

      poll_logs = wait_for_nodepool_logs_event(
          project_id=node_pool_info.project_id,
          base_filter=(
              'resource.type="gke_nodepool" '
              f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}" '
              f'AND resource.labels.location="{node_pool_info.location}" '
              f'AND resource.labels.nodepool_name="{node_pool_info.node_pool_name}"'
          ),
          op_id=operation_id,
          start_time=start_time_unixseconds,
          end_time=end_time_unixseconds,
      )

      cleanup_node_pool = node_pool.delete.override(
          trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_nodepool,
      )

      (
          create_nodepool
          >> original_disk_size
          >> operation_name
          >> wait_done
          >> op_meta
          >> updated_disk_size
          >> disk_resize_check
          >> [poll_metrics, poll_logs]
          >> cleanup_node_pool
      )
