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


@task
def check_duration_false_negative(
    start_time_unixseconds: int,
    end_time_unixseconds: int,
    threshold_s: int = 150,
) -> None:
  """
  Logs whether the observed update+recovery window is shorter than a threshold.

  This is an informational check used to indicate potential false negatives
  when downstream metrics/logs are expected to appear only after a certain
  minimum restart window.

  Args:
    start_time_unixseconds: Anchor timestamp (Unix seconds) recorded when the
      update command started.
    end_time_unixseconds: Anchor timestamp (Unix seconds) recorded after the
      node pool returns to RUNNING.
    threshold_s: Threshold in seconds used for the false-negative heuristic.
  """
  duration_s = max(0, int(end_time_unixseconds) - int(start_time_unixseconds))
  logging.info(
      "[duration_check] start=%d end=%d duration=%ss threshold=%ss",
      start_time_unixseconds,
      end_time_unixseconds,
      duration_s,
      threshold_s,
  )

  if duration_s < threshold_s:
    logging.info(
        "Restart shorter than %d seconds. This may cause a false negative.",
        threshold_s,
    )
  else:
    logging.info(
        "Restart longer than %d seconds. False negative should not occur.",
        threshold_s,
    )


@task
def capture_now() -> int:
  """Returns the current Unix timestamp in seconds."""
  return int(datetime.datetime.now())


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

  #

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

      #
      update_spec = node_pool.NodePoolUpdateSpec(
          disk_size_gb=original_disk_size + 50
      )
      updater = node_pool.NodePoolUpdater(node_pool_info)
      # need to check on the format
      update_start_time = updater.update(update_spec)

      task_id = "wait_for_running"
      wait_for_running = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RUNNING
      )

      update_end_time = capture_now()

      duration_check = check_duration_false_negative(
          update_start_time, update_end_time
      )

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
          start_time=update_start_time,
          end_time=update_end_time + QUERY_WINDOW_DURATION_SECONDS,
      )

      cleanup_node_pool = node_pool.delete.override(
          trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_nodepool,
      )

      (
          create_nodepool
          >> original_disk_size
          >> update_start_time  # need to check on the function decorator
          >> wait_for_running
          >> updated_disk_size
          >> disk_resize_check
          >> poll_metrics
          >> cleanup_node_pool
      )
