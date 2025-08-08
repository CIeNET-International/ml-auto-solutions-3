import datetime
import logging
import time

from airflow.operators.python import task, get_current_context
from airflow import models
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from google.cloud import monitoring_v3

from dags.tpu_obs.utils import node_pool_util
from dags.common.vm_resource import Project
from dags.map_reproducibility.utils import constants


@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def check_availability(
    node_pool: node_pool_util.Info, checking_available: bool
) -> bool:
  """Check current multi-host nodepool availability.

  This is a sensor task which runs every 30s for 900s. The task takes
  the current list of the multi_host availability outputs for the last 5
  minutes aggregated to 1 minute intervals. The results are listed, and
  the most recent result is checked to determine if it matches
  specified result, True or False.

  Args:
    checking_available(bool): True if the function is checking for the
    nodepool to become available, False if the function is checking for
    it to become unavailble.

  Returns:
    bool: True if intended event is detected. False if the intended event
    is not detected.
  """
  context = get_current_context()
  ti = context["ti"]
  now_in_seconds = int(ti.start_date.timestamp())
  logging.info("start_date: %s", now_in_seconds)

  api_client = monitoring_v3.MetricServiceClient()
  results = api_client.list_time_series(
      request={
          "name": f"projects/{node_pool.project_id}",
          "filter": (
              'metric.type="kubernetes.io/node_pool/multi_host/available" '
              f'AND resource.labels.cluster_name="{node_pool.cluster_name}" '
              f'AND resource.labels.node_pool_name="{node_pool.node_pool_name}"'
          ),
          "interval": monitoring_v3.TimeInterval({
              "end_time": {"seconds": now_in_seconds},
              "start_time": {"seconds": now_in_seconds - 300},
          }),
          "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
      }
  )

  # If no time series at all is returned
  if not results.time_series:
    logging.info("No time series.")
    return False

  state = False
  for time_series in results:
    for point in time_series.points:
      assert isinstance(point.value, monitoring_v3.types.common.TypedValue)
      assert isinstance(point.value.bool_value, bool)
      logging.info("Point value: %s", point.value.bool_value)
      state = point.value.bool_value
      break
    break

  logging.info("Nodepool available: %s", checking_available)
  return checking_available == state


with models.DAG(
    dag_id="multi-host-availability-rollback",
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    tags=[
        "cloud-ml-auto-solutions",
        "multi-host-availability",
        "tpu_obervability",
    ],
    start_date=datetime.datetime(2025, 6, 26),
    catchup=False,
) as dag:

  node_pool_info = node_pool_util.Info(
      project_id=Project.TPU_PROD_ENV_ONE_VM.value,
      cluster_name=Variable.get(
          "CLUSTER_NAME", default_var="qmcgarry-auto-test"
      ),
      node_pool_name=Variable.get(
          "NODE_POOL_NAME", default_var="nodepool-auto"
      ),
      location=Variable.get("LOCATION", default_var="asia-northeast1"),
      node_locations=Variable.get(
          "NODE_LOCATIONS", default_var="asia-northeast1-b"
      ),
      num_nodes=Variable.get("NUM_NODES", default_var=4),
      machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t"),
      tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="4x4"),
  )

  create_node_pool = node_pool_util.create(info=node_pool_info)

  wait_availability = check_availability(node_pool_info, True)

  wait_unavailability = check_availability(node_pool_info, False)

  # Checks if the cluster exists. If not, the DAG will fail.
  check_for_cluster = BashOperator(
      task_id="check_for_cluster",
      bash_command=f"""
          if gcloud container clusters describe {node_pool_info.cluster_name} \\
            --project {node_pool_info.project_id} --region {node_pool_info.location} &> /dev/null; then
            echo "GKE cluster {node_pool_info.cluster_name} already exists."
          else
            echo "ERROR: cluster does not exist."
            exit 1
          fi
        """,
  )

  # Performs a rollback of the nodepool, creating an interruption.
  run_rollback = BashOperator(
      task_id="run_rollback",
      bash_command=(
          "gcloud container node-pools rollback"
          f" {node_pool_info.node_pool_name} --project={node_pool_info.project_id} --cluster={node_pool_info.cluster_name} --region"
          f" {node_pool_info.location} --quiet"
      ),
  )

  # Cleanup task. Nodepool needs to be deleted.
  delete_node_pool = node_pool_util.delete.override(
      task_id="delete_node_pool", trigger_rule="all_done"
  )(info=node_pool_info)

  # This ensures the test will be properly marked as success or failure
  end = DummyOperator(task_id="end")

  (
      check_for_cluster
      >> create_node_pool
      >> wait_availability
      >> run_rollback
      >> wait_unavailability
      >> delete_node_pool
  )

  [wait_availability, wait_unavailability] >> end
