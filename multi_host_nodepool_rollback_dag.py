import datetime
import logging
import time

from airflow import models
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from google.cloud import monitoring_v3

import node_pool_util


@task
def check_availability(checking_available: bool) -> bool:
  """A sensor for multi-host nodepool availability.

  This task polls the node_pool/multi_host/available metric
  every 60 seconds for 45 minutes. Waiting for the nodepool to become
  available or unavailable

  Args:
    checking_available(bool): True if the function is checking for the
    nodepool to become available, False if the function is checking for
    it to become unavailble.

  Returns:
    bool: True if intended event is detected. False if the intended event
    is not detected.
  """

  now = time.time()

  # Must be converted to an int for the API
  seconds = int(now)

  # Look at the previous 60 seconds to see what the "current" state is. 60
  # seconds is the agregation period.
  fixed_time_interval = monitoring_v3.TimeInterval({
      "end_time": {"seconds": seconds},
      "start_time": {"seconds": seconds - 60},
  })

  mon_client = monitoring_v3.MetricServiceClient()
  project_name = f"projects/{PROJECT_ID}"
  results = mon_client.list_time_series(
      request={
          "name": project_name,
          "filter": (
              'metric.type="kubernetes.io/node_pool/multi_host/available" '
              f'AND resource.labels.cluster_name="{Variable.get("CLUSTER_NAME")}" '
              f'AND resource.labels.node_pool_name="{Variable.get("NODE_POOL_NAME")}"'
          ),
          "interval": fixed_time_interval,
          "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
      }
  )

  # If no time series at all is returned
  if not results.time_series:
    logging.info("No time series.")
    return False

  # The results must be converted to a list to be iterable.
  results_list = list(results)

  # The default sorting is the most recent events first.
  state = results_list[0].points[0].value.bool_value

  logging.info("Most recent state: %s", state)
  if checking_available == state:
    logging.info("Nodepool available: %s", checking_available)
    return True
  return False


with models.DAG(
    dag_id="multi-host-availability-rollback",
    schedule="00 06 * * *",
    tags=[
        "cloud-ml-auto-solutions",
        "multi-host-availability",
        "tpu_obervability",
    ],
    start_date=datetime.datetime(2025, 6, 26),
    catchup=False,
) as dag:

  PROJECT_ID = Variable.get("PROJECT_ID", default_var="tpu-prod-env-one-vm")
  CLUSTER_NAME = Variable.get("CLUSTER_NAME", default_var="qmcgarry-auto-test")
  NODE_POOL_NAME = Variable.get("NODE_POOL_NAME", default_var="nodepool-auto")
  REGION = Variable.get("LOCATION", default_var="asia-northeast1")
  NODE_LOCATIONS = Variable.get(
      "NODE_LOCATIONS", default_var="asia-northeast1-b"
  )
  NUM_NODES = Variable.get("NUM_NODES", default_var=4)
  MACHINE_TYPE = Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t")
  TPU_TOPOLOGY = Variable.get("TPU_TOPOLOGY", default_var="4x4")

  node_pool_info = node_pool_util.Info(
      project_id=PROJECT_ID,
      cluster_name=CLUSTER_NAME,
      node_pool_name=NODE_POOL_NAME,
      location=REGION,
      node_locations=NODE_LOCATIONS,
      num_nodes=NUM_NODES,
      machine_type=MACHINE_TYPE,
      tpu_topology=TPU_TOPOLOGY,
  )

  create_node_pool = node_pool_util.create(info=node_pool_info)

  wait_availability = check_availability(True)

  wait_unavailability = check_availability(False)

  # Checks if the cluster exists. If not, the DAG will fail.
  check_for_cluster = BashOperator(
      task_id="check_for_cluster",
      bash_command=f"""
          if gcloud container clusters describe {CLUSTER_NAME} \\
            --project {PROJECT_ID} --region {REGION} &> /dev/null; then
            echo "GKE cluster {CLUSTER_NAME} already exists."
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
          f"gcloud container node-pools rollback {NODE_POOL_NAME} "
          f"--project={PROJECT_ID} --cluster={CLUSTER_NAME} "
          f"--region {REGION} --quiet"
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
