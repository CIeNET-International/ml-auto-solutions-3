"""A DAG to test the jobset time-to-recover metric from a node pool rollback."""

import dataclasses
import datetime
import logging
import subprocess
import time

from airflow import models
from airflow.decorators import task
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from dags.common.vm_resource import Project, Region
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import node_pool_util as node_pool
from google.cloud import monitoring_v3


@dataclasses.dataclass
class Info:
  """Encapsulates information related to a GKE node pool and jobset file."""

  project_id: str
  cluster_name: str
  node_pool_name: str
  location: str
  yaml_file_name: str
  bucket_path: str


@task
def run_workload(info: Info):
  """Runs a kubectl workload in the designated cluster.

  Downloads the file at the fiven path to a temporary folder.
  Gets the credentials for the cluster specified in the info
  input. Runs the workload on the cluster and waits 150s to
  let the workload initilize

  Args:
    info(Info): Configuration object with cluster
    and workload details.
  """
  command = (
      "export KUBECONFIG=/tmp/kubeconfig && "
      f"gsutil cp {info.bucket_path}{info.yaml_file_name} "
      f"/tmp/{info.yaml_file_name} && "
      f"gcloud container clusters get-credentials {info.cluster_name} "
      f"--region {info.location} --project {info.project_id} && "
      f"kubectl --kubeconfig $KUBECONFIG apply -f "
      f"/tmp/{info.yaml_file_name} -n default"
  )

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )

  logging.info("STDOUT message: %s", process.stdout)
  logging.info("STDERR message: %s", process.stderr)


@task
def wait(seconds: int):
  """sleeps for a given number of seconds.

  Args:
    seconds(int): The number of seconds to sleep for.
  """

  command = f"sleep {seconds}"

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )

  logging.info("STDOUT message: %s", process.stdout)
  logging.info("STDERR message: %s", process.stderr)


@task
def end_workload(info: Info):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info(Info): Configuration object with cluster details.
  """
  command = (
      "export KUBECONFIG=/tmp/kubeconfig && "
      f"gcloud container clusters get-credentials {info.cluster_name} "
      f"--region {info.location} --project {info.project_id} && "
      "kubectl delete jobsets --all -n default --timeout=60s"
  )

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )

  logging.info("STDOUT message: %s", process.stdout)
  logging.info("STDERR message: %s", process.stderr)


@task.sensor(poke_interval=60, timeout=3600, mode="reschedule")
def wait_for_jobset_ttr(info: Info) -> bool:
  """A sensor task which polls the jobset time_between_interruptions metric
  every 60 seconds for 60 minutes.

  Args:
      info(Info): An instance of the Info class that encapsulates
      the configuration and metadata of a GKE node pool and workload.
  """

  now = int(time.time())
  api_client = monitoring_v3.MetricServiceClient()
  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{info.project_id}",
      filter=(
          'metric.type="kubernetes.io/jobset/times_to_recover" '
          f'resource.labels.cluster_name="{info.cluster_name}" '
      ),
      interval=monitoring_v3.TimeInterval({
          # This particular metric takes a long time to update
          # to GCP, typically around 20-30 minutes.
          # This means that the sensor must be long running and
          # have a long search period to detect it.
          "end_time": {"seconds": now},
          "start_time": {"seconds": now - 3600},
      }),
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )
  page_result = api_client.list_time_series(request=request)

  # We just need to know that the even happened at all
  if page_result.time_series:
    logging.info("Event detected at %s", now)
    return True
  else:
    logging.info("No time series found at %s. Continuing...", now)
  return False


with models.DAG(
    dag_id="jobset_rollback_ttr-new",
    start_date=datetime.datetime(2025, 8, 10),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu_obervability",
        "rollback",
    ],
    description=(
        "This DAG tests the use of a node-pool rollback to interrupt a "
        "jobset, then polls the jobset time-to-recover metric to check "
        "if it is updated."
    ),
    doc_md="""
  # JobSet Time-To-Recover (TTR) Test Using Node-Pool Rollback

  ### Description
  This DAG automates the process of creating a node-pool, launching a jobset
  then using a node-pool rollback to interrupt the node-pool, and afterwards
  monitors if the jobset TTR metric gets updated. Finally the DAG cleans up the
  jobset and node-pool which were created.

  ### Prerequisites
  This test requires an existing cluster and a jobset file to run.

  ### Procedures
  First the node-pool is created, a jobset yaml is then launched on the cluster
  and given a short period of time to initiate. After this a rollback is run on
  the previously created node-pool to interrupt it. A sensor is finally run
  which will either detect that the jobset time-to-recover metric has been
  updated, resulting in a success, or timeout, and fail.
  """,
) as dag:
  cluster_info = Info(
      project_id=Project.TPU_PROD_ENV_ONE_VM.value,
      cluster_name=Variable.get(
          "CLUSTER_NAME", default_var="qmcgarry-auto-test"
      ),
      node_pool_name=Variable.get(
          "NODE_POOL_NAME", default_var="nodepool-auto"
      ),
      location=Variable.get(
          "LOCATION", default_var=Region.ASIA_NORTHEAST1.value
      ),
      yaml_file_name=Variable.get(
          "YAML_FILE_NAME", default_var="workload.yaml"
      ),
      yaml_path=Variable.get(
          "YAML_PATH",
          default_var="gs://cienet-tpu-observability-airflow/workloads/",
      ),
  )

  create_node_pool = node_pool.create(node_pool=cluster_info)

  start_workload = run_workload(info=cluster_info)

  wait_three_minutes = wait(seconds=180)

  rollback_node_pool = node_pool.rollback(node_pool=cluster_info)

  wait_for_metric_upload = wait_for_jobset_ttr(info=cluster_info)

  cleanup_workload = end_workload.override(trigger_rule=TriggerRule.ALL_DONE)(
      info=cluster_info
  )

  cleanup_node_pool = node_pool.delete.override(trigger_rule="all_done")(
      node_pool=cluster_info
  ).as_teardown(
      setups=create_node_pool,
  )

  (
      create_node_pool
      >> start_workload
      >> wait_three_minutes
      >> rollback_node_pool
      >> wait_for_metric_upload
      >> cleanup_workload
      >> cleanup_node_pool
  )
