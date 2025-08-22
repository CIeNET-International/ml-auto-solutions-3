"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

import dataclasses
import datetime
import logging
import re
from typing import List

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.utils.trigger_rule import TriggerRule
from google.cloud import logging_v2 as gcp_logging
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types

from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants


@dataclasses.dataclass
class Info:
  """Configuration for the GKE Node Pool and Monitoring.

  Attributes:
    project_id: The Google Cloud project ID.
    region: The region of the GKE cluster.
    zone: The zone of the GKE cluster.
    cluster_name: The name of the GKE cluster.
    container_name: The name of the container running the workload.
    yaml_file_name: The name of the YAML file defining the Kubernetes job.
    bucket_path: The GCS bucket path where the YAML file is stored.
  """

  project_id: str
  region: str
  zone: str
  cluster_name: str
  container_name: str
  yaml_file_name: str
  bucket_path: str


def compare_tensorcore_utilization_values(
    log_values: List[float],
    monitoring_values: List[float],
    tolerance: float = 1.0,
) -> bool:
  """Compares two lists of utilization values within a given tolerance.

  This function iterates through two lists of floating-point numbers,
  representing utilization metrics from logs and monitoring systems. It checks
  if the absolute difference between each corresponding pair of values is
  within the specified tolerance.

  Args:
    log_values: A list of utilization values extracted from Cloud Logging.
    monitoring_values: A list of utilization values from Cloud Monitoring.
    tolerance: The maximum allowed absolute difference between corresponding
      values for the comparison to be considered a "PASS".

  Returns:
    True if all value pairs are within the tolerance, False otherwise.

  Raises:
    AirflowException: If the lengths of the two input lists do not match.
  """
  if len(log_values) != len(monitoring_values):
    raise AirflowException(
        f"Data count mismatch. Data count mismatch. Logs have {log_values} values, "
        f"Monitoring has {monitoring_values}."
    )

  logging.info("--- Comparison Results ---")
  logging.info("Tolerance: %s", tolerance)
  logging.info(
      "%s%s%s%s%s",
      f"{'Device':<12}",
      f"{'Log Value':<12}",
      f"{'Monitor Value':<15}",
      f"{'Difference':<12}",
      f"{'Result':<10}",
  )
  logging.info("-" * 65)

  all_passed = True
  for i, (log_val, mon_val) in enumerate(zip(log_values, monitoring_values)):
    diff = abs(log_val - mon_val)
    passed = diff < tolerance
    if not passed:
      all_passed = False
    logging.info(
        "%-12s %-12.2f %-15.2f %-12.2f %-10s",
        f"Device {i}",
        log_val,
        mon_val,
        diff,
        "PASS" if passed else "FAIL",
    )

  logging.info("-" * 65)

  return all_passed


@task.bash()
def run_workload(info: Info):
  """Applies the workload YAML to the GKE cluster and returns the start time.

  This task executes a bash script that performs the following steps:
  1. Copies the workload YAML file from GCS to a temporary local path.
  2. Authenticates `gcloud` with the specified GKE cluster.
  3. Applies the YAML file to the `default` namespace using `kubectl`.
  4. Echoes the current UTC time, which serves as the job's start time.

  Args:
    info: Configuration object with cluster and workload details.

  Returns:
    The UTC timestamp (ISO 8601 format) of when the job was applied.
  """
  return f"""
    export KUBECONFIG=/tmp/kubeconfig
    gsutil cp {info.bucket_path}{info.yaml_file_name} \\
            /tmp/{info.yaml_file_name}

    gcloud container clusters get-credentials {info.cluster_name} \
        --region {info.region} \
        --project {info.project_id}

    kubectl --kubeconfig $KUBECONFIG apply -f /tmp/{info.yaml_file_name} -n default

    echo $(date -u +"%Y-%m-%dT%H:%M:%S.%3N%:z")
    """


@task.bash()
def end_workload(info: Info):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info: Configuration object with cluster details.
  """
  return f"""
    export KUBECONFIG=/tmp/kubeconfig

    gcloud container clusters get-credentials {info.cluster_name} \
        --region {info.region} \
        --project {info.project_id}

    kubectl delete jobsets --all -n default --timeout=60s # Add a timeout for deletion
    """


@task.bash()
def get_active_nodes_bash(info: Info):
  """Retrieves a list of GKE nodes where the workload pods are running.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Use `kubectl` to get the names of the nodes for all pods in the
     `default` namespace.
  3. Formats the output as a space-separated string of node names.

  Args:
    info: Configuration object with cluster details.

  Returns:
    A string containing the names of the active nodes, separated by spaces.
  """
  return f"""
    export KUBECONFIG=/tmp/kubeconfig

    echo "Configuring kubectl to connect to cluster: {info.cluster_name}"
    gcloud container clusters get-credentials {info.cluster_name} \
        --region {info.region} \
        --project {info.project_id}

    NODE_LIST_JSON=$(kubectl get pods -n default -o jsonpath='{{.items[*].spec.nodeName}}' | tr ' ' '\\n')
    echo $NODE_LIST_JSON
    """


@task.sensor(poke_interval=30, timeout=600, mode="reschedule")
def wait_for_jobset_start_logs(info: Info, job_apply_time_str: str) -> bool:
  """Waits for the first log entry indicating the job has started.

  This task polls Cloud Logging for a specific log pattern that appears
  shortly after the TPU job begins execution within the specified container.
  It times out if no such log is found within a defined period.

  Args:
    info: An Info dataclass instance containing project and cluster details.
    job_apply_time_str: The ISO formatted string of the time the job was
      applied.

  Returns:
    True if the start log is found, otherwise it will raise an Airflow timeout
    exception.
  """
  log_client = gcp_logging.Client(project=info.project_id)
  datetime_job_apply_time = datetime.datetime.fromisoformat(job_apply_time_str)

  lql_query = (
      f'logName="projects/{info.project_id}/logs/stdout" AND '
      'resource.type="k8s_container" AND '
      f'resource.labels.cluster_name="{info.cluster_name}" AND '
      f'resource.labels.container_name="{info.container_name}" AND '
      'textPayload =~ "printTimestamp.*"'
  )
  full_filter = f'timestamp>="{job_apply_time_str}" AND ({lql_query})'

  response_iterator = log_client.list_entries(
      filter_=full_filter, order_by=gcp_logging.DESCENDING, max_results=1
  )
  latest_entry = next(iter(response_iterator), None)

  return latest_entry and latest_entry.timestamp > datetime_job_apply_time


@task
def format_node_list(bash_output: str) -> List[str]:
  """Converts a space-separated string of node names into a Python list.

  Args:
    bash_output: The raw string output from the `get_active_nodes_bash` task.

  Returns:
    A list of node name strings. Returns an empty list if the input is empty.
  """
  if not bash_output or not bash_output.strip():
    logging.warning("Received empty node list from bash task.")
    return []

  node_list = bash_output.strip().split()

  return node_list


@task
def verify_tensorcore_utilization(
    info: Info, node_name: str, job_apply_time: str
) -> bool:
  """Fetches and compares TensorCore utilization from logs and monitoring.

  For a single GKE node, this function queries both Cloud Logging and Cloud
  Monitoring to retrieve TensorCore utilization metrics that were generated
  after the job started. It then compares these two sets of data to verify
  their consistency.

  Args:
    info: Configuration object with project and cluster details.
    node_name: The name of the GKE node to verify.
    job_apply_time: The ISO timestamp string indicating when the workload began.

  Returns:
    True if the utilization values from logs and monitoring match within the
    defined tolerance, False otherwise.
  """
  datetime_job_apply_time = datetime.datetime.fromisoformat(job_apply_time)
  end_time_utc = datetime_job_apply_time + datetime.timedelta(minutes=10)

  log_client = gcp_logging.Client(project=info.project_id)

  lql_query = (
      f'logName="projects/{info.project_id}/logs/stdout" AND '
      'resource.type="k8s_container" AND '
      f'resource.labels.cluster_name="{info.cluster_name}" AND '
      f'resource.labels.container_name="{info.container_name}" AND '
      f'labels."compute.googleapis.com/resource_name":"{node_name}"'
  )
  full_filter = f'timestamp>="{job_apply_time}" AND ({lql_query})'
  logging.info("Executing log query for node %s:%s", node_name, full_filter)

  response_iterator = log_client.list_entries(
      filter_=full_filter, order_by=gcp_logging.ASCENDING
  )
  util_values, search_timestamp, in_tensorcore_section = [], None, False
  for entry in response_iterator:
    log_text = entry.payload
    if not isinstance(log_text, str):
      continue
    if not search_timestamp:
      ts_match = re.search(r"printTimestamp:\s*(\d+)", log_text)
      if ts_match:
        search_timestamp = int(ts_match.group(1))
    if "TensorCore Utilization" in log_text:
      in_tensorcore_section = True
      continue
    if in_tensorcore_section:
      match = re.search(r"│\s*\d+\s*│\s*([\d.]+)%", log_text)
      if match:
        util_values.append(float(match.group(1)))
    if len(util_values) == 4:
      break

  mon_client = monitoring_v3.MetricServiceClient()
  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{info.project_id}",
      filter=(
          "metric.type ="
          ' "kubernetes.io/node/accelerator/tensorcore_utilization" AND'
          f' resource.labels.cluster_name = "{info.cluster_name}" AND'
          f' resource.labels.node_name = "{node_name}"'
      ),
      interval=types.TimeInterval({
          "end_time": {"seconds": int(end_time_utc.timestamp())},
          "start_time": {
              "seconds": int(datetime_job_apply_time.timestamp())
          },
      }),
      view="FULL",
  )
  time_series_data = mon_client.list_time_series(request)
  metric_values = {}
  for ts in time_series_data:
    accelerator_id = ts.metric.labels["accelerator_id"]
    closest_point, min_diff = None, float("inf")
    for point in ts.points:
      diff = abs(point.interval.end_time.timestamp() - search_timestamp)
      if diff < min_diff:
        min_diff, closest_point = diff, point
    if closest_point:
      metric_values[accelerator_id] = round(closest_point.value.double_value, 2)

  if not metric_values:
    logging.error("No matching monitoring data found for node %s.", node_name)
    return False

  monitoring_values = [
      metric_values[key]
      for key in sorted(
          metric_values.keys(), key=lambda x: int(x.split("-")[-1])
      )
  ]
  return compare_tensorcore_utilization_values(util_values, monitoring_values)


@task
def summarize_results(
    verification_results: List[bool], active_nodes: List[str]
):
  """Summarizes the results of the TensorCore utilization verification.

  This function logs the number of nodes verified, how many passed, and returns
  a boolean indicating the overall success of the verification process.

  Args:
    verification_results: A list of booleans, where each boolean indicates
      whether the verification passed (True) or failed (False) for a node.
    active_nodes: A list of node names that were included in the verification.

  Returns:
    True if all nodes passed verification, False otherwise. If no nodes were
    active, the task is skipped and does not affect the DAG's final state.
  """
  if not active_nodes:
    logging.info("No active nodes were found. Grand Result: SKIPPED")
    return

  total_successful_nodes = sum(1 for result in verification_results if result)

  logging.info("Total nodes verified: %d", len(active_nodes))
  logging.info("Nodes that passed verification: %d", total_successful_nodes)

  return total_successful_nodes == len(active_nodes)


with models.DAG(
    dag_id="tpu_info_tensorcore_utilization_dag",
    start_date=datetime.datetime(2025, 8, 15),
    schedule=constants.Schedule.WEEKDAY_PST_6_30PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-info", "tensorcore-utilization"],
    description=(
        "This DAG verifies TensorCore utilization metrics by comparing data"
        " from Cloud Logging and Cloud Monitoring."
    ),
    doc_md="""
      # TensorCore Utilization Verification DAG
      # This DAG verifies TensorCore utilization metrics by comparing data from Cloud Logging and Cloud Monitoring.""",
) as dag:
  cluster_info = Info(
      project_id=models.Variable.get(
          "TCU_PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
      ),
      cluster_name=models.Variable.get(
          "TCU_CLUSTER_NAME", default_var="yuna-xpk-v6e"
      ),
      region=models.Variable.get(
          "TCU_REGION", default_var=Region.ASIA_NORTHEAST1.value
      ),
      zone=models.Variable.get(
          "TCU_ZONE", default_var=Zone.ASIA_NORTHEAST1_B.value
      ),
      yaml_file_name=models.Variable.get(
          "YAML_FILE_NAME", default_var="v6e-tpu-info-workload.yaml"
      ),
      bucket_path=models.Variable.get(
          "BUCKET_PATH",
          default_var="gs://us-east1-dennis-airflow-tes-a24588e9-bucket/data/",
      ),
      container_name=models.Variable.get(
          "CONTAINER_NAME", default_var="jax-tpu-job"
      ),
  )
  # Clean up any pre-existing workloads to ensure a clean environment for the
  # test.
  start_cleanup = end_workload.override(
      task_id="start_cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
  )(info=cluster_info)

  apply_time = run_workload.override(task_id="run_workload")(info=cluster_info)

  wait_for_job_start = wait_for_jobset_start_logs.override(
      task_id="wait_for_job_start"
  )(cluster_info, job_apply_time_str=apply_time)

  raw_node_names_str = get_active_nodes_bash.override(
      task_id="get_raw_node_names"
  )(info=cluster_info)

  active_node = format_node_list(bash_output=raw_node_names_str)

  verify_utilization_per_node = (
      verify_tensorcore_utilization.override(
          task_id="verify_utilization_per_node"
      )
      .partial(info=cluster_info, job_apply_time=apply_time)
      .expand(node_name=active_node)
  )

  clean_up = end_workload.override(
      task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
  )(info=cluster_info)

  summary = summarize_results(verify_utilization_per_node, active_node)

  apply_time >> wait_for_job_start >> raw_node_names_str >> active_node

  summary >> clean_up
