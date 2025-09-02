"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

import dataclasses
import datetime
import logging
import os
import re
import subprocess
from typing import List, Tuple
import random

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.utils.trigger_rule import TriggerRule
from dags.common.vm_resource import Project
from dags.common.vm_resource import Region
from dags.common.vm_resource import Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils.monitoring import query_time_series

from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types


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


def compare_metric_values(
    cmd_values: List[float],
    monitoring_values: List[float],
    pod_name: str,
    tolerance: float = 1.0,
):
  """Compares two lists of utilization values within a given tolerance."""
  if len(cmd_values) != len(monitoring_values):
    raise AirflowException(
        f"For pod {pod_name}, data count mismatch. TPU-Info has"
        f" {len(cmd_values)} values, Monitoring has {len(monitoring_values)}."
    )

  logging.info("--- Comparison Results for pod: %s ---", pod_name)
  logging.info(
      "%-12s%-15s%-17s%-12s%-10s",
      "Device",
      "TPU-Info Val",
      "Monitoring Val",
      "Difference",
      "Result",
  )
  logging.info("-" * 70)

  all_passed = True
  for i, (log_val, mon_val) in enumerate(zip(cmd_values, monitoring_values)):
    diff = abs(log_val - mon_val)
    passed = diff <= tolerance
    if not passed:
      all_passed = False
    logging.info(
        "%-12s%-15.2f%-17.2f%-12.2f%-10s",
        f"Device {i}",
        log_val,
        mon_val,
        diff,
        "PASS" if passed else "FAIL",
    )
  logging.info("-" * 70)

  if not all_passed:
    raise AirflowException(
        f"Overall Result for Pod {pod_name}: FAIL - Utilization values do not"
        " match within tolerance."
    )
  logging.info("Overall Result for Pod %s: PASS", pod_name)


@task
def run_workload(info: Info, kubeconfig: str, yaml_path: str):
  """Applies the workload YAML to the GKE cluster using subprocess.

  This task executes a series of shell commands using Python's subprocess
  module to perform the following steps:
  1. Defines temporary paths for kubeconfig and the YAML file.
  2. Copies the workload YAML file from GCS to the local temp path.
  3. Authenticates gcloud and gets credentials for the specified GKE cluster,
      storing them in the temporary kubeconfig file.
  4. Applies the YAML file to the `default` namespace using kubectl, pointing
      to the temporary kubeconfig.
  5. Returns the current UTC time as the job's start time, generated in Python.

  Args:
      info: Configuration object with cluster and workload details.
      kubeconfig: The path to the kubeconfig file.
      yaml_path: The local path where the YAML file will be copied.

  Returns:
      The UTC timestamp (ISO 8601 format) of when the job was applied.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  gsutil_cmd = f"gsutil cp {info.bucket_path}{info.yaml_file_name} {yaml_path}"
  process = subprocess.run(
      gsutil_cmd, shell=True, check=True, capture_output=True, text=True
  )
  logging.info("STDOUT message: %s", process.stdout)
  logging.info("STDERR message: %s", process.stderr)

  gcloud_cmd = (
      f"gcloud container clusters get-credentials {info.cluster_name} "
      f"--region={info.region} "
      f"--project={info.project_id} "
  )

  subprocess.run(
      gcloud_cmd,
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )
  kubectl_cmd = (
      f"kubectl --kubeconfig={kubeconfig} apply -f {yaml_path} " "-n default"
  )
  subprocess.run(
      kubectl_cmd, shell=True, check=True, capture_output=True, text=True
  )
  logging.info("STDOUT message: %s", process.stdout)
  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  current_time_utc_format = current_time_utc.isoformat(timespec="milliseconds")
  return current_time_utc_format


@task
def end_workload(info: Info, kubeconfig: str):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  gcloud_cmd = (
      f"gcloud container clusters get-credentials {info.cluster_name} "
      f"--region={info.region} "
      f"--project={info.project_id} "
  )

  subprocess.run(
      gcloud_cmd,
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )

  kubectl_cmd = (
      f"kubectl --kubeconfig={kubeconfig} delete jobsets --all -n default"
      " --timeout=60s --ignore-not-found=true"
  )
  subprocess.run(
      kubectl_cmd, shell=True, check=True, capture_output=True, text=True
  )


@task
def get_active_pods(info: Info, kubeconfig: str):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  gcloud_cmd = (
      f"gcloud container clusters get-credentials {info.cluster_name} "
      f"--region={info.region} "
      f"--project={info.project_id} "
  )

  subprocess.run(
      gcloud_cmd,
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )

  kubectl_cmd = (
      f"kubectl --kubeconfig={kubeconfig} get pods -n default -o"
      " jsonpath={.items[*].metadata.name}"
  )
  process = subprocess.run(
      kubectl_cmd, shell=True, check=True, capture_output=True, text=True
  )
  if not process or not process.stdout.strip():
    logging.warning("Received empty pod list from bash task.")
    raise AirflowException("Received empty pod list from bash task.")

  pod_list = process.stdout.strip().split()
  return pod_list


@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def query_to_wait_for_jobset_start(
    info: Info, pod_name_list: str, job_apply_time: str
) -> bool:
  """Waits for the first log entry indicating the job has started.

  This task polls Cloud Logging for a specific log pattern that appears
  shortly after the TPU job begins execution within the specified container.
  It times out if no such log is found within a defined period.

  Args:
    info: An Info dataclass instance containing project and cluster details.
    pod_name_list: A list of pod names.
    job_apply_time: The ISO formatted string of the time the job was applied.

  Returns:
    True if the start log is found, otherwise it will raise an Airflow timeout
    exception.
  """

  datetime_job_apply_time = datetime.datetime.fromisoformat(job_apply_time)
  end_time_utc = datetime_job_apply_time + datetime.timedelta(minutes=10)

  if not pod_name_list:
    raise AirflowException("pod_name_list is empty, sensor cannot proceed.")

  pod_name = random.choice(pod_name_list)
  filter_string = (
      "metric.type ="
      ' "kubernetes.io/container/accelerator/tensorcore_utilization" AND'
      f' resource.labels.cluster_name = "{info.cluster_name}" AND'
      f' resource.labels.pod_name = "{pod_name}"'
  )
  time_series_data = query_time_series(
      project_id=info.project_id,
      filter_str=filter_string,
      start_time=datetime_job_apply_time,
      end_time=end_time_utc,
      view="FULL",
  )
  time_series_data_list = list(time_series_data)

  # Retrieve the last three records to ensure stable workload startup.
  if not time_series_data_list or len(time_series_data_list[0].points) < 3:
    return False
  last_n_data_points = [
      round(point.value.double_value, 2)
      for point in time_series_data_list[0].points[0:3]
  ]
  minimal_activity_threshold = 1.0
  return all(p > minimal_activity_threshold for p in last_n_data_points)


@task
def get_tpu_info_from_pod(kubeconfig: str, pod_name: str) -> str:
  """Executes the 'tpu-info' command within a specified pod and returns its output.

  This task uses kubectl to run the 'tpu-info' command inside the given pod
  in the 'default' namespace. The output of the command is captured and
  returned.

  Args:
    kubeconfig: The path to the kubeconfig file.
    pod_name: The name of the pod to execute the command in.

  Returns:
    The standard output from the 'tpu-info' command.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  command_string = (
      f"kubectl --kubeconfig={kubeconfig} "
      f"exec {pod_name} -n default "
      f"-- "
      f"tpu-info"
  )

  result = subprocess.run(
      command_string,
      shell=True,
      # Since tpu-info feature still has some issues, so the command will
      # inevitably throw an error. To avoid marking the task as failed,
      # I set check to False so that the task status does not show as failed.
      check=False,
      capture_output=True,
      text=True,
  )
  print("STDOUT:", result.stdout)
  return result.stdout


@task
def fetch_parse_and_compare_utilization(
    info: Info,
    job_apply_time: str,
    comparison_data: Tuple[str, str],) -> bool:
  """Parses outputs from Monitoring and tpu-info, then compares them."""
  pod_name, tpu_info_text = comparison_data
  logging.info("Getting monitoring data for pod: %s...", pod_name)

  datetime_job_apply_time = datetime.datetime.fromisoformat(job_apply_time)
  end_time_utc = datetime_job_apply_time + datetime.timedelta(minutes=10)

  filter_string = (
      "metric.type ="
      ' "kubernetes.io/container/accelerator/tensorcore_utilization" AND'
      f' resource.labels.cluster_name = "{info.cluster_name}" AND'
      f' resource.labels.pod_name = "{pod_name}"'
  )

  time_series_data = query_time_series(
      project_id=info.project_id,
      filter_str=filter_string,
      start_time=datetime_job_apply_time,
      end_time=end_time_utc,
  )

  ts_list = list(time_series_data)
  if not ts_list or not all(ts.points for ts in ts_list):
    raise AirflowException(
        f"Could not retrieve valid monitoring data for pod {pod_name}."
    )

  metric_values = {}
  for ts in time_series_data:
    accelerator_id = ts.metric.labels["accelerator_id"]
    if ts.points:
      point = ts.points[0]
      metric_values[accelerator_id] = round(point.value.double_value, 2)

  if not metric_values:
    raise AirflowException(
        "Failed to extract metric values from monitoring data for pod"
        f" {pod_name}."
    )

  monitoring_values = [
      metric_values[key]
      for key in sorted(
          metric_values.keys(), key=lambda x: int(x.split("-")[-1])
      )
  ]

  util_values = []
  in_tensorcore_section = False
  for line in tpu_info_text.strip().split("\n"):
    if "TensorCore Utilization" in line:
      in_tensorcore_section = True
      continue
    if in_tensorcore_section:
      match = re.search(r"│\s*\d+\s*│\s*([\d.]+)%", line)
      if match:
        util_values.append(float(match.group(1)))

  if not util_values:
    raise AirflowException(
        "Failed to parse TensorCore utilization from tpu-info output for pod"
        f" {pod_name}."
    )

  compare_metric_values(
      util_values, monitoring_values, pod_name
  )
  return True


@task
def summarize_results(verification_results: List[bool], active_pods: List[str]):
  """Summarizes the results of the TensorCore utilization verification.

  This function logs the number of nodes verified, how many passed, and returns
  a boolean indicating the overall success of the verification process.

  Args:
    verification_results: A list of booleans, where each boolean indicates
      whether the verification passed (True) or failed (False) for a node.
    active_pods: A list of pod names that were included in the verification.

  Returns:
    True if all nodes passed verification, False otherwise. If no nodes were
    active, the task is skipped and does not affect the DAG's final state.
  """
  if not active_pods:
    logging.info("No active nodes were found. Grand Result: SKIPPED")
    return
  total_successful_pods = len(verification_results)
  total_expected_pods = len(active_pods)

  logging.info("--- Overall Verification Summary ---")
  logging.info("Total pods scheduled for verification: %d", total_expected_pods)
  logging.info(
      "Pods that passed verification (succeeded): %d", total_successful_pods
  )

  if total_successful_pods != total_expected_pods:
    raise AirflowException(
        "Grand Result: FAILURE - The number of passed comparisons "
        f"({total_successful_pods}) did not meet the threshold of "
        f"{total_expected_pods}. Active pods: {active_pods}"
    )


with models.DAG(
    dag_id="tpu_info_tensorcore_utilization_dag_test",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=constants.Schedule.WEEKDAY_PST_6_30PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info", "tensorcore-utilization"],
    description=(
        "This DAG verifies TensorCore utilization metrics by comparing data"
        " from Cloud Logging and Cloud Monitoring."
    ),
    doc_md="""
      # TensorCore Utilization Verification DAG
      # This DAG verifies TensorCore utilization metrics by comparing data from
      # Cloud Logging and Cloud Monitoring.""",
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

  kubeconfig_path = "/tmp/kubeconfig"
  local_yaml_path = f"/tmp/{cluster_info.yaml_file_name}"
  # Clean up any pre-existing workloads to ensure a clean environment for the
  # test.
  start_cleanup = end_workload.override(
      task_id="start_cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
  )(info=cluster_info, kubeconfig=kubeconfig_path)

  apply_time = run_workload.override(task_id="run_workload")(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_path=local_yaml_path,
  )

  active_pods = get_active_pods.override(task_id="get_active_pod")(
      info=cluster_info, kubeconfig=kubeconfig_path
  )

  wait_for_job_start = query_to_wait_for_jobset_start.override(
      task_id="wait_for_job_start"
  )(cluster_info, pod_name_list=active_pods, job_apply_time=apply_time)

  tpu_info_outputs = (
      get_tpu_info_from_pod.override(task_id="get_tpu_info")
      .partial(kubeconfig=kubeconfig_path)
      .expand(pod_name=active_pods)
  )

  verify_utilization_per_pod = fetch_parse_and_compare_utilization.partial(
      info=cluster_info,
      job_apply_time=apply_time
  ).expand(
      comparison_data=active_pods.zip(tpu_info_outputs)
  )

  summary = summarize_results.override(
      task_id="summarize_results", trigger_rule=TriggerRule.ALL_DONE
  )(verify_utilization_per_pod, active_pods)

  clean_up = end_workload.override(
      task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
  )(info=cluster_info, kubeconfig=kubeconfig_path).as_teardown(
      setups=apply_time
  )

  (
      start_cleanup
      >> apply_time
      >> active_pods
      >> wait_for_job_start
      >> tpu_info_outputs
      >> verify_utilization_per_pod
  )

  summary >> clean_up
