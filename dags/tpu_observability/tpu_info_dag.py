"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""
from abc import ABC, abstractmethod
import dataclasses
import datetime
import logging
import os
import random
import re
import subprocess
from typing import Dict, List, Tuple

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from google.cloud.monitoring_v3 import types

from dags.common.vm_resource import Project
from dags.common.vm_resource import Region
from dags.common.vm_resource import Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils.jobset_yaml_generator import create_jobset_yaml
from dags.tpu_observability.utils.jobset_yaml_generator import YamlConfig
from dags.tpu_observability.utils.monitoring import query_time_series
from dags.tpu_observability.utils.node_pool_util import Info


class BaseMetricStrategy(ABC):
  """Abstract Base Class (Interface) for a metric verification strategy.

  It defines the contract that all concrete metric strategies must follow.
  """

  @property
  @abstractmethod
  def metric_name(self) -> str:
    """The name of the metric as it appears in the Monitoring filter."""
    pass

  @abstractmethod
  def parse_from_monitoring(
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
    """Parses the desired value from a list of TimeSeries objects."""
    pass

  @abstractmethod
  def parse_from_tpu_info(self, tpu_info_text: str) -> List[float]:
    """Parses the desired value from the raw tpu-info command output."""
    pass


class TensorcoreUtilizationStrategy(BaseMetricStrategy):
  """Strategy for verifying TensorCore Utilization."""

  @property
  def metric_name(self) -> str:
    return "tensorcore_utilization"

  def parse_from_monitoring(
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
    metric_values = {}
    for ts in time_series_data:
      if ts.points:
        accelerator_id = ts.metric.labels["accelerator_id"]
        point = ts.points[0]
        metric_values[accelerator_id] = round(point.value.double_value, 2)
    return [
        metric_values[key]
        for key in sorted(
            metric_values.keys(), key=lambda x: int(x.split("-")[-1])
        )
    ]

  def parse_from_tpu_info(self, tpu_info_text: str) -> List[float]:
    util_values = []
    in_section = False
    for line in tpu_info_text.strip().split("\n"):
      if "TensorCore Utilization" in line:
        in_section = True
        continue
      if in_section:
        match = re.search(r"│\s*\d+\s*│\s*([\d.]+)%", line)
        if match:
          util_values.append(float(match.group(1)))
    return util_values


class MemoryUsedStrategy(BaseMetricStrategy):
  """Strategy for verifying Used HBM Memory."""

  @property
  def metric_name(self) -> str:
    return "memory_used"

  def parse_from_monitoring(
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
    metric_values = {}
    for ts in time_series_data:
      if ts.points:
        accelerator_id = ts.metric.labels["accelerator_id"]
        point = ts.points[0]
        bytes_value = point.value.int64_value
        gib_value = bytes_value / (1024**3)
        metric_values[accelerator_id] = round(gib_value, 2)
    return [
        metric_values[key]
        for key in sorted(
            metric_values.keys(), key=lambda x: int(x.split("-")[-1])
        )
    ]

  def parse_from_tpu_info(self, tpu_info_text: str) -> List[float]:
    util_values = []
    in_section = False
    for line in tpu_info_text.strip().split("\n"):
      if "TPU Runtime Utilization" in line:
        in_section = True
        continue
      if in_section:
        match = re.search(r"(\d+\.\d+)\s*GiB\s*\/\s*(\d+\.\d+)\s*GiB", line)
        if match:
          util_values.append(float(match.group(1)))
      if len(util_values) == 4:
        break
    return util_values


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
def run_workload(info: Info, kubeconfig: str, yaml_config: YamlConfig):
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
      yaml_config: The YamlConfig object containing namespace information.

  Returns:
      The UTC timestamp (ISO 8601 format) of when the job was applied.
  """
  params = dataclasses.asdict(yaml_config)
  base_job_name = yaml_config.jobset_name

  logging.info("Generating YAML content for JobSet: %s", base_job_name)

  yaml_content = create_jobset_yaml(**params)

  yaml_path = f"/tmp/{base_job_name}.yaml"
  with open(yaml_path, "w") as f:
    f.write(yaml_content)
  logging.info("Successfully generated YAML file at: %s", yaml_path)
  with open(yaml_path, "r") as f:
    logging.info("--- File Content ---\n%s", f.read())

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
      f"kubectl --kubeconfig={kubeconfig} apply -f {yaml_path} -n"
      f" {yaml_config.namespace}"
  )
  subprocess.run(
      kubectl_cmd, shell=True, check=True, capture_output=True, text=True
  )
  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  current_time_utc_format = current_time_utc.isoformat(timespec="milliseconds")
  return current_time_utc_format


@task
def end_workload(info: Info, kubeconfig: str, yaml_config: YamlConfig):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    yaml_config: The YamlConfig object containing namespace information.
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
      f"kubectl --kubeconfig={kubeconfig} delete jobsets --all -n"
      f" {yaml_config.namespace} --timeout=60s --ignore-not-found=true"
  )
  subprocess.run(
      kubectl_cmd, shell=True, check=True, capture_output=True, text=True
  )


@task
def get_active_pods(info: Info, kubeconfig: str, yaml_config: YamlConfig):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    yaml_config: The YamlConfig object containing namespace information.
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
      f"kubectl --kubeconfig={kubeconfig} get pods -n"
      f" {yaml_config.namespace} -o jsonpath={{.items[*].metadata.name}}"
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
def run_metric_verification(
    info: Info,
    job_apply_time: str,
    metric_strategy: BaseMetricStrategy,
    comparison_data: Tuple[str, str],
) -> bool:
  """A generic task that uses a strategy object to verify a metric."""
  pod_name, tpu_info_text = comparison_data
  metric_name = metric_strategy.metric_name
  logging.info("Verifying metric '%s' for pod: %s...", metric_name, pod_name)
  datetime_job_apply_time = datetime.datetime.fromisoformat(job_apply_time)
  end_time_utc = datetime_job_apply_time + datetime.timedelta(minutes=10)

  filter_string = (
      f'metric.type = "kubernetes.io/container/accelerator/{metric_name}" '
      f'AND resource.labels.cluster_name = "{info.cluster_name}" '
      f'AND resource.labels.pod_name = "{pod_name}"'
  )
  time_series_data = query_time_series(
      project_id=info.project_id,
      filter_str=filter_string,
      start_time=datetime_job_apply_time,
      end_time=end_time_utc,
  )

  monitoring_values = metric_strategy.parse_from_monitoring(time_series_data)
  util_values = metric_strategy.parse_from_tpu_info(tpu_info_text)

  compare_metric_values(util_values, monitoring_values, pod_name)
  return True


@task
def summarize_results(
    verification_results_dict: Dict[str, List[bool]], active_pods: List[str]
):
  """
  Summarizes the results for multiple metric verifications, checking each
  metric group individually.
  """
  if not active_pods:
    logging.info("No active nodes were found. Grand Result: SKIPPED")
    return

  num_expected_pods = len(active_pods)
  overall_success = True
  failure_summary = []

  logging.info("--- Overall Verification Summary ---")
  logging.info("Total pods scheduled for verification: %s", num_expected_pods)
  logging.info("-" * 70)
  logging.info("%-35s | %-10s | %-20s", "Metric Name", "Result", "Details")
  logging.info("-" * 70)

  for metric_name, results in verification_results_dict.items():
    num_passes = len(results)  # Only successed task return result

    if num_passes < num_expected_pods:
      status = "FAIL"
      details = f"Passed {num_passes} of {num_expected_pods} pods."
      overall_success = False
      failure_summary.append(f"- {metric_name}: {details}")
    else:
      status = "PASS"
      details = f"All {num_expected_pods} pods passed."

    logging.info("%-35s | %-10s | %-20s", metric_name, status, details)

  logging.info("-" * 70)

  if not overall_success:
    error_message = (
        "Grand Result: FAILURE - One or more metric verifications failed.\n"
        "Failure Details:\n" + "\n".join(failure_summary)
    )
    raise AirflowException(error_message)

  logging.info(
      "Grand Result: SUCCESS - All metric verifications passed for all pods."
  )


with models.DAG(
    dag_id="tpu_info_dag",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=constants.Schedule.WEEKDAY_PST_6_30PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info"],
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
          "TPU_INFO_PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
      ),
      cluster_name=models.Variable.get(
          "TPU_INFO_CLUSTER_NAME", default_var="yuna-xpk-v6e"
      ),
      region=models.Variable.get(
          "TPU_INFO_REGION", default_var=Region.ASIA_NORTHEAST1.value
      ),
      zone=models.Variable.get(
          "TPU_INFO_ZONE", default_var=Zone.ASIA_NORTHEAST1_B.value
      ),
      machine_type=models.Variable.get(
          "TPU_INFO_MACHINE_TYPE", default_var="tpu-v6e-slice"
      ),
      tpu_topology=models.Variable.get(
          "TPU_INFO_TPU_TOPOLOGY", default_var="4x4"
      ),
  )

  kubeconfig_path = "/tmp/kubeconfig"
  yaml_config_instance = YamlConfig(
      jobset_name="tpu-info-v6e-workload",
      namespace="default",
      max_restarts=5,
      replicated_job_name="tpu-job-slice",
      replicas=2,
      backoff_limit=0,
      completions=4,
      parallelism=4,
      image="us-docker.pkg.dev/tpu-prod-env-one-vm/yuna-docker-repo/tpu-info:v0.4.0",
      container_name="jax-tpu-job",
      tpu_cores_per_pod=4,
      node_selector={
          "cloud.google.com/gke-tpu-accelerator": cluster_info.machine_type,
          "cloud.google.com/gke-tpu-topology": cluster_info.tpu_topology,
      },
      command=["/bin/bash", "-c"],
      command_args=[
          """
          python -c 'import jax; print("TPU cores:", jax.device_count())'
          python /app/jax_tpu_benchmark.py
          echo "sleep..."
          sleep 10000
          """
      ],
      volume_name="code",
      config_map_name="jax-tpu-benchmark-code-one-tpuinfo-output",
  )

  # Clean up any pre-existing workloads to ensure a clean environment for the
  # test.
  start_cleanup = end_workload.override(
      task_id="start_cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
  )(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=yaml_config_instance,
  )

  apply_time = run_workload.override(task_id="run_workload")(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=yaml_config_instance,
  )

  active_pods = get_active_pods.override(task_id="get_active_pod")(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=yaml_config_instance,
  )

  wait_for_job_start = query_to_wait_for_jobset_start.override(
      task_id="wait_for_job_start"
  )(cluster_info, pod_name_list=active_pods, job_apply_time=apply_time)

  with TaskGroup(group_id="verification_group") as verification_group:
    tpu_info_outputs = (
        get_tpu_info_from_pod.override(task_id="get_tpu_info")
        .partial(kubeconfig=kubeconfig_path)
        .expand(pod_name=active_pods)
    )

    verify_tensorcore = (
        run_metric_verification.override(
            task_id="verify_tensorcore_utilization"
        )
        .partial(
            info=cluster_info,
            job_apply_time=apply_time,
            metric_strategy=TensorcoreUtilizationStrategy(),
        )
        .expand(comparison_data=active_pods.zip(tpu_info_outputs))
    )

    verify_memory_used = (
        run_metric_verification.override(task_id="verify_memory_used")
        .partial(
            info=cluster_info,
            job_apply_time=apply_time,
            metric_strategy=MemoryUsedStrategy(),
        )
        .expand(comparison_data=active_pods.zip(tpu_info_outputs))
    )

    tpu_info_outputs >> [verify_tensorcore, verify_memory_used]

  summary = summarize_results.override(
      task_id="summarize_results", trigger_rule=TriggerRule.ALL_DONE
  )(
      verification_results_dict={
          "TensorCore Utilization": verify_tensorcore,
          "HBM Memory Used": verify_memory_used,
      },
      active_pods=active_pods,
  )

  clean_up = end_workload.override(
      task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
  )(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=yaml_config_instance,
  ).as_teardown(
      setups=apply_time
  )

  (
      start_cleanup
      >> apply_time
      >> active_pods
      >> wait_for_job_start
      >> verification_group
      >> summary
      >> clean_up
  )
