"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

import datetime
import logging
import os
import random
import re
import subprocess
from typing import Dict, Final, List, Tuple

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from dags.common.vm_resource import Project, Region, Zone, MachineVersion
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils.jobset_generator import JobSet
from dags.tpu_observability.utils.jobset_generator import Workload
from dags.tpu_observability.utils.monitoring import query_time_series
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.tpu_info_util import parse_tpu_info_output


MACHINE_TYPE_TO_TPU_VERSION = {
    "ct6e-standard-4t": "v6e",
    "ct5p-hightpu-4t": "v5p",
}


def _get_credentials_command(node_pool: node_pool.Info):
  for attr_name in ["cluster_name", "region", "project_id"]:
    if not getattr(node_pool, attr_name):
      raise ValueError(f"{attr_name} must be set in the Info object.")

  return " ".join([
      "gcloud container clusters",
      f"get-credentials {node_pool.cluster_name}",
      f"--region={node_pool.region}",
      f"--project={node_pool.project_id}",
  ])


def _k8s_apply_jobset_command(
    kubeconfig: str, yaml_content: str, namespace: str
):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} apply",
      f"-f - -n {namespace} <<EOF\n",
      f"{yaml_content}\nEOF",
  ])


def _k8s_delete_jobset_command(kubeconfig: str, namespace: str):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} delete jobsets --all",
      f"-n {namespace} --timeout=60s --ignore-not-found=true",
  ])


def _k8s_get_pod_command(kubeconfig: str, namespace: str):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} get pods",
      f"-n {namespace} -o jsonpath={{.items[*].metadata.name}}",
  ])


@task
def run_workload(
    info: node_pool.Info, kubeconfig: str, yaml_config: JobSet, script: str
) -> TimeUtil:
  """Applies the specified YAML file to the GKE cluster.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    yaml_config: The JobSet object containing YAML configuration.
    script: The workload script to be executed.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  yaml_content = generator.generate_yaml(workload_script=script)

  result = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_apply_jobset_command(
              kubeconfig, yaml_content, yaml_config.namespace
          ),
      ]),
      shell=True,
      check=False,
      env=env,
      capture_output=True,
      text=True,
  )
  logging.info(
      "Command Execute:\n %s",
      " && ".join([
          _get_credentials_command(info),
          _k8s_apply_jobset_command(
              kubeconfig, yaml_content, yaml_config.namespace
          ),
      ]),
  )
  if result.returncode != 0:
    raise AirflowFailException(
        f"Command failed with exit code {result.returncode}.\n ,STDERR"
        f" message: {result.stderr}"
    )
  logging.info("STDOUT message: %s", result.stdout)

  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  return current_time_utc


@task
def end_workload(info: node_pool.Info, kubeconfig: str, yaml_config: JobSet):
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

  result = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_delete_jobset_command(kubeconfig, yaml_config.namespace),
      ]),
      shell=True,
      check=False,
      env=env,
      capture_output=True,
      text=True,
  )
  if result.returncode != 0:
    logging.info("Command failed with exit code %s.", result.returncode)
    logging.info("STDERR message: %s", result.stderr)
  logging.info("STDOUT message: %s", result.stdout)


@task
def get_active_pods(info: node_pool.Info, kubeconfig: str, yaml_config: JobSet):
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

  process = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_get_pod_command(kubeconfig, yaml_config.namespace),
      ]),
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )

  if not process or not process.stdout.strip():
    logging.warning("Received empty pod list from bash task.")
    raise AirflowFailException("Received empty pod list from bash task.")

  pod_list = process.stdout.strip().split()
  return pod_list


@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def wait_for_jobset_started(
    info: node_pool.Info, pod_name_list: str, job_apply_time: datetime.datetime
) -> bool:
  """Waits for the jobset to start by polling Cloud Logging for positive tensorcore utilization metrics.

  This task polls Cloud Logging for a specific log pattern that appears
  shortly after the TPU job begins execution within the specified container.
  It times out if no such log is found within a defined period.

  Args:
    info: An Info dataclass instance containing project and cluster details.
    pod_name_list: A list of pod names.
    job_apply_time: The datetime object of the time the job was applied.
  """

  end_time_datatime = job_apply_time + datetime.timedelta(minutes=10)
  start_time = TimeUtil.from_datetime(job_apply_time)
  end_time = TimeUtil.from_datetime(end_time_datatime)

  if not pod_name_list:
    raise AirflowFailException("pod_name_list is empty, sensor cannot proceed.")

  pod_name = random.choice(pod_name_list)
  metric_name = "kubernetes.io/container/accelerator/tensorcore_utilization"
  filter_string = [
      f'metric.type = "{metric_name}"',
      f'resource.labels.cluster_name = "{info.cluster_name}"',
      f'resource.labels.pod_name = "{pod_name}"',
  ]
  time_series_data = query_time_series(
      project_id=info.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
      view="FULL",
  )

  # The value of this metric means percentage of tensorcore utilization,
  # any positive values can represent that the jobset has started.
  threshold_value: Final[float] = 0.0

  # The minimum number of consecutive initial data points that must all exceed
  # 'threshold_value' to confirm that the jobset has successfully started and
  # is active.
  threshold_records_count: Final[int] = 3

  if (
      not time_series_data
      or len(time_series_data[0].points) < threshold_records_count
  ):
    return False
  last_n_data_points = [
      round(point.value.double_value, 2)
      for point in time_series_data[0].points[0:threshold_records_count]
  ]

  return all(p > threshold_value for p in last_n_data_points)


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

  result = subprocess.run(
      (
          f"kubectl --kubeconfig={kubeconfig} "
          f"exec {pod_name} -n default "
          f"-- "
          f"tpu-info"
      ),
      shell=True,
      # Since tpu-info feature still has some issues, so the command will
      # inevitably throw an error. To avoid marking the task as failed,
      # I set check to False so that the task status does not show as failed.
      check=False,
      capture_output=True,
      text=True,
  )
  logging.info("STDOUT: %s", result.stdout)
  return result.stdout


@task
def verify_table_amount(tpu_info_dict: Dict[str, str]):
  """Verifies if all expected tables are present in the parsed tpu_info dictionary.
  """
  expect_table_keys = {
      "tpu_chips",
      "tpu_runtime_utilization",
      "tensorcore_utilization",
      "tpu_buffer_transfer_latency",
  }

  found_keys = set(tpu_info_dict.keys())

  missing_keys = expect_table_keys - found_keys

  if missing_keys:
    raise AirflowFailException(
        "The following required tables were not found in the tpu-info output: "
        f"{', '.join(sorted(list(missing_keys)))}"
    )

  print("All expected tables were found.")


@task
def validate_chips_table(tpu_info_dict: str, info: node_pool.Info):
  """Validates the row count and content for the 'TPU Chips' table."""
  errors = []
  content = tpu_info_dict["tpu_chips"]["body"]

  expected_rows = 4
  if len(content) != expected_rows:
    raise AirflowFailException(
        f"[TPU Chips] Row count is incorrect. (Actual: {len(content)},"
        f" Expected: {expected_rows})"
    )

  tpu_type = MACHINE_TYPE_TO_TPU_VERSION[info.machine_type]

  for i, row_dict in enumerate(content, 1):
    if not re.match(r"/dev/vfio/\d+", row_dict["Chip"]):
      errors.append(
          f"[TPU Chips] Row {i}: Invalid 'Chip' format: {row_dict['Chip']}"
      )
    if tpu_type not in row_dict["Type"]:
      errors.append(
          f"[TPU Chips] Row {i}: 'Type' column value '{row_dict['Type']}' does"
          f" not contain the expected version '{tpu_type}'."
      )
    if not (row_dict["PID"]).isdigit() and int((row_dict["PID"]) > 0):
      errors.append(
          f"[TPU Chips] Row {i}: 'PID' must be a number greater than 0, got:"
          f" {row_dict['PID']}"
      )
  if errors:
    raise AirflowFailException(errors)


@task
def validate_runtime_table(tpu_info_dict: str):
  """Validates the row count and content for the 'TPU Runtime Utilization' table."""
  errors = []
  content = tpu_info_dict["tpu_runtime_utilization"]["body"]

  expected_rows = 4
  if len(content) != expected_rows:
    raise AirflowFailException(
        f"[Runtime] Row count is incorrect. (Actual: {len(content)}, Expected:"
        f" {expected_rows})"
    )

  for i, row_dict in enumerate(content, 1):
    hbm_match = re.match(
        r"(\d+\.\d+)\s*GiB\s*/\s*(\d+\.\d+)\s*GiB", row_dict["HBM Usage (GiB)"]
    )
    if hbm_match:
      used, total = float(hbm_match.group(1)), float(hbm_match.group(2))
      if used > total:
        errors.append(
            f"[Runtime] Row {i}: Used HBM ({used}) cannot be greater than Total"
            f" HBM ({total})."
        )
    else:
      errors.append(
          f"[Runtime] Row {i}: Invalid 'HBM Usage' format:"
          f" {row_dict['HBM Usage (GiB)']}"
      )

    duty_match = re.match(r"(\d+\.\d+)%", row_dict["Duty cycle"])
    if not (duty_match and 0.0 <= float(duty_match.group(1)) <= 100.0):
      errors.append(
          f"[Runtime] Row {i}: 'Duty cycle' not between 0-100%:"
          f" {row_dict['Duty cycle']}"
      )
  if errors:
    raise AirflowFailException(errors)


@task
def validate_tensorcore_table(tpu_info_dict: str):
  """Validates the row count and content for the 'TensorCore Utilization' table."""
  errors = []
  content = tpu_info_dict["tensorcore_utilization"]["body"]

  expected_rows = 4
  if len(content) != expected_rows:
    raise AirflowFailException(
        f"[TensorCore] Row count is incorrect. (Actual: {len(content)},"
        f" Expected: {expected_rows})"
    )

  for i, row_dict in enumerate(content, 1):
    util_match = re.match(r"(\d+\.\d+)%", row_dict["TensorCore Utilization"])
    if not (util_match and 0.0 < float(util_match.group(1)) <= 100.0):
      errors.append(
          f"[TensorCore] Row {i}: 'Utilization' not between 0-100%:"
          f" {row_dict['TensorCore Utilization']}"
      )
  if errors:
    raise AirflowFailException(errors)


@task
def validate_latency_table(tpu_info_dict: str):
  """Validates the row count and content for the TPU Buffer Transfer Latency table."""
  errors = []
  content = tpu_info_dict["tpu_buffer_transfer_latency"]["body"]

  if len(content) == 0:
    raise AirflowFailException(
        "[Latency] Row count is incorrect. At least one row is expected."
    )

  for i, row_dict in enumerate(content, 1):
    for title, val_str in row_dict.items():
      if title == "Buffer Size":
        continue
      if not (
          val_str.endswith(" us") and float(val_str.replace(" us", "")) > 0
      ):
        errors.append(
            f"[Latency] Row {i}, Col {title}: Invalid latency value: {val_str}"
        )
  if errors:
    raise AirflowFailException(errors)


with models.DAG(
    dag_id="tpu_info_format_validation_dag",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=constants.Schedule.WEEKDAY_PDT_6AM_7AM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info"],
    # TODO check the description and after dag finish.
    description=(
        "This DAG verifies the format of the tables in the tpu-info output "
        "using tpu-info CLI tool. It includes 4 tables: TPU Chips, TPU "
        "Runtime Utilization, TensorCore Utilization, and TPU Buffer Transfer "
        "Latency."
    ),
    doc_md="""
      # Format Validation DAG
      # This DAG verifies the format of the tables in the tpu-info output.""",
) as dag:
  cluster_info = node_pool.Info(
      project_id=models.Variable.get(
          "TCU_PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
      ),
      cluster_name=models.Variable.get(
          "TCU_CLUSTER_NAME", default_var="yuna-auto-testing"
      ),
      region=models.Variable.get(
          "TCU_REGION", default_var=Region.US_EAST5.value
      ),
      machine_type=models.Variable.get(
          "TCU_MACHINE_TYPE", default_var=MachineVersion.CT6E_STAND_4T.value
      ),
  )

  kubeconfig_path = "/tmp/kubeconfig"
  generator = JobSet(
      jobset_name="tpu-info-v6e-workload",
      namespace="default",
      max_restarts=5,
      replicated_job_name="tpu-job-slice",
      replicas=2,
      backoff_limit=0,
      completions=4,
      parallelism=4,
      tpu_accelerator_type="tpu-v6e-slice",
      tpu_topology="4x4",
      container_name="jax-tpu-worker",
      image="us-docker.pkg.dev/tpu-prod-env-one-vm/yuna-docker-repo/tpu-info:v0.4.0",
      tpu_cores_per_pod=4,
  )

  workload_script = Workload.JAX_TPU_BENCHMARK

  # Clean up any pre-existing workloads to ensure a clean environment for the
  # test.
  start_cleanup = end_workload.override(
      task_id="start_cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
  )(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=generator,
  )

  apply_time = run_workload(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=generator,
      script=workload_script,
  )

  active_pods = get_active_pods.override(task_id="get_active_pod")(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=generator,
  )

  wait_for_job_start = wait_for_jobset_started.override(
      task_id="wait_for_job_start"
  )(cluster_info, pod_name_list=active_pods, job_apply_time=apply_time)

  tpu_info_outputs = (
      get_tpu_info_from_pod.override(task_id="get_tpu_info")
      .partial(kubeconfig=kubeconfig_path)
      .expand(pod_name=active_pods)
  )

  tpu_info_dict = (
      parse_tpu_info_output.override(task_id="get_each_metric_table")
      .partial()
      .expand(output=tpu_info_outputs)
  )

  with TaskGroup(group_id="verification_group") as verification_group:
    verify_table_amount_task = (
        verify_table_amount.override(task_id="verify_table_amount_task")
        .partial()
        .expand(tpu_info_dict=tpu_info_dict)
    )

    validate_tpu_chips_metric = (
        validate_chips_table.override(task_id="validate_tpu_chips_metric")
        .partial(info=cluster_info)
        .expand(tpu_info_dict=tpu_info_dict)
    )

    validate_runtime_metric = (
        validate_runtime_table.override(task_id="validate_runtime_metric")
        .partial()
        .expand(tpu_info_dict=tpu_info_dict)
    )

    validate_tensorcore_metric = (
        validate_tensorcore_table.override(task_id="validate_tensorcore_metric")
        .partial()
        .expand(tpu_info_dict=tpu_info_dict)
    )

    validate_latency_metric = (
        validate_latency_table.override(task_id="validate_latency_metric")
        .partial()
        .expand(tpu_info_dict=tpu_info_dict)
    )

  clean_up = end_workload.override(
      task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
  )(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=generator,
  ).as_teardown(
      setups=apply_time
  )

  (
      verify_table_amount_task
      >> [
          validate_tpu_chips_metric,
          validate_runtime_metric,
          validate_tensorcore_metric,
          validate_latency_metric,
      ]
  )

  (
      start_cleanup
      >> apply_time
      >> active_pods
      >> wait_for_job_start
      >> tpu_info_outputs
      >> tpu_info_dict
      >> verification_group
      >> clean_up
  )
