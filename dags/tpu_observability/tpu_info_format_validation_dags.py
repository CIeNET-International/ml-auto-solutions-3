"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

import dataclasses
import datetime
import logging
import os
import random
import re
import subprocess
from typing import List, Tuple, Dict

from airflow import models
from airflow.decorators import task
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import Project
from dags.common.vm_resource import Region
from dags.common.vm_resource import Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils.jobset_yaml_generator import JobSet
from dags.tpu_observability.utils.jobset_yaml_generator import Workload
from dags.tpu_observability.utils.monitoring import query_time_series


@dataclasses.dataclass
class Info:
  """Configuration for the GKE Node Pool and Monitoring.

  Attributes:
    project_id: The Google Cloud project ID.
    region: The region of the GKE cluster.
    zone: The zone of the GKE cluster.
    cluster_name: The name of the GKE cluster.
  """

  project_id: str
  region: str
  zone: str
  cluster_name: str


def _get_credentials_command(cluster_name: str, region: str, project_id: str):
  return " ".join([
      "gcloud container clusters",
      f"get-credentials {cluster_name}",
      f"--region={region}",
      f"--project={project_id}",
  ])


def _k8s_apply_command(kubeconfig: str, yaml_path: str, namespace: str):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} apply",
      f"-f {yaml_path}",
      f"-n {namespace}",
  ])


def _k8s_delete_command(kubeconfig: str, namespace: str):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} delete jobsets --all",
      f"-n {namespace} --timeout=60s --ignore-not-found=true",
  ])


@task
def run_workload(info: Info, kubeconfig: str, yaml_config: JobSet, script: str):
  """Applies the specified YAML file to the GKE cluster.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    yaml_config: The JobSet object containing YAML configuration.
    script: The workload script to be executed.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  # Get GKE cluster credentials
  yaml_path = yaml_config_instance.generate_yaml(workload_script=script)

  result = subprocess.run(
      " && ".join(
          [
              _get_credentials_command(
                  info.cluster_name, info.region, info.project_id
              ),
              _k8s_apply_command(kubeconfig, yaml_path, yaml_config.namespace),
          ]
      ),
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )

  print("STDOUT:", result.stdout)

  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  return current_time_utc.isoformat(timespec="milliseconds")


@task
def end_workload(info: Info, kubeconfig: str, yaml_config: JobSet):
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

  subprocess.run(
      " && ".join([
          _get_credentials_command(
              info.cluster_name, info.region, info.project_id
          ),
          _k8s_delete_command(kubeconfig, yaml_config.namespace),
      ]),
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )


@task
def get_active_pods(info: Info, kubeconfig: str, yaml_config: JobSet):
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

  subprocess.run(
      _get_credentials_command(info.cluster_name, info.region, info.project_id),
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


# TODO move to an tpu-info util file after validated this function work.
@task
def split_tables_by_structure(output: str) -> Dict[str, str]:
  """Splits a multi-table string output into a dictionary of tables.

  Each table is identified by a title line followed by a block enclosed
  in box-drawing characters.

  Args:
    output: The string containing one or more tables.

  Returns:
    A dictionary where keys are the table titles and values are the full
    text content of each table, including the box-drawing characters.
  """
  pattern = re.compile(
      # Capture Group 1: The Title Line
      # ^ - matches the beginning of a line (because of the re.M flag)
      # [^\n] - matches any character that is not a '\n'
      # .*? - non-greedily matches any character until the end of the line
      r"(^[^\n].*?)\n\s*"
      # Capture Group 2: The Full Table Block
      r"(┏[━┳]+┓[\s\S]*?└[─┴]+┘)",
      re.MULTILINE,
  )

  matches = pattern.findall(output)

  tables_dict = {}
  for title, table_block in matches:
    tables_dict[title.strip()] = table_block.strip()

  return tables_dict


@task
def verify_all_table_exist(
    tables_dict: Dict[str, str], expected_count: int = 4
):
  """Receives a dictionary of parsed tables and verifies if the total number

  of tables matches the expected count.

  Args:
      tables_dict: A dictionary where keys are table titles and values are the
        full text content for each table.
      expected_count: The expected number of tables, defaults to 4.

  Returns:
      A tuple containing:
      - bool: True if the table count is correct, otherwise False.
      - str: A message describing the verification result.
  """
  actual_count = len(tables_dict)

  if actual_count != expected_count:
    raise AirflowException(
        f"Verification FAILED: Found {actual_count} tables, but expected"
        f" {expected_count}.\nFound table titles: {list(tables_dict.keys())}"
    )


@task
def validate_chips_table(metric_tables_dict: str, yaml_config: str):
  """Validates the row count and content for the 'TPU Chips' table."""
  errors = []
  content = metric_tables_dict["TPU Chips"]
  match = re.search(r"┡[━╇]+┩\s*(.*?)\s*└[─┴]+┘", content, re.S)
  if not match:
    raise AirflowException(
        "[TPU Chips] Table structure is incomplete (missing body or footer)."
    )
  tpu_type = yaml_config.tpu_accelerator_type.split("-")[1]
  rows = match.group(1).strip().split("\n")

  expected_rows = 4
  if len(rows) != expected_rows:
    raise AirflowException(
        f"[TPU Chips] Row count is incorrect. (Actual: {len(rows)}, Expected:"
        f" {expected_rows})"
    )

  for i, row_str in enumerate(rows, 1):
    cols = [col.strip() for col in row_str.split("│") if col.strip()]

    chip, type_str, _, pid_str = cols
    if not re.match(r"/dev/vfio/\d+", chip):
      errors.append(f"[TPU Chips] Row {i}: Invalid 'Chip' format: {chip}")
    if tpu_type not in type_str:
      errors.append(
          f"[TPU Chips] Row {i}: 'Type' column value '{type_str}' does not"
          f" contain the expected version '{tpu_type}'."
      )
    if not (pid_str.isdigit() and int(pid_str) > 0):
      errors.append(
          f"[TPU Chips] Row {i}: 'PID' must be a number greater than 0, got:"
          f" {pid_str}"
      )
  if errors:
    raise AirflowException(errors)


@task
def validate_runtime_table(metric_tables_dict: str) -> List[str]:
  """Validates the row count and content for the 'TPU Runtime Utilization' table."""
  errors = []
  content = metric_tables_dict["TPU Runtime Utilization"]
  match = re.search(r"┡[━╇]+┩\s*(.*?)\s*└[─┴]+┘", content, re.S)
  if not match:
    raise AirflowException(
        "[Runtime] Table structure is incomplete (missing body or footer)."
    )

  rows = match.group(1).strip().split("\n")

  expected_rows = 4
  if len(rows) != expected_rows:
    raise AirflowException(
        f"[Runtime] Row count is incorrect. (Actual: {len(rows)}, Expected:"
        f" {expected_rows})"
    )

  for i, row_str in enumerate(rows, 1):
    cols = [col.strip() for col in row_str.split("│") if col.strip()]

    _, hbm_usage, duty_cycle = cols
    hbm_match = re.match(r"(\d+\.\d+)\s*GiB\s*/\s*(\d+\.\d+)\s*GiB", hbm_usage)
    if hbm_match:
      used, total = float(hbm_match.group(1)), float(hbm_match.group(2))
      if used > total:
        errors.append(
            f"[Runtime] Row {i}: Used HBM ({used}) cannot be greater than Total"
            f" HBM ({total})."
        )
    else:
      errors.append(
          f"[Runtime] Row {i}: Invalid 'HBM Usage' format: {hbm_usage}"
      )

    duty_match = re.match(r"(\d+\.\d+)%", duty_cycle)
    if not (duty_match and 0.0 <= float(duty_match.group(1)) <= 100.0):
      errors.append(
          f"[Runtime] Row {i}: 'Duty cycle' not between 0-100%: {duty_cycle}"
      )
  if errors:
    raise AirflowException(errors)


@task
def validate_tensorcore_table(metric_tables_dict: str):
  """Validates the row count and content for the 'TensorCore Utilization' table."""
  errors = []
  content = metric_tables_dict["TensorCore Utilization"]
  match = re.search(r"┡[━╇]+┩\s*(.*?)\s*└[─┴]+┘", content, re.S)
  if not match:
    raise AirflowException(
        "[TensorCore] Table structure is incomplete (missing body or footer)."
    )

  rows = match.group(1).strip().split("\n")

  expected_rows = 4
  if len(rows) != expected_rows:
    raise AirflowException(
        f"[TensorCore] Row count is incorrect. (Actual: {len(rows)}, Expected:"
        f" {expected_rows})"
    )

  for i, row_str in enumerate(rows, 1):
    cols = [col.strip() for col in row_str.split("│") if col.strip()]

    _, util_str = cols
    util_match = re.match(r"(\d+\.\d+)%", util_str)
    if not (util_match and 0.0 < float(util_match.group(1)) <= 100.0):
      errors.append(
          f"[TensorCore] Row {i}: 'Utilization' not between 0-100%: {util_str}"
      )
  if errors:
    raise AirflowException(errors)


@task
def validate_latency_table(metric_tables_dict: str) -> List[str]:
  """Validates the row count and content for the TPU Buffer Transfer Latency table."""
  errors = []
  content = metric_tables_dict["TPU Buffer Transfer Latency"]
  match = re.search(r"┡[━╇]+┩\s*(.*?)\s*└[─┴]+┘", content, re.S)
  if not match:
    raise AirflowException(
        "[Latency] Table structure is incomplete (missing body or footer)."
    )

  rows = match.group(1).strip().split("\n")

  if len(rows) == 0:
    raise AirflowException(
        "[Latency] Row count is incorrect. At least one row is expected."
    )

  for i, row_str in enumerate(rows, 1):
    cols = [col.strip() for col in row_str.split("│") if col.strip()]

    for j, val_str in enumerate(cols[1:]):
      if not (
          val_str.endswith(" us") and float(val_str.replace(" us", "")) > 0
      ):
        errors.append(
            f"[Latency] Row {i}, Col {j+2}: Invalid latency value: {val_str}"
        )
  if errors:
    raise AirflowException(errors)


with models.DAG(
    dag_id="tpu_info_format_validation_dag",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=constants.Schedule.WEEKDAY_PDT_6AM_7AM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info"],
    # TODO modify the description and after dag finish.
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
          "TCU_PROJECT_ID", default_var="cienet-cmcs"
      ),
      cluster_name=models.Variable.get(
          "TCU_CLUSTER_NAME", default_var="yuna-xpk-v6e-ew4"
      ),
      region=models.Variable.get("TCU_REGION", default_var="europe-west4"),
      zone=models.Variable.get("TCU_ZONE", default_var=Zone.EUROPE_WEST4_A),
  )

  kubeconfig_path = "/tmp/kubeconfig"
  yaml_config_instance = JobSet(
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
      image="asia-northeast1-docker.pkg.dev/cienet-cmcs/yuna-docker/tpu-info:v0.4.0",
      command=["bash", "-c"],
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
      yaml_config=yaml_config_instance,
  )

  apply_time = run_workload(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=yaml_config_instance,
      script=workload_script,
  )

  active_pods = get_active_pods.override(task_id="get_active_pod")(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=yaml_config_instance,
  )

  wait_for_job_start = query_to_wait_for_jobset_start.override(
      task_id="wait_for_job_start"
  )(cluster_info, pod_name_list=active_pods, job_apply_time=apply_time)

  tpu_info_outputs = (
      get_tpu_info_from_pod.override(task_id="get_tpu_info")
      .partial(kubeconfig=kubeconfig_path)
      .expand(pod_name=active_pods)
  )

  metric_tables_dict = (
      split_tables_by_structure.override(task_id="get_each_metric_table")
      .partial()
      .expand(output=tpu_info_outputs)
  )

  with TaskGroup(group_id="verification_group") as verification_group:
    verify_all_table_exist_task = (
        verify_all_table_exist.override(task_id="verify_all_table_exist_task")
        .partial(expected_count=4)
        .expand(tables_dict=metric_tables_dict)
    )

    validate_tpu_chips_metric = (
        validate_chips_table.override(task_id="validate_tpu_chips_metric")
        .partial(yaml_config=yaml_config_instance)
        .expand(metric_tables_dict=metric_tables_dict)
    )

    validate_runtime_metric = (
        validate_runtime_table.override(task_id="validate_runtime_metric")
        .partial()
        .expand(metric_tables_dict=metric_tables_dict)
    )

    validate_tensorcore_metric = (
        validate_tensorcore_table.override(task_id="validate_tensorcore_metric")
        .partial()
        .expand(metric_tables_dict=metric_tables_dict)
    )

    validate_latency_metric = (
        validate_latency_table.override(task_id="validate_latency_metric")
        .partial()
        .expand(metric_tables_dict=metric_tables_dict)
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
      verify_all_table_exist_task
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
      >> metric_tables_dict
      >> verification_group
      >> clean_up
  )
