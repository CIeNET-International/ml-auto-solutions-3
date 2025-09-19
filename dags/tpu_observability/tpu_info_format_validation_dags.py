"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

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
from airflow.exceptions import AirflowFailException
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils.jobset_generator import JobSet
from dags.tpu_observability.utils.jobset_generator import Workload
from dags.tpu_observability.utils.monitoring import query_time_series
from dags.tpu_observability.utils.node_pool_util import Info


def _get_credentials_command(node_pool: Info):
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

  yaml_content = yaml_config_instance.generate_yaml(workload_script=script)

  result = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_apply_jobset_command(
              kubeconfig, yaml_content, yaml_config.params.get("namespace")
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
              kubeconfig, yaml_content, yaml_config.params.get("namespace")
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

  result = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_delete_jobset_command(kubeconfig, yaml_config.params.get("namespace")),
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
      " && ".join([
          _get_credentials_command(info),
          _k8s_get_pod_command(kubeconfig, yaml_config.params.get("namespace")),
      ]),
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )

  kubectl_cmd = (
      f"kubectl --kubeconfig={kubeconfig} get pods -n"
      f" {yaml_config.params.get('namespace')} -o"
      " jsonpath={.items[*].metadata.name}"
  )
  process = subprocess.run(
      kubectl_cmd, shell=True, check=True, capture_output=True, text=True
  )
  if not process or not process.stdout.strip():
    logging.warning("Received empty pod list from bash task.")
    raise AirflowFailException("Received empty pod list from bash task.")

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
    raise AirflowFailException("pod_name_list is empty, sensor cannot proceed.")

  pod_name = random.choice(pod_name_list)
  metric_name = "kubernetes.io/container/accelerator/tensorcore_utilization"
  filter_string = [
      f'metric.type = "{metric_name}"',
      f'resource.labels.cluster_name = "{info.cluster_name}"',
      f'resource.labels.pod_name = "{pod_name}"'
  ]
  time_series_data = query_time_series(
      project_id=info.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=datetime_job_apply_time,
      end_time=end_time_utc,
      view="FULL",
  )

  # Retrieve the last three records to ensure stable workload startup.
  if not time_series_data or len(time_series_data[0].points) < 3:
    return False
  last_n_data_points = [
      round(point.value.double_value, 2)
      for point in time_series_data[0].points[0:3]
  ]
  # 0 means 0.0% of tensorcore util
  return all(p > 0 for p in last_n_data_points)


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
  title_pattern = re.compile(
      r"(^[^\n].*)\n┏",
      re.MULTILINE
  )
  table_block_pattern = re.compile(
      r"(^┏[\s\S]*?┘)",
      re.MULTILINE
  )

  title = title_pattern.findall(output)
  clean_title = [s.strip() for s in title]
  table_block = table_block_pattern.findall(output)

  tables_dict = dict(zip(clean_title, table_block))

  return tables_dict


@task
def verify_all_table_exist(
    tables_dict: Dict[str, str], expected_count: int = 4
):
  """Receives a dictionary of parsed tables and verifies if the total number.

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
    raise AirflowFailException(
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
    raise AirflowFailException(
        "[TPU Chips] Table structure is incomplete (missing body or footer)."
    )
  tpu_type = yaml_config.params.get("tpu_accelerator_type").split("-")[1]
  rows = match.group(1).strip().split("\n")

  expected_rows = 4
  if len(rows) != expected_rows:
    raise AirflowFailException(
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
    raise AirflowFailException(errors)


@task
def validate_runtime_table(metric_tables_dict: str):
  """Validates the row count and content for the 'TPU Runtime Utilization' table."""
  errors = []
  content = metric_tables_dict["TPU Runtime Utilization"]
  match = re.search(r"┡[━╇]+┩\s*(.*?)\s*└[─┴]+┘", content, re.S)
  if not match:
    raise AirflowFailException(
        "[Runtime] Table structure is incomplete (missing body or footer)."
    )

  rows = match.group(1).strip().split("\n")

  expected_rows = 4
  if len(rows) != expected_rows:
    raise AirflowFailException(
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
    raise AirflowFailException(errors)


@task
def validate_tensorcore_table(metric_tables_dict: str):
  """Validates the row count and content for the 'TensorCore Utilization' table."""
  errors = []
  content = metric_tables_dict["TensorCore Utilization"]
  match = re.search(r"┡[━╇]+┩\s*(.*?)\s*└[─┴]+┘", content, re.S)
  if not match:
    raise AirflowFailException(
        "[TensorCore] Table structure is incomplete (missing body or footer)."
    )

  rows = match.group(1).strip().split("\n")

  expected_rows = 4
  if len(rows) != expected_rows:
    raise AirflowFailException(
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
    raise AirflowFailException(errors)


@task
def validate_latency_table(metric_tables_dict: str):
  """Validates the row count and content for the TPU Buffer Transfer Latency table."""
  errors = []
  content = metric_tables_dict["TPU Buffer Transfer Latency"]
  match = re.search(r"┡[━╇]+┩\s*(.*?)\s*└[─┴]+┘", content, re.S)
  if not match:
    raise AirflowFailException(
        "[Latency] Table structure is incomplete (missing body or footer)."
    )

  rows = match.group(1).strip().split("\n")

  if len(rows) == 0:
    raise AirflowFailException(
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
  cluster_info = Info(
      project_id=models.Variable.get(
          "TCU_PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
      ),
      cluster_name=models.Variable.get(
          "TCU_CLUSTER_NAME", default_var="yuna-auto-testing"
      ),
      region=models.Variable.get(
          "TCU_REGION", default_var=Region.US_EAST5.value
      ),
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
      image="us-docker.pkg.dev/tpu-prod-env-one-vm/yuna-docker-repo/tpu-info:v0.4.0",
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
