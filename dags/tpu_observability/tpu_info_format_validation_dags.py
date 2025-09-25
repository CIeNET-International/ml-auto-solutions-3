"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

import datetime
import logging
import os
import re
import subprocess
from typing import Dict

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from dags.common.vm_resource import Project, Region, MachineVersion
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils.jobset_generator import JobSet
from dags.tpu_observability.utils.jobset_generator import Workload
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.tpu_info_util import parse_tpu_info_output
from dags.tpu_observability.utils import jobset_util as jobset_util



MACHINE_TYPE_TO_TPU_VERSION = {
    "ct6e-standard-4t": "v6e",
    "ct5p-hightpu-4t": "v5p",
}


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
  jobset = JobSet(
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
      image="us-docker.pkg.dev/tpu-prod-env-one-vm/yuna-docker-repo/tpu-info:v0.5.1",
      tpu_cores_per_pod=4,
  )

  workload_script = Workload.JAX_TPU_BENCHMARK

  apply_time = jobset_util.run_workload(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=jobset.generate_yaml(workload_script=workload_script),
      namespace=jobset.namespace,
  )

  active_pods = jobset_util.get_active_pods.override(task_id="get_active_pod")(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      namespace=jobset.namespace,
  )

  wait_for_job_start = jobset_util.wait_for_jobset_started.override(
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

  clean_up_workload = jobset_util.end_workload.override(
      task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
  )(
      info=cluster_info,
      kubeconfig=kubeconfig_path,
      namespace=jobset.namespace,
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
      apply_time
      >> active_pods
      >> wait_for_job_start
      >> tpu_info_outputs
      >> tpu_info_dict
      >> verification_group
      >> clean_up_workload
  )
