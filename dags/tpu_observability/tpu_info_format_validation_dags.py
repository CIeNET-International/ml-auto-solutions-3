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
from typing import List, Tuple

from airflow import models
from airflow.decorators import task
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import Project
from dags.common.vm_resource import Region
from dags.common.vm_resource import Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils.jobset_yaml_generator import create_jobset_yaml
from dags.tpu_observability.utils.jobset_yaml_generator import YamlConfig
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
      f"-n {namespace}"
  ])


def _k8s_delete_command(kubeconfig: str, namespace: str):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} delete jobsets --all",
      f"-n {namespace} --timeout=60s --ignore-not-found=true"
  ])



@task
def run_workload(info: Info, kubeconfig: str, yaml_config: YamlConfig):
  """
  Applies the specified YAML file to the GKE cluster.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  # Get GKE cluster credentials
  yaml_path = create_jobset_yaml(yaml_config)
  result = subprocess.run(
        " && ".join([
          _get_credentials_command(info.cluster_name, info.region, info.project_id),
          _k8s_apply_command(kubeconfig, yaml_path, yaml_config.namespace),
      ]),
      shell=True, check=True, env=env, capture_output=True, text=True
  )

  print("STDOUT:", result.stdout)

  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  return current_time_utc.isoformat(timespec="milliseconds")


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

  subprocess.run(
      " && ".join([
          _get_credentials_command(info.cluster_name, info.region, info.project_id),
          _k8s_delete_command(kubeconfig, yaml_config.namespace),
      ]),
      shell=True, check=True, env=env, capture_output=True, text=True
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


@task
def validate_table_structure(tpu_info_output: str):
  """Validates the structural integrity of tables within the tpu-info output.

  This task checks if expected tables ("TPU Chips", "TPU Runtime Utilization",
  "TensorCore Utilization", and "TPU Buffer Transfer Latency") are present
  and have a valid structure (header, body, and footer) in the provided
  `tpu_info_output`.

  Args:
    tpu_info_output: The string output from the 'tpu-info' command.

  Raises:
    AirflowException: If any of the expected tables are not found or have
      an incomplete structure.
  """
  table_title = [
      "TPU Chips",
      "TPU Runtime Utilization",
      "TensorCore Utilization",
      "TPU Buffer Transfer Latency",
  ]
  flag = True
  invaild_metric = ""
  for title in table_title:
    escaped_title = re.escape(title)
    pattern = re.compile(
        rf"{escaped_title}\s*"
        rf"┏[━┳]+┓\s*"
        rf"┃.*?┃\s*"
        rf"┡[━╇]+┩"
        rf"[\s\S]*?"
        rf"└[─┴]+┘",
        re.S,
    )
    output_validated = pattern.search(tpu_info_output)
    if not output_validated:
      logging.error("Table %s is not exist", title)
      invaild_metric += title + " "
      flag = False
      continue
    output_ = pattern.search(tpu_info_output).group(0)
    print(output_)

    if flag is False:
      raise AirflowException(
          f"Structure Error: Table '{invaild_metric}' not found or its"
          " structure is incomplete."
      )


@task
def validate_row_counts(tpu_info_output: str):
  """Validates the number of rows in specific tables within the tpu-info output.

  This task checks if the "TPU Chips", "TPU Runtime Utilization", and
  "TensorCore Utilization" tables in the provided `tpu_info_output` contain
  the expected number of data rows (excluding headers and footers).

  Args:
    tpu_info_output: The string output from the 'tpu-info' command.

  Returns:
    A tuple containing:
      - bool: True if all checked tables have the expected row counts, False
      otherwise.
      - list[str]: A list of error messages for tables with incorrect row
      counts.
  """
  errors = []
  tables_to_check = {
      "TPU Chips": 4,
      "TPU Runtime Utilization": 4,
      "TensorCore Utilization": 4,
  }

  for title, expected_rows in tables_to_check.items():
    escaped_title = re.escape(title)
    pattern = re.compile(
        rf"{escaped_title}\s*"
        rf"┏[━┳]+┓\s*"
        rf"┃.*?┃\s*"
        rf"┡[━╇]+┩"
        rf"([\s\S]*?)"
        rf"└[─┴]+┘",
        re.S,
    )
    match = pattern.search(tpu_info_output)
    if not match:
      errors.append(
          f"Row count check error: Could not find table '{title}' to count its"
          " rows."
      )
      continue

    body_content = match.group(1).strip()
    actual_rows = len(body_content.split("\n")) if body_content else 0
    if actual_rows != expected_rows:
      error_msg = (
          f"Row count for '{title}' is incorrect"
          f"(Actual: {actual_rows}, Expected: {expected_rows})."
      )
      print(error_msg)
      errors.append(error_msg)
  if errors:
    raise AirflowException(f"Row count check error: {errors}")


@task
def validate_table_contents(tpu_info_output: str):
  """Validates the content format of tables within the tpu-info output.

  This task parses the output of the 'tpu-info' command and checks if the
  data within each expected table ("TPU Chips", "TPU Runtime Utilization",
  "TensorCore Utilization", and "TPU Buffer Transfer Latency") conforms to
  the expected format and constraints (e.g., numeric ranges, specific units).

  Args:
    tpu_info_output: The string output from the 'tpu-info' command.

  Returns:
    A boolean indicating whether all table contents are valid.
    (Note: Currently, this function only populates an `errors` list but doesn't
    explicitly return it or raise an exception based on it).
  """
  table_title = [
      "TPU Chips",
      "TPU Runtime Utilization",
      "TensorCore Utilization",
      "TPU Buffer Transfer Latency",
  ]
  errors = []
  for title in table_title:
    escaped_title = re.escape(title)
    pattern = re.compile(
        rf"{escaped_title}\s*"
        rf"┏[━┳]+┓\s*"
        rf"┃.*?┃\s*"
        rf"┡[━╇]+┩"
        rf"([\s\S]*?)"
        rf"└[─┴]+┘",
        re.S,
    )
    output_validated = pattern.search(tpu_info_output)
    rows = output_validated.group(1).strip().split("\n")

    match title:
      case "TPU Chips":
        for i, row_str in enumerate(rows, 1):
          cols = [col.strip() for col in row_str.split("│") if col.strip()]

          chip, type_str, devices_str, pid_str = cols
          if not re.match(r"/dev/vfio/\d+", chip):
            errors.append(f"TPU Chips, Row {i}: Invalid 'Chip' format: {chip}")
          if "TPU" not in type_str:
            errors.append(
                f"TPU Chips, Row {i}: 'Type' does not seem valid: {type_str}"
            )
          if not (devices_str.isdigit() and int(devices_str) > 0):
            errors.append(
                f"TPU Chips, Row {i}: 'Devices' must be > 0, but got:"
                f" {devices_str}"
            )

          if not (pid_str.isdigit() and int(pid_str) > 0):
            errors.append(
                f"TPU Chips, Row {i}: 'PID' must be > 0, but got: {pid_str}"
            )

      case "TPU Runtime Utilization":
        for i, row_str in enumerate(rows, 1):
          cols = [col.strip() for col in row_str.split("│") if col.strip()]

          _, hbm_usage, duty_cycle = cols
          hbm_match = re.match(
              r"(\d+\.\d+)\s*GiB\s*/\s*(\d+\.\d+)\s*GiB", hbm_usage
          )
          if hbm_match:
            used, total = float(hbm_match.group(1)), float(hbm_match.group(2))
            if total == 0:
              errors.append(f"Runtime Util, Row {i}: Total HBM cannot be 0.")
            if used > total:
              errors.append(
                  f"Runtime Util, Row {i}: Used HBM ({used}) cannot be"
                  f" greater than Total HBM ({total})."
              )
          else:
            errors.append(
                f"Runtime Util, Row {i}: Invalid 'HBM Usage' format:"
                f" {hbm_usage}"
            )

          duty_match = re.match(r"(\d+\.\d+)%", duty_cycle)
          if duty_match:
            percent = float(duty_match.group(1))
            if not (0.0 <= percent <= 100.0):
              errors.append(
                  f"Runtime Util, Row {i}: 'Duty cycle' is not between"
                  f" 0-100%: {duty_cycle}"
              )
          else:
            errors.append(
                f"Runtime Util, Row {i}: Invalid 'Duty cycle' format:"
                f" {duty_cycle}"
            )

      case "TensorCore Utilization":
        for i, row_str in enumerate(rows, 1):
          cols = [col.strip() for col in row_str.split("│") if col.strip()]

          _, util_str = cols
          util_match = re.match(r"(\d+\.\d+)%", util_str)
          if util_match:
            percent = float(util_match.group(1))
            if not (0.0 <= percent <= 100.0):
              errors.append(
                  f"TensorCore Util, Row {i}: 'Utilization' is not between"
                  f" 0-100%: {util_str}"
              )
          else:
            errors.append(
                f"TensorCore Util, Row {i}: Invalid 'Utilization' format:"
                f" {util_str}"
            )

      case "TPU Buffer Transfer Latency":
        for i, row_str in enumerate(rows, 1):
          cols = [col.strip() for col in row_str.split("│") if col.strip()]

          if not cols[0]:
            errors.append(f"Latency, Row {i}: 'Buffer Size' cannot be empty.")
          for j, val_str in enumerate(cols[1:], 1):  # Check P50, P90, etc.
            if val_str.endswith(" us"):
              try:
                if float(val_str.replace(" us", "")) <= 0:
                  errors.append(
                      f"Latency, Row {i}, Col {j+1}: Value must be > 0,"
                      f" but got {val_str}"
                  )
              except ValueError:
                errors.append(
                    f"Latency, Row {i}, Col {j+1}: Could not parse numeric"
                    f" part of {val_str}"
                )
            else:
              errors.append(
                  f"Latency, Row {i}, Col {j+1}: Value must end with ' us',"
                  f" but got {val_str}"
              )

  if errors:
    raise AirflowException(f"Table content check error: {errors}")


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
          "TCU_CLUSTER_NAME", default_var="qmcgarry-auto"
      ),
      region=models.Variable.get("TCU_REGION", default_var="europe-west4"),
      zone=models.Variable.get("TCU_ZONE", default_var="europe-west4-a"),
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
      tpu_accelerator_type="tpu-v6e-slice",
      tpu_topology="4x4",
      container_name="jax-tpu-worker",
      image="asia-northeast1-docker.pkg.dev/cienet-cmcs/yuna-docker/tpu-info:v0.4.0",
      command=["bash", "-c"],
      command_args=[
          """
          python -c 'import jax; print("TPU cores:", jax.device_count())'
          python -c '
          import jax
          import jax.numpy as jnp
          import time
          import os
          from jax.sharding import Mesh, NamedSharding
          from jax.experimental.pjit import pjit

          os.environ.setdefault("JAX_USE_PJIT", "true")
          jax.distributed.initialize()

          global_devices = jax.devices()
          print(f"[Host {jax.process_index()}] Got {len(global_devices)} global devices")
          mesh = Mesh(global_devices, ("x",))

          print(f"[Host {jax.process_index()}] Allocating data...")
          size = 32768
          x_global = jnp.ones((size, size), dtype=jnp.float32)
          y_global = jnp.ones((size, size), dtype=jnp.float32)

          print(f"[Host {jax.process_index()}] Sharding data...")
          sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("x", None))
          x = jax.device_put(x_global, sharding)
          y = jax.device_put(y_global, sharding)
          print(f"[Host {jax.process_index()}] Data on device")

          # ========= Define heavy workload =========
          @pjit
          def matmul_ultra_heavy(x, y):
              tmp1 = jnp.dot(x, y)
              tmp2 = jnp.dot(tmp1, y.T)
              tmp3 = jnp.dot(tmp2, x.T)
              tmp4 = jnp.dot(tmp3, x)
              tmp5 = jnp.dot(tmp4, y)
              return tmp5

          print(f"[Host {jax.process_index()}] Warming up...")
          matmul_ultra_heavy(x, y).block_until_ready()

          # ========= Benchmark =========
          print(f"[Host {jax.process_index()}] Starting benchmark...")

          start = time.time()
          for i in range(1_000_000): # Remember to control loop time to control experiment time
              result = matmul_ultra_heavy(x, y)
          result.block_until_ready()
          end = time.time()

          if jax.process_index() == 0:
              print(f"Total time: {end - start:.2f} seconds (on full v6e-16)")
          '
          echo "sleep..."
          sleep 10000
          """
      ],
      tpu_cores_per_pod=4,
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

  apply_time = run_workload(
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

  tpu_info_outputs = (
      get_tpu_info_from_pod.override(task_id="get_tpu_info")
      .partial(kubeconfig=kubeconfig_path)
      .expand(pod_name=active_pods)
  )

  with TaskGroup(group_id="verification_group") as verification_group:
    validate_table_structure_task = (
        validate_table_structure.override(
            task_id="validate_table_structure_task"
        )
        .partial()
        .expand(tpu_info_output=tpu_info_outputs)
    )

    validate_row_counts_task = (
        validate_row_counts.override(task_id="validate_row_counts_task")
        .partial()
        .expand(tpu_info_output=tpu_info_outputs)
    )

    validate_table_contents_task = (
        validate_table_contents.override(task_id="validate_table_contents_task")
        .partial()
        .expand(tpu_info_output=tpu_info_outputs)
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
      validate_table_structure_task
      >> validate_row_counts_task
      >> validate_table_contents_task
  )

  (
      start_cleanup
      >> apply_time
      >> active_pods
      >> wait_for_job_start
      >> tpu_info_outputs
      >> verification_group
      >> clean_up
  )
