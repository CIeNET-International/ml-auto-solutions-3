# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A DAG to validate the `tpu-info` CLI tool, ensuring help documentation,
version metadata, and process monitoring are functional inside TPU worker pods.
"""

import datetime
import tempfile
import os

from airflow import models
from airflow.decorators import task
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH


def _get_tpu_info_help_output(info: node_pool.Info, pod_name: str) -> str:
  """
  Retrieves the raw output of `tpu-info -help` from a specific pod.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- tpu-info -help",
    ])

    return subprocess.run_exec(cmd, env=env)


@task
def validate_tpu_info_help(info: node_pool.Info, pod_name: str) -> str:
  """
  Validates that the `tpu-info -help` output contains all required fields.
  """
  output = _get_tpu_info_help_output(info, pod_name)
  required_patterns = [
      "Display TPU info and metrics.",
      "options:",
      "-h, --help",
      "-v, --version",
      "-p, --process",
      "--streaming",
      "--rate RATE",
      "--list_metrics",
  ]

  for pattern in required_patterns:
    if pattern not in output:
      raise AssertionError(
          "Validation failed: Missing expected string "
          f"'{pattern}' in tpu-info help.\n"
          f"Output received:\n{output}"
      )

  return output


def _get_tpu_version_output(info: node_pool.Info, pod_name: str) -> str:
  """
  Executes the version command in the pod to get libtpu version and accelerator type.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- tpu-info --version",
    ])
    return subprocess.run_exec(cmd, env=env)


@task
def validate_tpu_info_version(info: node_pool.Info, pod_name: str) -> str:
  """
  Validates that `tpu-info --version` returns version and accelerator info.
  """
  output = _get_tpu_version_output(info, pod_name)
  required_patterns = [
      "tpu-info version:",
      "libtpu version:",
      "accelerator type:",
  ]

  for pattern in required_patterns:
    if pattern not in output:
      raise AssertionError(
          f"Validation failed: Missing expected string '{pattern}' "
          f"in tpu-info --version output.\nOutput:\n{output}"
      )

  return output


def _get_tpu_process_output(info: node_pool.Info, pod_name: str) -> str:
  """
  Executes the process command in the pod to get the TPU process table.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- tpu-info --process",
    ])
    return subprocess.run_exec(cmd, env=env)


@task
def validate_tpu_info_process(info: node_pool.Info, pod_name: str) -> str:
  """
  Validates that `tpu-info --process` displays the TPU process table.
  """
  output = _get_tpu_process_output(info, pod_name)
  required_patterns = [
      "TPU Process Info",
      "Chip",
      "PID",
      "Process Name",
      "/dev/vfio/",
      "python",
  ]

  for pattern in required_patterns:
    if pattern not in output:
      raise AssertionError(
          f"Validation failed: Missing expected string '{pattern}' "
          f"in tpu-info --process output.\nOutput:\n{output}"
      )

  return output


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="tpu_info_help_validation_dags",
    start_date=datetime.datetime(2025, 8, 10),
    schedule="0 18 * * *" if composer_env.is_prod_env() else None,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu-observability",
        "TPU",
        "v6e-16",
    ],
    description=(
        "Validates tpu-info CLI tool: help documentation, version metadata, "
        "and process monitoring capabilities inside TPU worker pods."
    ),
    doc_md="""
        ### Description
        This DAG performs an end-to-end validation of the `tpu-info` observability tool
        within TPU worker pods. It ensures the CLI tool is correctly installed and
        functional across different TPU configurations.

        ### Validation Steps:
        1. **Help Menu Validation**: Verifies `tpu-info -help` displays all required
           options (streaming, rate, etc.) and specific usage instructions.
        2. **Process Table Validation**: Confirms `tpu-info --process` can successfully
           map PIDs to TPU chips.
        3. **Version Validation**: Ensures `tpu-info --version` correctly reports
           the tool version, libtpu version, and accelerator type.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    jobset_config = JobSet(
        jobset_name="tpu-info-help-validation-jobset",
        namespace="default",
        max_restarts=5,
        replicated_job_name="tpu-job-slice",
        replicas=1,
        backoff_limit=0,
        completions=4,
        parallelism=4,
        tpu_accelerator_type="tpu-v6e-slice",
        tpu_topology="4x4",
        container_name="jax-tpu-worker",
        image="asia-northeast1-docker.pkg.dev/cienet-cmcs/"
        "yuna-docker/tpu-info:v0.5.1",
        tpu_cores_per_pod=4,
    )

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="tpu_info_help_validation_dags",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      create_node_pool = node_pool.create(
          node_pool=cluster_info,
      )

      apply_time = jobset.run_workload(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=Workload.JAX_TPU_BENCHMARK
          ),
          namespace=jobset_config.namespace,
      )

      pod_names = jobset.list_pod_names.override(task_id="list_pod_names")(
          node_pool=cluster_info,
          namespace=jobset_config.namespace,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start", timeout=1200
      )(cluster_info, pod_name_list=pod_names, job_apply_time=apply_time)

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="verification_group"
      ) as verification_group:
        validate_help = validate_tpu_info_help.partial(
            info=cluster_info
        ).expand(pod_name=pod_names)

        validate_version = validate_tpu_info_version.partial(
            info=cluster_info
        ).expand(pod_name=pod_names)

        validate_process = validate_tpu_info_process.partial(
            info=cluster_info
        ).expand(pod_name=pod_names)

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          namespace=jobset_config.namespace,
      ).as_teardown(
          setups=apply_time
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          cluster_info
          >> create_node_pool
          >> apply_time
          >> pod_names
          >> wait_for_job_start
          >> verification_group
          >> cleanup_workload
          >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
