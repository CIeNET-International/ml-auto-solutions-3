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
A
"""

import datetime
import os
import subprocess
import tempfile
from typing import List

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


# --- Helper Methods ---


def execute_tpu_info_cli_command(info, pod_name: str, tpu_args: str) -> str:
  """Helper to handle KUBECONFIG and execute kubectl exec."""
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    # Note: jobset.Command.get_credentials_command(info) must be accessible
    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- {tpu_args}",
    ])
    # Returning the stdout of the command execution
    return subprocess.run_exec(cmd, env=env)


def verify_output_contains_patterns(
    output: str, patterns: List[str], context: str
):
  """Verifies that expected strings exist in the output."""
  for pattern in patterns:
    if pattern not in output:
      raise AssertionError(
          f"Validation failed for '{context}': Missing '{pattern}'."
      )


def verify_streaming_frequency(output: str, rate: float, duration: int):
  """Quantitatively verifies if the tool outputs data at the requested rate."""
  # Count occurrences of a specific header unique to each data burst
  sample_count = output.count("TPU Metrics")
  expected_samples = duration / rate

  # 50% tolerance for Kubernetes network latency and tool initialization
  threshold = expected_samples * 0.5

  print(
      f"Rate: {rate}s | Expected: ~{expected_samples} | Actual: {sample_count}"
  )

  if sample_count < threshold:
    raise AssertionError(
        f"Frequency too low. Expected at least {threshold} samples, got {sample_count}."
    )


# --- Task Definitions ---


@task
def validate_streaming_rate(info, pod_name: str, rate: float) -> str:
  """
  Mapped Task: Validates tpu-info streaming performance.
  Each instance tests one pod at one specific rate.
  """
  # Ensure duration is long enough to catch slow rates (e.g., 5s rate needs >10s)
  duration = max(10, int(rate * 3))

  # Use timeout to stop the infinite streaming output
  tpu_args = f"timeout {duration}s tpu-info --streaming --rate {rate}"

  print(f"Validating Pod: {pod_name} at Rate: {rate}s for {duration}s")
  output = execute_tpu_info_cli_command(info, pod_name, tpu_args)

  # 1. Check for required UI elements/metrics
  verify_output_contains_patterns(
      output,
      ["TPU Metrics", "Duty Cycle", "Memory Usage"],
      f"Streaming check on {pod_name}",
  )

  # 2. Check for timing accuracy
  verify_streaming_frequency(output, rate, duration)

  return f"Completed: {pod_name} at {rate}s"


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="tpu_info_verify_streaming_rate_dags",
    start_date=datetime.datetime(2025, 8, 10),
    schedule="0 19 * * *" if composer_env.is_prod_env() else None,
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
        jobset_name="tpu-info-verify-streaming-rate-jobset",
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
          dag_name="tpu_info_verify_streaming_rate_dags",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
      )

      apply_time = jobset.run_workload.override(task_id="run_workload")(
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
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=pod_names, job_apply_time=apply_time)

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="verification_group"
      ) as verification_group:
        # 1. Define the static rates you want to test
        test_rates = [0.1, 0.5, 1.0, 5.0]

        # 2. Use Dynamic Task Mapping (.expand)
        # This creates a Cross-Product: (number of pods) x (number of rates)
        # If pod_names has 8 pods, this will trigger 32 task instances.
        streaming_validation_results = (
            validate_streaming_rate.override(task_id="streaming_rate_test")
            .partial(info=cluster_info)
            .expand(
                pod_name=pod_names,  # XComArg from a previous task
                rate=test_rates,  # Standard Python list
            )
        )

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
