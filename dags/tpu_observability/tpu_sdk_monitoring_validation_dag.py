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

"""A DAG to validate the `tpumonitoring` SDK, ensuring help() and
list_supported_metrics() are functional inside TPU worker pods."""

import datetime
import tempfile
import subprocess
import os
from typing import List

from airflow import models
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH

def execute_python_monitoring_command(info, pod_name: str, python_code: str) -> str:
    with tempfile.NamedTemporaryFile() as temp_config_file:
        env = os.environ.copy()
        env["KUBECONFIG"] = temp_config_file.name
        full_python_cmd = f'python3 -c "{python_code}"'

        cmd = " && ".join([
            jobset.Command.get_credentials_command(info),
            f"kubectl exec {pod_name} -n default -- {full_python_cmd}",
        ])
        return subprocess.run_exec(cmd, env=env)

def verify_output_contains_patterns(output: str, patterns: List[str], context: str):

    for pattern in patterns:
        if pattern not in output:
            raise AssertionError(
                f"Validation failed for '{context}': Missing '{pattern}'."
            )

@task
def validate_monitoring_help(info, pod_name: str) -> str:

    code = "from libtpu.sdk import tpumonitoring; tpumonitoring.help()"
    output = execute_python_monitoring_command(info, pod_name, code)


    patterns = [
        "List all supported functionality",
        "list_supported_metrics()",
        "get_metric(metric_name:str)",
        "snapshot mode"
    ]
    verify_output_contains_patterns(output, patterns, "tpumonitoring.help()")
    return output

@task
def validate_metrics_list(info, pod_name: str) -> str:

    code = "from libtpu.sdk import tpumonitoring; print(tpumonitoring.list_supported_metrics())"
    output = execute_python_monitoring_command(info, pod_name, code)


    patterns = [
        "tensorcore_util",
        "ici_link_health",
        "hbm_capacity_usage",
        "duty_cycle_pct",
        "host_to_device_transfer_latency"
    ]
    verify_output_contains_patterns(output, patterns, "tpumonitoring.list_supported_metrics()")
    return output

with models.DAG(
    dag_id="tpu_sdk_monitoring_validation",
    start_date=datetime.datetime(2026, 1, 13),
    schedule=None,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "tpu-observability",
        "delete_pod",
        "TPU",
        "v6e-16",
        "SDK",
        "Validation"
    ],
    description=("Validates tpumonitoring SDK: help() and list_supported_metrics() inside TPU worker pods.")
    ,
    doc_md="""
        ### Description
        This DAG performs an end-to-end validation of the `tpumonitoring` Python SDK
        within TPU worker pods. It ensures the SDK is correctly installed and its
        monitoring functions are accessible via `libtpu.sdk`.

        ### Validation Steps:
        1. **SDK Help Documentation Validation**:
           Executes `tpumonitoring.help()` to verify that the API documentation is
           correctly rendered and includes essential methods like `list_supported_metrics`.

        2. **Metric Catalog Validation**:
           Executes `tpumonitoring.list_supported_metrics()` and verifies that
           core TPU metrics (e.g., `tensorcore_util`, `hbm_capacity_usage`, `ici_link_health`)
           are present in the returned list.

        3. **Environment Integrity Check**:
           Ensures the `libtpu` library can correctly interface with the TPU driver
           and hardware devices inside the container.
      """,
) as dag:

    for machine in MachineConfigMap:
      config = machine.value

      jobset_config = JobSet(
          jobset_name="sdk-monitoring-v6e-workload",
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
          image="python:3.11",
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
          dag_name="jobset_ttr_pod_delete",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
      )

      start_workload = jobset.run_workload.override(task_id="start_workload")(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=Workload.JAX_TPU_BENCHMARK
          ),
          namespace=jobset_config.namespace,
      )

      ensure_all_pods_running = jobset.wait_for_all_pods_running.override(
          task_id="ensure_all_pods_running"
      )(
          num_pods=(jobset_config.replicas * jobset_config.parallelism),
          node_pool=cluster_info,
      )

      pod_name = jobset.list_pod_names.override(task_id="list_pod_names")(
          node_pool=cluster_info,
          namespace=jobset_config.namespace,
      )
      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="sdk_verification"
      ) as verification_group:
        sdk_help_validation = (
            validate_monitoring_help.override(task_id="sdk_help_validation")
            .partial(info=cluster_info)
            .expand(pod_name=pod_name)
        )

        metrics_list_validation = (
            validate_metrics_list.override(task_id="metrics_list_validation")
            .partial(info=cluster_info)
            .expand(pod_name=pod_name)
        )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          namespace=jobset_config.namespace,
      ).as_teardown(
          setups=start_workload
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
          >> start_workload
          >> ensure_all_pods_running
          >> pod_name
          >> verification_group
          >> cleanup_workload
          >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
