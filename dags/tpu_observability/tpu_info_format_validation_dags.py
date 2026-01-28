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
tpu_info_format_validation_dag:
A DAG orchestrates the process of verifying TensorCore utilization metrics.
This is done by comparing data from Cloud Logging and Cloud Monitoring.

tpu_info_cli_validation_dags:
A DAG to validate the `tpu-info` CLI tool, ensuring help documentation,
version metadata, and process monitoring are functional inside TPU worker pods.
"""

import datetime

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils import tpu_info_format_util as tpu_info_format
from dags.tpu_observability.utils.jobset_util import JobSet, Workload


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="tpu_info_format_validation_dag",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule="0 20 * * *" if composer_env.is_prod_env() else None,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info", "TPU", "v6e-16"],
    description=(
        "This DAG verifies the format of the tables in the tpu-info output "
        "using tpu-info CLI tool. It includes 4 tables: TPU Chips, TPU "
        "Runtime Utilization, TensorCore Utilization, and TPU Buffer Transfer "
        "Latency."
    ),
    doc_md="""
      # Format Validation DAG
      # This DAG verifies the format of the tables in the tpu-info output.

      ### Description
      This DAG automates the validation of the tpu-info command-line tool's
      output format.It verifies the structure and content of key metric tables,
      including "TPU Chips", "TPU Runtime Utilization", "TensorCore
      Utilization", and "TPU Buffer Transfer Latency", by running the tool on a
      live GKE cluster with TPU node pools.

      ### Prerequisites
      This test requires an existing GKE cluster.
      A pre-built Docker image containing the necessary jax, libtpu, and
      tpu-info packages must also be available in a repository accessible
      by the GKE cluster.

      ### Procedures
      The DAG begins by creating temporary GKE TPU node pools for the test.
      Once the node pools are running, it schedules a Kubernetes JobSet and
      waits for the pods to become active. It then executes the tpu-info
      command within these pods to capture the raw text output. This output is
      parsed into structured tables, and a series of validation tasks check
      each table for the correct structure, row counts, and data formats.
      Finally, regardless of the test outcome, the DAG cleans up all created
      resources, including the JobSet and the temporary node pools.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    @task
    def generate_second_node_pool_name(
        node_pool_info: node_pool.Info,
    ) -> str:
      """Generates a second node pool name."""
      return f"{node_pool_info.node_pool_name}-2"

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name="tpu_info_format_validation_dag",
      )

      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="tpu_info_format_validation_dag",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      cluster_info_2 = node_pool.copy_node_pool_info_with_override.override(
          task_id="copy_node_pool_info_with_override"
      )(
          info=cluster_info,
          node_pool_name=generate_second_node_pool_name(cluster_info),
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="create_node_pool"
      ) as create_node_pool:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1",
            retries=2,
        )(
            node_pool=cluster_info,
        )

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
            retries=2,
        )(
            node_pool=cluster_info_2,
        )

      apply_time = jobset.run_workload.override(
          owner=test_owner.YUNA_T, task_id="run_workload"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      pod_names = jobset.list_pod_names.override(
          task_id="list_pod_names",
          retries=5,
          retry_delay=datetime.timedelta(seconds=10),
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=pod_names, job_apply_time=apply_time)

      outputs_of_tpu_info = (
          tpu_info_format.get_tpu_info_from_pod.override(task_id="get_tpu_info")
          .partial(info=cluster_info)
          .expand(pod_name=pod_names)
      )

      output_of_tpu_info = (
          tpu_info.parse_tpu_info_output.override(
              task_id="get_each_metric_table"
          )
          .partial()
          .expand(output=outputs_of_tpu_info)
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="verification_group"
      ) as verification_group:
        verify_table_amount_task = (
            tpu_info_format.verify_table_amount.override(
                task_id="verify_table_amount_task"
            )
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_tpu_chips_metric = (
            tpu_info_format.validate_chips_table.override(
                task_id="validate_tpu_chips_metric"
            )
            .partial(tpu_config=config)
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_runtime_metric = (
            tpu_info_format.validate_runtime_table.override(
                task_id="validate_runtime_metric"
            )
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_tensorcore_metric = (
            tpu_info_format.validate_tensorcore_table.override(
                task_id="validate_tensorcore_metric"
            )
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_latency_metric = (
            tpu_info_format.validate_latency_table.override(
                task_id="validate_latency_metric"
            )
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

      clean_up_workload = jobset.end_workload.override(
          task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      ).as_teardown(
          setups=apply_time
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="cleanup_node_pool"
      ) as cleanup_node_pool:
        cleanup_first_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_1",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info).as_teardown(
            setups=create_node_pool,
        )

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info_2).as_teardown(
            setups=create_node_pool,
        )

      chain(
          verify_table_amount_task,
          [
              validate_tpu_chips_metric,
              validate_runtime_metric,
              validate_tensorcore_metric,
              validate_latency_metric,
          ],
      )

      chain(create_first_node_pool, create_second_node_pool)

      chain(cleanup_first_node_pool, cleanup_second_node_pool)

      chain(
          jobset_config,
          cluster_info,
          cluster_info_2,
          create_node_pool,
          apply_time,
          pod_names,
          wait_for_job_start,
          outputs_of_tpu_info,
          output_of_tpu_info,
          verification_group,
          clean_up_workload,
          cleanup_node_pool,
      )
      # pylint: enable=pointless-statement


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="tpu_info_cli_validation_dags",
    start_date=datetime.datetime(2025, 8, 10),
    schedule=None,
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
        jobset_name="tpu-info-cli-validation-jobset",
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
          dag_name="tpu_info_cli_validation_dags",
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
        help_validation = (
            tpu_info_format.validate_help.override(task_id="validate_help")
            .partial(info=cluster_info)
            .expand(pod_name=pod_names)
        )

        version_validation = (
            tpu_info_format.validate_version.override(
                task_id="validate_version"
            )
            .partial(info=cluster_info)
            .expand(pod_name=pod_names)
        )

        process_validation = (
            tpu_info_format.validate_process.override(
                task_id="validate_process"
            )
            .partial(info=cluster_info)
            .expand(pod_name=pod_names)
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

      chain(
          cluster_info,
          create_node_pool,
          apply_time,
          pod_names,
          wait_for_job_start,
          verification_group,
          cleanup_workload,
          cleanup_node_pool,
      )
      # pylint: enable=pointless-statement
