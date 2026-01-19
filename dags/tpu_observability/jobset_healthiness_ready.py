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

"""A DAG to test "Jobset Ready Healthiness" metric."""

import datetime

from airflow import models
from airflow.decorators import task
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_healthiness_ready",
    start_date=datetime.datetime(2025, 8, 10),
    schedule="30 19 * * *" if composer_env.is_prod_env() else None,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "healthiness",
        "tpu-obervability",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the 'Ready' status of jobset healthiness by "
        "comparing the number of 'Ready' replicas before and after "
        "a jobset is running."
    ),
    doc_md="""
      # JobSet Healthiness Test For the "Ready" Status

      ### Description
      This DAG automates the process of creating node-pools, ensuring the
      correct number of "Ready" replicas appear, then launching a jobset on
      multiple replicas to ensure the correct number begin running.

      ### Prerequisites
      This test requires an existing cluster to run.

      ### Procedures
      First two node-pools are created. The validation test is then run to
      check if the number of "Ready" replicas is 0. A jobset is then launched
      which uses 2 replicas. Once the jobset is running the jobs should
      quickly enter the "Ready" state. The number of found replicas is
      tested against the number of replicas which should be "Ready". If they
      match the DAG is a success.
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

    jobset_config = JobSet(
        jobset_name="jobset-healthiness-ready",
        namespace="default",
        max_restarts=0,
        replicated_job_name="tpu-job-slice",
        replicas=2,
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
          dag_name="jobset_healthiness_ready",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      cluster_info_2 = node_pool.copy_node_pool_info_with_override(
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

      validate_zero_replicas = jobset.validate_jobset_replica_number(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          replica_type="ready",
          correct_replica_num=0,
      )

      start_workload = jobset.run_workload(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=Workload.IDLE_READY_TPU_20M
          ),
          namespace=jobset_config.namespace,
      )

      validate_ready_replicas = jobset.validate_jobset_replica_number(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          replica_type="ready",
          correct_replica_num=jobset_config.replicas,
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

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          cluster_info
          >> cluster_info_2
          >> create_node_pool
          >> validate_zero_replicas
          >> start_workload
          >> validate_ready_replicas
          >> cleanup_workload
          >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
