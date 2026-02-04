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

"""A DAG to test "Jobset Suspended Healthiness" metric."""

import datetime

from airflow import models
from airflow.decorators import task
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.models.baseoperator import chain

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_healthiness_suspended",
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
        "This DAG tests the 'Suspended' status of jobset healthiness by "
        "comparing the number of 'Ready' replicas before and after "
        "a jobset is running."
    ),
    doc_md="""
      # JobSet Healthiness Test For the "Suspended" Status
      ### Description
      This DAG automates the process of creating node-pools, ensuring the
      correct number of "Ready" replicas appear, then launching a jobset on
      multiple replicas to ensure the correct number begin running.
      ### Prerequisites
      This test requires an existing cluster to run.
      ### Procedures
      First two node-pools are created. The validation test is then run to
      check if the number of "Suspended" replicas is 0. A jobset is then launched
      which uses 2 replicas. Once the jobset is running the jobs should
      quickly enter the "Ready" state. Then using command to suspend entire jobset.
      The number of found replicas is tested against the number of replicas which
      should be "Suspended". If they match the DAG is a success.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    @task
    def generate_second_node_pool_name(
        pool_info: node_pool.Info,
    ) -> str:
      """Generates a second node pool name."""
      return f"{pool_info.node_pool_name}-2"

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name="jobset_healthiness_suspended",
      )

      node_pool_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="jobset_healthiness_suspended",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      node_pool_info_2 = node_pool.copy_node_pool_info_with_override(
          info=node_pool_info,
          node_pool_name=generate_second_node_pool_name(node_pool_info),
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="create_node_pool"
      ) as create_node_pool:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1",
        )(
            node_pool=node_pool_info,
        )

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
        )(
            node_pool=node_pool_info_2,
        )

      validate_zero_replicas = jobset.wait_for_jobset_replica_number.override(
          task_id="validate_zero_replicas"
      )(
          node_pool=node_pool_info,
          jobset_config=jobset_config,
          replica_type="suspended",
          correct_replica_num=0,
      )

      start_workload = jobset.run_workload.override(task_id="start_workload")(
          node_pool=node_pool_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      suspend_jobset = jobset.suspended_jobset.override(
        task_id="suspend_jobset"
      )(
          node_pool=node_pool_info,
          jobset_config=jobset_config,
      )

      validate_suspended_replicas = (
          jobset.wait_for_jobset_replica_number.override(
              task_id="validate_suspended_replicas"
          )
      )(
          node_pool=node_pool_info,
          jobset_config=jobset_config,
          replica_type="suspended",
          correct_replica_num=jobset_config.replicas,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=node_pool_info, jobset_config=jobset_config
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
        )(node_pool=node_pool_info).as_teardown(
            setups=create_node_pool,
        )

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
        )(node_pool=node_pool_info_2).as_teardown(
            setups=create_node_pool,
        )

      chain(
          node_pool_info,
          node_pool_info_2,
          create_node_pool,
          validate_zero_replicas,
          start_workload,
          suspend_jobset,
          validate_suspended_replicas,
          cleanup_workload,
          cleanup_node_pool,
      )
