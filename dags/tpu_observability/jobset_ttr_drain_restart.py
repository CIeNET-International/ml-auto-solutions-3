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

"""A DAG to test jobset time-to-recover metric after a node-pool drained."""

import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.node_pool_util import NodeOperationSpec
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_ttr_drain_restart",
    start_date=datetime.datetime(2025, 8, 10),
    schedule="0 20 * * *" if composer_env.is_prod_env() else None,
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
        "This DAG tests using a node drain to interrupt a jobset, then "
        "verifies if the jobset restarts and polls the time-to-recover-"
        "metric to check if it is updated."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Node Drain

      ### Description
      This DAG automates the process of creating a node-pool and launching a JobSet.
      It then performs a node drain on a node where the JobSet is running to
      trigger an interruption. The DAG monitors if the JobSet successfully
      restarts and verifies if the JobSet TTR (Time-To-Recover) metric is
      properly updated. Finally, the DAG cleans up the JobSet and node-pool.

      ### Prerequisites
      This test requires an existing GKE cluster to run.

      ### Procedures
      1. Setup: A dedicated node-pool is created to host the JobSet.
      2. Launch: A JobSet YAML is deployed and given time to reach a 'Running' state.
      3. Interruption (Drain): The DAG identifies a node hosting a JobSet Pod and
         executes a `kubectl drain` command to evict the Pod and trigger a restart.
      4. Verification: A sensor runs to detect if the JobSet has recovered and if the
         time-to-recover metric has been updated in the monitoring system.
         Success is determined by the metric update; otherwise, it will timeout and fail.
      5. Cleanup: The JobSet is deleted and the node-pool is torn down.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name="jobset_ttr_drain_restart",
      )

      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="jobset_ttr_drain_restart",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
      )

      start_workload = jobset.run_workload.override(task_id="start_workload")(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      ensure_all_pods_running = jobset.wait_for_all_pods_running.override(
          task_id="ensure_all_pods_running"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      drained_node = node_pool.delete_one_random_node.override(
          task_id="drained_node"
      )(
          node_pool=cluster_info,
          spec=NodeOperationSpec.Drain(),
      )

      uncordon_node = node_pool.uncordon_node.override(task_id="uncordon_node")(
          node_pool=cluster_info, node_name=drained_node
      )

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_metric_upload"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      ).as_teardown(
          setups=start_workload
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          jobset_config,
          cluster_info,
          create_node_pool,
          start_workload,
          ensure_all_pods_running,
          drained_node,
          uncordon_node,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
