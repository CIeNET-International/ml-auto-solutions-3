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

"""A DAG to test jobset time-to-recover metric using a jobset pod delete."""

import datetime
import random
import logging

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout


DAG_ID = "jobset_ttr_node_reboot"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 1, 21),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu-observability",
        "node_reboot",
        "TPU",
        "v6e-16",
    ],
    description=(
        "Tests JobSet TTR metric by rebooting a random TPU node to trigger "
        "recovery, then polls the metric for updates."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Random Node Reboot

      ### Description
      This DAG verifies that a TPU JobSet can recover from a hardware-level failure.
      It launches a JobSet, executes a `reboot` command on a random node via SSH,
      and uses a sensor to confirm that the TTR (Time-To-Recover) metric is recorded.

      ### Prerequisites
      This test requires an existing cluster to run.
      GKE Cluster with TPU v6e support.
      SSH access enabled (OS Login).

      ### Procedures
      First the node-pool is created, a jobset yaml is then launched on the cluster and given a short
      period of time to initialize. After this, a random node reboot is triggered via SSH to interrupt
      the jobset by taking one of the TPU nodes offline. A sensor is finally run which will poll
      Cloud Monitoring to detect that the jobset time-to-recover (TTR) metric has been updated,
      resulting in a success, or timeout and failure.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      selector = jobset.generate_node_pool_selector("jobset-ttr-node-reboot")

      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name=DAG_ID,
          node_pool_selector=selector,
          privileged=True,
      )

      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name=DAG_ID,
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
          node_pool_selector=selector,
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

      reboot_node = jobset.reboot_one_random_node.override(
          task_id="reboot_one_random_node"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_jobset_ttr_to_be_found"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          start_time=reboot_node,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info, jobset_config=jobset_config).as_teardown(
          setups=start_workload
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          selector,
          jobset_config,
          cluster_info,
          create_node_pool,
          start_workload,
          ensure_all_pods_running,
          reboot_node,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
