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
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task
from airflow.providers.google.cloud.hooks.compute_ssh import ComputeEngineSSHHook

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH


@task
def random_node_reboot(info: node_pool.Info):
  """
  Selects a random node from the pool and triggers a system-level reboot via SSH.

  This task simulates a hardware failure or a maintenance event to verify that
  the JobSet can gracefully recover. By using a node reboot instead of a simple
  pod deletion, we test the full recovery path, including node availability
  checks and TPU stack re-initialization.

  Args:
    info: Node pool and cluster information.

  Returns:
    str: The name of the node that was targeted for the reboot.
  """
  nodes = node_pool.list_nodes(info)
  target = random.choice(nodes)

  logging.info(f"Targeting node {target} for reboot.")

  node_pool.execute_ssh_command(
      node_name=target, node_pool=info, command=node_pool.NodeCommands.REBOOT
  )
  return target


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_ttr_node_reboot",
    start_date=datetime.datetime(2026, 1, 21),
    schedule="0 18 * * *" if composer_env.is_prod_env() else None,
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
        "This DAG tests the JobSet time-to-recover metric by rebooting a random "
        "TPU node to trigger a recovery, then polls the metric to check for updates."
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

    jobset_config = JobSet(
        jobset_name="ttr-node-reboot-v6e-workload",
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
          dag_name="jobset_ttr_node_reboot",
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

      node_reboot = random_node_reboot.override(task_id="random_node_reboot")(
          info=cluster_info,
      )

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_jobset_ttr_to_be_found"
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
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

      chain(
          create_node_pool,
          start_workload,
          ensure_all_pods_running,
          node_reboot,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
