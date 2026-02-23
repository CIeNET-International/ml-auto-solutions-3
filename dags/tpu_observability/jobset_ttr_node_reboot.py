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
import fabric
import paramiko
import io

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.log.secrets_masker import mask_secret
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task
from google.cloud import compute_v1

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from xlml.utils.tpu import add_ssh_key_to_oslogin
from xlml.utils.ssh import SshKeys, obtain_persist_ssh_keys


@task
def random_node_reboot(info: node_pool.Info, keys: SshKeys):
  """
  Selects a random node from the node pool and triggers a system reboot via SSH.

  Args:
    info: Configuration and metadata of the GKE node pool.
    keys: SSH key pair used for authentication.

  Returns:
    The name of the node that was rebooted.
  """
  # Retrieve all nodes from the pool and select one at random
  nodes = node_pool.list_nodes(info)
  if not nodes:
    raise RuntimeError(f"No nodes found in node pool {info.node_pool_name}")

  target_node_url = random.choice(nodes)
  target_node_name = target_node_url.split("/")[-1]
  logging.info(f"Selected node name: {target_node_name}")

  # Get the internal IP address of the instance
  instance_client = compute_v1.InstancesClient()
  instance = instance_client.get(
      project=info.project_id, zone=info.zone, instance=target_node_name
  )
  target_ip = instance.network_interfaces[0].network_i_p
  logging.info(f"Targeting node {target_node_name} at {target_ip} for reboot.")

  # OS Login Setup
  # Register public key and retrieve the POSIX username
  mask_secret(keys.private)
  add_ssh_key_to_oslogin(keys.public, info.project_id)
  os_user = keys.user

  # Execute Reboot via Fabric
  pkey = paramiko.RSAKey.from_private_key(io.StringIO(keys.private))

  # Establish connection using the dynamic OS Login username
  conn = fabric.Connection(
      host=target_ip,
      user=os_user,
      connect_kwargs={"pkey": pkey, "banner_timeout": 200},
  )

  try:
    logging.info(
        f"Sending 'sudo reboot' command to {target_ip} as user '{os_user}'..."
    )
    # Use warn=True because the connection will drop immediately upon reboot,
    # which is expected behavior for this operation.
    conn.run("sudo reboot", warn=True)
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Log unexpected errors but allow the task to proceed
    logging.warning(
        f"Reboot command issued, but connection closed with error: {e}"
    )
  finally:
    conn.close()

  return target_node_name


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

  @task
  def get_project_id(info: node_pool.Info):
    return info.project_id

  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH, dag_name="jobset_ttr_node_reboot"
      )

      get_keys = obtain_persist_ssh_keys.override(task_id="get_ssh_keys")()

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
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      ensure_all_pods_running = jobset.wait_for_all_pods_running.override(
          task_id="ensure_all_pods_running"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      node_reboot = random_node_reboot.override(task_id="random_node_reboot")(
          info=cluster_info, keys=get_keys
      )

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_jobset_ttr_to_be_found"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
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
          jobset_config,
          get_keys,
          cluster_info,
          create_node_pool,
          start_workload,
          ensure_all_pods_running,
          node_reboot,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
