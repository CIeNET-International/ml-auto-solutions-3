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

"""A DAG to validate the status of a GKE node pool through its lifecycle."""

import copy
import datetime
import logging

from airflow import models
from airflow.decorators import task
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags.map_reproducibility.utils import constants
from dags.common.vm_resource import Region, Zone
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.configs.common import MachineConfigMap
from dags.tpu_observability.utils import subprocess_util as subprocess


@task.branch
def branch_on_node_pool_existence(
    node_pool_info: node_pool.Info, cleanup_task_id: str, skip_task_id: str
):
  """
  Checks if a GKE node pool exists and branches to either the cleanup task
  or a skip task.

  Args:
      node_pool_info: An instance of the node_pool.Info class.
      cleanup_task_id: The task ID to branch to if the node pool exists.
      skip_task_id: The task ID to branch to if the node pool does not exist.

  Returns:
      The task ID string to branch to.
  """
  command = (
      f"gcloud container node-pools describe {node_pool_info.node_pool_name} "
      f"--project={node_pool_info.project_id} "
      f"--cluster={node_pool_info.cluster_name} "
      f"--location={node_pool_info.location} "
      "--format=json"
  )
  try:
    subprocess.run_exec(command)
    logging.info(
        "Node pool '%s' exists. Branching to '%s'.",
        node_pool_info.node_pool_name,
        cleanup_task_id,
    )
    return cleanup_task_id
  except Exception as e:
    error_message = str(e)
    logging.info("Error while checking node pool existence: %s", error_message)
    # When gcloud describe fails because the resource is not found, subprocess.run_exec
    # will raise an exception, and the error message will contain "message=Not found" or "code=404".
    if "message=Not found" in error_message or "code=404" in error_message:
      logging.info(
          "Node pool '%s' does not exist (caught exception). Branching to '%s'.",
          node_pool_info.node_pool_name,
          skip_task_id,
      )
      return skip_task_id
    else:
      logging.warning(
          "An unexpected error occurred while checking node pool '%s' existence: %s. "
          "Defaulting to attempting cleanup by branching to '%s'.",
          node_pool_info.node_pool_name,
          error_message,
          cleanup_task_id,
      )
      return cleanup_task_id


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="gke_node_pool_status",
    start_date=datetime.datetime(2025, 8, 1),
    schedule=constants.Schedule.DAILY_PST_6PM,
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status", "TPU", "v6e-16"],
    description=(
        "This DAG tests whether the status of a GKE node pool changes as "
        "expected according to its lifecycle."
    ),
    doc_md="""
      # GKE Node Pool Status Validation DAG

      ### Description
      This DAG automates the process of going through the lifecycle of a GKE
      node pool and verifies whether the node pool status is reported correctly.

      ### Prerequisites
      This test requires an existing cluster.

      ### Procedures
      It creates a node pool, waits for it from provisioning to be running,
      deletes a random node to trigger reconciliation, waits for it to become
      running again, and finally cleans up.
      It also tests the error state by creating a node pool with invalid
      parameters and verifies that the status changes to error.
      All node-pool will be cleaned up clean it up after the tests.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value
    node_pool_info = node_pool.Info(
        project_id=models.Variable.get("PROJECT_ID", default_var="cienet-cmcs"),
        cluster_name=models.Variable.get(
            "CLUSTER_NAME", default_var="tpu-observability-automation"
        ),
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME", default_var="node-pool-status-v6e-autotest"
        ),
        location=models.Variable.get(
            "LOCATION", default_var=Region.US_CENTRAL1.value
        ),
        node_locations=models.Variable.get(
            "NODE_LOCATIONS", default_var=Zone.US_CENTRAL1_B.value
        ),
        num_nodes=models.Variable.get("NUM_NODES", default_var=4),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    problematic_node_pool_info = copy.deepcopy(node_pool_info)
    problematic_node_pool_info.node_pool_name += "-wrong"
    # Choosing a region that is different from the cluster location but still
    # compatible with the specified TPU cause the cluster creation to fail
    # due to mismatched node locations.
    problematic_node_pool_info.node_locations = models.Variable.get(
        "WRONG_NODE_LOCATION", default_var=Zone.ASIA_EAST1_C.value
    )

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      task_id = "create_node_pool"
      create_node_pool = node_pool.create.override(task_id=task_id)(
          node_pool=node_pool_info,
          reservation="cloudtpu-20251107233000-1246578561",
      )

      task_id = "wait_for_provisioning"
      wait_for_provisioning = node_pool.wait_for_status.override(
          task_id=task_id
      )(node_pool=node_pool_info, status=node_pool.Status.PROVISIONING)

      task_id = "wait_for_running"
      wait_for_running = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RUNNING
      )

      task_id = "delete_node"
      delete_node = node_pool.delete_one_random_node.override(task_id=task_id)(
          node_pool=node_pool_info
      )

      task_id = "wait_for_repair"
      wait_for_repair = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RECONCILING
      )

      task_id = "wait_for_recovered"
      wait_for_recovered = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RUNNING
      )

      task_id = "delete_node_pool"
      delete_node_pool = node_pool.delete.override(task_id=task_id)(
          node_pool=node_pool_info
      )

      task_id = "wait_for_stopping"
      wait_for_stopping = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.STOPPING
      )

      check_and_branch_cleanup = branch_on_node_pool_existence.override(
          task_id="check_and_branch_cleanup",
          trigger_rule=TriggerRule.ONE_SUCCESS,
      )(
          node_pool_info=node_pool_info,
          cleanup_task_id=f"v{config.tpu_version.value}.cleanup_node_pool",
          skip_task_id=f"v{config.tpu_version.value}.skip_cleanup",
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_node_pool,
      )

      skip_cleanup = DummyOperator(task_id="skip_cleanup")

      # Intentionally create a node pool with problematic configurations
      # to validate that it enters the ERROR state.
      task_id = "create_problematic_node_pool_info"
      create_problematic_node_pool_info = node_pool.create.override(
          task_id=task_id
      )(
          node_pool=problematic_node_pool_info,
          # The failure is intentionally ignored because we want to validate
          # that the status of the node pool (which fails to be created) is
          # "ERROR".
          ignore_failure=True,
      )

      task_id = "wait_for_error"
      wait_for_error = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=problematic_node_pool_info, status=node_pool.Status.ERROR
      )

      task_id = "cleanup_wrong_node_pool"
      cleanup_wrong_node_pool = node_pool.delete.override(
          task_id=task_id, trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=problematic_node_pool_info).as_teardown(
          setups=create_problematic_node_pool_info,
      )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      normal_flow = (
          create_node_pool
          >> wait_for_provisioning
          >> wait_for_running
          >> delete_node
          >> wait_for_repair
          >> wait_for_recovered
          >> delete_node_pool
          >> wait_for_stopping
          >> check_and_branch_cleanup
      )
      check_and_branch_cleanup >> [cleanup_node_pool, skip_cleanup]

      flow_for_error_state = (
          create_problematic_node_pool_info
          >> wait_for_error
          >> cleanup_wrong_node_pool
      )
      # pylint: enable=pointless-statement
