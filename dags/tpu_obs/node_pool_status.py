"""Manages the lifecycle of a GKE node pool and verifies its status as an Airflow DAG.
"""

import copy
import datetime

from airflow import models
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator

from dags.common.vm_resource import Project
from dags.common.vm_resource import Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_obs.utils import node_pool_util as node_pool


@task.branch(trigger_rule="all_done")
def check_delete_node_pool_status(**context):
  """Checks the status of the upstream task: delete_node_pool.

  The 'all_done' trigger rule ensures this check runs regardless of
  whether delete_node_pool succeeds or fails.
  """
  # Get the DAG Run object from the Airflow context
  dag_run = context["dag_run"]
  delete_node_pool_task_instance = dag_run.get_task_instance(
      task_id="delete_node_pool"
  )

  if delete_node_pool_task_instance.state in ["failed", "skipped"]:
    return "cleanup_node_pool"
  else:
    return "wait_for_stopping"


with models.DAG(
    dag_id="gke_node_pool_status",
    start_date=datetime.datetime(2025, 7, 30),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status"],
    description=(
        "This DAG tests whether the status of a GKE node pool changes as"
        " expected according to its lifecycle."
    ),
    doc_md="""
        ### GKE Node Pool Status Management DAG

        This DAG automates the lifecycle of a GKE node pool for testing purposes.
        This test requires an existing cluster, and if there are any other limitations
        It will create a node-pool under the specified cluster, and clean it up after the tests

        It creates a node pool, waits for it to be running, deletes a random node to trigger reconciliation,
        waits for it to become running again, and finally cleans up.It also
        tests the error state by creating a node pool with invalid parameters
        and verifies thatthe status changes to ERROR."
    """,
) as dag:

  node_pool_info = node_pool.Info(
      project_id=models.Variable.get(
          "PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
      ),
      cluster_name=models.Variable.get(
          "CLUSTER_NAME", default_var="yuna-xpk-v6e-2"
      ),
      node_pool_name=models.Variable.get(
          "NODE_POOL_NAME", default_var="yuna-v6e-autotest"
      ),
      location=models.Variable.get("LOCATION", default_var="asia-northeast1"),
      node_locations=models.Variable.get(
          "NODE_LOCATIONS", default_var=Zone.ASIA_NORTHEAST1_B.value
      ),
      num_nodes=models.Variable.get("NUM_NODES", default_var=4),
      machine_type=models.Variable.get(
          "MACHINE_TYPE", default_var="ct6e-standard-4t"
      ),
      tpu_topology=models.Variable.get("TPU_TOPOLOGY", default_var="4x4"),
  )

  problematic_node_pool_info = copy.deepcopy(node_pool_info)
  problematic_node_pool_info.node_pool_name += "-wrong"
  # Choosing a region that is different from the cluster location but still
  # compatible with the specified TPU cause the cluster creation to fail
  # due to mismatched node locations.
  problematic_node_pool_info.node_locations = models.Variable.get(
      "WRONG_NODE_LOCATION", default_var=Zone.ASIA_EAST1_C.value
  )

  id = "create_node_pool"
  create_node_pool = node_pool.create.override(task_id=id)(
      node_pool=node_pool_info
  )

  id = "wait_for_provisioning"
  wait_for_provisioning = node_pool.wait_for_status.override(task_id=id)(
      node_pool=node_pool_info, status=node_pool.Status.PROVISIONING
  )

  id = "wait_for_running"
  wait_for_running = node_pool.wait_for_status.override(task_id=id)(
      node_pool=node_pool_info, status=node_pool.Status.RUNNING
  )

  id = "delete_node"
  delete_node = node_pool.delete_one_random_node.override(task_id=id)(
      node_pool=node_pool_info
  )

  id = "wait_for_repair"
  wait_for_repair = node_pool.wait_for_status.override(task_id=id)(
      node_pool=node_pool_info, status=node_pool.Status.RECONCILING
  )

  id = "wait_for_recovered"
  wait_for_recovered = node_pool.wait_for_status.override(task_id=id)(
      node_pool=node_pool_info, status=node_pool.Status.RUNNING
  )

  id = "delete_node_pool"
  delete_node_pool = node_pool.delete.override(task_id=id)(
      node_pool=node_pool_info
  )

  id = "wait_for_stopping"
  wait_for_stopping = node_pool.wait_for_status.override(task_id=id)(
      node_pool=node_pool_info, status=node_pool.Status.STOPPING
  )

  id = "cleanup_node_pool"
  cleanup_node_pool = node_pool.delete.override(
      task_id=id, trigger_rule="all_done"
  )(node_pool=node_pool_info)

  # Intentionally create a node pool with problematic configurations
  # to validate that it enters the ERROR state.
  id = "create_problematic_node_pool_info"
  create_problematic_node_pool_info = node_pool.create.override(task_id=id)(
      # The failure is intentionally ignored because we want to validate
      # that the status of the node pool (which fails to be created) is "ERROR".
      node_pool=problematic_node_pool_info,
      ignore_failure=True,
  )

  id = "wait_for_error"
  wait_for_error = node_pool.wait_for_status.override(task_id=id)(
      node_pool=problematic_node_pool_info, status=node_pool.Status.ERROR
  )

  id = "cleanup_wrong_node_pool"
  cleanup_wrong_node_pool = node_pool.delete.override(
      task_id=id, trigger_rule="all_done"
  )(node_pool=problematic_node_pool_info)

  check_clean_up_status = check_delete_node_pool_status()

  # Add a final task with trigger_rule=all_success to ensure the DAG's
  # final status accurately reflects upstream failures.
  id = "final_status_check"
  final_status_check = EmptyOperator(
      task_id=id,
      trigger_rule="all_success",
  )

  normal_path_flow = (
      create_node_pool
      >> wait_for_provisioning
      >> wait_for_running
      >> delete_node
      >> wait_for_repair
      >> wait_for_recovered
      >> delete_node_pool
      >> check_clean_up_status
      >> [cleanup_node_pool, wait_for_stopping]
  )
  # delete_node_pool >> check_clean_up_status >> cleanup_node_pool

  # check_clean_up_status >> final_status_check

  wrong_path_flow = (
      create_problematic_node_pool_info
      >> wait_for_error
      >> cleanup_wrong_node_pool
  )

  [
      wait_for_provisioning,
      wait_for_running,
      wait_for_repair,
      wait_for_recovered,
      wait_for_stopping,
  ] >> final_status_check
