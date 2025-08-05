"""Manages the lifecycle of a GKE node pool and verifies its status as an Airflow DAG.
"""

import datetime

from airflow import models
from airflow.operators.empty import EmptyOperator

from dags.common.vm_resource import Project
from dags.common.vm_resource import Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_obs.utils import node_pool_util as node_pool

with models.DAG(
    dag_id="gke_node_pool_status_2",
    start_date=datetime.datetime(2025, 7, 30),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status"],
    description=(
        "This DAG tests whether the status of a GKE node pool changes as"
        " expected according to its lifecycle.It creates a node pool, waits for"
        " it to be running, deletes a random node to trigger reconciliation,"
        " waits for it to become running again, and finally cleans up.It also"
        " tests the error state by creating a node pool with invalid parameters"
        " and verifies thatthe status changes to ERROR."
    ),
    doc_md="""
    ### GKE Node Pool Status Management DAG

    This DAG automates the lifecycle of a GKE node pool for testing purposes.
    This test requires an existing cluster, and if there are any other limitations
    It will create a node-pool under the specified cluster, and clean it up after the tests

    It performs the following steps:
    1.  **Normal Path**: Creates a node pool, waits for it to be running, deletes a random node to trigger reconciliation, waits for it to become running again, and finally cleans up.
    2.  **Error Path**: Concurrently, it attempts to create a node pool with invalid parameters to test the error state, and then cleans up.
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

  problematic_node_pool_info = node_pool.Info(
      project_id=node_pool_info.project_id,
      cluster_name=node_pool_info.cluster_name,
      node_pool_name=node_pool_info.node_pool_name + "-wrong",
      location=node_pool_info.location,
      # Choosing a region that is different from the cluster location but still
      # compatible with the specified TPU cause the cluster creation to fail
      # due to mismatched node locations.
      node_locations=models.Variable.get(
          "WRONG_NODE_LOCATION", default_var="asia-east1-c"
      ),
      num_nodes=node_pool_info.num_nodes,
      machine_type=node_pool_info.machine_type,
      tpu_topology=node_pool_info.tpu_topology,
  )

  """STEP 1: Creates the GKE node pool."""
  create_node_pool = node_pool.create.override(task_id="create_node_pool")(
      node_pool=node_pool_info)

  """STEP 2: Validating Provisioning Status."""
  wait_for_provisioning = node_pool.wait_for_status.override(
      task_id="wait_for_provisioning"
  )(node_pool=node_pool_info,
    status=node_pool.Status.PROVISIONING,)

  """STEP 3: Validating Running Status."""
  wait_for_running = node_pool.wait_for_status.override(
      task_id="wait_for_running"
  )(node_pool=node_pool_info, status=node_pool.Status.RUNNING)

  """STEP 4: Deleting a random node to trigger reconciliation."""
  delete_node = node_pool.delete_one_random_node.override(
      task_id="delete_node")(node_pool=node_pool_info)

  """STEP 5: Validating Reconciling Status."""
  wait_for_repair = node_pool.wait_for_status.override(
      task_id="wait_for_repair"
  )(node_pool=node_pool_info, status=node_pool.Status.RECONCILING)

  """STEP 6: Validating Running Status After Repair."""
  wait_for_repair_completes = node_pool.wait_for_status.override(
      task_id="wait_for_repair_completes"
  )(node_pool=node_pool_info, status=node_pool.Status.RUNNING)

  # TODO: In this DAG, cleaning up and verifying the state is part of the
  # requirement. However, in other DAGs, resource cleanup may fail, but it is
  # not part of the verification requirement. Therefore, even if the
  # cleanup process fails, the overall task should still be considered
  # successful. Please confirm what the overall task result would be when
  # trigger_rule=ALL_DONE.
  """STEP 7: Cleaning up - Deleting Node Pool."""
  delete_node_pool = node_pool.delete.override(
      task_id="delete_node_pool", trigger_rule="all_done"
  )(node_pool=node_pool_info)

  """STEP 8: Validating Stopping Status."""
  wait_for_stopping = node_pool.wait_for_status.override(
      task_id="wait_for_stopping", trigger_rule="all_done"
  )(node_pool=node_pool_info, status=node_pool.Status.STOPPING)

  # This intentionally creates a node pool in an ERROR state. The GKE NodePool
  # object is created, but VM provisioning fails because the invalid
  # 'node_location' prevents a required GCE placement policy from being found or
  # created, resulting in a "resource not found" error from the GCE API.

  # This task must be successful in airflow, if it have some issues
  # next task will go to error state.
  """STEP 1: Creating Error Node Pool."""
  create_problematic_node_pool_info = node_pool.create.override(
      task_id="create_problematic_node_pool_info"
  )(
      # We ignore the failure is because we intensionally want to validate
      # that the status of this not created node-pool should be "ERROR"
      node_pool=problematic_node_pool_info, ignore_failure=True
  )
  """STEP 2: Validating Error Status."""
  wait_for_error = node_pool.wait_for_status.override(task_id="wait_for_error")(
      node_pool=problematic_node_pool_info,
      status=node_pool.Status.ERROR
  )
  """STEP 3: Cleaning up - Deleting Error Node Pool."""
  delete_wrong_node_pool = node_pool.delete.override(
      task_id="delete_wrong_node_pool",
      trigger_rule="all_done"
    )(node_pool=problematic_node_pool_info)

  # Add a final task with trigger_rule=all_success to ensure the DAG's
  # final status accurately reflects upstream failures.
  """Final Task to ensure the DAG completes."""
  end = EmptyOperator(
      task_id="final_status_check",
      trigger_rule="all_success",
  )

  normal_path_flow = (
      create_node_pool
      >> wait_for_provisioning
      >> wait_for_running
      >> delete_node
      >> wait_for_repair
      >> wait_for_repair_completes
      >> delete_node_pool
      >> wait_for_stopping
      >> end
  )

  wrong_path_flow = (
      create_problematic_node_pool_info
      >> wait_for_error
      >> delete_wrong_node_pool
      >> end
  )
