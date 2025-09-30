"""A DAG to validate the status of a GKE node pool after changing its node label."""

import datetime

from airflow import models
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import node_pool_util as node_pool


with models.DAG(
    dag_id="change_node_pool_label",
    start_date=datetime.datetime(2025, 9, 30),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "change-node-pool-label"],
    description=(
        "This DAG tests the GKE nodel pool's status by changing its label and "
        "confirming the state transitions are correct."
    ),
    doc_md="""
      # GKE Node Pool Status Validation DAG

      ### Description
      This DAG automates the process of creating a GKE
      node pool, changing its node pool label, and verifying whether the node pool status is reported correctly.

      ### Prerequisites
      This test requires an existing cluster.

      ### Procedures
      It creates a node pool, waits for it from provisioning to be running,
      changes the node pool label to trigger reconciliation, waits for it to become
      running again, and finally cleans up.
      The node pool will be cleaned up after the tests.
    """,
) as dag:
  node_pool_info = node_pool.Info(
      project_id=models.Variable.get(
          "PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
      ),
      cluster_name=models.Variable.get(
          "CLUSTER_NAME", default_var="ryan-cluster-asia"
      ),
      node_pool_name=models.Variable.get(
          "NODE_POOL_NAME", default_var="ryan-v6e-autotest"
      ),
      location=models.Variable.get(
          "LOCATION", default_var=Region.ASIA_NORTHEAST1.value
      ),
      node_locations=models.Variable.get(
          "NODE_LOCATIONS", default_var=Zone.ASIA_NORTHEAST1_B.value
      ),
      num_nodes=models.Variable.get("NUM_NODES", default_var=4),
      machine_type=models.Variable.get(
          "MACHINE_TYPE", default_var="ct6e-standard-4t"
      ),
      tpu_topology=models.Variable.get("TPU_TOPOLOGY", default_var="4x4"),
  )

  task_id = "create_node_pool"
  create_node_pool = node_pool.create.override(task_id=task_id)(
      node_pool=node_pool_info
  )

  task_id = "wait_for_provisioning"
  wait_for_provisioning = node_pool.wait_for_status.override(task_id=task_id)(
      node_pool=node_pool_info, status=node_pool.Status.PROVISIONING
  )

  task_id = "wait_for_running"
  wait_for_running = node_pool.wait_for_status.override(task_id=task_id)(
      node_pool=node_pool_info, status=node_pool.Status.RUNNING
  )

  task_id = "change_node_label"
  change_node_label = node_pool.change_node_label.override(task_id=task_id)(
      node_pool=node_pool_info
  )

  task_id = "wait_for_reconciling"
  wait_for_reconciling = node_pool.wait_for_status.override(task_id=task_id)(
      node_pool=node_pool_info, status=node_pool.Status.RECONCILING
  )

  task_id = "wait_for_recovered"
  wait_for_recovered = node_pool.wait_for_status.override(task_id=task_id)(
      node_pool=node_pool_info, status=node_pool.Status.RUNNING
  )

  task_id = "cleanup_node_pool"
  cleanup_node_pool = node_pool.delete.override(
      task_id=task_id, trigger_rule=TriggerRule.ALL_DONE
  )(node_pool=node_pool_info).as_teardown(
      setups=create_node_pool,
  )

  normal_flow = (
      create_node_pool
      >> wait_for_provisioning
      >> wait_for_running
      >> change_node_label
      >> wait_for_reconciling
      >> wait_for_recovered
      >> cleanup_node_pool
  )
