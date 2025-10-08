"""A DAG to validate the status of a GKE node pool after changing its label."""

import datetime

from airflow import models
from airflow.exceptions import AirflowFailException
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import node_pool_util as node_pool
from google.cloud import logging


_THRESHOLD_SECONDS = 150.0


def check_duration_and_branch(**kwargs) -> str:
  """Determines which task to do next based on the given duration and threshold

  Reads the duration from XCom and returns the next Task ID
  based on the threshold.

  Returns:
      The Task ID ('wait_for_ttr' or 'skip_ttr_check') to proceed to.
  Raises:
      RuntimeError: If the operation duration could not be retrieved from XCom.
  """
  ti = kwargs["ti"]

  duration_seconds = ti.xcom_pull(task_ids="get_node_pool_update_duration")

  if duration_seconds is None:
    error_msg = (
        f"No update duration found for node pool {node_pool.node_pool_name}."
    )
    raise AirflowFailException(error_msg)

  if duration_seconds >= _THRESHOLD_SECONDS:
    logging.info(
        f"Duration ({duration_seconds:.2f}s) >= {_THRESHOLD_SECONDS}s. "
        f"Proceeding to TTR check."
    )
    return "wait_for_ttr"
  else:
    logging.info(
        f"Duration ({duration_seconds:.2f}s) < {_THRESHOLD_SECONDS}s. "
        f"Skipping TTR check."
    )
    return "skip_ttr_check"


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
      node pool, changing its node pool label, and verifying whether the node pool
      status is reported correctly.

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

  task_id = "wait_for_running_initial"
  wait_for_running_initial = node_pool.wait_for_status.override(
      task_id=task_id
  )(node_pool=node_pool_info, status=node_pool.Status.RUNNING)

  task_id = "change_node_pool_label"
  change_node_pool_label = node_pool.change_node_pool_label.override(
      task_id=task_id
  )(node_pool=node_pool_info)

  task_id = "wait_for_reconciling"
  wait_for_reconciling = node_pool.wait_for_status.override(task_id=task_id)(
      node_pool=node_pool_info, status=node_pool.Status.RECONCILING
  )

  task_id = "wait_for_recovered"
  wait_for_recovered = node_pool.wait_for_status.override(task_id=task_id)(
      node_pool=node_pool_info, status=node_pool.Status.RUNNING
  )

  task_id = "get_update_duration"
  get_update_duration = node_pool.get_node_pool_update_duration.override(
      task_id=task_id
  )(node_pool=node_pool_info)

  task_id = "check_duration_branch"
  check_ttr_threshold = BranchPythonOperator(
      task_id=task_id,
      python_callable=check_duration_and_branch,
      op_kwargs={"node_pool_info": node_pool_info},
      provide_context=True,
      dag=dag,
  )

  # Task: if node pool update duration >= 150 seconds, do the TTR check.
  task_id = "wait_for_ttr"
  wait_for_ttr = node_pool.wait_for_ttr.override(task_id=task_id)(
      node_pool=node_pool_info
  )

  # Task: if node pool update duration < 150 seconds, skip the TTR check.
  task_id = "skip_ttr_check"
  skip_ttr_check = EmptyOperator(task_id=task_id)

  task_id = "cleanup_node_pool"
  cleanup_node_pool = node_pool.delete.override(
      task_id=task_id, trigger_rule=TriggerRule.ALL_DONE
  )(node_pool=node_pool_info).as_teardown(
      setups=create_node_pool,
  )

  setup_and_execute_flow = (
      create_node_pool
      >> wait_for_provisioning
      >> wait_for_running_initial
      >> change_node_pool_label
      >> wait_for_reconciling
      >> wait_for_recovered
  )

  measurement_and_decision_flow = (
      wait_for_recovered >> get_update_duration >> check_ttr_threshold
  )

  check_ttr_threshold >> [wait_for_ttr, skip_ttr_check]

  [wait_for_ttr, skip_ttr_check] >> cleanup_node_pool
