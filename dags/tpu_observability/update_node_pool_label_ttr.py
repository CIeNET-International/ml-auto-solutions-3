"""A DAG to validate the status of a GKE node pool after changing its label."""

import datetime
import logging

from airflow import models
from airflow.exceptions import AirflowFailException
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.configs.common import MachineConfigMap
from dags.tpu_observability.utils import node_pool_util as node_pool

_THRESHOLD_SECONDS = 150.0

with models.DAG(
    dag_id="update_node_pool_label_ttr",
    start_date=datetime.datetime(2025, 9, 30),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=[
        "gke",
        "tpu-observability",
        "update-node-pool-label-ttr",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the GKE nodel pool's status by updating its label and "
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
  for machine in MachineConfigMap:
    config = machine.value
    node_pool_info = node_pool.Info(
        project_id=models.Variable.get("PROJECT_ID", default_var="cienet-cmcs"),
        cluster_name=models.Variable.get(
            "CLUSTER_NAME", default_var="tpu-observability-automation"
        ),
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME",
            default_var="update-node-pool-label-ttr-v6e-autotest",
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

    LABELS_TO_UPDATE = {"test_key": "test_val"}

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
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

      task_id = "update_node_pool_label"
      update_node_pool_label = node_pool.update_labels.override(
          task_id=task_id
      )(node_pool=node_pool_info, node_labels=LABELS_TO_UPDATE)

      task_id = "wait_for_recovered"
      wait_for_recovered = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RUNNING
      )

      task_id = "get_node_pool_update_duration"
      get_node_pool_update_duration = (
          node_pool.get_node_pool_update_duration.override(task_id=task_id)(
              node_pool=node_pool_info
          )
      )

      task_id = "determine_next_branch"
      determine_next_branch = BranchPythonOperator(
          task_id=task_id,
          python_callable=node_pool.check_duration_and_determine_branch,
          op_kwargs={
              "config": config,
              "_THRESHOLD_SECONDS": _THRESHOLD_SECONDS,
          },
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

      _ = (
          create_node_pool
          >> wait_for_provisioning
          >> wait_for_running
          >> update_node_pool_label
          >> wait_for_recovered
          >> get_node_pool_update_duration
          >> determine_next_branch
          >> [wait_for_ttr, skip_ttr_check]
          >> cleanup_node_pool
      )
