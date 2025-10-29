"""A DAG to update the label of a node pool to make node pool unavailable for a while"""

import datetime

from airflow import models
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.configs.common import MachineConfigMap


LABELS_TO_UPDATE = {"env": "prod"}

with models.DAG(
    dag_id="gke_node_pool_label_update",
    start_date=datetime.datetime(2025, 8, 1),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status"],
    description=(
        "This DAG tests whether the status of a GKE node pool changes as "
        "expected after its labels are updated, triggering reconciliation."
    ),
    doc_md="""
      # GKE Node Pool Label Update Status Validation DAG

      ### Description
      This DAG automates the process of going through the lifecycle of a GKE
      node pool and verifies whether the node pool status is reported correctly
      after a configuration change (label update) is applied.

      ### Prerequisites
      This test requires an existing cluster.

      ### Procedures
      It creates a node pool, waits for it to be running, updates a label to
      trigger reconciliation, waits for it to become running again (recovering
      from the update), and finally cleans up by deleting the node pool.
    """,
) as dag:
  for machine_config_enum in MachineConfigMap:
    config = machine_config_enum.value
    node_pool_info = node_pool.Info(
        project_id=models.Variable.get(
            "PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
        ),
        cluster_name=models.Variable.get(
            "CLUSTER_NAME", default_var="tpu-observability-automation"
        ),
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME", default_var="node-pool-status-v6e-autotest"
        ),
        location=models.Variable.get(
            "LOCATION", default_var=Region.US_EAST5.value
        ),
        node_locations=models.Variable.get(
            "NODE_LOCATIONS", default_var=Zone.US_EAST5_B.value
        ),
        num_nodes=models.Variable.get("NUM_NODES", default_var=4),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    with TaskGroup(group_id=config.tpu_version.value):
      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=node_pool_info,
          reservation="cloudtpu-20250131131310-2118578099",
      )

      wait_for_availability = node_pool.wait_for_availability.override(
          task_id="wait_for_initial_availability"
      )(node_pool=node_pool_info, availability=True)

      update_node_pool_label = node_pool.update_labels.override(
          task_id="update_node_pool_label"
      )(node_pool=node_pool_info, node_labels=LABELS_TO_UPDATE)

      wait_for_unavailable = node_pool.wait_for_availability.override(
          task_id="wait_for_unavailability_after_update"
      )(node_pool=node_pool_info, availability=False)

      wait_node_pool_recovered = node_pool.wait_for_availability.override(
          task_id="wait_for_recovery"
      )(node_pool=node_pool_info, availability=True)

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=[create_node_pool],
      )

      (
          create_node_pool
          >> wait_for_availability
          >> update_node_pool_label
          >> wait_for_unavailable
          >> wait_node_pool_recovered
          >> cleanup_node_pool
      )
