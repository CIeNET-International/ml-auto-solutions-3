import copy
import datetime
import logging  # CHANGE: Added import for logging.
import os  # CHANGE: Added import for os, although not strictly used in the final version, good practice if file paths need manipulation.
import yaml  # CHANGE: Added import for yaml to parse the configuration file.

from airflow import models
from airflow.operators.python import PythonOperator  # CHANGE: Added import for PythonOperator.
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

# Assuming these modules are available in the Airflow environment.
from dags.map_reproducibility.utils import constants
from dags.common.vm_resource import Project, Region, Zone
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.configs.common import MachineConfigMap

# --- Configuration Loading from GCS ---
# CHANGE: Defined the path to the YAML configuration file in GCS.
# Composer environment syncs the contents of the GCS bucket's 'dags/' folder
# to the Airflow worker's local filesystem, typically at /home/airflow/gcs/dags/.
CONFIG_FILE_PATH = "/home/airflow/gcs/dags/dags/tpu_observability/configs/gke_dag_config.yaml"

# CHANGE: Added a function to load DAG configurations from a YAML file.
def load_dag_config(file_path: str) -> dict:
  """Loads DAG configuration from a YAML file in GCS (synced locally).

  Args:
    file_path: The local path to the YAML configuration file.

  Returns:
    A dictionary containing the DAG configurations.

  Raises:
    yaml.YAMLError: If the YAML file is malformed.
  """
  try:
    with open(file_path, 'r') as f:
      return yaml.safe_load(f)
  except FileNotFoundError:
    print(f"Configuration file not found: {file_path}. Using hardcoded defaults.")
    # CHANGE: Hardcoded default values are provided as a fallback.
    return {
        "project_id": Project.TPU_PROD_ENV_ONE_VM.value,
        "cluster_name": "tpu-observability-automation",
        "node_pool_name": "node-pool-status-v6e-autotest",
        "location": Region.US_EAST5.value,
        "node_locations": Zone.US_EAST5_B.value,
        "num_nodes": 4,
        "wrong_node_location": Zone.ASIA_EAST1_C.value,
        "reservation": "cloudtpu-20250131131310-2118578099",
    }
  except yaml.YAMLError as e:
    print(f"Error parsing YAML file {file_path}: {e}")
    raise

# CHANGE: Load configurations when the DAG file is parsed, using the new function.
dag_config = load_dag_config(CONFIG_FILE_PATH)

# CHANGE: Added a Python function to print the loaded configurations.
def print_dag_configurations(**kwargs) -> None:
  """Prints the loaded DAG configurations to Airflow logs.

  Args:
    **kwargs: Expected to contain 'dag_config' with the loaded configurations.
  """
  logger = logging.getLogger(__name__)
  dag_config = kwargs.get("dag_config", {})
  logger.info("--- Loaded DAG Configurations ---")
  if not dag_config:
    logger.warning("No DAG configurations loaded.")
    return

  for key, value in dag_config.items():
    logger.info(f"{key}: {value}")
  logger.info("-------------------------------")

with models.DAG(
    dag_id="gke_node_pool_status",
    start_date=datetime.datetime(2025, 8, 1),
    schedule=constants.Schedule.DAILY_PST_6PM,
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status"],
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
  # CHANGE: Added a PythonOperator task to print the loaded configurations.
  print_config_task = PythonOperator(
      task_id="print_dag_configurations",
      python_callable=print_dag_configurations,
      op_kwargs={"dag_config": dag_config}, # Pass the loaded config to the callable.
  )

  for machine in MachineConfigMap:
    config = machine.value
    # CHANGE: Modified node_pool.Info initialization to use values from `dag_config`.
    # Previously, this used models.Variable.get() for each parameter.
    node_pool_info = node_pool.Info(
        project_id=dag_config["project_id"],
        cluster_name=dag_config["cluster_name"],
        node_pool_name=dag_config["node_pool_name"],
        location=dag_config["location"],
        node_locations=dag_config["node_locations"],
        num_nodes=dag_config["num_nodes"],
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    problematic_node_pool_info = copy.deepcopy(node_pool_info)
    problematic_node_pool_info.node_pool_name += "-wrong"
    # CHANGE: Modified to use `dag_config["wrong_node_location"]`.
    # Previously, this used models.Variable.get("WRONG_NODE_LOCATION", ...).
    problematic_node_pool_info.node_locations = dag_config["wrong_node_location"]

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      task_id = "create_node_pool"
      create_node_pool = node_pool.create.override(task_id=task_id)(
          node_pool=node_pool_info,
          # CHANGE: Modified to use `dag_config["reservation"]`.
          # Previously, this was a hardcoded string.
          reservation=dag_config["reservation"],
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

      task_id = "cleanup_node_pool"
      cleanup_node_pool = node_pool.delete.override(
          task_id=task_id, trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_node_pool,
      )

      # Intentionally create a node pool with problematic configurations
      # to validate that it enters the ERROR state.
      task_id = "create_problematic_node_pool_info"
      create_problematic_node_pool_info = node_pool.create.override(
          task_id=task_id
      )(
          node_pool=problematic_node_pool_info,
          # The failure is intentionally ignored because we want to validate
          # that the status of the node pool (which fails to be created) is "ERROR".
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

      # CHANGE: Added dependencies to make `print_config_task` run before node pool creations.
      print_config_task >> create_node_pool
      print_config_task >> create_problematic_node_pool_info

      normal_flow = (
          create_node_pool
          >> wait_for_provisioning
          >> wait_for_running
          >> delete_node
          >> wait_for_repair
          >> wait_for_recovered
          >> delete_node_pool
          >> wait_for_stopping
      )

      flow_for_error_state = (
          create_problematic_node_pool_info
          >> wait_for_error
      )
