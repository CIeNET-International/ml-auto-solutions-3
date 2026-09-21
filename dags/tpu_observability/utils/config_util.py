import copy
from airflow.decorators import task
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool


@task
def prepare_tpu_configs(
    dag_id: str,
    machine_config,
    selector: str,
    gcs_config_path: str,
    gcs_jobset_config_path: str,
    is_prod: bool,
) -> dict:
  """Prepares and resolves configuration objects for TPU node pools and JobSets.

  This task runs at execution time to intercept the `selector` XComArg and
  resolve it into a plain string. It then passes the resolved selector to
  underlying utility functions to construct clean configuration objects
  without any unresolved Airflow expressions. This prevents downstream tasks
  from encountering initialization errors due to unrendered Jinja templates
  or raw XComArg objects wrapped inside custom classes.

  Args:
    dag_id: The unique identifier of the DAG invoking this function.
    machine_config: The machine configuration object containing TPU version,
      topology, and machine type details.
    selector: The node pool selector string. If passed as an XComArg, Airflow
      will automatically resolve it to its actual value prior to execution.
    gcs_config_path: The GCS URI path to the node pool configuration YAML.
    gcs_jobset_config_path: The GCS URI path to the JobSet configuration YAML.
    is_prod: A boolean flag indicating whether the current environment is
      production.

  Returns:
    A dictionary containing the following resolved configuration objects:
      - "jobset_config": The fully initialized JobSet configuration.
      - "cluster_info": The configuration info for the first TPU node pool.
      - "cluster_info_2": A deep copy of the first node pool info, configured
        with a modified name suffix for the second node pool.
  """

  jobset_config = jobset.build_jobset_from_gcs_yaml(
      gcs_path=gcs_jobset_config_path,
      dag_name=dag_id,
      node_pool_selector=selector,
  )

  cluster_info = node_pool.build_node_pool_info_from_gcs_yaml(
      gcs_path=gcs_config_path,
      dag_name=dag_id,
      is_prod=is_prod,
      machine_type=machine_config.machine_version.value,
      tpu_topology=machine_config.tpu_topology,
      node_pool_selector=selector,
  )

  cluster_info_2 = copy.deepcopy(cluster_info)
  cluster_info_2.node_pool_name = f"{cluster_info.node_pool_name}-2"

  return {
      "jobset_config": jobset_config,
      "cluster_info": cluster_info,
      "cluster_info_2": cluster_info_2,
  }
