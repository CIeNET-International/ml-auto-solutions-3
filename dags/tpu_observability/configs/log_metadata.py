from airflow.decorators import task
from xlml.utils import composer


@task(task_id="log_xlml_dashboard_metadata")
def log_metadata(
    cluster_project,
    region,
    zone,
    cluster_name,
    node_pool_name,
    workload_id,
    docker_image,
    accelerator_type,
    num_slices,
):

  composer.log_metadata_for_xlml_dashboard({
      "cluster_project": cluster_project,
      "region": region,
      "zone": zone,
      "cluster_name": cluster_name,
      "node_pool_name": node_pool_name,
      "workload_id": workload_id,
      "docker_image": docker_image,
      "accelerator_type": accelerator_type,
      "num_slices": num_slices,
  })

  return "Metadata Logged"
