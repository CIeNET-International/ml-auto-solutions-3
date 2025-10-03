import datetime

from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from airflow.providers.cncf.kubernetes.utils.pod_manager import OnFinishAction
from kubernetes.client import models as k8s

from dags.common.vm_resource import DockerImage
from dags.maxtext_pathways.configs.parameters import PARAMETERS
from dags.maxtext_pathways.utils.tasks import get_parameters


with DAG(
    dag_id='pw_mcjax_benchmark_recipe_dag',
    start_date=datetime.datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        'retries': 0,
    },
    tags=[
        'maxtext',
        'pathways',
        'mcjax',
        'benchmark',
        'nightly',
    ],
    description='A DAG to run a MaxText pw_mcjax_benchmark_recipe on GKE.',
    params=PARAMETERS,
    doc_md="""
  # A DAG to run a MaxText pw_mcjax_benchmark_recipe on GKE.

  ### Description
  Specify different models and number of slices to test the MaxText pw_mcjax_benchmark_recipe on different clusters.
  The DAG first generates recipe command through UI parameters, then runs the workload, waits and monitors the workload logs, and finally cleans up the workload.

  ### Prerequisites
  - This test requires an existing cluster.
  - This test requires that a bucket with the same name as the UI parameter "[User]-[Region]" exists in the UI parameter [Project].
  - Create a service account with the following roles: `Artifact Registry Reader`, `Kubernetes Engine Admin`, `Monitoring Viewer`.
    - Generate a new service account key and download the JSON file to retrieve its contents. Next, create a secret manager named `one-click-key` and store the key contents there for use when switching service accounts.
    - Make sure the default service account has the `Secret Manager Secret Accessor` role.
  - If you're using a service account to pull an image from a different project, you need to grant the service account the `Artifact Registry Reader` role in that project.

  ### Procedures
  An Airflow Composer environment must be created, and the required DAG code must be deployed to the associated GCS bucket.
  To initiate the recipe, the user must access the Airflow UI, locate the specific DAG, and trigger its execution.
  """,
) as dag:
  # Define task dependencies by instantiating and linking tasks.
  params = get_parameters()
  start_recipe = GKEStartPodOperator(
      task_id='pw_mcjax_benchmark_recipe',
      project_id=params['project'],
      cluster_name=params['cluster_name'],
      location=params['region'],
      namespace='default',
      hostnetwork=True,
      image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
      on_finish_action=OnFinishAction.DELETE_POD.value,
      get_logs=True,
      cmds=['/bin/bash', '-cxue', params['commands']],
      container_security_context=k8s.V1SecurityContext(privileged=True),
  )

  # Set the execution order.
  params >> start_recipe
