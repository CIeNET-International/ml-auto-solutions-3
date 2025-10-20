# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from kubernetes.client import models as k8s

from dags.common.vm_resource import DockerImage
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from dags.maxtext_pathways.utils import tasks
from xlml.utils.xpk import wait_for_workload_completion, clean_up_workload


RECIPE_NAME = recipe_cfg.RecipeConfigs.PW_MCJAX_BENCHMARK_RECIPE.value
RECIPE = RECIPE_NAME.lower()

with DAG(
    dag_id=RECIPE,
    start_date=datetime.datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        "retries": 0,
    },
    tags=[
        "maxtext",
        "pathways",
        "mcjax",
        "benchmark",
        "nightly",
    ],
    description=f"A DAG to run a MaxText {RECIPE} on GKE.",
    params=ui_params.PARAMETERS,
    doc_md=f"""
  # A DAG to run a MaxText {RECIPE} on GKE.

  ### Description
  Specify different models and number of slices to test the MaxText {RECIPE} on different clusters.  
  The DAG first generates recipe command through UI parameters, then runs the workload, waits and monitors the workload logs, and finally cleans up the workload.

  ### Prerequisites
  - This test requires an existing cluster.
  - This test requires that a dataset with the same name as the UI parameter "[BigQuery Database Dataset]".
  - Create a service account named `one-click` with the following roles: `Artifact Registry Reader`, `Kubernetes Engine Admin`, `Monitoring Viewer`.
    - Generate a new service account key and download the JSON file to retrieve its contents. 
      Next, create a secret manager named `one-click-key` and store the key contents there for use when switching service accounts.
    - Make sure the default service account has the `Secret Manager Secret Accessor` role.  
      ex: [PROJECT_NUMBER]-compute@developer.gserviceaccount.com
  - If you're using a service account to pull an image from a different project, you need to grant the service account the `Artifact Registry Reader` role in that project.

  ### Procedures
  An Airflow Composer environment must be created, and the required DAG code must be deployed to the associated GCS bucket.  
  To initiate the recipe, the user must access the Airflow UI, locate the specific DAG, and trigger its execution.  

  ### Model Configuration
  If you want to add other TPU type models, you need to manually modify `/ml-auto-solutions/dags/maxtext_pathways/configs/model_configs.py`.
  """,
) as dag:
  recipe_runtime = (
      RECIPE.replace("_", "-") + '-{{ execution_date.strftime("%H%M%S") }}'
  )

  # Define task dependencies by instantiating and linking tasks.
  dag_params = tasks.get_dag_parameters()
  derived_params = tasks.generate_derived_parameters(dag_params)
  commands = tasks.generate_commands(dag_params, derived_params, RECIPE_NAME)

  start_recipe = GKEStartPodOperator(
      task_id="start_recipe",
      name=RECIPE.replace("_", "-"),
      project_id=dag_params["project"],
      cluster_name=dag_params["cluster_name"],
      location=derived_params["region"],
      namespace="default",
      hostnetwork=True,
      image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
      # TODO(b/452777428): Unable to delete the pod, may need to update the Airflow version.
      # on_finish_action=OnFinishAction.DELETE_POD.value,
      get_logs=True,
      cmds=["/bin/bash", "-cxue", commands],
      container_security_context=k8s.V1SecurityContext(privileged=True),
      labels={"airflow-runtime": recipe_runtime},
  )

  clean_up_pod = tasks.clean_up_pod(
      cluster_name=dag_params["cluster_name"],
      region=derived_params["region"],
      project=dag_params["project"],
      airflow_runtime=recipe_runtime,
  ).as_teardown(setups=[commands])

  check_recipe_log = wait_for_workload_completion.override(
      task_id="check_recipe_log",
      poke_interval=30,
      on_execute_callback=[tasks.set_sensor_timeout],
  )(
      workload_id=derived_params["recipe_workload_id"],
      project_id=dag_params["project"],
      region=derived_params["region"],
      cluster_name=dag_params["cluster_name"],
  )

  clean_up_recipe = clean_up_workload.override(task_id="clean_up_recipe")(
      workload_id=derived_params["recipe_workload_id"],
      project_id=dag_params["project"],
      zone=dag_params["zone"],
      cluster_name=dag_params["cluster_name"],
  ).as_teardown(setups=[start_recipe])

  # Set the execution order.
  (
      dag_params
      >> derived_params
      >> commands
      >> start_recipe
      >> check_recipe_log
      >> clean_up_recipe
  )
  start_recipe >> clean_up_pod
