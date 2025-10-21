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
import random
import string
import time
from absl import logging
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from kubernetes.client import models as k8s

from dags.common.vm_resource import DockerImage
from dags.maxtext_pathways.configs import commands as cmds
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from xlml.utils import xpk, gke


@task.python(multiple_outputs=True)
def get_dag_parameters(**context) -> dict:
  """
  Fetches and returns the DAG run's configuration parameters.
  """
  dag_params = context.get("params", {})

  return dag_params


@task.python(multiple_outputs=True)
def generate_derived_parameters(dag_params: dict) -> dict:
  """
  Generates new parameters based on the initial DAG parameters.
  """
  derived_params = {}

  # Generate recipe workload_id and temp_key.
  name, temp_post_fix = generate_recipe_workload_id(dag_params)
  derived_params["temp_key"] = temp_post_fix
  derived_params["recipe_workload_id"] = name

  # Generate region by zone
  derived_params["region"] = gke.zone_to_region(dag_params["zone"])

  return derived_params


@task
def generate_commands(
    dag_params: dict, derived_params: dict, recipe_name: str
) -> str:
  """
  Generates a command string using the initial DAG parameters and derived parameters.
  """
  dag_params = dag_params.copy()

  # Initialization command.
  env_cmds_list = generate_install_dependencies_commands()

  env_cmds = " && ".join(env_cmds_list)
  recipe_cmds = " && ".join(cmds.RUN_RECIPE)

  env_cmds = env_cmds.format(service_account = dag_params["service_account"])
  recipe_cmds = recipe_cmds.format(recipe_name=recipe_name)

  # Generate device_type.
  device_type = (
      dag_params["device_version"] + "-" + str(dag_params["core_count"])
  )
  dag_params["device_type"] = device_type

  # Confirm whether to use customized_model_name.
  if dag_params["selected_model_names"] == "customized_model_name":
    dag_params["selected_model_names"] = dag_params["customized_model_name"]

  # Combine command.
  all_params = {**dag_params, **derived_params}
  for key, value in all_params.items():
    if key in recipe_cfg.RECIPE_FLAG:
      if isinstance(value, int):
        recipe_cmds += f" --{key}={value}"
      else:
        recipe_cmds += f" --{key}='{value}'"

  formatted_cmds = recipe_cmds.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")

  commands = " && ".join([env_cmds, recipe_cmds])

  return commands


def generate_install_dependencies_commands() -> list[str]:
  """
  Generate the list of shell commands to install necessary dependencies in the Pod.
  """
  env_cmds_list = (
      cmds.UPDATE_APT
      + cmds.INSTALL_MAKE
      + cmds.INSTALL_KUBECTL
      + cmds.INSTALL_DOCKER
      + cmds.INSTALL_GCLOUD
      + cmds.SWITCH_SERVICE_ACCOUNT
      + cmds.INSTALL_KUBECTL_KJOB
      + cmds.INSTALL_KUBECTL_KUEUE
      + cmds.INSTALL_XPK
      + cmds.BACK_MAXTEXT
  )

  return env_cmds_list


def generate_recipe_workload_id(params: dict) -> tuple[str, str]:
  """
  Generate a random value in advance to fix the workload_id so that the workload can be deleted later.
  Please refer to the `generate_xpk_workload_cmd` function in the `/maxtext/benchmarks/maxtext_xpk_runner.py` file.
  """
  time.localtime()
  length_of_random_str = 3
  temp_post_fix = "".join(
      random.choice(string.ascii_lowercase + string.digits)
      for _ in range(length_of_random_str)
  )

  truncate_model_name = 10
  truncate_prefix = 3
  post_fix = f'-{params["num_slices_list"]}-{time.strftime("%m%d%H", time.localtime())}-{temp_post_fix}'
  common_prefix = params["user"]

  pw_prefix = "pw-"

  if params["selected_model_framework"] == "pathways":
    post_fix = f'-{params["num_slices_list"]}-{temp_post_fix}'
    name = f'{pw_prefix}{params["selected_model_names"].replace("_", "-")[:truncate_model_name - len(pw_prefix)]}'
  else:
    name = f'{params["selected_model_names"].replace("_", "-")[:truncate_model_name]}'

  name = f"{common_prefix[:truncate_prefix]}-{name}{post_fix}"

  return name, temp_post_fix


@task
def clean_up_pod(
    cluster_name: str, region: str, project: str, airflow_runtime: str
) -> None:
  """
  Use SubprocessHook to execute shell commands to delete Pods.
  """
  hook = SubprocessHook()

  commands = [
      "set -xue",
      "export KUBECONFIG=/tmp/kubeconfig",  # Change KUBECONFIG from /home/airflow to /tmp to avoid permission issue.
      f"gcloud container clusters get-credentials {cluster_name} --region={region} --project={project}",
      f"kubectl delete pod -l airflow-runtime={airflow_runtime} --namespace=default --force --grace-period=0",
  ]

  result = hook.run_command(
      ["bash", "-c", ";".join(commands)],
  )

  assert (
      result.exit_code == 0
  ), f"kubectl clean-up failed with code {result.exit_code}"


def set_sensor_timeout(context: dict) -> None:
  """
  Dynamically sets the Airflow task timeout based on a custom flag or calculates it using a benchmark step count.
  """
  if context["params"]["timeout_enable"]:
    context["ti"].task.timeout = context["params"]["timeout_in_min"] * 60
  else:
    max_step_min = 5  # Average time required for each step in lama3-1-405b.
    context["ti"].task.timeout = (
        context["params"]["benchmark_steps"] * max_step_min * 60
    )


RECIPE_NAME = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE.value
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
  dag_params = get_dag_parameters()
  derived_params = generate_derived_parameters(dag_params)
  commands = generate_commands(dag_params, derived_params, RECIPE_NAME)

  start_recipe = GKEStartPodOperator(
      task_id="start_recipe",
      name=RECIPE.replace("_", "-"),
      project_id=dag_params["project"],
      cluster_name=dag_params["cluster_name"],
      location=derived_params["region"],
      namespace="default",
      hostnetwork=True,
      image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
      # TODO(b/452777428): Unable to delete the pod, may need to upgrade Airflow providers to "apache-airflow-providers-google==16.0.0".
      # on_finish_action=OnFinishAction.DELETE_POD.value,
      get_logs=True,
      cmds=["/bin/bash", "-cxue", commands],
      container_security_context=k8s.V1SecurityContext(privileged=True),
      labels={"airflow-runtime": recipe_runtime},
  )

  clean_up_pod = clean_up_pod(
      cluster_name=dag_params["cluster_name"],
      region=derived_params["region"],
      project=dag_params["project"],
      airflow_runtime=recipe_runtime,
  ).as_teardown(setups=[commands])
  # TODO(b/453860040): If start_recipe fails, clean_up_pod is not executed, even with teardown set. 
  # Currently relying on the task output (command) before start_recipe, may need to update the Airflow version.

  check_recipe_log = xpk.wait_for_workload_completion.override(
      task_id="check_recipe_log",
      poke_interval=30,
      on_execute_callback=[set_sensor_timeout],
  )(
      workload_id=derived_params["recipe_workload_id"],
      project_id=dag_params["project"],
      region=derived_params["region"],
      cluster_name=dag_params["cluster_name"],
  )

  clean_up_recipe = xpk.clean_up_workload.override(task_id="clean_up_recipe")(
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
