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

import random
import string
import time
from absl import logging
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook

from dags.maxtext_pathways.configs import commands as cmds
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from xlml.utils.gke import zone_to_region


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
  derived_params["region"] = zone_to_region(dag_params["zone"])

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
  env_cmds = " && ".join(cmds.ENV_COMMAND)
  recipe_cmds = " && ".join(cmds.RUN_RECIPE)

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

  recipe_cmds = recipe_cmds.format(recipe_name=recipe_name)
  env_cmds = env_cmds.format(service_account=dag_params["service_account"])

  formatted_cmds = recipe_cmds.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")

  commands = " && ".join([env_cmds, recipe_cmds])

  return commands


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
