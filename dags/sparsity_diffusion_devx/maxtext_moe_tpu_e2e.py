# Copyright 2024 Google LLC
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
# limitations under the License."""A DAG to run end-to-end MoE tests."""

"""A DAG to run end-to-end moe tests."""

import datetime
from datetime import timedelta
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config
from xlml.utils import name_format


# Run once a day at 1 am UTC (5 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None
HF_TOKEN = models.Variable.get("HF_TOKEN", None)

DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = timedelta(minutes=5)


with models.DAG(
    dag_id="maxtext_moe_tpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "jax_models_and_performance",
        "multipod_team",
        "maxtext",
        "tpu",
        "stable",
        "nightly",
        "mlscale_devx",
        "v6e-256",
        "v5p-8",
    ],
    start_date=datetime.datetime(2024, 11, 14),
    catchup=False,
    default_args={
        "retries": DEFAULT_RETRIES,
        "retry_delay": DEFAULT_RETRY_DELAY,
    },
) as dag:
  TEST_NAME_PREFIX = "maxtext"
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )
  docker_image = {
      "stable": DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK.value,
      "nightly": DockerImage.MAXTEXT_TPU_STABLE_STACK_NIGHTLY_JAX.value,
  }

  test_models_tpu = {
      "mixtral-8x22b": {
          "script_name": "tpu/mixtral/8x22b/2_test_mixtral",
          "cluster": XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
          "time_out_in_min": 60,
      },
      "gpt-oss-20b": {
          "script_name": "tpu/gpt_oss/20b/test_gpt_oss",
          "cluster": XpkClusters.TPU_V5P_8_CLUSTER,
          "time_out_in_min": 90,
      },
  }
  unchained_tests = []
  for model, test_scripts_details in test_models_tpu.items():
    for image_name, image_value in docker_image.items():
      if model.startswith("gpt-oss") and image_name == "stable":
        continue
      training_tpu = gke_config.get_gke_config(
          time_out_in_min=test_scripts_details["time_out_in_min"],
          test_name=f"{TEST_NAME_PREFIX}_{image_name}_{model}",
          run_model_cmds=(
              f"export HF_TOKEN={HF_TOKEN}; "
              "export BASE_OUTPUT_PATH=$GCS_OUTPUT; "
              "bash end_to_end/{test_scripts_details['script_name']}.sh",
          ),
          docker_image=image_value,
          test_owner=test_owner.SHUNING_J,
          cluster=test_scripts_details["cluster"],
      ).run_with_quarantine(quarantine_task_group)
      unchained_tests.append(training_tpu)

  for i in range(len(unchained_tests) - 1):
    downstream_task = unchained_tests[i + 1]
    downstream_task.trigger_rule = "all_done"
    _ = unchained_tests[i] >> downstream_task

  multicluster_test_models = {
      "mixtral-8x7b": [
          {
              "script_name": "tpu/mixtral/8x7b/1_test_mixtral",
              "cluster": XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER,
              "time_out_in_min": 240,
          },
          {
              "script_name": "tpu/mixtral/8x7b/2_test_mixtral",
              "cluster": XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
              "time_out_in_min": 90,
          },
      ],
      "llama4": [
          {
              "script_name": "tpu/llama4/1_test_llama4",
              "cluster": XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER,
              "time_out_in_min": 240,
          },
          {
              "script_name": "tpu/llama4/2_test_llama4",
              "cluster": XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
              "time_out_in_min": 90,
          },
      ],
  }

  def convert_checkpoint_and_run_training(
      run_config, scripts_config, upstream_task=None
  ):
    """Orchestrates the CPU conversion and TPU training tasks for a given model.

    Creates a shared GCS location, configures the CPU conversion task and the
    subsequent TPU training task, and enforces sequential execution if an
    upstream task is provided.
    """

    cpu_details = scripts_config[0]
    tpu_details = scripts_config[1]

    group_id = run_config["group_id"]
    test_name = run_config["test_name"]
    image_uri = run_config["image_uri"]

    test_id_suffix = group_id.replace("chained_tests_", "")
    gcs_location_task_id = f"generate_gcs_location_{test_id_suffix}"

    trigger_rule = "all_done" if upstream_task is not None else "all_success"

    shared_gcs_location = name_format.generate_gcs_folder_location.override(
        task_id=gcs_location_task_id, trigger_rule=trigger_rule
    )(
        GCS_SUBFOLDER,
        group_id,
    )

    if upstream_task is not None:
      _ = upstream_task >> shared_gcs_location

    conversion_cpu = gke_config.get_maxtext_cpu_end_to_end_gke_config(
        time_out_in_min=cpu_details["time_out_in_min"],
        test_name=test_name,
        run_model_cmds=(
            "export BASE_OUTPUT_PATH=$GCS_OUTPUT; "
            f"bash end_to_end/{cpu_details['script_name']}.sh",
        ),
        docker_image=image_uri,
        test_owner=test_owner.SHUNING_J,
        cluster=cpu_details["cluster"],
    ).run(gcs_location=shared_gcs_location)

    _ = shared_gcs_location >> conversion_cpu

    tpu_train_task = gke_config.get_gke_config(
        time_out_in_min=tpu_details["time_out_in_min"],
        test_name=test_name,
        run_model_cmds=(
            "export BASE_OUTPUT_PATH=$GCS_OUTPUT; "
            f"bash end_to_end/{tpu_details['script_name']}.sh",
        ),
        docker_image=image_uri,
        test_owner=test_owner.SHUNING_J,
        cluster=tpu_details["cluster"],
    ).run(gcs_location=shared_gcs_location)

    _ = shared_gcs_location >> training_tpu
    _ = conversion_cpu >> tpu_train_task

    return tpu_train_task

  last_task = None
  GCS_SUBFOLDER = f"{test_owner.Team.JAX_MODELS_AND_PERFORMANCE.value}/maxtext"


  for model, test_scripts_details in multicluster_test_models.items():
    for image_key, image_value in docker_image.items():
      current_group_id = f"chained_tests_{model}_{image_key}"
      current_test_name = f"{TEST_NAME_PREFIX}_{image_key}_{model}"

      config = {
          "group_id": current_group_id,
          "test_name": current_test_name,
          "image_uri": image_value,
      }

      mode_tpu = convert_checkpoint_and_run_training(
          run_config=config,
          scripts_config=test_scripts_details,
          upstream_task=last_task,
      )

      last_task = mode_tpu
