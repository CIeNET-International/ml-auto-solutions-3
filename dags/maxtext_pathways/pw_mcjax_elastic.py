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

"""DAG definition for running MaxText Pathways Elastic benchmarks on GKE."""

import datetime

from absl import logging
from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from dags.maxtext_pathways.configs.utils import (
    get_dag_parameters,
    generate_install_dependencies_commands,
    generate_derived_parameters,
    worker_pod_interruption,
    check_pod_status,
    COLOCATED_PYTHON_IMAGE,
)
from xlml.utils import kpo, xpk


ELASTIC_TYPE = ["Pause-resume", "Replica-resize"]
RECIPE_INSTANCE = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
RECIPE_NAME = RECIPE_INSTANCE.value.lower()

elastic_params = ui_params.PARAMETERS.copy()
elastic_params.update({
    "colocated_python_image": ui_params.Param(
        COLOCATED_PYTHON_IMAGE,
        type="string",
        title="Colocated Python Image",
        description="Colocated Python image for pathways.",
    ),
})


@task
def generate_commands(
    dag_params: dict, derived_params: dict, recipe_instance: recipe_cfg.Recipe
) -> str:
  """Generates a command string using config and derived parameters.

  Runtime modifications are made to the recipe command to enable elastic
  training and colocated Python data input.
  """
  env_cmds = generate_install_dependencies_commands()
  recipe_cmd = recipe_instance.run_command
  elastic_type = dag_params.get("elastic_type", ELASTIC_TYPE[0])

  # Patch benchmarks/maxtext_xpk_runner.py
  if elastic_type == "Pause-resume":
    patch_cmd_runner = (
        r'sed -i "/python3 -m maxtext.trainers.pre_train.train/a '
        r"          f\"elastic_enabled=True\",\n"
        r"          f\"enable_single_controller=True\",\n"
        f'          \\"elastic_min_slice_count='
        f'{derived_params["elastic_min_slice_count"]}\\",\\n'
        r"          f\"async_checkpointing=True\",\n"
        r"          f\"enable_checkpoint_cloud_logger=True\",\n"
        r"          f\"checkpoint_period=10\","
        r"          f\"colocated_python_data_input=True\",\n"
        r"          f\"tokenizer_path="
        r'"src/maxtext/assets/tokenizers/tokenizer.llama2"\"," '
        r"benchmarks/maxtext_xpk_runner.py"
    )
  else:  # Replica-resize
    patch_cmd_runner = (
        r'sed -i "/python3 -m maxtext.trainers.pre_train.train/a '
        r"          f\"elastic_enabled=True\",\n"
        r"          f\"enable_single_controller=True\",\n"
        f'          \\"elastic_min_slice_count='
        f'{derived_params["elastic_min_slice_count"]}\\",\\n'
        r"          f\"enable_pathways_goodput=True\",\n"
        r"          f\"enable_goodput_recording=True\",\n"
        r"          f\"goodput_upload_interval_seconds=30\",\n"
        r"          f\"monitor_goodput=True\",\n"
        r"          f\"async_checkpointing=True\",\n"
        r"          f\"enable_checkpoint_cloud_logger=True\",\n"
        r"          f\"checkpoint_period=10\",\n"
        r'" benchmarks/maxtext_xpk_runner.py'
    )

  # 2. Patch benchmarks/maxtext_trillium_model_configs.py
  model_configs_checkpointing = (
      r'sed -i "/model_name=\"default-basic-1\"/,/xla_flags/ { '
      r"s/\"enable_checkpointing\": False/\"enable_checkpointing\": True/; "
      r"/\"profiler\":/d; "
      r'}" benchmarks/maxtext_trillium_model_configs.py'
  )

  cmds_to_run = [env_cmds, patch_cmd_runner, model_configs_checkpointing]

  # Additional patches specifically for Pause-resume
  if elastic_type == "Pause-resume":
    model_configs_grain = (
        r'sed -i "/model_name=\"default-basic-1\"/,/)/ { '
        r"s/\"dataset_type\": \"synthetic\"/\"dataset_type\": \"grain\"/; "
        r"s/\"dataset_path\":/# \"dataset_path\":/; "
        r'}" benchmarks/maxtext_trillium_model_configs.py'
    )
    insert_line = (
        r"            \"grain_train_files\": \"gs://tess-tpu-dataloading-"
        r"us-central1/array-record/c4/en/3.0.1/c4-train.array_record*\","
    )
    model_configs_grain_sub = (
        'sed -i $\'/"dataset_type": "grain"/a \\\n'
        + insert_line
        + "' benchmarks/maxtext_trillium_model_configs.py"
    )
    cmds_to_run.extend([model_configs_grain, model_configs_grain_sub])

  # Combine parameter flags
  all_params = {**dag_params, **derived_params}
  for key, value in all_params.items():
    if key in recipe_cfg.RECIPE_FLAG:
      if isinstance(value, int):
        recipe_cmd += f" --{key}={value}"
      else:
        recipe_cmd += f" --{key}='{value}'"

  # Apply step overrides and proxy flags
  if elastic_type == "Pause-resume":
    recipe_cmd += " --benchmark_steps=1000"
    recipe_cmd += (
        f" --colocated_python_image='{dag_params['colocated_python_image']}'"
    )
    recipe_cmd += (
        f" --proxy_flags='--virtual_slices={derived_params['topology']} "
        f"--num_elastic_slices={derived_params['num_elastic_slices']} "
        " --sidecar_name=external'"
    )
  else:  # Replica-resize
    recipe_cmd += " --benchmark_steps=3000"
    recipe_cmd += (
        f" --proxy_flags='--virtual_slices={derived_params['topology']} "
        f"--num_elastic_slices={derived_params['num_elastic_slices']}'"
    )

  recipe_cmd += " --skip-validation"
  formatted_cmds = recipe_cmd.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")

  cmds_to_run.append(recipe_cmd)
  return " && ".join(cmds_to_run)


def create_elastic_dag(
    dag_id: str,
    elastic_type: str,
    doc_md: str,
    entry_log_pattern: str,
    end_log_pattern: str,
    extra_params: dict = None,
) -> models.DAG:
  """Factory function to build Elastic training DAGs."""

  params = elastic_params.copy()
  params["elastic_type"] = ui_params.Param(
      elastic_type,
      type="string",
      title="Elastic Type",
      description="Pause-resume/Replica-resize",
      enum=ELASTIC_TYPE,
  )
  if extra_params:
    params.update(extra_params)

  schedule = SchedulingHelper.arrange_schedule_time(dag_id)

  dag = models.DAG(
      dag_id=dag_id,
      start_date=datetime.datetime(2025, 1, 1),
      schedule_interval=schedule if composer_env.is_prod_env() else None,
      catchup=False,
      default_args={"retries": 0},
      tags=[
          "maxtext",
          "pathways",
          "mcjax",
          "benchmark",
          "nightly",
          "TPU",
          "v6e",
      ],
      description=f"A DAG to run a MaxText {RECIPE_NAME} with elastic training on GKE.",
      params=params,
      doc_md=doc_md,
  )

  with dag:
    fetched_params = get_dag_parameters()
    calculated_params = generate_derived_parameters(fetched_params, dag_id)
    generated_cmds = generate_commands(
        fetched_params, calculated_params, RECIPE_INSTANCE
    )

    start_recipe = kpo.run_command_in_kpo(
        start_cli_command=generated_cmds,
        workload_id="start_recipe",
        task_owner=test_owner.DORA_H,
        provisioning_timeout=datetime.timedelta(minutes=5),
        workload_run_timeout=datetime.timedelta(minutes=15),
        image_full_url=fetched_params["runner"],
    )

    check_pod = check_pod_status.override(
        task_id="check_pod_status",
        timeout=180,
    )(
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
        workload_id=calculated_params["workload_id"],
    )

    # TODO(cienet): Add comments or documentation to explain expected log patterns.
    interruption_task = worker_pod_interruption(
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
        workload_id=calculated_params["workload_id"],
        entry_log_pattern=entry_log_pattern,
        elastic_log_pattern="Elastic attempt",
        end_log_pattern=end_log_pattern,
    )

    wait_for_workload_complete = xpk.wait_for_workload_completion.override(
        task_id="wait_for_workload_complete",
        timeout=3600,
        trigger_rule=TriggerRule.ALL_DONE,
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
    )

    clean_up_recipe = xpk.clean_up_workload.override(
        task_id="clean_up_recipe", trigger_rule=TriggerRule.ALL_DONE
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        zone=fetched_params["zone"],
        cluster_name=fetched_params["cluster_name"],
    )

    chain(
        fetched_params,
        calculated_params,
        generated_cmds,
        start_recipe,
        check_pod,
        interruption_task,
        wait_for_workload_complete,
        clean_up_recipe,
    )

  return dag


# Instantiate Pause-resume DAG
PAUSE_RESUME_DOC = f"""
# A DAG to run a MaxText {RECIPE_NAME} with elastic training on GKE.

### Description
Pause-resume refers to the process of halting the training execution,
saving its state (typically to a checkpoint), and later restarting
the training, loading the state from the checkpoint to continue.
Stop the training process when slices become unavailable, and starts it
again later on the new set inherently. This mechanism is crucial for
fault tolerance and elasticity. Resuming can occur on the same
set of resources or a different set.

### Prerequisites
- This test requires an existing cluster.
- If you're using a service account to pull an image from a different
  project, you need to grant the service account the
  `Artifact Registry Reader` role in that project.

### Procedures
An Airflow Composer environment must be created, and the required DAG code
must be deployed to the associated GCS bucket. To initiate the recipe, the
user must access the Airflow UI, locate the specific DAG, and trigger it.

### Model Configuration
If you want to add other TPU type models, you need to manually modify
`/ml-auto-solutions/dags/maxtext_pathways/configs/model_configs.py`.
"""

pause_resume_dag = create_elastic_dag(
    dag_id="pw_elastic_pause_resume",
    elastic_type=ELASTIC_TYPE[0],
    doc_md=PAUSE_RESUME_DOC,
    entry_log_pattern="completed step:",
    end_log_pattern="Sufficient slices active: 1 >= 1",
)


# Instantiate Replica-resize DAG
REPLICA_RESIZE_DOC = f"""
# A DAG to run a MaxText {RECIPE_NAME} with elastic replica resize on GKE.

### Description
Replica-resize refers to the ability of the training job to dynamically
adjust the number of active TPU slices (replicas) it uses during execution.
Expected Behavior:
- A change in slice availability (failure or addition)
triggers an event. Often, a slice failure results in an error.
- The elastic training framework detects this change.
- Training on the previous configuration halts, and try to identify
the new set of healthy, available slice.
- The training job is automatically relaunched, loading the model
state from the most recent checkpoint. The relaunched job now runs on
the new set of available slices.

### Prerequisites
- This test requires an existing cluster.
- If you're using a service account to pull an image from a different
  project, you need to grant the service account the
  `Artifact Registry Reader` role in that project.

### Procedures
An Airflow Composer environment must be created, and the required DAG code
must be deployed to the associated GCS bucket. To initiate the recipe, the
user must access the Airflow UI, locate the specific DAG, and trigger it.

### Model Configuration
If you want to add other TPU type models, you need to manually modify
`/ml-auto-solutions/dags/maxtext_pathways/configs/model_configs.py`.
"""

replica_resize_dag = create_elastic_dag(
    dag_id="pw_elastic_replica_resize",
    elastic_type=ELASTIC_TYPE[1],
    doc_md=REPLICA_RESIZE_DOC,
    entry_log_pattern="live slice count: 2",
    end_log_pattern="Sufficient slices active: 2 >= 1",
    extra_params={
        "num_slices_list": ui_params.Param(
            2,
            type="integer",
            title="Number Slices",
            description="Number of slices",
        )
    },
)
