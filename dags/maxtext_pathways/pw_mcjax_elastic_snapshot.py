# Copyright 2026 Google LLC
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

"""DAG definitions for MaxText Pathways Elastic In-Memory Snapshot E2E tests."""

import datetime

from absl import logging
from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain

from dags import composer_env
from dags.common import test_owner
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from dags.maxtext_pathways.configs.snapshot_utils import (
    snapshot_fault_injection,
    verify_snapshot_logs,
)
from dags.maxtext_pathways.configs.utils import (
    COLOCATED_PYTHON_IMAGE,
    generate_derived_parameters,
    generate_install_dependencies_commands,
    get_dag_parameters,
)
from xlml.utils import gcluster, gke, kpo

C4_ARRAY_RECORD_GLOB = (
    "gs://tess-tpu-dataloading-us-central1/array-record/"
    "c4/en/3.0.1/c4-train.array_record*"
)
LLAMA2_TOKENIZER_PATH = "src/maxtext/assets/tokenizers/tokenizer.llama2"
SHARED_CAPACITY_PROJECT = "cloud-tpu-shared-capacity"
SHARED_CAPACITY_OUTPUT_DIR = "gs://ml-auto-solutions/pw-snapshot/"

_DEVICE_MODEL_CONFIG_TARGETS = {
    "v6e": (
        "benchmarks/maxtext_trillium_model_configs.py",
        "trillium_model_dict",
    ),
    "v5p": ("benchmarks/maxtext_v5p_model_configs.py", "v5p_model_dict"),
    "v5litepod": ("benchmarks/maxtext_v5e_model_configs.py", "v5e_model_dict"),
    "v5e": ("benchmarks/maxtext_v5e_model_configs.py", "v5e_model_dict"),
}


def build_snapshot_params(
    *,
    elastic_type: str,
    num_slices: int,
    snapshot_interval: int,
    checkpoint_period: int,
    benchmark_steps: int,
    enable_checkpointing: bool = False,
    async_checkpointing: bool = False,
) -> dict:
  """Builds Airflow UI parameters for an elastic snapshot DAG."""
  return {
      **ui_params.PARAMETERS,
      "elastic_type": ui_params.Param(
          elastic_type,
          type="string",
          enum=["Pause-resume", "Replica-resize"],
      ),
      "num_slices_list": ui_params.Param(num_slices, type="integer"),
      "elastic_snapshot_interval": ui_params.Param(
          snapshot_interval, type="integer"
      ),
      "checkpoint_period": ui_params.Param(checkpoint_period, type="integer"),
      "benchmark_steps": ui_params.Param(benchmark_steps, type="integer"),
      "enable_checkpointing": ui_params.Param(
          enable_checkpointing, type="boolean"
      ),
      "async_checkpointing": ui_params.Param(
          async_checkpointing, type="boolean"
      ),
      "use_colocated_python": ui_params.Param(False, type="boolean"),
      "colocated_python_image": ui_params.Param(
          COLOCATED_PYTHON_IMAGE, type="string"
      ),
      "base_output_directory": ui_params.Param(
          None,
          type=["null", "string"],
          title="Base Output Directory",
          description=(
              "Optional GCS output prefix (e.g."
              " gs://ml-auto-solutions/pw-snapshot/). Automatically defaults"
              " to gs://ml-auto-solutions/pw-snapshot/ when project is"
              " cloud-tpu-shared-capacity."
          ),
      ),
  }


@task
def generate_commands(
    dag_params: dict,
    derived_params: dict,
    recipe_instance: recipe_cfg.Recipe,
) -> str:
  """Builds shell commands to patch and launch an elastic snapshot workload."""
  env_cmds = generate_install_dependencies_commands(
      cluster_name=dag_params["cluster_name"],
      location=derived_params["region"],
      project_id=dag_params["project"],
  )
  recipe_cmd = recipe_instance.run_command

  snapshot_interval = int(dag_params["elastic_snapshot_interval"])
  checkpoint_period = int(dag_params["checkpoint_period"])
  benchmark_steps = int(dag_params["benchmark_steps"])
  enable_ckpt = bool(dag_params.get("enable_checkpointing", False))
  async_ckpt = bool(dag_params.get("async_checkpointing", False))
  use_colocated = bool(dag_params.get("use_colocated_python", False))
  selected_model = derived_params.get(
      "selected_model_names", dag_params["selected_model_names"]
  )
  colocated_line = (
      r"          f\"colocated_python_data_input=True\",\n"
      if use_colocated
      else ""
  )
  ckpt_sed_rule = (
      's/"enable_checkpointing": False/"enable_checkpointing": True/; '
      if enable_ckpt
      else ""
  )

  patch_cmd_runner = (
      r'sed -i "/python3 -m maxtext.trainers.pre_train.train/a '
      r"          f\"elastic_enabled=True\",\n"
      r"          f\"elastic_backup_kind=snapshot\",\n"
      f'          \\"elastic_snapshot_interval={snapshot_interval}\\",\\n'
      r"          f\"enable_single_controller=True\",\n"
      r"          f\"grain_use_elastic_iterator=True\",\n"
      r"          f\"packing=False\",\n"
      f'          \\"elastic_min_slice_count='
      f'{derived_params["elastic_min_slice_count"]}\\",\\n'
      r"          f\"elastic_timeout_seconds=900\",\n"
      f'          f\\"async_checkpointing={async_ckpt}\\",\\n'
      r"          f\"save_checkpoint_on_completion=False\",\n"
      f'          \\"checkpoint_period={checkpoint_period}\\",\\n'
      f"{colocated_line}"
      r"          f\"tokenizer_path="
      f'"{LLAMA2_TOKENIZER_PATH}"'
      r"\","
      r'" benchmarks/maxtext_ctk_runner.py'
  )
  device_version = dag_params.get("device_version", "v6e")
  is_multihost_v6e_8 = (
      device_version == "v6e" and int(dag_params.get("core_count", 16)) == 8
  )
  patch_cmds = [patch_cmd_runner]
  if is_multihost_v6e_8:
    patch_cmds.append(
        "sed -i 's/size in (1, 8)/size == 1/' benchmarks/maxtext_ctk_runner.py"
    )
  target_cfg_file, target_dict_name = _DEVICE_MODEL_CONFIG_TARGETS.get(
      device_version, _DEVICE_MODEL_CONFIG_TARGETS["v6e"]
  )
  sed_target_file = (
      "benchmarks/maxtext_trillium_model_configs.py"
      if selected_model == "default_basic_1"
      else target_cfg_file
  )
  patch_cmd_model_configs = (
      f"sed -i '/^{selected_model} = _add_to_model_dictionary/,/xla_flags/ {{ "
      's|"dataset_type": "synthetic"|"dataset_type": "grain",'
      f' "grain_train_files": "{C4_ARRAY_RECORD_GLOB}"|; '
      's/"dataset_path":/# "dataset_path":/; '
      's/"reuse_example_batch": 1/"reuse_example_batch": 0/; '
      f"{ckpt_sed_rule}"
      '/"profiler":/d; '
      f"}}' {sed_target_file}"
  )
  patch_cmds.append(patch_cmd_model_configs)
  if (
      target_cfg_file != "benchmarks/maxtext_trillium_model_configs.py"
      and selected_model == "default_basic_1"
  ):
    patch_cmds.append(
        "sed -i '$a\\from benchmarks.maxtext_trillium_model_configs import"
        " default_basic_1\\n"
        f'{target_dict_name}["default_basic_1"] = default_basic_1\' '
        f"{target_cfg_file}"
    )

  for key, value in {**dag_params, **derived_params}.items():
    if key in recipe_cfg.RECIPE_FLAG:
      flag_val = (
          value
          if isinstance(value, int) and not isinstance(value, bool)
          else f"'{value}'"
      )
      recipe_cmd += f" --{key}={flag_val}"

  base_output_dir = dag_params.get("base_output_directory")
  if (
      not base_output_dir
      and dag_params.get("project") == SHARED_CAPACITY_PROJECT
  ):
    base_output_dir = SHARED_CAPACITY_OUTPUT_DIR
  if base_output_dir:
    recipe_cmd += f" --base_output_directory='{base_output_dir}'"

  num_elastic_slices = int(derived_params["num_elastic_slices"])
  virtual_slices = (
      ",".join(["tpuv6e:2x4"] * num_elastic_slices)
      if is_multihost_v6e_8
      else derived_params["topology"]
  )
  proxy_flags = (
      f"--virtual_slices={virtual_slices} "
      f"--num_elastic_slices={num_elastic_slices}"
  )
  if use_colocated:
    recipe_cmd += (
        f" --colocated_python_image='{dag_params['colocated_python_image']}'"
    )
    proxy_flags += " --sidecar_name=external"

  recipe_cmd += (
      f" --benchmark_steps={benchmark_steps}"
      f" --proxy_flags='{proxy_flags}' --skip-validation"
  )
  logging.info(
      "Generated snapshot recipe command:\n %s",
      recipe_cmd.replace(" --", " \n  --"),
  )
  return " && ".join([
      env_cmds,
      *patch_cmds,
      recipe_cmd,
  ])


def create_elastic_snapshot_dag(
    *,
    dag_id: str,
    workload_prefix: str,
    description: str,
    doc_md: str,
    post_interrupt_checks: tuple[tuple[str, str], ...],
    elastic_type: str = "Replica-resize",
    num_slices: int = 2,
    snapshot_interval: int = 5,
    checkpoint_period: int = 1000,
    benchmark_steps: int = 500,
    enable_checkpointing: bool = False,
    async_checkpointing: bool = False,
    interrupt_after_snapshot_step: int = 5,
    post_snapshot_steps: int = 3,
    expect_fallback: bool = False,
    expect_scale_up: bool = False,
    expected_min_restores: int = 1,
    expect_data_iterator: bool = True,
    recipe_instance: recipe_cfg.Recipe = (
        recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
    ),
) -> models.DAG:
  """Factory function to build an elastic in-memory snapshot E2E DAG."""
  schedule = SchedulingHelper.arrange_schedule_time(dag_id)
  params = build_snapshot_params(
      elastic_type=elastic_type,
      num_slices=num_slices,
      snapshot_interval=snapshot_interval,
      checkpoint_period=checkpoint_period,
      benchmark_steps=benchmark_steps,
      enable_checkpointing=enable_checkpointing,
      async_checkpointing=async_checkpointing,
  )
  with models.DAG(
      dag_id=dag_id,
      start_date=datetime.datetime(2026, 1, 1),
      schedule_interval=schedule if composer_env.is_prod_env() else None,
      catchup=False,
      default_args={"retries": 0},
      tags=[
          "maxtext",
          "pathways",
          "mcjax",
          "elastic",
          "snapshot",
          "nightly",
          "TPU",
          "v6e",
      ],
      description=description,
      params=params,
      doc_md=doc_md,
  ) as dag:
    fetched_params = get_dag_parameters()
    calculated_params = generate_derived_parameters(
        fetched_params, workload_prefix
    )
    generated_cmds = generate_commands(
        fetched_params, calculated_params, recipe_instance
    )
    start_recipe = kpo.run_command_in_kpo(
        start_cli_command=generated_cmds,
        workload_id="start_recipe",
        task_owner=test_owner.KEVIN_Y,
        provisioning_timeout=datetime.timedelta(minutes=10),
        workload_run_timeout=datetime.timedelta(minutes=25),
        image_full_url=fetched_params["runner"],
    )
    setattr(
        dag.get_task("run_command_in_kpo.run_kpo-cli"),
        "startup_timeout_seconds",
        600,
    )
    interruption_task = snapshot_fault_injection(
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
        workload_id=calculated_params["workload_id"],
        required_live_slices=calculated_params["num_elastic_slices"],
        interrupt_after_snapshot_step=interrupt_after_snapshot_step,
        post_snapshot_steps=post_snapshot_steps,
        post_interrupt_checks=post_interrupt_checks,
    )
    wait_for_workload_complete = gke.wait_for_workload_completion.override(
        task_id="wait_for_workload_complete",
        timeout=3600,
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
    )
    verify_logs = verify_snapshot_logs.override(task_id="verify_snapshot_logs")(
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
        workload_id=calculated_params["workload_id"],
        expect_fallback=expect_fallback,
        expect_scale_up=expect_scale_up,
        expected_min_restores=expected_min_restores,
        expect_data_iterator=expect_data_iterator,
    )
    clean_up_recipe = gcluster.clean_up_workload.override(
        task_id="clean_up_recipe"
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        zone=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
    ).as_teardown()

    chain(
        fetched_params,
        calculated_params,
        generated_cmds,
        start_recipe,
        interruption_task,
        wait_for_workload_complete,
        verify_logs,
        clean_up_recipe,
    )
  return dag


dag_snapshot_replica_resize = create_elastic_snapshot_dag(
    dag_id="pw_elastic_snapshot_replica_resize",
    workload_prefix="pw_snap_rr",
    elastic_type="Replica-resize",
    num_slices=2,
    benchmark_steps=500,
    enable_checkpointing=False,
    expect_scale_up=True,
    expected_min_restores=2,
    post_interrupt_checks=(
        (
            "wait_for_slice_down_restore",
            "Restored the in-memory snapshot at step ",
        ),
        (
            "wait_for_scale_up_snapshot",
            "New slices are available. Snapshotting step ",
        ),
        (
            "wait_for_scale_up_restore",
            "Restored the in-memory snapshot at step ",
        ),
    ),
    description=(
        "MaxText Pathways elastic in-memory snapshot E2E test for 2->1->2"
        " replica resize and scale-up."
    ),
    doc_md=(
        "# MaxText Pathways Elastic In-Memory Snapshot: Replica-Resize\n"
        "Validates 2->1 slice-down host-RAM restore and 1->2 pre-scale-up"
        " snapshot/restore with Grain `ElasticIterator` and zero step loss."
    ),
)

dag_snapshot_pause_resume = create_elastic_snapshot_dag(
    dag_id="pw_elastic_snapshot_pause_resume",
    workload_prefix="pw_snap_pr",
    elastic_type="Pause-resume",
    num_slices=2,
    benchmark_steps=500,
    enable_checkpointing=False,
    post_interrupt_checks=(
        ("wait_for_slices_active", "Sufficient slices active:"),
        (
            "wait_for_snapshot_restore",
            "Restored the in-memory snapshot at step ",
        ),
    ),
    description=(
        "MaxText Pathways elastic in-memory snapshot E2E test for 2-slice"
        " pause-resume from surviving slice RAM."
    ),
    doc_md=(
        "# MaxText Pathways Elastic In-Memory Snapshot: 2-Slice Pause-Resume\n"
        "Validates 2->0->2 pause-resume restoring train state and Grain"
        " `ElasticIterator` from the surviving slice's pinned host RAM."
    ),
)

dag_snapshot_ckpt_fallback = create_elastic_snapshot_dag(
    dag_id="pw_elastic_snapshot_ckpt_fallback",
    workload_prefix="pw_snap_fb",
    elastic_type="Pause-resume",
    num_slices=1,
    snapshot_interval=155,
    checkpoint_period=150,
    benchmark_steps=280,
    enable_checkpointing=True,
    async_checkpointing=False,
    interrupt_after_snapshot_step=155,
    post_snapshot_steps=3,
    expect_fallback=True,
    expected_min_restores=0,
    expect_data_iterator=False,
    post_interrupt_checks=(
        (
            "wait_for_ckpt_fallback",
            "In-memory snapshot restore failed; falling back to checkpoint"
            " restore.",
        ),
    ),
    description=(
        "MaxText Pathways elastic in-memory snapshot E2E test for 1-slice"
        " graceful fallback to GCS checkpoint."
    ),
    doc_md=(
        "# MaxText Pathways Elastic In-Memory Snapshot: 1-Slice Fallback\n"
        "Validates graceful fallback to GCS checkpoint when a 1-slice FSDP"
        " host-RAM snapshot loses shards with no surviving replica."
    ),
)
