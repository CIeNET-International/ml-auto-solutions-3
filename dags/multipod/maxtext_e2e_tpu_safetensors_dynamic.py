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

"""MaxText E2E TPU safetensors_dynamic vs. to_maxtext Checkpoint Comparison DAG.

Compares two checkpoint loading/generation workflows end-to-end on TPU across
configurable models and single-slice / multi-slice topologies:
1. Original `to_maxtext` workflow:
   - Offline conversion via `maxtext.checkpoint_conversion.to_maxtext` producing
     the reference untrained Orbax checkpoint (`.../to_maxtext/0/items`).
   - Deterministic pre-training with `save_checkpoint_on_start=true` producing
     Step-0 (`.../train_to_maxtext/run/checkpoints/0/items`) and trained
     Step-(N-1) (`.../train_to_maxtext/run/checkpoints/{steps-1}/items`) Orbax
     checkpoints plus training metrics.
2. `safetensors_dynamic` workflow:
   - On-the-fly `.safetensors` loading via
     `source_checkpoint_layout=safetensors_dynamic` with deterministic
     pre-training, optionally writing Step-0 and trained Step-(N-1) Orbax
     checkpoints plus training metrics.
3. Checkpoint & Metrics Comparison:
   - Compares (a) in-memory `safetensors_dynamic` load vs. `to_maxtext/0/items`,
     (b) saved Step-0 Orbax checkpoints, (c) saved trained Step-(N-1) Orbax
     checkpoints, and (d) per-step training losses.
"""

import dataclasses
import datetime
import re

from airflow import models
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.utils.task_group import TaskGroup
from dags import gcs_bucket
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import DockerImage, GkeClusters
from dags.multipod.configs import gke_config
from dags.multipod.util import safetensors_util
from xlml.apis.gke_cluster_config import GkeClusterConfig

HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)


@dataclasses.dataclass(frozen=True)
class SafetensorsComparisonSpec:
  """Declarative specification for a model & slice-topology safetensors test."""

  model_name: str
  hf_repo: str
  cluster: GkeClusterConfig = dataclasses.field(
      default_factory=lambda: GkeClusters.TPU_V5P_BODABORG_NAP_CLUSTER
  )
  core_count: int = 8
  num_slices: int = 1
  scan_layers: bool = True
  weight_dtype: str = "float32"
  steps: int = 2
  per_device_batch_size: int = 1
  max_target_length: int = 2048
  atol: float = 0.0
  enable_comparison: bool = True
  save_checkpoints: bool | None = None
  prestage_hf_to_gcs: bool = True
  extra_maxtext_flags: tuple[str, ...] = ()
  extra_to_maxtext_flags: tuple[str, ...] = ()
  time_out_in_min: int = 60
  priority: str = "medium"
  group_id_override: str | None = None

  @property
  def should_save_checkpoints(self) -> bool:
    """Whether training runs write Step-0 & trained Orbax checkpoints to GCS."""
    if self.save_checkpoints is not None:
      return self.save_checkpoints
    return self.enable_comparison

  @property
  def group_id(self) -> str:
    """Returns an Airflow-safe TaskGroup ID (alphanumeric, dashes, `_`)."""
    if self.group_id_override:
      return self.group_id_override
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", self.model_name)
    tpu_ver = f"v{self.cluster.device_version.value}"
    if self.num_slices > 1:
      return f"{safe_model}-{self.num_slices}x-{tpu_ver}-{self.core_count}"
    return f"{safe_model}-{tpu_ver}-{self.core_count}"


def create_safetensors_comparison_task_group(
    spec: SafetensorsComparisonSpec,
    owner: str = test_owner.KEVIN_Y,
) -> TaskGroup:
  """Builds the `safetensors-dyn` (and optional `to-mt` + `cmp-ckpt`) group."""
  with TaskGroup(group_id=spec.group_id) as model_group:
    run_name = (
        "{{ params.run_name if params.run_name else 'sd-cmp-' ~ ts_nodash }}"
    )
    model = spec.model_name
    hf_repo = spec.hf_repo
    weight_dtype = spec.weight_dtype
    scan_layers = "true" if spec.scan_layers else "false"
    trained_step = spec.steps - 1
    extra_flags_str = (
        " " + " ".join(spec.extra_maxtext_flags)
        if spec.extra_maxtext_flags
        else ""
    )
    extra_to_mt_str = (
        " " + " ".join(spec.extra_to_maxtext_flags)
        if spec.extra_to_maxtext_flags
        else ""
    )

    cluster = spec.cluster.override(core_count=spec.core_count)

    topology_tag = (
        f"{spec.num_slices}x_v{cluster.device_version.value}_{spec.core_count}"
    )
    base_dir = (
        f"{gcs_bucket.BASE_OUTPUT_DIR}/{model}/safetensors_compare/"
        f"{topology_tag}/{run_name}"
    )
    markers_dir = f"{base_dir}/markers"
    to_maxtext_dir = f"{base_dir}/to_maxtext"
    to_maxtext_step0_ckpt = f"{to_maxtext_dir}/0/items"

    train_ref_dir = f"{base_dir}/train_to_maxtext"
    train_ref_run = "run"
    train_ref_step0_ckpt = (
        f"{train_ref_dir}/{train_ref_run}/checkpoints/0/items"
    )
    train_ref_trained_ckpt = (
        f"{train_ref_dir}/{train_ref_run}/checkpoints/{trained_step}/items"
    )
    ref_metrics_gcs = f"{train_ref_dir}/metrics_to_maxtext.txt"

    train_dyn_dir = f"{base_dir}/train_safetensors_dynamic"
    train_dyn_run = "run"
    train_dyn_step0_ckpt = (
        f"{train_dyn_dir}/{train_dyn_run}/checkpoints/0/items"
    )
    train_dyn_trained_ckpt = (
        f"{train_dyn_dir}/{train_dyn_run}/checkpoints/{trained_step}/items"
    )
    dyn_metrics_gcs = f"{train_dyn_dir}/metrics_safetensors_dynamic.txt"

    report_gcs = f"{base_dir}/comparison_report.json"

    shared_hf_gcs_cache = (
        f"{gcs_bucket.BASE_OUTPUT_DIR}/hf_safetensors_cache/"
        f"{hf_repo.replace('/', '_')}"
    )
    dyn_load_parameters_path = (
        shared_hf_gcs_cache if spec.prestage_hf_to_gcs else f"hf://{hf_repo}"
    )

    # Keep `enable_checkpointing=true` even when not writing checkpoints so
    # multi-host / multi-slice TPU runs initialize `jax.distributed` on `main`.
    ckpt_save_flags = (
        "checkpoint_period=1 save_checkpoint_on_start=true"
        if spec.should_save_checkpoints
        else (
            "save_checkpoint_on_start=false "
            "save_checkpoint_on_completion=false"
        )
    )

    common_train_flags = (
        "src/maxtext/configs/base.yml "
        f"model_name={model} "
        f"scan_layers={scan_layers} "
        f"weight_dtype={weight_dtype} "
        f"tokenizer_path={hf_repo} "
        "tokenizer_type=huggingface "
        "dataset_type=hf "
        "hf_path=allenai/c4 "
        "hf_data_dir=en "
        "train_split=train "
        "enable_data_shuffling=false "
        f"per_device_batch_size={spec.per_device_batch_size} "
        f"max_target_length={spec.max_target_length} "
        f"steps={spec.steps} "
        "enable_checkpointing=true "
        "async_checkpointing=false "
        f"{ckpt_save_flags}{extra_flags_str}"
    )

    # `safetensors_dynamic` workflow (always runs)
    dyn_pod0_prep_cmds = [
        (
            f"if gsutil ls {train_dyn_dir} >/dev/null 2>&1; then "
            f"gsutil -m rm -rf {train_dyn_dir} || true; "
            "fi"
        ),
    ]
    if spec.prestage_hf_to_gcs:
      dyn_pod0_prep_cmds.extend(
          safetensors_util.build_prestage_cmds(
              hf_repo=hf_repo,
              gcs_cache_dir=shared_hf_gcs_cache,
          )
      )
    dyn_pod0_prep = tuple(dyn_pod0_prep_cmds)

    safetensors_dynamic_cmds = (
        f"export HF_TOKEN={HF_TOKEN}",
        'export HF_HOME="/mnt/shm/hf_cache"',
        'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
        safetensors_util.pod0_barrier_cmd(
            dyn_pod0_prep, f"{markers_dir}/sd_dyn_prep.done"
        ),
        (
            "python3 -m maxtext.trainers.pre_train.train "
            f"{common_train_flags} "
            f"base_output_directory={train_dyn_dir} "
            f"run_name={train_dyn_run} "
            f"load_parameters_path={dyn_load_parameters_path} "
            "source_checkpoint_layout=safetensors_dynamic "
            "metrics_file=/tmp/metrics_safetensors_dynamic.txt"
        ),
        (
            f"if {safetensors_util.POD0_CONDITION}; then "
            f"gsutil cp /tmp/metrics_safetensors_dynamic.txt {dyn_metrics_gcs};"
            " fi"
        ),
    )

    safetensors_dynamic_task = gke_config.get_gke_config(
        time_out_in_min=spec.time_out_in_min,
        test_name="safetensors-dyn",
        run_model_cmds=safetensors_dynamic_cmds,
        docker_image="{{ params.docker_image }}",
        cluster=cluster,
        num_slices=spec.num_slices,
        test_owner=owner,
        priority=spec.priority,
        use_gcluster=True,
        mounts="/dev/shm;/mnt/shm;rw",
    ).run(skip_post_process=True)

    if spec.enable_comparison:
      local_hf_dir = f"/mnt/shm/{model}_hf"

      # Task 1: Original `to_maxtext` workflow (Pod 0 converts; all pods train)
      to_mt_pod0_prep = (
          (
              f"if gsutil ls {to_maxtext_dir} >/dev/null 2>&1; then "
              f"gsutil -m rm -rf {to_maxtext_dir} {train_ref_dir} || true; "
              "fi"
          ),
          (
              'python3 -c "import os, huggingface_hub as h; '
              f"h.snapshot_download('{hf_repo}', local_dir='{local_hf_dir}', "
              "allow_patterns=["
              "'*.safetensors','*.json','*.model','*.txt','*.jinja','*.py'"
              "], "
              "token=os.environ.get('HF_TOKEN'), max_workers=16)\""
          ),
          (
              "env JAX_PLATFORMS=cpu python3 -m "
              "maxtext.checkpoint_conversion.to_maxtext "
              "src/maxtext/configs/base.yml "
              f"model_name={model} "
              f"base_output_directory={to_maxtext_dir} "
              f"scan_layers={scan_layers} "
              "hardware=cpu "
              f"skip_jax_distributed_system=True{extra_to_mt_str} "
              f"--hf_model_path={local_hf_dir} "
              "--lazy_load_tensors=True "
              f"--save_dtype={weight_dtype}"
          ),
          f"rm -rf {local_hf_dir}",
      )

      to_maxtext_cmds = (
          f"export HF_TOKEN={HF_TOKEN}",
          'export HF_HOME="/mnt/shm/hf_cache"',
          'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
          safetensors_util.pod0_barrier_cmd(
              to_mt_pod0_prep, f"{markers_dir}/to_maxtext_prep.done"
          ),
          (
              "python3 -m maxtext.trainers.pre_train.train "
              f"{common_train_flags} "
              f"base_output_directory={train_ref_dir} "
              f"run_name={train_ref_run} "
              f"load_parameters_path={to_maxtext_step0_ckpt} "
              "metrics_file=/tmp/metrics_ref.jsonl"
          ),
          (
              f"if {safetensors_util.POD0_CONDITION}; then "
              f"gsutil cp /tmp/metrics_ref.jsonl {ref_metrics_gcs}; "
              "fi"
          ),
      )

      to_maxtext_task = gke_config.get_gke_config(
          time_out_in_min=spec.time_out_in_min,
          test_name="to-mt",
          run_model_cmds=to_maxtext_cmds,
          docker_image="{{ params.docker_image }}",
          cluster=cluster,
          num_slices=spec.num_slices,
          test_owner=owner,
          priority=spec.priority,
          use_gcluster=True,
          mounts="/dev/shm;/mnt/shm;rw",
      ).run(skip_post_process=True)

      # Task 3: Compare in-memory weights, Step-0 & trained ckpts, and metrics
      compare_maxtext_args = (
          "src/maxtext/configs/base.yml "
          f"model_name={model} "
          f"base_output_directory={base_dir} "
          "run_name=compare_ckpts "
          f"load_parameters_path={dyn_load_parameters_path} "
          "source_checkpoint_layout=safetensors_dynamic "
          f"scan_layers={scan_layers} "
          f"weight_dtype={weight_dtype} "
          "dtype=float32 "
          f"per_device_batch_size={spec.per_device_batch_size} "
          "max_target_length=512 "
          "enable_checkpointing=true "
          f"async_checkpointing=false{extra_flags_str}"
      )
      compare_cmds = (
          f"export HF_TOKEN={HF_TOKEN}",
          'export HF_HOME="/mnt/shm/hf_cache"',
          'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
          *safetensors_util.build_compare_cmds(
              to_maxtext_ckpt=to_maxtext_step0_ckpt,
              ref_step0_ckpt=train_ref_step0_ckpt,
              dyn_step0_ckpt=train_dyn_step0_ckpt,
              ref_trained_ckpt=train_ref_trained_ckpt,
              dyn_trained_ckpt=train_dyn_trained_ckpt,
              trained_step=trained_step,
              ref_metrics=ref_metrics_gcs,
              dyn_metrics=dyn_metrics_gcs,
              report_gcs=report_gcs,
              atol=spec.atol,
              maxtext_args=compare_maxtext_args,
          ),
      )

      compare_checkpoints_task = gke_config.get_gke_config(
          time_out_in_min=spec.time_out_in_min,
          test_name="cmp-ckpt",
          run_model_cmds=compare_cmds,
          docker_image="{{ params.docker_image }}",
          cluster=cluster,
          num_slices=spec.num_slices,
          test_owner=owner,
          priority=spec.priority,
          use_gcluster=True,
          mounts="/dev/shm;/mnt/shm;rw",
      ).run(skip_post_process=True)

      chain(
          [to_maxtext_task, safetensors_dynamic_task],
          compare_checkpoints_task,
      )

  return model_group


# Add new models or single-slice / multi-slice specs to `TEST_SPECS` below.
# Set `enable_comparison=True` for full `to-mt` + `safetensors-dyn` + `cmp-ckpt`
# parity tests, or `enable_comparison=False` for standalone `safetensors-dyn`
# functional runs.
# Example full comparison spec (with direct `hf://` URI test on a small model):
#   SafetensorsComparisonSpec(
#       model_name="qwen3-0.6b",
#       hf_repo="Qwen/Qwen3-0.6B",
#       core_count=8,
#       num_slices=1,
#       weight_dtype="float32",
#       scan_layers=True,
#       atol=0.0,
#       enable_comparison=True,
#       prestage_hf_to_gcs=False,
#   ),
TEST_SPECS: tuple[SafetensorsComparisonSpec, ...] = (
    # 1. Single-slice Llama-3.1-70B vanilla functional run (1x v5p-32)
    SafetensorsComparisonSpec(
        model_name="llama3.1-70b",
        hf_repo="meta-llama/Llama-3.1-70B",
        core_count=32,
        num_slices=1,
        weight_dtype="bfloat16",
        scan_layers=True,
        steps=2,
        enable_comparison=False,
        time_out_in_min=90,
    ),
    # 2. Multi-slice Llama-3.1-70B vanilla functional run (2x v5p-32)
    SafetensorsComparisonSpec(
        model_name="llama3.1-70b",
        hf_repo="meta-llama/Llama-3.1-70B",
        core_count=32,
        num_slices=2,
        weight_dtype="bfloat16",
        scan_layers=True,
        steps=2,
        enable_comparison=False,
        extra_maxtext_flags=("dcn_data_parallelism=2",),
        time_out_in_min=90,
    ),
)

with models.DAG(
    dag_id="maxtext_e2e_tpu_safetensors_dynamic",
    schedule=None,
    tags=[
        "multipod_team",
        "maxtext",
        "safetensors_dynamic",
        "checkpoint",
        "e2e",
        "TPU",
    ],
    start_date=datetime.datetime(2026, 6, 10),
    catchup=False,
    default_args={
        "owner": test_owner.KEVIN_Y,
    },
    params={
        "docker_image": Param(
            default=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
            type="string",
            description="MaxText Docker image URI to test",
        ),
        "run_name": Param(
            default="",
            type="string",
            description=(
                "Unique run name for checkpoints and comparison outputs "
                "(defaults to sd-cmp-{{ ts_nodash }})"
            ),
        ),
    },
) as dag:
  for test_spec in TEST_SPECS:
    create_safetensors_comparison_task_group(test_spec)
