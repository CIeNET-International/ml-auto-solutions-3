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
   - On-the-fly `.safetensors` loading via `source_checkpoint_layout=safetensors_dynamic`
     with `save_checkpoint_on_start=true` and deterministic pre-training,
     producing Step-0 (`.../train_safetensors_dynamic/run/checkpoints/0/items`)
     and trained Step-(N-1) (`.../train_safetensors_dynamic/run/checkpoints/{steps-1}/items`)
     Orbax checkpoints plus training metrics.
3. Checkpoint & Metrics Comparison:
   - Compares (a) in-memory `safetensors_dynamic` load vs. `to_maxtext/0/items`,
     (b) saved Step-0 Orbax checkpoints, (c) saved trained Step-(N-1) Orbax
     checkpoints, and (d) per-step training losses.
"""

import base64
import dataclasses
import datetime
import re
import textwrap

from airflow import models
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.utils.task_group import TaskGroup
from dags import gcs_bucket
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import DockerImage, GkeClusters
from dags.multipod.configs import gke_config
from xlml.apis.gke_cluster_config import GkeClusterConfig

HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)

COMPARE_SCRIPT_CONTENT = textwrap.dedent(
    """\
    import argparse
    import json
    import sys
    import time

    from absl import logging as absl_logging
    from etils import epath
    from flax import nnx
    from flax import traverse_util
    from flax.core.spmd import logical_axis_rules as axis_rules
    import jax
    import jax.numpy as jnp

    from maxtext.common import checkpointing
    from maxtext.common import train_state_nnx
    from maxtext.configs import pyconfig
    from maxtext.utils import max_logging
    from maxtext.utils import maxtext_utils
    from maxtext.utils import model_creation_utils
    from maxtext.utils import train_utils


    def _abstract_train_state(config, mesh):
      create_model_partial, abstract_model = (
          model_creation_utils.create_nnx_abstract_model(config, mesh)
      )
      _, tx = train_utils.create_training_optimizer(
          config, abstract_model, mesh=mesh
      )

      def init_state_fn():
        model = create_model_partial()
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return train_state_nnx.TrainStateNNX(model, optimizer)

      abstract_state, _, _ = maxtext_utils.get_abstract_state(
          config, mesh, init_state_fn, is_training=True
      )
      return abstract_state


    def _load_params(config, abstract_state, path, layout="orbax"):
      with axis_rules(config.logical_axis_rules):
        _, params = checkpointing.load_state_if_possible(
            None,
            None,
            path,
            "",
            config.checkpoint_storage_concurrent_gb,
            abstract_state,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
            source_checkpoint_layout=layout,
            maxtext_config=config,
        )
      pure = params.to_pure_dict() if hasattr(params, "to_pure_dict") else params
      jax.block_until_ready(pure)
      return pure


    @jax.jit
    def _leaf_stats(a, b):
      a32 = a.astype(jnp.float32)
      b32 = b.astype(jnp.float32)
      equal = jnp.all((a == b) | (jnp.isnan(a) & jnp.isnan(b)))
      return jnp.max(jnp.abs(a32 - b32)), equal


    def _compare_checkpoint_pair(
        config,
        abstract_state,
        label,
        ref_path,
        dyn_path,
        atol,
        ref_layout="orbax",
        dyn_layout="orbax",
    ):
      t0 = time.time()
      ref_params = _load_params(config, abstract_state, ref_path, layout=ref_layout)
      ref_load_s = time.time() - t0

      t0 = time.time()
      dyn_params = _load_params(config, abstract_state, dyn_path, layout=dyn_layout)
      dyn_load_s = time.time() - t0

      rflat = traverse_util.flatten_dict(ref_params)
      dflat = traverse_util.flatten_dict(dyn_params)

      only_ref = sorted("/".join(map(str, k)) for k in rflat.keys() - dflat.keys())
      only_dyn = sorted("/".join(map(str, k)) for k in dflat.keys() - rflat.keys())

      leaves = []
      mismatches = []
      for key in sorted(rflat.keys() & dflat.keys()):
        a = dflat[key]
        b = rflat[key]
        name = "/".join(map(str, key))
        shape_ok = a.shape == b.shape
        dtype_ok = a.dtype == b.dtype
        sharding_ok = getattr(a, "sharding", None) == getattr(b, "sharding", None)
        entry = {
            "name": name,
            "shape": list(a.shape),
            "ref_shape": list(b.shape),
            "dtype": str(a.dtype),
            "ref_dtype": str(b.dtype),
            "shape_match": shape_ok,
            "dtype_match": dtype_ok,
            "sharding_match": sharding_ok,
        }
        if not shape_ok:
          entry["status"] = "SHAPE_MISMATCH"
          mismatches.append(entry)
        else:
          max_diff, exact = _leaf_stats(a, b)
          max_diff_val = float(max_diff)
          exact_val = bool(exact)
          within_tol = exact_val or (max_diff_val <= atol)
          entry.update({
              "max_abs_diff": max_diff_val,
              "exact": exact_val,
              "status": "PASS" if (within_tol and dtype_ok and sharding_ok) else "MISMATCH",
          })
          if entry["status"] != "PASS":
            mismatches.append(entry)
        leaves.append(entry)

      overall_max_diff = max(
          (e.get("max_abs_diff", float("inf")) for e in leaves), default=0.0
      )
      passed = bool(leaves) and not mismatches and not only_ref and not only_dyn
      summary = {
          "label": label,
          "ref_ckpt": ref_path,
          "dyn_ckpt": dyn_path,
          "ref_load_seconds": round(ref_load_s, 2),
          "dyn_load_seconds": round(dyn_load_s, 2),
          "num_leaves_compared": len(leaves),
          "num_mismatches": len(mismatches),
          "overall_max_abs_diff": overall_max_diff,
          "only_in_ref": only_ref,
          "only_in_dyn": only_dyn,
          "mismatches": mismatches[:50],
          "result": "PASS" if passed else "FAIL",
      }
      max_logging.log(
          f"CKPT_COMPARE [{label}]: {summary['result']} "
          f"(leaves={len(leaves)}, mismatches={len(mismatches)}, "
          f"max_abs_diff={overall_max_diff}, only_ref={len(only_ref)}, "
          f"only_dyn={len(only_dyn)})"
      )
      return summary


    def _read_losses(metrics_path):
      path = epath.Path(metrics_path)
      if not path.exists():
        return []
      losses = []
      for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
          continue
        rec = json.loads(line)
        if "learning/loss" in rec:
          losses.append((int(rec.get("step", len(losses))), float(rec["learning/loss"])))
      return losses


    def _compare_metrics(ref_metrics_path, dyn_metrics_path, atol):
      ref_losses = _read_losses(ref_metrics_path)
      dyn_losses = _read_losses(dyn_metrics_path)
      diffs = []
      for (s_ref, l_ref), (s_dyn, l_dyn) in zip(ref_losses, dyn_losses):
        diff = abs(l_ref - l_dyn)
        diffs.append({
            "step_ref": s_ref,
            "step_dyn": s_dyn,
            "loss_ref": l_ref,
            "loss_dyn": l_dyn,
            "abs_diff": diff,
            "pass": s_ref == s_dyn and diff <= atol,
        })
      max_loss_diff = max((d["abs_diff"] for d in diffs), default=0.0)
      passed = (
          bool(ref_losses)
          and len(ref_losses) == len(dyn_losses)
          and all(d["pass"] for d in diffs)
      )
      summary = {
          "ref_metrics": ref_metrics_path,
          "dyn_metrics": dyn_metrics_path,
          "num_steps_ref": len(ref_losses),
          "num_steps_dyn": len(dyn_losses),
          "max_loss_abs_diff": max_loss_diff,
          "steps": diffs,
          "result": "PASS" if passed else "FAIL",
      }
      max_logging.log(
          f"METRICS_COMPARE: {summary['result']} "
          f"(steps={len(diffs)}, max_loss_abs_diff={max_loss_diff})"
      )
      return summary


    def main():
      absl_logging.set_verbosity(absl_logging.INFO)
      parser = argparse.ArgumentParser()
      parser.add_argument("--to_maxtext_ckpt", required=True)
      parser.add_argument("--ref_step0_ckpt", required=True)
      parser.add_argument("--dyn_step0_ckpt", required=True)
      parser.add_argument("--ref_trained_ckpt", required=True)
      parser.add_argument("--dyn_trained_ckpt", required=True)
      parser.add_argument("--trained_step", type=int, default=1)
      parser.add_argument("--ref_metrics", required=True)
      parser.add_argument("--dyn_metrics", required=True)
      parser.add_argument("--report", required=True)
      parser.add_argument("--atol", type=float, default=0.0)
      args, model_rest = parser.parse_known_args(sys.argv[1:])

      config = pyconfig.initialize(["compare_ckpts"] + model_rest)
      mesh = maxtext_utils.get_mesh_from_config(config)
      abstract_state = _abstract_train_state(config, mesh)

      in_memory_summary = _compare_checkpoint_pair(
          config,
          abstract_state,
          "in_memory_safetensors_vs_to_maxtext",
          args.to_maxtext_ckpt,
          config.load_parameters_path,
          args.atol,
          ref_layout="orbax",
          dyn_layout="safetensors_dynamic",
      )
      step0_summary = _compare_checkpoint_pair(
          config,
          abstract_state,
          "step0_ckpt",
          args.ref_step0_ckpt,
          args.dyn_step0_ckpt,
          args.atol,
      )
      trained_label = f"step{args.trained_step}_trained_ckpt"
      trained_summary = _compare_checkpoint_pair(
          config,
          abstract_state,
          trained_label,
          args.ref_trained_ckpt,
          args.dyn_trained_ckpt,
          args.atol,
      )
      metrics_summary = _compare_metrics(
          args.ref_metrics, args.dyn_metrics, args.atol
      )

      overall_pass = (
          in_memory_summary["result"] == "PASS"
          and step0_summary["result"] == "PASS"
          and trained_summary["result"] == "PASS"
          and metrics_summary["result"] == "PASS"
      )
      report = {
          "model_name": config.model_name,
          "scan_layers": bool(config.scan_layers),
          "weight_dtype": str(config.weight_dtype),
          "num_processes": jax.process_count(),
          "num_devices": jax.device_count(),
          "device_kind": jax.devices()[0].device_kind,
          "mesh": {k: int(v) for k, v in mesh.shape.items()},
          "overall_result": "PASS" if overall_pass else "FAIL",
          "in_memory_comparison": in_memory_summary,
          "step0_checkpoint_comparison": step0_summary,
          "trained_checkpoint_comparison": trained_summary,
          "training_metrics_comparison": metrics_summary,
      }
      if jax.process_index() == 0:
        epath.Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True))
      max_logging.log("OVERALL_COMPARE_RESULT: " + report["overall_result"])
      if not overall_pass:
        sys.exit(1)


    if __name__ == "__main__":
      main()
    """
)

COMPARE_SCRIPT_B64 = base64.b64encode(
    COMPARE_SCRIPT_CONTENT.encode("utf-8")
).decode("ascii")

POD0_CONDITION = (
    '[[ "${MEGASCALE_SLICE_ID:-${JOB_INDEX:-0}}" == "0" && '
    '"${JOB_COMPLETION_INDEX:-0}" == "0" ]]'
)


@dataclasses.dataclass(frozen=True)
class SafetensorsComparisonSpec:
  """Declarative specification for a model & slice-topology comparison test."""

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
  extra_maxtext_flags: tuple[str, ...] = ()
  extra_to_maxtext_flags: tuple[str, ...] = ()
  time_out_in_min: int = 60
  priority: str = "medium"
  group_id_override: str | None = None

  @property
  def group_id(self) -> str:
    """Returns an Airflow-safe TaskGroup ID (alphanumeric, dashes, underscores)."""
    if self.group_id_override:
      return self.group_id_override
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", self.model_name)
    tpu_ver = f"v{self.cluster.device_version.value}"
    if self.num_slices > 1:
      return f"{safe_model}-{self.num_slices}x-{tpu_ver}-{self.core_count}"
    return f"{safe_model}-{tpu_ver}-{self.core_count}"


def _pod0_barrier_cmd(
    pod0_cmds: tuple[str, ...],
    marker_gcs: str,
    poll_attempts: int = 720,
    poll_interval_sec: int = 10,
) -> str:
  """Runs `pod0_cmds` only on slice 0 / worker 0 and synchronizes all pods via GCS."""
  joined_pod0 = " && ".join(pod0_cmds)
  return (
      f"if {POD0_CONDITION}; then "
      f"gsutil -q rm -f {marker_gcs} 2>/dev/null || true; "
      f"{joined_pod0} && "
      f"echo done | gsutil -q cp - {marker_gcs}; "
      "else "
      f'echo "Waiting for pod 0 marker {marker_gcs}..."; '
      f"for _ in $(seq 1 {poll_attempts}); do "
      f"gsutil -q stat {marker_gcs} && break; sleep {poll_interval_sec}; "
      "done; "
      f"gsutil -q stat {marker_gcs}; "
      "fi"
  )


def create_safetensors_comparison_task_group(
    spec: SafetensorsComparisonSpec,
    owner: str = test_owner.KEVIN_Y,
) -> TaskGroup:
  """Builds the 3-task comparison TaskGroup (`to-mt`, `safetensors-dyn`, `cmp-ckpt`)."""
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
        f"checkpoint_period=1{extra_flags_str}"
    )

    local_hf_dir = f"/mnt/shm/{model}_hf"

    # Task 1: Original `to_maxtext` workflow (Pod 0 converts on CPU; all pods train)
    to_mt_pod0_prep = (
        (
            f"if gsutil ls {to_maxtext_dir} >/dev/null 2>&1; then "
            f"gsutil -m rm -rf {to_maxtext_dir} {train_ref_dir} || true; "
            "fi"
        ),
        (
            "python3 -c \"import os, huggingface_hub as h; "
            f"h.snapshot_download('{hf_repo}', local_dir='{local_hf_dir}', "
            "allow_patterns=['*.safetensors','*.json','*.model','*.txt','*.jinja','*.py'], "
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
        _pod0_barrier_cmd(to_mt_pod0_prep, f"{markers_dir}/to_maxtext_prep.done"),
        (
            "python3 -m maxtext.trainers.pre_train.train "
            f"{common_train_flags} "
            f"base_output_directory={train_ref_dir} "
            f"run_name={train_ref_run} "
            f"load_parameters_path={to_maxtext_step0_ckpt} "
            "save_checkpoint_on_start=true "
            "metrics_file=/tmp/metrics_ref.jsonl"
        ),
        (
            f"if {POD0_CONDITION}; then "
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

    # Task 2: `safetensors_dynamic` workflow (all pods load dynamically & train)
    dyn_pod0_prep = (
        (
            f"if gsutil ls {train_dyn_dir} >/dev/null 2>&1; then "
            f"gsutil -m rm -rf {train_dyn_dir} || true; "
            "fi"
        ),
    )

    safetensors_dynamic_cmds = (
        f"export HF_TOKEN={HF_TOKEN}",
        'export HF_HOME="/mnt/shm/hf_cache"',
        'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
        _pod0_barrier_cmd(dyn_pod0_prep, f"{markers_dir}/sd_dyn_prep.done"),
        (
            "python3 -m maxtext.trainers.pre_train.train "
            f"{common_train_flags} "
            f"base_output_directory={train_dyn_dir} "
            f"run_name={train_dyn_run} "
            f"load_parameters_path=hf://{hf_repo} "
            "source_checkpoint_layout=safetensors_dynamic "
            "save_checkpoint_on_start=true "
            "metrics_file=/tmp/metrics_safetensors_dynamic.txt"
        ),
        (
            f"if {POD0_CONDITION}; then "
            f"gsutil cp /tmp/metrics_safetensors_dynamic.txt {dyn_metrics_gcs}; "
            "fi"
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

    # Task 3: Compare in-memory weights, Step-0 & trained Orbax checkpoints, and metrics
    compare_cmds = (
        f"export HF_TOKEN={HF_TOKEN}",
        'export HF_HOME="/mnt/shm/hf_cache"',
        'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
        (
            f"echo '{COMPARE_SCRIPT_B64}' | base64 -d > "
            "/tmp/compare_safetensors_checkpoints.py"
        ),
        (
            "python3 /tmp/compare_safetensors_checkpoints.py "
            f"--to_maxtext_ckpt={to_maxtext_step0_ckpt} "
            f"--ref_step0_ckpt={train_ref_step0_ckpt} "
            f"--dyn_step0_ckpt={train_dyn_step0_ckpt} "
            f"--ref_trained_ckpt={train_ref_trained_ckpt} "
            f"--dyn_trained_ckpt={train_dyn_trained_ckpt} "
            f"--trained_step={trained_step} "
            f"--ref_metrics={ref_metrics_gcs} "
            f"--dyn_metrics={dyn_metrics_gcs} "
            f"--report={report_gcs} "
            f"--atol={spec.atol} "
            "src/maxtext/configs/base.yml "
            f"model_name={model} "
            f"base_output_directory={base_dir} "
            "run_name=compare_ckpts "
            f"load_parameters_path=hf://{hf_repo} "
            "source_checkpoint_layout=safetensors_dynamic "
            f"scan_layers={scan_layers} "
            f"weight_dtype={weight_dtype} "
            "dtype=float32 "
            f"per_device_batch_size={spec.per_device_batch_size} "
            "max_target_length=512 "
            "enable_checkpointing=true "
            f"async_checkpointing=false{extra_flags_str}"
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
# Examples for future extension:
#   SafetensorsComparisonSpec(
#       model_name="qwen3-0.6b",
#       hf_repo="Qwen/Qwen3-0.6B",
#       core_count=8,
#       num_slices=2,
#       extra_maxtext_flags=("dcn_data_parallelism=2", "ici_fsdp_parallelism=4"),
#   ),
#   SafetensorsComparisonSpec(
#       model_name="llama3.1-8b",
#       hf_repo="meta-llama/Llama-3.1-8B",
#       core_count=16,
#       num_slices=1,
#   ),
TEST_SPECS: tuple[SafetensorsComparisonSpec, ...] = (
    SafetensorsComparisonSpec(
        model_name="qwen3-0.6b",
        hf_repo="Qwen/Qwen3-0.6B",
        core_count=8,
        num_slices=1,
        weight_dtype="float32",
        scan_layers=True,
        atol=0.0,
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
