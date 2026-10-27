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

"""Utilities and remote TPU pod scripts for MaxText safetensors_dynamic tests.

This module serves two roles without embedding multiline script strings:
1. Airflow DAG command builders (`pod0_barrier_cmd`, `build_prestage_cmds`,
   `build_compare_cmds`) imported by `maxtext_e2e_tpu_safetensors_dynamic.py`,
   which base64-stage this standalone Python file onto remote TPU pods.
2. Remote TPU pod CLI (`/tmp/safetensors_util.py`) supporting two subcommands:
   - `prestage`: Atomically caches Hugging Face `*.safetensors` shards into a
     shared GCS directory using a generation-matched GCS lock.
   - `compare`: Compares in-memory `safetensors_dynamic` weights, Step-0 Orbax
     checkpoints, trained Step-(N-1) Orbax checkpoints, and per-step losses
     against the reference `to_maxtext` workflow.
"""

import argparse
import base64
import concurrent.futures
import datetime
import functools
import json
import os
import pathlib
import random
import shlex
import sys
import time
from typing import Any, Callable

try:
  from google.api_core import exceptions as gcs_exceptions
  from google.cloud import storage
except ImportError:
  gcs_exceptions = None
  storage = None

try:
  import huggingface_hub
except ImportError:
  huggingface_hub = None

try:
  from absl import logging as absl_logging
except ImportError:
  absl_logging = None

try:
  from etils import epath
except ImportError:
  epath = None

try:
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
except ImportError:
  nnx = None
  traverse_util = None
  axis_rules = None
  jax = None
  jnp = None
  checkpointing = None
  train_state_nnx = None
  pyconfig = None
  max_logging = None
  maxtext_utils = None
  model_creation_utils = None
  train_utils = None

POD0_CONDITION = (
    '[[ "${MEGASCALE_SLICE_ID:-${JOB_INDEX:-0}}" == "0" && '
    '"${JOB_COMPLETION_INDEX:-0}" == "0" ]]'
)
REMOTE_UTIL_PATH = "/tmp/safetensors_util.py"


@functools.lru_cache(maxsize=1)
def _encoded_module_source() -> str:
  """Returns base64-encoded UTF-8 bytes of this Python module."""
  source_bytes = pathlib.Path(__file__).read_bytes()
  return base64.b64encode(source_bytes).decode("ascii")


def stage_remote_util_cmd(remote_path: str = REMOTE_UTIL_PATH) -> str:
  """Returns a shell command that materializes this module on a remote pod."""
  encoded = _encoded_module_source()
  return f"echo '{encoded}' | base64 -d > {shlex.quote(remote_path)}"


def pod0_barrier_cmd(
    pod0_cmds: tuple[str, ...],
    marker_gcs: str,
    poll_attempts: int = 720,
    poll_interval_sec: int = 10,
) -> str:
  """Runs `pod0_cmds` on slice 0 / worker 0 and syncs all pods via GCS."""
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


def build_prestage_cmds(
    hf_repo: str,
    gcs_cache_dir: str,
    remote_path: str = REMOTE_UTIL_PATH,
) -> tuple[str, ...]:
  """Builds shell commands to stage this util and cache HF shards to GCS."""
  return (
      stage_remote_util_cmd(remote_path),
      (
          f"python3 {shlex.quote(remote_path)} prestage "
          f"--hf_repo={shlex.quote(hf_repo)} "
          f"--gcs_cache_dir={shlex.quote(gcs_cache_dir)}"
      ),
  )


def build_compare_cmds(
    *,
    to_maxtext_ckpt: str,
    ref_step0_ckpt: str,
    dyn_step0_ckpt: str,
    ref_trained_ckpt: str,
    dyn_trained_ckpt: str,
    trained_step: int,
    ref_metrics: str,
    dyn_metrics: str,
    report_gcs: str,
    atol: float,
    maxtext_args: str,
    remote_path: str = REMOTE_UTIL_PATH,
) -> tuple[str, ...]:
  """Builds shell commands to stage this util and compare checkpoints."""
  return (
      stage_remote_util_cmd(remote_path),
      (
          f"python3 {shlex.quote(remote_path)} compare "
          f"--to_maxtext_ckpt={to_maxtext_ckpt} "
          f"--ref_step0_ckpt={ref_step0_ckpt} "
          f"--dyn_step0_ckpt={dyn_step0_ckpt} "
          f"--ref_trained_ckpt={ref_trained_ckpt} "
          f"--dyn_trained_ckpt={dyn_trained_ckpt} "
          f"--trained_step={trained_step} "
          f"--ref_metrics={ref_metrics} "
          f"--dyn_metrics={dyn_metrics} "
          f"--report={report_gcs} "
          f"--atol={atol} "
          f"{maxtext_args}"
      ),
  )


# ==============================================================================
# Subcommand 1: `prestage` (Hugging Face `*.safetensors` -> shared GCS cache)
# ==============================================================================


def parse_gcs_uri(uri: str) -> tuple[str, str]:
  """Splits a `gs://bucket/prefix` URI into `(bucket_name, prefix)`."""
  if not uri.startswith("gs://"):
    raise ValueError(f"Expected gs:// URI, got {uri}")
  bucket_name, _, prefix = uri[5:].partition("/")
  if not bucket_name or not prefix.strip("/"):
    raise ValueError(f"Expected gs://<bucket>/<prefix>, got {uri}")
  return bucket_name, prefix.rstrip("/")


def list_valid_safetensors(
    client: Any, bucket_name: str, cache_prefix: str
) -> set[str]:
  """Lists non-empty `.safetensors` filenames under `gs://bucket/prefix/`."""
  return {
      os.path.basename(blob.name)
      for blob in client.list_blobs(bucket_name, prefix=f"{cache_prefix}/")
      if blob.name.endswith(".safetensors") and (blob.size or 0) > 0
  }


def cache_one_file(
    fpath: str,
    bucket_name: str,
    cache_prefix: str,
    hf_token: str | None,
) -> str:
  """Streams a single `.safetensors` shard from Hugging Face Hub into GCS."""
  if storage is None or huggingface_hub is None:
    raise RuntimeError(
        "google-cloud-storage and huggingface_hub are required for prestage."
    )
  fname = os.path.basename(fpath)
  blob_name = f"{cache_prefix}/{fname}"
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  existing_blob = bucket.get_blob(blob_name, client=client)
  if existing_blob is not None and (existing_blob.size or 0) > 0:
    print(f"[Prestage] Cache hit: {fname}", flush=True)
    return fname

  time.sleep(random.uniform(0.0, 2.0))
  fs = huggingface_hub.HfFileSystem(token=hf_token)
  blob = bucket.blob(blob_name)
  max_retries = 8
  for attempt in range(max_retries):
    t0 = time.time()
    try:
      with fs.open(fpath, "rb") as remote_f:
        blob.chunk_size = 32 * 1024 * 1024  # 32 MiB
        blob.upload_from_file(remote_f, client=client, timeout=600)
      print(
          f"[Prestage] Uploaded {fname} in {time.time() - t0:.1f}s",
          flush=True,
      )
      return fname
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if attempt == max_retries - 1:
        raise
      backoff = min(30 * (attempt + 1), 120)
      print(
          f"[Prestage] Retry {attempt + 1}/{max_retries} for {fname} "
          f"after error ({exc}); sleeping {backoff}s...",
          flush=True,
      )
      time.sleep(backoff)
  return fname


def run_prestage(args: argparse.Namespace) -> None:
  """Atomically stages Hugging Face `.safetensors` shards into GCS."""
  if storage is None or gcs_exceptions is None or huggingface_hub is None:
    raise RuntimeError(
        "google-cloud-storage and huggingface_hub are required for prestage."
    )
  bucket_name, cache_prefix = parse_gcs_uri(args.gcs_cache_dir)
  parent_prefix = os.path.dirname(cache_prefix)
  safe_repo = args.hf_repo.replace("/", "_")
  complete_blob_name = f"{parent_prefix}/_locks/{safe_repo}.complete"
  lock_blob_name = f"{parent_prefix}/_locks/{safe_repo}.lock"

  client = storage.Client()
  bucket = client.bucket(bucket_name)
  complete_blob = bucket.blob(complete_blob_name)
  lock_blob = bucket.blob(lock_blob_name)

  wait_start = time.time()
  while True:
    if complete_blob.exists(client=client):
      cached = list_valid_safetensors(client, bucket_name, cache_prefix)
      if cached:
        print(
            f"[Prestage] Shared GCS cache ready ({len(cached)} shards): "
            f"{args.gcs_cache_dir}",
            flush=True,
        )
        return
      try:
        complete_blob.delete(client=client)
      except gcs_exceptions.NotFound:
        pass

    if time.time() - wait_start > args.wait_timeout_sec:
      raise TimeoutError(
          f"Timed out waiting for shared GCS cache {args.gcs_cache_dir}"
      )

    try:
      lock_payload = json.dumps({"pid": os.getpid(), "ts": time.time()})
      lock_blob.upload_from_string(
          lock_payload, if_generation_match=0, client=client
      )
      break
    except gcs_exceptions.PreconditionFailed:
      existing_lock = bucket.get_blob(lock_blob_name, client=client)
      if existing_lock is not None and existing_lock.time_created is not None:
        age = (
            datetime.datetime.now(datetime.timezone.utc)
            - existing_lock.time_created
        ).total_seconds()
        if age > args.lock_stale_sec:
          print(
              f"[Prestage] Removing stale lock ({age:.0f}s old)...",
              flush=True,
          )
          try:
            existing_lock.delete(
                if_generation_match=existing_lock.generation, client=client
            )
          except (
              gcs_exceptions.NotFound,
              gcs_exceptions.PreconditionFailed,
          ):
            pass
          continue
      print(
          f"[Prestage] Another job is staging {args.hf_repo} to "
          f"{args.gcs_cache_dir}; waiting 15s...",
          flush=True,
      )
      time.sleep(15)

  try:
    hf_token = os.environ.get("HF_TOKEN") or None
    fs = huggingface_hub.HfFileSystem(token=hf_token)
    files = sorted(fs.glob(f"{args.hf_repo}/*.safetensors"))
    if not files:
      raise RuntimeError(f"No .safetensors files found in hf://{args.hf_repo}")

    existing = list_valid_safetensors(client, bucket_name, cache_prefix)
    missing = [f for f in files if os.path.basename(f) not in existing]
    print(
        f"[Prestage] {args.hf_repo}: {len(files)} total shards, "
        f"{len(missing)} missing in {args.gcs_cache_dir}",
        flush=True,
    )

    if missing:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=args.max_workers
      ) as executor:
        futures = [
            executor.submit(
                cache_one_file, fpath, bucket_name, cache_prefix, hf_token
            )
            for fpath in missing
        ]
        for fut in concurrent.futures.as_completed(futures):
          fut.result()

    final_cached = list_valid_safetensors(client, bucket_name, cache_prefix)
    expected_names = {os.path.basename(f) for f in files}
    still_missing = sorted(expected_names - final_cached)
    if still_missing:
      raise RuntimeError(
          f"Incomplete GCS cache {args.gcs_cache_dir}; missing: "
          f"{still_missing}"
      )

    complete_blob.upload_from_string(
        json.dumps(
            {
                "hf_repo": args.hf_repo,
                "num_shards": len(files),
                "completed_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
        ),
        client=client,
    )
    print(
        f"[Prestage] Completed staging {len(files)} shards to "
        f"{args.gcs_cache_dir}",
        flush=True,
    )
  finally:
    try:
      lock_blob.delete(client=client)
    except gcs_exceptions.NotFound:
      pass


# ==============================================================================
# Subcommand 2: `compare` (Checkpoint & Metrics Comparison on TPU)
# ==============================================================================


def _log(message: str) -> None:
  """Logs via `max_logging` on TPU pods or falls back to stdout."""
  if max_logging is not None:
    max_logging.log(message)
  else:
    print(message, flush=True)


def _path_exists(path_str: str) -> bool:
  """Checks whether a local or GCS path exists."""
  if epath is not None:
    return epath.Path(path_str).exists()
  return pathlib.Path(path_str).exists()


def _read_text(path_str: str) -> str:
  """Reads UTF-8 text from a local or GCS path."""
  if epath is not None:
    return epath.Path(path_str).read_text()
  return pathlib.Path(path_str).read_text(encoding="utf-8")


def _write_text(path_str: str, content: str) -> None:
  """Writes UTF-8 text to a local or GCS path."""
  if epath is not None:
    epath.Path(path_str).write_text(content)
  else:
    pathlib.Path(path_str).write_text(content, encoding="utf-8")


def _abstract_train_state(config: Any, mesh: Any) -> Any:
  """Constructs the abstract NNX training state for checkpoint loading."""
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


def _load_params(
    config: Any, abstract_state: Any, path: str, layout: str = "orbax"
) -> Any:
  """Loads sharded model parameters from an Orbax or safetensors checkpoint."""
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


@functools.lru_cache(maxsize=1)
def _get_leaf_stats_fn() -> Callable[[Any, Any], tuple[Any, Any]]:
  """Returns a JIT-compiled leaf comparison function."""

  def _leaf_stats(a, b):
    a32 = a.astype(jnp.float32)
    b32 = b.astype(jnp.float32)
    equal = jnp.all((a == b) | (jnp.isnan(a) & jnp.isnan(b)))
    return jnp.max(jnp.abs(a32 - b32)), equal

  return jax.jit(_leaf_stats)


def clear_caches() -> None:
  """Clears internal LRU caches (used by unit tests)."""
  _get_leaf_stats_fn.cache_clear()
  _encoded_module_source.cache_clear()


def compare_checkpoint_pair(
    config: Any,
    abstract_state: Any,
    label: str,
    ref_path: str,
    dyn_path: str,
    atol: float,
    ref_layout: str = "orbax",
    dyn_layout: str = "orbax",
) -> dict[str, Any]:
  """Compares every parameter leaf between reference and dynamic checkpoints."""
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

  leaf_stats_fn = _get_leaf_stats_fn()
  leaves = []
  mismatches = []
  for key in sorted(rflat.keys() & dflat.keys()):
    a = dflat[key]
    b = rflat[key]
    name = "/".join(map(str, key))
    shape_ok = a.shape == b.shape
    dtype_ok = a.dtype == b.dtype
    sharding_ok = getattr(a, "sharding", None) == getattr(b, "sharding", None)
    entry: dict[str, Any] = {
        "name": name,
        "dyn_shape": list(a.shape),
        "ref_shape": list(b.shape),
        "dyn_dtype": str(a.dtype),
        "ref_dtype": str(b.dtype),
        "shape_match": shape_ok,
        "dtype_match": dtype_ok,
        "sharding_match": sharding_ok,
    }
    if shape_ok:
      max_abs_arr, exact_arr = leaf_stats_fn(a, b)
      max_abs = float(max_abs_arr)
      exact = bool(exact_arr)
      within_tol = exact if atol == 0.0 else (max_abs <= atol)
      entry["max_abs_diff"] = max_abs
      entry["exact_equal"] = exact
      entry["pass"] = bool(dtype_ok and sharding_ok and within_tol)
    else:
      entry["max_abs_diff"] = None
      entry["exact_equal"] = False
      entry["pass"] = False
    leaves.append(entry)
    if not entry["pass"]:
      mismatches.append(entry)

  max_diff = max(
      (x["max_abs_diff"] for x in leaves if x["max_abs_diff"] is not None),
      default=0.0,
  )
  passed = not only_ref and not only_dyn and not mismatches
  summary = {
      "label": label,
      "ref_path": ref_path,
      "dyn_path": dyn_path,
      "ref_layout": ref_layout,
      "dyn_layout": dyn_layout,
      "ref_load_seconds": round(ref_load_s, 3),
      "dyn_load_seconds": round(dyn_load_s, 3),
      "num_common_leaves": len(leaves),
      "only_in_ref": only_ref,
      "only_in_dyn": only_dyn,
      "num_mismatches": len(mismatches),
      "max_abs_diff_across_all_leaves": max_diff,
      "atol": atol,
      "result": "PASS" if passed else "FAIL",
      "mismatches": mismatches,
      "leaves": leaves,
  }
  _log(
      f"[{label}] {summary['result']}: common_leaves={len(leaves)}, "
      f"mismatches={len(mismatches)}, max_abs_diff={max_diff}, "
      f"ref_load={ref_load_s:.2f}s, dyn_load={dyn_load_s:.2f}s"
  )
  return summary


def load_losses(metrics_path: str) -> list[tuple[int, float]]:
  """Parses `(step, learning/loss)` pairs from a MaxText JSONL metrics file."""
  if not _path_exists(metrics_path):
    return []
  losses = []
  for idx, line in enumerate(_read_text(metrics_path).splitlines()):
    line = line.strip()
    if not line:
      continue
    row = json.loads(line)
    if "learning/loss" in row:
      losses.append((int(row.get("step", idx)), float(row["learning/loss"])))
  return losses


def compare_metrics(
    ref_metrics_path: str, dyn_metrics_path: str, atol: float
) -> dict[str, Any]:
  """Compares per-step training losses between reference and dynamic runs."""
  ref_losses = load_losses(ref_metrics_path)
  dyn_losses = load_losses(dyn_metrics_path)
  diffs = []
  for (s_ref, l_ref), (s_dyn, l_dyn) in zip(ref_losses, dyn_losses):
    abs_diff = abs(l_dyn - l_ref)
    tol = max(atol, 1e-5)
    diffs.append(
        {
            "ref_step": s_ref,
            "dyn_step": s_dyn,
            "ref_loss": l_ref,
            "dyn_loss": l_dyn,
            "abs_diff": abs_diff,
            "pass": (s_ref == s_dyn) and (abs_diff <= tol),
        }
    )
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
  _log(
      f"METRICS_COMPARE: {summary['result']} "
      f"(steps={len(diffs)}, max_loss_abs_diff={max_loss_diff})"
  )
  return summary


def run_compare(
    args: argparse.Namespace, model_rest: list[str]
) -> dict[str, Any]:
  """Runs 4-way checkpoint and metrics parity verification on TPU."""
  if jax is None or pyconfig is None or maxtext_utils is None:
    raise RuntimeError("JAX and MaxText are required for compare.")
  if absl_logging is not None:
    absl_logging.set_verbosity(absl_logging.INFO)

  config = pyconfig.initialize(["compare_ckpts"] + model_rest)
  mesh = maxtext_utils.get_mesh_from_config(config)
  abstract_state = _abstract_train_state(config, mesh)

  in_memory_summary = compare_checkpoint_pair(
      config,
      abstract_state,
      "in_memory_safetensors_vs_to_maxtext",
      args.to_maxtext_ckpt,
      config.load_parameters_path,
      args.atol,
      ref_layout="orbax",
      dyn_layout="safetensors_dynamic",
  )
  step0_summary = compare_checkpoint_pair(
      config,
      abstract_state,
      "step0_ckpt",
      args.ref_step0_ckpt,
      args.dyn_step0_ckpt,
      args.atol,
  )
  trained_label = f"step{args.trained_step}_trained_ckpt"
  trained_summary = compare_checkpoint_pair(
      config,
      abstract_state,
      trained_label,
      args.ref_trained_ckpt,
      args.dyn_trained_ckpt,
      args.atol,
  )
  metrics_summary = compare_metrics(
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
      "overall_result": "PASS" if overall_pass else "FAIL",
      "in_memory_safetensors_vs_to_maxtext": in_memory_summary,
      "step0_checkpoint_comparison": step0_summary,
      f"step{args.trained_step}_trained_checkpoint_comparison": (
          trained_summary
      ),
      "training_metrics_comparison": metrics_summary,
  }
  if jax.process_index() == 0:
    _write_text(args.report, json.dumps(report, indent=2, sort_keys=True))
  _log("OVERALL_COMPARE_RESULT: " + report["overall_result"])
  if not overall_pass:
    sys.exit(1)
  return report


def build_arg_parser() -> argparse.ArgumentParser:
  """Constructs the CLI argument parser for `prestage` and `compare`."""
  parser = argparse.ArgumentParser(
      description="MaxText safetensors_dynamic E2E helper utilities."
  )
  subparsers = parser.add_subparsers(dest="subcommand", required=True)

  prestage_parser = subparsers.add_parser(
      "prestage",
      help="Atomically stage Hugging Face *.safetensors shards into GCS.",
  )
  prestage_parser.add_argument("--hf_repo", required=True)
  prestage_parser.add_argument("--gcs_cache_dir", required=True)
  prestage_parser.add_argument("--max_workers", type=int, default=8)
  prestage_parser.add_argument("--lock_stale_sec", type=int, default=2700)
  prestage_parser.add_argument("--wait_timeout_sec", type=int, default=3600)

  compare_parser = subparsers.add_parser(
      "compare",
      help="Compare reference to_maxtext and safetensors_dynamic checkpoints.",
  )
  compare_parser.add_argument("--to_maxtext_ckpt", required=True)
  compare_parser.add_argument("--ref_step0_ckpt", required=True)
  compare_parser.add_argument("--dyn_step0_ckpt", required=True)
  compare_parser.add_argument("--ref_trained_ckpt", required=True)
  compare_parser.add_argument("--dyn_trained_ckpt", required=True)
  compare_parser.add_argument("--trained_step", type=int, default=1)
  compare_parser.add_argument("--ref_metrics", required=True)
  compare_parser.add_argument("--dyn_metrics", required=True)
  compare_parser.add_argument("--report", required=True)
  compare_parser.add_argument("--atol", type=float, default=0.0)

  return parser


def main(argv: list[str] | None = None) -> None:
  """Entrypoint when executed as a standalone script on TPU pods."""
  parser = build_arg_parser()
  args, extra_args = parser.parse_known_args(argv)
  if args.subcommand == "prestage":
    if extra_args:
      parser.error(f"Unrecognized arguments for prestage: {extra_args}")
    run_prestage(args)
  elif args.subcommand == "compare":
    run_compare(args, extra_args)


if __name__ == "__main__" and len(sys.argv) > 1:
  main()
