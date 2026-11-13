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

"""Unit tests for safetensors_util and maxtext_e2e_tpu_safetensors_dynamic."""

import contextlib
import datetime
import io
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from airflow import models
import jinja2
import numpy as np

from dags.common import vm_resource
from dags.multipod import maxtext_e2e_tpu_safetensors_dynamic as safetensors_dag
from dags.multipod.util import safetensors_util
from xlml.apis import test_config


class _FakePreconditionFailed(Exception):
  """Stub for `google.api_core.exceptions.PreconditionFailed`."""


class _FakeNotFound(Exception):
  """Stub for `google.api_core.exceptions.NotFound`."""


class _FakeBlob:
  """In-memory GCS blob fake supporting generation-matched locks."""

  def __init__(self, bucket: "_FakeBucket", name: str):
    self._bucket = bucket
    self.name = name
    self.chunk_size = None
    self.generation = 1
    self.time_created = None
    self._data: bytes | None = None

  @property
  def size(self) -> int:
    return 0 if self._data is None else len(self._data)

  def exists(self, client: object = None) -> bool:
    del client
    return self.name in self._bucket.blobs

  def upload_from_string(
      self,
      data: str | bytes,
      if_generation_match: int | None = None,
      client: object = None,
  ) -> None:
    del client
    if if_generation_match == 0 and self.name in self._bucket.blobs:
      raise _FakePreconditionFailed("Lock exists")
    raw = data.encode("utf-8") if isinstance(data, str) else data
    self._data = raw
    if self.time_created is None:
      self.time_created = datetime.datetime.now(datetime.timezone.utc)
    self._bucket.blobs[self.name] = self

  def upload_from_file(
      self,
      file_obj: io.BufferedReader,
      client: object = None,
      timeout: int | None = None,
  ) -> None:
    del client, timeout
    self._data = file_obj.read()
    self.time_created = datetime.datetime.now(datetime.timezone.utc)
    self._bucket.blobs[self.name] = self

  def download_as_bytes(
      self, start: int | None = None, end: int | None = None
  ) -> bytes:
    """Returns the blob bytes; `end` is inclusive, as in the GCS client."""
    data = self._data or b""
    return data[start : None if end is None else end + 1]

  def delete(
      self,
      if_generation_match: int | None = None,
      client: object = None,
  ) -> None:
    del client
    existing = self._bucket.blobs.get(self.name)
    if existing is None:
      raise _FakeNotFound(self.name)
    if (
        if_generation_match is not None
        and existing.generation != if_generation_match
    ):
      raise _FakePreconditionFailed("Generation mismatch")
    del self._bucket.blobs[self.name]


class _FakeBucket:
  """In-memory GCS bucket fake."""

  def __init__(self, name: str):
    self.name = name
    self.blobs: dict[str, _FakeBlob] = {}

  def blob(self, name: str) -> _FakeBlob:
    return self.blobs.get(name) or _FakeBlob(self, name)

  def get_blob(self, name: str, client: object = None) -> _FakeBlob | None:
    del client
    return self.blobs.get(name)


class _FakeStorageClient:
  """In-memory GCS client fake."""

  def __init__(self, buckets: dict[str, _FakeBucket]):
    self._buckets = buckets

  def bucket(self, name: str) -> _FakeBucket:
    return self._buckets.setdefault(name, _FakeBucket(name))

  def list_blobs(self, bucket_name: str, prefix: str = "") -> list[_FakeBlob]:
    bkt = self.bucket(bucket_name)
    return [b for k, b in bkt.blobs.items() if k.startswith(prefix)]


def _safetensors_bytes(dtypes: dict[str, str]) -> bytes:
  """Builds a minimal `.safetensors` payload: header length + JSON header."""
  header = {
      "__metadata__": {"format": "pt"},
      **{
          name: {"dtype": dtype, "shape": [1], "data_offsets": [0, 0]}
          for name, dtype in dtypes.items()
      },
  }
  raw = json.dumps(header).encode("utf-8")
  return struct.pack("<Q", len(raw)) + raw + b"\x00" * 16


def _flatten_dict(tree: dict, prefix: tuple[str, ...] = ()) -> dict:
  """Minimal `flax.traverse_util.flatten_dict` replacement for local tests."""
  flat = {}
  for key, val in tree.items():
    path = prefix + (str(key),)
    if isinstance(val, dict):
      flat.update(_flatten_dict(val, path))
    else:
      flat[path] = val
  return flat


class SafetensorsUtilTest(unittest.TestCase):
  """Local verification suite for `safetensors_util` and the E2E DAG."""

  def setUp(self):
    super().setUp()
    safetensors_util.clear_caches()

  def test_stage_remote_util_cmd_roundtrip_and_zero_arg_execution(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      staged_path = pathlib.Path(tmp_dir) / "safetensors_util.py"
      cmd = safetensors_util.stage_remote_util_cmd(str(staged_path))
      subprocess.run(["bash", "-c", cmd], check=True)

      original = pathlib.Path(safetensors_util.__file__).read_bytes()
      self.assertEqual(staged_path.read_bytes(), original)

      # Verify `scripts/dag-check.sh` behavior: running with no CLI args exits 0
      proc = subprocess.run(
          [sys.executable, str(staged_path)],
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stderr)

      # `require_remote_util_cmd` passes once the module is staged
      guard = safetensors_util.require_remote_util_cmd(str(staged_path))
      self.assertEqual(
          subprocess.run(["bash", "-c", guard], check=False).returncode, 0
      )

      # A corrupted payload fails loudly and leaves no file behind, and the
      # guard then reports that the module is not staged.
      broken_path = pathlib.Path(tmp_dir) / "broken_util.py"
      broken_cmd = safetensors_util.stage_remote_util_cmd(str(broken_path))
      encoded = safetensors_util._encoded_module_source()  # pylint: disable=protected-access
      broken_cmd = broken_cmd.replace(encoded, encoded[: len(encoded) // 2])
      broken = subprocess.run(
          ["bash", "-c", broken_cmd],
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(broken.returncode, 0)
      self.assertFalse(broken_path.exists())
      missing_guard = subprocess.run(
          [
              "bash",
              "-c",
              safetensors_util.require_remote_util_cmd(str(broken_path)),
          ],
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(missing_guard.returncode, 0)
      self.assertIn("is not staged", missing_guard.stderr)

  def test_workload_commands_fit_in_one_argument(self):
    # The whole workload command is passed to `gcluster` as one `--command=`
    # argument, which Linux caps at 128 KiB (MAX_ARG_STRLEN). Every task of
    # every spec must stay well below that and embed this module exactly once.
    encoded = safetensors_util._encoded_module_source()  # pylint: disable=protected-access
    checked = 0
    for dag_task in safetensors_dag.dag.tasks:
      if not dag_task.task_id.endswith(".launch_workload.run_workload"):
        continue
      run_cmds = str(dag_task.op_kwargs["run_cmds"])
      checked += 1
      self.assertLess(
          len(run_cmds.encode("utf-8")),
          100_000,
          msg=f"Workload command too long for {dag_task.task_id}",
      )
      self.assertEqual(
          run_cmds.count(encoded),
          1,
          msg=f"Module must be embedded exactly once in {dag_task.task_id}",
      )
    # Every phase-1 spec supports comparison, so each builds 3 workloads.
    self.assertEqual(checked, len(safetensors_dag.TEST_SPECS) * 3)
    self.assertGreater(checked, 0)

  def test_parse_gcs_uri(self):
    bucket, prefix = safetensors_util.parse_gcs_uri(
        "gs://ml-auto-solutions/output/hf_safetensors_cache/meta_llama/"
    )
    self.assertEqual(bucket, "ml-auto-solutions")
    self.assertEqual(prefix, "output/hf_safetensors_cache/meta_llama")

    with self.assertRaises(ValueError):
      safetensors_util.parse_gcs_uri("hf://meta-llama/Llama-3.1-70B")
    with self.assertRaises(ValueError):
      safetensors_util.parse_gcs_uri("gs://bucket-only")

  def test_prestage_cold_upload_warm_cache_hit_and_force_download(self):
    buckets: dict[str, _FakeBucket] = {}
    fake_storage = types.SimpleNamespace(
        Client=lambda: _FakeStorageClient(buckets)
    )
    fake_exc = types.SimpleNamespace(
        PreconditionFailed=_FakePreconditionFailed,
        NotFound=_FakeNotFound,
    )

    class _FakeHfFs:
      """Fake Hugging Face filesystem stub."""

      def __init__(self, token=None):
        self.token = token

      def glob(self, pattern):
        self.last_pattern = pattern
        return [
            "Qwen/Qwen3-0.6B/model-00001-of-00002.safetensors",
            "Qwen/Qwen3-0.6B/model-00002-of-00002.safetensors",
        ]

      def open(self, fpath, mode="rb"):
        del mode
        return io.BytesIO(f"weights:{fpath}".encode("utf-8"))

    fake_hf = types.SimpleNamespace(HfFileSystem=_FakeHfFs)

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        mock.patch.object(safetensors_util, "epath", None),
        mock.patch.object(safetensors_util, "storage", fake_storage),
        mock.patch.object(safetensors_util, "gcs_exceptions", fake_exc),
        mock.patch.object(safetensors_util, "huggingface_hub", fake_hf),
        mock.patch.object(safetensors_util.time, "sleep"),
    ):
      cold_timing = pathlib.Path(tmp_dir) / "cold_timing.json"
      warm_timing = pathlib.Path(tmp_dir) / "warm_timing.json"
      forced_timing = pathlib.Path(tmp_dir) / "forced_timing.json"

      safetensors_util.main([
          "prestage",
          "--hf_repo=Qwen/Qwen3-0.6B",
          "--gcs_cache_dir=gs://test-bkt/output/hf_cache/Qwen_Qwen3-0.6B",
          "--max_workers=2",
          f"--timing_out={cold_timing}",
      ])

      bkt = buckets["test-bkt"]
      self.assertIn(
          "output/hf_cache/Qwen_Qwen3-0.6B/model-00001-of-00002.safetensors",
          bkt.blobs,
      )
      self.assertIn(
          "output/hf_cache/Qwen_Qwen3-0.6B/model-00002-of-00002.safetensors",
          bkt.blobs,
      )
      self.assertIn(
          "output/hf_cache/_locks/Qwen_Qwen3-0.6B.complete", bkt.blobs
      )
      self.assertNotIn("output/hf_cache/_locks/Qwen_Qwen3-0.6B.lock", bkt.blobs)

      cold_meta = json.loads(cold_timing.read_text(encoding="utf-8"))
      self.assertFalse(cold_meta["cache_hit"])
      self.assertEqual(cold_meta["num_total_shards"], 2)
      self.assertEqual(cold_meta["num_downloaded_files"], 2)

      # Second run hits `.complete` fast path without calling HfFileSystem
      failing_hf = types.SimpleNamespace(
          HfFileSystem=mock.Mock(side_effect=AssertionError("Should hit cache"))
      )
      with mock.patch.object(safetensors_util, "huggingface_hub", failing_hf):
        safetensors_util.main([
            "prestage",
            "--hf_repo=Qwen/Qwen3-0.6B",
            "--gcs_cache_dir=gs://test-bkt/output/hf_cache/Qwen_Qwen3-0.6B",
            f"--timing_out={warm_timing}",
        ])
      warm_meta = json.loads(warm_timing.read_text(encoding="utf-8"))
      self.assertTrue(warm_meta["cache_hit"])
      self.assertEqual(warm_meta["num_total_shards"], 2)
      self.assertEqual(warm_meta["num_downloaded_files"], 0)

      # Third run with `--force_download` bypasses the warm cache, downloads
      # every shard again and reports a cold run.
      safetensors_util.main([
          "prestage",
          "--hf_repo=Qwen/Qwen3-0.6B",
          "--gcs_cache_dir=gs://test-bkt/output/hf_cache/Qwen_Qwen3-0.6B",
          "--force_download",
          f"--timing_out={forced_timing}",
      ])
      forced_meta = json.loads(forced_timing.read_text(encoding="utf-8"))
      self.assertFalse(forced_meta["cache_hit"])
      self.assertTrue(forced_meta["force_download"])
      self.assertEqual(forced_meta["num_downloaded_files"], 2)
      self.assertIn(
          "output/hf_cache/_locks/Qwen_Qwen3-0.6B.complete", bkt.blobs
      )

  def test_prestage_evicts_stale_lock(self):
    buckets: dict[str, _FakeBucket] = {}
    bkt = buckets.setdefault("test-bkt", _FakeBucket("test-bkt"))
    stale_lock = bkt.blob("output/hf_cache/_locks/Qwen_Qwen3-0.6B.lock")
    stale_lock.upload_from_string("old-lock")
    stale_lock.time_created = datetime.datetime.now(
        datetime.timezone.utc
    ) - datetime.timedelta(seconds=3000)

    fake_storage = types.SimpleNamespace(
        Client=lambda: _FakeStorageClient(buckets)
    )
    fake_exc = types.SimpleNamespace(
        PreconditionFailed=_FakePreconditionFailed,
        NotFound=_FakeNotFound,
    )
    fake_hf = types.SimpleNamespace(
        HfFileSystem=lambda token=None: types.SimpleNamespace(
            glob=lambda _: ["Qwen/Qwen3-0.6B/model.safetensors"],
            open=lambda fpath, mode="rb": io.BytesIO(b"tensor-bytes"),
        )
    )

    with (
        mock.patch.object(safetensors_util, "storage", fake_storage),
        mock.patch.object(safetensors_util, "gcs_exceptions", fake_exc),
        mock.patch.object(safetensors_util, "huggingface_hub", fake_hf),
        mock.patch.object(safetensors_util.time, "sleep"),
    ):
      safetensors_util.main([
          "prestage",
          "--hf_repo=Qwen/Qwen3-0.6B",
          "--gcs_cache_dir=gs://test-bkt/output/hf_cache/Qwen_Qwen3-0.6B",
          "--lock_stale_sec=60",
      ])
    self.assertIn(
        "output/hf_cache/Qwen_Qwen3-0.6B/model.safetensors", bkt.blobs
    )

  def test_summarize_stages_cli_and_log_parsers(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp = pathlib.Path(tmp_dir)
      prestage_json = tmp / "prestage_timing.json"
      dyn_log = tmp / "safetensors_dynamic.log"
      ref_log = tmp / "to_maxtext.log"
      summary_out = tmp / "stage_timings.json"

      prestage_json.write_text(
          json.dumps({
              "cache_hit": False,
              "hf_to_gcs_seconds": 41.25,
              "num_total_shards": 30,
              "num_downloaded_files": 30,
          }),
          encoding="utf-8",
      )
      dyn_log.write_text(
          "INFO: [1/3] Mappings derived in 0.04s\n"
          "INFO: [2/3] Distributed Sharded GCS load completed in 18.50s\n"
          "INFO: [3/3] CPU Transformations completed in 6.20s\n"
          "INFO: Total safetensors_dynamic duration: 24.74s\n",
          encoding="utf-8",
      )
      ref_log.write_text(
          "[to_maxtext] HF snapshot_download completed in 52.10s\n"
          "Elapse for transform: 0.50 min\n"
          "Overall Elapse: 3.50 min\n",
          encoding="utf-8",
      )

      with mock.patch.object(safetensors_util, "epath", None):
        safetensors_util.main([
            "summarize_stages",
            f"--dyn_log={dyn_log}",
            f"--prestage_timing={prestage_json}",
            f"--ref_log={ref_log}",
            f"--out={summary_out}",
            "--weight_dtype=bfloat16",
            "--to_maxtext_save_dtype=bfloat16",
            "--dyn_log_gcs=gs://bkt/logs/safetensors_dynamic.log",
            "--ref_log_gcs=gs://bkt/logs/to_maxtext.log",
        ])

      stage_summary = json.loads(summary_out.read_text(encoding="utf-8"))
      self.assertEqual(stage_summary["weight_dtype"], "bfloat16")
      self.assertEqual(stage_summary["to_maxtext_save_dtype"], "bfloat16")
      # No `--gcs_cache_dir`: the dtype block still records the spec dtypes
      self.assertEqual(
          stage_summary["dtype"],
          {
              "hf_checkpoint": None,
              "weight_dtype": "bfloat16",
              "to_maxtext_save_dtype": "bfloat16",
          },
      )
      dyn = stage_summary["safetensors_dynamic"]
      self.assertEqual(dyn["hf_to_gcs_seconds"], 41.25)
      self.assertFalse(dyn["gcs_cache_hit"])
      self.assertEqual(dyn["cache_status"], "COLD")
      self.assertEqual(dyn["num_total_shards"], 30)
      self.assertEqual(dyn["num_downloaded_files"], 30)
      self.assertEqual(dyn["derive_mappings_seconds"], 0.04)
      self.assertEqual(dyn["sharded_gcs_load_seconds"], 18.5)
      self.assertEqual(dyn["in_memory_transform_seconds"], 6.2)
      self.assertEqual(dyn["total_dynamic_load_seconds"], 24.74)

      to_mt = stage_summary["to_maxtext"]
      self.assertEqual(to_mt["hf_download_seconds"], 52.1)
      self.assertEqual(to_mt["offline_convert_transform_seconds"], 30.0)
      self.assertEqual(to_mt["offline_convert_save_orbax_seconds"], 180.0)
      self.assertEqual(to_mt["offline_convert_total_seconds"], 210.0)

  def test_hf_dtype_record_happy_path_and_exception_handling(self):
    buckets: dict[str, _FakeBucket] = {}
    fake_storage = types.SimpleNamespace(
        Client=lambda: _FakeStorageClient(buckets)
    )
    bkt = buckets.setdefault("test-bkt", _FakeBucket("test-bkt"))
    cache_dir = "gs://test-bkt/output/hf_cache/org_model"
    bkt.blob("output/hf_cache/org_model/a.safetensors").upload_from_string(
        _safetensors_bytes({"w1": "BF16", "w2": "BF16"})
    )
    bkt.blob("output/hf_cache/org_model/b.safetensors").upload_from_string(
        _safetensors_bytes({"w3": "BF16", "norm": "F32"})
    )
    bkt.blob("output/hf_cache/org_model/config.json").upload_from_string("{}")

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        mock.patch.object(safetensors_util, "epath", None),
        mock.patch.object(safetensors_util, "storage", fake_storage),
    ):
      # 1. Reads only the `.safetensors` headers and counts tensor dtypes
      self.assertEqual(
          safetensors_util.read_hf_dtypes(cache_dir), {"BF16": 3, "F32": 1}
      )
      self.assertEqual(
          safetensors_util.format_dtype_line(
              safetensors_util.build_dtype_record(
                  {"BF16": 723}, "bfloat16", "bfloat16"
              )
          ),
          "[dtype] hf=BF16 weight=bfloat16 save=bfloat16",
      )
      self.assertEqual(
          safetensors_util.format_dtype_line(
              safetensors_util.build_dtype_record(
                  {"BF16": 3, "F32": 1}, "bfloat16", "float32"
              )
          ),
          "[dtype] hf=BF16:3,F32:1 weight=bfloat16 save=float32",
      )

      # 2. `summarize_stages` writes the dtype block and logs the `[dtype]` line
      out = pathlib.Path(tmp_dir) / "stage_timings.json"
      stdout = io.StringIO()
      with contextlib.redirect_stdout(stdout):
        safetensors_util.main([
            "summarize_stages",
            f"--dyn_log={pathlib.Path(tmp_dir) / 'missing.log'}",
            f"--out={out}",
            f"--gcs_cache_dir={cache_dir}",
            "--weight_dtype=bfloat16",
            "--to_maxtext_save_dtype=float32",
        ])
      summary = json.loads(out.read_text(encoding="utf-8"))
      self.assertEqual(
          summary["dtype"],
          {
              "hf_checkpoint": {"BF16": 3, "F32": 1},
              "weight_dtype": "bfloat16",
              "to_maxtext_save_dtype": "float32",
          },
      )
      self.assertIn(
          "[dtype] hf=BF16:3,F32:1 weight=bfloat16 save=float32",
          stdout.getvalue(),
      )

      # 3. A rerun (the EXIT trap) reuses the recorded dtypes without reading
      # the shards again.
      with mock.patch.object(
          safetensors_util,
          "read_hf_dtypes",
          side_effect=AssertionError("should reuse the recorded dtypes"),
      ):
        safetensors_util.main([
            "summarize_stages",
            f"--dyn_log={pathlib.Path(tmp_dir) / 'missing.log'}",
            f"--out={out}",
            f"--gcs_cache_dir={cache_dir}",
            "--weight_dtype=bfloat16",
            "--to_maxtext_save_dtype=float32",
        ])
      self.assertEqual(
          json.loads(out.read_text(encoding="utf-8"))["dtype"]["hf_checkpoint"],
          {"BF16": 3, "F32": 1},
      )

      # 4. No shards, a truncated header and an implausible header length all
      # raise from the reader...
      with self.assertRaisesRegex(FileNotFoundError, "No .safetensors shards"):
        safetensors_util.read_hf_dtypes("gs://test-bkt/output/hf_cache/empty")
      bkt.blob("output/hf_cache/bad/x.safetensors").upload_from_string(
          struct.pack("<Q", 64) + b'{"w1": {"dtype": "BF'
      )
      with self.assertRaises(ValueError):
        safetensors_util.read_hf_dtypes("gs://test-bkt/output/hf_cache/bad")
      bkt.blob("output/hf_cache/huge/x.safetensors").upload_from_string(
          struct.pack("<Q", 2**40) + b"{}"
      )
      with self.assertRaisesRegex(ValueError, "Implausible safetensors header"):
        safetensors_util.read_hf_dtypes("gs://test-bkt/output/hf_cache/huge")

      # 5. ...but `summarize_stages` never fails because of it: the summary is
      # still written and records why the HF dtype is missing.
      out_bad = pathlib.Path(tmp_dir) / "stage_timings_bad.json"
      safetensors_util.main([
          "summarize_stages",
          f"--dyn_log={pathlib.Path(tmp_dir) / 'missing.log'}",
          f"--out={out_bad}",
          "--gcs_cache_dir=gs://test-bkt/output/hf_cache/bad",
          "--weight_dtype=bfloat16",
      ])
      bad = json.loads(out_bad.read_text(encoding="utf-8"))["dtype"]
      self.assertIsNone(bad["hf_checkpoint"])
      self.assertIn("hf_checkpoint_error", bad)
      self.assertEqual(bad["weight_dtype"], "bfloat16")
      self.assertEqual(bad["to_maxtext_save_dtype"], "bfloat16")

  def test_compare_metrics_pass_and_fail(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      ref_file = pathlib.Path(tmp_dir) / "ref.jsonl"
      dyn_ok = pathlib.Path(tmp_dir) / "dyn_ok.jsonl"
      dyn_bad = pathlib.Path(tmp_dir) / "dyn_bad.jsonl"

      ref_file.write_text(
          '{"step": 0, "learning/loss": 1.897}\n'
          '{"step": 1, "learning/loss": 2.207}\n',
          encoding="utf-8",
      )
      dyn_ok.write_text(
          '{"step": 0, "learning/loss": 1.897001}\n'
          '{"step": 1, "learning/loss": 2.207}\n',
          encoding="utf-8",
      )
      dyn_bad.write_text(
          '{"step": 0, "learning/loss": 1.897}\n'
          '{"step": 1, "learning/loss": 2.500}\n',
          encoding="utf-8",
      )

      with mock.patch.object(safetensors_util, "epath", None):
        ok_summary = safetensors_util.compare_metrics(
            str(ref_file), str(dyn_ok), atol=0.0
        )
        self.assertEqual(ok_summary["result"], "PASS")
        self.assertEqual(ok_summary["num_steps_ref"], 2)

        bad_summary = safetensors_util.compare_metrics(
            str(ref_file), str(dyn_bad), atol=0.0
        )
        self.assertEqual(bad_summary["result"], "FAIL")

  def test_compare_cli_end_to_end_with_stubs(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp = pathlib.Path(tmp_dir)
      ref_metrics = tmp / "ref_metrics.jsonl"
      dyn_metrics = tmp / "dyn_metrics.jsonl"
      stage_timings_path = tmp / "stage_timings.json"
      ref_log_path = tmp / "to_maxtext.log"
      report_path = tmp / "report.json"

      rows = '{"step": 0, "learning/loss": 1.5}\n'
      ref_metrics.write_text(rows, encoding="utf-8")
      dyn_metrics.write_text(rows, encoding="utf-8")
      stage_timings_path.write_text(
          json.dumps({
              "safetensors_dynamic": {
                  "hf_to_gcs_seconds": 0.01,
                  "total_dynamic_load_seconds": 21.5,
              },
              "to_maxtext": None,
          }),
          encoding="utf-8",
      )
      ref_log_path.write_text(
          "[to_maxtext] HF snapshot_download completed in 40.00s\n"
          "Elapse for transform: 1.00 min\n"
          "Overall Elapse: 4.00 min\n",
          encoding="utf-8",
      )

      ckpt_store = {
          "gs://bkt/to_mt": {"decoder": {"w": np.array([1.0, 2.0])}},
          "gs://bkt/hf_cache": {"decoder": {"w": np.array([1.0, 2.0])}},
          "gs://bkt/ref_0": {"decoder": {"w": np.array([1.0, 2.0])}},
          "gs://bkt/dyn_0": {"decoder": {"w": np.array([1.0, 2.0])}},
          "gs://bkt/ref_1": {"decoder": {"w": np.array([0.9, 1.8])}},
          "gs://bkt/dyn_1": {"decoder": {"w": np.array([0.9, 1.8])}},
      }

      fake_jax = types.SimpleNamespace(
          jit=lambda fn: fn,
          block_until_ready=lambda x: x,
          process_index=lambda: 0,
      )
      fake_pyconfig = types.SimpleNamespace(
          initialize=mock.Mock(
              return_value=types.SimpleNamespace(
                  model_name="llama3.1-70b",
                  weight_dtype="bfloat16",
                  load_parameters_path="gs://bkt/hf_cache",
                  logical_axis_rules=(),
                  checkpoint_storage_concurrent_gb=32,
                  checkpoint_storage_use_ocdbt=True,
                  checkpoint_storage_use_zarr3=True,
              )
          )
      )
      fake_ckpt = types.SimpleNamespace(
          load_state_if_possible=lambda *args, **kwargs: (
              None,
              ckpt_store[args[2]],
          )
      )
      fake_mt_utils = types.SimpleNamespace(
          get_mesh_from_config=lambda cfg: "fake_mesh",
          get_abstract_state=lambda *a, **kw: ("abstract_state", None, None),
      )
      fake_model_creation = types.SimpleNamespace(
          create_nnx_abstract_model=lambda cfg, mesh: (lambda: "m", "abs_m")
      )
      fake_train_utils = types.SimpleNamespace(
          create_training_optimizer=lambda cfg, m, mesh=None: (None, "tx")
      )
      fake_traverse = types.SimpleNamespace(flatten_dict=_flatten_dict)

      with (
          mock.patch.object(safetensors_util, "epath", None),
          mock.patch.object(safetensors_util, "jax", fake_jax),
          mock.patch.object(safetensors_util, "jnp", np),
          mock.patch.object(safetensors_util, "pyconfig", fake_pyconfig),
          mock.patch.object(safetensors_util, "checkpointing", fake_ckpt),
          mock.patch.object(safetensors_util, "maxtext_utils", fake_mt_utils),
          mock.patch.object(
              safetensors_util, "model_creation_utils", fake_model_creation
          ),
          mock.patch.object(safetensors_util, "train_utils", fake_train_utils),
          mock.patch.object(safetensors_util, "traverse_util", fake_traverse),
          mock.patch.object(
              safetensors_util, "axis_rules", contextlib.nullcontext
          ),
      ):
        safetensors_util.main([
            "compare",
            "--to_maxtext_ckpt=gs://bkt/to_mt",
            "--ref_step0_ckpt=gs://bkt/ref_0",
            "--dyn_step0_ckpt=gs://bkt/dyn_0",
            "--ref_trained_ckpt=gs://bkt/ref_1",
            "--dyn_trained_ckpt=gs://bkt/dyn_1",
            "--trained_step=1",
            f"--ref_metrics={ref_metrics}",
            f"--dyn_metrics={dyn_metrics}",
            f"--report={report_path}",
            "--atol=0.0",
            "--weight_dtype=bfloat16",
            "--to_maxtext_save_dtype=bfloat16",
            f"--stage_timings={stage_timings_path}",
            "--dyn_log_gcs=gs://bkt/logs/safetensors_dynamic.log",
            f"--ref_log_gcs={ref_log_path}",
            "--cmp_log_gcs=gs://bkt/logs/compare_ckpts.log",
            "src/maxtext/configs/base.yml",
            "model_name=llama3.1-70b",
        ])

        fake_pyconfig.initialize.assert_called_once_with([
            "compare_ckpts",
            "src/maxtext/configs/base.yml",
            "model_name=llama3.1-70b",
        ])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["overall_result"], "PASS")
        self.assertEqual(report["weight_dtype"], "bfloat16")
        self.assertEqual(report["dtype"]["weight_dtype"], "bfloat16")
        self.assertIn("hf_checkpoint", report["dtype"])
        self.assertEqual(report["to_maxtext_save_dtype"], "bfloat16")
        self.assertEqual(
            report["log_links"]["safetensors_dynamic_log"],
            "gs://bkt/logs/safetensors_dynamic.log",
        )
        self.assertEqual(
            report["stage_timings"]["to_maxtext"]["hf_download_seconds"], 40.0
        )
        self.assertIsNotNone(
            report["stage_timings"]["to_maxtext"]["orbax_restore_seconds"]
        )

        # Mutate one checkpoint leaf and verify CLI exits with status 1
        ckpt_store["gs://bkt/dyn_1"] = {"decoder": {"w": np.array([9.9, 1.8])}}
        with self.assertRaises(SystemExit) as ctx:
          safetensors_util.main([
              "compare",
              "--to_maxtext_ckpt=gs://bkt/to_mt",
              "--ref_step0_ckpt=gs://bkt/ref_0",
              "--dyn_step0_ckpt=gs://bkt/dyn_0",
              "--ref_trained_ckpt=gs://bkt/ref_1",
              "--dyn_trained_ckpt=gs://bkt/dyn_1",
              "--trained_step=1",
              f"--ref_metrics={ref_metrics}",
              f"--dyn_metrics={dyn_metrics}",
              f"--report={report_path}",
              "--atol=0.0",
              "src/maxtext/configs/base.yml",
          ])
        self.assertEqual(ctx.exception.code, 1)

  def test_tpu7x_device_name_and_dtype_spec(self):
    # TPU7x uses the `tpu7x-<cores>` device name; other TPU versions keep
    # `v<version>-<cores>`.
    v7x = test_config.Tpu(version=vm_resource.TpuVersion.V7X, cores=32)
    self.assertEqual(v7x.name, "tpu7x-32")
    v5p = test_config.Tpu(version=vm_resource.TpuVersion.V5P, cores=32)
    self.assertEqual(v5p.name, "v5p-32")
    self.assertEqual(
        vm_resource.MachineVersion.TPU7X_STAND_4T.value, "tpu7x-standard-4t"
    )

    spec_default = safetensors_dag.SafetensorsComparisonSpec(
        model_name="llama3.1-70b",
        hf_repo="meta-llama/Llama-3.1-70B",
        weight_dtype="bfloat16",
    )
    self.assertEqual(spec_default.effective_to_maxtext_save_dtype, "bfloat16")

    spec_override = safetensors_dag.SafetensorsComparisonSpec(
        model_name="llama3.1-70b",
        hf_repo="meta-llama/Llama-3.1-70B",
        weight_dtype="bfloat16",
        to_maxtext_save_dtype="float32",
    )
    self.assertEqual(spec_override.effective_to_maxtext_save_dtype, "float32")

  def test_pod0_detection_with_two_containers_per_pod(self):
    # TPU7x pods run two containers that share MEGASCALE_SLICE_ID and
    # JOB_COMPLETION_INDEX; only the one with TPU_WORKER_ID=0 is "pod 0".
    cases = [
        ({"MEGASCALE_SLICE_ID": "0", "JOB_COMPLETION_INDEX": "0"}, True),
        (
            {
                "MEGASCALE_SLICE_ID": "0",
                "JOB_COMPLETION_INDEX": "0",
                "TPU_WORKER_ID": "0",
            },
            True,
        ),
        (
            {
                "MEGASCALE_SLICE_ID": "0",
                "JOB_COMPLETION_INDEX": "0",
                "TPU_WORKER_ID": "1",
            },
            False,
        ),
        (
            {
                "MEGASCALE_SLICE_ID": "1",
                "JOB_COMPLETION_INDEX": "0",
                "TPU_WORKER_ID": "0",
            },
            False,
        ),
        ({"MEGASCALE_SLICE_ID": "0", "JOB_COMPLETION_INDEX": "3"}, False),
    ]
    base_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "MEGASCALE_SLICE_ID",
            "JOB_INDEX",
            "JOB_COMPLETION_INDEX",
            "TPU_WORKER_ID",
        )
    }
    for env, expected in cases:
      with self.subTest(env=env):
        proc = subprocess.run(
            ["bash", "-c", safetensors_util.POD0_CONDITION],
            env={**base_env, **env},
            check=False,
        )
        self.assertEqual(proc.returncode == 0, expected)

  def test_resolve_comparison_enabled(self):
    # `resolve_comparison_enabled` requires a valid `run_mode`
    self.assertTrue(
        safetensors_dag.resolve_comparison_enabled(
            params={"run_mode": "compare"}
        )
    )
    self.assertFalse(
        safetensors_dag.resolve_comparison_enabled(
            params={"run_mode": "e2e_only"}
        )
    )
    # `dag_run.conf` overrides `params`
    self.assertTrue(
        safetensors_dag.resolve_comparison_enabled(
            params={"run_mode": "e2e_only"},
            conf={"run_mode": "compare"},
        )
    )
    with self.assertRaises(ValueError):
      safetensors_dag.resolve_comparison_enabled(params={})
    with self.assertRaises(ValueError):
      safetensors_dag.resolve_comparison_enabled(
          params={"run_mode": "invalid_mode"}
      )

  def test_dag_structure_jinja_rendering_and_bash_syntax(self):
    # Verify default DAG loads cleanly with all specs and unique group_ids
    dag = safetensors_dag.dag
    self.assertEqual(dag.dag_id, "maxtext_e2e_tpu_safetensors_dynamic")
    self.assertIn("run_mode", dag.params)
    self.assertIn("force_hf_download", dag.params)
    self.assertFalse(dag.params["force_hf_download"])
    self.assertNotIn("models", dag.params)
    self.assertEqual(safetensors_dag.VALID_RUN_MODES, ("e2e_only", "compare"))
    spec_group_ids = [s.group_id for s in safetensors_dag.TEST_SPECS]
    self.assertEqual(
        len(spec_group_ids),
        len(set(spec_group_ids)),
        msg="Duplicate group_id in TEST_SPECS",
    )
    group_ids = set(dag.task_group.children.keys())
    self.assertEqual(len(group_ids), len(safetensors_dag.TEST_SPECS))
    # Phase 1: four models, each with 1 and 2 slices on the TPU7x cluster
    self.assertEqual(
        group_ids,
        {
            "qwen3-0_6b-v7x-8",
            "qwen3-0_6b-2x-v7x-8",
            "gemma2-2b-v7x-8",
            "gemma2-2b-2x-v7x-8",
            "qwen3-30b-a3b-v7x-8",
            "qwen3-30b-a3b-2x-v7x-8",
            "llama3_1-70b-v7x-32",
            "llama3_1-70b-2x-v7x-32",
        },
    )
    for spec in safetensors_dag.TEST_SPECS:
      self.assertEqual(spec.cluster.name, "bodaborg-tpu7x-nap")
      self.assertEqual(spec.cluster.namespace, "default")
      self.assertEqual(spec.cluster.queue, "multislice-queue")
      self.assertIn(spec.num_slices, (1, 2))
      self.assertEqual(
          "dcn_data_parallelism=2" in spec.extra_maxtext_flags,
          spec.num_slices == 2,
      )
    # gemma2-2b exercises the dtype record: F32 checkpoint, bfloat16 weights
    gemma = next(
        s for s in safetensors_dag.TEST_SPECS if s.model_name == "gemma2-2b"
    )
    self.assertEqual(gemma.weight_dtype, "bfloat16")

    # Build TaskGroups covering:
    # 1. `comparison_supported=True` (3 tasks built, toggled by `run_mode`)
    # 2. `comparison_supported=False` (1 task built, for >200B models that
    #    cannot run single-host `to_maxtext`)
    captured_cmds: list[tuple[str, tuple[str, ...]]] = []
    real_get_gke_config = safetensors_dag.gke_config.get_gke_config

    def _spy_get_gke_config(**kwargs):
      captured_cmds.append(
          (kwargs["test_name"], tuple(kwargs["run_model_cmds"]))
      )
      return real_get_gke_config(**kwargs)

    with mock.patch.object(
        safetensors_dag.gke_config,
        "get_gke_config",
        side_effect=_spy_get_gke_config,
    ):
      with models.DAG(
          dag_id="test_safetensors_full_and_standalone",
          schedule=None,
          start_date=datetime.datetime(2026, 6, 10),
      ):
        safetensors_dag.create_safetensors_comparison_task_group(
            safetensors_dag.SafetensorsComparisonSpec(
                model_name="llama3.1-70b",
                hf_repo="meta-llama/Llama-3.1-70B",
                core_count=32,
                num_slices=1,
                comparison_supported=True,
            )
        )
        safetensors_dag.create_safetensors_comparison_task_group(
            safetensors_dag.SafetensorsComparisonSpec(
                model_name="qwen3-480b-a35b",
                hf_repo="Qwen/Qwen3-480B-A35B",
                core_count=64,
                num_slices=2,
                comparison_supported=False,
                extra_maxtext_flags=("dcn_data_parallelism=2",),
            )
        )

    test_names = [name for name, _ in captured_cmds]
    self.assertEqual(
        test_names, ["safetensors-dyn", "to-mt", "cmp-ckpt", "safetensors-dyn"]
    )

    # Verify runtime Jinja rendering of `safetensors-dyn` flags:
    dyn_70b_cmds = "\n".join(captured_cmds[0][1])
    rendered_e2e = jinja2.Template(dyn_70b_cmds).render(
        params={
            "run_name": "",
            "run_mode": "e2e_only",
        },
        ts_nodash="20261027T060000",
    )
    self.assertIn(
        "save_checkpoint_on_start=false save_checkpoint_on_completion=false",
        rendered_e2e,
    )
    self.assertIn("safetensors-dynamic-20261027T060000", rendered_e2e)
    self.assertIn("safetensors_dynamic_prep.done", rendered_e2e)
    self.assertNotIn("sd_dyn_prep.done", rendered_e2e)
    self.assertNotIn("--force_download", rendered_e2e)
    self.assertIn("stage_timings.json", rendered_e2e)
    self.assertIn("trap ", rendered_e2e)

    rendered_compare = jinja2.Template(dyn_70b_cmds).render(
        params={
            "run_name": "unit-run-01",
            "run_mode": "compare",
            "force_hf_download": True,
        },
        ts_nodash="20261027T060000",
    )
    self.assertIn(
        "checkpoint_period=1 save_checkpoint_on_start=true",
        rendered_compare,
    )
    self.assertIn("--force_download", rendered_compare)
    self.assertIn("--gcs_cache_dir=gs://", rendered_compare)

    # Render every command in every `run_mode` and check `bash -n`
    for run_mode in safetensors_dag.VALID_RUN_MODES:
      for test_name, cmds in captured_cmds:
        rendered = [
            jinja2.Template(c).render(
                params={
                    "run_name": "unit-run-01",
                    "run_mode": run_mode,
                    "force_hf_download": run_mode == "compare",
                },
                ts_nodash="20261027T060000",
            )
            for c in cmds
        ]
        script = "\n".join(rendered)
        self.assertNotIn("{{", script, msg=f"Unrendered Jinja in {test_name}")
        self.assertNotIn("{%", script, msg=f"Unrendered Jinja in {test_name}")
        proc = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"Invalid bash syntax in {test_name}: {proc.stderr}",
        )

  def test_exception_handling_across_prestage_barrier_summarize_and_compare(
      self,
  ):
    buckets: dict[str, _FakeBucket] = {}
    fake_storage = types.SimpleNamespace(
        Client=lambda: _FakeStorageClient(buckets)
    )
    fake_exc = types.SimpleNamespace(
        PreconditionFailed=_FakePreconditionFailed,
        NotFound=_FakeNotFound,
    )

    class _EmptyHfFs:
      """Fake Hugging Face filesystem stub returning no shards."""

      def __init__(self, token=None):
        self.token = token

      def glob(self, pattern):
        del pattern
        return []

    empty_hf = types.SimpleNamespace(HfFileSystem=_EmptyHfFs)
    # 1. `run_prestage` when repo has no `.safetensors` files records `error` in
    # `timing_out` and cleans up the `.lock` blob in `finally:`.
    with tempfile.TemporaryDirectory() as tmpdir:
      timing_file = pathlib.Path(tmpdir) / "prestage_err.json"
      with (
          mock.patch.object(safetensors_util, "epath", None),
          mock.patch.object(safetensors_util, "storage", fake_storage),
          mock.patch.object(safetensors_util, "gcs_exceptions", fake_exc),
          mock.patch.object(safetensors_util, "huggingface_hub", empty_hf),
      ):
        with self.assertRaisesRegex(RuntimeError, "No .safetensors files"):
          safetensors_util.main([
              "prestage",
              "--hf_repo=org/empty-model",
              "--gcs_cache_dir=gs://test-bkt/output/hf_cache/org_empty-model",
              f"--timing_out={timing_file}",
          ])
      err_timing = json.loads(timing_file.read_text(encoding="utf-8"))
      self.assertFalse(err_timing["cache_hit"])
      self.assertIn("No .safetensors files", err_timing["error"])
      bucket = _FakeStorageClient(buckets).bucket("test-bkt")
      self.assertFalse(
          bucket.blob("output/hf_cache/_locks/org_empty-model.lock").exists()
      )

      # 2. `run_summarize_stages` surfaces `prestage_error` and tolerates
      # corrupted JSON in `--prestage_timing` and missing log files.
      out_err_summary = pathlib.Path(tmpdir) / "stage_timings_err.json"
      safetensors_util.main([
          "summarize_stages",
          f"--dyn_log={pathlib.Path(tmpdir) / 'missing_dyn.log'}",
          f"--prestage_timing={timing_file}",
          f"--out={out_err_summary}",
      ])
      summary_payload = json.loads(out_err_summary.read_text(encoding="utf-8"))
      self.assertIn(
          "No .safetensors files",
          summary_payload["safetensors_dynamic"]["prestage_error"],
      )

      corrupt_timing = pathlib.Path(tmpdir) / "corrupt.json"
      corrupt_timing.write_text("{not valid json", encoding="utf-8")
      out_corrupt_summary = pathlib.Path(tmpdir) / "stage_timings_corrupt.json"
      safetensors_util.main([
          "summarize_stages",
          f"--dyn_log={pathlib.Path(tmpdir) / 'missing_dyn.log'}",
          f"--prestage_timing={corrupt_timing}",
          f"--out={out_corrupt_summary}",
      ])
      self.assertTrue(out_corrupt_summary.exists())

      # 3. `load_losses` skips malformed/truncated JSONL lines without crashing.
      bad_metrics = pathlib.Path(tmpdir) / "bad_metrics.txt"
      bad_metrics.write_text(
          '{"step": 0, "learning/loss": 2.5}\n'
          "{truncated_json_line\n"
          '{"step": 1, "learning/loss": 2.1}\n',
          encoding="utf-8",
      )
      losses = safetensors_util.load_losses(str(bad_metrics))
      self.assertEqual(losses, [(0, 2.5), (1, 2.1)])

      # 4. `pod0_barrier_cmd` writes `.failed` marker on Pod 0 error and causes
      # non-Pod-0 workers to abort immediately instead of polling forever.
      bin_dir = pathlib.Path(tmpdir) / "bin"
      bin_dir.mkdir()
      gcs_dir = pathlib.Path(tmpdir) / "gcs_markers"
      gcs_dir.mkdir()
      gsutil_script = bin_dir / "gsutil"
      gsutil_script.write_text(
          f"""#!/bin/bash
if [ "$1" = "-q" ]; then shift; fi
cmd="$1"; shift
if [ "$cmd" = "cp" ]; then
  src="$1"; dst="${{2/#gs:\\/\\/bkt/{gcs_dir}}}"
  mkdir -p "$(dirname "$dst")"
  if [ "$src" = "-" ]; then cat > "$dst"; else cp "$src" "$dst"; fi
elif [ "$cmd" = "rm" ]; then
  if [ "$1" = "-f" ]; then shift; fi
  for p in "$@"; do rm -f "${{p/#gs:\\/\\/bkt/{gcs_dir}}}"; done
elif [ "$cmd" = "stat" ]; then
  test -f "${{1/#gs:\\/\\/bkt/{gcs_dir}}}"
fi
""",
          encoding="utf-8",
      )
      gsutil_script.chmod(0o755)
      barrier_cmd = safetensors_util.pod0_barrier_cmd(
          ("exit 23",),
          "gs://bkt/prep.done",
          poll_attempts=5,
          poll_interval_sec=0,
      )
      env_pod0 = {
          **os.environ,
          "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
          "MEGASCALE_SLICE_ID": "0",
          "JOB_COMPLETION_INDEX": "0",
      }
      res_pod0 = subprocess.run(
          ["bash", "-c", barrier_cmd], env=env_pod0, check=False
      )
      self.assertEqual(res_pod0.returncode, 23)
      self.assertTrue((gcs_dir / "prep.done.failed").exists())

      env_worker1 = {
          **env_pod0,
          "MEGASCALE_SLICE_ID": "1",
      }
      res_worker1 = subprocess.run(
          ["bash", "-c", barrier_cmd],
          env=env_worker1,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(res_worker1.returncode, 1)
      self.assertIn("Pod 0 prep failed", res_worker1.stderr)

      # 5. `compare_checkpoint_pair` catches checkpoint load exceptions and
      # returns a structured FAIL dict with `error`.
      with mock.patch.object(
          safetensors_util,
          "_load_params",
          side_effect=FileNotFoundError("Missing Orbax checkpoint"),
      ):
        fail_pair = safetensors_util.compare_checkpoint_pair(
            config=types.SimpleNamespace(),
            abstract_state=object(),
            label="step0_ckpt",
            ref_path="gs://bkt/missing_ref",
            dyn_path="gs://bkt/missing_dyn",
            atol=1e-2,
        )
      self.assertEqual(fail_pair["result"], "FAIL")
      self.assertIn("Missing Orbax checkpoint", fail_pair["error"])


if __name__ == "__main__":
  unittest.main()
