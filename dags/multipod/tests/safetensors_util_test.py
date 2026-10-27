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
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from airflow import models
import jinja2
import numpy as np

from dags.multipod import maxtext_e2e_tpu_safetensors_dynamic as safetensors_dag
from dags.multipod.util import safetensors_util


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

  def test_prestage_cold_upload_and_warm_cache_hit(self):
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
        mock.patch.object(safetensors_util, "storage", fake_storage),
        mock.patch.object(safetensors_util, "gcs_exceptions", fake_exc),
        mock.patch.object(safetensors_util, "huggingface_hub", fake_hf),
        mock.patch.object(safetensors_util.time, "sleep"),
    ):
      safetensors_util.main(
          [
              "prestage",
              "--hf_repo=Qwen/Qwen3-0.6B",
              "--gcs_cache_dir=gs://test-bkt/output/hf_cache/Qwen_Qwen3-0.6B",
              "--max_workers=2",
          ]
      )

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

      # Second run hits `.complete` fast path without calling HfFileSystem
      failing_hf = types.SimpleNamespace(
          HfFileSystem=mock.Mock(side_effect=AssertionError("Should hit cache"))
      )
      with mock.patch.object(safetensors_util, "huggingface_hub", failing_hf):
        safetensors_util.main(
            [
                "prestage",
                "--hf_repo=Qwen/Qwen3-0.6B",
                "--gcs_cache_dir=gs://test-bkt/output/hf_cache/Qwen_Qwen3-0.6B",
            ]
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
      safetensors_util.main(
          [
              "prestage",
              "--hf_repo=Qwen/Qwen3-0.6B",
              "--gcs_cache_dir=gs://test-bkt/output/hf_cache/Qwen_Qwen3-0.6B",
              "--lock_stale_sec=60",
          ]
      )
    self.assertIn(
        "output/hf_cache/Qwen_Qwen3-0.6B/model.safetensors", bkt.blobs
    )

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
      report_path = tmp / "report.json"

      rows = '{"step": 0, "learning/loss": 1.5}\n'
      ref_metrics.write_text(rows, encoding="utf-8")
      dyn_metrics.write_text(rows, encoding="utf-8")

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
        safetensors_util.main(
            [
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
                "model_name=llama3.1-70b",
            ]
        )

        fake_pyconfig.initialize.assert_called_once_with(
            [
                "compare_ckpts",
                "src/maxtext/configs/base.yml",
                "model_name=llama3.1-70b",
            ]
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["overall_result"], "PASS")

        # Mutate one checkpoint leaf and verify CLI exits with status 1
        ckpt_store["gs://bkt/dyn_1"] = {"decoder": {"w": np.array([9.9, 1.8])}}
        with self.assertRaises(SystemExit) as ctx:
          safetensors_util.main(
              [
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
              ]
          )
        self.assertEqual(ctx.exception.code, 1)

  def test_dag_structure_jinja_rendering_and_bash_syntax(self):
    # Verify default DAG loads cleanly with expected 1-slice & 2-slice groups
    dag = safetensors_dag.dag
    self.assertEqual(dag.dag_id, "maxtext_e2e_tpu_safetensors_dynamic")
    group_ids = set(dag.task_group.children.keys())
    self.assertIn("llama3_1-70b-v5p-32", group_ids)
    self.assertIn("llama3_1-70b-2x-v5p-32", group_ids)

    # Also build a full 3-task comparison TaskGroup and capture all commands
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
                model_name="qwen3-0.6b",
                hf_repo="Qwen/Qwen3-0.6B",
                core_count=8,
                num_slices=1,
                enable_comparison=True,
                prestage_hf_to_gcs=True,
            )
        )
        safetensors_dag.create_safetensors_comparison_task_group(
            safetensors_dag.SafetensorsComparisonSpec(
                model_name="llama3.1-70b",
                hf_repo="meta-llama/Llama-3.1-70B",
                core_count=32,
                num_slices=2,
                enable_comparison=False,
                prestage_hf_to_gcs=True,
                extra_maxtext_flags=("dcn_data_parallelism=2",),
            )
        )

    test_names = [name for name, _ in captured_cmds]
    self.assertEqual(
        test_names, ["safetensors-dyn", "to-mt", "cmp-ckpt", "safetensors-dyn"]
    )

    # Render every shell command with Jinja2 and check bash syntax via `bash -n`
    for test_name, cmds in captured_cmds:
      rendered = [
          jinja2.Template(c).render(
              params={"run_name": "unit-run-01"},
              ts_nodash="20261027T060000",
          )
          for c in cmds
      ]
      script = "\n".join(rendered)
      self.assertNotIn("{{", script, msg=f"Unrendered Jinja in {test_name}")
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


if __name__ == "__main__":
  unittest.main()
