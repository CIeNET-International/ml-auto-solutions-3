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

"""Unit tests for MaxText Pathways Elastic In-Memory Snapshot DAGs."""

import ast
import importlib
import pathlib
import subprocess
import tempfile
import unittest

from airflow.exceptions import AirflowFailException
from dags.common.scheduling_helper import scheduling_helper
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from dags.maxtext_pathways.configs import snapshot_utils
from dags.maxtext_pathways.configs import utils as pw_utils

_SAMPLE_CTK_RUNNER = """\
import os
MAXTEXT_PKG_DIR = "/deps/src/maxtext"


def compute_chip_suffix(size: int) -> int:
  chip_suffix = size if size in (1, 8) else 4
  return chip_suffix


def build_user_command(wl_config, hlo_dump="", config_tuning_params=""):
  command = " ".join(
      [
          "export JAX_PLATFORMS=proxy &&",
          "export MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets &&"
          f'{hlo_dump} python3 -m maxtext.trainers.pre_train.train {os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")}',
          f"{config_tuning_params}",
          f"steps={wl_config.num_steps}",
      ]
  )
  return command
"""

_SAMPLE_TRILLIUM_MODEL_CONFIGS = """\
trillium_model_dict = {}


class MaxTextModel:

  def __init__(self, **kwargs):
    self.kwargs = kwargs


def _add_to_model_dictionary(d, m):
  d["default_basic_1"] = m
  return m


default_basic_1 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="default-basic-1",
        model_type="default",
        tuning_params={
            "per_device_batch_size": 1,
            "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic",
            "reuse_example_batch": 1,
            "enable_checkpointing": False,
            "profiler": "xplane",
        },
        xla_flags="",
    ),
)
"""


def _extract_sed_commands(full_cmd: str) -> list[str]:
  """Extracts only the sed patch commands from a chained KPO shell string."""
  return [
      part.strip()
      for part in full_cmd.split(" && ")
      if part.strip().startswith("sed ")
  ]


class PwMcjaxElasticSnapshotDagTest(unittest.TestCase):
  """Tests DAG structure, command generation, and log verification."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.snap_mod = importlib.import_module(
        "dags.maxtext_pathways.pw_mcjax_elastic_snapshot"
    )
    cls.pw_mcjax_elastic = importlib.import_module(
        "dags.maxtext_pathways.pw_mcjax_elastic"
    )
    cls.pw_mcjax_elastic_goodput = importlib.import_module(
        "dags.maxtext_pathways.pw_mcjax_elastic_goodput"
    )

  def _run_sed_patches_and_parse(self, full_cmd: str) -> tuple[str, str]:
    sed_cmds = _extract_sed_commands(full_cmd)
    self.assertGreaterEqual(len(sed_cmds), 2)
    with tempfile.TemporaryDirectory() as tmpdir:
      bench_dir = pathlib.Path(tmpdir) / "benchmarks"
      bench_dir.mkdir(parents=True, exist_ok=True)
      runner_file = bench_dir / "maxtext_ctk_runner.py"
      configs_file = bench_dir / "maxtext_trillium_model_configs.py"
      runner_file.write_text(_SAMPLE_CTK_RUNNER, encoding="utf-8")
      configs_file.write_text(_SAMPLE_TRILLIUM_MODEL_CONFIGS, encoding="utf-8")

      subprocess.run(
          ["bash", "-euo", "pipefail", "-c", " && ".join(sed_cmds)],
          cwd=tmpdir,
          check=True,
      )
      patched_runner = runner_file.read_text(encoding="utf-8")
      patched_configs = configs_file.read_text(encoding="utf-8")
      ast.parse(patched_runner)
      ast.parse(patched_configs)
      return patched_runner, patched_configs

  def test_dags_registered_and_have_teardown(self):
    expected = [
        (
            "pw_elastic_snapshot_replica_resize",
            self.snap_mod.dag_snapshot_replica_resize,
        ),
        (
            "pw_elastic_snapshot_pause_resume",
            self.snap_mod.dag_snapshot_pause_resume,
        ),
        (
            "pw_elastic_snapshot_ckpt_fallback",
            self.snap_mod.dag_snapshot_ckpt_fallback,
        ),
    ]
    pw_registered = scheduling_helper.REGISTERED_DAGS[
        scheduling_helper.PW_MCJAX_CLUSTER.name
    ]
    for expected_id, dag in expected:
      self.assertEqual(dag.dag_id, expected_id)
      self.assertIn(expected_id, pw_registered)
      cleanup_task = dag.get_task("clean_up_recipe")
      verify_task = dag.get_task("verify_snapshot_logs")
      self.assertTrue(cleanup_task.is_teardown)
      self.assertIn("clean_up_recipe", verify_task.downstream_task_ids)

  def test_fault_injection_task_topologies(self):
    rr_tasks = {
        t.task_id for t in self.snap_mod.dag_snapshot_replica_resize.tasks
    }
    self.assertTrue(
        {
            "snapshot_fault_injection.wait_and_interrupt_worker",
            "snapshot_fault_injection.wait_for_slice_down_restore",
            "snapshot_fault_injection.wait_for_scale_up_snapshot",
            "snapshot_fault_injection.wait_for_scale_up_restore",
            "verify_snapshot_logs",
            "clean_up_recipe",
        }.issubset(rr_tasks)
    )

    pr_tasks = {
        t.task_id for t in self.snap_mod.dag_snapshot_pause_resume.tasks
    }
    self.assertTrue(
        {
            "snapshot_fault_injection.wait_and_interrupt_worker",
            "snapshot_fault_injection.wait_for_slices_active",
            "snapshot_fault_injection.wait_for_snapshot_restore",
        }.issubset(pr_tasks)
    )

    fb_tasks = {
        t.task_id for t in self.snap_mod.dag_snapshot_ckpt_fallback.tasks
    }
    self.assertTrue(
        {
            "snapshot_fault_injection.wait_and_interrupt_worker",
            "snapshot_fault_injection.wait_for_ckpt_fallback",
        }.issubset(fb_tasks)
    )
    for d in (
        self.snap_mod.dag_snapshot_replica_resize,
        self.snap_mod.dag_snapshot_pause_resume,
        self.snap_mod.dag_snapshot_ckpt_fallback,
    ):
      kpo_task = d.get_task("run_command_in_kpo.run_kpo-cli")
      self.assertEqual(kpo_task.startup_timeout_seconds, 600)

  def test_derived_parameters_supports_1_and_2_slice_pause_resume(self):
    one_slice = pw_utils.generate_derived_parameters.function(
        {
            "zone": "us-central1-b",
            "device_version": "v6e",
            "core_count": 16,
            "selected_model_names": "default_basic_1",
            "elastic_type": "Pause-resume",
            "num_slices_list": 1,
        },
        "pw_snap_fb",
    )
    self.assertEqual(one_slice["elastic_min_slice_count"], -1)
    self.assertEqual(one_slice["topology"], "tpuv6e:4x4")
    self.assertEqual(one_slice["num_elastic_slices"], 1)

    two_slice = pw_utils.generate_derived_parameters.function(
        {
            "zone": "us-central1-b",
            "device_version": "v6e",
            "core_count": 16,
            "selected_model_names": "default_basic_1",
            "elastic_type": "Pause-resume",
            "num_slices_list": 2,
        },
        "pw_snap_pr",
    )
    self.assertEqual(two_slice["elastic_min_slice_count"], -1)
    self.assertEqual(two_slice["topology"], "tpuv6e:4x4,tpuv6e:4x4")
    self.assertEqual(two_slice["num_elastic_slices"], 2)

  def test_wait_for_running_head_pod_handles_gcluster_and_xpk(self):
    pods = [
        {
            "metadata": {"name": "pw-snap-rr-1027-worker-0-1-abc"},
            "status": {"phase": "Failed"},
            "spec": {"containers": [{"name": "pathways-worker"}]},
        },
        {
            "metadata": {"name": "pw-snap-rr-1027-pathways-head-0-0-xyz"},
            "status": {
                "phase": "Running",
                "containerStatuses": [{
                    "name": "workload-container",
                    "state": {"running": {}},
                }],
            },
            "spec": {
                "containers": [
                    {"name": "pathways-proxy"},
                    {"name": "workload-container"},
                ]
            },
        },
    ]
    with unittest.mock.patch.object(
        pw_utils, "_list_workload_pods_kubectl", return_value=pods
    ):
      head_pod, container = pw_utils.wait_for_running_head_pod(
          "pw-snap-rr-1027", startup_timeout=5, poll_interval=1
      )
    self.assertEqual(
        head_pod["metadata"]["name"], "pw-snap-rr-1027-pathways-head-0-0-xyz"
    )
    self.assertEqual(container, "workload-container")

    failed_head = [{
        "metadata": {"name": "pw-snap-rr-1027-pathways-head-0-0-xyz"},
        "status": {"phase": "Failed"},
        "spec": {"containers": [{"name": "workload-container"}]},
    }]
    with unittest.mock.patch.object(
        pw_utils, "_list_workload_pods_kubectl", return_value=failed_head
    ):
      with self.assertRaisesRegex(AirflowFailException, "Head pod"):
        pw_utils.wait_for_running_head_pod(
            "pw-snap-rr-1027", startup_timeout=5, poll_interval=1
        )

  def test_snapshot_sed_patches_produce_valid_python_and_flags(self):
    params = dict(self.snap_mod.dag_snapshot_replica_resize.params)
    derived = pw_utils.generate_derived_parameters.function(
        params, "pw_snap_rr"
    )
    cmd = self.snap_mod.generate_commands.function(
        params, derived, recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
    )
    self.assertIn("--benchmark_steps=500", cmd)
    self.assertNotIn("--sidecar_name=external", cmd)
    patched_runner, patched_configs = self._run_sed_patches_and_parse(cmd)
    self.assertIn("elastic_backup_kind=snapshot", patched_runner)
    self.assertIn("elastic_snapshot_interval=5", patched_runner)
    self.assertIn("grain_use_elastic_iterator=True", patched_runner)
    self.assertIn("packing=False", patched_runner)
    self.assertIn("elastic_timeout_seconds=900", patched_runner)
    self.assertIn("async_checkpointing=False", patched_runner)
    self.assertIn("save_checkpoint_on_completion=False", patched_runner)
    self.assertIn("checkpoint_period=1000", patched_runner)
    self.assertNotIn("colocated_python_data_input=True", patched_runner)
    self.assertIn('"dataset_type": "grain"', patched_configs)
    self.assertIn('"reuse_example_batch": 0', patched_configs)
    self.assertIn('"enable_checkpointing": False', patched_configs)
    self.assertIn("c4-train.array_record*", patched_configs)
    self.assertNotIn('"profiler":', patched_configs)

    fb_params = dict(self.snap_mod.dag_snapshot_ckpt_fallback.params)
    fb_derived = pw_utils.generate_derived_parameters.function(
        fb_params, "pw_snap_fb"
    )
    fb_cmd = self.snap_mod.generate_commands.function(
        fb_params, fb_derived, recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
    )
    self.assertIn("--benchmark_steps=280", fb_cmd)
    fb_runner, fb_configs = self._run_sed_patches_and_parse(fb_cmd)
    self.assertIn("async_checkpointing=False", fb_runner)
    self.assertIn("checkpoint_period=150", fb_runner)
    self.assertIn("elastic_snapshot_interval=155", fb_runner)
    self.assertIn('"enable_checkpointing": True', fb_configs)

    params_coloc = dict(params, use_colocated_python=True)
    cmd_coloc = self.snap_mod.generate_commands.function(
        params_coloc, derived, recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
    )
    self.assertIn("--sidecar_name=external", cmd_coloc)
    self.assertIn("--colocated_python_image=", cmd_coloc)
    patched_runner_coloc, _ = self._run_sed_patches_and_parse(cmd_coloc)
    self.assertIn("colocated_python_data_input=True", patched_runner_coloc)

    params_v6e8 = dict(params, core_count=8)
    derived_v6e8 = pw_utils.generate_derived_parameters.function(
        params_v6e8, "pw_snap_rr"
    )
    cmd_v6e8 = self.snap_mod.generate_commands.function(
        params_v6e8, derived_v6e8, recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
    )
    self.assertIn("--virtual_slices=tpuv6e:2x4,tpuv6e:2x4", cmd_v6e8)
    runner_v6e8, _ = self._run_sed_patches_and_parse(cmd_v6e8)
    self.assertIn("chip_suffix = size if size == 1 else 4", runner_v6e8)

  def test_v5p_and_v5litepod_cluster_support(self):
    base_params = dict(self.snap_mod.dag_snapshot_replica_resize.params)
    v5p_params = dict(
        base_params,
        cluster_name="bodaborg-v5p-nap",
        project="cloud-tpu-shared-capacity",
        zone="europe-west4",
        device_version="v5p",
        core_count=16,
    )
    v5p_derived = pw_utils.generate_derived_parameters.function(
        v5p_params, "pw_snap_rr"
    )
    self.assertEqual(v5p_derived["region"], "europe-west4")
    self.assertEqual(v5p_derived["device_type"], "v5p-16")
    self.assertEqual(v5p_derived["topology"], "tpuv5:2x2x2,tpuv5:2x2x2")
    v5p_cmd = self.snap_mod.generate_commands.function(
        v5p_params, v5p_derived, recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
    )
    self.assertIn("--virtual_slices=tpuv5:2x2x2,tpuv5:2x2x2", v5p_cmd)
    self.assertIn(
        "--base_output_directory='gs://ml-auto-solutions/pw-snapshot/'", v5p_cmd
    )

    sed_cmds = _extract_sed_commands(v5p_cmd)
    with tempfile.TemporaryDirectory() as tmpdir:
      bench_dir = pathlib.Path(tmpdir) / "benchmarks"
      bench_dir.mkdir(parents=True, exist_ok=True)
      (bench_dir / "__init__.py").write_text("", encoding="utf-8")
      (bench_dir / "maxtext_ctk_runner.py").write_text(
          _SAMPLE_CTK_RUNNER, encoding="utf-8"
      )
      (bench_dir / "maxtext_trillium_model_configs.py").write_text(
          _SAMPLE_TRILLIUM_MODEL_CONFIGS, encoding="utf-8"
      )
      v5p_cfg = bench_dir / "maxtext_v5p_model_configs.py"
      v5p_cfg.write_text("v5p_model_dict = {}\n", encoding="utf-8")

      subprocess.run(
          ["bash", "-euo", "pipefail", "-c", " && ".join(sed_cmds)],
          cwd=tmpdir,
          check=True,
      )
      ast.parse(v5p_cfg.read_text(encoding="utf-8"))
      subprocess.run(
          [
              "python3",
              "-c",
              (
                  "import benchmarks.maxtext_v5p_model_configs as m; "
                  "cfg = m.v5p_model_dict['default_basic_1'].kwargs; "
                  "assert cfg['tuning_params']['dataset_type'] == 'grain'; "
                  "assert cfg['tuning_params']['reuse_example_batch'] == 0"
              ),
          ],
          cwd=tmpdir,
          check=True,
      )

    v5e_params = dict(
        base_params,
        cluster_name="v5e-256-bodaborg-europe-west4",
        project="tpu-prod-env-multipod",
        zone="europe-west4-b",
        device_version="v5litepod",
        core_count=256,
    )
    v5e_derived = pw_utils.generate_derived_parameters.function(
        v5e_params, "pw_snap_rr"
    )
    self.assertEqual(v5e_derived["topology"], "tpuv5e:16x16,tpuv5e:16x16")

  def test_interrupt_worker_pod_falls_back_on_single_host_slices(self):
    single_host_pods = [
        {
            "metadata": {"name": "pw-snap-rr-1102-pathways-head-0-0-aaa"},
            "status": {"phase": "Running"},
        },
        {
            "metadata": {"name": "pw-snap-rr-1102-worker-0-0-bbb"},
            "status": {"phase": "Running"},
        },
        {
            "metadata": {"name": "pw-snap-rr-1102-worker-1-0-ccc"},
            "status": {"phase": "Running"},
        },
    ]
    with (
        unittest.mock.patch.object(
            pw_utils, "_get_kubeconfig_env", return_value={}
        ),
        unittest.mock.patch.object(
            pw_utils,
            "_list_workload_pods_kubectl",
            return_value=single_host_pods,
        ),
        unittest.mock.patch.object(
            pw_utils.subprocess_utils, "run_exec"
        ) as mock_exec,
    ):
      pw_utils.interrupt_worker_pod.function(
          workload_id="pw-snap-rr-1102",
          cluster_name="bodaborg-v5p-nap",
          region="europe-west4",
          project_id="cloud-tpu-shared-capacity",
      )
    self.assertEqual(mock_exec.call_count, 1)
    self.assertIn("pw-snap-rr-1102-worker-1-0-ccc", mock_exec.call_args[0][0])

  def test_stream_until_snapshot_ready_enforces_slices_and_snapshot(self):
    staggered_lines = [
        "2026-10-27T07:56:00.000Z Sufficient slices active: 1 >= 1",
        "2026-10-27T07:56:05.000Z Saving in-memory snapshot at step 5...",
        "2026-10-27T07:56:06.000Z completed step: 8, live slice count: 1",
        "2026-10-27T07:56:20.000Z Sufficient slices active: 2 >= 1",
        "2026-10-27T07:56:25.000Z Saving in-memory snapshot at step 10...",
        "2026-10-27T07:56:27.000Z completed step: 13, live slice count: 2",
    ]
    ts = snapshot_utils.stream_until_snapshot_ready(
        staggered_lines,
        required_live_slices=2,
        interrupt_after_snapshot_step=5,
        post_snapshot_steps=3,
    )
    self.assertEqual(ts, "2026-10-27T07:56:27.000Z")

    single_slice_only = [
        "2026-10-27T07:56:00.000Z Sufficient slices active: 1 >= 1",
        "2026-10-27T07:56:05.000Z Saving in-memory snapshot at step 5...",
        "2026-10-27T07:56:08.000Z completed step: 8, live slice count: 1",
    ]
    with self.assertRaisesRegex(AirflowFailException, "2 active TPU slice"):
      snapshot_utils.stream_until_snapshot_ready(
          single_slice_only,
          required_live_slices=2,
          interrupt_after_snapshot_step=5,
          post_snapshot_steps=3,
      )

  def test_existing_checkpoint_dags_explicitly_set_checkpoint_mode(self):
    pr_params = {
        k: v.value for k, v in self.pw_mcjax_elastic.elastic_params.items()
    }
    pr_derived = pw_utils.generate_derived_parameters.function(
        pr_params, "pw_elastic_pause_resume"
    )
    pr_cmd = self.pw_mcjax_elastic.generate_commands.function(
        pr_params,
        pr_derived,
        recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE,
        self.pw_mcjax_elastic.ElasticType.PAUSE_RESUME,
    )
    pr_runner, pr_configs = self._run_sed_patches_and_parse(pr_cmd)
    self.assertIn("elastic_backup_kind=checkpoint", pr_runner)
    self.assertIn("packing=False", pr_runner)
    self.assertIn('"enable_checkpointing": True', pr_configs)

    rr_params = {
        k: v.value for k, v in self.pw_mcjax_elastic.replica_params.items()
    }
    rr_derived = pw_utils.generate_derived_parameters.function(
        rr_params, "pw_elastic_replica_resize"
    )
    rr_cmd = self.pw_mcjax_elastic.generate_commands.function(
        rr_params,
        rr_derived,
        recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE,
        self.pw_mcjax_elastic.ElasticType.REPLICA_RESIZE,
    )
    rr_runner, _ = self._run_sed_patches_and_parse(rr_cmd)
    self.assertIn("elastic_backup_kind=checkpoint", rr_runner)

    gp_params = {
        k: v.value
        for k, v in self.pw_mcjax_elastic_goodput.replica_params.items()
    }
    gp_derived = pw_utils.generate_derived_parameters.function(
        gp_params, "pw_elastic_goodput_replica"
    )
    gp_cmd = self.pw_mcjax_elastic_goodput.generate_commands.function(
        gp_params, gp_derived, recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
    )
    gp_runner, _ = self._run_sed_patches_and_parse(gp_cmd)
    self.assertIn("elastic_backup_kind=checkpoint", gp_runner)
    self.assertIn("grain_use_elastic_iterator=True", gp_runner)
    self.assertIn("packing=False", gp_runner)

  def test_validate_snapshot_log_text_replica_resize_happy_path(self):
    logs = "\n".join([
        "Saving in-memory snapshot at step 0...",
        "completed step: 0, seconds: 1.1, live slice count: 2",
        "Saving in-memory snapshot at step 5...",
        "completed step: 5, seconds: 0.4, live slice count: 2",
        "completed step: 6, seconds: 0.4, live slice count: 2",
        "completed step: 7, seconds: 0.4, live slice count: 2",
        "Elastic attempt 1 starting",
        "Restoring the in-memory snapshot at step 5...",
        "Restored the data iterator state: {'global_next_index': 192}.",
        "Restored the in-memory snapshot at step 5.",
        "completed step: 6, seconds: 0.5, live slice count: 1",
        "completed step: 12, seconds: 0.5, live slice count: 1",
        "New slices are available. Snapshotting step 12 before scaling up.",
        "Saving in-memory snapshot at step 12...",
        "Restoring the in-memory snapshot at step 12...",
        "Restored the data iterator state: {'global_next_index': 304}.",
        "Restored the in-memory snapshot at step 12.",
        "completed step: 13, seconds: 0.4, live slice count: 2",
    ])
    summary = snapshot_utils.validate_snapshot_log_text(
        logs,
        expect_fallback=False,
        expect_scale_up=True,
        expected_min_restores=2,
        expect_data_iterator=True,
    )
    self.assertEqual(summary["restored_snapshot_steps"], [5, 12])
    self.assertEqual(summary["scale_up_snapshot_steps"], [12])
    self.assertEqual(summary["restored_iterator_indices"], [192, 304])

    staggered_startup_logs = "\n".join([
        "Saving in-memory snapshot at step 0...",
        "New slices are available. Snapshotting step 0 before scaling up.",
        "completed step: 0, seconds: 1.1, live slice count: 1",
        "Restoring the in-memory snapshot at step 0...",
        "Restored the data iterator state: {'global_next_index': 8}.",
        "Restored the in-memory snapshot at step 0.",
        "completed step: 1, seconds: 0.4, live slice count: 2",
        "Saving in-memory snapshot at step 30...",
        "completed step: 30, seconds: 0.4, live slice count: 2",
        "Restoring the in-memory snapshot at step 30...",
        "Restored the data iterator state: {'global_next_index': 488}.",
        "Restored the in-memory snapshot at step 30.",
        "completed step: 31, seconds: 0.5, live slice count: 1",
        "New slices are available. Snapshotting step 31 before scaling up.",
        "Restoring the in-memory snapshot at step 31...",
        "Restored the data iterator state: {'global_next_index': 496}.",
        "Restored the in-memory snapshot at step 31.",
        "completed step: 32, seconds: 0.4, live slice count: 2",
    ])
    staggered_summary = snapshot_utils.validate_snapshot_log_text(
        staggered_startup_logs,
        expect_fallback=False,
        expect_scale_up=True,
        expected_min_restores=2,
        expect_data_iterator=True,
    )
    self.assertEqual(staggered_summary["restored_snapshot_steps"], [0, 30, 31])

  def test_validate_snapshot_log_text_ckpt_fallback_happy_path(self):
    logs = "\n".join([
        "Saving in-memory snapshot at step 0...",
        "completed step: 8, seconds: 0.4, live slice count: 1",
        "Saving in-memory snapshot at step 11...",
        "completed step: 11, seconds: 0.4, live slice count: 1",
        "completed step: 12, seconds: 0.4, live slice count: 1",
        "Restoring the in-memory snapshot at step 11...",
        "Restoring the in-memory snapshot at step 11 failed: RuntimeError",
        (
            "In-memory snapshot restore failed; falling back to checkpoint"
            " restore."
        ),
        "restoring from this run's directory step 8",
        "completed step: 9, seconds: 0.4, live slice count: 1",
    ])
    summary = snapshot_utils.validate_snapshot_log_text(
        logs,
        expect_fallback=True,
        expect_scale_up=False,
        expected_min_restores=0,
        expect_data_iterator=False,
    )
    self.assertEqual(summary["fallback_checkpoint_steps"], [8])

  def test_validate_snapshot_log_text_rejects_lost_or_replayed_steps(self):
    bad_continuity_logs = "\n".join([
        "Saving in-memory snapshot at step 5...",
        "completed step: 5, seconds: 0.4, live slice count: 2",
        "Restoring the in-memory snapshot at step 5...",
        "Restored the data iterator state: {'global_next_index': 192}.",
        "Restored the in-memory snapshot at step 5.",
        "completed step: 7, seconds: 0.4, live slice count: 2",
    ])
    with self.assertRaisesRegex(
        AirflowFailException, "Step continuity violation"
    ):
      snapshot_utils.validate_snapshot_log_text(
          bad_continuity_logs,
          expect_fallback=False,
          expect_scale_up=False,
          expected_min_restores=1,
          expect_data_iterator=True,
      )

  def test_validate_snapshot_log_text_rejects_silent_ckpt_fallback(self):
    unexpected_fallback_logs = "\n".join([
        "Saving in-memory snapshot at step 5...",
        "Restoring the in-memory snapshot at step 5 failed: RuntimeError",
        (
            "In-memory snapshot restore failed; falling back to checkpoint"
            " restore."
        ),
        "restoring from this run's directory step 0",
        "completed step: 1, seconds: 0.4, live slice count: 1",
    ])
    with self.assertRaisesRegex(
        AirflowFailException, "Unexpected snapshot failure"
    ):
      snapshot_utils.validate_snapshot_log_text(
          unexpected_fallback_logs,
          expect_fallback=False,
          expect_scale_up=False,
          expected_min_restores=1,
          expect_data_iterator=True,
      )

  def test_validate_snapshot_log_text_rejects_scale_up_step_mismatch(self):
    mismatch_logs = "\n".join([
        "Saving in-memory snapshot at step 5...",
        "Restored the data iterator state: {'global_next_index': 192}.",
        "Restored the in-memory snapshot at step 5.",
        "completed step: 6, seconds: 0.4, live slice count: 1",
        "New slices are available. Snapshotting step 12 before scaling up.",
        "Restored the data iterator state: {'global_next_index': 192}.",
        "Restored the in-memory snapshot at step 10.",
        "completed step: 11, seconds: 0.4, live slice count: 2",
    ])
    with self.assertRaisesRegex(
        AirflowFailException, "Scale-up restore step mismatch"
    ):
      snapshot_utils.validate_snapshot_log_text(
          mismatch_logs,
          expect_fallback=False,
          expect_scale_up=True,
          expected_min_restores=2,
          expect_data_iterator=True,
      )

  def test_validate_snapshot_log_text_rejects_regressing_iterator_index(self):
    regressing_logs = "\n".join([
        "Saving in-memory snapshot at step 5...",
        "Restored the data iterator state: {'global_next_index': 300}.",
        "Restored the in-memory snapshot at step 5.",
        "completed step: 6, seconds: 0.4, live slice count: 1",
        "New slices are available. Snapshotting step 10 before scaling up.",
        "Restored the data iterator state: {'global_next_index': 100}.",
        "Restored the in-memory snapshot at step 10.",
        "completed step: 11, seconds: 0.4, live slice count: 2",
    ])
    with self.assertRaisesRegex(
        AirflowFailException, "global_next_index regressed"
    ):
      snapshot_utils.validate_snapshot_log_text(
          regressing_logs,
          expect_fallback=False,
          expect_scale_up=True,
          expected_min_restores=2,
          expect_data_iterator=True,
      )


if __name__ == "__main__":
  unittest.main()
