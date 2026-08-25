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

"""Tests for task.py, gke_cluster_config.py, and GKE config builders."""

import datetime
import unittest
from unittest import mock

from airflow import models
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import QuarantineTests
from dags.common.vm_resource import GkeClusters, Project, TpuVersion
from dags.multipod.configs import gke_config
from xlml.apis import gcp_config, metric_config, task, test_config


class GclusterTaskTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dag = models.DAG(
        dag_id="test_gcluster_dag",
        start_date=datetime.datetime(2026, 8, 1),
        schedule=None,
    )
    self.test_cfg = test_config.TpuGkeTest(
        test_config.Tpu(version=TpuVersion.V4, cores=8),
        test_name="test-gcluster-workload",
        run_model_cmds=("echo 'hello'",),
        set_up_cmds=None,
        timeout=datetime.timedelta(minutes=30),
        task_owner=test_owner.SURBHI_J,
        cluster_name="test-cluster",
        docker_image="gcr.io/test/image:latest",
        namespace="automation-testing",
    )
    self.gcp_cfg = gcp_config.GCPConfig(
        project_name=Project.CLOUD_TPU_MULTIPOD_DEV.value,
        zone="us-central1-a",
        dataset_name=metric_config.DatasetOption.XLML_DATASET,
    )
    self.runner_cfg = task.GclusterRunnerConfig(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
    )
    self.gcluster_task = task.GclusterTask(
        runner_config=self.runner_cfg,
    )

  def test_gcluster_config_override(self):
    """Overrides GkeClusterConfig attributes while preserving defaults."""
    cfg = GkeClusters.TPU_V5P_MLPERF_CLUSTER
    self.assertEqual(cfg.core_count, 8)
    self.assertEqual(cfg.namespace, "automation-testing")

    overridden = cfg.override(core_count=64, namespace="custom-ns")
    self.assertEqual(overridden.core_count, 64)
    self.assertEqual(overridden.namespace, "custom-ns")
    self.assertEqual(overridden.name, cfg.name)

  def test_get_gke_config_gcluster(self):
    """Constructs a GclusterTask when use_gcluster=True is enabled."""
    gcluster_task_obj = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-gcluster",
        run_model_cmds=("echo 'gcluster'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V5P_MLPERF_CLUSTER,
        test_owner=test_owner.SURBHI_J,
        use_gcluster=True,
    )
    self.assertIsInstance(gcluster_task_obj, task.GclusterTask)
    self.assertIsInstance(gcluster_task_obj, task.BaseRunnerTask)
    self.assertIsInstance(
        gcluster_task_obj.runner_config, task.GclusterRunnerConfig
    )

  def test_get_gke_config_with_name_gen_and_quarantine_gcluster(self):
    """Constructs GclusterNameGenAndQuarantineTask for Gcluster workloads."""
    gcluster_task_obj = gke_config.get_gke_config_with_name_gen_and_quarantine(
        time_out_in_min=30,
        test_name="test-gcluster-namegen",
        run_model_cmds=("echo 'gcluster'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V5P_MLPERF_CLUSTER,
        test_owner=test_owner.SURBHI_J,
        use_gcluster=True,
    )
    self.assertIsInstance(
        gcluster_task_obj, task.GclusterNameGenAndQuarantineTask
    )
    self.assertIsInstance(gcluster_task_obj, task.BaseRunnerTask)
    self.assertIsInstance(
        gcluster_task_obj.runner_config, task.GclusterRunnerConfig
    )

  def test_get_gke_config_xpk(self):
    """Constructs an XpkTask when use_gcluster=False is defaulted."""
    xpk_task_obj = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-xpk",
        run_model_cmds=("echo 'xpk'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V4_8_MAXTEXT_CLUSTER,
        test_owner=test_owner.SURBHI_J,
    )
    self.assertIsInstance(xpk_task_obj, task.XpkTask)
    self.assertIsInstance(xpk_task_obj, task.BaseRunnerTask)
    self.assertIsInstance(xpk_task_obj.runner_config, task.XpkRunnerConfig)

  def test_get_gke_config_with_name_gen_and_quarantine_xpk(self):
    """Constructs XpkNameGenAndQuarantineTask when use_gcluster=False."""
    xpk_task_obj = gke_config.get_gke_config_with_name_gen_and_quarantine(
        time_out_in_min=30,
        test_name="test-xpk-namegen",
        run_model_cmds=("echo 'xpk'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V4_8_MAXTEXT_CLUSTER,
        test_owner=test_owner.SURBHI_J,
    )
    self.assertIsInstance(xpk_task_obj, task.XpkNameGenAndQuarantineTask)
    self.assertIsInstance(xpk_task_obj, task.BaseRunnerTask)
    self.assertIsInstance(xpk_task_obj.runner_config, task.XpkRunnerConfig)

  def test_gcluster_task_run_structure(self):
    """Generates TaskGroup containing run, monitor, and cleanup tasks."""
    runner_cfg = task.GclusterRunnerConfig(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        priority="high",
        gcluster_version="v1.101.0",
    )
    gcluster_task = task.GclusterTask(runner_config=runner_cfg)
    with self.test_dag:
      tg = gcluster_task.run(
          skip_post_process=True,
      )

    self.assertIsNotNone(tg)
    self.assertEqual(tg.group_id, self.test_cfg.benchmark_id)

    b_id = self.test_cfg.benchmark_id
    expected_task_ids = {
        f"{b_id}.pre_process.generate_workload_id",
        f"{b_id}.run_model.dummy_op_for_teardown",
        f"{b_id}.run_model.launch_workload.run_workload",
        f"{b_id}.run_model.launch_workload.wait_for_workload_start",
        f"{b_id}.run_model.wait_for_workload_completion",
        f"{b_id}.run_model.clean_up_workload",
    }
    actual_task_ids = {t.task_id for t in self.test_dag.tasks}
    self.assertTrue(
        expected_task_ids.issubset(actual_task_ids),
        f"Missing expected tasks: {expected_task_ids - actual_task_ids}",
    )

    cleanup_task = self.test_dag.get_task(f"{b_id}.run_model.clean_up_workload")
    self.assertEqual(cleanup_task.trigger_rule, "all_done_setup_success")

  def test_gcluster_task_with_mounts(self):
    """Passes volume mounts to GclusterRunnerConfig and task execution."""
    runner_cfg = task.GclusterRunnerConfig(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        mounts=["/dev/shm;/dev/shm;rw"],
    )
    gcluster_task = task.GclusterTask(runner_config=runner_cfg)
    with self.test_dag:
      tg = gcluster_task.run(
          skip_post_process=True,
      )
    self.assertIsNotNone(tg)
    self.assertEqual(runner_cfg.mounts, ["/dev/shm;/dev/shm;rw"])

  def test_gcluster_name_gen_and_quarantine_task(self):
    """Injects dynamic run name generator without mutating input configs."""
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    runner_cfg = task.GclusterRunnerConfig(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        task_metric_config=metric_cfg,
        priority="very-high",
        mounts="/dev/shm;/dev/shm;rw",
    )
    original_cmds = self.test_cfg.run_model_cmds
    namegen_task = task.GclusterNameGenAndQuarantineTask(
        runner_config=runner_cfg,
    )
    with self.test_dag:
      tg = namegen_task.run(
          skip_post_process=True,
      )

    self.assertIsNotNone(tg)
    task_ids = [t.task_id for t in self.test_dag.tasks]
    self.assertTrue(any("generate_run_name" in tid for tid in task_ids))
    # Assert configs remain unmutated
    self.assertEqual(self.test_cfg.run_model_cmds, original_cmds)
    self.assertEqual(runner_cfg.priority, "very-high")

  @mock.patch.object(QuarantineTests, "is_quarantined", return_value=True)
  def test_gcluster_task_with_quarantine_true(self, mock_quarantine):
    """Links quarantined test execution to the quarantine TaskGroup."""
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    with self.test_dag:
      quarantine_group = TaskGroup(group_id="quarantine_group")
      runner_cfg = task.GclusterRunnerConfig(
          task_test_config=self.test_cfg,
          task_gcp_config=self.gcp_cfg,
          task_metric_config=metric_cfg,
          priority="very-high",
      )
      namegen_task = task.GclusterNameGenAndQuarantineTask(
          runner_config=runner_cfg,
          quarantine_task_group=quarantine_group,
      )
      tg = namegen_task.run(
          skip_post_process=True,
      )
    self.assertIsNotNone(tg)
    mock_quarantine.assert_called_once_with(self.test_cfg.benchmark_id)
    self.assertEqual(tg.parent_group, quarantine_group)

  @mock.patch.object(QuarantineTests, "is_quarantined", return_value=False)
  def test_gcluster_task_with_quarantine_false(self, mock_quarantine):
    """Does not attach TaskGroup to quarantine group when not quarantined."""
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    with self.test_dag:
      quarantine_group = TaskGroup(group_id="quarantine_group")
      runner_cfg = task.GclusterRunnerConfig(
          task_test_config=self.test_cfg,
          task_gcp_config=self.gcp_cfg,
          task_metric_config=metric_cfg,
          priority="very-high",
      )
      namegen_task = task.GclusterNameGenAndQuarantineTask(
          runner_config=runner_cfg,
          quarantine_task_group=quarantine_group,
      )
      tg = namegen_task.run(
          skip_post_process=True,
      )
    self.assertIsNotNone(tg)
    mock_quarantine.assert_called_once_with(self.test_cfg.benchmark_id)
    self.assertNotEqual(tg.parent_group, quarantine_group)


if __name__ == "__main__":
  unittest.main()
