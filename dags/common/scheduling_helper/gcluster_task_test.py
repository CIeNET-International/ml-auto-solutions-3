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

"""Integration tests for GclusterTask and polymorphic GKE config factory."""

import datetime
import unittest
from unittest import mock

from airflow import models
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import QuarantineTests
from dags.common.vm_resource import Gclusters, Project, TpuVersion, XpkClusters
from dags.multipod.configs import gke_config
from xlml.apis import gcp_config, gcluster_config, metric_config, task, test_config


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
    self.gcluster_task = task.GclusterTask(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
    )

  def test_gcluster_config_override(self):
    cfg = Gclusters.TPU_V5P_MLPERF_CLUSTER
    self.assertEqual(cfg.core_count, 4)
    self.assertEqual(cfg.namespace, "automation-testing")

    overridden = cfg.override(core_count=64, namespace="custom-ns")
    self.assertEqual(overridden.core_count, 64)
    self.assertEqual(overridden.namespace, "custom-ns")
    self.assertEqual(overridden.name, cfg.name)

  def test_get_gke_config_polymorphic_dispatch(self):
    xpk_task_obj = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-xpk",
        run_model_cmds=("echo 'xpk'",),
        docker_image="gcr.io/test/image:latest",
        cluster=XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
        test_owner=test_owner.SURBHI_J,
    )
    self.assertIsInstance(xpk_task_obj, task.XpkTask)
    self.assertNotIsInstance(xpk_task_obj, task.GclusterTask)

    gcluster_task_obj = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-gcluster",
        run_model_cmds=("echo 'gcluster'",),
        docker_image="gcr.io/test/image:latest",
        cluster=Gclusters.TPU_V5P_MLPERF_CLUSTER,
        test_owner=test_owner.SURBHI_J,
    )
    self.assertIsInstance(gcluster_task_obj, task.GclusterTask)

  def test_gcluster_task_run_structure(self):
    with self.test_dag:
      tg = self.gcluster_task.run(
          skip_post_process=True,
          priority="high",
          gcluster_version="v1.99.0",
      )

    self.assertIsNotNone(tg)
    self.assertEqual(tg.group_id, self.test_cfg.benchmark_id)

    task_ids = [t.task_id for t in self.test_dag.tasks]
    self.assertTrue(
        any("run_workload" in tid for tid in task_ids),
        f"Missing run_workload in {task_ids}",
    )
    self.assertTrue(
        any("wait_for_workload_completion" in tid for tid in task_ids),
        f"Missing wait_for_workload_completion in {task_ids}",
    )
    self.assertTrue(
        any("clean_up_workload" in tid for tid in task_ids),
        f"Missing clean_up_workload in {task_ids}",
    )

  def test_gcluster_task_with_mounts(self):
    with self.test_dag:
      _ = self.gcluster_task.run(
          skip_post_process=True,
          mounts=["/dev/shm;/dev/shm;rw"],
      )

  def test_gcluster_name_gen_and_quarantine_task(self):
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    namegen_task = task.GclusterNameGenAndQuarantineTask(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        task_metric_config=metric_cfg,
    )
    with self.test_dag:
      tg = namegen_task.run(
          skip_post_process=True,
          priority="very-high",
          mounts="/dev/shm;/dev/shm;rw",
      )

    self.assertIsNotNone(tg)
    task_ids = [t.task_id for t in self.test_dag.tasks]
    self.assertTrue(any("generate_run_name" in tid for tid in task_ids))

  @mock.patch.object(QuarantineTests, "is_quarantined", return_value=True)
  def test_gcluster_task_with_quarantine_true(self, mock_quarantine):
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    with self.test_dag:
      quarantine_group = TaskGroup(group_id="quarantine_group")
      namegen_task = task.GclusterNameGenAndQuarantineTask(
          task_test_config=self.test_cfg,
          task_gcp_config=self.gcp_cfg,
          task_metric_config=metric_cfg,
          quarantine_task_group=quarantine_group,
      )
      tg = namegen_task.run(
          skip_post_process=True,
          priority="very-high",
      )
    self.assertIsNotNone(tg)


if __name__ == "__main__":
  unittest.main()
