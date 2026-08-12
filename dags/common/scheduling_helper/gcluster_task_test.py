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

"""Unit tests for GclusterTask and GclusterRunner in xlml/apis/task.py."""

import datetime as dt
import unittest
from unittest import mock

import airflow
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import QuarantineTests
from dags.common.vm_resource import Gclusters, Project, TpuVersion, XpkClusters
from dags.multipod.configs import gke_config
from xlml.apis import gcp_config, metric_config, task, test_config
from xlml.apis.gcluster_config import GclusterConfig
from xlml.utils import gcluster


class GclusterTaskTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dag = models.DAG(
        dag_id="test_gcluster_dag",
        start_date=dt.datetime(2026, 1, 1),
        schedule=None,
    )
    self.gcp_cfg = gcp_config.GCPConfig(
        project_name="test-project",
        zone="us-central1-a",
        dataset_name=metric_config.DatasetOption.XLML_DATASET,
    )
    self.test_cfg = test_config.TpuGkeTest(
        test_config.Tpu(
            version=TpuVersion.V4,
            cores=8,
        ),
        test_name="test-gcluster-workload",
        run_model_cmds=["python3 -m train"],
        set_up_cmds=None,
        timeout=dt.timedelta(minutes=30),
        task_owner=test_owner.SURBHI_J,
        num_slices=1,
        cluster_name="test-tpu-cluster",
        docker_image="us-docker.pkg.dev/test/image:latest",
    )
    self.gcluster_task = task.GclusterTask(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
    )

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
        any("run_model.run_workload" in tid for tid in task_ids),
        f"Missing run_workload in {task_ids}",
    )
    self.assertTrue(
        any(
            "run_model.wait_for_workload_completion" in tid for tid in task_ids
        ),
        f"Missing wait_for_workload_completion in {task_ids}",
    )
    self.assertTrue(
        any("run_model.clean_up_workload" in tid for tid in task_ids),
        f"Missing clean_up_workload in {task_ids}",
    )

  def test_gcluster_task_run_with_mounts(self):
    with self.test_dag:
      tg = self.gcluster_task.run(
          skip_post_process=True,
          priority="high",
          mounts=gcluster.VolumeMounts.DSHM,
          gcluster_version="v1.99.0",
      )

    self.assertIsNotNone(tg)
    self.assertEqual(tg.group_id, self.test_cfg.benchmark_id)

  def test_gcluster_name_gen_and_quarantine_task(self):
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            use_regex_file_location=True,
        ),
    )
    task_with_metric = task.GclusterNameGenAndQuarantineTask(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        task_metric_config=metric_cfg,
    )

    with self.test_dag:
      tg = task_with_metric.run(
          gcluster_version="v1.99.0",
          skip_post_process=True,
      )

    self.assertIsNotNone(tg)
    task_ids = [t.task_id for t in self.test_dag.tasks]
    self.assertTrue(
        any("generate_run_name" in tid for tid in task_ids),
        f"Missing generate_run_name in {task_ids}",
    )
    self.assertTrue(
        any("generate_tb_file_location" in tid for tid in task_ids),
        f"Missing generate_tb_file_location in {task_ids}",
    )

  @mock.patch.object(QuarantineTests, "is_quarantined", return_value=True)
  def test_gcluster_task_with_quarantine_true(self, mock_quarantine):
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            use_regex_file_location=True,
        ),
    )
    quarantine_group = TaskGroup(group_id="quarantine_group")
    task_with_metric = task.GclusterNameGenAndQuarantineTask(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        task_metric_config=metric_cfg,
        quarantine_task_group=quarantine_group,
    )

    with self.test_dag:
      tg = task_with_metric.run(
          gcluster_version="v1.99.0",
          skip_post_process=True,
      )

    self.assertIsNotNone(tg)
    mock_quarantine.assert_called_once_with(self.test_cfg.benchmark_id)

  def test_gcluster_config_override(self):
    base_config = GclusterConfig(
        name="test-cluster",
        device_version=TpuVersion.V5P,
        core_count=8,
        project="test-project",
        zone="us-central1-a",
        namespace="default",
    )
    overridden = base_config.override(core_count=128, namespace="custom-ns")
    self.assertEqual(overridden.core_count, 128)
    self.assertEqual(overridden.namespace, "custom-ns")
    self.assertEqual(overridden.name, "test-cluster")
    # Base config must remain unmodified
    self.assertEqual(base_config.core_count, 8)
    self.assertEqual(base_config.namespace, "default")

  def test_get_gke_config_polymorphic_dispatch(self):
    gcluster_t = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-gcluster",
        docker_image="gcr.io/test/image:latest",
        test_owner=test_owner.JACKY_F,
        run_model_cmds=["echo test"],
        cluster=Gclusters.TPU_V5P_MLPERF_CLUSTER,
    )
    self.assertIsInstance(gcluster_t, task.GclusterTask)

    xpk_t = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-xpk",
        docker_image="gcr.io/test/image:latest",
        test_owner=test_owner.JACKY_F,
        run_model_cmds=["echo test"],
        cluster=XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
    )
    self.assertIsInstance(xpk_t, task.XpkTask)


if __name__ == "__main__":
  unittest.main()
