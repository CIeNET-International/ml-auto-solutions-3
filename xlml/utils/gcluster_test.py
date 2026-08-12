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

"""Unit tests for gcluster.py utility."""

import unittest
from unittest import mock
from airflow.exceptions import AirflowFailException
from dags.common.vm_resource import GpuVersion
from xlml.utils import gcluster


class GclusterTest(unittest.TestCase):

  def test_get_gcluster_setup_cmd(self):
    cmds = gcluster.get_gcluster_setup_cmd("/tmp/test_dir", "v1.99.0")
    self.assertEqual(len(cmds), 3)
    self.assertEqual(
        cmds[0], "set -xueo pipefail; export PATH=/tmp/test_dir/bin:$PATH"
    )
    self.assertIn("mkdir -p /tmp/test_dir/bin", cmds[1])
    self.assertIn(
        "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/v1.99.0/gcluster_bundle_linux_amd64.tgz",
        cmds[1],
    )
    self.assertIn("curl -fsSL", cmds[1])
    self.assertIn("chmod +x /tmp/test_dir/bin/gcluster", cmds[1])
    self.assertIn("gcloud auth configure-docker", cmds[2])

  def test_is_valid_gpu_version(self):
    self.assertTrue(gcluster.is_valid_gpu_version(GpuVersion.XPK_H100.value))
    self.assertTrue(
        gcluster.is_valid_gpu_version(GpuVersion.XPK_H100_MEGA.value)
    )
    self.assertFalse(gcluster.is_valid_gpu_version("v4-8"))
    self.assertFalse(gcluster.is_valid_gpu_version("v5e-16"))

  def test_generate_workload_id_format_and_length(self):
    benchmark_id = "maxtext-e2e-pre-training-llama3-70b-tpu-test"
    workload_id = gcluster.generate_workload_id.function(benchmark_id)

    # Must be <= 28 chars to satisfy Kubernetes/GCE and gcluster limits
    self.assertLessEqual(len(workload_id), 28)
    self.assertRegex(workload_id, r"^[a-zA-Z0-9-]+-[a-f0-9]{8}$")

  def test_generate_workload_id_uniqueness(self):
    benchmark_id = "test-job"
    id1 = gcluster.generate_workload_id.function(benchmark_id)
    id2 = gcluster.generate_workload_id.function(benchmark_id)
    self.assertNotEqual(id1, id2)

  def test_generate_workload_id_empty_or_special_chars(self):
    id_empty = gcluster.generate_workload_id.function("!!!###")
    self.assertLessEqual(len(id_empty), 28)
    self.assertTrue(id_empty.startswith("job-"))

  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_tpu(self, mock_hook_cls, mock_log_meta):
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345678",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type="v4-8",
        run_cmds="bash run.sh",
        num_slices=1,
        priority="very-high",
        namespace="automation-testing",
        mounts=["/dev/shm;/dev/shm;rw", "/local/path;/container/path;ro"],
    )

    mock_log_meta.assert_called_once()
    mock_hook.run_command.assert_called_once()
    cmd_args = mock_hook.run_command.call_args[0][0]
    full_cmd = cmd_args[2]

    self.assertIn("job submit", full_cmd)
    self.assertIn("--cluster=test-cluster", full_cmd)
    self.assertIn("--location=us-central1-a", full_cmd)
    self.assertIn("--project=test-project", full_cmd)
    self.assertIn("--name=test-workload-12345678", full_cmd)
    self.assertIn("--compute-type=v4-8", full_cmd)
    self.assertIn("--image=us-docker.pkg.dev/img:latest", full_cmd)
    self.assertIn("--num-slices=1", full_cmd)
    self.assertNotIn("--num-nodes", full_cmd)
    self.assertIn("--priority=very-high", full_cmd)
    self.assertIn("--env GCS_OUTPUT=gs://test-bucket/output", full_cmd)
    self.assertIn("--download-dependencies", full_cmd)
    self.assertIn("--mount=/dev/shm\\;/dev/shm\\;rw", full_cmd)
    self.assertIn("--mount=/local/path\\;/container/path\\;ro", full_cmd)
    self.assertIn(
        "kubectl config set-context --current --namespace=automation-testing",
        full_cmd,
    )

  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_gpu_and_mtc_and_restarts(
      self, mock_hook_cls, mock_log_meta
  ):
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345678",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type=GpuVersion.XPK_H100_MEGA.value,
        run_cmds="python3 -c 'print(\"hello\")'",
        num_slices=2,
        ramdisk_directory="/dev/shm/ramdisk",
        mtc_enabled=True,
        max_restart=5,
        mounts="/dev/shm;/dev/shm;rw",
    )

    full_cmd = mock_hook.run_command.call_args[0][0][2]
    self.assertIn("--num-nodes=2", full_cmd)
    self.assertNotIn("--num-slices", full_cmd)
    self.assertIn("--gke-mtc-ramdisk-dir=/dev/shm/ramdisk", full_cmd)
    self.assertIn("--gke-mtc-enabled", full_cmd)
    self.assertIn("--restarts=5", full_cmd)
    self.assertIn("--gke-scheduler=gke.io/topology-aware-auto", full_cmd)
    self.assertIn("--mount=/dev/shm\\;/dev/shm\\;rw", full_cmd)

  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_pathways(self, mock_hook_cls, mock_log_meta):
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345678",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type="v4-8",
        run_cmds="bash run.sh",
        use_pathways=True,
    )

    full_cmd = mock_hook.run_command.call_args[0][0][2]
    self.assertIn("--pathways", full_cmd)
    self.assertIn("--pathways-gcs-location=gs://test-bucket/output", full_cmd)

  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_clean_up_workload_success(self, mock_hook_cls):
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.clean_up_workload.function(
        workload_id="test-workload-12345678",
        project_id="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
    )

    mock_hook.run_command.assert_called_once()
    full_cmd = mock_hook.run_command.call_args[0][0][2]
    self.assertIn("job cancel test-workload-12345678", full_cmd)
    self.assertIn("--cluster=test-cluster", full_cmd)
    self.assertIn("--location=us-central1-a", full_cmd)
    self.assertIn("--project=test-project", full_cmd)
    self.assertIn("--download-dependencies", full_cmd)

  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_clean_up_workload_failure_raises(self, mock_hook_cls):
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 1
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    with self.assertRaises(AssertionError):
      gcluster.clean_up_workload.function(
          workload_id="test-workload-12345678",
          project_id="test-project",
          zone="us-central1-a",
          cluster_name="test-cluster",
      )

  @mock.patch("xlml.utils.gcluster._list_workload_pods")
  @mock.patch("xlml.utils.gcluster._get_core_api_client")
  def test_wait_for_workload_start(self, mock_get_client, mock_list_pods):
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "pod-1"
    mock_pod.status.phase = "Running"
    mock_pod.status.container_statuses = []

    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_list_pods.return_value = mock_pod_list

    started = gcluster.wait_for_workload_start.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    self.assertTrue(started)

  @mock.patch("xlml.utils.gcluster._list_workload_pods")
  @mock.patch("xlml.utils.gcluster._get_core_api_client")
  def test_wait_for_workload_completion_success(
      self, mock_get_client, mock_list_pods
  ):
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "pod-1"
    mock_pod.status.phase = "Succeeded"
    mock_pod.status.container_statuses = []
    mock_pod.spec.containers = [mock_pod]

    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_list_pods.return_value = mock_pod_list

    mock_core_api = mock.MagicMock()
    mock_core_api.read_namespaced_pod_log.return_value = "Step 50 complete"
    mock_get_client.return_value = mock_core_api

    completed = gcluster.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    self.assertTrue(completed)

  @mock.patch("xlml.utils.gcluster._list_workload_pods")
  @mock.patch("xlml.utils.gcluster._get_core_api_client")
  def test_wait_for_workload_completion_failure(
      self, mock_get_client, mock_list_pods
  ):
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "pod-1"
    mock_pod.status.phase = "Failed"
    mock_pod.status.container_statuses = []
    mock_pod.spec.containers = [mock_pod]

    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_list_pods.return_value = mock_pod_list

    mock_core_api = mock.MagicMock()
    # Ensure even if pod log read raises an exception, the failure is still raised
    mock_core_api.read_namespaced_pod_log.side_effect = Exception(
        "K8s log stream closed"
    )
    mock_get_client.return_value = mock_core_api

    with self.assertRaises(AirflowFailException):
      gcluster.wait_for_workload_completion.function(
          workload_id="test-workload",
          project_id="test-project",
          region="us-central1",
          cluster_name="test-cluster",
      )

  @mock.patch("xlml.utils.gcluster._get_workload_job")
  @mock.patch("xlml.utils.gcluster._get_batch_api_client")
  @mock.patch("xlml.utils.gcluster._list_workload_pods")
  @mock.patch("xlml.utils.gcluster._get_core_api_client")
  def test_wait_for_workload_completion_no_pods_conditions_none(
      self, mock_get_client, mock_list_pods, mock_get_batch, mock_get_job
  ):
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list

    mock_job = mock.MagicMock()
    mock_job.status.conditions = None
    mock_get_job.return_value = mock_job

    completed = gcluster.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    self.assertFalse(completed)

  @mock.patch("xlml.utils.gcluster._get_workload_job")
  @mock.patch("xlml.utils.gcluster._get_batch_api_client")
  @mock.patch("xlml.utils.gcluster._list_workload_pods")
  @mock.patch("xlml.utils.gcluster._get_core_api_client")
  def test_wait_for_workload_completion_no_pods_job_completed(
      self, mock_get_client, mock_list_pods, mock_get_batch, mock_get_job
  ):
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list

    mock_condition = mock.MagicMock()
    mock_condition.type = "Complete"
    mock_job = mock.MagicMock()
    mock_job.status.conditions = [mock_condition]
    mock_get_job.return_value = mock_job

    completed = gcluster.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    self.assertTrue(completed)


if __name__ == "__main__":
  unittest.main()
