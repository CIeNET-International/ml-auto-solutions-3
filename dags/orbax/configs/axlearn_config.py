# Copyright 2025 Google LLC
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

"""Utilities to construct configs for AXLearn framework DAG."""


import datetime

from xlml.apis import gcp_config, task, test_config
from xlml.apis.xpk_cluster_config import XpkClusterConfig


# TODO: simplify
def get_axlearn_tpu_config(
    test_name: str,
    test_owner: str,
    docker_image_full_url: str,
    docker_image_name: str,
    docker_image_repo: str,
    cluster: XpkClusterConfig,
    workload_provision_timeout: datetime.timedelta,
    workload_run_timeout: datetime.timedelta,
    workload_post_test_timeout: datetime.timedelta,
    num_slices: int = 1,
) -> task.AXLearnTask:
  """Setup the AXLearn tpu env config."""

  return task.AXLearnTask(
      test_config=test_config.TpuGkeTest(
          accelerator=test_config.Tpu(
              version=cluster.device_version,
              cores=cluster.core_count,
          ),
          test_name=test_name,
          cluster_name=cluster.name,
          docker_image=docker_image_full_url,
          set_up_cmds=None,
          run_model_cmds=None,
          task_owner=test_owner,
          num_slices=num_slices,
      ),
      gcp_config=gcp_config.GCPConfig(
          project_name=cluster.project,
          zone=cluster.zone,
      ),
      workload_provision_timeout=workload_provision_timeout,
      workload_run_timeout=workload_run_timeout,
      workload_post_test_timeout=workload_post_test_timeout,
      image_name=docker_image_name,
      image_repo=docker_image_repo,
      image_full_url=docker_image_full_url,
  )
