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
from xlml.apis.metric_config import DatasetOption
from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags.common.vm_resource import XpkClusters


def get_axlearn_tpu_config(
    test_name: str,
    docker_image: str,
    test_owner: str,
    time_out_in_min: int,
    cluster: XpkClusterConfig = XpkClusters.TPU_V5P_128_CLUSTER,
    dataset_name: DatasetOption = DatasetOption.XLML_DATASET,
    num_slices: int = 1,
) -> task.AXLearnTask:
  """Setup the AXLearn tpu env config."""

  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=dataset_name,
  )

  latest_docker_image = f"{docker_image.split(':')[0]}:latest"
  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=cluster.device_version,
          cores=cluster.core_count,
      ),
      test_name=test_name,
      run_model_cmds=None,
      set_up_cmds=None,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner,
      num_slices=num_slices,
      cluster_name=cluster.name,
      docker_image=latest_docker_image,
  )

  return task.AXLearnTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
