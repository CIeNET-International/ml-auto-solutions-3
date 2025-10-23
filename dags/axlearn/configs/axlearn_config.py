# Copyright 2024 Google LLC
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

"""Utilities to construct configs for solutionsteam_jax_bite DAG."""


import datetime
from typing import Optional, Iterable
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags.common.vm_resource import TpuVersion, Project, XpkClusters
from airflow.models.taskmixin import DAGNode

def get_axlearn_tpu_config(
    test_name: str,
    docker_image: str,
    test_owner: str,
    run_model_cmds: Iterable[str],
    time_out_in_min: int,
    cluster: XpkClusterConfig = XpkClusters.TPU_V5P_128_CLUSTER,
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.XLML_DATASET,
    num_slices: int = 1,
) -> task.AxlearnTask:
  """Setup the axlearn tpu env config."""

  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=dataset_name,
  )
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
      docker_image=docker_image,
  )

  return task.AxlearnTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
