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

"""A self-contained DAG to demonstrate running workloads with Cluster Toolkit (gcluster) on GKE."""

import datetime
from airflow import models
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from xlml.apis import gcp_config, metric_config, task, test_config

with models.DAG(
    dag_id="gcluster_example_dag",
    schedule=None,
    tags=[
        "example",
        "gke",
        "xlml",
        "benchmark",
        "TPU",
        "gcluster",
        "cluster-toolkit",
    ],
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
) as dag:
  cluster = XpkClusters.TPU_V4_8_MAS_CLUSTER

  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=cluster.device_version,
          cores=cluster.core_count,
      ),
      test_name="gcluster-v4-8-smoke-test",
      cluster_name=cluster.name,
      docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
      run_model_cmds=[
          "python3 -c \"import jax; print('=== TPU DEVICES ===', jax.devices('tpu')); assert len(jax.devices('tpu')) > 0, 'No TPU devices found'\""
      ],
      set_up_cmds=None,
      timeout=datetime.timedelta(minutes=15),
      task_owner=test_owner.JACKY_F,
      num_slices=1,
  )

  gcluster_smoke_test = task.GclusterTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  ).run()
