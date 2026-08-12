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
from dags.common.vm_resource import DockerImage, Gclusters
from dags.multipod.configs import gke_config

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
  gcluster_smoke_test = gke_config.get_gke_config(
      test_name="gcluster-v5p-smoke-test",
      cluster=Gclusters.TPU_V5P_MLPERF_CLUSTER,
      docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
      run_model_cmds=[
          "python3 -c \"import jax; print('=== TPU DEVICES ===', jax.devices('tpu')); assert len(jax.devices('tpu')) > 0, 'No TPU devices found'\""
      ],
      time_out_in_min=15,
      test_owner=test_owner.JACKY_F,
  ).run()
