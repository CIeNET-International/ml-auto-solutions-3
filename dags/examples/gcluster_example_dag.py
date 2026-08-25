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

"""Example DAG demonstrating Cluster Toolkit (gcluster) orchestration."""

import datetime
from airflow import models
from dags.common import test_owner
from dags.common.vm_resource import GkeClusters
from dags.multipod.configs import gke_config

with models.DAG(
    dag_id="gcluster_example_dag",
    schedule=None,
    tags=["example", "gcluster", "tpu"],
    start_date=datetime.datetime(2026, 8, 1),
    catchup=False,
) as dag:
  test_cmds = (
      "echo 'Running Cluster Toolkit test workload'",
      "python3 -c 'import jax; print(jax.devices())'",
  )

  gcluster_task = gke_config.get_gke_config(
      time_out_in_min=30,
      test_name="gcluster-example",
      run_model_cmds=test_cmds,
      docker_image=(
          "gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:latest"
      ),
      cluster=GkeClusters.TPU_V5P_MLPERF_CLUSTER,
      test_owner=test_owner.JACKY_F,
      priority="very-high",
      use_gcluster=True,
  ).run(
      skip_post_process=True,
  )
