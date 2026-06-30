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

"""
A DAG to run MaxText Pathways workloads on v6e-32.
"""
import datetime
from airflow.models.baseoperator import chain
from airflow import DAG
from dags import composer_env
from dags.maxtext_pathways.configs import jobset_util_pw as jobset
from dags.tpu_observability.utils.jobset_util import Workload
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from dags.tpu_observability.utils import node_pool_util as node_pool
from xlml.utils import xpk

# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 22 * * *" if composer_env.is_prod_env() else None

PROJECT_ID = "cienet-cmcs"
REGION = "us-central1"
CLUSTER = "pw-v6e-32x4"
MODEL = "llama3.1-8b"
device = "v6e-32"

with DAG(
    f"pw_elastic_training_{MODEL}",
    schedule_interval=None,
    start_date=datetime.datetime(2026, 3, 1),
    tags=["pw"],
) as dag:

    dataset_types = ["synthetic", "tfds"]
    slices_type = [[1, 1], [4, 0]]
    last_group = None
    for slices in slices_type:
        for dtype in dataset_types:
            group_id = f"{device}x{slices[0]}-{slices[1]}-{dtype}"
            with TaskGroup(group_id=group_id) as current_group:
                jobset_config = jobset.JobSet(
                    jobset_name=group_id,
                    namespace="default",
                    dataset_type=dtype,
                    model=MODEL,
                    num_chips=32,
                    max_restarts=4,
                    hardware_slice=slices[0],
                    spare_slices=slices[1],
                    steps=15,
                    replicated_job_name="tpu-job-slice",
                    backoff_limit=0,
                    completions=8,
                    parallelism=8,
                    tpu_accelerator_type="tpu-v6e-slice",
                    tpu_topology="4x8",
                    container_name="jax-tpu-worker",
                    image="python:3.11",
                    tpu_cores_per_pod=4,
                )
                cluster_info = node_pool.Info(
                    project_id=PROJECT_ID,
                    cluster_name=CLUSTER,
                    region=REGION,
                )

                start_workload = jobset.run_workload(
                    node_pool=cluster_info,
                    yaml_config=jobset_config.generate_yaml(
                        workload_script=Workload.JAX_TPU_BENCHMARK
                    ),
                    namespace=jobset_config.namespace,
                )
                wait_for_workload_start = xpk.wait_for_workload_start.override(
                    timeout=2400,
                )(
                    workload_id=jobset_config.jobset_name,
                    project_id=PROJECT_ID,
                    region=REGION,
                    cluster_name=CLUSTER,
                )
                wait_for_workload_completion = xpk.wait_for_workload_completion.override(
                    timeout=1800,
                )(
                    workload_id=jobset_config.jobset_name,
                    project_id=PROJECT_ID,
                    region=REGION,
                    cluster_name=CLUSTER,
                )
                cleanup_workload = jobset.end_workload.override(
                    task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
                )(
                    node_pool=cluster_info,
                    jobset_name=jobset_config.jobset_name,
                    namespace=jobset_config.namespace,
                ).as_teardown(
                    setups=start_workload
                )

                chain(
                    start_workload,
                    wait_for_workload_start,
                    wait_for_workload_completion,
                    cleanup_workload,
                )

            if last_group:
                _ = last_group >> current_group
            last_group = current_group
