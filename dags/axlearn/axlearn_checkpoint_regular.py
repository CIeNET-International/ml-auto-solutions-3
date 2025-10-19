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

"""A DAG to run all supported ML models with the latest JAX/FLAX version."""

import datetime
from os  import path
from airflow import models
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, RuntimeVersion, XpkClusters
from dags.axlearn.configs import axlearn_config as config
from dags.axlearn.util import test_config_util, validation_util
from airflow.utils.task_group import TaskGroup
from datetime import timedelta
from xlml.utils.gke import zone_to_region


SCHEDULE = "0 21 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "axlearn_reg_save"


with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 30),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "axlearn",
        "regular",
        "nightly"
        "jax0.5.3"
        "python3.10"
    ],
    description="DAG that verifies the axlearn regular (Native) checkpointing saving functionality",
    doc_md="""
      # Axlearn Regular Checkpoint Validation DAG.

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      checkpoint saving when using **Axlearn Native Checkpointer ** features.
      It will check that the checkpoints are being stored in the GCS bucket.
      Also the steps flag controls how many steps the job will run.

      ### Prerequisites
      To run this test, you need an existing cluster.

      ### Procedures
      1.  **Install dependencies for Axlearn:** Setup axlearn CLI and all
      axlearn neccessary dependecies.
      2.  **Run Axelarn Jobsets:** The DAG runs a Axlearn jobset.
      3.  The DAG validates that **Axelarn checkpoints** are being saved correctly
      in the `GCS bucket` by checking bucket and pod logs.
    """,
    concurrency=2,
) as dag:
  checkpointing = test_config_util.Checkpointing(
      name="reg",
      enable_multi_tier_checkpointing=False,
  )
  test_configs = [
    test_config_util.TestConfig(
        cluster=XpkClusters.TPU_V5P_128_CLUSTER_TEST,
        run_name="gke_tpu_single",
        slices=[2],
        instance_type="tpu-v5p-128",
        mesh_type="tpu-v5p-128",
        short_id="axl-sav",
        module="text.gpt.c4_trainer",
        model_config="fuji-7B-v2-flash",
        step=200,
        checkpoint_step=50,
        trainer_dir=test_config_util.DEFAULT_BUCKET,
        data_dir="gs://axlearn-public/tensorflow_datasets",
    ),
  ]
  for mode, image in test_config_util.DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:

        #TODO: Need to discuss. This is really important for the axlearn CLI
        # command.
        run_name = validation_util.get_image_name(
            project_id=test_config.cluster.project,
            path_repository=image.value.split(":")[0],
          )

        start_time = validation_util.generate_timestamp()

        # AXLearn head against JAX 0.5.3
        # Runs Fuji training on v5p-128 in the provided GCP Project
        axlearn_regular_run = config.get_axlearn_tpu_config(
            cluster=test_config.cluster,
            num_slices=slice_num,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-reg",
            run_model_cmds="",
            docker_image=run_name,
            test_owner=test_owner.CAMILO_Q,
          ).run(
            test_configs=test_config,
            axlearn_branch="main",
            run_name=run_name,
            trace_steps=[40, 90, 140, 190, 240]
          )

        end_time = validation_util.generate_timestamp()

        steps_to_validate = test_config.generate_step_to_validate()

        validate_steps = validation_util.validate_checkpoints_save_regular_axlearn(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            run_name=run_name,
            pod_pattern=".*-0",
            start_time=start_time,
            end_time=end_time,
            steps_to_validate=steps_to_validate,
        )
        (
          run_name
          >> start_time
          >> axlearn_regular_run
          >> end_time
          >> validate_steps
        )


