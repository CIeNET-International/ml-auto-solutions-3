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

"""
Airflow DAG for validating AXLearn regular orbax (Native) checkpoint saving
functionality
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.orbax.configs import axlearn_config as config
from dags.orbax.util import test_config_util, validation_util
from xlml.utils.gke import zone_to_region
from xlml.utils import axlearn


SCHEDULE = "* 13 * * *" if composer_env.is_prod_env() else None
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
        "nightly",
        "jax0.5.3",
        "python3.10",
        "TPU",
        "v5p-128",
    ],
    description="""
      DAG that verifies the AXLearn regular (Native) checkpointing saving
      functionality
      """,
    doc_md="""
      # AXLearn Regular Checkpoint Validation DAG.

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      checkpoint saving when using **AXLearn Native Checkpointer ** features.
      It will check that the checkpoints are being stored in the GCS bucket.
      Also the steps flag controls how many steps the job will run.

      ### Prerequisites
      To run this test, you need an existing cluster.

      ### Procedures
      1.  **Install necessary dependencies for AXLearn:** Setup AXLearn CLI and all
      AXLearn necessary dependencies.
      2.  **Run AXLearn JobSets:** The DAG runs a AXLearn JobSet.
      3.  The DAG validates that **AXLearn checkpoints** are being saved correctly
      in the `GCS bucket` by checking bucket and pod logs.
    """,
    concurrency=2,
) as dag:
  checkpointing = test_config_util.Checkpointing(
      name="reg",
      enable_multi_tier_checkpointing=False,
      enable_emergency_checkpoint=False,
  )
  test_configs = [
      test_config_util.TestConfigAXLearn(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          run_name="gke_tpu_single",
          slices=[2],
          instance_type="tpu-v5p-128",
          mesh_type="tpu-v5p-128",
          short_id=f"axlearn-{checkpointing.name}-sav",
          module="text.gpt.c4_trainer",
          label="tpu-v5p",
          model_name="fuji-7B-v2-flash",
          steps=200,
          trainer_dir=test_config_util.DEFAULT_BUCKET_AXLEARN,
          data_dir="gs://axlearn-public/tensorflow_datasets",
          trace_steps=[40, 90, 140, 190],
      ),
  ]
  for mode, image_full_path in test_config_util.DOCKER_IMAGES_AXLEARN:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        # This task will get all remote repository images. Will print
        # latest image full path --> gcr.io/cienet-cmcs/axlearn-custom:<DATE>.
        name_image_full_path_latest = axlearn.get_image_name(
            project_id=test_config.cluster.project,
            path_repository=image_full_path.value.split(":")[0],
        )

        # This task will create the run_name_id always with latest.
        # e.g: gcr.io/cienet-cmcs/axlearn-custom:latest
        run_name_id = axlearn.generate_run_name(
            short_id=test_config.short_id,
            slice_number=slice_num,
            accelerator=test_config.instance_type,
        )

        start_time = validation_util.generate_timestamp()

        # AXLearn head against JAX 0.5.3
        # Runs Fuji training on v5p-128 in the provided GCP Project
        axlearn_regular_run = config.get_axlearn_tpu_config(
            cluster=test_config.cluster,
            num_slices=slice_num,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-reg",
            docker_image=image_full_path.value,
            test_owner=test_owner.CAMILO_Q,
        ).run(
            test_configs=test_config,
            axlearn_branch="main",
            run_name=run_name_id,
        )

        end_time = validation_util.generate_timestamp()

        steps_to_validate = test_config.generate_step_to_validate()

        validate_steps = (
            validation_util.validate_checkpoints_save_regular_axlearn(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                run_name=run_name_id,
                pod_pattern=".*-0",
                start_time=start_time,
                end_time=end_time,
                steps_to_validate=steps_to_validate,
            )
        )

        _ = (
            name_image_full_path_latest
            >> start_time
            >> axlearn_regular_run
            >> end_time
            >> validate_steps
        )
