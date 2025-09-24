"""
GRPO (Group Relative Policy Optimization) training DAG for Llama3.1 8B model.

This DAG runs GRPO training validation on a single TPU slice to test
the MaxText RL pipeline. It executes E2E training with a 30-minute
timeout and validates completion through log monitoring.
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import validation_util, test_config_util
from xlml.utils.xpk import MAIN_BRANCH
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 20 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_grpo_rl"

DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_GRPO_RL_IMAGE,
)]

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 9, 21),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "grpo",
        "nightly",
        "rl",
    ],
    description="GRPO training for MaxText RL pipeline validation.",
    doc_md="""
      # GRPO MaxText RL Training

      ### Overview
      This DAG runs GRPO (Group Relative Policy Optimization) training 
      to validate the MaxText reinforcement learning pipeline.

      ### Execution Flow
      1. **Job Launch:** Deploy GRPO training job to GKE cluster using Pathways
      2. **Training Run:** Execute grpo_llama3_demo.py with JAX proxy/CPU platforms
      3. **Log Validation:** Check for "Post GRPO Training" completion signal
      4. **Success/Failure:** Report status based on timeout and log validation

      ### Success Criteria
      The test passes when:
      1. Training job completes within 30-minute timeout
      2. "Post GRPO Training" log message appears in jax-tpu container
      3. No infrastructure or container launch failures occur
    """,
    concurrency=1,
) as dag:
  training_config = test_config_util.TestConfig(
      cluster=XpkClusters.TPU_V5P_128_CLUSTER,
      machine_type="ct5p-hightpu-4t",
      accelerator="v5p-128",
      slices=[1],  # Single slice for GRPO training
      model_name="llama3.1-8b",
      short_id="max-rl",
      step=200,
      local_checkpoint_step=None,
      replicator_backup_time=None,
      base_dir=test_config_util.DEFAULT_BUCKET,
  )

  for mode, image in DOCKER_IMAGES:
    for slice_num in training_config.slices:
      run_name = validation_util.generate_run_name(
          short_id=training_config.short_id,
          checkpointing_type="grpo",
          slice_number=slice_num,
          accelerator=training_config.accelerator,
      )

      # TODO: use secret manager for HF token and extract parameters from script
      grpo_training_command = [
          "HF_TOKEN=$HF_TOKEN JAX_PLATFORMS=proxy,cpu python src/MaxText/examples/grpo_llama3_demo.py",
      ]

      start_time = validation_util.generate_timestamp()

      grpo_training_task = gke_config.get_gke_config(
          num_slices=slice_num,
          cluster=training_config.cluster,
          time_out_in_min=30,
          test_name=f"{training_config.short_id}",
          run_model_cmds=grpo_training_command,
          docker_image=image.value,
          test_owner=test_owner.JACKY_F,
      ).run(
          use_pathways=True,
          xpk_branch=MAIN_BRANCH,
          skip_post_process=True,
      )

      end_time = validation_util.generate_timestamp()

      validate_grpo_training = validation_util.validate_log_exist(
          project_id=training_config.cluster.project,
          location=zone_to_region(training_config.cluster.zone),
          cluster_name=training_config.cluster.name,
          text_filter="Post GRPO Training",
          namespace="default",
          container_name="jax-tpu",
          pod_pattern=f"{training_config.short_id}.*",
          start_time=start_time,
          end_time=end_time,
      )

      (
          run_name
          >> start_time
          >> grpo_training_task
          >> end_time
          >> validate_grpo_training
      )
