"""
SFT training DAG for Llama3.1 70B model.

This DAG runs SFT training validation to test the MaxText supervised
fine-tuning pipeline. The workflow deploys training jobs to GKE clusters
using Pathways, executes the SFT trainer, and validates successful
completion through comprehensive log monitoring of training signals.
"""

import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.post_training.util import validation_util, test_config_util
from xlml.utils.xpk import MAIN_BRANCH
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 21 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_sft"

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 9, 21),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "maxtext",
        "post-training",
        "sft",
        "TPU",
        "v5p-128",
        "nightly",
    ],
    description="SFT training for MaxText pipeline validation.",
    doc_md="""
      # Supervised Fine-Tuning (SFT) MaxText Training

      ### Overview
      This DAG runs Supervised Fine-Tuning (SFT) to validate the MaxText
      fine-tuning pipeline. The workflow tests the complete SFT training stack,
      including infrastructure setup, model initialization, training execution,
      and result validation.

      ### Execution Flow
      1. **Job Launch:** Deploy SFT training jobs to GKE cluster using Pathways infrastructure
      2. **Model Loading:** Initialize Llama3.1 70B model with HuggingFace authentication
      3. **Training Run:** Execute `MaxText.sft.sft_trainer` with JAX proxy/CPU platforms
      4. **Log Validation:** Monitor and check for training completion signals
      5. **Success/Failure:** Report final status based on log validation and job completion

      ### Success Criteria
      The test passes when:
      1. Training jobs complete successfully without errors
      2. Training completion log messages appear in the jax-tpu container logs
      3. No infrastructure failures or container launch issues occur
      4. All training steps execute within expected parameters
    """,
    concurrency=2,
) as dag:
  training_steps = 30
  training_config = test_config_util.SFTTestConfig(
      cluster=XpkClusters.TPU_V5P_128_CLUSTER,
      accelerator="v5p-128",
      slices=[1],  # Single slice for SFT training
      model_name="llama3.1-70b",
      steps=training_steps,
      short_id="msft",
      base_dir=(
          f"{test_config_util.DEFAULT_BUCKET}/llama3.1-70b-Instruct/outputs"
      ),
      tokenizer_path="meta-llama/Llama-3.1-70B-Instruct",
      load_parameters_path=(
          f"{test_config_util.DEFAULT_BUCKET}/llama3.1-70b-Instruct/"
          "scanned-pathways/0/items/"
      ),
      sft_config_path="src/MaxText/configs/sft.yml",
  )
  # HF token retrieved from Airflow Variables for secure credential management
  HF_TOKEN_LLAMA3_1 = models.Variable.get("HF_TOKEN_LLAMA3_1", None)

  for mode, image in test_config_util.POST_TRAINING_DOCKER_IMAGES:
    # TODO: Enable stable mode once a new version of MaxText is available
    if mode == test_config_util.SetupMode.STABLE:
      continue  # Skip stable for RL training tests

    for slice_num in training_config.slices:
      current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
      run_name = f"{mode.value}-{current_datetime}"

      sft_training_command = training_config.generate_sft_training_command(
          run_name=run_name,
          hf_token=HF_TOKEN_LLAMA3_1,
      )
      with TaskGroup(
          group_id=f"sft-{mode.value}-{slice_num}x{training_config.accelerator}"
      ) as group:
        with TaskGroup(group_id="run_training") as training_group:
          start_time = validation_util.generate_timestamp.override(
              task_id="generate_start_time"
          )()

          training_task = gke_config.get_gke_config(
              num_slices=slice_num,
              cluster=training_config.cluster,
              time_out_in_min=30,
              test_name=training_config.short_id,
              run_model_cmds=sft_training_command,
              docker_image=image.value,
              test_owner=test_owner.JACKY_F,
          ).run(
              use_pathways=True,
              xpk_branch=MAIN_BRANCH,
              # skip_post_process=True,
          )

          end_time = validation_util.generate_timestamp.override(
              task_id="generate_end_time"
          )()

          chain(
              start_time,
              training_task,
              end_time,
          )

        with TaskGroup(group_id="validate_training") as validation_group:
          validate_training_logs = validation_util.validate_log_exist.override(
              task_id="validate_training_logs"
          )(
              project_id=training_config.cluster.project,
              location=zone_to_region(training_config.cluster.zone),
              cluster_name=training_config.cluster.name,
              text_filter=f"\"'step': {training_steps}\"",
              namespace="default",
              container_name="jax-tpu",
              pod_pattern=f"{training_config.short_id}.*",
              start_time=start_time,
              end_time=end_time,
          )

        chain(
            training_group,
            validation_group,
        )
