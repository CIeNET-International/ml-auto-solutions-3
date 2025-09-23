"""
A DAG to run GRPO (Group Relative Policy Optimization) training on Llama3.1 8B model.

This DAG performs GRPO training on the GSM8K math reasoning benchmark
to enhance the model's problem-solving skills on mathematical word problems.
The training is executed on a TPU multi-pod cluster.
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

# TODO: Add Docker image configuration for GRPO training with MaxText and Tunix
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
    description="A DAG to run GRPO training on Llama3.1 8B model for math reasoning.",
    doc_md="""
      # GRPO Llama3.1 8B Training DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of training
      the Llama3.1 8B-IT model using Group Relative Policy Optimization (GRPO)
      on the GSM8K math reasoning benchmark.

      ### Test Scenario
      The DAG performs reinforcement learning training to enhance the model's
      problem-solving skills on mathematical word problems:
      1. Load pre-trained Llama3.1 8B model as policy and reference models
      2. Train using GRPO algorithm on GSM8K dataset
      3. Evaluate model performance before and after training
      4. Save checkpoints and training metrics

      ### Prerequisites
      - An existing TPU cluster configured for MaxText training
      - Access to Llama3.1 8B pre-trained model checkpoints
      - Access to GSM8K dataset
      - Tunix library for GRPO implementation

      ### Training Flow
      1. **Data Preparation:** Load and preprocess GSM8K dataset
         - Extract math word problems and answers
         - Format prompts with reasoning and answer templates
      2. **Model Loading:** Load Llama3.1 8B model as policy and reference
         - Policy model: trainable model with weight updates
         - Reference model: frozen model for KL divergence computation
      3. **GRPO Training:** Train using Group Relative Policy Optimization
         - Generate multiple responses per prompt
         - Compute rewards based on answer correctness and format
         - Update policy using relative advantage within groups
      4. **Evaluation:** Evaluate model on test set
         - Answer accuracy: percentage of correct numerical answers
         - Format accuracy: percentage of correctly formatted responses
         - Partial accuracy: answers within 10% of correct value

      ### Key Parameters
      - **Model:** Llama3.1 8B-Instruct
      - **Dataset:** GSM8K (grade school math word problems)
      - **Training Steps:** 200 batches with configurable epochs
      - **Generation:** 2 responses per prompt with temperature 0.9
      - **Rewards:** Multi-faceted reward system for format and correctness
      - **Optimizer:** AdamW with cosine decay and warmup

      ### Success Criteria
      The training succeeds when:
      1. GRPO training completes without errors
      2. Model checkpoints are saved at specified intervals
      3. Post-training evaluation shows improvement in accuracy metrics
      4. Training metrics are logged successfully
    """,
    concurrency=1,
) as dag:
  # GRPO training configuration
  training_config = test_config_util.TestConfig(
      cluster=XpkClusters.TPU_V5P_128_CLUSTER,
      machine_type="ct5p-hightpu-4t",
      accelerator="v5p-128",
      slices=[1],  # Single slice for GRPO training
      model_name="llama3.1-8b",
      short_id="max-rl", # This short_id can't not exceed 6 characters due to Pathway limitation
      step=200,  # Number of training batches
      local_checkpoint_step=None,
      replicator_backup_time=None,
      base_dir=test_config_util.DEFAULT_BUCKET,
  )

  for mode, image in DOCKER_IMAGES:
    for slice_num in training_config.slices:
      # Generate consistent run name.
      run_name = validation_util.generate_run_name(
          short_id=training_config.short_id,
          checkpointing_type="grpo",
          slice_number=slice_num,
          accelerator=training_config.accelerator,
      )

      # TODO: Extract these parameters into the grpo_llama3_demo.py script (or accept via env/CLI)
      grpo_training_command = [
          f'HF_TOKEN=$HF_TOKEN JAX_PLATFORMS=proxy,cpu src/MaxText/examples/grpo_llama3_demo.py --run_name={run_name} '
          f"--checkpoint_dir=gs://{test_config_util.DEFAULT_BUCKET}/{DAG_TEST_NAME}/{run_name} "
          f"--log_dir=gs://{test_config_util.DEFAULT_BUCKET}/{DAG_TEST_NAME}/{run_name}/logs "
          f"--profile_dir=gs://{test_config_util.DEFAULT_BUCKET}/{DAG_TEST_NAME}/{run_name}/profiles "
          f"--max_steps=200",
      ]

      start_time = validation_util.generate_timestamp()

      grpo_training_task = gke_config.get_gke_config(
          num_slices=slice_num,
          cluster=training_config.cluster,
          time_out_in_min=180,  # 3 hours timeout for GRPO training
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

      # Validate GRPO training completion by checking training logs
      validate_grpo_training = validation_util.validate_log_exist(
          project_id=training_config.cluster.project,
          location=zone_to_region(training_config.cluster.zone),
          cluster_name=training_config.cluster.name,
          text_filter="Post GRPO Training:",
          namespace="default",
          container_name="train",
          pod_pattern="grpo-llama3",
          start_time=start_time,
          end_time=end_time,
      )

      # Validate model evaluation metrics improvement
      validate_model_performance = validation_util.validate_log_exist(
          project_id=training_config.cluster.project,
          location=zone_to_region(training_config.cluster.zone),
          cluster_name=training_config.cluster.name,
          text_filter="accuracy=",
          namespace="default",
          container_name="train",
          pod_pattern="grpo-llama3",
          start_time=start_time,
          end_time=end_time,
      )

      (
          run_name
          >> start_time
          >> grpo_training_task
          >> end_time
          >> validate_grpo_training
          >> validate_model_performance
      )
