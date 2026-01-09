"""
Gemma3 4B multimodal workflow DAG for MaxText.

This DAG runs the Gemma3 4B multimodal workflow including:
1. Multimodal decode (inference with text+images)
2. Supervised Fine-Tuning (SFT) with visual-question-answering dataset

The workflow validates the MaxText multimodal pipeline,
testing inference and fine-tuning capabilities.

Note: Requires a pre-converted MaxText checkpoint at the specified GCS path.
"""

import datetime

from airflow import models
from airflow.models.baseoperator import chain


from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import Project, Zone, TpuVersion, RuntimeVersion, V6E_GCE_NETWORK, V6E_GCE_SUBNETWORK
from dags.post_training.util import validation_util
from xlml.apis import gcp_config, metric_config, task, test_config

SCHEDULE = "0 21 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_multimodal"
DEFAULT_BUCKET = gcs_bucket.RL_AUTOMATION_BUCKET

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 1, 5),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "maxtext",
        "gemma3",
        "decode",
        "multimodal",
        "sft",
        "post-training",
        "TPU",
        "v6e-8",
        "nightly",
    ],
    description="Gemma3 4B multimodal workflow: decode and SFT.",
    doc_md="""
      # Gemma3 4B Multimodal Workflow

      ### Overview
      This DAG runs the Gemma3 4B multimodal workflow to validate
      the MaxText multimodal pipeline. The workflow includes multimodal
      inference with images and supervised fine-tuning with
      visual-question-answering datasets.

      ### Prerequisites
      A pre-converted MaxText checkpoint must be available at the specified GCS path.

      ### Execution Flow
      1. **TPU Creation:** Create a TPU VM with required specifications
      2. **Environment Setup:** Install dependencies and prepare MaxText environment
      3. **Decode Execution:** Run multimodal decode with image input
      4. **SFT Training:** Fine-tune model on VQA dataset
      5. **Cleanup:** Delete TPU resources

      ### Success Criteria
      The test passes when:
      1. TPU VM is created successfully
      2. Decode command produces valid text output
      3. SFT training completes without errors
      4. No infrastructure or runtime failures occur
    """,
    concurrency=1,
) as dag:
  # HF token retrieved from Airflow Variables for secure credential management
  HF_TOKEN_GEMMA3 = models.Variable.get("HF_TOKEN_CIENET", None)

  # Test configuration
  test_run_name = "gemma3_multimodal_test"
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  base_output_dir = f"{DEFAULT_BUCKET}/gemma3-4b"
  maxtext_ckpt_path = f"{base_output_dir}/checkpoints/0/items"

  # Setup commands for MaxText environment following install_maxtext.md
  setup_script = """
  set -e
  set -x

  # Clone MaxText repository
  if [ ! -d "maxtext" ]; then
    git clone https://github.com/AI-Hypercomputer/maxtext.git
  fi
  cd maxtext

  # Install uv package installer
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  # Create and activate virtual environment
  uv venv --python 3.12 --seed maxtext_venv
  source maxtext_venv/bin/activate

  # Install MaxText and dependencies
  uv pip install -e .[tpu] --resolution=lowest
  python3 -m pip install uv
  
  # Install additional GitHub dependencies
  install_maxtext_github_deps
  """

  # Decode command matching the multimodal.md tutorial
  decode_command = f"""
  set -e
  set -x
  cd maxtext
  source maxtext_venv/bin/activate

  # Environment variables
  export HF_TOKEN={HF_TOKEN_GEMMA3}
  export MAXTEXT_CKPT_GCS_PATH={maxtext_ckpt_path}
  export BASE_OUTPUT_DIRECTORY={base_output_dir}

  # Run Gemma3 decode with multimodal image input
  python -m MaxText.decode \
    MaxText/configs/base.yml \
    model_name=gemma3-4b \
    hf_access_token='$HF_TOKEN' \
    tokenizer_path=src/MaxText/assets/tokenizer.gemma3 \
    load_parameters_path=$MAXTEXT_CKPT_GCS_PATH \
    per_device_batch_size=1 \
    run_name={test_run_name}_decode_{timestamp} \
    max_prefill_predict_length=272 \
    max_target_length=300 \
    steps=1 \
    async_checkpointing=false \
    scan_layers=false \
    use_multimodal=true \
    prompt='Describe image <start_of_image>' \
    image_path='src/MaxText/test_assets/test_image.jpg' \
    attention='dot_product'

  echo "Decode completed successfully"
  """

  # SFT training command
  sft_command = """

  # Run Supervised Fine-Tuning with vision dataset

  python -m MaxText.sft_trainer \
    src/MaxText/configs/sft-vision-chartqa.yml \
    run_name=chartqa-sft_{timestamp} \
    model_name=gemma3-4b \
    tokenizer_path="google/gemma-3-4b-it" \
    hf_access_token='$HF_TOKEN' \
    load_parameters_path=$MAXTEXT_CKPT_GCS_PATH \
    base_output_directory=$BASE_OUTPUT_DIRECTORY \
    per_device_batch_size=1 \
    steps=20 \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    checkpoint_period=1000 \
    scan_layers=False \
    async_checkpointing=True \
    enable_checkpointing=True \
    attention=dot_product \
    max_num_images_per_example=1 \
    dataset_type=hf \
    profiler=xplane

  echo "SFT training completed successfully"
  """

  # Create common test configuration for multimodal workflow
  # Note: run_model_cmds execute sequentially in the same shell, so environment
  # setup and exports from the first command persist for subsequent commands
  multimodal_test = test_config.TpuVmTest(
      test_config.Tpu(
          version=TpuVersion.TRILLIUM,
          cores=8,
          runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
          reserved=False,
          network=V6E_GCE_NETWORK,
          subnetwork=V6E_GCE_SUBNETWORK,
      ),
      test_name=f"{DAG_TEST_NAME}_multimodal",
      set_up_cmds=[setup_script],
      run_model_cmds=(decode_command, sft_command),
      timeout=datetime.timedelta(minutes=120),
      task_owner=test_owner.JACKY_F,
      num_slices=1,
      gcs_subfolder=f"{DEFAULT_BUCKET}/{DAG_TEST_NAME}",
  )

  # Run combined multimodal test using the standard API
  start_time = validation_util.generate_timestamp.override(
      task_id="start_time"
  )()

  multimodal_run = task.run_queued_resource_test(
      task_test_config=multimodal_test,
      task_gcp_config=gcp_config.GCPConfig(
          project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
          zone=Zone.EUROPE_WEST4_A.value,
          dataset_name=metric_config.DatasetOption.XLML_DATASET,
      ),
  )

  end_time = validation_util.generate_timestamp.override(task_id="end_time")()

  # Validation tasks to ensure specific output markers exist in logs
  validate_decode = validation_util.validate_tpu_vm_log_exist.override(
      task_id="validate_decode",
      owner=test_owner.JACKY_F,
  )(
      project_id=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      zone=Zone.EUROPE_WEST4_A.value,
      node_id_pattern=f"{multimodal_test.benchmark_id}.*",
      text_filter='"Decode completed successfully"',
      start_time=start_time,
      end_time=end_time,
  )

  validate_sft = validation_util.validate_tpu_vm_log_exist.override(
      task_id="validate_sft",
      owner=test_owner.JACKY_F,
  )(
      project_id=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      zone=Zone.EUROPE_WEST4_A.value,
      node_id_pattern=f"{multimodal_test.benchmark_id}.*",
      text_filter='"SFT training completed successfully"',
      start_time=start_time,
      end_time=end_time,
  )

  chain(
      start_time,
      multimodal_run,
      end_time,
      validate_decode,
      validate_sft,
  )
