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

"""Utilities to construct configs for maxtext inference DAG."""

import datetime
import json
from typing import Dict
from xlml.apis import gcp_config, metric_config, task, test_config
from dags.common import test_owner
from dags.multipod.configs import common
from dags.common.vm_resource import TpuVersion, Project, RuntimeVersion

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value
VENV_DIR = "venv-312"


def get_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    test_mode: common.SetupMode,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = True,
    num_slices: int = 1,
    model_configs: Dict = {},
    maxtext_branch: str = "",
    jetstream_branch: str = "",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = (
      "pip install --upgrade pip",
      # Download jetstream and maxtext
      f"if [ ! -d maxtext ]; then git clone {maxtext_branch} https://github.com/google/maxtext.git; fi",
      f"if [ ! -d JetStream ]; then git clone {jetstream_branch} https://github.com/google/JetStream.git; fi",
      "sudo apt-get -y update",
      "sudo apt-get -y install jq",
      "cd JetStream && pip install -e . && cd benchmarks && pip install -r requirements.in",
      "pip install torch --index-url https://download.pytorch.org/whl/cpu",
      "cd ..",
  )

  # Setup uv and and instll maxtext from PypI (Recommended by the maxtext repo: https://maxtext.readthedocs.io/en/latest/guides/install_maxtext.html)
  setup_maxtext_cmds = (
      # Install uv using the standalone installer and add to PATH permanently in the current session
      "curl -LsSf https://astral.sh/uv/install.sh | sh",
      'export PATH="$HOME/.local/bin:$PATH"',
      # Make the PATH change permanent for subsequent sessions (optional, but good practice)
      "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc",
      "source ~/.bashrc",
      f"uv venv --python 3.12 {VENV_DIR} --seed --clear",
      f"source {VENV_DIR}/bin/activate",
      "pip install uv",
      "uv pip install maxtext --resolution=lowest",
      "install_maxtext_github_deps",
      "uv pip install rouge-score",
      "sudo apt-get install -y git-lfs",
      "git lfs install",
  )

  set_up_cmds += setup_maxtext_cmds

  additional_metadata_dict = {
      "model_name": f"{model_configs['model_name']}",
      "model_mode": f"{model_configs['model_mode']}",
      "quant_mode": f"{model_configs['quant_mode']}",
      "tokenizer": f"{model_configs['tokenizer']}",
      "weight_dtype": f"{model_configs['weight_dtype']}",
      "scan_layers": f"{model_configs['scan_layers']}",
      "max_prefill_predict_length": f"{model_configs['max_prefill_predict_length']}",
      "max_target_length": f"{model_configs['max_target_length']}",
      "attention": f"{model_configs['attention']}",
      "ici_fsdp_parallelism": f"{model_configs['ici_fsdp_parallelism']}",
      "ici_autoregressive_parallelism": f"{model_configs['ici_autoregressive_parallelism']}",
      "ici_tensor_parallelism": f"{model_configs['ici_tensor_parallelism']}",
      "checkpoint": f"{model_configs['checkpoint']}",
      "quantization": f"{model_configs['quantization']}",
      "quantize_kvcache": f"{model_configs['quantize_kvcache']}",
      "weight_quant_dtype": f"{model_configs['quantization'] if model_configs['quantization'] else 'bf16'}",
      "kv_quant_dtype": f"{model_configs['kv_quant_dtype']}",
      "per_device_batch_size": f"{model_configs['per_device_batch_size']}",
      "dataset": f"{model_configs['dataset']}",
      "dataset_path": f"{model_configs['dataset_path']}",
      "request_rate": f"{model_configs['request_rate']}",
      "num_prompts": f"{model_configs['num_prompts']}",
      "max_output_length": f"{model_configs['max_output_length']}",
      "warmup_mode": f"{model_configs['warmup_mode']}",
      "prefill_cache_axis_order": f"{model_configs['prefill_cache_axis_order']}",
      "ar_cache_axis_order": f"{model_configs['ar_cache_axis_order']}",
      "compute_axis_order": f"{model_configs['compute_axis_order']}",
      "reshape_q": f"{model_configs['reshape_q']}",
      "kv_quant_axis": f"{model_configs['kv_quant_axis']}",
  }

  # Let gcs path be directly used, else use maxtext/assets dir
  if not model_configs["tokenizer"].startswith("gs://"):
    tokenizer_path = (
        f"src/maxtext/assets/tokenizers/{model_configs['tokenizer']}"
    )
    full_tokenizer_path = (
        f"maxtext/src/maxtext/assets/tokenizers/{model_configs['tokenizer']}"
    )
  else:
    tokenizer_path = model_configs["tokenizer"]
    full_tokenizer_path = model_configs["tokenizer"]

  run_model_cmds = (
      # Start virtual environment
      f"source {VENV_DIR}/bin/activate",
      "wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json > /dev/null 2>&1",
      # Get commit hash of the maxtext and jetstream repos
      f"export METADATA_DICT='{json.dumps(additional_metadata_dict)}'",
      'cd maxtext && export MAXTEXT_COMMIT_HASH=$(git log -1 --format="%H") && cd ..',
      'cd JetStream && export JETSTREAM_COMMIT_HASH=$(git log -1 --format="%H") && cd ..',
      'export METADATA_DICT=$(jq -c \'. + { "maxtext_commit_hash": $newVal}\' --arg newVal ${MAXTEXT_COMMIT_HASH} <<<"$METADATA_DICT")',
      'export METADATA_DICT=$(jq -c \'. + { "jetstream_commit_hash": $newVal}\' --arg newVal ${JETSTREAM_COMMIT_HASH} <<<"$METADATA_DICT")',
      ### Benchmark
      "cd maxtext",
      # Start JetStream MaxText server in the background
      f"""python3 -m MaxText.maxengine_server \\
        src/maxtext/configs/inference/inference_jetstream.yml \\
        model_name='{model_configs['model_name']}' \\
        tokenizer_path='{tokenizer_path}' \\
        weight_dtype='{model_configs['weight_dtype']}' \\
        scan_layers='{model_configs['scan_layers']}' \\
        max_prefill_predict_length='{model_configs['max_prefill_predict_length']}' \\
        max_target_length='{model_configs['max_target_length']}' \\
        attention='{model_configs['attention']}' \\
        ici_fsdp_parallelism='{model_configs['ici_fsdp_parallelism']}' \\
        ici_autoregressive_parallelism='{model_configs['ici_autoregressive_parallelism']}' \\
        ici_tensor_parallelism='{model_configs['ici_tensor_parallelism']}' \\
        load_parameters_path='{model_configs['checkpoint']}' \\
        quantization='"{model_configs.get('quantization') or ""}"' \\
        quantize_kvcache='{model_configs['quantize_kvcache']}' \\
        kv_quant_dtype='"{model_configs.get('kv_quant_dtype') or ""}"' \\
        per_device_batch_size='{model_configs['per_device_batch_size']}' \\
        prefill_cache_axis_order='{model_configs['prefill_cache_axis_order']}' \\
        ar_cache_axis_order='{model_configs['ar_cache_axis_order']}' \\
        compute_axis_order='{model_configs['compute_axis_order']}' \\
        reshape_q='{model_configs['reshape_q']}' \\
        kv_quant_axis='"{model_configs.get('kv_quant_axis') or ""}"' &""",
      "cd ..",
      # Give server time to start
      f"sleep {model_configs['sleep_time']}",
      "cd JetStream",
      # Since we change to the Jetstream dir as root, we need to change the ownership of the dir to the user running the script
      "sudo chown -R $USER:$USER $HOME/JetStream",
      "git lfs pull",
      "cd ..",
      # Run benchmark, run eval, save benchmark and eval results, and save predictions to /tmp/request-outputs.json
      f"""python JetStream/benchmarks/benchmark_serving.py \
      --tokenizer {full_tokenizer_path} \
      --model {model_configs['model_name']} \
      --dataset {model_configs['dataset']} \\"""
      + (
          f"""--dataset-path {model_configs['dataset_path']} \\"""
          if model_configs["dataset_path"] != ""
          else ""
      )
      + f"""--request-rate {model_configs['request_rate']} \
      --num-prompts {model_configs['num_prompts']}  \
      --max-output-length {model_configs['max_output_length']} \
      --warmup-mode {model_configs['warmup_mode']} \
      --save-result \
      --additional-metadata-metrics-to-save ${{METADATA_DICT}} \
      --save-request-outputs \
      --run-eval {model_configs['run_eval']}""",
      'export BENCHMARK_OUTPUT=$(find . -name "*JetStream*" -type f -printf "%T@ %Tc %p\n" | sort -n | head -1 | awk \'NF>1{print $NF}\')',
      # Stop JetStream server
      "kill -9 %%",
      # Upload results (in jsonlines format) to GCS to be post-processed into
      # our BigQuery table
      "mv ${BENCHMARK_OUTPUT} metric_report.jsonl",
      f"gcloud storage cp metric_report.jsonl {metric_config.SshEnvVars.GCS_OUTPUT.value}",
      f"gcloud storage cp /tmp/request-outputs.json {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.XIANG_S,
      num_slices=num_slices,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/maxtext",
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )
  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
