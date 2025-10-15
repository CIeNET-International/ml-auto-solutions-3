# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0 #
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to run workloads with AXLearn."""

from datetime import timedelta
import re
import uuid
from absl import logging
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from kubernetes import client as k8s_client, stream
from xlml.utils import gke

MAIN_BRANCH = "main"

# This function do some hacks to get Axlearn working with Airlfow
# One of them is deleting some unuseful packages in [dev] dependencies.
# We only need to run axlearn CLI
@task
def set_up_axlearn_dpd(
    cluster_name: str,
    project_id: str,
    zone: str,
    branch: str = MAIN_BRANCH,
  ):
  """setup axlearn dependencies."""
  logging.info(f"Using custom branch  ==> {branch}")
  clone_branch = (
      f"git clone --branch {branch} https://github.com/Borklet-Labs/axlearn $HOME/axlearn"
  )

  # Generate Axlearn config file.
  axlearn_config_cmd = f'cat << \'CONFIG_EOF\' > ~/axlearn/.axlearn/axlearn.default.config\n    [gcp]\n_active = "{project_id}:{zone}"\n\n[gcp."{project_id}:{zone}"]\nproject = "{project_id}"\nregion = "{zone[:-2]}"\nzone = "{zone}"\ngke_cluster = "{cluster_name}"\ncluster = "{cluster_name}"\nlabels = "tpu-v5p"\ndocker_repo = "gcr.io/{project_id}"\ndefault_dockerfile = "Dockerfile"\nservice_account_email = "ml-auto-solutions-dev@cloud-tpu-multipod-dev.iam.gserviceaccount.com"\npermanent_bucket = "axlearn-ml-solutions"\nprivate_bucket = "axlearn-ml-solutions"\nttl_bucket = "axlearn-ml-solutions"\nCONFIG_EOF\n'
  create_axlearn_conf = [axlearn_config_cmd.rstrip("\n")]

  # Bypass permission issues for gcloud
  KUBECONFIG_FILE = "/tmp/kubeconfig_gke"

  # Maybe add these lines
  install_python3_cmd = [
      "rm -rf ~/.pyenv",
      "rm -rf ~/my_venv",
      "curl https://pyenv.run | bash",
      f"echo 'export PYENV_ROOT=\"$HOME/.pyenv\"' >> ~/.bashrc ",
      f"echo 'export PYENV_ROOT=\"$HOME/.pyenv\"' >> ~/.profile ",
      f"echo 'export KUBECONFIG=\"{KUBECONFIG_FILE}\"' >> ~/.profile ",
      f"echo 'export KUBECONFIG=\"{KUBECONFIG_FILE}\"' >> ~/.bashrc",
      f"echo '[[ -d $PYENV_ROOT/bin ]] && export PATH=\"$PYENV_ROOT/bin:$PATH\"' >> ~/.bashrc ",
      f"echo '[[ -d $PYENV_ROOT/bin ]] && export PATH=\"$PYENV_ROOT/bin:$PATH\"' >> ~/.profile",
      f"echo 'eval \"$(pyenv init -)\"' >> ~/.bashrc ",
      f"echo 'eval \"$(pyenv init -)\"' >> ~/.profile",
      f"source ~/.bashrc ",
      f"source ~/.profile",
      f"pyenv install 3.10.12 && pyenv global 3.10.12",
      "python -m venv ~/my_venv",
      f"source ~/my_venv/bin/activate",
  ]

  # We need to limit the number of total steps. Default is to 5000.
  # reduce_steps = (
  #   "sed -i 's|max_step = TOTAL_TOKENS\[version\]\[model_size\] // tokens_per_batch|max_step = 100|; /max_step = 100/a save_every_n_steps=500' axlearn/experiments/text/gpt/fuji.py"
  # )

  cmds = [
      "set -xue",
      "rm -rf $HOME/axlearn",
      clone_branch,
      *install_python3_cmd,
      "python --version",
      f"cd ~/axlearn/ ",
      f"pip  install -e '.[core,tpu]'",
      "pip list",
      "pyenv rehash",
      "which axlearn",
  ]
  cmds.append(*create_axlearn_conf)
  hook = SubprocessHook()
  result = hook.run_command(["bash", "-c", ";".join(cmds)])

  assert (
      result.exit_code == 0
  ), f"Set up axlearn dependencies command failed with code {result.exit_code}"


@task
def activate_axlearn(
    cluster_name: str,
    project_id: str,
    zone: str,
):
  """Activate axlearn."""
  cmds = [
      "set -xue",
      "source ~/.bashrc",
      "source ~/.profile",
      "cd ~/axlearn",
      "source ~/my_venv/bin/activate",
      "python --version",
      "which axlearn",
      "echo $KUBECONFIG",
      f"gcloud container clusters get-credentials {cluster_name} \
        --region {zone[:-2]} --project {project_id}",
  ]

  # Execute given commands
  hook = SubprocessHook()
  result = hook.run_command(["bash", "-c", ";".join(cmds)])
  assert (
      result.exit_code == 0
  ), f"Set up axlearn dependencies command failed with code {result.exit_code}"


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate a valid workload ID."""

  # Remove all non-alphanumeric characters, and truncate to ensure the result
  # is less than 40 characters.
  short_benchmark = re.sub(r"[^a-zA-Z0-9-]+", "", benchmark_id)[:32]
  short_id = str(uuid.uuid4())[:8]
  return f"{short_benchmark}{short_id}"


@task(execution_timeout=timedelta(hours=1))
def run_workload_axlearn(
    task_id:str,
    benchmark_id:str,
    workload_id: str,
    gcs_path: str,
    cluster_project: str,
    zone: str,
    cluster_name: str,
    run_name: str,
    run_cmds: str,
    axlearn_branch: str,
    accelerator_type: str = "",
    module: str = "",
    model_config: str = "",
    trainer_dir: str = "",
    num_replicas: int = 1,
    trace_steps: list[str] = None,
):
  """Run workload through axlearn tool."""

  trace_list = (
      ("--trace_at_steps=" + ",".join(map(str, trace_steps)))
      if trace_steps
      else " "
  )
  export_var = [
      f"export CLUSTER={cluster_name}",
      f"export NAME=axlearn-image-test-euh",
      f"export BASTION_TIER=disabled",
      f"export DEFAULT_PROJECT_ID=$(gcloud config get project)",
      "export PROJECT_ID=${PROJECT_ID:-$DEFAULT_PROJECT_ID}",
      f"export INSTANCE_TYPE={accelerator_type}",
      f"export MESH_SELECTOR={accelerator_type}",
      f"export NUM_REPLICAS={num_replicas}",
      f"export MODULE={module}",
      f"export MODEL_CONFIG={model_config}",
      f"export TRAIN_DIR={trainer_dir}",
  ]
  logging.info(
      f" Cluster: {cluster_name} \
      -- num-replicas={num_replicas} \
      --run_name={run_name} \
      --project={cluster_project} \
      --zone={zone} \
      --instance-type={accelerator_type} \
      --module={module} \
      --config={model_config} \
      --trainer_dir={trainer_dir} \
      --data_dir=FAKE \
      --jax_backend=tpu \
      --mesh_selector={accelerator_type} \
      --initialization_timeout=1200 Trace: {trace_list}"
  )

# f"--service_account_email=ml-auto-solutions-dev@cloud-tpu-multipod-dev.iam.gserviceaccount.com "
  workload_create_cmd = (
      f"axlearn gcp launch run --cluster=$CLUSTER "
      f"--runner_name gke_tpu_single "
      f"--name=$NAME "
      f"--instance_type=$INSTANCE_TYPE "
      f"--max_tries=100 "
      f"--num_replicas=$NUM_REPLICAS "
      f"--bundler_spec=allow_dirty=True "
      f"--bundler_type=artifactregistry "
      f"--bundler_spec=image=tpu "
      f"-- \"ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main\" "
      f"--module=$MODULE --config=$MODEL_CONFIG "
      f"--trainer_dir=$TRAIN_DIR "
      f"--data_dir=gs://axlearn-public/tensorflow_datasets "
      f"--mesh_selector=$MESH_SELECTOR "
      f"--jax_backend=tpu "
      f"--initialization_timeout=1200 {trace_list} "
  )

  cmds = [
      "set -xue",
      "source ~/.bashrc",
      "source ~/.profile",
      "source ~/my_venv/bin/activate",
      "cd ~/axlearn",
      "echo $KUBECONFIG",
      "axlearn gcp config activate",
      *export_var,
      workload_create_cmd,
  ]

  # Execute given commands
  hook = SubprocessHook()
  result = hook.run_command(["bash", "-c", ";".join(cmds)])
  assert (
      result.exit_code == 0
  ), f"Error when runninf Axlearn workload check logs to confirm values are correct {result.exit_code}"
