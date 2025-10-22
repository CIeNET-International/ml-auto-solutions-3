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
import os
from typing import List
from absl import logging
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from xlml.utils import composer


MAIN_BRANCH = "main"


@task
def install_axlearn_cli(
    cluster_name: str,
    project_id: str,
    zone: str,
    branch: str = MAIN_BRANCH,
    commit: str = "df7ed09"
  ):
  """
  Installs the Axlearn CLI and its dependencies into the execution environment
  and configures it to communicate with a specific Google Cloud TPU/GKE cluster.

  This task performs a sequence of low-level shell operations including:
  1. Cloning the Axlearn repository from the specified branch.
  2. Generating the mandatory `axlearn.default.config` file with all GCP
    and cluster-specific details (project ID, zone, GKE cluster name,
    and service accounts). This configures the CLI's target environment.
  3. Installing Python 3.10.12 via `pyenv` and setting up a dedicated
    virtual environment (`~/my_venv`) to isolate dependencies.
  4. Installing the Axlearn framework and its core/TPU dependencies in
    editable mode (`pip install -e '.[core,tpu]'`).
  5. Setting critical environment variables (`KUBECONFIG`, `PYENV_ROOT`)
    to ensure the installed CLI and tools function correctly within the
    distributed environment.

  Args:
      cluster_name: The name of the GKE cluster (e.g., 'tpu-v5p-cluster')
                    that Axlearn will target for job submission.
      project_id: The Google Cloud Project ID where the resources reside.
      zone: The GCP zone (e.g., 'us-central1-a') where the cluster is located.
      branch: The Git branch of the Axlearn repository to clone (default: MAIN_BRANCH).
  """
  clone_branch = (
      f"git clone --branch {branch} https://github.com/apple/axlearn.git $HOME/axlearn"
  )
  checkout_commit = (
    f"git checkout {commit}"
  )
  axlearn_config_cmd = f'cat << \'CONFIG_EOF\' > ~/axlearn/.axlearn/axlearn.default.config\n    [gcp]\n_active = "{project_id}:{zone}"\n\n[gcp."{project_id}:{zone}"]\nproject = "{project_id}"\nregion = "{zone[:-2]}"\nzone = "{zone}"\ngke_cluster = "{cluster_name}"\ncluster = "{cluster_name}"\nlabels = "tpu-v5p"\ndocker_repo = "gcr.io/{project_id}"\ndefault_dockerfile = "Dockerfile"\nservice_account_email = "ml-auto-solutions-dev@cloud-tpu-multipod-dev.iam.gserviceaccount.com"\npermanent_bucket = "axlearn-ml-solutions-bucket"\nprivate_bucket = "axlearn-ml-solutions-bucket"\nttl_bucket = "axlearn-ml-solutions-bucket"\nCONFIG_EOF\n'
  create_axlearn_conf = [axlearn_config_cmd.rstrip("\n")]
  install_python3_cmd = _construct_cmds_cli_install()
  cmds = [
      "set -xue",
      "rm -rf $HOME/axlearn",
      clone_branch,
      *install_python3_cmd,
      "python --version",
      f"cd ~/axlearn/ ",
      checkout_commit,
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

  # TODO: Need to refactor. Since these commands are really hard to configure
  # and takes a long time to to try them in airflow need time to adjust them.
  # Probably we can delete some of them. Or created them on a higher module.
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

  hook = SubprocessHook()
  result = hook.run_command(["bash", "-c", ";".join(cmds)])
  assert (
      result.exit_code == 0
  ), f"Set up axlearn dependencies command failed with code {result.exit_code}"


@task
def generate_workload_id(run_name_workload: str) -> str:
  """Generate a valid workload ID."""

  #TODO: Find a way to run workload with a better name
  #For now the name will be only the tag of the image
  real_run_name_from_image = run_name_workload.split("-")[0]
  logging.info(f"Run_name used: {real_run_name_from_image}")
  return f"{real_run_name_from_image}"


@task(execution_timeout=timedelta(hours=1))
def run_workload_axlearn(
    task_id:str,
    gcs_path: str,
    cluster_project: str,
    cluster_name: str,
    zone: str,
    docker_image: str,
    benchmark_id:str,
    workload_id: str,
    run_name: str,
    steps: int,
    checkpoint_steps: int,
    run_cmds: str,
    accelerator_type: str = "",
    module: str = "",
    model_config: str = "",
    trainer_dir: str = "",
    num_slices: int = 1,
    trace_steps: list[str] = None,
):
  """Run workload through axlearn CLI command."""

  # Log required info for XLML PLX Dashboard
  composer.log_metadata_for_xlml_dashboard({
      "cluster_project": cluster_project,
      "zone": zone,
      "cluster_name": cluster_name,
      "task_id": task_id,
      "workload_id": workload_id,
      "gcs_path": gcs_path,
      "benchmark_id": benchmark_id,
      "docker_image": docker_image,
      "accelerator_type": accelerator_type,
      "num_slices": num_slices,
  })

  # Get  image run name and tag separatedly since we will need it for Axlearn CLI
  # e.g iamge_run_name = axlearn-custom:xynzb3zkn
  image_with_tag = docker_image.split("/")[-1]
  tag = image_with_tag.split(":")[1]
  image_run_name = image_with_tag.split(":")[0]


  export_var = [
      f"export BASTION_TIER=disabled",
      f"export PROJECT_ID={cluster_project}",
  ]
  trace_list = (
      ("--trace_at_steps=" + ",".join(map(str, trace_steps)))
      if len(trace_steps) > 0
      else " "
  )

  # Injection of sed commands to modify at runtime apple/axlearn repo.
  # Need to change:
  #     - Batch size: Depends on the TPU topology
  #     - Logging for debugging purposes
  #     - Comment out XLA flag. Having errors during tests.
  #     - Modify FSDP since depending on topology and Batch Size per Device.

  # Eg.  We need to limit the number of total steps. Default is to 5000.
  #      reduce_steps = (
  #         "sed -i 's|max_step = TOTAL_TOKENS\[version\]\[model_size\] // tokens_per_batch|max_step = 100|; /max_step = 100/a save_every_n_steps=500' axlearn/experiments/text/gpt/fuji.py"
  #         )
  # This will be injected in the following Axlearn command.

  # The main Axlearn command to run.
  workload_create_cmd = (
      f"axlearn gcp launch run --cluster={cluster_name} "
      f"--runner_name gke_tpu_single "
      f"--name={tag} "
      f"--instance_type={accelerator_type} "
      f"--max_tries=100 "
      f"--num_replicas={num_slices} "
      f"--bundler_spec=allow_dirty=True "
      f"--bundler_type=artifactregistry "
      f"--bundler_spec=image={image_run_name} "
      f"-- \""
    f"ulimit -n 1048576; ulimit -c 0; "
    rf"sed -i '/num_kv_heads = None/a \ \ \ \ max_step = {steps}' axlearn/experiments/text/gpt/fuji.py; "
    rf"sed -i 's/^[ \t]*if self.step % 100 == 0 or 0 <= self.step <= 5:/if self.step % 5 == 0:/' axlearn/common/trainer.py; "
    rf"sed -i 's/^[ \t]*mesh_shape=mesh_shape_from_axes(data=-1, fsdp=64)/mesh_shape=mesh_shape_from_axes(data=1, fsdp=128)/' axlearn/experiments/text/gpt/fuji.py; "
    rf"sed -i 's/^\([ \t]*\)train_batch_size = tokens_per_batch \/\/ max_sequence_length/\1train_batch_size = 128/' axlearn/experiments/text/gpt/fuji.py; "
    rf"sed -i 's/\(lr_warmup_steps: int = \)2000/\150/' axlearn/experiments/text/gpt/common.py; "
    rf"sed -i '/max_step=max_step,/a \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ save_every_n_steps={checkpoint_steps},' axlearn/experiments/text/gpt/fuji.py; "
    f"python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main\" "
      f"--module={module} --config={model_config} "
      f"--trainer_dir={trainer_dir}/{run_name} "
      f"--data_dir=gs://axlearn-public/tensorflow_datasets "
      f"--mesh_selector={accelerator_type} "
      f"--jax_backend=tpu "
      f"--initialization_timeout=1200 {trace_list} "
  )

  #TODO Need to find a better way to activate KUBECONFIG env variable. Instead
  # of source ~/.bashrc....
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
  hook = SubprocessHook()
  result = hook.run_command(["bash", "-c", ";".join(cmds)])
  assert (
      result.exit_code == 0
  ), f"Error when running Axlearn workload check logs to confirm values are correct {result.exit_code}"


@task(trigger_rule="all_done")
def clean_up_workload(
    workload_id: str,
    project_id: str,
    zone: str,
    cluster_name: str,
    xpk_branch: str = MAIN_BRANCH,
) -> bool:
  """Delete workload."""
  pass

def _construct_cmds_cli_install()->List[str]:
  """
    Constructs a list of shell commands necessary to install and configure
    pyenv, Python 3.10.12, create a virtual environment, and optionally set
    the KUBECONFIG environment variable.

    The KUBECONFIG and PYENV_ROOT environment variable exports are only
    appended to the user's profile files (~/.bashrc and ~/.profile) if they
    are not already set in the current execution environment.

    Returns:
        List[str]: A list of sequential shell commands ready for execution.
  """
  KUBECONFIG_FILE = "/tmp/kubeconfig_gke"
  env_var_pyenv = [
      f"echo 'export PYENV_ROOT=\"$HOME/.pyenv\"' >> ~/.bashrc ",
      f"echo 'export PYENV_ROOT=\"$HOME/.pyenv\"' >> ~/.profile ",
      f"echo '[[ -d $PYENV_ROOT/bin ]] && export PATH=\"$PYENV_ROOT/bin:$PATH\"' >> ~/.bashrc ",
      f"echo '[[ -d $PYENV_ROOT/bin ]] && export PATH=\"$PYENV_ROOT/bin:$PATH\"' >> ~/.profile",
  ]
  env_var_kube = [
      f"echo 'export KUBECONFIG=\"{KUBECONFIG_FILE}\"' >> ~/.profile ",
      f"echo 'export KUBECONFIG=\"{KUBECONFIG_FILE}\"' >> ~/.bashrc",
  ]
  install_python3_cmd = [
    "rm -rf ~/.pyenv",
    "rm -rf ~/my_venv",
    "curl https://pyenv.run | bash",
  ]

  # Insert the combined environment variable setup commands here
  # for kubeconfig and pyenv_root
  default_kubeconfig = "/home/airflow/composer_kube_config"
  current_kubeconfig = os.getenv("KUBECONFIG")
  current_pyenvconfig = os.getenv("PYENV_ROOT")
  print(f"DEBUG: Current Kubeconfig: '{current_kubeconfig}'\tCurrent PYENV_ROOT {current_pyenvconfig}")

  # We do this so we dont duplicate ENV_VARS in ~/.bashrc file.
  if current_kubeconfig == default_kubeconfig:
    install_python3_cmd.extend(env_var_kube)
  if current_pyenvconfig is None:
    install_python3_cmd.extend(env_var_pyenv)

  # Continue with the rest of the python installation commands
  install_python3_cmd.extend([
      f"echo 'eval \"$(pyenv init -)\"' >> ~/.bashrc ",
      f"echo 'eval \"$(pyenv init -)\"' >> ~/.profile",
      f"source ~/.bashrc ",
      f"source ~/.profile",
      f"pyenv install 3.10.12 && pyenv global 3.10.12",
      "python -m venv ~/my_venv",
      f"source ~/my_venv/bin/activate",
  ])
  return install_python3_cmd
