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
import tempfile
import os
from typing import List
from absl import logging
from kubernetes import client as k8s_client
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.exceptions import AirflowFailException
from xlml.utils import composer, gke


MAIN_BRANCH = "main"


LOGGING_URL_FORMAT = (
    "https://pantheon.corp.google.com/logs/query;"
    + "query=resource.type%3D%22k8s_container%22%0A"
    + "resource.labels.project_id%3D%22{project}%22%0A"
    + "resource.labels.location%3D%22{region}%22%0A"
    + "resource.labels.cluster_name%3D%22{cluster}%22%0A"
    + "resource.labels.namespace_name%3D%22default%22%0A"
    + "labels.k8s-pod%2Fjobset_sigs_k8s_io%2F"
    + "jobset-name%3D%22{workload_id}%22%20severity%3E%3DDEFAULT;"
    + "storageScope=project;duration=P7D?e=13803378&"
    + "mods=allow_workbench_image_override&project={project}"
)

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

  # Clone commit ping to 0.5.3 Jax version.
  clone_branch = (
      f"git clone --branch {branch} https://github.com/apple/axlearn.git $HOME/axlearn"
  )
  checkout_commit = (
    f"git checkout {commit}"
  )
  axlearn_config_cmd = f'cat << \'CONFIG_EOF\' > ~/axlearn/.axlearn/axlearn.default.config\n    [gcp]\n_active = "{project_id}:{zone}"\n\n[gcp."{project_id}:{zone}"]\nproject = "{project_id}"\nregion = "{zone[:-2]}"\nzone = "{zone}"\ngke_cluster = "{cluster_name}"\ncluster = "{cluster_name}"\nlabels = "tpu-v5p"\ndocker_repo = "gcr.io/{project_id}"\ndefault_dockerfile = "Dockerfile"\nservice_account_email = "ml-auto-solutions-dev@cloud-tpu-multipod-dev.iam.gserviceaccount.com"\npermanent_bucket = "axlearn-ml-solutions-bucket"\nprivate_bucket = "axlearn-ml-solutions-bucket"\nttl_bucket = "axlearn-ml-solutions-bucket"\nCONFIG_EOF\n'
  create_axlearn_conf = [axlearn_config_cmd.rstrip("\n")]

  # Clean previous venv environments and set up WITHOUT writing to .bashrc
  clean_cmds = _clean_env()
  env_var_cmds = _setup_env_var()
  venv_cmds = _create_venv()

  cmds = [
      "set -xue",
      *clean_cmds,
      *env_var_cmds,
      f"pyenv install 3.10.12 && pyenv global 3.10.12",
      *venv_cmds,
      clone_branch,
      f"cd ~/axlearn/ ",
      checkout_commit,
      f"pip  install -e '.[core,tpu]'",
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
def generate_workload_id(run_name_workload: str) -> str:
  """Generate a valid workload ID."""

  #TODO: Find a way to run workload with a better name
  #For now the name will be only the tag of the image
  real_run_name__running = run_name_workload.split("-")[0]
  logging.info(f"Run_name used: {real_run_name__running}")
  return f"{real_run_name__running}"


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
    data: int,
    fsdp: int,
    train_batch_size: int,
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

  # The main Axlearn command to run.
  image_with_tag = docker_image.split("/")[-1]
  tag = image_with_tag.split(":")[1]
  image_run_name = image_with_tag.split(":")[0]

  trace_list = (
      ("--trace_at_steps=" + ",".join(map(str, trace_steps)))
      if len(trace_steps) > 0
      else " "
  )
  workload_create_cmd = (
    f"export BASTION_TIER=disabled && export PROJECT_ID={cluster_project} && axlearn gcp launch run --cluster={cluster_name} "
    f"--runner_name gke_tpu_single "
    f"--name={tag} "
    f"--instance_type={accelerator_type} "
    f"--max_tries=20 "
    f"--num_replicas={num_slices} "
    f"--bundler_spec=allow_dirty=True "
    f"--bundler_type=artifactregistry "
    f"--bundler_spec=image={image_run_name} "
    f"-- \""
  f"ulimit -n 1048576; ulimit -c 0; "
  rf"sed -i '/num_kv_heads = None/a \ \ \ \ max_step = {steps}' axlearn/experiments/text/gpt/fuji.py; "
  rf"sed -i 's/^[ \t]*if self.step % 100 == 0 or 0 <= self.step <= 5:/if self.step % 5 == 0:/' axlearn/common/trainer.py; "
  rf"sed -i 's/^[ \t]*mesh_shape=mesh_shape_from_axes(data=-1, fsdp=64)/mesh_shape=mesh_shape_from_axes(data={data}, fsdp={fsdp})/' axlearn/experiments/text/gpt/fuji.py; "
  rf"sed -i 's/^\([ \t]*\)train_batch_size = tokens_per_batch \/\/ max_sequence_length/\1train_batch_size = {train_batch_size}/' axlearn/experiments/text/gpt/fuji.py; "
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
  env_cmds = _setup_env_var()
  cmds = [
      "set -xue",
      *env_cmds,
      "cd ~/axlearn",
      f"source ~/my_venv/bin/activate",
      "axlearn gcp config activate",
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
    region: str,
    cluster_name: str,
) -> bool:
  """Delete jobset."""
  core_api = _get_core_api_client(project_id, region, cluster_name)
  pods = _list_workload_pods(core_api, workload_id)

  if any(pod.status.phase in ["Pending", "Running"] for pod in pods.items):
    logging.info("At least one pod has yet to complete.")
    return False

  try:
    for pod in pods.items:
      if pod.status.phase == "Failed":
        # Don't keep retrying if the pod has failed
        pass
      elif pod.status.phase in ["Unknown"]:
        raise RuntimeError(f"Bad pod phase: {pod.status.phase}")

  finally:
    #TODO: Need to complete logic to first kill process in Airflow pod and
    # second delete jobset. In this order otherwise does not work.
    logging.info("All pods are complete ")
  logging.info("All pod(s) phase are succeeded.")
  return True




def _create_venv()->List[str]:
  """
    Generates a list of shell commands to create and activate a Python virtual environment.
    The virtual environment is created in the user's home directory under '~/my_venv'.
  """
  return [
      "python -m venv ~/my_venv",
      f"source ~/my_venv/bin/activate",
      "python --version",
  ]


def _clean_env()->List[str]:
  """
    Generates a list of shell commands to clean up existing directories and
    set up a fresh pyenv installation.

    The commands perform destructive removal of existing directories before
    downloading and running the pyenv installation script.
  """
  return [
      "rm -rf $HOME/axlearn",
      "rm -rf ~/.pyenv",
      "rm -rf ~/my_venv",
      "curl https://pyenv.run | bash",
  ]


def _setup_env_var()->List[str]:
  """
    Constructs a list of shell commands necessary to install and configure
    pyenv, Python 3.10.12, create a virtual environment.
  """
  return [
      "export KUBECONFIG=/tmp/kubeconfig_gke",
      f'export PYENV_ROOT="$HOME/.pyenv"',
      f'export PATH="$PYENV_ROOT/bin:$PATH"',
      f'eval "$(pyenv init -)"',
  ]


def _get_core_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.CoreV1Api:
  """Create a core API client for the given cluster."""
  client = gke.get_authenticated_client(project_id, region, cluster_name)

  # Initilize the client
  core_api = k8s_client.CoreV1Api(client)
  logging.info("Successful initilize k8s client from cluster response.")
  return core_api


def _list_workload_pods(
    core_api: k8s_client.CoreV1Api, workload_id: str
) -> k8s_client.V1PodList:
  """List all pods for the given workload."""
  logging.info(f"Getting pods for workload_id: {workload_id}")
  pods = core_api.list_namespaced_pod(
      label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      namespace="default",
  )
  return pods
