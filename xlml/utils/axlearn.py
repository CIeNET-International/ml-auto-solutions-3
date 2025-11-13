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

from datetime import datetime
from typing import List
from absl import logging
from airflow.decorators import task
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
def generate_workload_id(run_name_workload: str) -> str:
  """Generate a valid workload ID."""

  #TODO: Find a way to run workload with a better name
  #For now the name will be only the tag of the image
  real_run_name__running = run_name_workload.split("-")[0]
  logging.info(f"Run_name used: {real_run_name__running}")
  return f"{real_run_name__running}"

def build_axlearn_cmd(
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
  """Run workload through AXLearn CLI command."""

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

  # Get  image run name and tag separatedly since we will need it for AXLearn CLI
  # Here tag always gonna be latest.
  image_with_tag = docker_image.split("/")[-1]
  tag = image_with_tag.split(":")[1]
  image_run_name = image_with_tag.split(":")[0]


  # Create a run_name id for output directory.
  outpu_dir_name = "-".join(run_name.split("-")[:4])

  export_var = [
      f"&& export BASTION_TIER=disabled",
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
  # This will be injected in the following AXLearn command.

  # The main AXLearn command to run.
  workload_create_cmd = (
      f"axlearn gcp launch run --cluster={cluster_name} "
      f"--runner_name gke_tpu_single "
      f"--name={tag} "
      f"--instance_type={accelerator_type} "
      f"--max_tries=3 "
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
      f"--trainer_dir={trainer_dir}/{outpu_dir_name}/{run_name} "
      f"--data_dir=gs://axlearn-public/tensorflow_datasets "
      f"--mesh_selector={accelerator_type} "
      f"--jax_backend=tpu "
      f"--initialization_timeout=1200 {trace_list} "
  )

  cmds = [
      *export_var,
      workload_create_cmd,
  ]
  final_command_string = ' && '.join(cmds)
  return final_command_string


def create_axlearn_config_cmd(
    cluster_name: str,
    project_id: str,
    zone: str,
)-> List[str]:
  cmds = ""
  cmds += "mkdir -p .axlearn/"
  # config_axlearn = f'cat << \'CONFIG_EOF\' > ~/.axlearn/axlearn.default.config\n    [gcp]\n_active = "{project_id}:{zone}"\n\n[gcp."{project_id}:{zone}"]\nproject = "{project_id}"\nregion = "{zone[:-2]}"\nzone = "{zone}"\ngke_cluster = "{cluster_name}"\ncluster = "{cluster_name}"\nlabels = "tpu-v5p"\ndocker_repo = "gcr.io/{project_id}"\ndefault_dockerfile = "Dockerfile"\nservice_account_email = "ml-auto-solutions-dev@cloud-tpu-multipod-dev.iam.gserviceaccount.com"\npermanent_bucket = "axlearn-bucket-multipod"\nprivate_bucket = "axlearn-bucket-multipod"\nttl_bucket = "axlearn-bucket-multipod"\nCONFIG_EOF\n'
  config_content = f"""[gcp]
_active = "{project_id}:{zone}"

[gcp."{project_id}:{zone}"]
project = "{project_id}"
region = "{zone[:-2]}"
zone = "{zone}"
gke_cluster = "{cluster_name}"
cluster = "{cluster_name}"
labels = "tpu-v5p"
docker_repo = "gcr.io/{project_id}"
default_dockerfile = "Dockerfile"
service_account_email = "ml-auto-solutions-dev@cloud-tpu-multipod-dev.iam.gserviceaccount.com"
permanent_bucket = "axlearn-bucket-multipod"
private_bucket = "axlearn-bucket-multipod"
ttl_bucket = "axlearn-bucket-multipod"
"""
  escaped_config = config_content.replace('"', r'\"').replace('\n', r'\n')
  cmds += f" && printf \"{escaped_config}\" > ~/.axlearn/axlearn.default.config"
  return cmds

def setup_cmds(
    cluster_name: str,
    project_id: str,
    zone: str,
):
  cmds =[
  '&& export PYTHONPATH=$PYTHONPATH:/root',
  'axlearn gcp config activate',
  'which axlearn',
  'apt-get install -y kubectl google-cloud-sdk-gke-gcloud-auth-plugin',
  f'gcloud container clusters get-credentials {cluster_name} \
        --region {zone[:-2]} --project {project_id}'
  ]
  final_command_string = ' && '.join(cmds)
  return final_command_string


def generate_run_name(
    short_id: str,
    slice_number: int,
    accelerator: str,
    name_image: str,
) -> str:
  """
  Generates a unique run name for a MaxText run based on given parameters.

  The function creates a formatted string that includes a short identifier,
  the number of slices, the accelerator type, and the current timestamp. This
  run name is useful for uniquely identifying a specific training run,
  especially for checkpointing and logging purposes.

  Args:
      short_id: A short identifier for the specific model or experiment.
      checkpointing_type: The name of the checkpointing strategy (e.g., 'emc').
      slice_number: The number of TPU slices used for the training run.
      accelerator: The type of accelerator used (e.g., 'tpu-v4').

  Returns:
      A string formatted as '{short_id}-mtc-{slice_number}x-{accelerator}-{timestamp}'.
  """

  run_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  run_name_id = f"{name_image.split(':')[1]}-{short_id}-{slice_number}x-{accelerator}-{run_time}"
  return run_name_id
