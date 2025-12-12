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
import os
import re
from absl import logging
import textwrap
import json

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.exceptions import AirflowFailException

from dags import composer_env
from xlml.utils import gke
from xlml.utils import composer


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
def setup_airflow_cluster_context() -> None:
  """Get credential for in-cluster to setup CLI AXLearn command."""

  cluster_name = os.environ["COMPOSER_GKE_NAME"]
  project_id = os.environ["GCP_PROJECT"]
  region = os.environ["COMPOSER_LOCATION"]

  logging.info(f"{' LOGGING AIRFLOW CLUSTER ':=^80}")
  logging.info("CLUSTER_NAME: %s", cluster_name)
  logging.info("PROJECT_ID: %s", project_id)
  logging.info("REGION: %s", region)

  hook = SubprocessHook()
  result = hook.run_command([
      "bash",
      "-c",
      (
          f"gcloud container clusters get-credentials {cluster_name} "
          f"--region {region}  --project {project_id}"
      ),
  ])
  assert (
      result.exit_code == 0
  ), f"XPK clean-up failed with code {result.exit_code}"


def build_axlearn_cmd(
    task_id: str,
    gcs_path: str,
    cluster_project: str,
    cluster_name: str,
    zone: str,
    docker_image: str,
    benchmark_id: str,
    workload_id: str,
    accelerator_type: str = "",
    module: str = "",
    model_config: str = "",
    trainer_dir: str = "",
    num_slices: int = 1,
    trace_steps: list[str] = None,
) -> str:
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

  # TODO: project ID
  # valid command:
  # gcloud container images add-tag gcr.io/cloud-tpu-multipod-dev/axlearn-custom:2025-11-21 gcr.io/cloud-tpu-multipod-dev/axlearn-custom:automation-dev-2025-12-12-11-43-2 --project=cloud-tpu-multipod-dev --quiet
  update_image_tag_cmd = (
      "gcloud artifacts docker tags add "
      f"{docker_image}:latest "
      f"{docker_image}:{workload_id} "
      "--project=cloud-tpu-multipod-dev "
      "--quiet"
  )

  # TODO: can we name the folder? a fixed one
  # The output directory will be construct from the workload_id
  # workload_id: "automation-prod-2025-11-19-05-15"
  # output_dir_name: automation-prod
  regex = r"^(?P<output_dir_name>(?:[^-]+-){2}[^-]+)"
  match = re.search(regex, workload_id)
  if not match:
    raise AirflowFailException(f"Invalid run name format: {workload_id}")
  output_dir_name = match.group("output_dir_name")

  export_var = (
      "export BASTION_TIER=disabled; "
      f"export PROJECT_ID={cluster_project}"
  )
  trace_list = ""
  if len(trace_steps) > 0:
    trace_list = "--trace_at_steps=" + ",".join(map(str, trace_steps))

  # Injection of sed commands to modify at runtime apple/axlearn repo.
  # Need to change:
  #     - Batch size: Depends on the TPU topology
  #     - Logging for debugging purposes
  #     - Comment out XLA flag. Having errors during tests.
  #     - Modify FSDP since depending on topology and Batch Size per Device.

  # Eg.  We need to limit the number of total steps. Default is to 5000.
  #      reduce_steps = (
  #          "sed -i 's|max_step = TOTAL_TOKENS\[version\]\[model_size\] // "
  #          "tokens_per_batch|max_step = 100|; /max_step = 100/a "
  #          "save_every_n_steps=500' axlearn/experiments/text/gpt/fuji.py"
  #      )
  # This will be injected in the following AXLearn command.

  # The main AXLearn command to run.
  workload_create_cmd = (
      f"axlearn gcp launch run --cluster={cluster_name} "
      f"--runner_name gke_tpu_single "
      f"--name={workload_id} "
      f"--instance_type={accelerator_type} "
      f"--max_tries=10 "
      f"--num_replicas={num_slices} "
      f"--bundler_spec=allow_dirty=True "
      f"--bundler_type=artifactregistry "
      f"--bundler_spec=image={docker_image} "
      f'-- "'
      f"ulimit -n 1048576; ulimit -c 0; "
      f"python3 -c 'import jax; jax.devices()'; "
      f"python3 -m axlearn.common.launch_trainer_main"
      f'" '
      f"--module={module} --config={model_config} "
      f"--trainer_dir={trainer_dir}/{output_dir_name}/{workload_id} "
      f"--data_dir=gs://axlearn-public/tensorflow_datasets "
      f"--mesh_selector={accelerator_type} "
      f"--jax_backend=tpu "
      f"--initialization_timeout=1200 {trace_list} "
  )

  # TODO: type incompatible
  return [export_var, update_image_tag_cmd, workload_create_cmd]


def create_axlearn_config_cmd(
    cluster_name: str, project_id: str, zone: str, label: str = "tpu-v5p"
) -> str:
  """
  Generates a shell command string to create the .axlearn.default.config
  configuration file using textwrap.dedent for clean multiline string
  formatting.
  """
  cmds = ""
  cmds += "mkdir -p ~/.axlearn/"

  config_content = textwrap.dedent(
      f"""
      [gcp]
      _active = "{project_id}:{zone}"

      [gcp."{project_id}:{zone}"]
      project = "{project_id}"
      region = "{gke.zone_to_region(zone)}"
      zone = "{zone}"
      gke_cluster = "{cluster_name}"
      cluster = "{cluster_name}"
      labels = "{label}"
      docker_repo = "gcr.io/{project_id}"
      default_dockerfile = "Dockerfile"
      permanent_bucket = "axlearn-bucket-multipod"
      private_bucket = "axlearn-bucket-multipod"
      ttl_bucket = "axlearn-bucket-multipod"
      """
  ).strip()

  escaped_config = config_content.replace('"', r"\"").replace("\n", r"\n")

  cmds += f'&& printf "{escaped_config}" > ~/.axlearn/axlearn.default.config'

  return cmds


def setup_cmds(
    cluster_name: str,
    project_id: str,
    zone: str,
) -> str:
  return [
      "export PYTHONPATH=$PYTHONPATH:/root",
      "axlearn gcp config activate",
      "apt-get install -y kubectl google-cloud-sdk-gke-gcloud-auth-plugin",
      f"gcloud container clusters get-credentials {cluster_name} \
            --region {gke.zone_to_region(zone)} --project {project_id}",
  ]


def generate_workload_id() -> str:
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
    A string formatted as
      '{short_id}-mtc-{slice_number}x-{accelerator}-{timestamp}'.
  """

  run_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  env = "prod" if composer_env.is_prod_env() else "dev"
  return f"automation-{env}-{run_time}"
