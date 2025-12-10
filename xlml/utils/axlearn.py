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


@task
def generate_workload_id(run_name_workload: str) -> str:
  """Generate a valid workload ID."""

  real_run_name__running = run_name_workload.split("-")[0]
  logging.info(f"Run_name used: {real_run_name__running}")
  return f"{real_run_name__running}"


def build_axlearn_cmd(
    task_id: str,
    gcs_path: str,
    cluster_project: str,
    cluster_name: str,
    zone: str,
    docker_image: str,
    benchmark_id: str,
    workload_id: str,
    run_name: str,
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

  # Extract the full path (e.g., "gcr.io/cienet-cmcs/axlearn-custom")
  # and the tag (e.g., "latest")
  regex = r"^(?P<full_path>.*)/(?P<run_image>[^/:]+):(?P<tag>.+)$"
  match = re.search(regex, docker_image)
  if not match:
    raise AirflowFailException(f"Invalid docker image format: {docker_image}")

  # These two values will determine the name of the pod run in AXLearn.
  image_run_name = match.group("run_image")
  tag = match.group("tag")

  # The output directory will be construct from the run_name
  # run_name: latest-axlearn-reg-rest-2x-tpu-v5p-64-2025-11-19-05-15
  # outpu_dir_name: latest-axlearn-reg-rest
  regex = r"^(?P<output_dir_name>(?:[^-]+-){3}[^-]+)"
  match = re.search(regex, run_name)
  if not match:
    raise AirflowFailException(f"Invalid run name format: {run_name}")
  outpu_dir_name = match.group("output_dir_name")

  export_var = [
      "export BASTION_TIER=disabled",
      f"export PROJECT_ID={cluster_project}",
  ]
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
      f"--name={tag} "
      f"--instance_type={accelerator_type} "
      f"--max_tries=10 "
      f"--num_replicas={num_slices} "
      f"--bundler_spec=allow_dirty=True "
      f"--bundler_type=artifactregistry "
      f"--bundler_spec=image={image_run_name} "
      f'-- "'
      f"ulimit -n 1048576; ulimit -c 0; "
      f"python3 -c 'import jax; jax.devices()'; "
      f"python3 -m axlearn.common.launch_trainer_main"
      f'" '
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
  return cmds


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


def generate_run_name(
    short_id: str,
    slice_number: int,
    accelerator: str,
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
    A string formatted as
      '{short_id}-mtc-{slice_number}x-{accelerator}-{timestamp}'.
  """

  run_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  run_name_id = f"latest-{short_id}-{slice_number}x-{accelerator}-{run_time}"
  return run_name_id


@task
def get_image_name(
    project_id: str,
    path_repository: str,
) -> str | None:
  """
  Retrieves repository details by calling the gcloud CLI command
  and parsing the output as JSON.

  Args:
    project_id: Your Google Cloud Project ID.
    repository_id: The ID of the repository.
      e.g gcr.io/cienet-cmcs/axlearn-custom
  Returns:
    A string with the name of the latest daily image.
  """
  list_tags_cmds = (
      f"gcloud container images list-tags {path_repository} "
      f"--project={project_id} "
      f"--format=json | tr -d '\\n\\r'"
  )
  cmds = [
      "set -ue",
      list_tags_cmds,
  ]

  hook = SubprocessHook()
  result = hook.run_command(["bash", "-c", ";".join(cmds)])
  assert (
      result.exit_code == 0
  ), f"XPK command failed with code {result.exit_code}"

  try:
    repo_details: list[dict] = json.loads(result.output)
  except json.JSONDecodeError as e:
    raise ValueError("Failed to parse JSON output from gcloud command.") from e

  logging.info(f"{f'First 5 Images: {repo_details[:5]}':=^80}")
  image_name = ""
  for image_info in repo_details:
    tags_list = image_info.get("tags", [])

    # We are expected to get the tags of all our images.
    # If tag list contain more than two we know is the latest.
    # (Since only the latest contains 2 tags.)
    if len(tags_list) >= 2:
      tags_list.remove("latest")
      image_name = f"{path_repository}:{tags_list[0]}"
      logging.info(f"{f'Running wirh Image: {image_name}':=^80}")
      return f"{path_repository}:latest"
  raise AirflowFailException("Image not found or is not latest image")
