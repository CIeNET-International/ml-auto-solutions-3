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

import airflow
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.exceptions import AirflowFailException
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.utils.pod_manager import OnFinishAction

from dags import composer_env
from xlml.utils import gke
from xlml.utils import composer
from dags.common.vm_resource import DockerImage


MAIN_BRANCH = "main"

KPO_LABEL_KEY = "axlearn-kpo-label"
KPO_LABEL_VAL = "axlearn_cli_kpo_worker"
# pylint: disable=line-too-long
# MUST use this fixed namespace for Cloud Composer 2.
# See: https://cloud.google.com/composer/docs/composer-2/use-kubernetes-pod-operator#composer-2-kpo-access-project-resources
KPO_NAMESPACE = "composer-user-workloads"
# pylint: enable=line-too-long

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
def reset_kube_config() -> None:
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
def update_image_tag_cmd(image_name: str, workload_id: str):
  # TODO: (dev only) (just for backup)
  # gcloud container images add-tag gcr.io/cloud-tpu-multipod-dev/axlearn-custom gcr.io/cloud-tpu-multipod-dev/axlearn-custom:automation-dev-2025-12-12-11-43-2 --quiet
  #
  # AXLearn pulls this particular image {docker_image}:{workload_id} when
  # creating the Pod.
  # Tag the image with {workload_id} before submitting the workload via the
  # AXLearn CLI.

  hook = SubprocessHook()
  result = hook.run_command([
      "bash",
      "-c",
      (
          "gcloud container images add-tag "
          f"{image_name} "
          f"{image_name}:{workload_id} "
          "--quiet"
      ),
  ])
  assert (
      result.exit_code == 0
  ), f"Failed to update image tag; exit code {result.exit_code}"


@task.kubernetes(
    name="cli-axlearn-pod",
    namespace=KPO_NAMESPACE,
    config_file="/home/airflow/composer_kube_config",
    on_finish_action=OnFinishAction.KEEP_POD,
    labels={KPO_LABEL_KEY: KPO_LABEL_VAL},
)
def run_in_kpo(command_to_execute: str, **kwargs):
  return KubernetesPodOperator(
      task_id="run_axlearn-cli",
      name="cli-axlearn-pod",
      namespace=KPO_NAMESPACE,
      config_file="/home/airflow/composer_kube_config",
      image=kwargs["image"],
      cmds=["bash", "-cx", f"{command_to_execute} & exit 0"],
      on_finish_action=OnFinishAction.KEEP_POD,
      labels={KPO_LABEL_KEY: KPO_LABEL_VAL},
  )


@task
def generate_workload_id() -> str:
  """Generates a unique run name for a AXLearn run."""

  run_time = datetime.now().strftime("%Y%m%d%H%M")
  env = "prod" if composer_env.is_prod_env() else "dev"
  return f"automation-{env}-{run_time}"


@task
def generate_axlearn_cli_command(
    task_id: str,
    gcs_path: str,
    project_id: str,
    cluster_name: str,
    zone: str,
    docker_image_name: str,
    docker_image_repo: str,
    docker_image_full_url: str,
    benchmark_id: str,
    workload_id: str,
    accelerator_type: str = "",
    module: str = "",
    model_config: str = "",
    trainer_dir: str = "",
    num_slices: int = 1,
    trace_steps: list[int] = None,
    label: str = "tpu-v5p",
) -> str:
  # Log required info for XLML PLX Dashboard
  composer.log_metadata_for_xlml_dashboard({
      "cluster_project": project_id,
      "zone": zone,
      "cluster_name": cluster_name,
      "task_id": task_id,
      "workload_id": workload_id,
      "gcs_path": gcs_path,
      "benchmark_id": benchmark_id,
      "docker_image": docker_image_full_url,
      "accelerator_type": accelerator_type,
      "num_slices": num_slices,
  })

  cfg_content = textwrap.dedent(
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
      docker_repo = "{docker_image_repo}"
      default_dockerfile = "Dockerfile"
      permanent_bucket = "axlearn-bucket-multipod"
      private_bucket = "axlearn-bucket-multipod"
      ttl_bucket = "axlearn-bucket-multipod"
      """
  ).strip()

  cfg_file = "~/.axlearn/axlearn.default.config"

  gen_config_cmds = [
      "mkdir -p ~/.axlearn",
      f"cat > {cfg_file} <<'EOF'\n{cfg_content}\nEOF\necho 'file created'",
  ]

  setup_cmds = [
      "export PYTHONPATH=$PYTHONPATH:/root",
      "axlearn gcp config activate",
      "apt-get install -y kubectl google-cloud-sdk-gke-gcloud-auth-plugin",
      f"gcloud container clusters get-credentials {cluster_name} \
            --region {gke.zone_to_region(zone)} --project {project_id}",
  ]

  axlearn_cli_cmd = (
      f"axlearn gcp launch run --cluster={cluster_name} "  # TODO
      f"--runner_name gke_tpu_single "
      f"--name={workload_id} "
      f"--instance_type={accelerator_type} "
      f"--max_tries=10 "
      f"--num_replicas={num_slices} "
      f"--bundler_spec=allow_dirty=True "
      f"--bundler_type=artifactregistry "
      f"--bundler_spec=image={docker_image_name} "
      f"--bundler_spec=skip_bundle=True "  # TODO (only for test)
      f'-- "'
      f"ulimit -n 1048576; ulimit -c 0; "
      f"python3 -c 'import jax; jax.devices()'; "
      f"python3 -m axlearn.common.launch_trainer_main"
      f'" '
      f"--module={module} "
      f"--config={model_config} "
      f"--trainer_dir={trainer_dir}/{workload_id} "
      f"--data_dir=gs://axlearn-public/tensorflow_datasets "
      f"--mesh_selector={accelerator_type} "
      f"--jax_backend=tpu "
      f"--initialization_timeout=1200 "
  )
  if trace_steps:
    axlearn_start_cmd += f"--trace_at_steps={','.join(map(str, trace_steps))}"

  # 300 seconds should be sufficient for the CLI to deploy the workload,
  # the CLI is useless once the workload is deployed.
  exit_with_delay = " & sleep 300 && exit 0"

  run_workload_cmds = [
      "export BASTION_TIER=disabled",
      f"export PROJECT_ID={project_id}",
      axlearn_cli_cmd + exit_with_delay,
  ]

  return " && ".join([
      *gen_config_cmds,
      *setup_cmds,
      *run_workload_cmds,
  ])


class CommandBuilder:

  @staticmethod
  @task
  def start_workload(
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
      trace_steps: list[int] = None,
  ) -> str:
    """Generates the command for AXLearn CLI to start the workload"""

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

    # TODO: do we even need this?
    # # TODO: can we name the folder? a fixed one
    # # The output directory will be construct from the workload_id
    # # workload_id: "automation-prod-202511190515"
    # # output_dir_name: automation-prod
    # regex = r"^(?P<output_dir_name>.+)-\d{12}$"
    # match = re.search(regex, workload_id)
    # if not match:
    #   raise AirflowFailException(f"Invalid run name format: {workload_id}")
    # output_dir_name = match.group("output_dir_name")

    # TODO: move to `AXLEARN_CUSTOM`?
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
    axlearn_start_cmd = (
        f"axlearn gcp launch run --cluster={cluster_name} "  # TODO
        f"--runner_name gke_tpu_single "
        f"--name={workload_id} "
        f"--instance_type={accelerator_type} "
        f"--max_tries=10 "
        f"--num_replicas={num_slices} "
        f"--bundler_spec=allow_dirty=True "
        f"--bundler_type=artifactregistry "
        f"--bundler_spec=image=axlearn-custom "  # TODO
        f'-- "'
        f"ulimit -n 1048576; ulimit -c 0; "
        f"python3 -c 'import jax; jax.devices()'; "
        f"python3 -m axlearn.common.launch_trainer_main"
        f'" '
        f"--module={module} --config={model_config} "
        f"--trainer_dir={trainer_dir}/{workload_id} "
        f"--data_dir=gs://axlearn-public/tensorflow_datasets "
        f"--mesh_selector={accelerator_type} "
        f"--jax_backend=tpu "
        f"--initialization_timeout=1200 "
    )
    if trace_steps:
      axlearn_start_cmd += f"--trace_at_steps={','.join(map(str, trace_steps))}"

    # 300 seconds should be sufficient for the CLI to deploy the workload,
    # the CLI is useless once the workload is deployed.
    exit_with_delay = " & sleep 300 && exit 0"

    return " && ".join([
        "export BASTION_TIER=disabled",
        f"export PROJECT_ID={cluster_project}",
        axlearn_start_cmd + exit_with_delay,
    ])

  @staticmethod
  @task
  def generate_config_file(
      cluster_name: str, project_id: str, zone: str, label: str = "tpu-v5p"
  ) -> str:
    """
    Returns a shell command which generates the AXLearn configs file.
    """
    cfg_content = textwrap.dedent(
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
    ).strip()  # TODO: docker_repo

    cfg_file = "~/.axlearn/axlearn.default.config"

    return " && ".join([
        "mkdir -p ~/.axlearn",
        f"cat > {cfg_file} <<'EOF'\n{cfg_content}\nEOF\necho 'file created'",
    ])

  @staticmethod
  @task
  def setup_axlearn(
      cluster_name: str,
      project_id: str,
      zone: str,
  ) -> str:
    return " && ".join([
        "export PYTHONPATH=$PYTHONPATH:/root",
        "axlearn gcp config activate",
        "apt-get install -y kubectl google-cloud-sdk-gke-gcloud-auth-plugin",
        f"gcloud container clusters get-credentials {cluster_name} \
              --region {gke.zone_to_region(zone)} --project {project_id}",
    ])

  @staticmethod
  @task
  def join_cmds(cmds: list[str]) -> str:
    return " && ".join(cmds)
