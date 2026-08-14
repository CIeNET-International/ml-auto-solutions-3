# Copyright 2026 Google LLC
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

"""Utility functions for Cluster Toolkit (gcluster) orchestration."""

from collections.abc import Iterable
import logging
import os
import re
import shlex
import tempfile
from typing import Any, Union
import uuid

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.hooks.subprocess import SubprocessHook
from dags.common.vm_resource import GpuVersion
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from xlml.utils import composer

DEFAULT_GCLUSTER_VERSION = "v1.99.0"

LOGGING_URL_FORMAT = (
    "https://console.cloud.google.com/logs/viewer"
    "?project={project}&resource=k8s_container"
    "&minLogLevel=0&expandAll=false"
    "&customFacets=&limitCustomFacetWidth=true"
    "&filters=text:job-name%3D{workload_id}"
)


def get_gcluster_setup_cmd(tmpdir: str, version: str) -> list[str]:
  """Construct shell commands to download and unpack the pinned gcluster binary."""
  bin_dir = f"{tmpdir}/bin"
  bundle_url = (
      "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/"
      f"download/{version}/gcluster_bundle_linux_amd64.tgz"
  )

  bash_setup = f"set -xueo pipefail; export PATH={bin_dir}:$PATH"
  download_and_extract = (
      f"mkdir -p {bin_dir} && "
      f"curl -fsSL {bundle_url} | tar -xz -C {bin_dir} && "
      f"chmod +x {bin_dir}/gcluster"
  )
  docker_auth = (
      "gcloud auth configure-docker"
      " gcr.io,us-docker.pkg.dev,europe-docker.pkg.dev,asia-docker.pkg.dev,"
      "us-central1-docker.pkg.dev,us-central2-docker.pkg.dev,"
      "europe-west4-docker.pkg.dev,us-east5-docker.pkg.dev"
      " --quiet || true"
  )

  return [bash_setup, download_and_extract, docker_auth]


def is_valid_gpu_version(accelerator_type: str) -> bool:
  """Check whether the given accelerator type corresponds to a valid GPU."""
  return accelerator_type in [member.value for member in GpuVersion]


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate a valid workload ID for gcluster (<= 28 characters)."""
  short_id = str(uuid.uuid4())[:8]
  clean_benchmark = (
      re.sub(r"[^a-z0-9]+", "-", benchmark_id.lower())
      .strip("-")[:19]
      .rstrip("-")
  )
  if not clean_benchmark:
    clean_benchmark = "job"
  return f"{clean_benchmark}-{short_id}"


@task
def run_workload(
    task_id: str,
    cluster_project: str,
    zone: str,
    cluster_name: str,
    benchmark_id: str,
    workload_id: str,
    gcs_path: str,
    docker_image: str,
    accelerator_type: str,
    run_cmds: str,
    num_slices: int = 1,
    use_vertex_tensorboard: bool = False,
    use_pathways: bool = False,
    ramdisk_directory: str = "",
    mtc_enabled: bool = False,
    gcluster_version: str = DEFAULT_GCLUSTER_VERSION,
    max_restart: int = 0,
    priority: str = "high",
    namespace: str = "default",
    mounts: str | Iterable[str] | None = None,
):
  """Run workload through Cluster Toolkit (gcluster) tool."""
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

  with tempfile.TemporaryDirectory() as tmpdir:
    gcluster_bin = f"{tmpdir}/bin/gcluster"
    base_args = [
        f"{gcluster_bin} job submit",
        f"--cluster={cluster_name}",
        f"--location={zone}",
        f"--project={cluster_project}",
        f"--name={workload_id}",
        f"--command={shlex.quote(run_cmds)}",
        f"--compute-type={accelerator_type}",
        f"--image={docker_image}",
    ]

    if is_valid_gpu_version(accelerator_type):
      base_args.append(f"--num-nodes={num_slices}")
    else:
      base_args.append(f"--num-slices={num_slices}")

    base_args.append(f"--priority={priority}")
    base_args.append(f"--env GCS_OUTPUT={gcs_path}")
    base_args.append("--download-dependencies")

    if mounts is not None:
      if isinstance(mounts, str):
        mount_list = [mounts]
      else:
        mount_list = list(mounts)
      for m in mount_list:
        base_args.append(f"--mount={shlex.quote(m)}")

    if ramdisk_directory:
      base_args.append(f"--gke-mtc-ramdisk-dir={ramdisk_directory}")
    if mtc_enabled:
      base_args.append("--gke-mtc-enabled")
    if max_restart > 0:
      base_args.append(f"--restarts={max_restart}")
    if is_valid_gpu_version(accelerator_type):
      base_args.append("--gke-scheduler=gke.io/topology-aware-auto")

    workload_submit_cmd = " ".join(base_args)

    get_credentials_cmd = (
        f"gcloud container clusters get-credentials {cluster_name}"
        f" --location={zone} --project={cluster_project} && "
        f"kubectl config set-context --current --namespace={namespace} || true"
    )

    cmds = get_gcluster_setup_cmd(tmpdir, gcluster_version)
    cmds.append(get_credentials_cmd)
    cmds.append(workload_submit_cmd)

    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={
            **os.environ,
            "KUBECONFIG": os.path.join(tmpdir, "gcluster_kube.conf"),
        },
    )
    assert (
        result.exit_code == 0
    ), f"Cluster Toolkit submit failed with code {result.exit_code}"


def _get_core_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.CoreV1Api:
  k8s_config.load_kube_config()
  return k8s_client.CoreV1Api()


def _get_batch_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.BatchV1Api:
  k8s_config.load_kube_config()
  return k8s_client.BatchV1Api()


def _get_custom_objects_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.CustomObjectsApi:
  k8s_config.load_kube_config()
  return k8s_client.CustomObjectsApi()


def _list_workload_pods(
    core_api: k8s_client.CoreV1Api, workload_id: str, namespace: str = "default"
) -> Any:
  label_selector = (
      f"jobset.sigs.k8s.io/jobset-name={workload_id}," f"job-name={workload_id}"
  )
  try:
    pods = core_api.list_namespaced_pod(
        namespace=namespace, label_selector=f"job-name={workload_id}"
    )
    if len(pods.items) == 0:
      pods = core_api.list_namespaced_pod(
          namespace=namespace,
          label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      )
    return pods
  except Exception as e:
    logging.warning(f"Error listing pods for {workload_id}: {e}")
    mock_list = type("PodList", (), {"items": []})()
    return mock_list


def _get_workload_job(
    batch_api: k8s_client.BatchV1Api,
    workload_id: str,
    namespace: str = "default",
) -> Any:
  try:
    return batch_api.read_namespaced_job(name=workload_id, namespace=namespace)
  except Exception as e:
    logging.info(f"Could not read Kubernetes Job {workload_id}: {e}")
    return None


def _get_workload_jobset(
    custom_api: k8s_client.CustomObjectsApi,
    workload_id: str,
    namespace: str = "default",
) -> Any:
  try:
    return custom_api.get_namespaced_custom_object(
        group="jobset.sigs.k8s.io",
        version="v1alpha2",
        namespace=namespace,
        plural="jobsets",
        name=workload_id,
    )
  except Exception as e:
    logging.info(f"Could not read JobSet {workload_id}: {e}")
    return None


def _log_workload_pod_statuses(workload_id: str, pods: Any) -> None:
  for pod in pods.items:
    logging.info(
        f"Workload {workload_id} pod {pod.metadata.name} phase:"
        f" {pod.status.phase}"
    )


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_workload_start(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    namespace: str = "default",
) -> bool:
  """Wait for workload to start running."""
  core_api = _get_core_api_client(project_id, region, cluster_name)
  pods = _list_workload_pods(core_api, workload_id, namespace=namespace)
  _log_workload_pod_statuses(workload_id, pods)

  if len(pods.items) == 0:
    logging.info(f"Waiting for pods of workload: {workload_id} to be created.")
    return False

  for pod in pods.items:
    if pod.status.phase in ["Pending", "Unknown"]:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
      return False

  logging.info("All pod(s) phase are ready to run.")
  return True


@task.sensor(poke_interval=60, timeout=18000, mode="reschedule")
def wait_for_workload_completion(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    namespace: str = "default",
) -> bool:
  """Wait for workload to finish successfully."""
  core_api = _get_core_api_client(project_id, region, cluster_name)
  pods = _list_workload_pods(core_api, workload_id, namespace=namespace)
  _log_workload_pod_statuses(workload_id, pods)

  if len(pods.items) == 0:
    batch_api = _get_batch_api_client(project_id, region, cluster_name)
    job = _get_workload_job(batch_api, workload_id, namespace=namespace)
    if job and job.status.conditions:
      conditions = job.status.conditions
      if any(condition.type == "Failed" for condition in conditions):
        url = LOGGING_URL_FORMAT.format(
            project=project_id,
            region=region,
            cluster=cluster_name,
            namespace=namespace,
            workload_id=workload_id,
        )
        raise AirflowFailException(
            f"Workload {workload_id} failed. Logs: {url}"
        )
      if any(condition.type == "Complete" for condition in conditions):
        logging.info(f"Workload {workload_id} Job completed successfully.")
        return True
    logging.info(f"No pods found for workload: {workload_id}")
    return False

  for pod in pods.items:
    if pod.status.phase in ["Pending", "Running", "Unknown"]:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
      return False
    if pod.status.phase == "Failed":
      url = LOGGING_URL_FORMAT.format(
          project=project_id,
          region=region,
          cluster=cluster_name,
          namespace=namespace,
          workload_id=workload_id,
      )
      raise AirflowFailException(
          f"Workload {workload_id} failed with pod phase: {pod.status.phase}."
          f" Link to logs: {url}"
      )

  if len(pods.items) > 0:
    last_pod = pods.items[-1]
    if (
        last_pod.status.container_statuses
        and last_pod.status.container_statuses[0].state.terminated
        and last_pod.status.container_statuses[0].state.terminated.exit_code
        != 0
    ):
      try:
        logs = core_api.read_namespaced_pod_log(
            name=last_pod.metadata.name,
            namespace=namespace,
            container=last_pod.spec.containers[0].name,
        )
        for line in logs.split("\n"):
          logging.info(line)
      except Exception as e:
        logging.warning(
            f"Could not retrieve pod logs for {last_pod.metadata.name}: {e}"
        )
  logging.info("All pod(s) phase are succeeded.")
  return True


@task(trigger_rule="all_done")
def clean_up_workload(
    workload_id: str,
    project_id: str,
    zone: str,
    cluster_name: str,
    gcluster_version: str = DEFAULT_GCLUSTER_VERSION,
    namespace: str = "default",
) -> None:
  """Delete/cancel workload using Cluster Toolkit."""
  with tempfile.TemporaryDirectory() as tmpdir:
    gcluster_bin = f"{tmpdir}/bin/gcluster"
    workload_delete_cmd = (
        f"{gcluster_bin} job cancel {workload_id}"
        f" --cluster={cluster_name}"
        f" --location={zone}"
        f" --project={project_id}"
        " --download-dependencies"
    )

    get_credentials_cmd = (
        f"gcloud container clusters get-credentials {cluster_name}"
        f" --location={zone} --project={project_id} && "
        f"kubectl config set-context --current --namespace={namespace} || true"
    )
    cmds = get_gcluster_setup_cmd(tmpdir, gcluster_version)
    cmds.append(get_credentials_cmd)
    cmds.append(workload_delete_cmd)
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={
            **os.environ,
            "KUBECONFIG": os.path.join(tmpdir, "gcluster_kube.conf"),
        },
    )
    assert (
        result.exit_code == 0
    ), f"Cluster Toolkit clean-up failed with code {result.exit_code}"
