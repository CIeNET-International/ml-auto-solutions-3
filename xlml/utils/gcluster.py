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

"""Utilities to run workloads with Cluster Toolkit (gcluster)
(https://github.com/GoogleCloudPlatform/cluster-toolkit)."""

import os
import re
import shlex
import tempfile
from typing import Iterable, List, Optional, Union
import uuid
from absl import logging

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.hooks.subprocess import SubprocessHook
from kubernetes import client as k8s_client

from dags.common.vm_resource import GpuVersion
from xlml.apis import metric_config
from xlml.utils import composer, gke

# Pinned release tag for Cluster Toolkit binary distribution
DEFAULT_GCLUSTER_VERSION = "v1.99.0"

# Duration = past 7 days
LOGGING_URL_FORMAT = (
    "https://pantheon.corp.google.com/logs/query;"
    + "query=resource.type%3D%22k8s_container%22%0A"
    + "resource.labels.project_id%3D%22{project}%22%0A"
    + "resource.labels.location%3D%22{region}%22%0A"
    + "resource.labels.cluster_name%3D%22{cluster}%22%0A"
    + "resource.labels.namespace_name%3D%22{namespace}%22%0A"
    + "labels.k8s-pod%2Fjobset_sigs_k8s_io%2F"
    + "jobset-name%3D%22{workload_id}%22%20severity%3E%3DDEFAULT;"
    + "storageScope=project;duration=P7D?e=13803378&"
    + "mods=allow_workbench_image_override&project={project}"
)


def get_gcluster_setup_cmd(
    tmpdir: str, version: str = DEFAULT_GCLUSTER_VERSION
) -> List[str]:
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
  # gcluster enforces a 28-character limit on workload names.
  # Truncate sanitized benchmark prefix to at most 19 chars:
  # 19 + 1 ('-') + 8 = 28 max.
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
    mounts: Optional[Union[str, Iterable[str]]] = None,
):
  """Run workload through Cluster Toolkit (gcluster) tool."""
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

  with tempfile.TemporaryDirectory() as tmpdir:
    gcluster_bin = f"{tmpdir}/bin/gcluster"

    if accelerator_type in [
        GpuVersion.XPK_H100.value,
        GpuVersion.XPK_H100_MEGA.value,
    ]:
      slicing_arg = f" --num-nodes={num_slices}"
    else:
      slicing_arg = f" --num-slices={num_slices}"

    workload_create_cmd = (
        f"{gcluster_bin} job submit"
        f" --cluster={cluster_name}"
        f" --location={zone}"
        f" --project={cluster_project}"
        f" --name={workload_id}"
        f" --command={shlex.quote(run_cmds)}"
        f" --compute-type={accelerator_type}"
        f" --image={docker_image}"
        f"{slicing_arg}"
        f" --priority={priority}"
        f" --env {metric_config.SshEnvVars.GCS_OUTPUT.name}={gcs_path}"
        " --download-dependencies"
    )

    if mounts:
      if isinstance(mounts, str):
        workload_create_cmd += f" --mount={shlex.quote(mounts)}"
      elif isinstance(mounts, Iterable):
        for m in mounts:
          workload_create_cmd += f" --mount={shlex.quote(m)}"

    if ramdisk_directory and not use_pathways:
      workload_create_cmd += f" --gke-mtc-ramdisk-dir={ramdisk_directory}"

    if mtc_enabled:
      workload_create_cmd += " --gke-mtc-enabled"

    if max_restart > 0:
      workload_create_cmd += f" --restarts={max_restart}"

    if accelerator_type == GpuVersion.XPK_H100_MEGA.value:
      workload_create_cmd += " --gke-scheduler=gke.io/topology-aware-auto"

    if use_pathways:
      workload_create_cmd += f" --pathways --pathways-gcs-location={gcs_path}"

    get_credentials_cmd = (
        f"gcloud container clusters get-credentials {cluster_name}"
        f" --location={zone} --project={cluster_project} && "
        f"kubectl config set-context --current --namespace={namespace} || true"
    )
    cmds = get_gcluster_setup_cmd(tmpdir, gcluster_version)
    cmds.append(get_credentials_cmd)
    if use_vertex_tensorboard:
      vertex_ai_dependency = (
          "pip install -U google-cloud-aiplatform cloud-accelerator-diagnostics"
      )
      cmds.append(vertex_ai_dependency)
    cmds.append(workload_create_cmd)

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
    ), f"Cluster Toolkit command failed with code {result.exit_code}"


def _get_core_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.CoreV1Api:
  """Create a core API client for the given cluster."""
  client = gke.get_authenticated_client(project_id, region, cluster_name)
  core_api = k8s_client.CoreV1Api(client)
  logging.info("Successfully initialized k8s client from cluster response.")
  return core_api


def _list_workload_pods(
    core_api: k8s_client.CoreV1Api,
    workload_id: str,
    namespace: str = "default",
) -> k8s_client.V1PodList:
  """List all pods for the given JobSet workload."""
  logging.info(
      f"Getting pods for workload_id: {workload_id} in namespace: {namespace}"
  )
  pods = core_api.list_namespaced_pod(
      label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      namespace=namespace,
  )
  return pods


def _get_batch_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.BatchV1Api:
  """Create a batch API client for the given cluster."""
  client = gke.get_authenticated_client(project_id, region, cluster_name)
  batch_api = k8s_client.BatchV1Api(client)
  logging.info(
      "Successfully initialized k8s batch api client from cluster response."
  )
  return batch_api


def _get_workload_job(
    batch_api: k8s_client.BatchV1Api,
    workload_id: str,
    namespace: str = "default",
) -> Optional[k8s_client.V1Job]:
  """Get the job for a given JobSet workload."""
  logging.info(
      f"Getting job for workload_id: {workload_id} in namespace: {namespace}"
  )
  jobs = batch_api.list_namespaced_job(
      label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      namespace=namespace,
  )
  if len(jobs.items) == 0:
    logging.info(f"No job found for workload_id: {workload_id}")
    return None

  if len(jobs.items) > 1:
    logging.info(f"Got more than one job for workload_id: {workload_id}")
    for i, job in enumerate(jobs.items):
      logging.info(f"Job {i=}")
      logging.info(f"{job}")

  return jobs.items[0]


def _log_workload_pod_statuses(
    workload_id: str, pods: k8s_client.V1PodList
) -> None:
  """Logs the status of each retrieved pod and its containers for troubleshooting."""
  if not pods.items:
    return

  logging.info(f"{f' Pod Statuses for Workload {workload_id} ':-^80}")
  for pod in pods.items:
    pod_name = pod.metadata.name
    pod_phase = pod.status.phase
    logging.info(f"Pod: {pod_name} | Phase: {pod_phase}")
    if pod.status.container_statuses:
      for cs in pod.status.container_statuses:
        state_str = "Unknown"
        if cs.state.running:
          state_str = f"Running (started: {cs.state.running.started_at})"
        elif cs.state.waiting:
          state_str = (
              f"Waiting (reason: {cs.state.waiting.reason},"
              f" msg: {cs.state.waiting.message})"
          )
        elif cs.state.terminated:
          state_str = (
              f"Terminated (exit_code: {cs.state.terminated.exit_code},"
              f" reason: {cs.state.terminated.reason},"
              f" msg: {cs.state.terminated.message})"
          )
        logging.info(
            f"  Container: {cs.name} | Ready: {cs.ready} | State: {state_str}"
        )


@task
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
    logging.info(f"No pods found for workload: {workload_id}")
    return False

  for pod in pods.items:
    if pod.status.phase in ["Pending", "Unknown"]:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
      return False

  logging.info("All pod(s) phase are ready to run.")
  return True


@task
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
