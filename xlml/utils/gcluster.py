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
import uuid

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.hooks.subprocess import SubprocessHook
from dags.common.vm_resource import GpuVersion
from kubernetes import client as k8s_client
from xlml.utils import composer, gke

DEFAULT_GCLUSTER_VERSION = "v1.101.0"

LOGGING_URL_FORMAT = (
    "https://console.cloud.google.com/logs/viewer"
    "?project={project}&resource=k8s_container"
    "&minLogLevel=0&expandAll=false"
    "&customFacets=&limitCustomFacetWidth=true"
    "&filters=text:job-name%3D{workload_id}"
)


def get_gcluster_setup_cmd(
    tmpdir: str, version: str = DEFAULT_GCLUSTER_VERSION
) -> list[str]:
  """Construct shell commands to download and unpack gcluster binary."""
  bin_dir = f"{tmpdir}/bin"
  bundle_url = (
      "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/"
      f"download/{version}/gcluster_bundle_linux_amd64.tgz"
  )

  return [
      f"set -xueo pipefail; export PATH={bin_dir}:$PATH",
      (
          f"mkdir -p {bin_dir} && "
          f"curl -fsSL {bundle_url} | tar -xz -C {bin_dir} && "
          f"chmod +x {bin_dir}/gcluster"
      ),
  ]


def is_valid_gpu_version(accelerator_type: str) -> bool:
  """Check whether the given accelerator type corresponds to a valid GPU."""
  return accelerator_type in [member.value for member in GpuVersion]


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate unique workload ID conforming to RFC 1123 for Cluster Toolkit."""
  short_id = uuid.uuid4().hex[:5]
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
    pathways_gcs_location: str = "",
    ramdisk_directory: str = "",
    mtc_enabled: bool = False,
    gcluster_version: str = DEFAULT_GCLUSTER_VERSION,
    max_restart: int = 0,
    priority: str = "high",
    namespace: str = "default",
    mounts: str | Iterable[str] | None = None,
    queue: str = "default",
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
    slice_keyword = (
        "--num-nodes"
        if is_valid_gpu_version(accelerator_type)
        else "--num-slices"
    )

    args = [
        f"{gcluster_bin} job submit",
        f"--cluster={cluster_name}",
        f"--location={zone}",
        f"--project={cluster_project}",
        f"--name={workload_id}",
        f"--command={shlex.quote(run_cmds)}",
        f"--compute-type={accelerator_type}",
        f"--image={docker_image}",
        f"{slice_keyword}={num_slices}",
        f"--priority={priority}",
        f"--env GCS_OUTPUT={gcs_path}",
        "--download-dependencies",
    ]

    if queue:
      args.append(f"--queue={queue}")
    if use_pathways:
      args.append("--pathways")
      location = pathways_gcs_location or f"{gcs_path}/pathways"
      args.append(f"--pathways-gcs-location={location}")
    if use_vertex_tensorboard:
      args.append("--use-vertex-tensorboard")
    if ramdisk_directory:
      args.append(f"--gke-mtc-ramdisk-dir={ramdisk_directory}")
    if mtc_enabled:
      args.append("--gke-mtc-enabled")
    if max_restart > 0:
      args.append(f"--restarts={max_restart}")
    if is_valid_gpu_version(accelerator_type):
      args.append("--gke-scheduler=gke.io/topology-aware-auto")

    if mounts:
      mount_list = [mounts] if isinstance(mounts, str) else list(mounts)
      for m in mount_list:
        args.append(f"--mount={shlex.quote(m)}")

    get_credentials_cmd = (
        f"gcloud container clusters get-credentials {cluster_name}"
        f" --location={zone} --project={cluster_project} && "
        f"kubectl config set-context --current --namespace={namespace}"
    )

    cmds = [
        *get_gcluster_setup_cmd(tmpdir, gcluster_version),
        get_credentials_cmd,
        " ".join(args),
    ]

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


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_workload_start(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    namespace: str = "default",
) -> bool:
  """Wait for workload to start running."""
  core_api = gke.get_core_api_client(project_id, region, cluster_name)
  pods = gke.list_workload_pods(core_api, workload_id, namespace=namespace)
  gke.log_workload_pod_statuses(workload_id, pods)

  if not pods.items:
    logging.info(f"Waiting for pods of workload: {workload_id} to be created.")
    return False

  for pod in pods.items:
    if pod.status.phase in ["Pending", "Unknown"]:
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
          f"Workload {workload_id} failed during startup with pod phase:"
          f" {pod.status.phase}. Link to logs: {url}"
      )

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
  core_api = gke.get_core_api_client(project_id, region, cluster_name)
  pods = gke.list_workload_pods(core_api, workload_id, namespace=namespace)
  gke.log_workload_pod_statuses(workload_id, pods)

  if not pods.items:
    batch_api = gke.get_batch_api_client(project_id, region, cluster_name)
    job = gke.get_workload_job(batch_api, workload_id, namespace=namespace)
    if job and job.status and job.status.conditions:
      conditions = job.status.conditions
      if any(
          c.type == "Failed" and getattr(c, "status", "True") != "False"
          for c in conditions
      ):
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
      if any(
          c.type == "Complete" and getattr(c, "status", "True") != "False"
          for c in conditions
      ):
        logging.info(f"Workload {workload_id} Job completed successfully.")
        return True

    custom_api = gke.get_custom_objects_api_client(
        project_id, region, cluster_name
    )
    jobset = gke.get_workload_jobset(
        custom_api, workload_id, namespace=namespace
    )
    if jobset and "status" in jobset and "conditions" in jobset["status"]:
      conditions = jobset["status"]["conditions"]
      if any(
          c.get("type") == "Failed" and c.get("status", "True") != "False"
          for c in conditions
      ):
        url = LOGGING_URL_FORMAT.format(
            project=project_id,
            region=region,
            cluster=cluster_name,
            namespace=namespace,
            workload_id=workload_id,
        )
        raise AirflowFailException(
            f"Workload {workload_id} JobSet failed. Logs: {url}"
        )
      if any(
          c.get("type") == "Completed" and c.get("status", "True") != "False"
          for c in conditions
      ):
        logging.info(f"Workload {workload_id} JobSet completed successfully.")
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

  for pod in pods.items:
    if pod.status.container_statuses:
      for container_status in pod.status.container_statuses:
        if (
            container_status.state
            and container_status.state.terminated
            and container_status.state.terminated.exit_code != 0
        ):
          try:
            container_name = (
                container_status.name or pod.spec.containers[0].name
            )
            logs = core_api.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
                container=container_name,
            )
            for line in logs.split("\n"):
              logging.info(line)
          except k8s_client.exceptions.ApiException as e:
            logging.warning(
                f"Could not retrieve pod logs for {pod.metadata.name}: {e}"
            )
          url = LOGGING_URL_FORMAT.format(
              project=project_id,
              region=region,
              cluster=cluster_name,
              namespace=namespace,
              workload_id=workload_id,
          )
          raise AirflowFailException(
              f"Workload {workload_id} failed with container exit code "
              f"{container_status.state.terminated.exit_code}. Logs: {url}"
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
