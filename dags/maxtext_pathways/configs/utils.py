# Copyright 2023 Google LLC
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

"""Common funcitons and tasks for MaxText Pathways DAGs"""

from datetime import datetime, timezone
import json
import os
import re
import tempfile
import time

from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowException, AirflowFailException
from airflow.models.taskmixin import DAGNode
from airflow.models.baseoperator import chain
from airflow.operators.python import get_current_context
from airflow.utils.task_group import TaskGroup
from google.cloud import logging as gcp_logging

from xlml.utils import gcluster, gke, subprocess_utils

# TODO(cienet): Replace this with an official one.
COLOCATED_PYTHON_IMAGE = (
    "gcr.io/tpu-prod-env-multipod/lidanny_maxtext-colocated-python:latest"
)

_V5E_V6E_TOPOLOGIES = {
    1: "1x1",
    4: "2x2",
    8: "2x4",
    16: "4x4",
    32: "4x8",
    64: "8x8",
    128: "8x16",
    256: "16x16",
}

_V5P_TOPOLOGIES = {
    8: "2x2x1",
    16: "2x2x2",
    32: "2x2x4",
    64: "2x4x4",
    128: "4x4x4",
    256: "4x4x8",
    512: "4x8x8",
    1024: "8x8x8",
    2048: "8x8x16",
    4096: "8x16x16",
}


def resolve_pathways_slice_topology(
    device_version: str, core_count: int
) -> str:
  """Maps (device_version, core_count) to a Pathways virtual_slices token."""
  cores = int(core_count)
  if device_version == "v5p":
    if cores not in _V5P_TOPOLOGIES:
      raise ValueError(f"Unsupported v5p core_count for Pathways: {cores}")
    return f"tpuv5:{_V5P_TOPOLOGIES[cores]}"
  if device_version in ("v5litepod", "v5e"):
    shape = _V5E_V6E_TOPOLOGIES.get(cores, f"4x{cores // 4}")
    return f"tpuv5e:{shape}"
  shape = _V5E_V6E_TOPOLOGIES.get(cores, f"4x{cores // 4}")
  return f"tpuv6e:{shape}"


def _get_kubeconfig_env(
    project_id: str, region: str, cluster_name: str, kubeconfig_path: str
) -> dict[str, str]:
  """Authenticates kubectl against the specified GKE cluster."""
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig_path
  cmd = f"gcloud container clusters get-credentials {cluster_name} --region {region} --project {project_id}"
  logging.info(f"Authenticating kubectl for cluster: {cluster_name}")
  subprocess_utils.run_exec(cmd, env=env)
  return env


def _list_workload_pods_kubectl(
    workload_id: str, env: dict[str, str] | None = None
) -> list:
  """Lists pods matching the workload label using kubectl and returns them as a list of dicts."""
  logging.info(f"Getting pods for workload_id: {workload_id}")
  cmd = f"kubectl get pods -n default -l jobset.sigs.k8s.io/jobset-name={workload_id} -o json"
  result = subprocess_utils.run_exec(cmd, env=env, log_output=False)
  pod_list_json = json.loads(result)
  # Returns the 'items' list from the Kubernetes List object
  return pod_list_json.get("items", [])


def generate_recipe_workload_id(dag_id: str) -> tuple[str, str]:
  """Generate a workload_id following the standard naming convention."""
  time.localtime()
  timestamp = time.strftime("%m%d%H%M%S", time.localtime())
  name = f"{dag_id[:10]}-{timestamp[:10]}"
  name = name[:40].replace("_", "-")

  return name


def generate_install_dependencies_commands(
    cluster_name: str,
    location: str,
    project_id: str,
) -> str:
  """Generate shell commands to install necessary dependencies in the Pod."""
  # fmt: off
  return " && ".join([
      # Update apt package list
      "sudo apt-get update",

      # Install kubectl and GKE auth plugin
      "sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin",

      # Install gcluster and configure cluster credentials
      *gcluster.get_gcluster_setup_cmd(
          tmpdir="/root",
          cluster_name=cluster_name,
          location=location,
          project_id=project_id,
      ),

      # Install dependencies for maxtext
      "pip install omegaconf",

      # Prepare environment for further pip installs
      "cd /deps",
      "export USER=root",
  ])
  # fmt: on


@task.python(multiple_outputs=True)
def get_dag_parameters(**context) -> dict:
  """Fetches and returns the DAG run's configuration parameters."""
  dag_params = context.get("params", {})

  return dag_params


@task.python(multiple_outputs=True)
def generate_derived_parameters(dag_params: dict, dag_id: str) -> dict:
  """Generates new parameters based on the initial DAG parameters."""
  derived_params = {}

  # Generate recipe workload_id.
  name = generate_recipe_workload_id(dag_id)
  derived_params["workload_id"] = name

  # Generate region by zone
  derived_params["region"] = gke.zone_to_region(dag_params["zone"])

  # Generate device_type.
  device_type = (
      dag_params["device_version"] + "-" + str(dag_params["core_count"])
  )
  derived_params["device_type"] = device_type

  # Confirm whether to use customized_model_name.
  if dag_params["selected_model_names"] == "customized_model_name":
    derived_params["selected_model_names"] = dag_params["customized_model_name"]

  if dag_params["elastic_type"] in ["Pause-resume", "Replica-resize"]:
    slice_topology = resolve_pathways_slice_topology(
        dag_params["device_version"], dag_params["core_count"]
    )
    if dag_params["elastic_type"] == "Pause-resume":
      num_slices = max(1, int(dag_params.get("num_slices_list", 1)))
      derived_params["elastic_min_slice_count"] = -1
    else:
      num_slices = max(2, int(dag_params.get("num_slices_list", 2)))
      derived_params["elastic_min_slice_count"] = 1
    derived_params["topology"] = ",".join([slice_topology] * num_slices)
    derived_params["num_elastic_slices"] = num_slices

  return derived_params


@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def check_gcp_logs_exist(
    project_id: str,
    cluster_name: str,
    workload_id: str,
    expect_log_contains: str,
    location: str,  # e.g., 'us-central1' or zone 'us-central1-a'
    expected_count: int = 1,
) -> bool:
  """
  Counts occurrences of a string pattern in GCP
  Cloud Logging for a specific workload.
  """
  # Initialize the GCP Logging Client
  client = gcp_logging.Client(project=project_id)

  log_filter = (
      f'resource.type="k8s_container" '
      f'resource.labels.cluster_name="{cluster_name}" '
      f'resource.labels.location="{location}" '
      'resource.labels.namespace_name="default" '
      f'resource.labels.pod_name=~"{workload_id}.*"'
  )

  logging.info(f"Querying GCP Logging with filter: {log_filter}")

  # Fetch the entries. (Adjust page_size based on log volume to optimize speed)
  entries = client.list_entries(filter_=log_filter, page_size=500)

  # Consolidate all log payloads into a single text body
  log_lines = []
  for entry in entries:
    payload = entry.payload

    if payload:
      if isinstance(payload, str):
        log_lines.append(payload)
      elif isinstance(payload, dict):
        message = payload.get("message") or payload.get("textPayload")
        if message:
          log_lines.append(str(message))
        else:
          log_lines.append(str(payload))

  full_logs_text = "\n".join(log_lines)

  if not full_logs_text:
    logging.info("No logs found yet in Cloud Logging for filter.")
    return False

  # Normalize input to a list if a single string is passed
  if isinstance(expect_log_contains, str):
    patterns = [expect_log_contains]
  else:
    patterns = expect_log_contains

  all_patterns_found = True
  for pattern in patterns:
    # re.escape matches your original literal string search logic
    log_matches = re.findall(re.escape(pattern), full_logs_text)
    log_count = len(log_matches)
    logging.info(f"Logs: '{pattern}' found {log_count} times, ")

    if log_count < expected_count:
      logging.info(
          f"Pattern '{pattern}' found {log_count} times, "
          f"which is less than expected_count ({expected_count})."
      )
      all_patterns_found = False
      break

  if all_patterns_found:
    logging.info(
        "All expected log patterns found successfully in GCP Cloud Logging."
    )
    return True

  logging.info("Waiting for matching log pattern in GCP...")
  return False


def wait_for_running_head_pod(
    workload_id: str,
    env: dict[str, str] | None = None,
    startup_timeout: int = 600,
    poll_interval: int = 10,
) -> tuple[dict, str | None]:
  """Polls until the head pod exists and its workload container starts."""
  head_prefixes = (
      f"{workload_id}-pathways-head-0-0-",
      f"{workload_id}-head-0-0-",
  )
  deadline = time.time() + startup_timeout
  while True:
    pods = _list_workload_pods_kubectl(workload_id, env=env)
    if pods:
      matching_heads = [
          p
          for p in pods
          if p.get("metadata", {}).get("name", "").startswith(head_prefixes)
      ]
      active_heads = [
          p
          for p in matching_heads
          if p.get("status", {}).get("phase") in ("Running", "Succeeded")
      ]
      head_pod = (
          active_heads[0]
          if active_heads
          else (matching_heads[0] if matching_heads else pods[0])
      )
      phase = head_pod.get("status", {}).get("phase")
      head_name = head_pod.get("metadata", {}).get("name", "<unknown>")
      if phase in ("Failed", "Unknown"):
        raise AirflowFailException(
            f"Head pod {head_name} in bad phase: {phase}"
        )

      containers = [
          c.get("name")
          for c in head_pod.get("spec", {}).get("containers", [])
          if c.get("name")
      ]
      container_name = next(
          (c for c in ("workload-container", "jax-tpu") if c in containers),
          containers[0] if containers else None,
      )
      statuses = head_pod.get("status", {}).get("containerStatuses") or []
      is_waiting = (
          phase == "Pending"
          if not statuses
          else any(
              "waiting" in s.get("state", {})
              for s in statuses
              if not container_name or s.get("name") == container_name
          )
      )
      if phase in ("Running", "Succeeded") and not is_waiting:
        return head_pod, container_name
      logging.info(
          "Head pod %s in phase %s (container %s initializing); waiting...",
          head_name,
          phase,
          container_name,
      )
    else:
      logging.info(
          "No pods found yet for workload %s (queued or starting); waiting...",
          workload_id,
      )

    if time.time() >= deadline:
      raise AirflowException(
          f"Timed out after {startup_timeout}s waiting for head pod of"
          f" workload: {workload_id}"
      )
    time.sleep(poll_interval)


# TODO(cienet): remove pod detection logic (caller must specify it)
@task
def interrupt_worker_pod(
    workload_id: str, cluster_name: str, region: str, project_id: str
) -> str:
  """
  Authenticates with the GKE cluster and sends SIGILL to worker pod 0-1.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = _get_kubeconfig_env(
        project_id, region, cluster_name, kubeconfig_path=temp_config_file.name
    )

    candidate_prefixes = (
        f"{workload_id}-worker-0-1-",
        f"{workload_id}-worker-1-0-",
        f"{workload_id}-worker-0-0-",
    )
    pods = _list_workload_pods_kubectl(workload_id, env=env)

    if not pods:
      raise AirflowException(
          f"No pods found for workload selector: {workload_id}"
      )

    for pod in pods:
      phase = pod.get("status", {}).get("phase")
      pod_name = pod.get("metadata", {}).get("name", "")
      if phase in ("Failed", "Unknown") and not pod_name.startswith(
          candidate_prefixes
      ):
        raise AirflowFailException(f"Bad pod phase: {phase}")

    target_pod_name = None
    for prefix in candidate_prefixes:
      for pod in pods:
        phase = pod.get("status", {}).get("phase")
        pod_name = pod.get("metadata", {}).get("name", "")
        if pod_name.startswith(prefix) and phase == "Running":
          target_pod_name = pod_name
          break
      if target_pod_name:
        break

    if not target_pod_name:
      raise AirflowFailException(
          f"No pod found matching prefixes: {candidate_prefixes}"
      )

    logging.info(f"Found target pod: {target_pod_name}")

    cmd = (
        f"kubectl exec {target_pod_name} -n default -- /bin/sh -c 'kill -s"
        " SIGILL 1'"
    )

    try:
      subprocess_utils.run_exec(cmd, env=env)
    except subprocess_utils.ProcessKilledException:
      logging.info("Process was terminated with SIGKILL")
    except AirflowException as exc:
      if any(
          code in str(exc)
          for code in (
              "command terminated with exit code 132",
              "command terminated with exit code 137",
          )
      ):
        logging.info("Worker process terminated via SIGILL as expected.")
      elif "Forbidden" in str(exc) and "pods/exec" in str(exc):
        logging.info(
            "pods/exec forbidden by cluster RBAC; deleting pod %s instead.",
            target_pod_name,
        )
        delete_cmd = (
            f"kubectl delete pod {target_pod_name} -n default"
            " --grace-period=5 --wait=false"
        )
        subprocess_utils.run_exec(delete_cmd, env=env)
      else:
        raise

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# TODO(cienet): timeout might have conflict with steam
def _stream_pod_logs(
    pod_name: str,
    namespace: str,
    pattern: str,
    since_time: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
    container_name: str | None = None,
) -> str:
  """Streams pod logs in real time and exits as soon as pattern is found."""

  cmd = f"kubectl logs -f {pod_name} -n {namespace} --timestamps=true"
  if container_name:
    cmd += f" -c {container_name}"
  if since_time:
    cmd += f" --since-time={since_time}"

  # Open persistent streaming process
  process = subprocess_utils.run_streaming(cmd, env=env)

  # Establish deadline for bounded wait
  deadline = time.time() + timeout
  try:
    for line in iter(process.stdout.readline, ""):
      # Check if deadline has been exceeded
      if time.time() > deadline:
        raise AirflowException(
            f"Timeout after {timeout}s: pattern '{pattern}' was not found in logs."
        )
      # Direct line-by-line read prevents internal buffer locking
      clean_line = line.strip()
      if not clean_line:
        continue

      logging.info("[pod-log] %s", clean_line)

      # Check for target pattern match
      if pattern in clean_line:
        logging.info("Target log pattern '%s' matched!", pattern)

        if "Z " in clean_line:
          return clean_line.split("Z ")[0] + "Z"

        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Process ended naturally (container finished) without finding the string
    raise AirflowException(
        f"Log stream ended, but pattern '{pattern}' was not found."
    )

  # Exceptional outcome: the pipe crashed or dropped prematurely
  except (IOError, OSError) as e:
    raise AirflowException(f"Failed while reading process output: {e}") from e

  finally:
    process.kill()
    # Close the pipe to prevent resource leaks
    process.stdout.close()
    # Reaps the process to prevent zombie processes
    process.wait()


@task
def check_logs_stream(
    workload_id: str,
    cluster_name: str,
    region: str,
    project_id: str,
    expect_log_contains: str,
    since_time: str | None = None,
) -> str:
  """
  Checks if the running pod's logs contain a specific substring.
  """
  context = get_current_context()
  ti = context["task_instance"]

  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = _get_kubeconfig_env(
        project_id, region, cluster_name, kubeconfig_path=temp_config_file.name
    )
    head_pod, container_name = wait_for_running_head_pod(workload_id, env=env)

    if since_time:
      effective_since_time = since_time
      logging.info("Using upstream XCom since_time: %s", effective_since_time)
    else:
      effective_since_time = ti.start_date.astimezone(timezone.utc).strftime(
          "%Y-%m-%dT%H:%M:%S.%fZ"
      )
      logging.info(
          "No upstream or internal since_time. Using task start time: %s",
          effective_since_time,
      )

    pod_name = head_pod["metadata"]["name"]
    pod_namespace = head_pod["metadata"].get("namespace", "default")

    # Stream logs continuously until string is found
    timestamp = _stream_pod_logs(
        pod_name=pod_name,
        namespace=pod_namespace,
        pattern=expect_log_contains,
        since_time=effective_since_time,
        env=env,
        container_name=container_name,
    )

  return timestamp


def worker_pod_interruption(
    project_id: str = "",
    region: str = "",
    cluster_name: str = "",
    workload_id: str = "",
    times: int = 3,
    entry_log_pattern: str = "completed step:",
    elastic_log_pattern: str = "Elastic attempt",
    end_log_pattern: str = "Sufficient slices active:",
) -> DAGNode:
  """Run a test job with worker pod interruption."""
  with TaskGroup(group_id="worker_pod_interruption") as group:
    last_timestamp = None
    for i in range(1, times + 1):
      wait_for_step = check_logs_stream.override(
          task_id=f"wait_for_step_starts_{i}",
          retries=5,
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=entry_log_pattern,
          since_time=last_timestamp,
      )

      trigger_interrupt = interrupt_worker_pod.override(
          task_id=f"interrupt_worker_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
      )

      wait_for_step >> trigger_interrupt

      wait_for_elastic_attempt = check_logs_stream.override(
          task_id=f"wait_for_elastic_attempt_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=elastic_log_pattern,
          since_time=trigger_interrupt,
      )

      wait_for_slices_active = check_logs_stream.override(
          task_id=f"wait_for_slices_active_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=end_log_pattern,
          since_time=wait_for_elastic_attempt,
      )

      chain(
          wait_for_step,
          trigger_interrupt,
          wait_for_elastic_attempt,
          wait_for_slices_active,
      )

      last_timestamp = wait_for_slices_active

    return group
