"""Utilities for GKE."""

import base64
import concurrent.futures
import datetime
import logging
import tempfile
import time
from typing import Any, Dict, Optional

from airflow.decorators import task, task_group
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
import google.auth
import google.auth.transport.requests
from google.cloud import container_v1
import kubernetes
import urllib3

from xlml.apis import gcp_config, test_config
from xlml.utils import composer


class PodsNotReadyError(Exception):
  """Exception raised when pods are not ready within the expected timeout."""

  def __init__(self, message):
    super().__init__(message)


def get_authenticated_client(
    project_name: str, region: str, cluster_name: str
) -> kubernetes.client.ApiClient:
  container_client = container_v1.ClusterManagerClient()
  cluster_path = (
      f"projects/{project_name}/locations/{region}/clusters/{cluster_name}"
  )
  response = container_client.get_cluster(name=cluster_path)
  creds, _ = google.auth.default()
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  configuration = kubernetes.client.Configuration()
  configuration.host = f"https://{response.endpoint}"

  ca_cert_content = base64.b64decode(
      response.master_auth.cluster_ca_certificate
  )
  with tempfile.NamedTemporaryFile(delete=False) as ca_cert:
    ca_cert.write(ca_cert_content)
    configuration.ssl_ca_cert = ca_cert.name
  configuration.api_key_prefix["authorization"] = "Bearer"
  configuration.api_key["authorization"] = creds.token

  return kubernetes.client.ApiClient(configuration)


def get_core_api_client(
    project_id: str, region: str, cluster_name: str
) -> kubernetes.client.CoreV1Api:
  """Create a core API client for the given cluster."""
  client = get_authenticated_client(project_id, region, cluster_name)
  core_api = kubernetes.client.CoreV1Api(client)
  logging.info(
      "Successfully initialized k8s core API client from cluster response."
  )
  return core_api


def get_batch_api_client(
    project_id: str, region: str, cluster_name: str
) -> kubernetes.client.BatchV1Api:
  """Create a batch API client for the given cluster."""
  client = get_authenticated_client(project_id, region, cluster_name)
  batch_api = kubernetes.client.BatchV1Api(client)
  logging.info(
      "Successfully initialized k8s batch API client from cluster response."
  )
  return batch_api


def get_custom_objects_api_client(
    project_id: str, region: str, cluster_name: str
) -> kubernetes.client.CustomObjectsApi:
  """Create a custom objects API client for the given cluster."""
  client = get_authenticated_client(project_id, region, cluster_name)
  return kubernetes.client.CustomObjectsApi(client)


_JOBSET_CRD_GROUP = "jobset.x-k8s.io"
_JOBSET_CRD_VERSION = "v1alpha2"
_JOBSET_CRD_PLURAL = "jobsets"

_JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"
_REPLICATED_JOB_LABEL = "jobset.sigs.k8s.io/replicatedjob-name"
_RESTART_ATTEMPT_LABEL = "jobset.sigs.k8s.io/restart-attempt"
_HEAD_REPLICATED_JOB = "pathways-head"

_JOBSET_STATE_COMPLETED = "completed"
_JOBSET_STATE_RESTARTING = "restarting"
_JOBSET_STATE_FAILED = "failed"


def list_workload_pods(
    core_api: kubernetes.client.CoreV1Api,
    workload_id: str,
    namespace: str = "default",
) -> kubernetes.client.V1PodList:
  """List all pods for the given workload (Job or JobSet)."""
  logging.info(
      f"Getting pods for workload_id: {workload_id} in namespace: {namespace}"
  )
  pods = core_api.list_namespaced_pod(
      namespace=namespace, label_selector=f"job-name={workload_id}"
  )
  if not pods.items:
    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"{_JOBSET_NAME_LABEL}={workload_id}",
    )
  return pods


def get_workload_job(
    batch_api: kubernetes.client.BatchV1Api,
    workload_id: str,
    namespace: str = "default",
) -> Optional[kubernetes.client.V1Job]:
  """Get the Kubernetes Job object for a given workload."""
  logging.info(
      f"Getting job for workload_id: {workload_id} in namespace: {namespace}"
  )
  try:
    return batch_api.read_namespaced_job(name=workload_id, namespace=namespace)
  except kubernetes.client.exceptions.ApiException as e:
    logging.info(
        f"Direct job read failed for {workload_id} ({e}); trying label"
        " selector..."
    )

  try:
    jobs = batch_api.list_namespaced_job(
        label_selector=f"{_JOBSET_NAME_LABEL}={workload_id}",
        namespace=namespace,
    )
    if not jobs.items:
      return None
    if len(jobs.items) > 1:
      logging.info(f"Got more than one job for workload_id: {workload_id}")
    return jobs.items[0]
  except kubernetes.client.exceptions.ApiException as e:
    logging.info(f"Could not list Kubernetes Jobs for {workload_id}: {e}")
    return None


def get_workload_jobset(
    custom_api: kubernetes.client.CustomObjectsApi,
    workload_id: str,
    namespace: str = "default",
) -> Optional[Dict[str, Any]]:
  """Get the Kubernetes JobSet CRD object for a given workload."""
  try:
    return custom_api.get_namespaced_custom_object(
        group=_JOBSET_CRD_GROUP,
        version=_JOBSET_CRD_VERSION,
        namespace=namespace,
        plural=_JOBSET_CRD_PLURAL,
        name=workload_id,
    )
  except kubernetes.client.exceptions.ApiException as e:
    logging.info(f"Could not read JobSet {workload_id}: {e}")
    return None


def print_container_logs(
    core_api: kubernetes.client.CoreV1Api,
    namespace: str,
    pod_name: str,
    container_name: str,
) -> None:
  """Prints the full logs of a single container."""
  try:
    # `tail_lines` is intentionally omitted so that the whole log is returned.
    # `_preload_content=False` streams the logs instead of buffering the whole
    # response in memory. It returns the raw `urllib3.HTTPResponse`, which
    # yields one `bytes` line per iteration, hence the decoding below.
    response = core_api.read_namespaced_pod_log(
        name=pod_name,
        namespace=namespace,
        container=container_name,
        _preload_content=False,
    )
  except (
      kubernetes.client.exceptions.ApiException,
      # The Kubernetes client only converts SSL errors into `ApiException`, so
      # connection errors surface as raw urllib3 errors.
      urllib3.exceptions.HTTPError,
  ) as e:
    logging.warning(
        "Could not retrieve logs for pod %s, container %s: %s",
        pod_name,
        container_name,
        e,
    )
    return

  logging.info(
      "--- Logs for pod %s, container %s ---", pod_name, container_name
  )
  try:
    for line in response:
      logging.info(line.decode("utf-8", "replace").rstrip("\n"))
  except urllib3.exceptions.HTTPError as e:
    logging.warning(
        "Log stream of pod %s, container %s was interrupted: %s",
        pod_name,
        container_name,
        e,
    )
  finally:
    # The connection is not returned to the pool automatically when the
    # response is streamed.
    response.release_conn()


def print_pod_logs(
    core_api: kubernetes.client.CoreV1Api,
    pod: kubernetes.client.V1Pod,
) -> None:
  """Prints the full logs of all containers in a pod."""
  containers = pod.spec.containers if pod.spec else None
  if not containers:
    logging.warning("No containers found for pod %s.", pod.metadata.name)
    return

  for container in containers:
    print_container_logs(
        core_api,
        namespace=pod.metadata.namespace,
        pod_name=pod.metadata.name,
        container_name=container.name,
    )


def log_workload_pod_statuses(
    workload_id: str, pods: kubernetes.client.V1PodList
) -> None:
  """Logs the status of each retrieved pod and its containers."""
  if not pods or not pods.items:
    return

  logging.info(f"{f' Pod Statuses for Workload {workload_id} ':-^80}")
  for pod in pods.items:
    logging.info(f"Pod: {pod.metadata.name}, Status: {pod.status.phase}")
    if not pod.status.container_statuses:
      continue
    for container_status in pod.status.container_statuses:
      match container_status.state:
        case state if state.waiting:
          w = state.waiting
          logging.warning(
              f"  Container '{container_status.name}' WAITING. "
              f"Reason: {w.reason}. Message: {w.message}"
          )
        case state if state.terminated:
          t = state.terminated
          logging.error(
              f"  Container '{container_status.name}' TERMINATED. "
              f"Reason: {t.reason}. Exit Code: {t.exit_code}"
          )
  logging.info("-" * 80)


LOGGING_URL_FORMAT = (
    "https://console.cloud.google.com/logs/viewer"
    "?project={project}&resource=k8s_container"
    "&minLogLevel=0&expandAll=false"
    "&customFacets=&limitCustomFacetWidth=true"
    "&filters=text:job-name%3D{workload_id}"
)


def _restart_attempt(pod: kubernetes.client.V1Pod) -> Optional[int]:
  labels = getattr(pod.metadata, "labels", None)
  raw = labels.get(_RESTART_ATTEMPT_LABEL) if isinstance(labels, dict) else None
  return int(raw) if isinstance(raw, str) and raw.isdigit() else None


def _latest_attempt_pods(
    pods: list[kubernetes.client.V1Pod],
) -> list[kubernetes.client.V1Pod]:
  attempts = [a for p in pods if (a := _restart_attempt(p)) is not None]
  if not attempts:
    return list(pods)
  latest = max(attempts)
  return [p for p in pods if _restart_attempt(p) in (None, latest)]


def _has_failed_container(pod: kubernetes.client.V1Pod) -> bool:
  if not pod.status or not pod.status.container_statuses:
    return False
  return any(
      cs.state and cs.state.terminated and cs.state.terminated.exit_code != 0
      for cs in pod.status.container_statuses
  )


def _get_jobset_state(
    pod: kubernetes.client.V1Pod,
    project_id: str,
    region: str,
    cluster_name: str,
    workload_id: str,
    namespace: str,
    jobset_cache: Optional[Dict[str, Any]] = None,
) -> str:
  """Returns 'completed', 'restarting', or 'failed' for a JobSet pod."""
  labels = getattr(pod.metadata, "labels", None)
  if not isinstance(labels, dict) or (
      _JOBSET_NAME_LABEL not in labels and _REPLICATED_JOB_LABEL not in labels
  ):
    return _JOBSET_STATE_FAILED
  if jobset_cache is not None and "jobset" in jobset_cache:
    jobset = jobset_cache["jobset"]
  else:
    custom_api = get_custom_objects_api_client(project_id, region, cluster_name)
    jobset = get_workload_jobset(custom_api, workload_id, namespace=namespace)
    if jobset_cache is not None:
      jobset_cache["jobset"] = jobset
  if not isinstance(jobset, dict):
    return _JOBSET_STATE_FAILED
  status = jobset.get("status") or {}
  conditions = status.get("conditions") or []
  if any(
      c.get("type") == "Completed" and c.get("status", "True") != "False"
      for c in conditions
  ):
    return _JOBSET_STATE_COMPLETED
  if any(
      c.get("type") == "Failed" and c.get("status", "True") != "False"
      for c in conditions
  ):
    return _JOBSET_STATE_FAILED
  max_restarts = ((jobset.get("spec") or {}).get("failurePolicy") or {}).get(
      "maxRestarts", 0
  )
  restarts = status.get("restarts", 0)
  pod_attempt = _restart_attempt(pod)
  is_stale_attempt = (
      pod_attempt is not None
      and isinstance(restarts, int)
      and pod_attempt < restarts
  )
  if is_stale_attempt or (
      isinstance(max_restarts, int)
      and isinstance(restarts, int)
      and max_restarts > restarts
  ):
    return _JOBSET_STATE_RESTARTING
  return _JOBSET_STATE_FAILED


def _jobset_verdict(
    pod: kubernetes.client.V1Pod,
    project_id: str,
    region: str,
    cluster_name: str,
    workload_id: str,
    namespace: str,
    core_api: Optional[kubernetes.client.CoreV1Api] = None,
    pods_to_check: Optional[list[kubernetes.client.V1Pod]] = None,
    jobset_cache: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
  """Returns sensor verdict (True=done, False=keep polling, None=fail)."""
  jobset_state = _get_jobset_state(
      pod,
      project_id,
      region,
      cluster_name,
      workload_id,
      namespace,
      jobset_cache=jobset_cache,
  )
  if jobset_state == _JOBSET_STATE_COMPLETED:
    logging.info(f"Workload {workload_id} JobSet completed successfully.")
    if core_api is not None and pods_to_check:
      for succeeded_pod in pods_to_check:
        if (
            succeeded_pod.status.phase == "Succeeded"
            and not _has_failed_container(succeeded_pod)
        ):
          print_pod_logs(core_api, succeeded_pod)
    return True
  if jobset_state == _JOBSET_STATE_RESTARTING:
    logging.info(
        f"Pod {pod.metadata.name} failed, waiting for JobSet"
        f" {workload_id} to restart."
    )
    return False
  return None


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_workload_start(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    namespace: str = "default",
) -> bool:
  """Wait for workload to start running."""
  core_api = get_core_api_client(project_id, region, cluster_name)
  pods = list_workload_pods(core_api, workload_id, namespace=namespace)
  log_workload_pod_statuses(workload_id, pods)

  if not pods.items:
    logging.info(f"Waiting for pods of workload: {workload_id} to be created.")
    return False

  active_pods = _latest_attempt_pods(pods.items)
  jobset_cache: Dict[str, Any] = {}
  for pod in active_pods:
    if pod.status.phase in ["Pending", "Unknown"]:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
      return False
    if pod.status.phase == "Failed":
      verdict = _jobset_verdict(
          pod,
          project_id,
          region,
          cluster_name,
          workload_id,
          namespace,
          jobset_cache=jobset_cache,
      )
      if verdict is not None:
        return verdict
      print_pod_logs(core_api, pod)
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
  core_api = get_core_api_client(project_id, region, cluster_name)
  pods = list_workload_pods(core_api, workload_id, namespace=namespace)
  log_workload_pod_statuses(workload_id, pods)

  if not pods.items:
    batch_api = get_batch_api_client(project_id, region, cluster_name)
    job = get_workload_job(batch_api, workload_id, namespace=namespace)
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

    custom_api = get_custom_objects_api_client(project_id, region, cluster_name)
    jobset = get_workload_jobset(custom_api, workload_id, namespace=namespace)
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

  active_pods = _latest_attempt_pods(pods.items)
  # For Pathways workloads, the JobSet completes as soon as pathways-head
  # succeeds. Worker pods may exit non-zero after the head client disconnects
  # before garbage collection deletes them.
  head_pods = [
      pod
      for pod in active_pods
      if (
          isinstance(getattr(pod.metadata, "labels", None), dict)
          and pod.metadata.labels.get(_REPLICATED_JOB_LABEL)
          == _HEAD_REPLICATED_JOB
      )
      or (
          isinstance(getattr(pod.metadata, "name", None), str)
          and f"-{_HEAD_REPLICATED_JOB}-" in pod.metadata.name
      )
  ]
  pods_to_check = (
      head_pods
      if head_pods and all(pod.status.phase == "Succeeded" for pod in head_pods)
      else active_pods
  )
  jobset_cache: Dict[str, Any] = {}

  for pod in pods_to_check:
    if pod.status.phase in ["Pending", "Running", "Unknown"]:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
      return False
    if pod.status.phase == "Failed":
      verdict = _jobset_verdict(
          pod,
          project_id,
          region,
          cluster_name,
          workload_id,
          namespace,
          core_api=core_api,
          pods_to_check=pods_to_check,
          jobset_cache=jobset_cache,
      )
      if verdict is not None:
        return verdict
      print_pod_logs(core_api, pod)
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

  for pod in pods_to_check:
    if pod.status.container_statuses:
      for container_status in pod.status.container_statuses:
        if (
            container_status.state
            and container_status.state.terminated
            and container_status.state.terminated.exit_code != 0
        ):
          verdict = _jobset_verdict(
              pod,
              project_id,
              region,
              cluster_name,
              workload_id,
              namespace,
              core_api=core_api,
              pods_to_check=pods_to_check,
              jobset_cache=jobset_cache,
          )
          if verdict is not None:
            return verdict
          print_container_logs(
              core_api,
              namespace=namespace,
              pod_name=pod.metadata.name,
              container_name=(
                  container_status.name or pod.spec.containers[0].name
              ),
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

  # Fetch logs for successful pods before returning
  for pod in pods_to_check:
    print_pod_logs(core_api, pod)

  logging.info("All pod(s) phase are succeeded.")
  return True


@task_group
def run_job(
    body: Dict[str, Any],
    gcp: gcp_config.GCPConfig,
    gke_test_config: test_config.GpuGkeTest,
    cluster_name: str,
    job_create_timeout: datetime.timedelta,
    task_owner: str,
    gcs_location: str = "",
):
  """Run a batch job directly on a GKE cluster.

  Args:
    body: Dict that defines a Kubernetes `Job`.
    gcp: GCP config with the project name and zone of the GKE cluster.
    gke_test_config: Test config with the accelerator information of the GKE
      cluster.
    cluster_name: Name of the GCP cluster.
    job_create_timeout: Amount of time to wait for all pods to become active.
    task_owner: Task owner username or link.
    gcs_location: GCS path for all artifacts of the test.
  """

  @task
  def deploy_job(gcs_location):
    # Log required info for XLML PLX Dashboard
    composer.log_metadata_for_xlml_dashboard({
        "cluster_project": gcp.project_name,
        "zone": gcp.zone,
        "dataset_name": gcp.dataset_name.value,
        "composer_project": gcp.composer_project,
        "dataset_project": gcp.dataset_project,
        "cluster_name": cluster_name,
        "accelerator_type": gke_test_config.accelerator.machine_type,
    })

    body["spec"]["template"]["spec"]["containers"][0]["env"].append(
        {"name": "GCS_OUTPUT", "value": gcs_location}
    )
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    jobs_client = kubernetes.client.BatchV1Api(client)

    resp = jobs_client.create_namespaced_job(namespace="default", body=body)

    logging.info(f"response: {resp}")

    return resp.metadata.name

  @task.sensor(
      poke_interval=60,
      timeout=job_create_timeout.total_seconds(),
      mode="reschedule",
  )
  def wait_all_pods_ready(name: str):
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    batch_api = kubernetes.client.BatchV1Api(client)
    job = batch_api.read_namespaced_job(namespace="default", name=name)

    # TODO(wcromar): Handle other conditions (e.g. unschedulablility)
    logging.info(f"Job status: {job.status}")
    if job.status.failed:
      raise RuntimeError(f"Job has {job.status.failed} failed pods.")

    core_api = kubernetes.client.CoreV1Api(client)
    pod_label_selector = f"batch.kubernetes.io/job-name={name}"
    pods = core_api.list_namespaced_pod(
        namespace="default", label_selector=pod_label_selector
    )

    if len(pods.items) != body["spec"]["parallelism"]:
      logging.info("Waiting for all pods to be created...")
      return False

    return True

  @task(retries=6)
  def stream_logs(name: str):
    def _watch_pod(name, namespace) -> Optional[int]:
      logs_watcher = kubernetes.watch.Watch()

      logging.info(f"Waiting for pod {name} to start...")
      pod_watcher = kubernetes.watch.Watch()
      for event in pod_watcher.stream(
          core_api.list_namespaced_pod,
          namespace,
          field_selector=f"metadata.name={name}",
      ):
        status = event["object"].status
        logging.info(
            f'Pod {event["object"].metadata.name} status: {status.phase}'
        )
        if status.phase != "Pending":
          break

      logging.info(f"Streaming pod logs for {name}...")
      for line in logs_watcher.stream(
          core_api.read_namespaced_pod_log,
          name,
          namespace,
          _request_timeout=3600,
      ):
        logging.info(f"{name}] {line}")

      logging.warning(f"Lost logs stream for {name}.")

      pod = core_api.read_namespaced_pod(namespace="default", name=name)
      if pod.status.container_statuses:
        container_status = pod.status.container_statuses[0]
        if pod.status.container_statuses[0].state.terminated:
          exit_code = container_status.state.terminated.exit_code
          if exit_code:
            logging.error(f"Pod {name} had non-zero exit code {exit_code}")

          return exit_code

      logging.warning(f"Unknown status for pod {name}")
      return None

    # We need to re-authenticate if the stream_logs fail. This can happen when
    # the job runs for too long and the credential expire.
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    core_api = kubernetes.client.CoreV1Api(client)
    pod_label_selector = f"batch.kubernetes.io/job-name={name}"
    pods = core_api.list_namespaced_pod(
        namespace="default", label_selector=pod_label_selector
    )
    # TODO(piz): Use time.sleep may not be a good solution here. However, I
    # expect resources are all ready in wait_all_pods_ready stage. This just in
    # case authentication takes time. Check with Will for better solutions.
    time.sleep(30)
    if len(pods.items) != body["spec"]["parallelism"]:
      logging.info("Waiting for all pods to be re-connected...")
      raise PodsNotReadyError("pods are not ready after refreshing credential.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = []
      for pod in pods.items:
        f = executor.submit(
            _watch_pod, pod.metadata.name, pod.metadata.namespace
        )
        futures.append(f)

      # Wait for pods to complete, and exit with the first non-zero exit code.
      for f in concurrent.futures.as_completed(futures):
        try:
          # TODO(piz/wcromar): it looks like there is a delay between
          # as_completed and update of f.result(). exit_code can be None even
          # task is complete.
          exit_code = f.result()
        except kubernetes.client.ApiException as e:
          logging.error("Kubernetes error. Retrying...", exc_info=e)
          exit_code = None

        # Retry if status is unknown
        if exit_code is None:
          raise RuntimeError("unknown exit code")
        if exit_code:
          raise RuntimeError("Non-zero exit code")

  name = deploy_job.override(owner=task_owner)(gcs_location)
  chain(wait_all_pods_ready(name), stream_logs(name))


def zone_to_region(zone: str) -> str:
  zone_terms = zone.split("-")
  return zone_terms[0] + "-" + zone_terms[1]
