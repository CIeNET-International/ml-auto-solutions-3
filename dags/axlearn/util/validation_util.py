
"""Utilities to get workloads logs and some utils."""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from absl import logging
import re
import json

from airflow.providers.google.cloud.operators.gcs import GCSHook
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api
from airflow.hooks.subprocess import SubprocessHook


@task
def generate_timestamp():
  return datetime.now(timezone.utc)

@task
def get_image_name(
  project_id: str,
  path_repository: str,
)-> str | None :
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
  result = hook.run_command(
    ["bash", "-c", ";".join(cmds)]
  )
  assert (
      result.exit_code == 0
  ), f"XPK command failed with code {result.exit_code}"

  try:
    repo_details: List[Dict] = json.loads(result.output)
  except json.JSONDecodeError:
    raise ValueError("Failed to parse JSON output from gcloud command.")

  image_name = ""
  for image_info in repo_details:
    tags_list = image_info.get("tags", [])
    if len(tags_list) >= 2:
      tags_list.remove("latest")
      image_name = f"{path_repository}:{tags_list[0]}"
      return image_name
  raise AirflowFailException('Image not found or is not latest image')


@task
def validate_checkpoints_save_regular_axlearn(
    project_id: str,
    run_name: str,
    location: str,
    cluster_name: str,
    steps_to_validate: list,
    pod_pattern: Optional[str] = ".*",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:

  log_pattern = r"^Serialization.*?step_(?P<step>\d+).*"
  complied_pattern = re.compile(log_pattern)
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      pod_pattern=pod_pattern,
      text_filter=f'jsonPayload.message=~"{log_pattern}"',
      start_time=start_time,
      end_time=end_time,
  )
  steps_are_saved: set[int] = set()  # Use a set for faster lookup.
  for entry in entries:
    if not isinstance(entry, logging_api.StructEntry):
      raise AirflowFailException(
          "Log entry must be contain a jsonPayload attribute."
      )
    message = entry.payload.get("message")
    if not message:
      raise AirflowFailException(f"Failed to parse entry {entry}")

    m = complied_pattern.search(message)
    if m:
      steps_are_saved.add(int(m.group(1)))

  for step in steps_to_validate:
    if step not in steps_are_saved:
      logging.info(f"Found entries: {entries}")
      raise AirflowFailException(
          f"Failed to validate. Expect steps are saved: {steps_to_validate}; "
          f"got: {steps_are_saved}"
      )


def list_log_entries(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> list[logging_api.LogEntry]:
  """
  List log entries for the specified Google Cloud project.
  This function connects to Google Cloud Logging,
  constructs a filter for Kubernetes container logs
  within a specific project, location, cluster, namespace,
  and pod name pattern, and retrieves log
  entries from the specified time range.
  It prints the timestamp, severity, resource information,
  and payload for each log entry found.

  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    namespace: Kubernetes namespace (defaults to "default")
    pod_pattern: Pattern to match pod names (defaults to "*")
    container_name: Optional container name to filter logs
    text_filter: Optional comma-separated string to
      filter log entries by textPayload content
    start_time: Optional start time for log retrieval
      (defaults to 12 hours ago)
    end_time: Optional end time for log retrieval (defaults to now)
  Returns:
    bool: Number of log entries found
  """

  logging_client = logging_api.Client(project=project_id)

  # Set the time window for log retrieval:
  # default to last 12 hours if not provided
  if end_time is None:
    end_time = datetime.now(timezone.utc)
  if start_time is None:
    start_time = end_time - timedelta(hours=12)

  # Format times as RFC3339 UTC "Zulu" format required by the Logging API
  start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
  end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

  conditions = [
      f'resource.labels.project_id="{project_id}"',
      f'resource.labels.location="{location}"',
      f'resource.labels.cluster_name="{cluster_name}"',
      f'resource.labels.namespace_name="{namespace}"',
      f'resource.labels.pod_name=~"{pod_pattern}"',
      "severity>=DEFAULT",
      f'timestamp>="{start_time_str}"',
      f'timestamp<="{end_time_str}"',
  ]

  if container_name:
    conditions.append(f'resource.labels.container_name="{container_name}"')
  if text_filter:
    conditions.append(f"{text_filter}")

  log_filter = " AND ".join(conditions)

  logging.info(f"Log filter constructed: {log_filter}")
  return list(logging_client.list_entries(filter_=log_filter))
