"""Utilities to get workloads logs and some utils."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from absl import logging
import re

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api

from dags.orbax.util import gcs


@task
def generate_timestamp():
  return datetime.now(timezone.utc)


@task
def validate_log_exist(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """Validate the workload log `text filter` it is found during training."""

  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter=text_filter,
      start_time=start_time,
      end_time=end_time,
  )

  log_found = False

  for entry in entries:
    if entry.payload is not None and text_filter in str(entry.payload):
      payload_str = str(entry.payload)
      log_found = True
      for line in payload_str.split("\n"):
        logging.info("├─ Timestamp: %s", entry.timestamp)
        logging.info("└─ Payload:")
        logging.info("   %s", line)

  if log_found:
    logging.info("Validate success")
    return

  raise AirflowFailException("The log history is empty!")


@task
def validate_log_with_step(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    vali_step_list: Optional[list] = None,
) -> None:
  """
  Validates that a workload is training correctly by checking for specific log steps.

  This function queries logs from a specified GKE cluster and namespace.
  It searches for a log entry containing the string '(blocking + background)'
  and then compares the number of steps found against an expected list of steps.

  A mismatch in the number of steps will cause the validation to fail. This can
  happen if, for example, a restore operation causes the step count to restart
  from zero, leading to `len(vali_step_list) != len(found_steps)`.

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
    vali_step_list: Optional to validate list of steps
  Returns:
    bool: validate success or not
  """
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter=f'jsonPayload.message=~"{text_filter}"',
      start_time=start_time,
      end_time=end_time,
  )
  if vali_step_list is None:
    return
  new_step_list = []
  for entry in entries:
    if not entry.payload:
      continue
    payload_str = str(entry.payload)
    for line in payload_str.split("\n"):
      for step in vali_step_list:
        vali_str = "directory=/local/" + str(step)
        if vali_str in line and step not in new_step_list:
          logging.info(f"├─ Timestamp: {entry.timestamp}")
          logging.info("└─ Payload:")
          logging.info(f"   {line}")
          new_step_list.append(step)
  if len(vali_step_list) == len(new_step_list):
    logging.info("Validate success")
    return
  else:
    raise AirflowFailException(
        f"{len(vali_step_list)} saves are expected,"
        f"but got {len(new_step_list)}"
    )


@task
def validate_gcs_restore_log(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """
  Validates that a workload successfully restored from GCS checkpoints.

  This function queries logs from a specified GKE cluster and namespace
  to look for log entries with 'event_type': 'restore' and validates that
  the restored step corresponds to a previously saved step.

  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    namespace: Kubernetes namespace (defaults to "default")
    pod_pattern: Pattern to match pod names (defaults to "*")
    start_time: Optional start time for log retrieval
      (defaults to 12 hours ago)
    end_time: Optional end time for log retrieval (defaults to now)
  Returns:
    None: Raises AirflowFailException if restoration validation fails
  """
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      text_filter="event_type",
      start_time=start_time,
      end_time=end_time,
  )
  
  save_steps = set()
  restore_steps = set()
  
  for entry in entries:
    if not entry.payload:
      continue
    payload_str = str(entry.payload)
    
    for line in payload_str.split("\n"):
      if "'event_type': 'save'" in line or '"event_type": "save"' in line:
        # Extract step from save event
        import re
        step_match = re.search(r"'step': (\d+)|\"step\": (\d+)", line)
        if step_match:
          step = int(step_match.group(1) or step_match.group(2))
          save_steps.add(step)
          logging.info(f"├─ Found save event at step {step}")
          logging.info(f"└─ Timestamp: {entry.timestamp}")
      
      elif "'event_type': 'restore'" in line or '"event_type": "restore"' in line:
        # Extract step from restore event
        import re
        step_match = re.search(r"'step': (\d+)|\"step\": (\d+)", line)
        if step_match:
          step = int(step_match.group(1) or step_match.group(2))
          restore_steps.add(step)
          logging.info(f"├─ Found restore event at step {step}")
          logging.info(f"└─ Timestamp: {entry.timestamp}")
          logging.info(f"   Full log: {line}")
  
  if not restore_steps:
    raise AirflowFailException(
        "No restore events found. Emergency checkpoint restoration may have failed."
    )
  
  # Validate that all restore steps correspond to previously saved steps
  invalid_restores = restore_steps - save_steps
  if invalid_restores:
    raise AirflowFailException(
        f"Restore validation failed: Steps {invalid_restores} were restored "
        f"but not found in saved steps {save_steps}."
    )
  
  logging.info(f"GCS restore validation successful!")
  logging.info(f"Saved steps: {sorted(save_steps)}")
  logging.info(f"Restored steps: {sorted(restore_steps)}")
  logging.info(f"All {len(restore_steps)} restore events correspond to previously saved checkpoints.")


@task
def validate_gcs_checkpoint_files(
    bucket_path: str,
    vali_step_list: Optional[list] = None,
) -> None:
  """
  Validates that checkpoint files exist in GCS bucket for expected steps.

  This function uses the GCS utility to check that checkpoint files
  are properly saved in the bucket for each expected step.

  Args:
    bucket_path: The full gs:// path to the GCS bucket
    vali_step_list: Optional list of steps to validate
  Returns:
    None: Raises AirflowFailException if checkpoint validation fails
  """
  if vali_step_list is None:
    logging.info("No validation steps provided, skipping GCS checkpoint validation")
    return

  import dags.orbax.util.gcs_util as gcs_util
  
  try:
    checkpoint_files = gcs_util.get_gcs_checkpoint(bucket_path)
    logging.info(f"Found checkpoint files in GCS: {checkpoint_files}")
    
    # Extract step directories from checkpoint files
    found_steps = set()
    for file_path in checkpoint_files:
      # Extract directory names that are numeric (step numbers)
      path_parts = file_path.split('/')
      for part in path_parts:
        if part.isdigit():
          found_steps.add(int(part))
    
    expected_steps = set(vali_step_list)
    missing_steps = expected_steps - found_steps
    
    logging.info(f"Expected steps: {sorted(expected_steps)}")
    logging.info(f"Found steps: {sorted(found_steps)}")
    
    if missing_steps:
      raise AirflowFailException(
          f"GCS checkpoint validation failed: Missing checkpoint files for steps {sorted(missing_steps)}. "
          f"Expected steps: {sorted(vali_step_list)}, Found steps: {sorted(found_steps)}"
      )
    
    logging.info(f"GCS checkpoint validation successful!")
    logging.info(f"All {len(vali_step_list)} expected checkpoint files found in GCS")
    logging.info(f"Validated steps: {sorted(found_steps)}")
    
  except Exception as e:
    raise AirflowFailException(f"Error validating GCS checkpoints: {str(e)}")


@task
def validate_log_with_gcs_save(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    container_name: Optional[str] = None,
    text_filter: str = "(blocking + background).",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    vali_step_list: Optional[list] = None,
) -> None:
  """
  Validates that a workload is saving checkpoints to GCS correctly by checking for specific log steps.

  This function queries logs from a specified GKE cluster and namespace.
  It searches for log entries containing 'Finished async_save (blocking + background)'
  with GCS directory paths and compares the steps found against an expected list.

  Expected log format:
  "Finished async_save (blocking + background). Time taken: 124.161461s. 
   directory=gs://bucket/path/checkpoints/50"

  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    namespace: Kubernetes namespace (defaults to "default")
    pod_pattern: Pattern to match pod names (defaults to "*")
    container_name: Optional container name to filter logs
    text_filter: Text filter for async_save logs (defaults to specific pattern)
    start_time: Optional start time for log retrieval (defaults to 12 hours ago)
    end_time: Optional end time for log retrieval (defaults to now)
    vali_step_list: List of steps to validate
  Returns:
    None: Raises AirflowFailException if validation fails
  """
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter=text_filter,
      start_time=start_time,
      end_time=end_time,
  )
  
  if vali_step_list is None:
    raise AirflowFailException("vali_step_list is required for GCS save validation")
    
  found_steps = []
  
  for entry in entries:
    if not entry.payload:
      continue
    payload_str = str(entry.payload)
    
    for line in payload_str.split("\n"):
      if "Finished async_save (blocking + background)" in line and "directory=gs://" in line:
        # Extract step from GCS directory path
        # Example: directory=gs://cienet-mtc-bucket/max-ecm-res-gcs-ecm-2x-v6e-64-2025-09-01-08-39/checkpoints/50
        import re
        directory_match = re.search(r"directory=gs://[^/]+/[^/]+/checkpoints/(\d+)", line)
        if directory_match:
          step = int(directory_match.group(1))
          if step in vali_step_list and step not in found_steps:
            logging.info(f"├─ Found GCS save at step {step}")
            logging.info(f"├─ Timestamp: {entry.timestamp}")
            logging.info("└─ Payload:")
            logging.info(f"   {line}")
            found_steps.append(step)
  
  expected_count = len(vali_step_list)
  found_count = len(found_steps)
  
  if expected_count == found_count:
    logging.info(f"GCS save validation successful!")
    logging.info(f"Expected steps: {sorted(vali_step_list)}")
    logging.info(f"Found steps: {sorted(found_steps)}")
    return True
  else:
    missing_steps = set(vali_step_list) - set(found_steps)
    raise AirflowFailException(
        f"GCS save validation failed: {expected_count} saves expected, "
        f"but got {found_count}. Missing steps: {sorted(missing_steps)}"
    )


def list_log_entries(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
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

  # Construct the log filter
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
