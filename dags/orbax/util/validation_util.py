"""Utilities to get workloads logs and some utils."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from absl import logging
import re

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api

from dags.orbax.util import gcs
from xlml.utils import xpk


@task
def generate_timestamp():
  return datetime.now(timezone.utc)


@task
def validate_checkpoint_saves(
    project_id: str,
    location: str,
    cluster_name: str,
    steps_to_validate: list,
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """
  Validates that a workload is saving checkpoints correctly by checking for specific log steps.
  
  This function queries logs from a specified GKE cluster and namespace.
  It searches for log entries containing step numbers and compares 
  the steps found against an expected list.
  
  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    steps_to_validate: List of steps to validate (required)
    pod_pattern: Pattern to match pod names (defaults to "*")
    container_name: Optional container name to filter logs
    text_filter: Text filter for save event logs (defaults to "event_type")
    start_time: Optional start time for log retrieval (defaults to 12 hours ago)
    end_time: Optional end time for log retrieval (defaults to now)
  Returns:
    None: Raises AirflowFailException if validation fails
  """
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter='jsonPayload.message=~"\'event_type\': \'save\'"',
      start_time=start_time,
      end_time=end_time,
  )

  found_steps = []

  for entry in entries:
    if isinstance(entry, logging_api.StructEntry):
      message = entry.payload.get("message")
      if message:
        # Try to extract step number from the message
        try:
          # Use regex to find step number in the message
          step_match = re.search(r"'step':\s*(\d+)", message)
          if step_match:
            step = int(step_match.group(1))

            # Add all found steps to the list
            if step not in found_steps:
              logging.info(f"├─ Found checkpoint save at step {step}")
              logging.info(f"├─ Timestamp: {entry.timestamp}")
              logging.info("└─ Message:")
              logging.info(f"   {message}")
              found_steps.append(step)

        except (ValueError, AttributeError) as e:
          # Skip entries that can't be parsed
          logging.debug(f"Could not parse log entry: {e}")
          continue

  # Compare the found steps with expected steps directly
  expected_steps = sorted(steps_to_validate)
  found_steps_sorted = sorted(found_steps)

  if expected_steps == found_steps_sorted:
    logging.info(f"Checkpoint validation successful!")
    logging.info(f"Expected steps: {expected_steps}")
    logging.info(f"Found steps: {found_steps_sorted}")
    return
  else:
    missing_steps = set(steps_to_validate) - set(found_steps)
    extra_steps = set(found_steps) - set(steps_to_validate)
    
    error_msg = f"Checkpoint validation failed: Expected {expected_steps}, Found {found_steps_sorted}"
    if missing_steps:
      error_msg += f". Missing steps: {sorted(missing_steps)}"
    if extra_steps:
      error_msg += f". Extra steps: {sorted(extra_steps)}"
    
    raise AirflowFailException(error_msg)


@task
def generate_run_name(
    short_id: str,
    checkpointing_type: str,
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
      A string formatted as '{short_id}-mtc-{slice_number}x-{accelerator}-{timestamp}'.
  """

  run_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  run_name = f"{short_id}-{checkpointing_type}-{slice_number}x-{accelerator}-{run_time}"
  return run_name


@task
def validate_log_with_gcs(
    project_id: str,
    location: str,
    cluster_name: str,
    checkpoint_dir: str,
    namespace: str = "default",
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """
  Validates workload logs against GCS bucket checkpoints.

  This function first queries logs from a specified GKE cluster to determine the
  GCS bucket path used for checkpointing. It then retrieves logs related to
  checkpoint save operations, extracts the step numbers, and verifies that the
  corresponding checkpoint files exist in the GCS bucket. The function passes if the
  latest step found in the logs matches the latest step found in the GCS bucket's
  checkpoint filenames. It raises an exception on failure.

  Args:
    project_id: The Google Cloud project ID.
    location: The GKE cluster location.
    cluster_name: The GKE cluster name.
    namespace: The Kubernetes namespace. Defaults to "default".
    pod_pattern: A glob pattern to match pod names. Defaults to "*".
    container_name: An optional container name to filter logs by.
    text_filter: An optional string to filter log entries by their `textPayload`.
    start_time: The start time for log retrieval.
    end_time: The end time for log retrieval.

  Returns:
    None. The function completes successfully if all validation steps are found.

  Raises:
    AirflowFailException: If the bucket path format is invalid, if checkpoint files
      are missing, if steps cannot be extracted from log lines,if step lists
      are empty, or if the latest steps do not match.
  """

  # Get the entries for the backup steps in the bucket. To later compare the
  # latest stored step in bucket with the latest recorded step in training pod.
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter=f'textPayload=~"{text_filter}"',
      start_time=start_time,
      end_time=end_time,
  )
  gcs_save_step_list = []
  gcs_save_step_list_bucket = []
  for entry in entries:
    if entry.payload is not None:
      payload_str = str(entry.payload)
      for line in payload_str.split("\n"):
        # Extract the gcs bucket path from replicator logs
        gcs_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2,}"
        step_pattern = r"step (\d+)"
        match_gcs = re.search(gcs_pattern, line)
        match_step = re.search(step_pattern, line)
        validate_check_gcs = False

        # If could not found those valuses eg. gcs=2025-08-10_12-09 and step=60.
        if match_gcs and match_step and checkpoint_dir:
          gcs_checkpoint_path = match_gcs.group(0)
          step = match_step.group(1)
          logging.info(f"get gcs path from: {gcs_checkpoint_path}")
          bucket_files = gcs.get_gcs_checkpoint(
              f"{checkpoint_dir}/{gcs_checkpoint_path}/"
          )
          logging.info(f"gcs bucket files lenght: {len(bucket_files)}")
          if len(bucket_files) > 0:
            # Extract .meta file to future comparision
            for file in bucket_files:
              if ".meta" in file:
                gcs_save_step_list_bucket.append(file)
                break

            # Check for correct format .data
            for file in bucket_files:
              if ".data" in file:
                validate_check_gcs = True
                break

          if not validate_check_gcs:
            raise AirflowFailException(
                f"Checkpoint files can not found in {gcs_checkpoint_path}"
            )

          # Add it to a global list that we will use later to compare with bucket
          gcs_save_step_list.append(int(step))
        else:
          raise AirflowFailException(
              f"Could not find gcs_checkpoint_path or step in line: {line}"
          )

  # Compare last step found in replicator logs and last (only one)
  # step extracted from filename bucket
  if len(gcs_save_step_list_bucket) > 0 and len(gcs_save_step_list) > 0:
    # Extract s60 from  file name with extension .meta
    pattern_bucket_step = r"s(\d+)"
    raw_str_filename = gcs_save_step_list_bucket[-1]
    match = re.search(pattern_bucket_step, raw_str_filename)
    if match is None:
      raise AirflowFailException(
          f"Could not extract step from filename: {raw_str_filename}"
      )
    last_step_bucket = match.group(0)[1:]
    if int(last_step_bucket) == max(gcs_save_step_list):
      logging.info("Validate success")
  else:
    raise AirflowFailException(
        f"Steps in bucket or replicator logs are empty. "
        f"GCS bucket steps found: {len(gcs_save_step_list_bucket)}. "
        f"Replicator log steps found: {len(gcs_save_step_list)}."
    )
  return max(gcs_save_step_list), max(gcs_save_step_list_bucket)


@task
def validate_gcs_checkpoint_files(
    bucket_path: str,
    steps_to_validate: Optional[list] = None,
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
  if steps_to_validate is None:
    logging.info("No validation steps provided, skipping GCS checkpoint validation")
    return

  try:
    checkpoint_files = gcs.get_gcs_checkpoint(bucket_path)
    logging.info(f"Found checkpoint files in GCS: {checkpoint_files}")

    # Extract step directories from checkpoint files
    found_steps = set()
    for file_path in checkpoint_files:
      # Extract directory names that are numeric (step numbers)
      path_parts = file_path.split('/')
      for part in path_parts:
        if part.isdigit():
          found_steps.add(int(part))

    expected_steps = set(steps_to_validate)
    missing_steps = expected_steps - found_steps

    logging.info(f"Expected steps: {sorted(expected_steps)}")
    logging.info(f"Found steps: {sorted(found_steps)}")

    if missing_steps:
      raise AirflowFailException(
          f"GCS checkpoint validation failed: Missing checkpoint files for steps {sorted(missing_steps)}. "
          f"Expected steps: {sorted(steps_to_validate)}, Found steps: {sorted(found_steps)}"
      )

    logging.info(f"GCS checkpoint validation successful!")
    logging.info(
        f"All {len(steps_to_validate)} expected checkpoint files found in GCS"
    )
    logging.info(f"Validated steps: {sorted(found_steps)}")

  except Exception as e:
    raise AirflowFailException(f"Error validating GCS checkpoints: {str(e)}")


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


@task
def validate_log_exist(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """Validate the workload log text filter it is found during training."""

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

  if not entries:
    raise AirflowFailException("The log history is empty!")


@task
def validate_restored_correct_checkpoint(
    project_id: str,
    location: str,
    cluster_name: str,
    interrupt_at_step: int,
    pod_pattern: str = ".*",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """Validate the restored step is in the expected range."""

  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace="default",
      pod_pattern=pod_pattern,
      text_filter="jsonPayload.message:\"'event_type'\"",
      start_time=start_time,
      end_time=end_time,
  )

  if not entries:
    raise AirflowFailException("No event_type found in the log.")

  save_step_before_restore = None
  for entry in entries:
    if not isinstance(entry, logging_api.StructEntry):
      raise AirflowFailException(
          "Log entry must be contain a jsonPayload attribute."
      )
    message = entry.payload.get("message")
    if not message:
      continue

    if re.search(r"'event_type': 'save'", message):
      match = re.search(r"'step': (\d+)", message)
      if match:
        save_step_before_restore = int(match.group(1))
      else:
        raise AirflowFailException(
            f"Found save event with no step number, message: {message}"
        )
    elif re.search(r"'event_type': '(emergency_)?restore'", message):
      logging.info("Found restore event: %s", message)
      if save_step_before_restore is None:
        raise AirflowFailException(
            "Found a restore event before any save event."
        )
      match = re.search(r"'step':\s*(?:np\.int32\()?(\d+)", message)
      if match and int(match.group(1)) >= interrupt_at_step:
        return

      raise AirflowFailException(
          f"Restore event did not contain the expected step. Expect step "
          f"{match.group(1)} to be greater than or equal to step "
          f"{interrupt_at_step}, but the log was: {message}."
      )

  raise AirflowFailException(
      "Failed to validate that restored happened at correct step."
  )
