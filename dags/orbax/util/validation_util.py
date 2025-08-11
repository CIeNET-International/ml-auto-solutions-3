"""Utilities to get workloads logs and some utils."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass
from absl import logging
from typing import Tuple
import re
from abc import ABC, abstractmethod


from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api
from xlml.utils.gke import zone_to_region

import dags.orbax.util.gcs_util as gcs_util

class BaseValidation(ABC):
  @abstractmethod
  def list_log_entries(self) -> list:
    """
    Abstract method that must be implemented by all subclasses.
    """
    pass


@dataclass
class ValidationReplicator(BaseValidation):
  """
  A dataclass to hold information of a Validation Object. Will be used
  in a log explorer query.
  """

  project_id: str
  location: str
  cluster_name: str
  namespace: str
  pod_pattern: str
  container_name: Optional[str] = None
  text_filter: Optional[str] = None
  start_time: Optional[datetime] = None
  end_time: Optional[datetime] = None
  local_step_list: Optional[list] = None
  gcs_step_list: Optional[list] = None
  gcs_bucket_run_name: Optional[str] = None

  def list_log_entries(
      self,
  ) -> list:
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

    logging_client = logging_api.Client(project=self.project_id)

    # Set the time window for log retrieval:
    # default to last 12 hours if not provided
    if self.end_time is None:
      end_time = datetime.now(timezone.utc)
    if self.start_time is None:
      start_time = end_time - timedelta(hours=12)

    # Format times as RFC3339 UTC "Zulu" format required by the Logging API
    start_time_str = self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = self.end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Construct the log filter
    log_filter = (
        f'resource.labels.project_id="{self.project_id}" '
        f'resource.labels.location="{self.location}" '
        f'resource.labels.cluster_name="{self.cluster_name}" '
        f'resource.labels.namespace_name="{self.namespace}" '
        f'resource.labels.pod_name:"{self.pod_pattern}" '
        "severity>=DEFAULT "
        f'timestamp>="{start_time_str}" '
        f'timestamp<="{end_time_str}"'
    )

    if self.container_name:
      log_filter += f' resource.labels.container_name="{self.container_name}"'

    if self.text_filter:
      log_filter += f' SEARCH("{self.text_filter}")'

    # Retrieve log entries matching the filter
    logging.info(f"Log filter constructed: {log_filter}")
    entries = logging_client.list_entries(filter_=log_filter)

    return entries

  def get_replicated_steps(self) -> None:
    # Get the entries for the backup steps in the bucket. To later compare the
    # latest stored step in bucket with the latest recorded step in training pod.
    entries = self.list_log_entries()
    gcs_save_step_list = []
    gcs_save_step_list_bucket = []
    for entry in entries:
      if entry.payload is not None:
        payload_str = str(entry.payload)
        for line in payload_str.split("\n"):

          # Extract the gcs bucket path from replicator logs
          # eg. gcs=2025-08-10_12-09 and step=60.
          gcs_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2,}"
          step_pattern = r"step (\d+)"
          match_gcs = re.search(gcs_pattern, line)
          match_step = re.search(step_pattern, line)
          validate_check_gcs = False

          logging.info(f"Bucket run name: {self.gcs_bucket_run_name}")
          if match_gcs and match_step and self.gcs_bucket_run_name:
            gcs_checkpoint_path = match_gcs.group(0)
            step = match_step.group(1)
            logging.info(f"get gcs path from: {gcs_checkpoint_path}")
            bucket_files = gcs_util.get_gcs_checkpoint(
                f"{self.gcs_bucket_run_name}/{gcs_checkpoint_path}/"
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
              raise AirflowFailException(f"No Backup event happened. No \
                                        replication event happened")
    self.gcs_step_list = gcs_save_step_list_bucket
    self.local_step_list = gcs_save_step_list

@dataclass
class ValidationPod(BaseValidation):
  """
  A dataclass to hold information of a Validation Object. Will be used
  in a log explorer query.
  """

  project_id: str
  location: str
  cluster_name: str
  namespace: str
  pod_pattern: str
  container_name: Optional[str] = None
  text_filter: Optional[str] = None
  start_time: Optional[datetime] = None
  end_time: Optional[datetime] = None
  vali_step_list: Optional[list] = None
  local_step_list: Optional[list] = None
  gcs_step_list: Optional[list] = None
  gcs_bucket_run_name: Optional[str] = None

  def __init__(
    self,
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str,
    pod_pattern: str,
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    vali_step_list: Optional[list] = None,
    gcs_bucket_path: Optional[str] = None,
  ):
    """
    doc
    """
    self.project_id = project_id
    self.location= location
    self.cluster_name = cluster_name
    self.namespace = namespace
    self.pod_pattern = pod_pattern
    self.container_name = container_name
    self.text_filter = text_filter
    self.start_time = start_time
    self.end_time = end_time
    self.vali_step_list = vali_step_list
    self.gcs_bucket_run_name = gcs_bucket_path

  def list_log_entries(
      self
  ) -> list:
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
      text_filter:gg Optional comma-separated string to
        filter log entries by textPayload content
      start_time: Optional start time for log retrieval
        (defaults to 12 hours ago)
      end_time: Optional end time for log retrieval (defaults to now)
    Returns:
      bool: Number of log entries found
    """

    logging_client = logging_api.Client(project=self.project_id)

    start_time = self.start_time
    end_time = self.end_time

    # Set the time window for log retrieval:
    # default to last 12 hours if not provided
    if end_time is None:
      end_time = datetime.now(timezone.utc)

    if start_time  is None:
      start_time = end_time-timedelta(hours=12)

    # Format times as RFC3339 UTC "Zulu" format required by the Logging API
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Construct the log filter
    log_filter = (
        f'resource.labels.project_id="{self.project_id}" '
        f'resource.labels.location="{self.location}" '
        f'resource.labels.cluster_name="{self.cluster_name}" '
        f'resource.labels.namespace_name="{self.namespace}" '
        f'resource.labels.pod_name:"{self.pod_pattern}" '
        "severity>=DEFAULT "
        f'timestamp>="{start_time_str}" '
        f'timestamp<="{end_time_str}"'
    )

    if self.container_name:
      log_filter += f' resource.labels.container_name="{self.container_name}"'

    if self.text_filter:
      log_filter += f' SEARCH("{self.text_filter}")'

    # Retrieve log entries matching the filter
    logging.info(f"Log filter constructed: {log_filter}")
    entries = logging_client.list_entries(filter_=log_filter)

    return entries

  def get_backup_bucket_path(self)->str | None:
    # We need to match the original run_name to the run_name in the gcs bucket.
    # Dues to datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") behaviour
    # is better to get the run_name from already executed pod (post mortem)
    gcs_bucket_run_name = None
    entries = self.list_log_entries()
    for entry in entries:
      if entry.payload is not None:
        payload_str = str(entry.payload)
        for line in payload_str.split("\n"):
          if "Config param checkpoint_dir" in line:
            gcs_run_name_pattern = re.search(r'Config param checkpoint_dir: (.*?)\/checkpoints\/', line)
            if gcs_run_name_pattern:
              full_gcs_path = gcs_run_name_pattern.group(1)
              gcs_bucket_run_name = "gs://" + full_gcs_path + "/"
              break
            raise AirflowFailException(
                f"There is no Checkpoint Directory Configured, emptu checkpoint_dir flag"
            )

    return gcs_bucket_run_name

def factory_create_validation_config(
  test_config,
  namespace: str,
  text_filter:str,
  gcs_bucket_run_name: Optional[str] = None,
) -> BaseValidation:
  """Creates a specific validation object based on the namespace.

  This factory method determines which concrete `BaseValidation` subclass
  to instantiate (`ValidationPod` or `ValidationReplicator`) based on the
  `namespace` argument. It dynamically constructs the configuration parameters
  for the chosen class and returns the new object.

  Args:
    self: The class instance (typical for a method).
    test_config: A configuration object containing cluster and test details.
    start_time: The start of the time range for log queries.
    end_time: The end of the time range for log queries.
    namespace: The Kubernetes namespace, which dictates the type of
      validation object to create.
    text_filter: A string used to filter log entries.
    gcs_bucket_run_name: The name of the GCS bucket run, used for
      managed checkpointing.

  Returns:
    A concrete subclass of `BaseValidation` (either `ValidationPod` or
    `ValidationReplicator`).

  Raises:
    AirflowFailException: If an invalid or unsupported namespace is provided.
  """

  if namespace == "default":
    vali_step = test_config.step - 1
    # Use a more readable generator expression to create the list
    vali_step_list = list(range(0, vali_step, test_config.local_checkpoint_step))
    vali_step_list.append(vali_step)
    params = {
        "project_id": test_config.cluster.project,
        "location": zone_to_region(test_config.cluster.zone),
        "cluster_name": test_config.cluster.name,
        "namespace": "default",
        "pod_pattern":"*",
        "text_filter": text_filter,
        "vali_step_list":vali_step_list,
    }
    return ValidationPod(**params)

  elif namespace == "gke-managed-checkpointing":
    params = {
        "project_id": test_config.cluster.project,
        "location": zone_to_region(test_config.cluster.zone),
        "cluster_name": test_config.cluster.name,
        "namespace": "gke-managed-checkpointing",
        "pod_pattern":"multitier-driver",
        "container_name": "replication-worker",
        "text_filter": text_filter,
        "gcs_bucket_run_name": gcs_bucket_run_name
    }
    return ValidationReplicator(**params)
  else:
    raise AirflowFailException(f"Invalid namespace: {namespace}")


@task
def generate_timestamp():
  return datetime.now(timezone.utc)


@task
def validate_log_with_step(
  validation_info: ValidationPod,
  start_time: datetime,
  end_time: datetime,
) -> bool:
  """
  Validates workload logs by checking for specific training steps.

  This function queries logs from a specified GKE cluster and namespace,
  searching for log entries that indicate a checkpoint save event. It then
  compares the number of successfully found steps against a predefined list
  of expected steps to ensure the workload progressed as intended. A mismatch
  in the number of steps will cause the validation to fail, which can
  occur if, for example, a restore operation causes the step count to
  restart from zero.

  Args:
    val_info: A ValidationPod instance containing all necessary
      parameters for the log query, such as project ID, cluster details,
      and the list of validation steps to check.

  Returns:
    bool: True if validation is successful.
  """

  validation_info.start_time = start_time
  validation_info.end_time = end_time
  entries = validation_info.list_log_entries()
  if validation_info.vali_step_list is None:
    return False
  new_step_list = []
  for entry in entries:
    if not entry.payload:
      continue
    payload_str = str(entry.payload)
    for line in payload_str.split("\n"):
      for step in validation_info.vali_step_list:
        vali_str = "directory=/local/" + str(step)
        if vali_str in line and step not in new_step_list:
          logging.info(f"├─ Timestamp: {entry.timestamp}")
          logging.info("└─ Payload:")
          logging.info(f"   {line}")
          new_step_list.append(step)
  if len(validation_info.vali_step_list) == len(new_step_list):
    logging.info("Validate success")
    return True
  else:
    raise AirflowFailException(
        f"{len(validation_info.vali_step_list)} saves are expected,"
        f"but got {len(new_step_list)}"
    )


@task
def validate_log_with_gcs(
  validation_info: BaseValidation,
  start_time: datetime,
  end_time: datetime,
) -> None:
  """
  Validates GKE workload logs against GCS bucket checkpoints.

  This function retrieves log entries from a specified GKE cluster and
  compares the step numbers found in the logs with the step numbers
  extracted from the filenames of checkpoint files in a GCS bucket.
  It raises an `AirflowFailException` if the latest step in the logs
  does not match the latest step in the GCS bucket.

  Args:
    validation_info: A BaseValidation object, which must be a
      ValidationReplicator instance for this task.

  Returns:
    None
  """

  if isinstance(validation_info, ValidationReplicator):
    validation_info.start_time = start_time
    validation_info.end_time = end_time
    validation_info.get_replicated_steps()

  # Compare last step found in replicator logs and last (only one)
  # step extracted from filename bucket
  if len(validation_info.gcs_bucket_run_name) > 0 and len(validation_info.local_step_list) > 0:
    # Extract s60 from  file name with extension .meta
    pattern_bucket_step = r"s(\d+)"
    raw_str_filename = validation_info.gcs_step_list[-1]
    match = re.search(pattern_bucket_step, raw_str_filename)
    if match is None:
      raise AirflowFailException(
          f"Could not extract step from filename: {raw_str_filename}"
      )
    last_step_bucket = match.group(0)[1:]
    if int(last_step_bucket) == max(validation_info.gcs_step_list):
      logging.info("Validate success")
      return True
  else:
      raise AirflowFailException(
        f"Steps in bucket or replicator logs are empty. "
        f"GCS bucket steps found: {len(validation_info.gcs_step_list)}. "
        f"Replicator log steps found: {len(validation_info.local_step_list)}."
      )
  return max(validation_info.local_step_list), max(validation_info.gcs_step_list)


