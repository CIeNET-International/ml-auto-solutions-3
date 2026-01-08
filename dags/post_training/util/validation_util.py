"""Validation utilities for post training DAGs.

This module provides validation functions specific to post training
workflows, reusing generic utilities from the orbax module where applicable.
"""

import datetime
from typing import Optional
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api

# Re-export commonly used validation functions from orbax
from dags.orbax.util.validation_util import (
    generate_timestamp,
    validate_log_exist,
)

__all__ = [
    "generate_posttraining_run_name",
    "generate_timestamp",
    "validate_log_exist",
    "validate_tpu_vm_log_exist",
]


@task
def validate_tpu_vm_log_exist(
    project_id: str,
    zone: str,
    node_id_pattern: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    text_filter: Optional[str] = None,
) -> None:
  """
  Validates that specific logs exist for a TPU VM resource within a time window.

  Args:
      project_id: The Google Cloud project ID.
      zone: The TPU VM zone.
      node_id_pattern: Regex pattern to match node_id.
      start_time: Start time for log retrieval.
      end_time: End time for log retrieval.
      text_filter: Optional filter for log content (textPayload or jsonPayload).
  """
  logging_client = logging_api.Client(project=project_id)

  start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
  end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

  # Construct query specifically for tpu_worker resource type
  conditions = [
      'resource.type="tpu_worker"',
      f'resource.labels.project_id="{project_id}"',
      f'resource.labels.zone="{zone}"',
      f'resource.labels.node_id=~"{node_id_pattern}"',
      "severity>=DEFAULT",
      f'timestamp>="{start_time_str}"',
      f'timestamp<="{end_time_str}"',
  ]

  if text_filter:
    conditions.append(text_filter)

  query = " AND ".join(conditions)
  entries = list(logging_client.list_entries(filter_=query, page_size=1))

  if not entries:
    # Check if ANY logs exist for this TPU to provide a better error message
    if text_filter:
      base_query = " AND ".join(conditions[:-1])
      if list(logging_client.list_entries(filter_=base_query, page_size=1)):
        raise AirflowFailException(
            f"Logs found for TPU {node_id_pattern}, but text_filter {text_filter} "
            f"did not match. Query: {query}"
        )

    raise AirflowFailException(
        f"No logs found for TPU {node_id_pattern} in {zone} between "
        f"{start_time_str} and {end_time_str}. Query: {query}"
    )


@task
def generate_posttraining_run_name(
    short_id: str,
    checkpointing_type: str,
    slice_number: int,
    mode: str,
) -> str:
  """
  Generates a short run name for a post-training run.

  Args:
      short_id: A short identifier for the specific model or experiment.
      checkpointing_type: The name of the checkpointing strategy (e.g., 'grpo').
      slice_number: The number of TPU slices used.
      mode: The setup mode (e.g., 'nightly').

  Returns:
      A short string formatted as
      '{short_id}-{checkpointing_type}-{mode}-{slice_number}'.
  """
  run_name = f"{short_id}-{checkpointing_type}-{mode}-{slice_number}"
  return run_name
