"""Utility functions for querying Google Cloud Monitoring data."""
import logging as log
from typing import List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
)

from google.cloud import monitoring_v3, logging
from google.cloud.logging_v2 import types as logging_types
from google.cloud.monitoring_v3 import types as monitoring_types
from google.api_core.exceptions import ResourceExhausted
from google.api.error_reason_pb2 import ErrorReason

from dags.tpu_observability.utils.time_util import TimeUtil


LOG_READ_QUOTA_EXCEED_ERROR = ErrorReason.Name(ErrorReason.RATE_LIMIT_EXCEEDED)


def is_quota_exceeded(exception: BaseException) -> bool:
    """Checks if the exception message contains the specific quota exceeded identifier."""
    return LOG_READ_QUOTA_EXCEED_ERROR in str(exception)


# Composite Condition: Only retry if the exception is ResourceExhausted AND contains the specific error message.
RETRY_CONDITION = (
    retry_if_exception_type(ResourceExhausted)
    & retry_if_exception(is_quota_exceeded)
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=60, min=60, max=600),
    retry=RETRY_CONDITION,
    before_sleep=lambda retry_state: print(
        f"--- QUOTA HIT --- Retrying attempt {retry_state.attempt_number} in {retry_state.idle_for:.2f} seconds..."
    ),
)
def query_time_series(
    project_id: str,
    filter_str: str,
    start_time: TimeUtil,
    end_time: TimeUtil,
    aggregation: Optional[monitoring_types.Aggregation] = None,
    view: monitoring_types.ListTimeSeriesRequest.TimeSeriesView = monitoring_types.ListTimeSeriesRequest.TimeSeriesView.FULL,
    page_size: Optional[int] = None,
    log_enable: bool = False,
) -> List[monitoring_types.TimeSeries]:
  """A utility that queries metrics (time series data) from Google Cloud Monitoring API.

  This function provides a flexible interface to the list_time_series API,
  with robust error handling and convenient parameter types.

  Args:
    project_id: The Google Cloud project ID.
    filter_str: A Cloud Monitoring filter string that specifies which time
      series should be returned.
    start_time: The start of the time interval.
    end_time: The end of the time interval.
    aggregation: An Aggregation object that specifies how to align and combine
      time series. Defaults to None (raw data).
    view: The level of detail to return. Can be the TimeSeriesView enum (e.g.,
      TimeSeriesView.FULL) or a string ("FULL", "HEADERS"). Defaults to FULL.
    page_size: The maximum number of results to return per page.
    log_enable: Whether to enable logging. Defaults to False.

  Returns:
    A list of TimeSeries objects matching the query.

  Raises:
    ValueError: If the time format or view string is invalid.
    google.api_core.exceptions.GoogleAPICallError: If the API call fails.
  """
  if log_enable:
    log.info("Querying monitoring data for project '%s'", project_id)
    log.info("Filter: %s", filter_str)

  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{project_id}",
      filter=filter_str,
      interval=monitoring_types.TimeInterval(
          start_time=start_time.to_timestamp_pb2(),
          end_time=end_time.to_timestamp_pb2(),
      ),
      view=view,
  )

  if aggregation:
    request.aggregation = aggregation
  if page_size:
    request.page_size = page_size

  client = monitoring_v3.MetricServiceClient()
  results = client.list_time_series(request)

  return list(results)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=60, min=60, max=600),
    retry=RETRY_CONDITION,
    before_sleep=lambda retry_state: print(
        f"--- QUOTA HIT --- Retrying attempt {retry_state.attempt_number} in {retry_state.idle_for:.2f} seconds..."
    ),
)
def query_log_entries(
    project_id: str,
    filter_str: str,
    start_time: TimeUtil,
    end_time: TimeUtil,
    order_by: Optional[str] = logging.DESCENDING,
    max_results: Optional[int] = None,
    page_size: Optional[int] = None,
    log_enable: bool = False,
) -> List[logging_types.LogEntry]:
  """Queries log entries from Google Cloud Logging API.

  Args:
    project_id: The Google Cloud project ID.
    filter_str: A Cloud logging filter string that specifies which log entries
      should be returned.
    start_time: The start of the time interval.
    end_time: The end of the time interval.
    order_by: Optional. How to order the results (e.g., "timestamp desc").
      Defaults to descending timestamp.
    max_results: Optional. The maximum number of results to return overall.
    page_size: The maximum number of results to return per page.
    log_enable: Whether to enable logging. Defaults to False.

  Returns:
    A list of LogEntry objects matching the query.

  Raises:
    ValueError: If the time format is invalid.
    google.api_core.exceptions.GoogleAPICallError: If the API call fails.
  """
  if log_enable:
    log.info("Querying logging data for project '%s'", project_id)
    log.info("Filter: %s", filter_str)

  logging_api_client = logging.Client(project=project_id)

  time_range_str = (
      f'timestamp>="{start_time.to_iso_string()}" AND'
      f' timestamp<="{end_time.to_iso_string()}"'
  )

  log_entries = logging_api_client.list_entries(
      filter_=f"({time_range_str}) AND ({filter_str})",
      order_by=order_by,
      max_results=max_results,
      page_size=page_size,
  )

  return list(log_entries)
