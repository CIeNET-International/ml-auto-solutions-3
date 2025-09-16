"""Utility functions for querying Google Cloud Monitoring data."""
import datetime
import logging
from typing import List, Optional, Union

from airflow.exceptions import AirflowException
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types

# Define a type hint for the flexible time inputs.
TimeInput = Union[datetime.datetime, str, int, float]


def _to_timestamp_seconds(time_input: TimeInput, arg_name: str) -> int:
  """Internal helper to convert a flexible time input into an integer Unix timestamp."""
  if isinstance(time_input, datetime.datetime):
    return int(time_input.timestamp())
  elif isinstance(time_input, (int, float)):
    # Input is already a Unix timestamp, ensure it's an integer.
    return int(time_input)
  elif isinstance(time_input, str):
    try:
      dt_object = datetime.datetime.fromisoformat(
          time_input.replace("Z", "+00:00")
      )
      return int(dt_object.timestamp())
    except ValueError:
      raise AirflowException(
          f"Invalid ISO 8601 format for {arg_name}: '{time_input}'"
      )
  else:
    raise TypeError(
        f"Unsupported type for {arg_name}: {type(time_input)}. "
        "Must be datetime, str (ISO 8601), int, or float (Unix timestamp)."
    )


def query_time_series(
    project_id: str,
    filter_str: str,
    start_time: TimeInput,
    end_time: TimeInput,
    aggregation: Optional[types.Aggregation] = None,
    view: Union[
        str, types.ListTimeSeriesRequest.TimeSeriesView
    ] = types.ListTimeSeriesRequest.TimeSeriesView.FULL,
    page_size: Optional[int] = None,
) -> List[types.TimeSeries]:
  """A reusable function to query time series data from Google Cloud Monitoring.

  This function provides a flexible interface to the list_time_series API,
  with robust error handling and convenient parameter types.

  Args:
      project_id: The Google Cloud project ID.
      filter_str: A Cloud Monitoring filter string that specifies which time
        series should be returned.
        Example: 'metric.type = "your/metric" AND resource.labels.instance_id
          = "123"'
      start_time: The start of the time interval. Can be a datetime object, an
        ISO 8601 string, or a Unix timestamp (int/float).
      end_time: The end of the time interval. Can be a datetime object, an ISO
        8601 string, or a Unix timestamp (int/float).
      aggregation: An Aggregation object that specifies how to align and combine
        time series. Defaults to None (raw data).
      view: The level of detail to return. Can be the TimeSeriesView enum (e.g.,
        TimeSeriesView.FULL) or a string ("FULL", "HEADERS"). Defaults to FULL.
      page_size: The maximum number of results to return per page.

  Returns:
      A list of TimeSeries objects matching the query.

  Raises:
      ValueError: If the time format or view string is invalid.
      google.api_core.exceptions.GoogleAPICallError: If the API call fails.
  """
  logging.info("Querying monitoring data for project '%s'", project_id)
  logging.info("Filter: %s", filter_str)

  client = monitoring_v3.MetricServiceClient()
  project_name = f"projects/{project_id}"

  try:
    start_seconds = _to_timestamp_seconds(start_time, arg_name="start_time")
    end_seconds = _to_timestamp_seconds(end_time, arg_name="end_time")
  except (ValueError, TypeError) as e:
    logging.error("Time conversion error: %s", e)
    raise

  interval = types.TimeInterval(
      start_time={"seconds": start_seconds},
      end_time={"seconds": end_seconds},
  )

  if isinstance(view, str):
    try:
      # Convert string like "FULL" to the enum member TimeSeriesView.FULL
      view_enum = types.ListTimeSeriesRequest.TimeSeriesView[view.upper()]
    except KeyError:
      raise ValueError(
          f"Invalid view string: '{view}'. Must be 'FULL' or 'HEADERS'."
      )
  else:
    view_enum = view

  request_kwargs = {
      "name": project_name,
      "filter": filter_str,
      "interval": interval,
      "view": view_enum,
  }
  if aggregation:
    request_kwargs["aggregation"] = aggregation
  if page_size:
    request_kwargs["page_size"] = page_size

  request = monitoring_v3.ListTimeSeriesRequest(**request_kwargs)

  results = client.list_time_series(request)

  return results
