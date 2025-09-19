"""Utility functions for querying Google Cloud Monitoring data."""
import datetime
import logging
from typing import List, Optional, Union

from dags.tpu_observability.utils.time_util import TimeUtil, TimeInput

from airflow.exceptions import AirflowException
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types

TimeInput = Union[datetime.datetime, str, int, float]


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

  start_time_obj = TimeUtil.build(start_time, arg_name="start_time")
  end_time_obj = TimeUtil.build(end_time, arg_name="end_time")

  interval = types.TimeInterval(
      start_time=start_time_obj.to_protobuf_timestamp(),
      end_time=end_time_obj.to_protobuf_timestamp(),
  )

  if isinstance(view, str):
    try:
      # Convert string like "FULL" to the enum member TimeSeriesView.FULL
      view_enum = types.ListTimeSeriesRequest.TimeSeriesView[view.upper()]
    except KeyError as exc:
      raise ValueError(
          f"Invalid view string: '{view}'. Must be 'FULL' or 'HEADERS'."
      ) from exc
  else:
    view_enum = view

  request = monitoring_v3.ListTimeSeriesRequest({
      "name": project_name,
      "filter": filter_str,
      "interval": interval,
      "view": view_enum,
  })

  if aggregation:
    request.aggregation = aggregation
  if page_size:
    request.page_size = page_size

  results = client.list_time_series(request)

  return list(results)
