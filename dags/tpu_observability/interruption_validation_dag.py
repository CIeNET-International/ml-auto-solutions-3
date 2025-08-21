"""This script validates the consistency of interruption events between metrics and logs."""

import dataclasses
import datetime
import enum
import re
from typing import List

from airflow import models
from airflow.decorators import task
from airflow.operators.python import get_current_context
from dags.common.vm_resource import Project
from dags.map_reproducibility.utils.constants import Schedule
from dags.multipod.configs.common import Platform
from google.cloud import logging
from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2
from proto import datetime_helpers


_UNKNOWN_RESOURCE_NAME = 'Unknown'


@dataclasses.dataclass
class TimeRange:
  """Class containing proper time range for the validation."""

  start: int
  end: int


class InterruptionReason(str, enum.Enum):
  """Enum class for interruption reasons."""

  DEFRAGMENTATION = 'Defragmentation'
  EVICTION = 'Eviction'
  HOST_ERROR = 'HostError'
  MIGRATE_ON_HWSW_MAINTENANCE = 'Migrate on HW/SW Maintenance'
  HWSW_MAINTENANCE = 'HW/SW Maintenance'
  BARE_METAL_PREEMPTION = 'Bare Metal Preemption'
  OTHER = 'Other'

  def metric_label(self) -> str:
    """Returns the metric label for the interruption reason."""

    return self.value

  def log_filter(self) -> str:
    """Returns the log filter for the interruption reason."""

    filters = []
    match self:
      case InterruptionReason.DEFRAGMENTATION | InterruptionReason.EVICTION:
        filters = ['compute.instances.preempted']
      case InterruptionReason.HOST_ERROR:
        filters = ['compute.instances.hostError']
      case InterruptionReason.MIGRATE_ON_HWSW_MAINTENANCE:
        filters = ['compute.instances.migrateOnHostMaintenance']
      case InterruptionReason.HWSW_MAINTENANCE:
        filters = ['compute.instances.terminateOnHostMaintenance']
      case InterruptionReason.BARE_METAL_PREEMPTION:
        filters = ['compute.instances.baremetalCaretakerPreempted']
      case InterruptionReason.OTHER:
        filters = [
            'compute.instances.guestTerminate',
            'compute.instances.instanceManagerHaltForRestart',
            'compute.instances.stoppedDueToPdDoubleServe',
            'compute.instances.kmsKeyError',
            'compute.instances.shredmillKeyError',
            'compute.instances.invalidVmImage',
            'compute.instances.scratchDiskCreationFailed',
            'compute.instances.localSsdInitializationError',
            'compute.instances.localSsdInitializationKeyError',
            'compute.instances.localSsdVerifyTarError',
            'compute.instances.localSsdRecoveryAttempting',
            'compute.instances.localSsdRecoveryTimeoutError',
            'compute.instances.localSsdRecoveryFailedError',
        ]
      case _:
        raise ValueError(f'Unmapped interruption reason: {self}')
    return ' OR '.join(
        f'protoPayload.methodName="{filter}"' for filter in filters
    )


@dataclasses.dataclass
class Configs:
  """Validation configuration.

  Attributes:
      project_id: The ID of the GCP project.
      max_time_diff_sec: The maximum allowed time difference between a metric
        and its corresponding log event, in seconds. The corresponding log event
        will be considered valid if it is within `max_time_diff_sec` seconds
        before or after the metric event.
      max_log_results: The maximum number of log results to fetch in a single
        query. This is to avoid fetching too many logs.
      platform: The platform (GCE or GKE) where the validation is performed.
      interruption_reason: The specific interruption reason to validate.
      max_start_time_rewind_sec: The maximum number of seconds the validation
        start time can be rewound to find a proper start time, relative to the
        initial start time.
  """

  project_id: str
  max_time_diff_sec: int
  max_log_results: int
  platform: Platform
  interruption_reason: InterruptionReason
  max_start_time_rewind_sec: int


@dataclasses.dataclass
class EventRecord:
  """Represents lists of metric points and log events for a single resource.

  Attributes:
      resource_name: The name of the resource (e.g., node or instance).
      metric_points_timestamps: A list of timestamps for metric points related
        to the resource.
      log_events_timestamps: A list of timestamps for log events related to
        the resource.
  """

  resource_name: str
  metric_points_timestamps: List[int] = dataclasses.field(default_factory=list)
  log_events_timestamps: List[int] = dataclasses.field(default_factory=list)


def fetch_metric_timeseries_by_api(
    configs: Configs,
    start_time: int,
    end_time: int,
) -> List[EventRecord]:
  """Queries the monitoring API for a given validation_conf and time range to retrieve the timeseries data for each resource.

  This function will be called by the airflow task or decide_time_window().
  If start_time and end_time are not provided, it means it's called by the
  airflow task and we use the proper time range from the context.

  Args:
      configs: The configuration object containing the parameters for the
        validation.
      start_time: The start of the time interval when it's not an airflow task.
      end_time: The end of the time interval when it's not an airflow task.
      **context: The airflow context.

  Returns:
      A List of EventRecord objects. Each eventRecord must contain the metric
      points timestamps for the resource name.
  """
  project_id = configs.project_id
  interruption_reason = configs.interruption_reason.metric_label()
  platform = configs.platform

  match platform:
    case Platform.GCE:
      metric_type = 'tpu.googleapis.com/instance/interruption_count'
      resource_type = 'tpu.googleapis.com/GceTpuWorker'
    case Platform.GKE:
      metric_type = 'kubernetes.io/node/interruption_count'
      resource_type = 'k8s_node'
    case _:
      raise ValueError(f'Unsupported platform: {platform.value}')

  metric_filter = (
      f'resource.labels.project_id = "{project_id}" '
      f'metric.type = "{metric_type}" '
      f'resource.type = "{resource_type}" '
      f'metric.labels.interruption_reason = "{interruption_reason}" '
  )

  project_name = f'projects/{project_id}'
  # key: resource_name, value: EventRecord
  events_records: dict[str, EventRecord] = {}

  start_timestamp = timestamp_pb2.Timestamp()
  start_timestamp.FromSeconds(start_time)
  end_timestamp = timestamp_pb2.Timestamp()
  end_timestamp.FromSeconds(end_time)

  interval = monitoring_v3.TimeInterval(
      start_time=start_timestamp, end_time=end_timestamp
  )

  request = monitoring_v3.ListTimeSeriesRequest(
      name=project_name,
      filter=metric_filter,
      interval=interval,
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  monitoring_api_client = monitoring_v3.MetricServiceClient()
  # Here should raise the exception from the API. We don't catch it, just let
  # it raise to Airflow.
  response = monitoring_api_client.list_time_series(request=request)

  for time_series in response:
    match platform:
      case Platform.GKE:
        resource_key = 'node_name'
        resource_name = time_series.resource.labels.get(
            resource_key, _UNKNOWN_RESOURCE_NAME
        )
      case Platform.GCE:
        resource_key = 'instance_name'
        resource_name = time_series.metric.labels.get(
            resource_key, _UNKNOWN_RESOURCE_NAME
        )
      case _:
        raise RuntimeError(
            f"Unsupported platform '{platform.value}'. "
            'Please check the scenario configuration.'
        )

    # Ensure we actually got a name before proceeding
    if resource_name == _UNKNOWN_RESOURCE_NAME:
      raise RuntimeError(
          'Could not determine node/instance name for time series. '
          f'Failed to extract name for resource type "{platform.value}". '
          f'Time series data: {time_series}'
      )

    for point in time_series.points:
      end_time_obj: datetime_helpers.DatetimeWithNanoseconds = (
          point.interval.end_time
      )
      match monitoring_v3.TypedValue.pb(point.value).WhichOneof('value'):
        case 'int64_value':
          event_count = point.value.int64_value
        case 'double_value':
          event_count = int(point.value.double_value)
        case _:
          raise RuntimeError(
              'Unexpected TypedValue:'
              f" {monitoring_v3.TypedValue.pb(point.value).WhichOneof('value')}."
              f' Full point data: {point}'
          )

      # Process the event only if the event_count is greater than 0, meaning
      # that an interruption occurred at this time.
      if event_count > 0:
        if resource_name not in events_records:
          events_records[resource_name] = EventRecord(
              resource_name=resource_name,
          )
        # The event_count represents a count of interruption events occurring
        # at the same time.
        # We need to add each event separately to the list of metric points.
        events_records[resource_name].metric_points_timestamps.extend(
            [int(end_time_obj.timestamp()) * event_count]
        )

  if not events_records:
    raise RuntimeError('No metric events found in the specified time range.')

  return list(events_records.values())


@task
def fetch_log_entries_by_api(
    proper_time_range: TimeRange,
    event_records: List[EventRecord],
    configs: Configs,
) -> List[EventRecord]:
  """This function queries the Logging API for a given validation_conf and time range to fetch the log entries.

  Args:
      configs: The configuration object containing the parameters for the
        validation.

  Returns:
      A list of EventRecord objects. Each EventRecord must contain the log
      events timestamps for the resource name.
  """
  start_time = datetime.datetime.fromtimestamp(
      proper_time_range.start, tz=datetime.timezone.utc
  )
  end_time = datetime.datetime.fromtimestamp(
      proper_time_range.end, tz=datetime.timezone.utc
  )

  project_id = configs.project_id
  log_filter_query = configs.interruption_reason.log_filter()
  max_results = configs.max_log_results

  logging_api_client = logging.Client(project=project_id)

  start_time_str = start_time.isoformat().replace('+00:00', 'Z')
  end_time_str = end_time.isoformat().replace('+00:00', 'Z')
  time_range_str = (
      f'timestamp>="{start_time_str}" AND timestamp<="{end_time_str}"'
  )

  resource_filter_query = None
  event_records = {record.resource_name: record for record in event_records}
  if event_records:
    resource_filter_query = ' OR '.join(
        f'protoPayload.resourceName=~"^projects/[\\w-]+/zones/[\\w-]+/instances/{name}$"'
        for name in event_records
    )

  if resource_filter_query:
    log_filter_query = f'{log_filter_query} AND ({resource_filter_query})'

  log_entries = logging_api_client.list_entries(
      filter_=f'({time_range_str}) AND ({log_filter_query})',
      order_by=logging.DESCENDING,
      max_results=max_results,
  )

  if not log_entries:
    raise RuntimeError('No log entries found in the specified time range.')

  entry_count = 0
  for entry in log_entries:
    entry_count += 1
    # The 'resourceName' in the log entry payload typically looks like:
    # "projects/{project_id}/zones/{zone}/instances/{node_name}"
    regex_pattern = r'^projects/[\w-]+/zones/[\w-]+/instances/([\w-]+)$'
    resource_name = entry.payload.get('resourceName', '')
    match = re.match(regex_pattern, resource_name)
    if match:
      log_node_name = match.group(1)
      aware_timestamp = entry.timestamp.replace(tzinfo=datetime.timezone.utc)

      if log_node_name not in event_records:
        event_records[log_node_name] = EventRecord(
            resource_name=log_node_name,
        )
      event_records[log_node_name].log_events_timestamps.append(
          int(aware_timestamp.timestamp())
      )

  if entry_count == max_results:
    raise RuntimeError(
        f'Log entries limit reached ({max_results} entries). '
        'This might indicate we are missing data. '
        'Consider increasing max_results or narrowing the time range.'
    )

  return list(event_records.values())


@task
def determinate_time_range(
    configs: Configs,
) -> TimeRange:
  """Determines the time range for the validation.

  The function aims to find a time window [start, end] such that there are no
  metric events within `allowed_gap` of either the start or end boundaries.
  This ensures that all metric events within the determined time range can be
  properly correlated with log events, considering a potential time difference
  between metric and log timestamps.

  Args:
      configs: The configuration object containing the parameters for the
        validation.

  Returns:
      A TimeRange object representing the determined start and end times.
  """
  # We assume the max shift of the log is 30 minutes. (call it max_shift)
  # The allowed_gap should be 2 * 30 minutes. Here's why we need this buffer:
  #
  # A 1x max_shift is used to capture the last relevant metric event. This
  # ensures we can correlate it with its corresponding log event, even if the
  # log event occurs up to max_shift later, allowing us to find event pairs at
  # the very edge of the query window.
  #
  # The second 1x max_shift is crucial for preventing a different issue: it
  # explicitly excludes the next metric event. This is to avoid an incorrect
  # correlation, as the log for that next metric event might fall within our
  # query window, leading to misleading associations.
  allowed_gap = int(datetime.timedelta(minutes=30).total_seconds()) * 2

  # This test is scheduled to run every day,
  # so we validate the interruption within a day (at least)
  min_time_window = int(datetime.timedelta(days=1).total_seconds())
  time_window_step = min_time_window

  context = get_current_context()
  ti = context['ti']
  task_start_time = int(ti.start_date.timestamp())

  right_bound = task_start_time
  left_bound = right_bound - 2 * time_window_step

  found_right = False
  while not found_right:
    # Fail the test to indicate that manual inspection is required,
    # as the data has been too dense for a significant duration.
    if abs(task_start_time - right_bound) > int(
        datetime.timedelta(days=3).total_seconds()
    ):
      raise RuntimeError('the data has been too dense in past few days')

    # Call API to obtain metrics.
    metric_records = fetch_metric_timeseries_by_api(
        configs,
        left_bound,
        right_bound,
    )

    total_metric_timestamps = []
    for record in metric_records:
      total_metric_timestamps.extend(record.metric_points_timestamps)
    # Use the timestamp (in second) to sort the data.
    total_metric_timestamps.sort()

    # Find the right-most data that has a sufficient gap from the
    # right boundary.
    for r in reversed(total_metric_timestamps):
      if abs(r - right_bound) > allowed_gap:
        found_right = True
        break
      right_bound = r

    if not found_right:
      left_bound -= time_window_step
      continue  # Back to while-loop.
    else:
      # Right bound is determined.
      # We need to add an additional allowed_gap / 2 to the right bound,
      # to ensure that the log of the next metric event is not included in the
      # validation.
      right_bound = int(right_bound - allowed_gap / 2 - 1)

  found_left = False
  iteration_count = 0
  while not found_left:
    iteration_count += 1
    # At this point, the right bound has been determined.
    # However, since the left bound keeps shifting and the time window keeps
    # expanding, validating the interruption count over such a long duration
    # might not make sense.
    if right_bound-left_bound > int(datetime.timedelta(days=5).total_seconds()):
      raise RuntimeError('the time window has been too long')

    # Call API to obtain metrics.
    metric_records = fetch_metric_timeseries_by_api(
        configs,
        left_bound,
        right_bound,
    )

    total_metric_timestamps = []
    for record in metric_records:
      total_metric_timestamps.extend(record.metric_points_timestamps)
    # Use the timestamp (in second) to sort the data.
    total_metric_timestamps.sort()

    # Find the left-most data that has a sufficient gap from the
    # left boundary.
    for r in total_metric_timestamps:
      if abs(r - left_bound) > allowed_gap:
        found_left = True
        break
      left_bound = r

    if not found_left or (right_bound - left_bound) < min_time_window:
      # If a proper left bound is not found after iterating through all records,
      # the current left bound is equal to the rightmost record.
      # Expand the search window. The initial window size was 2 * time_window_step.
      # We extend the search by an additional time_window_step.
      left_bound -= (2 + iteration_count) * time_window_step
      continue  # Back to while-loop.
    else:
      # Left bound is determined.
      # We need to add an additional allowed_gap / 2 to the left bound,
      # to ensure that the log of the previous metric event is not included in
      # the validation.
      left_bound = int(left_bound + allowed_gap / 2 + 1)

  return TimeRange(start=left_bound, end=right_bound)


@task
def check_event_count_match(
    event_records: List[EventRecord],
):
  """Checks if the number of metric events matches the number of log events for each resource.

  Args:
      event_records: A list of EventRecord objects containing metric and log
        timestamps.
  """

  count_diff_records = [
      event_record
      for event_record in event_records
      if len(event_record.metric_points_timestamps)
      != len(event_record.log_events_timestamps)
  ]
  if count_diff_records:
    raise RuntimeError(
        'Event count mismatch for this following event records:'
        f' {count_diff_records}'
    )


def create_interruption_dag(
    dag_id: str,
    platform: Platform,
    interruption_reason: InterruptionReason,
) -> models.DAG:
  """Generates a single interruption event validation DAG.
  """

  with models.DAG(
      dag_id=dag_id,
      start_date=datetime.datetime(2025, 7, 20),
      schedule=Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
      catchup=False,
      tags=['gke', 'gce', 'tpu-observability', 'interruption_validation'],
      description=(
          'This DAG validates the interruption event metrics and logs for GKE and GCE'
      ),
      doc_md="""
      ### Interruption Event Validation DAG
      This DAG automatically validates the consistency of interruption events between metrics and logs for both GKE and GCE environments.
      """,
  ) as dag:
    configs = Configs(
        project_id=models.Variable.get(
            'INTERRUPTION_PROJECT_ID',
            default_var=Project.TPU_PROD_ENV_ONE_VM.value,
        ),
        max_time_diff_sec=int(
            models.Variable.get('INTERRUPTION_MAX_TIME_DIFF_SEC', default_var=150)
        ),
        max_log_results=int(
            models.Variable.get('INTERRUPTION_MAX_LOG_RESULTS', default_var=1000)
        ),
        platform=platform,
        interruption_reason=interruption_reason,
        max_start_time_rewind_sec=int(
            models.Variable.get(
                'INTERRUPTION_MAX_START_TIME_REWIND_SECONDS', default_var=3600
            )
        ),
    )

    @task
    def fetch_metric_timeseries_by_api_task(
        proper_time_range: TimeRange,
        configs: Configs,
    ) -> List[EventRecord]:
      return fetch_metric_timeseries_by_api(
          configs,
          proper_time_range.start,
          proper_time_range.end,
      )

    proper_time_range = determinate_time_range(configs)
    records_updated_with_metric_data = fetch_metric_timeseries_by_api_task(
        proper_time_range,
        configs,
    )
    records_updated_with_log_data = fetch_log_entries_by_api(
        proper_time_range,
        records_updated_with_metric_data,
        configs,
    )
    check_event_count = check_event_count_match(records_updated_with_log_data)

    (
        proper_time_range
        >> records_updated_with_metric_data
        >> records_updated_with_log_data
        >> check_event_count
    )

    return dag


dag_id_prefix = 'interruption_validation'
for platform in [Platform.GCE, Platform.GKE]:
  for reason in InterruptionReason:
    reason_value = reason.value.replace(' ', '_').replace('/', '').lower()
    dag_id = f'{dag_id_prefix}_{platform.value}_{reason_value}'
    _ = create_interruption_dag(
        dag_id,
        platform,
        reason
    )
