"""This script validates the consistency of interruption events between metrics and logs."""

import dataclasses
import datetime
import enum
import re
from typing import List

from airflow import models
from airflow.decorators import task
from dags.common.vm_resource import Project
from dags.map_reproducibility.utils.constants import Schedule
from dags.multipod.configs.common import Platform
from google.cloud import logging
from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2
from proto import datetime_helpers
import pytz


_UNKNOWN_RESOURCE_NAME = 'Unknown'


@dataclasses.dataclass
class ProperTimeRange:
  """Class containing proper time range for the validation."""

  proper_start_time: str
  proper_end_time: str

  def __init__(
      self,
      proper_start_time: datetime.datetime | str,
      proper_end_time: datetime.datetime | str,
  ):
    if isinstance(proper_start_time, str):
      self.proper_start_time = proper_start_time
    elif isinstance(proper_start_time, datetime.datetime):
      self.proper_start_time = proper_start_time.astimezone(
          datetime.timezone.utc
      ).isoformat()
    if isinstance(proper_end_time, str):
      self.proper_end_time = proper_end_time
    elif isinstance(proper_end_time, datetime.datetime):
      self.proper_end_time = proper_end_time.astimezone(
          datetime.timezone.utc
      ).isoformat()


class InterruptionReason(str, enum.Enum):
  """Enum class for interruption reasons."""

  DEFRAGMENTATION = 'Defragmentation'
  EVICTION = 'Eviction'
  HOST_ERROR = 'HostError'
  MIGRATE_ON_HWSW_MAINTENANCE = 'Migrate on HW/SW Maintenance'
  HW_SW_MAINTENANCE = 'HW/SW Maintenance'
  BARE_METAL_PREEMPTION = 'Bare Metal Preemption'
  OTHER = 'Other'

  def metric_label(self) -> str:
    """Returns the metric label for the interruption reason."""

    return self.value

  def log_filter(self) -> str:
    """Returns the log filter for the interruption reason."""

    match self:
      case InterruptionReason.DEFRAGMENTATION | InterruptionReason.EVICTION:
        return 'protoPayload.methodName="compute.instances.preempted" '
      case InterruptionReason.HOST_ERROR:
        return 'protoPayload.methodName="compute.instances.hostError" '
      case InterruptionReason.MIGRATE_ON_HWSW_MAINTENANCE:
        return 'protoPayload.methodName="compute.instances.migrateOnHostMaintenance" '
      case InterruptionReason.HW_SW_MAINTENANCE:
        return 'protoPayload.methodName="compute.instances.terminateOnHostMaintenance" '
      case InterruptionReason.BARE_METAL_PREEMPTION:
        return 'protoPayload.methodName="compute.instances.baremetalCaretakerPreempted" '
      case InterruptionReason.OTHER:
        parts = [
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
        return ' OR '.join(f'protoPayload.methodName="{p}"' for p in parts) + ' '
      case _:
        raise ValueError(f'Unmapped interruption reason: {self}')


@dataclasses.dataclass
class Configs:
  """Validation configuration."""

  initial_start_time: datetime.datetime
  initial_end_time: datetime.datetime
  project_id: str
  max_time_diff_sec: int
  max_log_results: int
  metric_aggregation: str
  platform: Platform
  interruption_reason: InterruptionReason
  max_start_time_rewind_sec: int


@dataclasses.dataclass
class EventRecord:
  """Represents lists of metric points and log events for a single resource.

  Attributes:
      resource_name: The name of the resource (e.g., node or instance).
      interruption_reason: The reason for the interruption event.
      log_filter: The log query filter used to fetch log events.
      metric_points_timestamps: A list of timestamps for metric points related
        to the resource.
      log_events_timestamps: A list of timestamps for log events related to
        the resource.
  """
  resource_name: str
  interruption_reason: str = ''
  log_filter: str = ''
  metric_points_timestamps: List[str] = dataclasses.field(default_factory=list)
  log_events_timestamps: List[str] = dataclasses.field(default_factory=list)


def query_metric_data_by_api(
    configs: Configs,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> List[EventRecord]:
  """Queries the monitoring API for a given validation_conf and time range to retrieve the timeseries data for each resource.

  Args:
      configs: The configuration object containing the parameters for the
        validation.
      start_time: The start of the time interval.
      end_time: The end of the time interval.

  Returns:
      A List of EventRecord objects. Each eventRecord must contain the metric
      points timestamps for the resource name.
  """
  project_id = configs.project_id
  interruption_reason = configs.interruption_reason.metric_label()
  platform = configs.platform
  aggregation = configs.metric_aggregation

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
  events_records: dict[str, EventRecord] = {}

  start_timestamp = timestamp_pb2.Timestamp()
  start_timestamp.FromDatetime(start_time)
  end_timestamp = timestamp_pb2.Timestamp()
  end_timestamp.FromDatetime(end_time)

  interval = monitoring_v3.TimeInterval(
      start_time=start_timestamp,
      end_time=end_timestamp
  )

  request = monitoring_v3.ListTimeSeriesRequest(
      name=project_name,
      filter=metric_filter,
      interval=interval,
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  # If aggregation is provided, add it to the request arguments
  if aggregation:
    request.aggregation = aggregation

  monitoring_api_client = monitoring_v3.MetricServiceClient()
  # Here should raise the exception from the API. We don't catch it, just let
  # it raise to Airflow.
  response = monitoring_api_client.list_time_series(request=request)

  for time_series in response:
    match platform:
      case Platform.GKE:
        resource_key = 'node_name'
      case Platform.GCE:
        resource_key = 'instance_name'
      case _:
        print(f"Warning: Unknown platform '{platform.value}'.")
        raise RuntimeError(
            f"Unsupported platform '{platform.value}'. "
            'Please check the scenario configuration.'
        )

    resource_name = time_series.resource.labels.get(
        resource_key, _UNKNOWN_RESOURCE_NAME
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
        aware_timestamp = end_time_obj.replace(tzinfo=datetime.timezone.utc)
        if resource_name not in events_records:
          events_records[resource_name] = EventRecord(
              resource_name=resource_name,
              interruption_reason=interruption_reason,
          )
        # The event_count represents a count of interruption events occurring
        # at the same time.
        # We need to add each event separately to the list of metric points.
        events_records[resource_name].metric_points_timestamps.extend(
            [aware_timestamp.isoformat()*event_count]
        )

  if not events_records:
    print(
        'No metric events found in the specified time range. Validation cannot'
        ' proceed.'
    )
    raise RuntimeError('No metric events found in the specified time range.')

  return list(events_records.values())


@task
def decide_time_window(
    configs: Configs,
    **context,
) -> List[EventRecord]:
  """Adjusts the proper time range for the validation and fetches the metric data with the proper time range.

  It will adjust the start_time and end_time to ensure there is idle_time_buffer
  before the earliest metric record and after the latest metric record, by
  querying the metric data with the monitoring API. This adjustment is crucial
  to capture all relevant log events, as logs might appear slightly earlier or
  later than the corresponding metric points due to delays (max_time_diff_sec)
  in their generation or transmission.
  The function iteratively refines the time range until no further adjustment is
  needed, or a maximum rewind limit is reached.
  In edge cases, the retrieved logs might not belong to the current time range,
  so it is important to ensure that the time range is wide enough to capture all
  the relevant logs.

  If the difference between the initial start_time and the adjusted start_time
  is more than max_start_time_rewind_seconds, it will raise a RuntimeError.

  This function performs the following steps:
  1.  Queries the monitoring API to fetch interruption metric timestamps within
      the current time range.
  2.  Determines the earliest and latest metric record timestamps.
  3.  Calculates a new start time by subtracting `idle_time_buffer` from the
      earliest metric record timestamp.
  4.  Calculates a new end time by adding `idle_time_buffer` to the latest
      metric record timestamp.
  5.  Adjusts the new end time to ensure it does not exceed the current end time
      or the previous last record time.
  6.  If the new time range is different from the current time range, updates
      the current time range and repeats from step 1.
  7.  If the time range has stabilized (no further adjustment is needed), it
      updates the `proper_time_range` with a refined time range (removing half
      of the idle_time_buffer from each side) and returns the metric records.
  8. If the start time rewind has reached the maximum limit, it will raise a
      RuntimeError.

  Args:
      configs: The configuration object containing the parameters for the
        validation.
      **context: The airflow context.

  Returns:
      A List of EventRecord objects.

  Raises:
      RuntimeError: If the start time rewind has reached the maximum limit or
      if all records have been removed during the end time adjustment.
  """
  max_time_diff_sec = configs.max_time_diff_sec
  max_start_time_rewind_sec = configs.max_start_time_rewind_sec
  initial_start_time = configs.initial_start_time
  initial_end_time = configs.initial_end_time

  current_start_time = initial_start_time
  current_end_time = initial_end_time

  # The `idle_time_buffer` is set to 2x `max_time_diff_sec` (D) to handle
  # potential boundary issues in our time-range queries.
  #
  # Let's consider a `max_time_diff_sec` of 120s.
  # A metric event at 12:05 might have a log event anywhere from 12:03 to 12:07.
  #
  # If we only used a 1x buffer (120s), our query window would end at 12:05 + 120s = 12:07.
  #
  # What if a metric event happens at 12:08? Its corresponding log could be at 12:06.
  # Our log query (ending at 12:07) would capture this log, but the metric event
  # itself would be outside the metric query's range.
  #
  # By using a 2x buffer, we ensure that the time range is wide enough to
  # consistently capture both the metric and its corresponding log event,
  # regardless of where they fall relative to the initial query boundaries.
  idle_time_buffer = datetime.timedelta(seconds=max_time_diff_sec * 2)
  max_rewind_delta = datetime.timedelta(seconds=max_start_time_rewind_sec)

  # Continue to adjust the time range until the time range has stabilized or
  # the start time rewind has reached the maximum limit.
  while True:
    print(
        '\n current range:'
        f' {current_start_time.isoformat()} to {current_end_time.isoformat()}'
    )

    # Query metric data for the current time range by API.
    metric_records = query_metric_data_by_api(
        configs,
        current_start_time,
        current_end_time,
    )

    total_metric_timestamps = []
    for record in metric_records:
      total_metric_timestamps.extend(record.metric_points_timestamps)
    total_metric_timestamps.sort(
        key=lambda timestamp: (datetime.datetime.fromisoformat(timestamp))
    )

    first_record_time = datetime.datetime.fromisoformat(
        total_metric_timestamps[0]
    )
    last_record_time = datetime.datetime.fromisoformat(
        total_metric_timestamps[-1]
    )

    # Adjust start time (ensure there is idle_time_buffer before the earliest
    # record).
    calculated_new_start_time = first_record_time - idle_time_buffer

    # Allow current_start_time to be pushed forward as long as it does not
    # exceed max_rewind_delta. Update only when calculated_new_start_time is
    # earlier than current_start_time.
    new_start_time = min(current_start_time, calculated_new_start_time).replace(
        microsecond=0
    )
    new_end_time = current_end_time

    temp_records = total_metric_timestamps.copy()

    # If the previous_last_record_time is max, then potential_new_end >
    # previous_last_record_time will always be false. (for the first iteration)
    previous_last_record_time = datetime.datetime.max.replace(
        tzinfo=datetime.timezone.utc
    )

    while temp_records:
      current_last_record = datetime.datetime.fromisoformat(temp_records[-1])
      potential_new_end = current_last_record + idle_time_buffer

      # If potential_new_end is still later than current_end_time or the
      # previous last record time (except for the first iteration), this means
      # that even if the current last record is taken as the basis, the buffer
      # exceeds current_end_time or the previous last record time.
      # Since end_time cannot be extended, this current_last_record cannot be
      # our final last record.
      # We need to remove the current_last_record and try again.
      if (
          potential_new_end > current_end_time
          or potential_new_end > previous_last_record_time
      ):
        print(
            f'The last record time {current_last_record.isoformat()} + buffer'
            f' {idle_time_buffer} exceeds the current end_time'
            f' {current_end_time.isoformat()} or the previous last record time'
            ' {previous_last_record_time.isoformat()}. Removing this record'
            ' and recalculating.'
        )
        previous_last_record_time = datetime.datetime.fromisoformat(
            temp_records.pop()
        )

        if not temp_records:
          raise RuntimeError(
              'All records have been removed, and a suitable end_time cannot'
              ' be found.'
          )
      else:
        # If potential_new_end is less than or equal to current_end_time
        # This means that based on the current last record, the buffer does not
        # exceed current_end_time.
        # This is the new end_time we want.
        new_end_time = potential_new_end.replace(microsecond=0)
        break

    print(f'Earliest record time: {first_record_time.isoformat()}')
    print(f'Latest record time: {last_record_time.isoformat()}')
    print(f'Calculated new start time: {new_start_time.isoformat()}')
    print(f'Calculated new end time: {new_end_time.isoformat()}')

    # confirm whether the time range has stabilized
    if (
        new_start_time == current_start_time
        and new_end_time == current_end_time
    ):
      print('Time range has stabilized, no need to adjust again.')

      # Update the proper_time_range with the new start_time and end_time.
      proper_time_range = ProperTimeRange(
          proper_start_time=current_start_time + idle_time_buffer / 2,
          proper_end_time=current_end_time - idle_time_buffer / 2
      )
      context['ti'].xcom_push(key='proper_time_range', value=proper_time_range)

      return metric_records
    else:
      current_start_time = new_start_time
      current_end_time = new_end_time
      print('Time range has been adjusted, need to check again.')

    if (initial_start_time - current_start_time) > max_rewind_delta:
      print(
          'Start time rewind has reached the maximum limit'
          f' ({max_start_time_rewind_sec} seconds), terminating adjustment.'
      )
      raise RuntimeError('Start time rewind has reached the maximum limit.')


@task
def fetch_interruption_logs_timestamps(
    configs: Configs,
    event_records: List[EventRecord],
    **context,
) -> List[EventRecord]:
  """This function queries the Logging API for a given validation_conf and time range to fetch the log entries.

  Args:
      configs: The configuration object containing the parameters for the
        validation.
      event_records: A list of EventRecord objects, containing metric events
        grouped by resource name.
      **context: The airflow context.

  Returns:
      A list of EventRecord objects. Each EventRecord must contain the log
      events timestamps for the resource name.
  """
  proper_time_range: ProperTimeRange = context['ti'].xcom_pull(
      task_ids='decide_time_window', key='proper_time_range'
  )

  project_id = configs.project_id
  interruption_reason = configs.interruption_reason
  log_filter_query = configs.interruption_reason.log_filter()
  max_results = configs.max_log_results

  logging_api_client = logging.Client(project=project_id)

  start_time_str = (
      proper_time_range.proper_start_time.replace('+00:00', 'Z')
  )
  end_time_str = (
      proper_time_range.proper_end_time.replace('+00:00', 'Z')
  )
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
            interruption_reason=interruption_reason,
        )
      event_records[log_node_name].log_filter = log_filter_query
      event_records[log_node_name].log_events_timestamps.append(
          aware_timestamp.isoformat()
      )

  if entry_count == max_results:
    raise RuntimeError(
        f'Log entries limit reached ({max_results} entries). '
        'This might indicate we are missing data. '
        'Consider increasing max_results or narrowing the time range.'
    )

  return list(event_records.values())


@task
def check_event_count_match(
    event_records: List[EventRecord],
) -> List[EventRecord]:
  """Checks if the number of metric events matches the number of log events for each resource.

  Args:
      event_records: A list of EventRecord objects, containing metric and log
        events grouped by resource name.

  Returns:
      A list of EventRecord objects, containing the updated validation results.
  """
  # We are primarily concerned with validating that the number of metric events
  # matches the number of log events.
  for event_record in event_records:
    # Check event count match first
    num_metric_events = len(event_record.metric_points_timestamps)
    num_log_events = len(event_record.log_events_timestamps)

    if num_metric_events != num_log_events:
      # If any mismatch is found, we will raise an error and terminate the DAG.
      raise RuntimeError(
          f'Event count mismatch. Expected {num_metric_events} metric'
          f' events but found {num_log_events} log events for node'
          f' "{event_record.resource_name}". One-to-one correspondence not'
          ' possible.'
      )

  return event_records


with models.DAG(
    dag_id='interruption_event_validation_dag',
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
  # Define time range for data fetching
  now = datetime.datetime.now(pytz.utc)
  initial_start_time = now - datetime.timedelta(hours=12)
  initial_end_time = now

  configs = Configs(
      initial_start_time=initial_start_time,
      initial_end_time=initial_end_time,
      project_id=models.Variable.get(
          'INTERRUPTION_PROJECT_ID', default_var=Project.TPU_PROD_ENV_ONE_VM.value,
      ),
      max_time_diff_sec=int(models.Variable.get(
          'INTERRUPTION_MAX_TIME_DIFF_SEC', default_var=150
      )),
      max_log_results=int(models.Variable.get(
          'INTERRUPTION_MAX_LOG_RESULTS', default_var=1000
      )),
      metric_aggregation=models.Variable.get(
          'INTERRUPTION_METRIC_AGGREGATION', default_var=None
      ),
      platform=Platform.GKE,
      interruption_reason=InterruptionReason.HOST_ERROR,
      max_start_time_rewind_sec=int(models.Variable.get(
          'INTERRUPTION_MAX_START_TIME_REWIND_SECONDS', default_var=3600
      )),
  )

  event_records_after_get_metrics = decide_time_window(
      configs=configs,
  )

  event_records_after_get_logs_and_metrics = fetch_interruption_logs_timestamps(
      configs=configs,
      event_records=event_records_after_get_metrics,
  )
  event_records_after_check_count_match = check_event_count_match(
      event_records_after_get_logs_and_metrics
  )

  # --- Task Workflow ---
  (
      event_records_after_get_metrics
      >> event_records_after_get_logs_and_metrics
      >> event_records_after_check_count_match
  )


