"""Utility class for handling various time representations."""

import datetime
from typing import Union
from airflow.exceptions import AirflowException
from google.protobuf import timestamp_pb2

TimeInput = Union[datetime.datetime, str, int, float, "TimeUtil"]


class TimeUtil:
  """A utility class to handle various time representations and provide a unified interface."""

  def __init__(self, unix_seconds: int):
    self.time: int = unix_seconds

  @classmethod
  def build(
      cls, time_input: TimeInput, arg_name: str = "time_input"
  ) -> "TimeUtil":
    """Builds a TimeUtil object from various possible input formats.

    This is the main entry point for creating an instance of this class.

    Args:
      time_input: The time value, can be a datetime, ISO string, timestamp
        (int/float), or another TimeUtil object.
      arg_name: The name of the argument being processed, used for error
        messages.

    Returns:
      A TimeUtil object.

    Raises:
      AirflowException: If a string input has an invalid format.
      TypeError: If the time_input is of an unsupported type.
    """
    if isinstance(time_input, cls):
      return time_input
    if isinstance(time_input, datetime.datetime):
      return cls._from_datetime(time_input)
    elif isinstance(time_input, (int, float)):
      return cls._from_timestamp(time_input)
    elif isinstance(time_input, str):
      try:
        return cls._from_iso_string(time_input)
      except ValueError as e:
        raise AirflowException(f"Invalid format for {arg_name}: {e}") from e
    else:
      raise TypeError(
          f"Unsupported type for {arg_name}: {type(time_input)}. "
          "Must be datetime, str (ISO 8601), int, float, or TimeUtil."
      )

  @classmethod
  def _from_iso_string(cls, time_str: str) -> "TimeUtil":
    dt_object = datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    return cls(int(dt_object.timestamp()))

  @classmethod
  def _from_timestamp(cls, ts: Union[int, float]) -> "TimeUtil":
    return cls(int(ts))

  @classmethod
  def _from_datetime(cls, dt: datetime.datetime) -> "TimeUtil":
    return cls(int(dt.timestamp()))

  def to_seconds(self) -> int:
    return self.time

  def to_protobuf_timestamp(self) -> timestamp_pb2.Timestamp:
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromSeconds(self.time)
    return timestamp

  def to_datetime(self) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(self.time, tz=datetime.timezone.utc)

  def to_iso_format(self) -> str:
    iso_str = self.to_datetime().isoformat()
    return iso_str.replace("+00:00", "Z")

if __name__ == "__main__":
  start_time = datetime.datetime.fromisoformat("2025-09-19T04:08:35.951+00:00")
  end_time = start_time + datetime.timedelta(minutes=10)

  start_time_obj = TimeUtil.build(start_time, arg_name="start_time")
  end_time_obj = TimeUtil.build(end_time, arg_name="end_time")

  print(start_time_obj.to_protobuf_timestamp())
  print(end_time_obj.to_protobuf_timestamp())
