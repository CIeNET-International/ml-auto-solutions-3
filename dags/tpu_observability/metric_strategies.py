from abc import ABC, abstractmethod
import re
from typing import Any

from google.cloud.monitoring_v3 import types
from airflow.exceptions import AirflowException


def _calculate_percentiles_from_histogram(
    percentiles: list[float],
    total_count: int,
    bounds: list[float],
    bucket_counts: list[int],
) -> dict[float, float]:
  """Estimates multiple percentile values from histogram data in a single pass."""
  if total_count == 0:
    return {p: 0.0 for p in percentiles}

  sorted_percentiles = sorted(percentiles)
  target_ranks = {p: total_count * (p / 100.0) for p in sorted_percentiles}
  results = {}
  cumulative_count = 0
  percentiles_to_find = list(sorted_percentiles)

  for i, count_in_bucket in enumerate(bucket_counts):
    if not percentiles_to_find:
      break

    prev_cumulative_count = cumulative_count
    cumulative_count += count_in_bucket

    while (
        percentiles_to_find
        and target_ranks[percentiles_to_find[0]] <= cumulative_count
    ):
      p = percentiles_to_find.pop(0)
      target_rank = target_ranks[p]
      lower_bound = bounds[i - 1] if i > 0 else 0.0
      upper_bound = bounds[i]

      if count_in_bucket == 0:
        results[p] = lower_bound
        continue

      rank_in_bucket = target_rank - prev_cumulative_count
      fraction = rank_in_bucket / count_in_bucket
      estimated_value = lower_bound + fraction * (upper_bound - lower_bound)
      results[p] = estimated_value

  return results


class BaseMetricStrategy(ABC):
  """
  Abstract Base Class (Interface) for a metric verification strategy.
  It defines the contract that all concrete metric strategies must follow.
  """

  @property
  @abstractmethod
  def metric_name(self) -> str:
    """The name of the metric as it appears in the Monitoring filter."""
    pass

  @property
  @abstractmethod
  def tpu_info_metric_name(self) -> str:
    """The name of the metric to be searched from the tpu-info command."""
    pass

  @property
  @abstractmethod
  def dag_id_suffix(self) -> str:
    """The suffix used to generate the DAG ID, which should be unique."""
    pass

  @abstractmethod
  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    """Parses the desired value from a list of TimeSeries objects."""
    pass

  @abstractmethod
  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> list[float]:
    """Parses the desired value from the raw tpu-info command output."""
    pass

  @property
  def tolerance_percent(self) -> float:
    """
    The relative tolerance (in percent) to use for this metric's verification.
    Subclasses should override this value to set a custom tolerance.
    """
    return 2.0


class MemoryUsedStrategy(BaseMetricStrategy):
  """Strategy for verifying Used HBM Memory."""

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/accelerator/memory_used"

  @property
  def tpu_info_metric_name(self) -> str:
    return "hbm_usage"

  @property
  def dag_id_suffix(self) -> str:
    return "memory_used"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    metric_values = {}
    for ts in time_series_data:
      if ts.points:
        accelerator_id = ts.metric.labels["accelerator_id"]
        point = ts.points[0]
        bytes_value = point.value.int64_value
        gib_value = bytes_value / (1024**3)
        metric_values[accelerator_id] = round(gib_value, 2)
    return [
        metric_values[key]
        for key in sorted(
            metric_values.keys(), key=lambda x: int(x.split("-")[-1])
        )
    ]

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU HBM Usage":
        for row_dict in metric_table.body:
          hbm_value = row_dict["HBM Usage (GiB)"]
          match = re.search(
              r"(\d+\.\d+)\s*GiB\s*\/\s*(\d+\.\d+)\s*GiB", hbm_value
          )
          if match:
            tpu_info_data_values.append(float(match.group(1)))
    return tpu_info_data_values

  @property
  def tolerance_percent(self) -> float:
    return 1.0


class MemoryTotalStrategy(BaseMetricStrategy):
  """Strategy for verifying Total HBM Memory."""

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/accelerator/memory_total"

  @property
  def tpu_info_metric_name(self) -> str:
    return "hbm_usage"

  @property
  def dag_id_suffix(self) -> str:
    return "memory_total"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    metric_values = {}
    for ts in time_series_data:
      if ts.points:
        accelerator_id = ts.metric.labels["accelerator_id"]
        point = ts.points[0]
        bytes_value = point.value.int64_value
        gib_value = bytes_value / (1024**3)
        metric_values[accelerator_id] = round(gib_value, 2)
    return [
        metric_values[key]
        for key in sorted(
            metric_values.keys(), key=lambda x: int(x.split("-")[-1])
        )
    ]

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU HBM Usage":
        for row_dict in metric_table.body:
          hbm_value = row_dict["HBM Usage (GiB)"]
          match = re.search(
              r"(\d+\.\d+)\s*GiB\s*\/\s*(\d+\.\d+)\s*GiB", hbm_value
          )
          if match:
            tpu_info_data_values.append(float(match.group(2)))
    return tpu_info_data_values

  @property
  def tolerance_percent(self) -> float:
    return 0.0


class DutyCycleStrategy(BaseMetricStrategy):
  """Strategy for verifying Duty Cycle."""

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/accelerator/duty_cycle"

  @property
  def tpu_info_metric_name(self) -> str:
    return "duty_cycle_percent"

  @property
  def dag_id_suffix(self) -> str:
    return "duty_cycle"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    metric_values = {}
    for ts in time_series_data:
      if ts.points:
        accelerator_id = ts.metric.labels["accelerator_id"]
        point = ts.points[0]
        metric_values[accelerator_id] = round(point.value.int64_value, 2)
    return [
        metric_values[key]
        for key in sorted(
            metric_values.keys(), key=lambda x: int(x.split("-")[-1])
        )
    ]

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Duty Cycle":
        for row_dict in metric_table.body:
          dutycycle_value = row_dict["Duty Cycle (%)"]
          match = re.search(r"(\d+\.\d+)%", dutycycle_value)
          if match:
            tpu_info_data_values.append(float(match.group(1)))
    return tpu_info_data_values

  @property
  def tolerance_percent(self) -> float:
    return 1.0


class TensorcoreUtilizationStrategy(BaseMetricStrategy):
  """Strategy for verifying TensorCore Utilization."""

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/accelerator/tensorcore_utilization"

  @property
  def tpu_info_metric_name(self) -> str:
    return "tensorcore_utilization"

  @property
  def dag_id_suffix(self) -> str:
    return "tensorcore_utilization"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    metric_values = {}
    for ts in time_series_data:
      if ts.points:
        accelerator_id = ts.metric.labels["accelerator_id"]
        point = ts.points[0]
        metric_values[accelerator_id] = round(point.value.double_value, 2)
    return [
        metric_values[key]
        for key in sorted(
            metric_values.keys(), key=lambda x: int(x.split("-")[-1])
        )
    ]

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TensorCore Utilization":
        for row_dict in metric_table.body:
          tcu_value = row_dict["TensorCore Utilization"].replace("%", "")
          tpu_info_data_values.append(float(tcu_value))
    return tpu_info_data_values

  @property
  def tolerance_percent(self) -> float:
    return 10.0


class BufferTransferLatencyStrategy(BaseMetricStrategy):
  """Strategy for verifying Buffer Transfer Latency from distribution data."""

  def __init__(self, percentiles_to_check: list[float]):
    """Initializes the Strategy.

    Args:
      percentiles_to_check: A list of percentiles to verify, e.g., [50, 90, 95,
        99.9].
    """
    if not percentiles_to_check:
      raise ValueError("percentiles_to_check cannot be empty.")
    self.percentiles_to_check = sorted(percentiles_to_check)
    super().__init__()

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/multislice/network/dcn_transfer_latencies"

  @property
  def tpu_info_metric_name(self) -> str:
    return "buffer_transfer_latency"

  @property
  def dag_id_suffix(self) -> str:
    return "buffer_transfer_latency"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    """Parses and calculates the specified percentiles from Cloud Monitoring's distribution data."""
    distributions_by_buffer: dict[str, dict[str, Any]] = {}
    for ts in time_series_data:
      if ts.points and ts.metric.labels.get("buffer_size"):
        buffer_size = ts.metric.labels["buffer_size"]
        dist_value = ts.points[0].value.distribution_value
        distributions_by_buffer[buffer_size] = {
            "count": dist_value.count,
            "bounds": dist_value.bucket_options.explicit_buckets.bounds,
            "bucket_counts": dist_value.bucket_counts,
        }

    percentile_values_by_buffer: dict[str, dict[float, float]] = {}
    for buffer_size, data in distributions_by_buffer.items():
      percentile_values_by_buffer[
          buffer_size
      ] = _calculate_percentiles_from_histogram(
          percentiles=self.percentiles_to_check,
          total_count=data["count"],
          bounds=data["bounds"],
          bucket_counts=data["bucket_counts"],
      )

    monitoring_values = []
    for buffer_size in sorted(distributions_by_buffer.keys()):
      for p in self.percentiles_to_check:
        monitoring_values.append(percentile_values_by_buffer[buffer_size][p])

    return monitoring_values

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[Any]
  ) -> list[float]:
    """Parses percentile values from the output table of the tpu-info tool.

    Args:
      tpu_info_metric_output: A list of tables from the tpu-info command output.

    Returns:
      A list of float values representing the parsed percentiles, ordered by
      buffer size and then by percentile.
    """
    parsed_values_by_buffer: dict[str, dict[float, float]] = {}

    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Buffer Transfer Latency":
        for row_dict in metric_table.body:
          buffer_size = row_dict.get("Buffer Size")
          if not buffer_size:
            continue

          parsed_values_by_buffer[buffer_size] = {}
          for p in self.percentiles_to_check:
            # Handle the special key name for P99.9
            p_key = f"P{p}" if p != 99.9 else "P999"
            value_str = row_dict.get(p_key, "")

            match = re.search(r"([\d\.]+)", value_str)
            if match:
              parsed_values_by_buffer[buffer_size][p] = float(match.group(1))

    tpu_info_data_values = []
    for buffer_size in sorted(parsed_values_by_buffer.keys()):
      for p in self.percentiles_to_check:
        tpu_info_data_values.append(parsed_values_by_buffer[buffer_size][p])

    return tpu_info_data_values


class HostToDeviceTransferLatenciesStrategy(BaseMetricStrategy):
  """Strategy for verifying Host to Device Transfer Latency from distribution data."""

  def __init__(self, percentiles_to_check: list[float]):
    """Initializes the Strategy.

    Args:
      percentiles_to_check: A list of percentiles to verify, e.g., [50, 90, 95,
        99.9].
    """
    if not percentiles_to_check:
      raise ValueError("percentiles_to_check cannot be empty.")
    self.percentiles_to_check = sorted(percentiles_to_check)
    super().__init__()

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/multislice/accelerator/host_to_device_transfer_latencies"

  @property
  def tpu_info_metric_name(self) -> str:
    return "host_to_device_transfer_latency"

  @property
  def dag_id_suffix(self) -> str:
    return "host_to_device_transfer_latency"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    """Parses and calculates the specified percentiles from Cloud Monitoring's distribution data."""
    distributions_by_buffer: dict[str, dict[str, Any]] = {}
    for ts in time_series_data:
      if ts.points and ts.metric.labels.get("buffer_size"):
        buffer_size = ts.metric.labels["buffer_size"]
        dist_value = ts.points[0].value.distribution_value
        distributions_by_buffer[buffer_size] = {
            "count": dist_value.count,
            "bounds": dist_value.bucket_options.explicit_buckets.bounds,
            "bucket_counts": dist_value.bucket_counts,
        }

    percentile_values_by_buffer: dict[str, dict[float, float]] = {}
    for buffer_size, data in distributions_by_buffer.items():
      percentile_values_by_buffer[
          buffer_size
      ] = _calculate_percentiles_from_histogram(
          percentiles=self.percentiles_to_check,
          total_count=data["count"],
          bounds=data["bounds"],
          bucket_counts=data["bucket_counts"],
      )

    monitoring_values = []
    for buffer_size in sorted(distributions_by_buffer.keys()):
      for p in self.percentiles_to_check:
        monitoring_values.append(percentile_values_by_buffer[buffer_size][p])

    return monitoring_values

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[Any]
  ) -> list[float]:
    """Parses percentile values from the output table of the tpu-info tool.

    Args:
      tpu_info_metric_output: A list of tables from the tpu-info command output.

    Returns:
      A list of float values representing the parsed percentiles, ordered by
      buffer size and then by percentile.
    """
    parsed_values_by_buffer: dict[str, dict[float, float]] = {}

    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Host to Device Transfer Latency":
        for row_dict in metric_table.body:
          buffer_size = row_dict.get("Buffer Size")
          if not buffer_size:
            continue

          parsed_values_by_buffer[buffer_size] = {}
          for p in self.percentiles_to_check:
            # Handle the special key name for P99.9
            p_key = f"P{p}" if p != 99.9 else "P999"
            value_str = row_dict.get(p_key, "")

            match = re.search(r"([\d\.]+)", value_str)
            if match:
              parsed_values_by_buffer[buffer_size][p] = float(match.group(1))

    tpu_info_data_values = []
    for buffer_size in sorted(parsed_values_by_buffer.keys()):
      for p in self.percentiles_to_check:
        tpu_info_data_values.append(parsed_values_by_buffer[buffer_size][p])

    return tpu_info_data_values


class DeviceToHostTransferLatenciesStrategy(BaseMetricStrategy):
  """Strategy for verifying Device to Host Transfer Latency from distribution data."""

  def __init__(self, percentiles_to_check: list[float]):
    """Initializes the Strategy.

    Args:
      percentiles_to_check: A list of percentiles to verify, e.g., [50, 90, 95,
        99.9].
    """
    if not percentiles_to_check:
      raise ValueError("percentiles_to_check cannot be empty.")
    self.percentiles_to_check = sorted(percentiles_to_check)
    super().__init__()

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/multislice/accelerator/device_to_host_transfer_latencies"

  @property
  def tpu_info_metric_name(self) -> str:
    return "device_to_host_transfer_latency"

  @property
  def dag_id_suffix(self) -> str:
    return "device_to_host_transfer_latency"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    """Parses and calculates the specified percentiles from Cloud Monitoring's distribution data."""
    distributions_by_buffer: dict[str, dict[str, Any]] = {}
    for ts in time_series_data:
      if ts.points and ts.metric.labels.get("buffer_size"):
        buffer_size = ts.metric.labels["buffer_size"]
        dist_value = ts.points[0].value.distribution_value
        distributions_by_buffer[buffer_size] = {
            "count": dist_value.count,
            "bounds": dist_value.bucket_options.explicit_buckets.bounds,
            "bucket_counts": dist_value.bucket_counts,
        }

    percentile_values_by_buffer: dict[str, dict[float, float]] = {}
    for buffer_size, data in distributions_by_buffer.items():
      percentile_values_by_buffer[
          buffer_size
      ] = _calculate_percentiles_from_histogram(
          percentiles=self.percentiles_to_check,
          total_count=data["count"],
          bounds=data["bounds"],
          bucket_counts=data["bucket_counts"],
      )

    monitoring_values = []
    for buffer_size in sorted(distributions_by_buffer.keys()):
      for p in self.percentiles_to_check:
        monitoring_values.append(percentile_values_by_buffer[buffer_size][p])

    return monitoring_values

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[Any]
  ) -> list[float]:
    """Parses percentile values from the output table of the tpu-info tool.

    Args:
      tpu_info_metric_output: A list of tables from the tpu-info command output.

    Returns:
      A list of float values representing the parsed percentiles, ordered by
      buffer size and then by percentile.
    """
    parsed_values_by_buffer: dict[str, dict[float, float]] = {}

    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Device to Host Transfer Latency":
        for row_dict in metric_table.body:
          buffer_size = row_dict.get("Buffer Size")
          if not buffer_size:
            continue

          parsed_values_by_buffer[buffer_size] = {}
          for p in self.percentiles_to_check:
            # Handle the special key name for P99.9
            p_key = f"P{p}" if p != 99.9 else "P999"
            value_str = row_dict.get(p_key, "")

            match = re.search(r"([\d\.]+)", value_str)
            if match:
              parsed_values_by_buffer[buffer_size][p] = float(match.group(1))

    tpu_info_data_values = []
    for buffer_size in sorted(parsed_values_by_buffer.keys()):
      for p in self.percentiles_to_check:
        tpu_info_data_values.append(parsed_values_by_buffer[buffer_size][p])

    return tpu_info_data_values


class CollectiveEndToEndLatencyLatenciesStrategy(BaseMetricStrategy):
  """Strategy for verifying Collective End to End Latency from distribution data."""

  def __init__(self, percentiles_to_check: list[float]):
    """Initializes the Strategy.

    Args:
      percentiles_to_check: A list of percentiles to verify, e.g., [50, 90, 95,
        99.9].
    """
    if not percentiles_to_check:
      raise ValueError("percentiles_to_check cannot be empty.")
    self.percentiles_to_check = sorted(percentiles_to_check)
    super().__init__()

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/multislice/network/collective_end_to_end_latencies"

  @property
  def tpu_info_metric_name(self) -> str:
    return "collective_e2e_latency"

  @property
  def dag_id_suffix(self) -> str:
    return "collective_e2e_latency"

  def parse_from_monitoring(
      self, time_series_data: list[types.TimeSeries], **kwargs
  ) -> list[float]:
    """Parses and calculates the specified percentiles from Cloud Monitoring's distribution data."""
    distributions_by_buffer: dict[str, dict[str, Any]] = {}
    for ts in time_series_data:
      if ts.points and ts.metric.labels.get("collective_type"):
        buffer_size = ts.metric.labels["collective_type"]
        dist_value = ts.points[0].value.distribution_value
        distributions_by_buffer[buffer_size] = {
            "count": dist_value.count,
            "bounds": dist_value.bucket_options.explicit_buckets.bounds,
            "bucket_counts": dist_value.bucket_counts,
        }

    percentile_values_by_buffer: dict[str, dict[float, float]] = {}
    for buffer_size, data in distributions_by_buffer.items():
      percentile_values_by_buffer[
          buffer_size
      ] = _calculate_percentiles_from_histogram(
          percentiles=self.percentiles_to_check,
          total_count=data["count"],
          bounds=data["bounds"],
          bucket_counts=data["bucket_counts"],
      )

    monitoring_values = []
    for buffer_size in sorted(distributions_by_buffer.keys()):
      for p in self.percentiles_to_check:
        monitoring_values.append(percentile_values_by_buffer[buffer_size][p])

    return monitoring_values

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[Any]
  ) -> list[float]:
    """Parses percentile values from the output table of the tpu-info tool.

    Args:
      tpu_info_metric_output: A list of tables from the tpu-info command output.

    Returns:
      A list of float values representing the parsed percentiles, ordered by
      buffer size and then by percentile.
    """
    parsed_values_by_buffer: dict[str, dict[float, float]] = {}

    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Collective End to End Latency":
        for i, row_dict in enumerate(metric_table.body):
          buffer_size = row_dict.get("Buffer Size")
          if not buffer_size:
            continue

          buffer_size_name = buffer_size + "(" + str(i) + ")"
          parsed_values_by_buffer[buffer_size_name] = {}
          for p in self.percentiles_to_check:
            # Handle the special key name for P99.9
            p_key = f"P{p}" if p != 99.9 else "P999"
            value_str = row_dict.get(p_key, "")

            match = re.search(r"([\d\.]+)", value_str)
            if match:
              parsed_values_by_buffer[buffer_size_name][p] = float(
                  match.group(1)
              )

    tpu_info_data_values = []
    for buffer_size_name in sorted(parsed_values_by_buffer.keys()):
      for p in self.percentiles_to_check:
        tpu_info_data_values.append(
            parsed_values_by_buffer[buffer_size_name][p]
        )

    return tpu_info_data_values


ALL_METRIC_STRATEGIES = [
    MemoryUsedStrategy(),
    MemoryTotalStrategy(),
    DutyCycleStrategy(),
    TensorcoreUtilizationStrategy(),
    BufferTransferLatencyStrategy([50, 90, 95, 99.9]),
    HostToDeviceTransferLatenciesStrategy([50, 90, 95, 99.9]),
    DeviceToHostTransferLatenciesStrategy([50, 90, 95, 99.9]),
    CollectiveEndToEndLatencyLatenciesStrategy([50, 90, 95, 99.9]),
]
