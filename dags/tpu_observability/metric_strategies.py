from abc import ABC, abstractmethod
import re
from typing import List

from google.cloud.monitoring_v3 import types

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
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
    """Parses the desired value from a list of TimeSeries objects."""
    pass

  @abstractmethod
  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> List[float]:
    """Parses the desired value from the raw tpu-info command output."""
    pass


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
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
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

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> List[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU HBM Usage":
        for row_dict in metric_table.body:
          hbm_value = row_dict["HBM Usage (GiB)"]
          match = re.search(r"(\d+\.\d+)\s*GiB\s*\/\s*(\d+\.\d+)\s*GiB", hbm_value)
          if match:
            tpu_info_data_values.append(float(match.group(1)))
    return tpu_info_data_values


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
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
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

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> List[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU HBM Usage":
        for row_dict in metric_table.body:
          hbm_value = row_dict["HBM Usage (GiB)"]
          match = re.search(r"(\d+\.\d+)\s*GiB\s*\/\s*(\d+\.\d+)\s*GiB", hbm_value)
          if match:
            tpu_info_data_values.append(float(match.group(2)))
    return tpu_info_data_values


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
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
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

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> List[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Duty Cycle":
        for row_dict in metric_table.body:
          dutycycle_value = row_dict["Duty Cycle (%)"]
          match = re.search(r"(\d+\.\d+)%", dutycycle_value)
          if match:
            tpu_info_data_values.append(float(match.group(2)))
    return tpu_info_data_values


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
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
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

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> List[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TensorCore Utilization":
        for row_dict in metric_table.body:
          tcu_value = row_dict["TensorCore Utilization"].replace("%", "")
          tpu_info_data_values.append(float(tcu_value))
    return tpu_info_data_values


class BufferTransferLatencyStrategy(BaseMetricStrategy):
  """Strategy for verifying Buffer Transfer Latency."""

  @property
  def metric_name(self) -> str:
    return "kubernetes.io/container/multislice/network/dcn_transfer_latencies"

  @property
  def tpu_info_metric_name(self) -> str:
    return "duty_cycle_percent"

  @property
  def dag_id_suffix(self) -> str:
    return "duty_cycle"

  def parse_from_monitoring(
      self, time_series_data: List[types.TimeSeries], **kwargs
  ) -> List[float]:
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

  def parse_from_tpu_info(self, tpu_info_metric_output: str) -> List[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Duty Cycle":
        for row_dict in metric_table.body:
          dutycycle_value = row_dict["Duty Cycle (%)"]
          match = re.search(r"(\d+\.\d+)%", dutycycle_value)
          if match:
            tpu_info_data_values.append(float(match.group(2)))
    return tpu_info_data_values


ALL_METRIC_STRATEGIES = [
    MemoryUsedStrategy(),
    MemoryTotalStrategy(),
    DutyCycleStrategy(),
    TensorcoreUtilizationStrategy(),
    BufferTransferLatencyStrategy(),
]
