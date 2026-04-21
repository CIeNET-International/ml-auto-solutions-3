"""Utility for parsing the output of the 'tpu-info' command."""

from dataclasses import dataclass
from enum import auto
from enum import IntEnum
import re

from airflow.decorators import task


# A type alias for a parsed row, mapping column headers to their values.
_TableRow = dict[str, str]


@dataclass
class Table:
  """Represents a single parsed table from the tpu-info output."""

  name: str
  raw_body: str
  body: list[_TableRow] | None = None

  def parse_body(self):
    """Parses the raw_body string to populate the structured body attribute."""

    class TableLineIndex(IntEnum):
      """Below is an example of the text returned by tpu-info, formatted as a table.

      | Chip        | Type         | Devices | PID |
      |-------------|--------------|---------|-----|
      | /dev/vfio/0 | TPU v6e chip | 1       | 24  |
      | /dev/vfio/1 | TPU v6e chip | 1       | 24  |
      | /dev/vfio/2 | TPU v6e chip | 1       | 24  |
      | /dev/vfio/3 | TPU v6e chip | 1       | 24  |
      """

      HEADER = 0
      SEPARATOR = 1
      DATA = 2

    lines = self.raw_body.strip().split("\n")

    if len(lines) < TableLineIndex.DATA:
      self.body = []
      return

    header_line = lines[TableLineIndex.HEADER]
    headers = [h.strip() for h in header_line.split("|")[1:-1]]

    data_lines = lines[TableLineIndex.DATA :]

    parsed_body = []
    for line in data_lines:
      columns = line.split("|")[1:-1]
      if len(columns) != len(headers):
        continue

      row_data: _TableRow = {
          header: col.strip() for header, col in zip(headers, columns)
      }
      parsed_body.append(row_data)

    self.body = parsed_body


@task
def parse_tpu_info_output(output: str) -> list[Table]:
  """Splits a multi-table Markdown string from tpu-info into a structured TpuInfo object.

  Args:
    output: The raw string output from the 'tpu-info' command.

  Returns:
    A list of Table objects with attributes populated for each found table.
  """
  pattern = re.compile(
      r"^([A-Za-z][^\n]*?)\s*\n+"
      r"(^\|[^\n]*(?:\n\|[^\n]*)*)",
      re.MULTILINE
  )

  parsed_tables = []

  for match in pattern.finditer(output):
    name = match.group(1).strip()
    raw_body = match.group(2).strip()

    table = Table(name=name, raw_body=raw_body, body=None)

    table.parse_body()
    parsed_tables.append(table)

  if not parsed_tables:
    raise ValueError("Failed to parse any tables from the tpu-info output.")

  return parsed_tables


if __name__ == "__main__":
  full_output = """
TPU Chips

| Chip        | Type         | Devices | PID |
|-------------|--------------|---------|-----|
| /dev/vfio/0 | TPU v6e chip | 1       | 24  |
| /dev/vfio/1 | TPU v6e chip | 1       | 24  |
| /dev/vfio/2 | TPU v6e chip | 1       | 24  |
| /dev/vfio/3 | TPU v6e chip | 1       | 24  |

TPU Runtime Utilization

| Chip | HBM Usage (GiB)      | Duty cycle |
|------|----------------------|------------|
| 0    | 8.44 GiB / 31.25 GiB | 100.00%    |
| 1    | 8.44 GiB / 31.25 GiB | 100.00%    |
| 4    | 8.44 GiB / 31.25 GiB | 100.00%    |
| 5    | 8.44 GiB / 31.25 GiB | 100.00%    |

TensorCore Utilization

| Core ID | TensorCore Utilization |
|---------|------------------------|
| 0       | 8.71%                  |
| 1       | 8.56%                  |
| 2       | 8.52%                  |
| 3       | 8.20%                  |

TPU Buffer Transfer Latency

| Buffer Size | P50         | P90         | P95          | P999         |
|-------------|-------------|-------------|--------------|--------------|
| 8MB+        | 54031.13 us | 96579.79 us | 103413.24 us | 131976.57 us |

TPU Inbound Buffer Transfer Latency

| Buffer Size | P50         | P90         | P95          | P999         |
|-------------|-------------|-------------|--------------|--------------|
| 8MB+        | 53198.63 us | 93329.45 us | 101954.57 us | 132554.97 us |

╭──────────────────────── Host Compute Latency Status ─────────────────────────╮
│ WARNING: Host Compute Latency metrics unavailable. Did you start a           │
│ MULTI_SLICE workload with `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?   │
╰──────────────────────────────────────────────────────────────────────────────╯
TPU gRPC TCP Minimum RTT

| P50      | P90      | P95      | P999     |
|----------|----------|----------|----------|
| 66.79 us | 82.26 us | 84.50 us | 86.18 us |

TPU gRPC TCP Delivery Rate

| P50           | P90           | P95           | P999          |
|---------------|---------------|---------------|---------------|
| 12408.00 Mbps | 27133.21 Mbps | 30492.25 Mbps | 35873.24 Mbps |
"""

  tpu_info_output = parse_tpu_info_output(full_output)
  print(tpu_info_output)
  for i in tpu_info_output:
    print(i.name)
    print(i.body)
