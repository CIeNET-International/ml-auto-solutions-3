"""Utility for parsing the output of the 'tpu-info' command."""

import re
from dataclasses import dataclass, field, asdict
from enum import IntEnum, auto
from typing import Dict, List

from airflow.decorators import task

# A type alias for a parsed row, mapping column headers to their values.
_TableRow = Dict[str, str]


@dataclass
class Table:
  """Represents a single parsed table from the tpu-info output."""

  name: str
  raw_body: str
  body: List[_TableRow] = field(init=False, repr=False)

  def __post_init__(self):
    """Parses the raw_body string to populate the structured body attribute."""

    class TableLineIndex(IntEnum):

      """Defines the expected line indices for different parts of a raw table.

      Output example:
                                      Libtpu version: 0.0.20.dev20250722+nightly
                                      Accelerator type: v6e

      TPU Chips
      ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
      ┃ Chip        ┃ Type         ┃ Devices ┃ PID  ┃
      ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
      │ /dev/vfio/0 │ TPU v6e chip │ 1       │ 1016 │
      │ /dev/vfio/1 │ TPU v6e chip │ 1       │ 1016 │
      │ /dev/vfio/2 │ TPU v6e chip │ 1       │ 1016 │
      │ /dev/vfio/3 │ TPU v6e chip │ 1       │ 1016 │
      └─────────────┴──────────────┴─────────┴──────┘
      TPU Runtime Utilization
      ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
      ┃ Device ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
      ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
      │ 0      │ 18.45 GiB / 31.25 GiB │ 100.00%    │
      │ 1      │ 10.40 GiB / 31.25 GiB │ 100.00%    │
      │ 4      │ 10.40 GiB / 31.25 GiB │ 100.00%    │
      │ 5      │ 10.40 GiB / 31.25 GiB │ 100.00%    │
      └────────┴───────────────────────┴────────────┘
      TensorCore Utilization
      ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃ Chip ID ┃ TensorCore Utilization ┃
      ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
      │ 0       │ 15.42%                 │
      │ 1       │ 15.28%                 │
      │ 2       │ 14.64%                 │
      │ 3       │ 14.52%                 │
      └─────────┴────────────────────────┘
      TPU Buffer Transfer Latency
      ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
      ┃ Buffer Size ┃ P50         ┃ P90          ┃ P95          ┃ P999         ┃
      ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
      │ 4MB+        │ 16524.14 us │ 29825.32 us  │ 33527.37 us  │ 44780.14 us  │
      │ 8MB+        │ 34693.85 us │ 564965.07 us │ 608747.76 us │ 650846.50 us │
      └─────────────┴─────────────┴──────────────┴──────────────┴──────────────┘
      """

      UPPER_BORDER = 0
      HEADER = auto()
      SEPARATOR = auto()
      DATA = auto()
      LOWER_BORDER = -1

    lines = self.raw_body.strip().split("\n")
    if len(lines) < max(TableLineIndex):
      return

    self.body = []
    header_line = lines[TableLineIndex.HEADER]
    headers = [h.strip() for h in header_line.split("┃") if h.strip()]

    data_lines = lines[TableLineIndex.DATA : TableLineIndex.LOWER_BORDER]

    for line in data_lines:
      columns = line.split("│")[1:-1]
      if len(columns) != len(headers):
        continue

      row_data: _TableRow = {
          header: col.strip() for header, col in zip(headers, columns)
      }

      self.body.append(row_data)


@task
def parse_tpu_info_output(output: str) -> List[Table]:
  """Splits a multi-table string from tpu-info into a structured TpuInfo object.

  Args:
    output: The raw string output from the 'tpu-info' command.

  Returns:
    A TpuInfo object with attributes populated for each found table.
  """
  title_pattern = re.compile(r"(^[^\n].*)\n┏", re.MULTILINE)
  table_block_pattern = re.compile(r"(^┏[\s\S]*?┘)", re.MULTILINE)

  titles = [s.strip() for s in title_pattern.findall(output)]
  blocks = table_block_pattern.findall(output)

  if len(titles) != len(blocks):
    raise ValueError(
        "Mismatch between found table titles and table blocks. "
        f"Found {len(titles)} titles and {len(blocks)} blocks."
    )

  parsed_tables = [
      Table(name=name, raw_body=body) for name, body in zip(titles, blocks)
  ]

  return parsed_tables


if __name__ == "__main__":
  full_output = """
                                       Libtpu version: 0.0.20.dev20250722+nightly
                                      Accelerator type: v6e

TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━┩
│ /dev/vfio/0 │ TPU v6e chip │ 1       │ 7   │
│ /dev/vfio/1 │ TPU v6e chip │ 1       │ 7   │
│ /dev/vfio/2 │ TPU v6e chip │ 1       │ 7   │
│ /dev/vfio/3 │ TPU v6e chip │ 1       │ 7   │
└─────────────┴──────────────┴─────────┴─────┘
TPU Runtime Utilization
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Device ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 8      │ 17.26 GiB / 31.25 GiB │ 100.00%    │
│ 9      │ 9.26 GiB / 31.25 GiB  │ 100.00%    │
│ 12     │ 9.26 GiB / 31.25 GiB  │ 100.00%    │
│ 13     │ 9.26 GiB / 31.25 GiB  │ 100.00%    │
└────────┴───────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Chip ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 7.71%                  │
│ 1       │ 7.62%                  │
│ 2       │ 7.64%                  │
│ 3       │ 7.62%                  │
└─────────┴────────────────────────┘
TPU Buffer Transfer Latency
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Buffer Size ┃ P50         ┃ P90         ┃ P95          ┃ P999         ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 8MB+        │ 53752.55 us │ 96313.78 us │ 103951.79 us │ 348016.24 us │
└─────────────┴─────────────┴─────────────┴──────────────┴──────────────┘
"""

  tpu_info_output = parse_tpu_info_output(full_output)
  print(tpu_info_output)
  content = next(
      (table.body for table in tpu_info_output if table.name == "TPU Chips"),
      None,
  )
  print(content)
