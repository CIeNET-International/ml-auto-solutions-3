"""Utility for parsing the output of the 'tpu-info' command."""

import re
from dataclasses import dataclass, field
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
    self.body = []
    lines = self.raw_body.strip().split("\n")
    if len(lines) < 4:
      return
    header_line = lines[1]
    headers = [h.strip() for h in header_line.split("┃") if h.strip()]
    data_lines = lines[3:-1]
    for line in data_lines:
      values = [v.strip() for v in line.split("│")][1:-1]
      if len(values) == len(headers):
        self.body.append(dict(zip(headers, values)))


# A mapping from the exact table titles in the raw output to the
# attribute names in the TpuInfo dataclass.
TABLE_NAME_TO_ATTR = {
    "TPU Chips": "chips",
    "TPU Runtime Utilization": "runtime_utilization",
    "TensorCore Utilization": "tensorcore_utilization",
    "TPU Buffer Transfer Latency": "buffer_transfer_latency",
}


@task
def parse_tpu_info_output(output: str) -> dict:
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

  tpu_info_dict = {}

  parsed_tables = [
      Table(name=name, raw_body=body) for name, body in zip(titles, blocks)
  ]

  for table in parsed_tables:
    attribute_name = TABLE_NAME_TO_ATTR.get(table.name)
    if attribute_name:
      tpu_info_dict[attribute_name] = {
          "name": table.name,
          "raw_body": table.raw_body,
          "body": table.body,
      }

  return tpu_info_dict


if __name__ == "__main__":
  full_output = """
TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ /dev/vfio/0 │ TPU v6e chip │ 1       │ 1019 │
│ /dev/vfio/1 │ TPU v6e chip │ 1       │ 1019 │
│ /dev/vfio/2 │ TPU v6e chip │ 1       │ 1019 │
│ /dev/vfio/3 │ TPU v6e chip │ 1       │ 1019 │
└─────────────┴──────────────┴─────────┴──────┘
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
│ 0       │ 8.06%                  │
│ 1       │ 7.51%                  │
│ 2       │ 7.56%                  │
│ 3       │ 8.75%                  │
└─────────┴────────────────────────┘
TPU Buffer Transfer Latency
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Buffer Size ┃ P50         ┃ P90         ┃ P95         ┃ P999         ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 8MB+        │ 44168.05 us │ 79499.95 us │ 85675.90 us │ 164516.53 us │
└─────────────┴─────────────┴─────────────┴─────────────┴──────────────┘
"""
  tpu_info = parse_tpu_info_output(full_output)
  print(tpu_info)
  print(type(tpu_info))
  print(tpu_info["chips"]["body"])
  print(tpu_info["runtime_utilization"]["body"])
  print(tpu_info["tensorcore_utilization"]["body"])
  print(tpu_info["buffer_transfer_latency"]["body"])
