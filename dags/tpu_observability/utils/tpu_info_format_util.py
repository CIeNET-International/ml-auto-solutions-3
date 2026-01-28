# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for executing TPU info commands and validating their outputs."""


import os
import re
import subprocess
import tempfile

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from dags.tpu_observability.configs.common import (
    TpuConfig,
)
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils import tpu_info_util as tpu_info


@task
def get_tpu_info_from_pod(info: node_pool.Info, pod_name: str) -> str:
  """
  Executes the `tpu-info` command in a specified pod and returns its output.

  This task uses kubectl to run the 'tpu-info' command inside the given pod
  in the 'default' namespace. The output of the command is captured and
  returned.

  Args:
    info: Information about the node pool.
    pod_name: The name of the pod to execute the command in.

  Returns:
    The standard output from the 'tpu-info' command.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- tpu-info",
    ])

    return subprocess.run_exec(cmd, env=env)


@task
def verify_table_amount(tpu_info_output: list[tpu_info.Table]):
  """
  Verifies if all expected tables are present.
  """
  expect_table_names = {
      "TPU Chips",
      "TPU Runtime Utilization",
      "TensorCore Utilization",
      "TPU Buffer Transfer Latency",
  }

  found_names = {table.name for table in tpu_info_output}

  missing_names = expect_table_names - found_names

  if missing_names:
    raise AirflowFailException(
        "Mismatched tpu-info tables; "
        f"required: {expect_table_names}; got: {found_names}"
    )


@task
def validate_chips_table(
    tpu_info_output: list[tpu_info.Table],
    tpu_config: TpuConfig,
):
  """
  Validates the row count and content for the 'TPU Chips' table.
  """
  errors = []
  content = next(
      (table for table in tpu_info_output if table.name == "TPU Chips"),
      None,
  )

  expected_rows = 4
  if len(content.body) != expected_rows:
    raise AirflowFailException(
        f"Unexpected row count; except: {expected_rows}; got:"
        f" {len(content.body)}"
    )

  tpu_type = tpu_config.tpu_version.value

  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "Chip":
          if not re.match(r"/dev/vfio/\d+", data):
            errors.append(
                f"Unexpected {header}; except: '/dev/vfio/NNN'; got: '{data}'"
            )
        case "Type":
          if tpu_type not in data:
            errors.append(
                f"Unexpected {header}; except: string contains '{tpu_type}';"
                f" got: '{data}'"
            )
        case "PID":
          if not (data.isdigit() and int(data) > 0):
            errors.append(
                f"Unexpected {header}; except: a positive integer; got: "
                f"'{data}'"
            )

  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with {len(errors)} "
        f"error(s):\n{error_summary}\n\n"
        f"Raw table output:\n{content.raw_body}"
    )


@task
def validate_runtime_table(tpu_info_output: list[tpu_info.Table]):
  """
  Validates the row count and content of table 'TPU Runtime Utilization'
  """
  errors = []
  content = next(
      (
          table
          for table in tpu_info_output
          if table.name == "TPU Runtime Utilization"
      ),
      None,
  )

  expected_rows = 4
  if len(content.body) != expected_rows:
    raise AirflowFailException(
        f"Unexpected row count; except: {expected_rows}; got:"
        f" {len(content.body)}"
    )

  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "HBM Usage (GiB)":
          regex = re.match(r"(\d+\.\d+)\s*GiB\s*/\s*(\d+\.\d+)\s*GiB", data)
          if regex:
            used, total = float(regex.group(1)), float(regex.group(2))
            if used > total:
              errors.append(
                  f"Unexpected {header}; expect: 'used HBM <= total HBM'; got:"
                  f" '{used} GiB > {total} GiB'"
              )
          else:
            errors.append(
                f"Unexpected {header}; expect: 'N.NN GiB / N.NN GiB'; got:"
                f" '{data}'"
            )
        case "Duty cycle":
          duty_match = re.match(r"(\d+\.\d+)%", data)
          if not (duty_match and 0.0 <= float(duty_match.group(1)) <= 100.0):
            errors.append(
                f"Unexpected {header}; expect: 'a percentage between"
                f" 0.0-100.0'; got: '{data}'"
            )
  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with"
        f" {len(errors)} error(s):\n{error_summary}\n\nRaw table"
        f" output:\n{content.raw_body}"
    )


@task
def validate_tensorcore_table(tpu_info_output: list[tpu_info.Table]):
  """
  Validates the row count and content of table 'TensorCore Utilization'
  """
  errors = []
  content = next(
      (
          table
          for table in tpu_info_output
          if table.name == "TensorCore Utilization"
      ),
      None,
  )

  expected_rows = 4
  if len(content.body) != expected_rows:
    raise AirflowFailException(
        f"Unexpected row count; except: {expected_rows}; got:"
        f" {len(content.body)}"
    )
  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "TensorCore Utilization":
          util_match = re.match(r"(\d+\.\d+)%", data)
          if not (util_match and 0.0 < float(util_match.group(1)) <= 100.0):
            errors.append(
                f"Unexpected {header}; expect: 'a percentage > 0.0 and <="
                f" 100.0'; got: '{data}'"
            )
  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with"
        f" {len(errors)} error(s):\n{error_summary}\n\nRaw table"
        f" output:\n{content.raw_body}"
    )


@task
def validate_latency_table(tpu_info_output: list[tpu_info.Table]):
  """
  Validates the row count and content of table 'TPU Buffer Transfer Latency'
  """
  errors = []
  content = next(
      (
          table
          for table in tpu_info_output
          if table.name == "TPU Buffer Transfer Latency"
      ),
      None,
  )

  if content.body is None or len(content.body) == 0:
    raise AirflowFailException(
        "Unexpected row count; expects at least one data row; got: 0"
    )

  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "Buffer Size":
          continue
        case "P50" | "P90" | "P95" | "P999":
          if not (data.endswith(" us") and float(data.replace(" us", "")) > 0):
            errors.append(
                f"Unexpected {header}; expect: 'a positive float ending in \""
                f" us\"'; got: '{data}'"
            )

  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with"
        f" {len(errors)} error(s):\n{error_summary}\n\nRaw table"
        f" output:\n{content.raw_body}"
    )


def execute_tpu_info_cli_command(info, pod_name: str, tpu_args: str) -> str:
  """Helper to handle KUBECONFIG and execute kubectl."""
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- {tpu_args}",
    ])
    return subprocess.run_exec(cmd, env=env)


def verify_output_contains_patterns(
    output: str, patterns: list[str], context: str
):
  """Helper to verify expected strings in output."""
  for pattern in patterns:
    if pattern not in output:
      raise AssertionError(
          f"Validation failed for '{context}': Missing '{pattern}'."
      )


@task
def validate_help(info, pod_name: str) -> str:
  output = execute_tpu_info_cli_command(info, pod_name, "tpu-info -help")
  patterns = [
      "Display TPU info and metrics.",
      "options:",
      "-h, --help",
      "-v, --version",
      "-p, --process",
      "--streaming",
      "--rate RATE",
      "--list_metrics",
  ]
  verify_output_contains_patterns(output, patterns, "tpu-info -help")
  return output


@task
def validate_version(info, pod_name: str) -> str:
  output = execute_tpu_info_cli_command(info, pod_name, "tpu-info --version")
  patterns = ["tpu-info version:", "libtpu version:", "accelerator type:"]
  verify_output_contains_patterns(output, patterns, "tpu-info --version")
  return output


@task
def validate_process(info, pod_name: str) -> str:
  output = execute_tpu_info_cli_command(info, pod_name, "tpu-info --process")
  patterns = [
      "TPU Process Info",
      "Chip",
      "PID",
      "Process Name",
      "/dev/vfio/",
      "python",
  ]
  verify_output_contains_patterns(output, patterns, "tpu-info --process")
  return output
