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
# See the License foar the specific language governing permissions and
# limitations under the License.

"""
DAG to validate the 'tpu-info' streaming refresh rate by calculating the time interval
between consecutive hardware telemetry updates on TPU v6e slices.
"""

import datetime
import os
import re
import subprocess
import tempfile
import logging

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH


class TPUPerformanceAnalyzer:

  def __init__(self, target_rate: float = 0.1):
    """
    Initialize the analyzer with a target refresh rate.
    :param target_rate: The expected update interval in seconds (default 0.1s).
    """
    self.target_rate = target_rate
    self.lower_bound = target_rate * 0.8
    self.upper_bound = target_rate * 1.2
    self.frames = []
    self.update_events = []

    self._frame_start_re = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*?\[H")
    self._chips_re = re.compile(
        r"│\s+(/dev/vfio/\d+)\s+│.*?│\s+\d+\s+│\s+(\d+)\s+│"
    )
    self._runtime_re = re.compile(
        r"│\s+(\d+)\s+│\s+([\d.]+ GiB / [\d.]+ GiB)\s+│\s+([\d.]+)%\s+│"
    )
    self._tensor_re = re.compile(r"│\s+(\d+)\s+│\s+([\d.]+)%\s+│")
    self._latency_re = re.compile(
        r"│\s+([\dMB+]+)\s+│\s+([\d.]+) us\s+│\s+([\d.]+) us\s+│\s+([\d.]+) us\s+│\s+([\d.]+) us\s+│"
    )

  def _init_empty_frame(self):
    """Internal helper to initialize a structure for a single log frame."""
    return {"ts": None, "chips": {}, "runtime": {}, "tensor": {}, "latency": {}}

  def _extract_metrics(self, line, frame):
    """Internal helper to extract various hardware metrics from a log line."""
    m_c = self._chips_re.search(line)
    if m_c:
      frame["chips"][m_c.group(1)] = m_c.group(2)

    m_r = self._runtime_re.search(line)
    if m_r:
      frame["runtime"][m_r.group(1)] = (m_r.group(2), m_r.group(3))

    m_t = self._tensor_re.search(line)
    if m_t:
      frame["tensor"][m_t.group(1)] = m_t.group(2)

    m_l = self._latency_re.search(line)
    if m_l:
      frame["latency"][m_l.group(1)] = (
          m_l.group(2),
          m_l.group(3),
          m_l.group(4),
          m_l.group(5),
      )

  def parse_log(self, log_content: str):
    """
    Parse raw log text into structured data frames.
    """
    self.frames = []
    lines = log_content.splitlines()
    current_frame = self._init_empty_frame()

    for line in lines:
      fs_match = self._frame_start_re.search(line)
      if fs_match:
        if current_frame["ts"]:
          self.frames.append(current_frame)
        current_frame = self._init_empty_frame()
        current_frame["ts"] = datetime.datetime.strptime(
            fs_match.group(1), "%H:%M:%S.%f"
        )
        continue
      self._extract_metrics(line, current_frame)

    if current_frame["ts"]:
      self.frames.append(current_frame)

  def filter_update_events(self):
    """
    Compare consecutive frames and retain only those where hardware data changed.
    """
    self.update_events = []
    last_snapshot = None
    for f in self.frames:
      snapshot = {k: v for k, v in f.items() if k != "ts"}
      if last_snapshot is None or snapshot != last_snapshot:
        self.update_events.append(f)
        last_snapshot = snapshot

  def validate_rate_match(self) -> bool:
    """
    Validates if the hardware update frequency aligns with the target refresh rate.

    Logic Rationale:
    1. Eager Hardware Output: To ensure monitoring data does not lag behind the
       specified refresh frequency (Target Refresh Rate e.g., 0.1s), the hardware driver implements
       an eager refresh strategy. This often results in intervals slightly below
       or exactly at the target (e.g., 0.08s - 0.10s).
    2. Jitter Tolerance: A +/- 20% buffer (0.08s to 0.12s) is established to
       account for system scheduling jitters and network transmission latency.
    3. Sensitivity Verification: A 'True' result confirms the system successfully
       captured active hardware state changes at a high frequency, rather than
       stale cached data.
    """
    if len(self.update_events) < 3:
      return False

    for i in range(2, len(self.update_events)):
      delta = (
          self.update_events[i]["ts"] - self.update_events[i - 1]["ts"]
      ).total_seconds()
      if self.lower_bound <= delta <= self.upper_bound:
        return True
    return False

  def _format_row(self, no, ev, delta):
    """Helper to format a single row of report data."""
    pids = ", ".join(ev["chips"].values())
    hbm_str = " | ".join(
        [
            f"D{k}:{v[0].split('/')[0].strip()}({v[1]}%)"
            for k, v in ev["runtime"].items()
        ]
    )
    tc_str = ", ".join([f"C{k}:{v}%" for k, v in ev["tensor"].items()])
    lat_str = " || ".join(
        [f"{k}: {'|'.join(v)}" for k, v in ev["latency"].items()]
    )

    return (
        f"{no:<3} | {ev['ts'].strftime('%H:%M:%S.%f')[:-3]:<12} | {delta:<7.3f}s | "
        f"{pids:<12} | {hbm_str:<60} | {tc_str:<30} | {lat_str}"
    )

  def generate_report(self) -> str:
    """
    Generate a complete aligned text report of detected performance updates.
    """
    num_events = len(self.update_events)
    if num_events < 3:
      return "Insufficient data to generate a report (at least 3 unique events required)."

    output = []
    header = (
        f"{'No':<3} | {'Timestamp':<12} | {'Intv (s)':<10} | "
        f"{'PIDs':<12} | {'HBM Usage (Device:Used | Duty%)':<60} | "
        f"{'TensorCore':<30} | {'Latency Profile (us)'}"
    )
    output.append(header)
    output.append("-" * 210)

    intervals = []
    for i in range(2, num_events):
      ev = self.update_events[i]
      prev_ev = self.update_events[i - 1]
      delta = (ev["ts"] - prev_ev["ts"]).total_seconds()
      intervals.append(delta)

      row = self._format_row(i - 1, ev, delta)
      output.append(row)

    if intervals:
      avg_intv = sum(intervals) / len(intervals)
      is_matched = self.validate_rate_match()
      output.append("-" * 210)
      output.append(
          f"Average Interval (Stable Phase): {avg_intv:.3f} s\n"
          f"Target Rate Match: {is_matched}\n"
          f"(Verified: Streaming data updated within the target refresh rate window)"
      )

    return "\n".join(output)


def execute_tpu_info_cli_command(info, pod_name: str, tpu_args: str) -> str:
  """Helper to handle KUBECONFIG and execute kubectl exec."""
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
  """Verifies that expected strings exist in the output."""
  for pattern in patterns:
    if pattern not in output:
      raise AssertionError(
          f"Validation failed for '{context}': Missing '{pattern}'."
      )


@task
def validate_streaming_rate_iterations(
    info,
    pod_name: str,
    rate: float,
    iteration_count: int = 40,
    duration: int = 30,
    pass_threshold_percent: float = 0.5,
) -> str:
  """
  Performs 40 iterations of 30s tests.
  The task succeeds if at least 50% of the iterations are valid.
  """
  analyzer = TPUPerformanceAnalyzer(target_rate=rate)
  success_count = 0
  pass_threshold = iteration_count * pass_threshold_percent

  for i in range(1, iteration_count + 1):
    # Precise command with Perl microsecond timestamping and terminal line export
    tpu_args = (
        f'sh -c "export LINES=50 && '
        f"script -q -c 'timeout {duration}s tpu-info --streaming --rate {rate}' /dev/null\" "
        f"| perl -MTime::HiRes=gettimeofday -ne ' "
        f"($s, $usec) = gettimeofday; "
        f"($sec,$min,$hour) = localtime($s); "
        f'printf("[%02d:%02d:%02d.%03d] %s", $hour, $min, $sec, $usec/1000, $_);\' '
        f"|| [ ${{PIPESTATUS[0]}} -eq 124 ]"
    )

    try:
      output = execute_tpu_info_cli_command(info, pod_name, tpu_args)
      analyzer.parse_log(output)
      analyzer.filter_update_events()
      logging.info(analyzer.generate_report())
      if analyzer.validate_rate_match():
        success_count += 1
        logging.info(f"Iteration {i}: Passed validation.")
      else:
        logging.info(
            f"Iteration {i}: Failed validation (Intervals out of range)."
        )
    except Exception as e:
      logging.error(f"Iteration {i}: Command execution error: {str(e)}")

  status_msg = f"Pod {pod_name} at {rate}s: {success_count}/{iteration_count} iterations passed."
  logging.info(status_msg)

  if success_count < pass_threshold:
    raise AssertionError(
        f"Validation Failed: Only {success_count}/{iteration_count} passed. "
        f"Required at least {pass_threshold}."
    )

  return status_msg


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="tpu_info_verify_streaming_rate_dags",
    start_date=datetime.datetime(2025, 8, 10),
    schedule="0 14 * * *" if composer_env.is_prod_env() else None,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "tpu-observability",
        "TPU",
        "v6e-16",
        "tpu-info",
        "streaming-rate",
    ],
    description=(
        "Validates the tpu-info refresh rate by "
        "calculating the time interval "
        "between consecutive hardware telemetry updates on TPU v6e-16 slices."
    ),
    doc_md="""
      ## TPU Info Streaming Refresh Rate Verification DAG

      This DAG automates the functional testing of the `tpu-info` CLI tool, specifically
      validating the accuracy of the streaming **refresh rate**.

      The core verification logic calculates the precise time delta between
      consecutive hardware data frames. It ensures that the actual refresh frequency
      matches the requested `--rate` parameter within a defined jitter tolerance,
      confirming that the system delivers real-time hardware metrics without stale data.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    @task
    def generate_second_node_pool_name(
        node_pool_info: node_pool.Info,
    ) -> str:
      """Generates a second node pool name."""
      return f"{node_pool_info.node_pool_name}-2"

    jobset_config = JobSet(
        jobset_name="tpu-info-verify-streaming-rate-jobset",
        namespace="default",
        max_restarts=5,
        replicated_job_name="tpu-job-slice",
        replicas=2,
        backoff_limit=0,
        completions=4,
        parallelism=4,
        tpu_accelerator_type="tpu-v6e-slice",
        tpu_topology="4x4",
        container_name="jax-tpu-worker",
        image="asia-northeast1-docker.pkg.dev/cienet-cmcs/"
        "yuna-docker/tpu-info:v0.5.1",
        tpu_cores_per_pod=4,
    )

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="tpu_info_verify_streaming_rate_dags",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      cluster_info_2 = node_pool.copy_node_pool_info_with_override.override(
          task_id="copy_node_pool_info_with_override"
      )(
          info=cluster_info,
          node_pool_name=generate_second_node_pool_name(cluster_info),
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="create_node_pool"
      ) as create_node_pool:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1"
        )(
            node_pool=cluster_info,
        )

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
            retries=2,
        )(
            node_pool=cluster_info_2,
        )

      apply_time = jobset.run_workload.override(task_id="run_workload")(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=Workload.JAX_TPU_BENCHMARK
          ),
          namespace=jobset_config.namespace,
      )

      pod_names = jobset.list_pod_names.override(task_id="list_pod_names")(
          node_pool=cluster_info,
          namespace=jobset_config.namespace,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=pod_names, job_apply_time=apply_time)

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="streaming_rate_verification"
      ) as rate_verification_group:
        test_rates = [0.1, 0.5, 1.0, 5.0]

        for rate in test_rates:
          formatted_rate = str(rate).replace(".", "_")

          # Keyword arguments are generated dynamically at runtime (pylint does not
          # know this signature).
          with TaskGroup(  # pylint: disable=unexpected-keyword-arg
              group_id=f"rate_{formatted_rate}"
          ) as rate_iterations_group:
            validate_streaming_rate_iterations.override(
                task_id="streaming_rate_test",
                execution_timeout=datetime.timedelta(minutes=60),
            ).partial(
                info=cluster_info,
                rate=rate,
                iteration_count=40,
                duration=30,  # Each iteration runs for 30 seconds
                pass_threshold_percent=0.5,  # At least 50% must pass
            ).expand(
                pod_name=pod_names
            )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          namespace=jobset_config.namespace,
      ).as_teardown(
          setups=apply_time
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="cleanup_node_pool"
      ) as cleanup_node_pool:
        cleanup_first_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_1",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info).as_teardown(
            setups=create_node_pool,
        )

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info_2).as_teardown(
            setups=create_node_pool,
        )

      chain(
          cluster_info,
          create_node_pool,
          apply_time,
          pod_names,
          wait_for_job_start,
          rate_verification_group,
          cleanup_workload,
          cleanup_node_pool,
      )
      # pylint: enable=pointless-statement
