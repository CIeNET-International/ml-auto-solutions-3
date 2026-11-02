# Copyright 2026 Google LLC
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

"""Fault-injection and log-verification utilities for Elastic Snapshot DAGs."""

from collections.abc import Iterable
from datetime import datetime, timezone
import re
import tempfile
import time

from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowException, AirflowFailException
from airflow.models.baseoperator import chain
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup

from dags.maxtext_pathways.configs.utils import (
    _get_kubeconfig_env,
    check_logs_stream,
    interrupt_worker_pod,
    wait_for_running_head_pod,
)
from xlml.utils import subprocess_utils

_SAVE_SNAPSHOT_RE = re.compile(r"Saving in-memory snapshot at step (\d+)\.\.\.")
_SCALE_UP_SNAPSHOT_RE = re.compile(
    r"New slices are available\. Snapshotting step (\d+) before scaling up\."
)
_RESTORE_OK_RE = re.compile(r"Restored the in-memory snapshot at step (\d+)\.")
_RESTORE_FAIL_RE = re.compile(
    r"Restoring the in-memory snapshot at step (\d+) failed:"
)
_FALLBACK_RE = re.compile(
    r"In-memory snapshot restore failed; falling back to checkpoint restore\."
)
_CKPT_RESTORE_RE = re.compile(r"restoring from this run's directory step (\d+)")
_ITERATOR_RESTORE_RE = re.compile(
    r"Restored the data iterator state: (\{[^}]*\})\."
)
_GLOBAL_NEXT_INDEX_RE = re.compile(r"['\"]global_next_index['\"]\s*:\s*(\d+)")
_COMPLETED_STEP_RE = re.compile(r"completed step:\s*(\d+)\b")


def stream_until_snapshot_ready(
    lines: Iterable[str],
    *,
    required_live_slices: int,
    interrupt_after_snapshot_step: int,
    post_snapshot_steps: int = 3,
    timeout: int = 1800,
) -> str:
  """Consumes head-pod log lines until slice count, snapshot, and step match."""
  required_slices = max(1, int(required_live_slices))
  saw_required_slices = False
  target_snap_step: int | None = None
  deadline = time.time() + timeout

  for raw_line in lines:
    if time.time() > deadline:
      raise AirflowException(
          f"Timeout after {timeout}s waiting for snapshot fault-injection"
          " trigger."
      )
    clean_line = raw_line.strip()
    if not clean_line:
      continue
    logging.info("[pod-log] %s", clean_line)

    if (
        f"live slice count: {required_slices}" in clean_line
        or f"Sufficient slices active: {required_slices} >=" in clean_line
    ):
      saw_required_slices = True

    if saw_required_slices and target_snap_step is None:
      if match := _SAVE_SNAPSHOT_RE.search(clean_line):
        snap_step = int(match.group(1))
        if snap_step >= interrupt_after_snapshot_step:
          target_snap_step = snap_step
          logging.info(
              "Matched periodic snapshot at step %d (required slices=%d).",
              snap_step,
              required_slices,
          )

    if target_snap_step is not None:
      if match := _COMPLETED_STEP_RE.search(clean_line):
        step_num = int(match.group(1))
        if step_num >= target_snap_step + post_snapshot_steps:
          logging.info(
              "Matched post-snapshot completed step %d; injecting SIGILL.",
              step_num,
          )
          if "Z " in clean_line:
            return clean_line.split("Z ")[0] + "Z"
          return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

  if not saw_required_slices:
    raise AirflowFailException(
        f"Head pod log stream ended before reaching {required_slices} active"
        " TPU slice(s); verify cluster TPU node pool capacity."
    )
  if target_snap_step is None:
    raise AirflowFailException(
        "Head pod log stream ended before periodic snapshot at step >="
        f" {interrupt_after_snapshot_step}."
    )
  raise AirflowFailException(
      "Head pod log stream ended before completed step >="
      f" {target_snap_step + post_snapshot_steps}."
  )


@task
def wait_and_interrupt_worker(
    project_id: str,
    region: str,
    cluster_name: str,
    workload_id: str,
    required_live_slices: int,
    interrupt_after_snapshot_step: int,
    post_snapshot_steps: int = 3,
) -> str:
  """Waits for live slices + periodic snapshot, then sends SIGILL."""
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = _get_kubeconfig_env(
        project_id, region, cluster_name, kubeconfig_path=temp_config_file.name
    )
    head_pod, container_name = wait_for_running_head_pod(workload_id, env=env)
    pod_name = head_pod["metadata"]["name"]
    namespace = head_pod["metadata"].get("namespace", "default")
    cmd = f"kubectl logs -f {pod_name} -n {namespace} --timestamps=true"
    if container_name:
      cmd += f" -c {container_name}"

    process = subprocess_utils.run_streaming(cmd, env=env)
    try:
      pre_interrupt_ts = stream_until_snapshot_ready(
          iter(process.stdout.readline, ""),
          required_live_slices=int(required_live_slices),
          interrupt_after_snapshot_step=interrupt_after_snapshot_step,
          post_snapshot_steps=post_snapshot_steps,
      )
    finally:
      process.kill()
      process.stdout.close()
      process.wait()

  interrupt_worker_pod.function(
      workload_id=workload_id,
      cluster_name=cluster_name,
      region=region,
      project_id=project_id,
  )
  return pre_interrupt_ts


def snapshot_fault_injection(
    project_id: str,
    region: str,
    cluster_name: str,
    workload_id: str,
    required_live_slices: int,
    interrupt_after_snapshot_step: int,
    post_snapshot_steps: int,
    post_interrupt_checks: tuple[tuple[str, str], ...],
) -> DAGNode:
  """Builds the live log-streaming fault-injection task group for a scenario."""
  common = {
      "project_id": project_id,
      "region": region,
      "cluster_name": cluster_name,
      "workload_id": workload_id,
  }
  with TaskGroup(group_id="snapshot_fault_injection") as group:
    interrupt = wait_and_interrupt_worker.override(
        task_id="wait_and_interrupt_worker"
    )(
        **common,
        required_live_slices=required_live_slices,
        interrupt_after_snapshot_step=interrupt_after_snapshot_step,
        post_snapshot_steps=post_snapshot_steps,
    )
    tasks = [interrupt]
    prev = interrupt
    for check_id, pattern in post_interrupt_checks:
      prev = check_logs_stream.override(task_id=check_id)(
          **common, expect_log_contains=pattern, since_time=prev
      )
      tasks.append(prev)
    chain(*tasks)
  return group


def validate_snapshot_log_text(
    logs_text: str,
    *,
    expect_fallback: bool,
    expect_scale_up: bool,
    expected_min_restores: int,
    expect_data_iterator: bool,
) -> dict:
  """Validates snapshot save/restore, step continuity, and iterator logs."""
  if not (logs_text and logs_text.strip()):
    raise AirflowFailException("Head pod logs are empty.")

  saved_steps: list[int] = []
  scale_up_steps: list[int] = []
  restored_steps: list[int] = []
  failed_steps: list[int] = []
  ckpt_steps: list[int] = []
  iterator_indices: list[int] = []
  completed_steps: list[int] = []
  fallback_count = 0
  pending_scale_up: int | None = None
  pending_restore: tuple[str, int] | None = None

  for line in logs_text.splitlines():
    if match := _SAVE_SNAPSHOT_RE.search(line):
      saved_steps.append(int(match.group(1)))
    if match := _SCALE_UP_SNAPSHOT_RE.search(line):
      pending_scale_up = int(match.group(1))
      scale_up_steps.append(pending_scale_up)
    if match := _RESTORE_FAIL_RE.search(line):
      failed_steps.append(int(match.group(1)))
    if _FALLBACK_RE.search(line):
      fallback_count += 1
    if match := _CKPT_RESTORE_RE.search(line):
      ckpt_step = int(match.group(1))
      ckpt_steps.append(ckpt_step)
      pending_restore = ("checkpoint", ckpt_step)
    if match := _ITERATOR_RESTORE_RE.search(line):
      idx_match = _GLOBAL_NEXT_INDEX_RE.search(match.group(1))
      if idx_match is None:
        raise AirflowFailException(
            f"Missing integer 'global_next_index' in {match.group(1)}"
        )
      iterator_indices.append(int(idx_match.group(1)))
    if match := _RESTORE_OK_RE.search(line):
      restored_step = int(match.group(1))
      restored_steps.append(restored_step)
      pending_restore = ("snapshot", restored_step)
      if pending_scale_up is not None:
        if restored_step != pending_scale_up:
          raise AirflowFailException(
              f"Scale-up restore step mismatch: snapshotted {pending_scale_up},"
              f" restored {restored_step}."
          )
        pending_scale_up = None
    if match := _COMPLETED_STEP_RE.search(line):
      step_num = int(match.group(1))
      completed_steps.append(step_num)
      if pending_restore is not None:
        kind, r_step = pending_restore
        if step_num != r_step + 1:
          raise AirflowFailException(
              f"Step continuity violation after {kind} restore at {r_step}:"
              f" expected {r_step + 1}, got {step_num}."
          )
        pending_restore = None

  if not any(step > 0 for step in saved_steps):
    raise AirflowFailException(
        f"Expected periodic snapshot at step > 0; got {saved_steps}."
    )
  if expect_fallback:
    if not (failed_steps and fallback_count > 0 and ckpt_steps):
      raise AirflowFailException(
          f"Expected 1-slice snapshot failure and checkpoint fallback; got"
          f" failed={failed_steps}, fallbacks={fallback_count},"
          f" ckpts={ckpt_steps}."
      )
  else:
    if failed_steps or fallback_count > 0 or ckpt_steps:
      raise AirflowFailException(
          f"Unexpected snapshot failure or checkpoint fallback:"
          f" failed={failed_steps}, fallbacks={fallback_count},"
          f" ckpts={ckpt_steps}."
      )
    positive_restores = [step for step in restored_steps if step > 0]
    if len(positive_restores) < expected_min_restores:
      raise AirflowFailException(
          f"Expected >={expected_min_restores} snapshot restores at step > 0;"
          f" got {restored_steps}."
      )
    if expect_scale_up and (not scale_up_steps or pending_scale_up is not None):
      raise AirflowFailException(
          f"Missing pre-scale-up snapshot or matching restore:"
          f" {scale_up_steps}."
      )
    if expect_data_iterator:
      if len(iterator_indices) < expected_min_restores or not all(
          idx > 0 for idx in iterator_indices
      ):
        raise AirflowFailException(
            f"Expected >={expected_min_restores} positive ElasticIterator"
            f" restores; got {iterator_indices}."
        )
      if not all(
          nxt >= prv for prv, nxt in zip(iterator_indices, iterator_indices[1:])
      ):
        raise AirflowFailException(
            f"ElasticIterator global_next_index regressed: {iterator_indices}."
        )

  if pending_restore is not None:
    raise AirflowFailException(
        f"No completed training step observed after restore: {pending_restore}."
    )
  return {
      "saved_snapshot_steps": saved_steps,
      "scale_up_snapshot_steps": scale_up_steps,
      "restored_snapshot_steps": restored_steps,
      "restored_iterator_indices": iterator_indices,
      "fallback_checkpoint_steps": ckpt_steps,
      "completed_steps_count": len(completed_steps),
  }


@task
def verify_snapshot_logs(
    project_id: str,
    region: str,
    cluster_name: str,
    workload_id: str,
    expect_fallback: bool,
    expect_scale_up: bool,
    expected_min_restores: int,
    expect_data_iterator: bool,
) -> dict:
  """Fetches Pathways head pod logs via kubectl and verifies invariants."""
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = _get_kubeconfig_env(
        project_id, region, cluster_name, kubeconfig_path=temp_config_file.name
    )
    head_pod, container_name = wait_for_running_head_pod(workload_id, env=env)
    pod_name = head_pod["metadata"]["name"]
    namespace = head_pod["metadata"].get("namespace", "default")
    cmd = f"kubectl logs {pod_name} -n {namespace}"
    if container_name:
      cmd += f" -c {container_name}"
    logs_text = subprocess_utils.run_exec(cmd, env=env, log_output=False)

  summary = validate_snapshot_log_text(
      logs_text,
      expect_fallback=expect_fallback,
      expect_scale_up=expect_scale_up,
      expected_min_restores=expected_min_restores,
      expect_data_iterator=expect_data_iterator,
  )
  logging.info("Verified elastic snapshot log summary: %s", summary)
  return summary
