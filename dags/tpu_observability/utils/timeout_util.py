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

import logging
import time
from typing import List

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowTaskTimeout
from airflow.models import TaskInstance
from airflow.utils.session import provide_session
from airflow.utils.state import State

"""Utility class for managing automated TaskGroup timeouts and cleanups."""


class TimeoutUtil:
  """Utility class for managing automated TaskGroup timeouts and cleanups."""

  @staticmethod
  def get_group_prefix(task_id: str) -> str:
    """Extracts the TaskGroup prefix from a full task_id.

    Args:
      task_id: The full string ID of the task.

    Returns:
      A string representing the group prefix (e.g., 'group_1.').
    """
    parts = task_id.split(".")
    prefix = ".".join(parts[:-1]) + "." if len(parts) > 1 else ""
    return prefix

  @staticmethod
  @provide_session
  def terminate_group(
      dag_run, group_prefix: str, timer_id: str, session=None
  ) -> int:
    """Forces all active tasks in the group to FAILED state.

    Args:
      dag_run: The current DagRun object.
      group_prefix: The prefix of the TaskGroup to clean up.
      timer_id: The ID of the timer task to exclude from termination.
      session: The SQL Alchemy session.

    Returns:
      The count of terminated tasks.
    """
    active_tis = dag_run.get_task_instances(
        state=[State.RUNNING, State.QUEUED], session=session
    )
    count = 0
    for ti in active_tis:
      if ti.task_id.startswith(group_prefix) and ti.task_id != timer_id:
        logging.info(f"[CLEANUP] Killing task: {ti.task_id}")
        ti.state = State.FAILED
        session.merge(ti)
        count += 1
    session.commit()
    return count

  @staticmethod
  def cleanup_callback(context) -> None:
    """Standard Airflow callback for timeout handling.

    Note: Uses staticmethod to avoid 'classmethod not callable' TypeErrors
    during Airflow callback execution.
    """
    exception = context.get("exception")
    ti = context["task_instance"]

    if not (
        isinstance(exception, AirflowTaskTimeout)
        or "AirflowTaskTimeout" in str(exception)
    ):
      logging.info(f"[CALLBACK] {ti.task_id} failed but not due to timeout.")
      return

    prefix = TimeoutUtil.get_group_prefix(ti.task_id)
    logging.info(f"[CALLBACK] Starting automated cleanup for group: {prefix}")
    timeout_count = TimeoutUtil.terminate_group(
        context["dag_run"], prefix, ti.task_id
    )
    logging.info(
        f"[CALLBACK] Cleanup finished. {timeout_count} tasks terminated."
    )

  @staticmethod
  def find_leaf_task(
      dag_obj: models.DAG, group_prefix: str, timer_id: str
  ) -> str:
    """Automatically finds the leaf task (end of chain) in a TaskGroup."""
    group_tasks = [
        t
        for t in dag_obj.tasks
        if t.task_id.startswith(group_prefix) and t.task_id != timer_id
    ]
    leaf_ids = [
        t.task_id
        for t in group_tasks
        if not [d for d in t.downstream_task_ids if d.startswith(group_prefix)]
    ]
    if not leaf_ids:
      raise ValueError(f"No leaf task found in {group_prefix}")
    return leaf_ids[-1]

  @staticmethod
  @provide_session
  def get_fresh_states(
      dag_id: str, run_id: str, task_id: str, session=None
  ) -> List[str]:
    """Queries Metadata DB for fresh states, bypassing session cache."""
    session.expire_all()
    tis = (
        session.query(TaskInstance)
        .filter(
            TaskInstance.dag_id == dag_id,
            TaskInstance.run_id == run_id,
            TaskInstance.task_id == task_id,
        )
        .all()
    )
    return [str(ti.state) for ti in tis]

  @staticmethod
  @task(
      task_id="timer",
      retries=0,
      on_failure_callback=cleanup_callback.__func__
      if hasattr(cleanup_callback, "__func__")
      else cleanup_callback,
  )
  def monitor_group(timeout_minutes: int, **context):
    """The @task decorated monitor logic for TaskGroups."""
    ti = context["task_instance"]
    dag_obj = context["dag"]
    dag_run = context["dag_run"]

    prefix = TimeoutUtil.get_group_prefix(ti.task_id)
    target_id = TimeoutUtil.find_leaf_task(dag_obj, prefix, ti.task_id)

    logging.info(
        f"[MONITOR] Monitoring group: {prefix} via target: {target_id}"
    )

    elapsed = 0
    interval = 1
    while elapsed < timeout_minutes:
      remaining = timeout_minutes - elapsed
      progress_pct = (elapsed / timeout_minutes) * 100
      states = TimeoutUtil.get_fresh_states(
          dag_run.dag_id, dag_run.run_id, target_id
      )
      logging.info(
          f"Progress: {elapsed}min/{timeout_minutes}min \({progress_pct:.1f}%)"
          " | "
          f"Remaining: {remaining}min | Target: {target_id} | States: {states}"
      )

      if states:
        logging.info(f"{target_id} states: {states}")
        if all(s == "success" for s in states):
          return "Phase Success"
        if any(s in ["failed", "upstream_failed"] for s in states):
          return "Phase Failed Early"

      time.sleep(interval * 60)
      elapsed += interval

    logging.error(f"[MONITOR TIMEOUT] Reached limit of {timeout_minutes}min.")
    raise AirflowTaskTimeout(
        f"Group {prefix} timed out waiting for {target_id}"
    )
