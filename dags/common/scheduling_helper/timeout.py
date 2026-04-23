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

"""TaskGroupWithTimeout: timeout enforcement for Airflow TaskGroups."""

import logging
from datetime import datetime, timedelta, timezone

from airflow.exceptions import AirflowFailException
from airflow.models import BaseOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.timeout import timeout as AirflowTimeout

_START_TASK_ID = "_timeout_start"


class TaskGroupWithTimeout(TaskGroup):
  """A TaskGroup that enforces a per-task timeout.

  Each task in the group shares a single deadline: the first task to run
  sets the deadline to ``now + timeout``, and each subsequent task receives
  only the time remaining until that deadline.

  Args:
    group_id: Unique identifier for this TaskGroup.
    timeout: Timeout as a timedelta (e.g. ``timedelta(minutes=30)``).
    **kwargs: Additional arguments passed to TaskGroup.

  Usage:
    with TaskGroupWithTimeout(
        group_id="testing",
        timeout=timedelta(minutes=30),
    ) as testing:
      task_a = my_task_a()
      task_b = my_task_b()
  """

  def __init__(self, group_id, timeout: timedelta, **kwargs):
    super().__init__(group_id=group_id, **kwargs)
    self.timeout = timeout
    self._start_task = None
    self._creating_start_task = False

  def __enter__(self):
    tg = super().__enter__()
    self._creating_start_task = True
    self._start_task = PythonOperator(
        task_id=_START_TASK_ID,
        python_callable=lambda: datetime.now(timezone.utc).isoformat(),
    )
    self._creating_start_task = False
    return tg

  def __exit__(self, *args):
    for child in self.children.values():
      if child is not self._start_task:
        child.set_upstream(self._start_task)
    return super().__exit__(*args)

  def add(self, base_op: BaseOperator):
    node = super().add(base_op)

    if self._creating_start_task:
      return node
    if not hasattr(node, "execute") or not callable(node.execute):
      logging.info(
          "Node %s is not an executable task (e.g., nested TaskGroup). Skipping timeout injection.",
          node,
      )
      return node
    original_execute = type(node).execute

    group_id = self.group_id
    timeout = self.timeout
    start_task = self._start_task

    def wrapped_execute(context):
      task_instance = context.get("task_instance")
      start_str = task_instance.xcom_pull(task_ids=start_task.task_id)
      group_start = (
          datetime.fromisoformat(start_str)
          if start_str
          else datetime.now(timezone.utc)
      )
      deadline = group_start + timeout

      remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
      logging.info(
          "TaskGroup '%s' deadline: group_start=%s, deadline=%s, remaining=%.1fs",
          group_id,
          group_start.isoformat(),
          deadline.isoformat(),
          remaining,
      )
      if remaining <= 0:
        raise AirflowFailException(
            f"TaskGroup '{group_id}' has already exceeded its timeout. "
            "Skipping retries."
        )

      with AirflowTimeout(seconds=int(remaining)):
        return original_execute(task_instance.task, context)

    node.execute = wrapped_execute
    return node
