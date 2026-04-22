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
from airflow.utils.task_group import TaskGroup
from airflow.utils.timeout import timeout as AirflowTimeout


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

  def add(self, task):
    dag_node = super().add(task)

    # Only wrap actual operators (with execute), not sub-TaskGroups.
    if isinstance(task, TaskGroup) or not hasattr(task, "execute"):
      return dag_node

    group_id = self.group_id
    timeout = self.timeout
    original_execute = type(task).execute

    def wrapped_execute(context):
      dag = context["dag"]
      ti = context["task_instance"]
      xcom_key = f"{group_id}.start"
      group_task_ids = [t for t in dag.task_ids if t.startswith(group_id + ".")]

      raw = ti.xcom_pull(task_ids=group_task_ids, key=xcom_key)
      results = list(raw) if raw is not None else []
      start_str = next(
          (v for v in results if isinstance(v, (str, datetime))), None
      )
      if start_str is None:
        group_start = datetime.now(timezone.utc)
        ti.xcom_push(key=xcom_key, value=group_start.isoformat())
      else:
        group_start = (
            start_str
            if isinstance(start_str, datetime)
            else datetime.fromisoformat(start_str)
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

      current_task = context["task_instance"].task
      with AirflowTimeout(seconds=int(remaining)):
        return original_execute(current_task, context)

    task.execute = wrapped_execute
    return dag_node
