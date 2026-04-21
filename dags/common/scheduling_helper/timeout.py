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
from airflow.models import TaskInstance
from airflow.utils.session import create_session
from airflow.utils.task_group import TaskGroup
from airflow.utils.timeout import timeout as AirflowTimeout


class _FailOnTimeout(AirflowTimeout):
  """Airflow timeout that marks the task as FAILED immediately.

  The default ``airflow.utils.timeout.timeout`` raises ``AirflowTaskTimeout``
  on expiry, which Airflow may retry.  This subclass raises
  ``AirflowFailException`` instead so the task stops permanently.
  """

  def __init__(self, seconds, group_id):
    super().__init__(seconds=seconds, error_message="Timeout")
    self._group_id = group_id

  def handle_timeout(self, signum, frame):
    raise AirflowFailException(
        f"TaskGroup '{self._group_id}' has exceeded its timeout "
        f"of {self.seconds}s. Skipping retries."
    )


class TaskGroupWithTimeout(TaskGroup):
  """A TaskGroup that enforces a per-task timeout.

  Each task in the group shares a single deadline: the first task to run
  sets the deadline to ``now + timeout``, and each subsequent task receives
  only the time remaining until that deadline.  A ``_FailOnTimeout`` (subclass
  of ``airflow.utils.timeout``) is used so that timeouts raise
  ``AirflowFailException`` (non-retryable / FAILED) instead of the
  default ``AirflowTaskTimeout`` (retryable / UP_FOR_RETRY).

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
      # Each TaskGroupWithTimeout has an independent deadline starting from
      # when the group's first task began executing.  Query the DB for the
      # earliest start_date among all task instances in this group so that
      # every task independently derives the same shared deadline.
      dag = context["dag"]
      run_id = context["run_id"]
      group_prefix = group_id + "."
      group_task_ids = [t for t in dag.task_ids if t.startswith(group_prefix)]

      with create_session() as session:
        earliest_ti = (
            session.query(TaskInstance)
            .filter(
                TaskInstance.dag_id == dag.dag_id,
                TaskInstance.run_id == run_id,
                TaskInstance.task_id.in_(group_task_ids),
                TaskInstance.start_date.isnot(None),
            )
            .order_by(TaskInstance.start_date)
            .first()
        )

      group_start = (
          earliest_ti.start_date
          if earliest_ti is not None
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

      current_task = context["task_instance"].task
      with _FailOnTimeout(int(remaining), group_id):
        return original_execute(current_task, context)

    task.execute = wrapped_execute
    return dag_node
