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

"""TaskGroupWithTimeout: shared deadline enforcement for Airflow TaskGroups."""

import logging
import signal
from datetime import datetime, timedelta, timezone

from airflow.exceptions import AirflowFailException
from airflow.utils.task_group import TaskGroup


class TaskGroupWithTimeout(TaskGroup):
  """A TaskGroup that enforces a shared deadline timeout across all tasks.

  All tasks added to this group share a common deadline. The first task to
  execute establishes the deadline (now + timeout). Subsequent tasks compute
  remaining time from the shared deadline and dynamically set Airflow's native
  ``execution_timeout`` so the framework handles the timeout.

  The hook is installed at interpretation (DAG parsing) stage via the
  overridden ``add()`` method.  At execution stage, the ``pre_execute`` hook
  computes the remaining budget and sets ``execution_timeout``; Airflow then
  wraps ``execute()`` with its built-in ``airflow.utils.timeout.timeout``.

  This approach deliberately avoids monkey-patching ``execute()``, because
  Airflow 2.10+ passes a sentinel kwarg to ``execute()`` based on
  ``callable.__name__ == "execute"``.  Replacing the method would break
  the sentinel check and, for ``@task``-decorated operators, prevent
  XComArg resolution inside ``_PythonDecoratedOperator.execute()``.

  Args:
    group_id: Unique identifier for this TaskGroup.
    timeout: Group-level timeout as timedelta, or numeric value in minutes.
    **kwargs: Additional arguments passed to TaskGroup.

  Usage:
    with TaskGroupWithTimeout(
        group_id="testing",
        timeout=timedelta(minutes=30),
    ) as testing:
      task_a = my_task_a()
      task_b = my_task_b()
  """

  _DEADLINE_XCOM_KEY_PREFIX = "__tg_deadline__"

  def __init__(self, group_id, timeout, **kwargs):
    super().__init__(group_id=group_id, **kwargs)
    if isinstance(timeout, (int, float)):
      timeout = timedelta(minutes=timeout)
    self.timeout = timeout
    self._wrapped_task_ids = []

  @property
  def _deadline_xcom_key(self):
    return f"{self._DEADLINE_XCOM_KEY_PREFIX}{self.group_id}"

  def add(self, task):
    result = super().add(task)

    # Only wrap actual operators (with pre_execute), not sub-TaskGroups.
    if isinstance(task, TaskGroup) or not hasattr(task, "pre_execute"):
      return result

    group_timeout = self.timeout
    group_id = self.group_id
    deadline_key = self._deadline_xcom_key
    wrapped_ids = self._wrapped_task_ids

    self._wrapped_task_ids.append(task.task_id)

    # Pre-set execution_timeout so Airflow creates its signal-based timeout
    # wrapper BEFORE pre_execute runs.  Airflow checks this attribute to
    # decide whether to wrap execute() with ``timeout(seconds)``; if it is
    # None at that point the timeout context is never created and any value
    # set later in pre_execute has no effect.
    task.execution_timeout = group_timeout

    original_pre_execute = task.pre_execute

    def wrapped_pre_execute(context):
      ti = context["task_instance"]
      now = datetime.now(timezone.utc)

      # Look for an existing deadline set by a sibling task in this group.
      deadline = None
      for sibling_id in wrapped_ids:
        if sibling_id == ti.task_id:
          continue
        val = ti.xcom_pull(task_ids=sibling_id, key=deadline_key)
        if val is not None:
          deadline = datetime.fromisoformat(val)
          break

      if deadline is None:
        # First task in the group: establish the shared deadline.
        deadline = now + group_timeout

      # Publish deadline for subsequent sibling tasks to discover.
      ti.xcom_push(key=deadline_key, value=deadline.isoformat())

      remaining = (deadline - now).total_seconds()
      logging.info(
          "[TaskGroupWithTimeout] Task '%s' in group '%s': "
          "deadline=%s, remaining=%ds",
          ti.task_id,
          group_id,
          deadline.isoformat(),
          int(remaining),
      )

      if remaining <= 0:
        raise AirflowFailException(
            f"TaskGroup '{group_id}' has exceeded its timeout "
            f"of {group_timeout}. Skipping retries."
        )

      # Replace Airflow's SIGALRM handler with one that raises
      # AirflowFailException (non-retryable) instead of the default
      # AirflowTaskTimeout (retryable).  Then re-arm the alarm to fire
      # at the actual remaining budget.
      def _handle_group_timeout(signum, frame):
        raise AirflowFailException(
            f"TaskGroup '{group_id}' has exceeded its timeout "
            f"of {group_timeout}. Skipping retries."
        )

      signal.signal(signal.SIGALRM, _handle_group_timeout)
      task.execution_timeout = timedelta(seconds=int(remaining))
      signal.alarm(int(remaining))

      original_pre_execute(context=context)

    task.pre_execute = wrapped_pre_execute
    return result
