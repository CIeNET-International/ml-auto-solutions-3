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
import signal
from datetime import timedelta

from airflow.exceptions import AirflowFailException
from airflow.utils.task_group import TaskGroup


class TaskGroupWithTimeout(TaskGroup):
  """A TaskGroup that enforces a per-task timeout.

  Each task added to this group gets ``execution_timeout`` set to the
  group-level timeout.  A custom SIGALRM handler is installed so that
  timeouts raise ``AirflowFailException`` (non-retryable / FAILED)
  instead of the default ``AirflowTaskTimeout`` (retryable / UP_FOR_RETRY).

  Args:
    group_id: Unique identifier for this TaskGroup.
    timeout: Timeout as timedelta, or numeric value in minutes.
    **kwargs: Additional arguments passed to TaskGroup.

  Usage:
    with TaskGroupWithTimeout(
        group_id="testing",
        timeout=timedelta(minutes=30),
    ) as testing:
      task_a = my_task_a()
      task_b = my_task_b()
  """

  def __init__(self, group_id, timeout, **kwargs):
    super().__init__(group_id=group_id, **kwargs)
    if isinstance(timeout, (int, float)):
      timeout = timedelta(minutes=timeout)
    self.timeout = timeout

  def add(self, task):
    result = super().add(task)

    # Only wrap actual operators (with pre_execute), not sub-TaskGroups.
    if isinstance(task, TaskGroup) or not hasattr(task, "pre_execute"):
      return result

    group_timeout = self.timeout
    group_id = self.group_id

    # Pre-set execution_timeout so Airflow creates its signal-based timeout
    # wrapper BEFORE pre_execute runs.  Airflow checks this attribute to
    # decide whether to wrap execute() with ``timeout(seconds)``; if it is
    # None at that point the timeout context is never created and any value
    # set later in pre_execute has no effect.
    task.execution_timeout = group_timeout

    original_pre_execute = task.pre_execute

    def wrapped_pre_execute(context):
      ti = context["task_instance"]
      remaining = int(group_timeout.total_seconds())

      logging.info(
          "[TaskGroupWithTimeout] Task '%s' in group '%s': timeout=%ds",
          ti.task_id,
          group_id,
          remaining,
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
      signal.alarm(remaining)

      original_pre_execute(context=context)

    task.pre_execute = wrapped_pre_execute
    return result
