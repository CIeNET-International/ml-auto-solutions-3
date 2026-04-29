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
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.timeout import timeout as AirflowTimeout
from airflow.utils.trigger_rule import TriggerRule


class TaskGroupWithTimeout(TaskGroup):
  """A TaskGroup that enforces a per-task timeout.

  Each task in the group shares a single deadline: the first task to run
  sets the deadline to `now + timeout`, and each subsequent task receives
  only the time remaining until that deadline.

  Args:
    group_id: Unique identifier for this TaskGroup.
    timeout: Timeout as a timedelta (e.g. `timedelta(minutes=30)`).
    is_teardown: When `True`, the group runs even if an upstream group
      has failed — suitable for cleanup/teardown groups (e.g. a `post_test`
      group following a `testing` group). Defaults to `False`.
    **kwargs: Additional arguments passed to TaskGroup.
  """

  ROOT_TASK_ID = "provision_taskgroup_session"

  def __init__(
      self,
      group_id,
      timeout: timedelta,
      is_teardown: bool = False,
      **kwargs,
  ):
    super().__init__(group_id=group_id, **kwargs)
    self.timeout = timeout
    self.trigger_rule = (
        TriggerRule.ALL_DONE if is_teardown else TriggerRule.ALL_SUCCESS
    )
    self._root_node = None

  def __enter__(self):
    """Enter the TaskGroup context and create the root timing task.

    Creates `_root_node`, records `datetime.now(UTC)` as an ISO-format
    string via XCom. This task serves as the *root node* of the group:
    all other tasks in the group are wired to run after it (see `__exit__`),
    so its XCom value represents the earliest possible group start time that
    every downstream task can reference.

    While `_root_node` is being constructed, `self._root_node` is still
    `None`; `.add()` uses that sentinel to skip timeout injection for the
    root node itself.
    """
    tg = super().__enter__()
    self._root_node = PythonOperator(
        task_id=self.ROOT_TASK_ID,
        python_callable=lambda: datetime.now(timezone.utc).isoformat(),
        trigger_rule=self.trigger_rule,
    )
    return tg

  def __exit__(self, *args):
    """Exit the TaskGroup context and enforce the root-node dependency.

    Wire `_root_node` as upstream of every in-group root child.
    A "root child" is a direct child with no upstream sibling within this
    group. Non-root children inherit the dependency transitively through
    their siblings, which avoids the N redundant edges that wiring every
    child directly would create.
    """
    children_ids = set(self.children.keys())
    for child in self.children.values():
      if child is self._root_node:
        continue
      # If a sibling already chains into this child, the dependency on
      # _root_node is satisfied transitively — no need to add a direct edge.
      if child.upstream_task_ids & children_ids:
        continue
      child.set_upstream(self._root_node)
    return super().__exit__(*args)

  def add(self, base_op: BaseOperator):
    node = super().add(base_op)

    if base_op.task_id.endswith(f".{self.ROOT_TASK_ID}"):
      return node
    # The node has to have the `execute` method (e.g., BaseOperator or
    # MappedOperator), or there will be nothing to intercept.
    #
    # Rationale for NOT using `isinstance(node, BaseOperator)`:
    # 1. Dynamic Task Mapping: Tasks generated via `.expand()` return a `MappedOperator`.
    # 2. Class Hierarchy: `MappedOperator` does NOT inherit from `BaseOperator`
    #    (both inherit from `AbstractOperator`). An `isinstance` check would silently
    #    skip mapped tasks, leaving them without timeout protection.
    # 3. Nested TaskGroups: `node` can be a nested `TaskGroup` (which lacks `execute`).
    # Therefore, Duck Typing (checking for a callable `execute` attribute) is the
    # most robust approach to intercept all executable nodes regardless of internal SDK changes.
    if not hasattr(node, "execute") or not callable(node.execute):
      logging.info(
          "Node %s is not an executable task (e.g., nested TaskGroup). Skipping timeout injection.",
          node,
      )
      return node
    # Use the unbound method so `self` binds at execution time, after Airflow
    # resolves XComArg placeholders. Binding via `node.execute` at parse time
    # leaks unresolved placeholders into XCom and breaks serialization.
    original_execute = type(node).execute

    group_id = self.group_id
    timeout = self.timeout
    root_node_id = self._root_node.task_id

    def wrapped_execute(context):
      task_instance = context.get("task_instance")
      start_str = task_instance.xcom_pull(task_ids=root_node_id)
      if not start_str:
        raise AirflowFailException(
            f"TaskGroup '{group_id}': no XCom value found from root node "
            f"'{root_node_id}'. Cannot determine group start time."
        )
      group_start = datetime.fromisoformat(start_str)
      deadline = group_start + timeout

      remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
      if remaining <= 0:
        raise AirflowFailException(
            f"TaskGroup '{group_id}' has already exceeded its timeout. "
            "Skipping retries."
        )
      task = task_instance.task

      # Take the minimum value as the effective timeout to ensure all tasks
      # are strictly bounded under this task group's shared deadline.
      effective_timeout_sec = min(remaining, _determine_task_timeout(task))
      logging.info(
          "TaskGroup '%s' task '%s': effective timeout=%ds",
          group_id,
          task_instance.task_id,
          effective_timeout_sec,
      )

      # Group-budget exhaustion is enforced by the `remaining <= 0` check
      # above on the next retry; let AirflowTaskTimeout propagate normally.
      with AirflowTimeout(seconds=int(effective_timeout_sec)):
        return original_execute(task, context)

    node.execute = wrapped_execute
    return node


def _determine_task_timeout(task: BaseOperator) -> float:
  """
  Determines the effective timeout for a task by identifying which limit
  triggers first.

  This method centralizes the logic for various operator types.
  - For sensors, it resolves the potential overlap between sensor-specific
    timeouts and general execution timeouts.
  - For standard operators, it takes "inf" as the value when no limit is
    set, which aligns with the API's behavior of allowing unlimited
    execution.
  """
  # Since Airflow treats an unset `execution_timeout` as unlimited,
  # we take "inf" as its value to align with this behavior
  is_set = task.execution_timeout is not None
  inf = float("inf")
  timeout_1 = task.execution_timeout.total_seconds() if is_set else inf

  if isinstance(task, BaseSensorOperator):
    # This attribute has a default value stored in the configuration file;
    # therefore, `timeout` will always be set.
    timeout_2 = task.timeout
    return min(timeout_1, timeout_2)

  return timeout_1
