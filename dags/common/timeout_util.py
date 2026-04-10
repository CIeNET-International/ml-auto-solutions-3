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

"""Utility class for managing automated TaskGroup timeouts and cleanups."""

import logging

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException, AirflowSensorTimeout
from airflow.models import TaskInstance
from airflow.utils.session import provide_session
from airflow.utils.state import State
from airflow.utils.task_group import TaskGroup


@provide_session
def cleanup_group_callback(context, session=None) -> None:
  """Callback for cascading cleanup upon task timeout or early failure."""
  ti = context["task_instance"]
  exception = context.get("exception")
  exception_msg = str(exception)

  # Condition check: Verify if it's a Sensor Timeout or our custom Failed Early exception
  is_timeout = (
      isinstance(exception, AirflowSensorTimeout) or "Timeout" in exception_msg
  )
  is_failed_early = "Failed Early" in exception_msg

  if not (is_timeout or is_failed_early):
    logging.info(
        f"[CALLBACK] {ti.task_id} failed, but not due to timeout or early failure. Skipping cleanup. Exception: {exception_msg}"
    )
    return

  # Retrieve target Group ID via params to eliminate the risk of hardcoded string parsing
  target_group_id = context["params"].get("target_group_id")
  if not target_group_id:
    logging.warning(
        "[CALLBACK] target_group_id not found, cannot execute cleanup."
    )
    return

  logging.info(
      f"[CALLBACK] Initiating automated cleanup for Group: {target_group_id}. Trigger cause: {'Timeout' if is_timeout else 'Failed Early'}"
  )

  # Retrieve all still-running TaskInstances within the current DAG Run
  dag_run = context["dag_run"]
  tis = dag_run.get_task_instances(
      state=[State.RUNNING, State.QUEUED, State.SCHEDULED, State.UP_FOR_RETRY],
      session=session,
  )

  count = 0
  group_prefix = f"{target_group_id}."

  for task_instance in tis:
    # Ensure the task belongs to the group and is not the monitor itself
    if (
        task_instance.task_id.startswith(group_prefix)
        and task_instance.task_id != ti.task_id
    ):
      logging.warning(
          f"[SIGTERM] Preparing to forcefully mark {task_instance.task_id} as FAILED"
      )
      # Use native API to change state, ensuring proper state machine transition
      task_instance.set_state(State.FAILED, session=session)
      count += 1

  session.commit()
  logging.info(
      f"[CALLBACK] Cleanup completed. Forcefully terminated {count} tasks in total."
  )


class TimeoutUtil:
  """Utility class focused on state querying to avoid polluting the main logic."""

  @staticmethod
  @provide_session
  def check_leaf_states(
      dag_id: str, run_id: str, leaf_task_ids: list[str], session=None
  ) -> list[str]:
    """Query the status of the leaf tasks in the Group."""
    session.expire_all()  # Ensure fetching the latest state from DB, not from cache
    tis = (
        session.query(TaskInstance.state)
        .filter(
            TaskInstance.dag_id == dag_id,
            TaskInstance.run_id == run_id,
            TaskInstance.task_id.in_(leaf_task_ids),
        )
        .all()
    )
    # SQLAlchemy returns tuples, need to extract them into a list of strings
    return [str(ti.state) for ti in tis if ti.state is not None]


@task.sensor(
    poke_interval=60,
    mode="reschedule",
    on_failure_callback=cleanup_group_callback,
)
def monitor_group_sensor(leaf_task_ids: list[str], **context):
  """
  Monitor the progress of leaf tasks.
  mode="reschedule" means if the condition is not met after each poke,
  the Worker resource will be immediately released back to the system.
  """
  dag_run = context["dag_run"]

  states = TimeoutUtil.check_leaf_states(
      dag_id=dag_run.dag_id, run_id=dag_run.run_id, leaf_task_ids=leaf_task_ids
  )

  logging.info(f"[MONITOR] Target: {leaf_task_ids} | Current states: {states}")

  if not states:
    return False  # Tasks haven't generated states yet, go back to sleep and wait for the next minute

  # Scenario 1: Pass (All successful) -> Return True, Sensor succeeds (green), Callback is not triggered
  if all(s == State.SUCCESS for s in states):
    logging.info("[MONITOR] All target tasks succeeded (Pass)!")
    return True

  # Scenario 3: Failed Early -> Actively raise a specific exception to force Sensor failure and trigger the cleanup Callback
  if any(s in [State.FAILED, State.UPSTREAM_FAILED] for s in states):
    logging.error(
        "[MONITOR] Detected early task failure within the Group, initiating Fail-Fast termination sequence."
    )
    raise AirflowException("Failed Early")

  # Scenario 2: Timeout -> Handled by the Sensor's timeout parameter.
  # As long as time is not up and conditions are unmet, return False to keep waiting;
  # once time is up, Airflow's underlying system directly raises AirflowSensorTimeout.
  return False


class TimeoutTaskGroup(TaskGroup):
  """
  TaskGroup with lifecycle monitoring capabilities.
  """

  def __init__(self, group_id, timeout_minutes, **kwargs):
    super().__init__(group_id=group_id, **kwargs)
    self.timeout_minutes = timeout_minutes

  def __exit__(self, _type, _value, _tb):
    # 1. Before adding the Sensor, fetch all leaf tasks under this Group
    leaf_tasks = self.get_leaves()
    leaf_task_ids = [t.task_id for t in leaf_tasks]

    # 2. Wrap the original TaskGroup execution
    super().__exit__(_type, _value, _tb)

    if not leaf_task_ids:
      return

    # 3. Dynamically create the Sensor and mount it under this TaskGroup as a parallel monitor
    # Note: the timeout parameter unit is in 'seconds', hence * 60
    monitor_group_sensor.override(
        task_id="task_group_timer",
        task_group=self,
        timeout=self.timeout_minutes * 60,
        params={"target_group_id": self.group_id},
    )(leaf_task_ids=leaf_task_ids)
