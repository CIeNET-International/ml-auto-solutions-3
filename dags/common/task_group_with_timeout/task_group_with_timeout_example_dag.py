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

"""Example DAG demonstrating TaskGroupWithTimeout edge cases.

Manually triggered (`schedule=None`). Each TaskGroup below is an isolated
"island" — they share no upstream/downstream edges, so a single DAG trigger
will fan out and exercise every scenario in parallel.

All group budgets are set to at least 60s so that Composer scheduling/queue
overhead does not eat into the visible budget. Within each case, the task
durations are kept clearly apart from the budget (no near-boundary values)
so the expected outcome is unambiguous.
"""

import datetime
import time

from airflow import models
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

from dags.common.task_group_with_timeout import TaskGroupWithTimeout


DAG_ID = "task_group_with_timeout_example_dag"


def _sleep_for(seconds: float):
  """Sleep helper used by example tasks to simulate workload duration."""
  time.sleep(seconds)


def _raise_workload_failure():
  """Task body that unconditionally raises to simulate a workload failure."""
  raise AirflowFailException("simulated workload failure")


def _never_satisfied() -> bool:
  """Sensor poke callable that never resolves, forcing the sensor to time out."""
  return False


with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 0},
    tags=["task_group_with_timeout", "example"],
) as dag:
  # case 1: every task finishes well within the group budget
  # (5s + 5s + 5s = 15s vs 120s budget).
  with TaskGroupWithTimeout(
      group_id="case_within_budget",
      timeout=datetime.timedelta(minutes=2),
  ):
    step_one = PythonOperator(
        task_id="step_one", python_callable=_sleep_for, op_args=[5]
    )
    step_two = PythonOperator(
        task_id="step_two", python_callable=_sleep_for, op_args=[5]
    )
    step_three = PythonOperator(
        task_id="step_three", python_callable=_sleep_for, op_args=[5]
    )
    chain(step_one, step_two, step_three)

  # case 2: a single task sleeps far beyond the group budget
  # (120s sleep vs 60s budget) — AirflowTaskTimeout fires after ~60s.
  with TaskGroupWithTimeout(
      group_id="case_single_task_exceeds_budget",
      timeout=datetime.timedelta(seconds=60),
  ):
    PythonOperator(
        task_id="long_running_task",
        python_callable=_sleep_for,
        op_args=[120],
    )

  # case 3: shared deadline — `consume_partial_budget` (30s sleep) succeeds and
  # uses up half of the 60s budget; `consume_remaining_budget` (60s sleep)
  # only has ~30s left and is interrupted by AirflowTaskTimeout.
  with TaskGroupWithTimeout(
      group_id="case_cumulative_runtime_exhausts_budget",
      timeout=datetime.timedelta(seconds=60),
  ):
    consume_partial_budget = PythonOperator(
        task_id="consume_partial_budget",
        python_callable=_sleep_for,
        op_args=[30],
    )
    consume_remaining_budget = PythonOperator(
        task_id="consume_remaining_budget",
        python_callable=_sleep_for,
        op_args=[60],
    )
    chain(consume_partial_budget, consume_remaining_budget)

  # case 4: once `budget_consumer` (90s sleep) exhausts the 60s budget,
  # `task_with_retries` raises AirflowFailException with `timeout exceeded`
  # immediately. Its `retries=3` is intentionally configured to demonstrate
  # that retries are *not* consumed when the group budget is already gone.
  with TaskGroupWithTimeout(
      group_id="case_retry_skipped_after_exhaustion",
      timeout=datetime.timedelta(seconds=60),
  ):
    budget_consumer = PythonOperator(
        task_id="budget_consumer",
        python_callable=_sleep_for,
        op_args=[90],
    )
    task_with_retries = PythonOperator(
        task_id="task_with_retries",
        python_callable=_raise_workload_failure,
        retries=3,
        retry_delay=datetime.timedelta(seconds=1),
    )
    chain(budget_consumer, task_with_retries)

  # case 5: task-level `execution_timeout` (60s) is tighter than the group's
  # 10min remaining budget, so the task-level limit wins. The task sleeps
  # 120s and is interrupted at ~60s.
  with TaskGroupWithTimeout(
      group_id="case_task_timeout_takes_priority",
      timeout=datetime.timedelta(minutes=10),
  ):
    PythonOperator(
        task_id="task_with_explicit_timeout",
        python_callable=_sleep_for,
        op_args=[120],
        execution_timeout=datetime.timedelta(seconds=60),
    )

  # case 6: a sensor's own `timeout=180s` is clamped to the group's remaining
  # ~60s, so the sensor trips at ~60s instead of 180s.
  with TaskGroupWithTimeout(
      group_id="case_sensor_timeout_constrained_by_group",
      timeout=datetime.timedelta(seconds=60),
  ):
    PythonSensor(
        task_id="never_satisfied_sensor",
        python_callable=_never_satisfied,
        poke_interval=10,
        timeout=180,
    )

  # case 7: dependency-edge reduction — Graph view should show that
  # `_root_node` connects only to `chain_head` and `parallel_root`;
  # `chain_middle` and `chain_tail` reach the root transitively via
  # `chain_head`.
  with TaskGroupWithTimeout(
      group_id="case_dependency_edge_reduction",
      timeout=datetime.timedelta(minutes=5),
  ):
    chain_head = PythonOperator(
        task_id="chain_head", python_callable=lambda: None
    )
    chain_middle = PythonOperator(
        task_id="chain_middle", python_callable=lambda: None
    )
    chain_tail = PythonOperator(
        task_id="chain_tail", python_callable=lambda: None
    )
    parallel_root = PythonOperator(
        task_id="parallel_root", python_callable=lambda: None
    )
    chain(chain_head, chain_middle, chain_tail)

  # case 8: is_teardown — main phase fails, teardown phase still runs.
  with TaskGroupWithTimeout(
      group_id="case_main_phase_failure",
      timeout=datetime.timedelta(minutes=2),
  ) as main_phase:
    PythonOperator(
        task_id="failing_task", python_callable=_raise_workload_failure
    )
  with TaskGroupWithTimeout(
      group_id="case_teardown_phase_runs_anyway",
      timeout=datetime.timedelta(minutes=2),
      is_teardown=True,
  ) as teardown_phase:
    PythonOperator(task_id="cleanup_task", python_callable=lambda: None)
  chain(main_phase, teardown_phase)
