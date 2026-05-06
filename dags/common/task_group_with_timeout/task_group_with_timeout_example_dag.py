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

Manually triggered (`schedule=None`). The DAG contains TaskGroups
(case_1 ... case_8_teardown, with case_5 split into three sub-cases
case_5_1..case_5_3 and case_6 split into nine sub-cases
case_6_1..case_6_9) that exercise different edge cases of
TaskGroupWithTimeout. Per-group descriptions live in each group's
`tooltip`, visible on hover in the Airflow Graph view.

Tasks register their expected outcome (PASS/FAIL) into the module-level
`validate_dict` via `gen_task`. A single `verify_task_states` task at
the end of the DAG (with `trigger_rule=ALL_DONE`) reads the dict and
asserts every task ended in its expected state — green when the demo
behaved as designed, red when reality drifted from the spec.

Tasks expected to fail are marked `.as_teardown(
on_failure_fail_dagrun=False)` so their failure does not propagate to
the dagrun's overall status. Combined with the single verification task,
the dagrun is green iff every demo behaved as designed.

case_5_1..case_5_3 and case_6_1..case_6_9 are independent groups
verifying _determine_task_timeout picks the minimum of
(group_remaining, sensor.timeout, execution_timeout). "unset" means
the parameter is omitted; for sensor.timeout this falls back to the
BaseSensorOperator default (7 days), effectively unbounded relative
to the group budget.
"""

import datetime
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.models.xcom_arg import XComArg
from airflow.sensors.python import PythonSensor
from airflow.utils.trigger_rule import TriggerRule

from dags.common.task_group_with_timeout import TaskGroupWithTimeout


DAG_ID = "task_group_with_timeout_example_dag"


class TaskRun(Enum):
  PASS = auto()
  FAIL = auto()


# global shared, to reduce args
validate_dict = {}


def gen_task(expect: TaskRun, op: Callable, **op_kwargs) -> Any:
  """Build a task and register its expected PASS/FAIL outcome."""
  task_obj = op(**op_kwargs)
  task_id = (
      task_obj.operator.task_id
      if isinstance(task_obj, XComArg)
      else task_obj.task_id
  )
  validate_dict[task_id] = expect
  return task_obj


@task
def sleep_for(seconds: int):
  """Task body that sleeps for the given number of seconds."""
  time.sleep(seconds)


@task
def raise_workload_failure():
  """Task body that unconditionally raises to simulate a workload failure."""
  raise AirflowFailException("simulated workload failure")


@task
def noop():
  """Task body that does nothing; used as a placeholder."""


@task(trigger_rule=TriggerRule.ALL_DONE)
def verify_task_states(expected_states: dict, dag_run=None):
  """Assert each task ended in its expected state.

  expected_states maps a fully-qualified task_id (group-prefixed) to one
  of the TaskInstanceState string values, e.g. 'success' or 'failed'.
  """
  mismatches = []
  for task_id, expected in expected_states.items():
    ti = dag_run.get_task_instance(task_id)
    actual = ti.state if ti else "<missing>"
    if actual != expected:
      mismatches.append(f"{task_id}: expected={expected!r}, actual={actual!r}")
  if mismatches:
    raise AirflowFailException(
        "Task state assertion failed:\n  " + "\n  ".join(mismatches)
    )


def _never_satisfied() -> bool:
  """Sensor poke callable that never resolves, forcing the sensor time out"""
  return False


with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 0},
    tags=["task_group_with_timeout", "example"],
) as dag:
  with TaskGroupWithTimeout(
      group_id="case_1",
      tooltip=(
          "Tasks finish well within the group budget (5+5+5=15s vs 120s)."
      ),
      timeout=datetime.timedelta(minutes=2),
  ) as case_1:
    step_one = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=5)
    step_two = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=5)
    step_three = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=5)
    chain(step_one, step_two, step_three)

  with TaskGroupWithTimeout(
      group_id="case_2",
      tooltip=(
          "A single task sleeps far beyond the group budget "
          "(120s vs 60s) - AirflowTaskTimeout fires at ~60s."
      ),
      timeout=datetime.timedelta(seconds=60),
  ) as case_2:
    gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for,
        seconds=120,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_3",
      tooltip=(
          "Shared deadline across a longer chain - five sequential tasks "
          "(5+8+10+12+30=65s sleep) against a 60s budget. The first four "
          "steps succeed; the last step's remaining budget is too small "
          "for its 30s sleep, so AirflowTaskTimeout interrupts it. "
          "(Sleep totals are kept comfortably under 60s to leave headroom "
          "for Composer scheduling overhead between tasks.)"
      ),
      timeout=datetime.timedelta(seconds=60),
  ) as case_3:
    step_1 = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=5)
    step_2 = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=8)
    step_3 = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=10)
    step_4 = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=12)
    step_5 = gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for,
        seconds=30,
    ).as_teardown(on_failure_fail_dagrun=False)
    chain(step_1, step_2, step_3, step_4, step_5)

  with TaskGroupWithTimeout(
      group_id="case_4",
      tooltip=(
          "task_group_timeout vs sensor retry. The first task uses 15s "
          "of the 60s budget, leaving ~45s for sensor_with_retries. Each "
          "sensor attempt times out at sensor.timeout=20s; with retries=3 "
          "(4 total attempts), successive retries chip away at the "
          "remaining budget. By the third retry the budget is gone, so "
          "wrapped_execute's `remaining<=0` guard fires immediately and "
          "the task is marked FAILED."
      ),
      timeout=datetime.timedelta(seconds=60),
  ) as case_4:
    short_task = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=15)
    sensor_with_retries = gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="sensor_with_retries",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=20,
        retries=3,
        retry_delay=datetime.timedelta(seconds=1),
    ).as_teardown(on_failure_fail_dagrun=False)
    chain(short_task, sensor_with_retries)

  with TaskGroupWithTimeout(
      group_id="case_5_1",
      tooltip="exec=120, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_5_1:
    gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for.override(
            execution_timeout=datetime.timedelta(seconds=120)
        ),
        seconds=120,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_5_2",
      tooltip="exec=30, group=60 -> exec wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_5_2:
    gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for.override(execution_timeout=datetime.timedelta(seconds=30)),
        seconds=120,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_5_3",
      tooltip="exec=unset, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_5_3:
    gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for,
        seconds=120,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_1",
      tooltip="sensor=120, exec=120, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_1:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=120,
        execution_timeout=datetime.timedelta(seconds=120),
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_2",
      tooltip="sensor=120, exec=30, group=60 -> exec wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_2:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=120,
        execution_timeout=datetime.timedelta(seconds=30),
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_3",
      tooltip="sensor=30, exec=120, group=60 -> sensor wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_3:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=30,
        execution_timeout=datetime.timedelta(seconds=120),
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_4",
      tooltip="sensor=30, exec=20, group=60 -> exec wins (~20s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_4:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=30,
        execution_timeout=datetime.timedelta(seconds=20),
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_5",
      tooltip="sensor=unset, exec=120, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_5:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        execution_timeout=datetime.timedelta(seconds=120),
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_6",
      tooltip="sensor=unset, exec=30, group=60 -> exec wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_6:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        execution_timeout=datetime.timedelta(seconds=30),
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_7",
      tooltip="sensor=120, exec=unset, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_7:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=120,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_8",
      tooltip="sensor=30, exec=unset, group=60 -> sensor wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_8:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=30,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_6_9",
      tooltip="sensor=unset, exec=unset, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_9:
    gen_task(
        expect=TaskRun.FAIL,
        op=PythonSensor,
        task_id="subject_sensor",
        python_callable=_never_satisfied,
        poke_interval=5,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_7",
      tooltip=(
          "Dependency-edge reduction. Graph view should show that "
          "_root_node connects only to the entry tasks of the group "
          "(the chain root and the parallel branch); the rest of the "
          "chain reaches _root_node transitively."
      ),
      timeout=datetime.timedelta(minutes=5),
  ) as case_7:
    chain_head = gen_task(expect=TaskRun.PASS, op=noop)
    chain_middle = gen_task(expect=TaskRun.PASS, op=noop)
    chain_tail = gen_task(expect=TaskRun.PASS, op=noop)
    gen_task(expect=TaskRun.PASS, op=noop)
    chain(chain_head, chain_middle, chain_tail)

  with TaskGroupWithTimeout(
      group_id="case_8_main",
      tooltip="Main phase fails immediately (the inner task raises).",
      timeout=datetime.timedelta(minutes=2),
  ) as case_8_main:
    gen_task(
        expect=TaskRun.FAIL,
        op=raise_workload_failure,
    ).as_teardown(on_failure_fail_dagrun=False)

  with TaskGroupWithTimeout(
      group_id="case_8_teardown",
      tooltip=(
          "Teardown phase (is_teardown=True) runs cleanup even when "
          "case_8_main fails."
      ),
      timeout=datetime.timedelta(minutes=2),
      is_teardown=True,
  ) as case_8_teardown:
    gen_task(expect=TaskRun.PASS, op=noop)

  chain(case_8_main, case_8_teardown)

  validate = verify_task_states(
      expected_states={
          task_id: ("success" if expect is TaskRun.PASS else "failed")
          for task_id, expect in validate_dict.items()
      }
  )
  chain([dag.task_dict[tid] for tid in validate_dict], validate)
