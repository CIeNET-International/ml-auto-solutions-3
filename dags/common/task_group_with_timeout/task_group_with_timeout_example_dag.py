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

"""Example DAG demonstrating TaskGroupWithTimeout edge cases."""

import datetime
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
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
  validate_dict[task_obj.operator.task_id] = expect
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


@task.sensor(poke_interval=5)
def never_satisfied() -> bool:
  """Sensor that never resolves; used to force timeout."""
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
    step_1 = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=5)
    step_2 = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=5)
    step_3 = gen_task(expect=TaskRun.PASS, op=sleep_for, seconds=5)
    chain(step_1, step_2, step_3)

  with TaskGroupWithTimeout(
      group_id="case_2",
      tooltip=(
          "A single task sleeps far beyond the group budget "
          "(120s vs 60s) - AirflowTaskTimeout fires at ~60s."
      ),
      timeout=datetime.timedelta(seconds=60),
  ) as case_2:
    gen_task(expect=TaskRun.FAIL, op=sleep_for, seconds=120)

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
    step_5 = gen_task(expect=TaskRun.FAIL, op=sleep_for, seconds=30)
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
        op=never_satisfied.override(
            timeout=20,
            retries=3,
            retry_delay=datetime.timedelta(seconds=1),
        ),
    )
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
    )

  with TaskGroupWithTimeout(
      group_id="case_5_2",
      tooltip="exec=30, group=60 -> exec wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_5_2:
    gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for.override(execution_timeout=datetime.timedelta(seconds=30)),
        seconds=120,
    )

  with TaskGroupWithTimeout(
      group_id="case_5_3",
      tooltip="exec=unset, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_5_3:
    gen_task(expect=TaskRun.FAIL, op=sleep_for, seconds=120)

  with TaskGroupWithTimeout(
      group_id="case_6_1",
      tooltip="sensor=120, exec=120, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_1:
    subject = gen_task(
        expect=TaskRun.FAIL,
        op=never_satisfied.override(
            timeout=120,
            execution_timeout=datetime.timedelta(seconds=120),
        ),
    )
    probe = gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_2",
      tooltip="sensor=120, exec=30, group=60 -> exec wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_2:
    subject = gen_task(
        expect=TaskRun.FAIL,
        op=never_satisfied.override(
            timeout=120,
            execution_timeout=datetime.timedelta(seconds=30),
        ),
    )
    probe = gen_task(
        expect=TaskRun.PASS,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_3",
      tooltip="sensor=30, exec=120, group=60 -> sensor wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_3:
    subject = gen_task(
        expect=TaskRun.FAIL,
        op=never_satisfied.override(
            timeout=30,
            execution_timeout=datetime.timedelta(seconds=120),
        ),
    )
    probe = gen_task(
        expect=TaskRun.PASS,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_4",
      tooltip="sensor=30, exec=20, group=60 -> exec wins (~20s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_4:
    subject = gen_task(
        expect=TaskRun.FAIL,
        op=never_satisfied.override(
            timeout=30,
            execution_timeout=datetime.timedelta(seconds=20),
        ),
    )
    probe = gen_task(
        expect=TaskRun.PASS,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_5",
      tooltip="sensor=unset, exec=120, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_5:
    subject = gen_task(
        expect=TaskRun.FAIL,
        op=never_satisfied.override(
            execution_timeout=datetime.timedelta(seconds=120)
        ),
    )
    probe = gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_6",
      tooltip="sensor=unset, exec=30, group=60 -> exec wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_6:
    subject = gen_task(
        expect=TaskRun.FAIL,
        op=never_satisfied.override(
            execution_timeout=datetime.timedelta(seconds=30)
        ),
    )
    probe = gen_task(
        expect=TaskRun.PASS,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_7",
      tooltip="sensor=120, exec=unset, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_7:
    subject = gen_task(
        expect=TaskRun.FAIL, op=never_satisfied.override(timeout=120)
    )
    probe = gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_8",
      tooltip="sensor=30, exec=unset, group=60 -> sensor wins (~30s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_8:
    subject = gen_task(
        expect=TaskRun.FAIL, op=never_satisfied.override(timeout=30)
    )
    probe = gen_task(
        expect=TaskRun.PASS,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

  with TaskGroupWithTimeout(
      group_id="case_6_9",
      tooltip="sensor=unset, exec=unset, group=60 -> group wins (~60s).",
      timeout=datetime.timedelta(seconds=60),
  ) as case_6_9:
    subject = gen_task(expect=TaskRun.FAIL, op=never_satisfied)
    probe = gen_task(
        expect=TaskRun.FAIL,
        op=sleep_for.override(trigger_rule=TriggerRule.ALL_DONE),
        seconds=2,
    )
    chain(subject, probe)

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
    gen_task(expect=TaskRun.FAIL, op=raise_workload_failure)

  with TaskGroupWithTimeout(
      group_id="case_8_teardown",
      tooltip=(
          "Teardown phase (is_teardown=True) runs cleanup even when "
          "the preceding group fails."
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
