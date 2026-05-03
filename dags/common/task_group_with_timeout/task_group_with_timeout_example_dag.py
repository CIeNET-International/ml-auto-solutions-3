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

Manually triggered (`schedule=None`). The DAG contains nine independent
TaskGroups (case_1 ... case_8_teardown) that exercise different edge
cases of TaskGroupWithTimeout. Per-group descriptions live in each
group's `tooltip`, visible on hover in the Airflow Graph view.

Each group is paired with a `verify_<case>` task placed outside the
group (with `trigger_rule=ALL_DONE`). The verify task asserts every
in-group task ended in its expected state — green when the demo behaved
as designed, red when reality drifted from the spec (e.g. a task that
was expected to fail unexpectedly succeeded).

Tasks expected to fail are also marked `.as_teardown(
on_failure_fail_dagrun=False)` so their failure does not propagate to
the dagrun's overall status. Combined, the dagrun's overall status
reflects whether all `verify_<case>` tasks passed.

case_5 and case_6 are cross-product groups (sub-cases case_5_1..3 and
case_6_1..9 respectively) verifying that _determine_task_timeout picks
the minimum of (group_remaining, sensor.timeout, execution_timeout).
"unset" means the parameter is omitted; for sensor.timeout this falls
back to the BaseSensorOperator default (7 days), effectively unbounded
relative to the group budget.
"""

import datetime
import time

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.sensors.python import PythonSensor
from airflow.utils.trigger_rule import TriggerRule

from dags.common.task_group_with_timeout import TaskGroupWithTimeout


DAG_ID = "task_group_with_timeout_example_dag"


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
    step_one = sleep_for.override(task_id="step_one")(seconds=5)
    step_two = sleep_for.override(task_id="step_two")(seconds=5)
    step_three = sleep_for.override(task_id="step_three")(seconds=5)
    chain(step_one, step_two, step_three)

  verify_case_1 = verify_task_states.override(task_id="verify_case_1")(
      expected_states={
          "case_1.step_one": "success",
          "case_1.step_two": "success",
          "case_1.step_three": "success",
      }
  )
  chain(case_1, verify_case_1)

  with TaskGroupWithTimeout(
      group_id="case_2",
      tooltip=(
          "A single task sleeps far beyond the group budget "
          "(120s vs 60s) - AirflowTaskTimeout fires at ~60s."
      ),
      timeout=datetime.timedelta(seconds=60),
  ):
    long_running_task = sleep_for.override(task_id="long_running_task")(
        seconds=120
    ).as_teardown(on_failure_fail_dagrun=False)

  verify_case_2 = verify_task_states.override(task_id="verify_case_2")(
      expected_states={
          "case_2.long_running_task": "failed",
      }
  )
  chain(long_running_task, verify_case_2)

  with TaskGroupWithTimeout(
      group_id="case_3",
      tooltip=(
          "Shared deadline across a longer chain - five sequential tasks "
          "(5+8+10+12+30=65s sleep) against a 60s budget. The first four "
          "steps succeed; step_5's remaining budget is too small for its "
          "30s sleep, so AirflowTaskTimeout interrupts it. (Sleep totals "
          "are kept comfortably under 60s to leave headroom for Composer "
          "scheduling overhead between tasks.)"
      ),
      timeout=datetime.timedelta(seconds=60),
  ):
    step_1 = sleep_for.override(task_id="step_1")(seconds=5)
    step_2 = sleep_for.override(task_id="step_2")(seconds=8)
    step_3 = sleep_for.override(task_id="step_3")(seconds=10)
    step_4 = sleep_for.override(task_id="step_4")(seconds=12)
    step_5 = sleep_for.override(task_id="step_5")(seconds=30).as_teardown(
        on_failure_fail_dagrun=False
    )
    chain(step_1, step_2, step_3, step_4, step_5)

  verify_case_3 = verify_task_states.override(task_id="verify_case_3")(
      expected_states={
          "case_3.step_1": "success",
          "case_3.step_2": "success",
          "case_3.step_3": "success",
          "case_3.step_4": "success",
          "case_3.step_5": "failed",
      }
  )
  chain(step_5, verify_case_3)

  with TaskGroupWithTimeout(
      group_id="case_4",
      tooltip=(
          "task_group_timeout vs sensor retry. short_task uses 15s of the "
          "60s budget, leaving ~45s for the sensor. Each sensor attempt "
          "times out at sensor.timeout=20s; with retries=3 (4 total "
          "attempts), successive retries chip away at the remaining "
          "budget. By the third retry the budget is gone, so "
          "wrapped_execute's `remaining<=0` guard fires immediately and "
          "the task is marked FAILED."
      ),
      timeout=datetime.timedelta(seconds=60),
  ):
    short_task = sleep_for.override(task_id="short_task")(seconds=15)
    sensor_with_retries = PythonSensor(
        task_id="sensor_with_retries",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=20,
        retries=3,
        retry_delay=datetime.timedelta(seconds=1),
    ).as_teardown(on_failure_fail_dagrun=False)
    chain(short_task, sensor_with_retries)

  verify_case_4 = verify_task_states.override(task_id="verify_case_4")(
      expected_states={
          "case_4.short_task": "success",
          "case_4.sensor_with_retries": "failed",
      }
  )
  chain(sensor_with_retries, verify_case_4)

  with TaskGroupWithTimeout(
      group_id="case_5",
      tooltip=(
          "Cross product of execution_timeout vs a 60s group budget "
          "(case_5_1=above, case_5_2=below, case_5_3=unset). All three "
          "tasks sleep 120s and time out; whichever limit (group or "
          "execution_timeout) is tighter wins."
      ),
      timeout=datetime.timedelta(seconds=60),
  ):
    case_5_1 = sleep_for.override(
        task_id="case_5_1_exec_above_group",
        execution_timeout=datetime.timedelta(seconds=120),
    )(seconds=120).as_teardown(on_failure_fail_dagrun=False)
    case_5_2 = sleep_for.override(
        task_id="case_5_2_exec_below_group",
        execution_timeout=datetime.timedelta(seconds=30),
    )(seconds=120).as_teardown(on_failure_fail_dagrun=False)
    case_5_3 = sleep_for.override(task_id="case_5_3_exec_unset")(
        seconds=120
    ).as_teardown(on_failure_fail_dagrun=False)

  verify_case_5 = verify_task_states.override(task_id="verify_case_5")(
      expected_states={
          "case_5.case_5_1_exec_above_group": "failed",
          "case_5.case_5_2_exec_below_group": "failed",
          "case_5.case_5_3_exec_unset": "failed",
      }
  )
  chain([case_5_1, case_5_2, case_5_3], verify_case_5)

  with TaskGroupWithTimeout(
      group_id="case_6",
      tooltip=(
          "Cross product of sensor.timeout x execution_timeout vs a 60s "
          "group budget (case_6_1..case_6_9). Expected trip points: "
          "6-1 ~60s (group), 6-2 ~30s (exec), 6-3 ~30s (sensor), "
          "6-4 ~20s (exec), 6-5 ~60s (group), 6-6 ~30s (exec), "
          "6-7 ~60s (group), 6-8 ~30s (sensor), 6-9 ~60s (group)."
      ),
      timeout=datetime.timedelta(seconds=60),
  ):
    case_6_1 = PythonSensor(
        task_id="case_6_1_sensor_above_exec_above",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=120,
        execution_timeout=datetime.timedelta(seconds=120),
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_2 = PythonSensor(
        task_id="case_6_2_sensor_above_exec_below",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=120,
        execution_timeout=datetime.timedelta(seconds=30),
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_3 = PythonSensor(
        task_id="case_6_3_sensor_below_exec_above",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=30,
        execution_timeout=datetime.timedelta(seconds=120),
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_4 = PythonSensor(
        task_id="case_6_4_sensor_below_exec_below",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=30,
        execution_timeout=datetime.timedelta(seconds=20),
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_5 = PythonSensor(
        task_id="case_6_5_sensor_unset_exec_above",
        python_callable=_never_satisfied,
        poke_interval=5,
        execution_timeout=datetime.timedelta(seconds=120),
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_6 = PythonSensor(
        task_id="case_6_6_sensor_unset_exec_below",
        python_callable=_never_satisfied,
        poke_interval=5,
        execution_timeout=datetime.timedelta(seconds=30),
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_7 = PythonSensor(
        task_id="case_6_7_sensor_above_exec_unset",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=120,
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_8 = PythonSensor(
        task_id="case_6_8_sensor_below_exec_unset",
        python_callable=_never_satisfied,
        poke_interval=5,
        timeout=30,
    ).as_teardown(on_failure_fail_dagrun=False)
    case_6_9 = PythonSensor(
        task_id="case_6_9_both_unset",
        python_callable=_never_satisfied,
        poke_interval=5,
    ).as_teardown(on_failure_fail_dagrun=False)

  verify_case_6 = verify_task_states.override(task_id="verify_case_6")(
      expected_states={
          "case_6.case_6_1_sensor_above_exec_above": "failed",
          "case_6.case_6_2_sensor_above_exec_below": "failed",
          "case_6.case_6_3_sensor_below_exec_above": "failed",
          "case_6.case_6_4_sensor_below_exec_below": "failed",
          "case_6.case_6_5_sensor_unset_exec_above": "failed",
          "case_6.case_6_6_sensor_unset_exec_below": "failed",
          "case_6.case_6_7_sensor_above_exec_unset": "failed",
          "case_6.case_6_8_sensor_below_exec_unset": "failed",
          "case_6.case_6_9_both_unset": "failed",
      }
  )
  chain(
      [
          case_6_1,
          case_6_2,
          case_6_3,
          case_6_4,
          case_6_5,
          case_6_6,
          case_6_7,
          case_6_8,
          case_6_9,
      ],
      verify_case_6,
  )

  with TaskGroupWithTimeout(
      group_id="case_7",
      tooltip=(
          "Dependency-edge reduction. Graph view should show that "
          "_root_node connects only to chain_head and parallel_root; "
          "chain_middle and chain_tail reach the root transitively via "
          "chain_head."
      ),
      timeout=datetime.timedelta(minutes=5),
  ) as case_7:
    chain_head = noop.override(task_id="chain_head")()
    chain_middle = noop.override(task_id="chain_middle")()
    chain_tail = noop.override(task_id="chain_tail")()
    noop.override(task_id="parallel_root")()
    chain(chain_head, chain_middle, chain_tail)

  verify_case_7 = verify_task_states.override(task_id="verify_case_7")(
      expected_states={
          "case_7.chain_head": "success",
          "case_7.chain_middle": "success",
          "case_7.chain_tail": "success",
          "case_7.parallel_root": "success",
      }
  )
  chain(case_7, verify_case_7)

  with TaskGroupWithTimeout(
      group_id="case_8_main",
      tooltip="Main phase fails (failing_task raises immediately).",
      timeout=datetime.timedelta(minutes=2),
  ) as case_8_main:
    failing_task = raise_workload_failure.override(
        task_id="failing_task"
    )().as_teardown(on_failure_fail_dagrun=False)
  with TaskGroupWithTimeout(
      group_id="case_8_teardown",
      tooltip=(
          "Teardown phase (is_teardown=True) runs cleanup even when "
          "case_8_main fails."
      ),
      timeout=datetime.timedelta(minutes=2),
      is_teardown=True,
  ) as case_8_teardown:
    noop.override(task_id="cleanup_task")()
  chain(case_8_main, case_8_teardown)

  verify_case_8_main = verify_task_states.override(
      task_id="verify_case_8_main"
  )(
      expected_states={
          "case_8_main.failing_task": "failed",
      }
  )
  chain(failing_task, verify_case_8_main)

  verify_case_8_teardown = verify_task_states.override(
      task_id="verify_case_8_teardown"
  )(
      expected_states={
          "case_8_teardown.cleanup_task": "success",
      }
  )
  chain(case_8_teardown, verify_case_8_teardown)
