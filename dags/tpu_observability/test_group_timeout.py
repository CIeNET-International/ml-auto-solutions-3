import datetime
import logging
import time
from typing import List, Optional

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowTaskTimeout
from airflow.models import TaskInstance
from airflow.operators.bash import BashOperator
from airflow.utils.session import provide_session
from airflow.utils.state import State
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

# --- 1. Utility Functions ---


def get_group_prefix(task_id: str) -> str:
  """Extracts the TaskGroup prefix from a full task_id."""
  parts = task_id.split(".")
  prefix = ".".join(parts[:-1]) + "." if len(parts) > 1 else ""
  logging.info(
      f"[UTIL] Extracted group prefix: '{prefix}' from task_id: '{task_id}'"
  )
  return prefix


def find_group_leaf_task(
    dag_obj: models.DAG, group_prefix: str, timer_id: str
) -> str:
  """Automatically finds the last task (leaf node) in the current TaskGroup."""
  logging.info(f"[UTIL] Scanning DAG for leaf tasks in group: {group_prefix}")
  group_tasks = [
      t
      for t in dag_obj.tasks
      if t.task_id.startswith(group_prefix) and t.task_id != timer_id
  ]
  leaf_task_ids = []
  for task_obj in group_tasks:
    internal_downstream = [
        d for d in task_obj.downstream_task_ids if d.startswith(group_prefix)
    ]
    if not internal_downstream:
      leaf_task_ids.append(task_obj.task_id)

  if not leaf_task_ids:
    raise ValueError(f"No target leaf task found in {group_prefix}")

  target_id = leaf_task_ids[-1]
  logging.info(f"[UTIL] Auto-detected target leaf task: {target_id}")
  return target_id


@provide_session
def get_latest_task_states(
    dag_id: str, run_id: str, task_id: str, session=None
) -> List[str]:
  """Queries the Metadata DB for the freshest states, bypassing session cache."""
  session.expire_all()
  tis = (
      session.query(TaskInstance)
      .filter(
          TaskInstance.dag_id == dag_id,
          TaskInstance.run_id == run_id,
          TaskInstance.task_id == task_id,
      )
      .all()
  )
  states = [str(ti.state) for ti in tis]
  logging.info(f"[DB] Query result for {task_id}: {states}")
  return states


@provide_session
def terminate_all_tasks_in_group(
    dag_run, group_prefix: str, timer_id: str, session=None
) -> int:
  """Forces all running or queued tasks in the group to FAILED state."""
  active_tis = dag_run.get_task_instances(
      state=[State.RUNNING, State.QUEUED], session=session
  )
  killed_count = 0
  for ti in active_tis:
    if ti.task_id.startswith(group_prefix) and ti.task_id != timer_id:
      logging.info(f"[CLEANUP] Terminating active task: {ti.task_id}")
      ti.state = State.FAILED
      session.merge(ti)
      killed_count += 1
  session.commit()
  return killed_count


# --- 2. Callback Function ---


def cleanup_group_on_failure(context) -> None:
  """Callback triggered on AirflowTaskTimeout."""
  exception = context.get("exception")
  if not (
      isinstance(exception, AirflowTaskTimeout)
      or "AirflowTaskTimeout" in str(exception)
  ):
    logging.info("[CALLBACK] Failure not caused by timeout. Skipping cleanup.")
    return

  ti = context["task_instance"]
  dag_run = context["dag_run"]
  prefix = get_group_prefix(ti.task_id)

  logging.info(f"[CALLBACK] Initiating group-wide termination for: {prefix}")
  killed_total = terminate_all_tasks_in_group(dag_run, prefix, ti.task_id)
  logging.info(f"[CALLBACK] Cleanup finished. Total killed: {killed_total}")


# --- 3. Timer Task Definition ---


@task(task_id="timer", retries=0, on_failure_callback=cleanup_group_on_failure)
def group_phase_timer(timeout_seconds: int, **context):
  """Monitors the group completion and raises timeout if threshold is met."""
  ti = context["task_instance"]
  dag_obj = context["dag"]
  dag_run = context["dag_run"]

  prefix = get_group_prefix(ti.task_id)
  target_id = find_group_leaf_task(dag_obj, prefix, ti.task_id)

  logging.info(
      f"[MONITOR] Starting loop. Target: {target_id} | Limit: {timeout_seconds}s"
  )

  elapsed = 0
  interval = 5
  while elapsed < timeout_seconds:
    states = get_latest_task_states(dag_run.dag_id, dag_run.run_id, target_id)
    if states:
      if all(s == "success" for s in states):
        logging.info(f"[MONITOR] Target {target_id} reached SUCCESS.")
        return "Phase Complete"
      if any(s in ["failed", "upstream_failed"] for s in states):
        logging.warning(f"[MONITOR] Target {target_id} failed early.")
        return "Phase Failed"

    time.sleep(interval)
    elapsed += interval

  raise AirflowTaskTimeout(f"Phase {prefix} timed out waiting for {target_id}")


# --- 4. DAG Definition ---

with models.DAG(
    dag_id="triple_group_task_timer",
    start_date=datetime.datetime(2025, 3, 1),
    schedule_interval=None,
    catchup=False,
    tags=["monitoring", "taskflow"],
) as dag:
  # --- Group 1 ---
  with TaskGroup(group_id="group_1") as tg1:
    t1_1 = BashOperator(task_id="t1_1", bash_command="sleep 5")
    t1_2 = BashOperator(task_id="t1_2", bash_command="sleep 5")
    t1_3 = BashOperator(task_id="t1_3", bash_command="echo 'G1 Done'")

    t1_1 >> t1_2 >> t1_3
    # Call the decorated task as a function
    group_phase_timer(timeout_seconds=60)

  # --- Group 2 ---
  with TaskGroup(group_id="group_2") as tg2:
    t2_1 = BashOperator(task_id="t2_1", bash_command="sleep 2")
    t2_2 = BashOperator(task_id="t2_2", bash_command="sleep 60")
    t2_3 = BashOperator(task_id="t2_3", bash_command="echo 'G2 Done'")

    t2_1 >> t2_2 >> t2_3
    group_phase_timer(timeout_seconds=15)

  # --- Group 3 ---
  with TaskGroup(group_id="group_3") as tg3:
    t3_1 = BashOperator(task_id="t3_1", bash_command="sleep 2")
    t3_2 = BashOperator(task_id="t3_2", bash_command="echo 'G3 Done'")

    t3_1 >> t3_2
    group_phase_timer(timeout_seconds=60)

  # --- Orchestration ---
  gatekeeper = BashOperator(
      task_id="gatekeeper",
      bash_command="echo 'Moving to next phase'",
      trigger_rule=TriggerRule.ALL_DONE,
  )

  tg1 >> tg2 >> gatekeeper >> tg3
