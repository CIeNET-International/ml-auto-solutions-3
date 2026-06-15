"""A integration DAG to test the behavior of TaskGroupWithTimeout."""

import datetime
from datetime import timedelta
import logging
import time

from airflow import models
from airflow.models.baseoperator import chain
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags.common.scheduling_helper.scheduling_helper import (
    SchedulingHelper,
    get_dag_timeout,
)
from dags.common.task_group_with_timeout.task_group_with_timeout import (
    TaskGroupWithTimeout,
)

DAG_ID = "test_task_group_with_timeout"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)


def simulate_workload(sleep_seconds: int):
  """Simulates a long-running workload with explicit sleep."""
  logging.info(f"Executing workload... Sleeping for {sleep_seconds}s.")
  time.sleep(sleep_seconds)
  logging.info("Workload completed successfully.")


# Test 1: Active enforcement against nested TaskGroups.
try:
  with TaskGroupWithTimeout(
      group_id="isolated_nested_test", timeout=timedelta(seconds=60)
  ):
    with TaskGroup(group_id="nested_native_group"):
      pass
except AirflowException as e:
  print(f"Standalone Parsing Check - Caught expected nested error: {e}")

# Test 2: Active enforcement against Dynamic Task Mapping inside the group.
try:
  with TaskGroupWithTimeout(
      group_id="isolated_mapping_test", timeout=timedelta(seconds=60)
  ):
    PythonOperator.partial(
        task_id="dummy_map", python_callable=lambda: None
    ).expand(op_args=[[]])
except AirflowException as e:
  print(f"Standalone Parsing Check - Caught expected mapping error: {e}")

# Runtime Timeout Tests
with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 6, 1),
    schedule=None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=["integration-test", "timeout-validation"],
    description=(
        "Validates continuous time-budget pool depletion across consecutive "
        "TaskGroups and tests teardown resource protection safeguards."
    ),
    doc_md="""
      # TaskGroupWithTimeout Integration Test Suite

      ### Description
      This DAG serves as a production-grade sandbox to verify the dynamic runtime
      capabilities of the custom `TaskGroupWithTimeout` infrastructure component.

      ### Procedures
      1. **Case 1 (Normal Flow):** Consumes 60 seconds from the global pool and succeeds.
      2. **Case 2 (Timeout Flow):** Dynamically inherits the remaining budget. Forces `t3`
         to fail due to budget exhaustion.
      3. **Teardown & Final Status:** Ensures external cleanup runs and forces the final
         DAG status to SUCCESS via an ALL_DONE end-node.
    """,
) as dag:
  # Case 1: Normal Flow
  with TaskGroupWithTimeout(
      group_id="case1_normal_flow", timeout=timedelta(seconds=100)
  ) as case1:
    c1_t1 = PythonOperator(
        task_id="t1", python_callable=simulate_workload, op_args=[20]
    )
    c1_t2 = PythonOperator(
        task_id="t2", python_callable=simulate_workload, op_args=[20]
    )
    c1_t3 = PythonOperator(
        task_id="t3", python_callable=simulate_workload, op_args=[20]
    )

    chain(c1_t1, c1_t2, c1_t3)

  # Case 2: Timeout Block & Teardown Bypass
  with TaskGroupWithTimeout(
      group_id="case2_block_and_teardown", timeout=timedelta(seconds=100)
  ) as case2:
    c2_t1 = PythonOperator(
        task_id="t1", python_callable=simulate_workload, op_args=[20]
    )
    c2_t2 = PythonOperator(
        task_id="t2", python_callable=simulate_workload, op_args=[40]
    )
    c2_t3 = PythonOperator(
        task_id="t3", python_callable=simulate_workload, op_args=[60], retries=0
    )

  c2_teardown = PythonOperator(
      task_id="cleanup", python_callable=simulate_workload, op_args=[5]
  ).as_teardown()

  dag_final_success = PythonOperator(
      task_id="dag_final_success",
      python_callable=lambda: logging.info(
          "Test pipeline finished. Overriding final status to SUCCESS."
      ),
      trigger_rule=TriggerRule.ALL_DONE,
  )

  chain(c2_t1, c2_t2, c2_t3, c2_teardown)
  chain(case1, case2, dag_final_success)
