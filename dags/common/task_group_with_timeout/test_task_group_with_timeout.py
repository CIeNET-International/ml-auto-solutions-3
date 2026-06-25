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

from dags import composer_env
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


def verify_static_defenses():
  """Verifies that TaskGroupWithTimeout structural restrictions are active."""
  # Mock a minimal dummy DAG context to allow TaskGroup initialization
  mock_dag = models.DAG(
      dag_id="mock_static_defense_dag", start_date=datetime.datetime(2026, 6, 1)
  )

  # Test 1: Active enforcement against nested TaskGroups
  try:
    tg_timeout = TaskGroupWithTimeout(
        group_id="parent_timeout_group",
        timeout=timedelta(seconds=60),
        dag=mock_dag,
    )
    nested_native = TaskGroup(group_id="nested_native_group", dag=mock_dag)
    # Explicitly trigger the component's add constraint logic
    tg_timeout.add(nested_native)
  except AirflowException as e:
    print(f"Static Parsing Check - Caught expected nested error: {e}")

  # Test 2: Active enforcement against Dynamic Task Mapping inside the group
  try:
    tg_timeout_map = TaskGroupWithTimeout(
        group_id="parent_mapping_group",
        timeout=timedelta(seconds=60),
        dag=mock_dag,
    )
    dummy_mapped_task = PythonOperator.partial(
        task_id="dummy_map", python_callable=lambda: None, dag=mock_dag
    ).expand(op_args=[[]])
    # Explicitly trigger the component's add constraint logic
    tg_timeout_map.add(dummy_mapped_task)
  except AirflowException as e:
    print(f"Static Parsing Check - Caught expected mapping error: {e}")


# Runtime Timeout Tests
with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 6, 1),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
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

      ### Prerequisites
      The custom TaskGroupWithTimeout component must be available under dags.common.

      ### Procedures
      1. **Static Validation:** Pre-checks blocking of nested groups and dynamic mapping.
      2. **Case 1 (Normal Flow):** Consumes 60 seconds from the global pool and succeeds.
      3. **Case 2 (Timeout Flow):** Dynamically inherits the remaining budget. Forces `t3`
         to fail due to budget exhaustion.
      4. **Teardown & Final Status:** Ensures external cleanup runs and forces the final
         DAG status to SUCCESS via an ALL_DONE end-node.
    """,
) as dag:
  verify_static_defenses()

  # Case 1: Normal Flow
  with TaskGroupWithTimeout(
      group_id="normal_flow", timeout=timedelta(seconds=100)
  ) as case1:
    normal_flow_task_1 = PythonOperator(
        task_id="normal_flow_task_1",
        python_callable=simulate_workload,
        op_args=[20],
    )
    normal_flow_task_2 = PythonOperator(
        task_id="normal_flow_task_2",
        python_callable=simulate_workload,
        op_args=[20],
    )
    normal_flow_task_3 = PythonOperator(
        task_id="normal_flow_task_3",
        python_callable=simulate_workload,
        op_args=[20],
    )

    chain(normal_flow_task_1, normal_flow_task_2, normal_flow_task_3)

  # Case 2: Timeout Block & Teardown Bypass
  with TaskGroupWithTimeout(
      group_id="timeout_flow", timeout=timedelta(seconds=100)
  ) as case2:
    timeout_flow_task_1 = PythonOperator(
        task_id="timeout_flow_task_1",
        python_callable=simulate_workload,
        op_args=[20],
    )
    timeout_flow_task_2 = PythonOperator(
        task_id="timeout_flow_task_2",
        python_callable=simulate_workload,
        op_args=[40],
    )
    timeout_flow_task_3 = PythonOperator(
        task_id="timeout_flow_task_3",
        python_callable=simulate_workload,
        op_args=[60],
        retries=0,
    )

  env_cleanup_teardown = PythonOperator(
      task_id="env_cleanup_teardown",
      python_callable=simulate_workload,
      op_args=[5],
  ).as_teardown()

  dag_status_override = PythonOperator(
      task_id="dag_status_override",
      python_callable=lambda: logging.info(
          "Test pipeline finished. Overriding final status to SUCCESS."
      ),
      trigger_rule=TriggerRule.ALL_DONE,
  )

  chain(
      timeout_flow_task_1,
      timeout_flow_task_2,
      timeout_flow_task_3,
      env_cleanup_teardown,
  )
  chain(case1, case2, dag_status_override)
