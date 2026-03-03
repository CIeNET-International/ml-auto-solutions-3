import datetime
from airflow import models
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

# Importing your validated utility class
from dags.tpu_observability.utils.timeout_util import TimeoutUtil

"""Expanded DAG testing complex TaskGroup dependencies and automated timeouts."""

with models.DAG(
    dag_id="multi_phase_tpu_stress_test",
    start_date=datetime.datetime(2025, 3, 1),
    schedule_interval=None,
    catchup=False,
    tags=["google_style", "tpu_observability", "production_test"],
) as dag:
  # --- Group 1: Linear Chain (Longer) ---
  with TaskGroup(group_id="setup_phase") as tg1:
    t1_1 = BashOperator(task_id="fetch_config", bash_command="sleep 2")
    t1_2 = BashOperator(task_id="validate_env", bash_command="sleep 2")
    t1_3 = BashOperator(task_id="provision_nodes", bash_command="sleep 5")
    t1_4 = BashOperator(
        task_id="check_connectivity", bash_command="echo 'Setup OK'"
    )

    # Automated timer: Will detect t1_4 as the leaf node
    timer = TimeoutUtil.monitor_group(timeout_seconds=60)

    t1_1 >> t1_2 >> t1_3 >> t1_4
    [t1_1, timer]

  # --- Group 2: Failure Scenario (Mid-chain stall) ---
  with TaskGroup(group_id="training_phase") as tg2:
    t2_1 = BashOperator(task_id="load_dataset", bash_command="sleep 2")
    t2_2 = BashOperator(
        task_id="train_model", bash_command="sleep 300"
    )  # This will stall
    t2_3 = BashOperator(task_id="export_model", bash_command="echo 'Exported'")

    # Tight timeout: will kill t2_2 at 20 seconds
    timer = TimeoutUtil.monitor_group(timeout_seconds=20)

    t2_1 >> t2_2 >> t2_3
    [t2_1, timer]

  # Inter-phase Gatekeeper
  gatekeeper_1 = BashOperator(
      task_id="gatekeeper_1",
      bash_command="echo 'Syncing states...'",
      trigger_rule=TriggerRule.ALL_DONE,
  )

  # --- Group 3: Complex Branching (Parallel Leaf detection) ---
  with TaskGroup(group_id="validation_phase") as tg3:
    t3_start = BashOperator(task_id="start_eval", bash_command="sleep 2")

    # Parallel evaluation paths
    t3_a = BashOperator(task_id="accuracy_metric", bash_command="sleep 5")
    t3_b = BashOperator(task_id="latency_metric", bash_command="sleep 5")

    # Leaf node that merges results
    t3_final = BashOperator(
        task_id="finalize_metrics", bash_command="echo 'Validation Done'"
    )

    # Timer: Must identify t3_final as the single leaf node
    timer = TimeoutUtil.monitor_group(timeout_seconds=60)

    t3_start >> [t3_a, t3_b] >> t3_final
    [t3_start, timer]

  # Final Gatekeeper
  gatekeeper_final = BashOperator(
      task_id="gatekeeper_final",
      bash_command="echo 'All phases finished or cleaned up.'",
      trigger_rule=TriggerRule.ALL_DONE,
  )

  # Final Orchestration
  tg1 >> gatekeeper_1 >> tg2 >> gatekeeper_final >> tg3
