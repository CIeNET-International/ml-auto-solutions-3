"""Test Dag."""

from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime as dt


def task_fn():
  print("Running")


with DAG(
    dag_id="normal_dag_1",
    start_date=dt.datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=dt.timedelta(hours=1),
) as dag1:
  t1 = PythonOperator(
      task_id="task1",
      python_callable=task_fn,
  )

with DAG(
    dag_id="normal_dag_2",
    start_date=dt.datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=dt.timedelta(hours=1),
) as dag2:
  t2 = PythonOperator(
      task_id="task2",
      python_callable=task_fn,
  )
