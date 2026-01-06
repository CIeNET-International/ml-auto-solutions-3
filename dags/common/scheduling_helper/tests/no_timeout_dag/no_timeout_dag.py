from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime as dt


def task_fn():
  print(f"Running")


with DAG(
    dag_id="no_timeout_dag",
    start_date=dt.datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag1:
  t1 = PythonOperator(
      task_id="task1",
      python_callable=task_fn,
  )
