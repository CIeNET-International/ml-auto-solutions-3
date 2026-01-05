from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime as dt

def task_fn(name):
  print(f"Running {name}")

with DAG(
    dag_id="overtime_dag_1",
    start_date=dt.datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=dt.timedelta(hours=23),
) as dag1:
  t1 = PythonOperator(
      task_id="task1",
      python_callable=lambda: task_fn("dag_one"),
  )

with DAG(
    dag_id="overtime_dag_2",
    start_date=dt.datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=dt.timedelta(hours=3),
) as dag2:
  t2 = PythonOperator(
      task_id="task2",
      python_callable=lambda: task_fn("dag_two"),
  )
