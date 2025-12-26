
from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
import time
from dags.common.vm_resource import XpkClusters
from dags.schedulinghelper_test.scheduling_helper import SchedulingHelper


DAG_ID = "test2"

SCHEDULE = SchedulingHelper.ArrangeScheduleTime(
    XpkClusters.TPU_V5P_128_CLUSTER,
    DAG_ID,
)

with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2025, 1, 1),
    schedule=SCHEDULE,
    catchup=False,
    dagrun_timeout=timedelta(minutes=120),
    tags=["timeout-test"],
) as dag:

  @task
  def sleep_121_minutes():

    print(
      f"schedule: {SCHEDULE}",
      f"Task: {DAG_ID} started, sleeping for 121 minutes...",
    )
    time.sleep(121 * 60)
    print("Task finished!")
    sleep_121_minutes()
