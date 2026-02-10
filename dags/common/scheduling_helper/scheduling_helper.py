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

import datetime as dt
import enum


class Dag:
  """
  Represents an Airflow DAG with its identification and expected runtime.
  """

  dag_id: str
  dag_run_timeout: dt.timedelta

  def __init__(
      self,
      dag_id: str,
      dag_run_timeout: dt.timedelta = dt.timedelta(minutes=60),
  ):
    self.dag_id = dag_id
    self.dag_run_timeout = dag_run_timeout


class DayOfWeek(enum.Enum):
  ALL = "*"
  WEEK_DAY = "1-5"
  WEEKEND = "0,6"


class Cluster:
  tpu_obs_prod = "tpu-observability-automation-prod"


class SchedulingHelper:
  """
  Helper class to automate the calculation of non-overlapping cron schedules.
  """

  DEFAULT_MARGIN = dt.timedelta(minutes=15)
  DEFAULT_ANCHOR = dt.datetime(2000, 1, 1, 13, 0, 0, tzinfo=dt.timezone.utc)

  # The registry defines which DAGs run in which clusters and their expected runtimes.
  registry: dict[Cluster, list[Dag]] = {
      Cluster.tpu_obs_prod: [
          Dag("gke_node_pool_label_update", dt.timedelta(minutes=30)),
          Dag("gke_node_pool_status", dt.timedelta(minutes=30)),
          Dag("jobset_rollback_ttr", dt.timedelta(minutes=90)),
          Dag("jobset_ttr_node_pool_resize", dt.timedelta(minutes=90)),
          Dag("jobset_ttr_pod_delete", dt.timedelta(minutes=90)),
          Dag("multi-host-availability-rollback", dt.timedelta(minutes=30)),
          Dag("node_pool_ttr_disk_size", dt.timedelta(minutes=90)),
          Dag("node_pool_ttr_update_label", dt.timedelta(minutes=90)),
          Dag("tpu_info_format_validation_dag", dt.timedelta(minutes=30)),
          Dag("tpu_sdk_monitoring_validation", dt.timedelta(minutes=30)),
          Dag("jobset_ttr_kill_process", dt.timedelta(minutes=90)),
      ],
  }

  @classmethod
  def ArrangeScheduleTime(
      cls,
      cluster: Cluster,
      dag_id: str,
      day_of_week: DayOfWeek = DayOfWeek.ALL,
  ) -> str:
    if cluster not in cls.registry:
      raise ValueError(f"Cluster {cluster} is not registered.")

    dags_in_cluster = cls.registry[cluster]
    anchor = cls.DEFAULT_ANCHOR
    offset = dt.timedelta(0)

    for dag in dags_in_cluster:
      if dag_id == dag.dag_id:
        schedule_time = anchor + offset
        return f"{schedule_time.minute} {schedule_time.hour} * * {day_of_week.value}"

      offset += dag.dag_run_timeout + cls.DEFAULT_MARGIN

      if offset >= dt.timedelta(hours=24):
        raise ValueError(f"Schedule exceeds 24h window at {dag_id}.")

    raise ValueError(f"DAG '{dag_id}' not found in registry.")


if __name__ == "__main__":
  cluster = Cluster.tpu_obs_prod
  print(f"{'DAG ID':<35} | {'Schedule (Cron)':<15} | {'Start Time (UTC)'}")
  print("-" * 70)
  for dag in SchedulingHelper.registry[cluster]:
    schedule = SchedulingHelper.ArrangeScheduleTime(cluster, dag.dag_id)
    # Just for visualization:
    idx = SchedulingHelper.registry[cluster].index(dag)
    start_time = SchedulingHelper.DEFAULT_ANCHOR + (
        idx * (dt.timedelta(minutes=30) + dt.timedelta(minutes=15))
    )
    print(f"{dag.dag_id:<35} | {schedule:<15} | {start_time.strftime('%H:%M')}")
