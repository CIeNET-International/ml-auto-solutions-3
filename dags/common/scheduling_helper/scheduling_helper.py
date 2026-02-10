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

"""Helper module for scheduling DAGs across clusters."""
import datetime as dt
import enum


class Dag:
  """Represents an Airflow DAG with its identification and expected runtime."""

  def __init__(
      self,
      dag_run_timeout: dt.timedelta = dt.timedelta(minutes=60),
  ):
    self.dag_run_timeout = dag_run_timeout


class DayOfWeek(enum.Enum):
  ALL = "*"
  WEEK_DAY = "1-5"
  WEEKEND = "0,6"


class Cluster:
  tpu_obs_prod = "tpu-observability-automation-prod"


class SchedulingHelper:
  """Manages DAG scheduling across different clusters."""

  DEFAULT_MARGIN = dt.timedelta(minutes=15)
  DEFAULT_ANCHOR = dt.datetime(2000, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)

  registry: dict[Cluster, dict[str, Dag]] = {
      Cluster.tpu_obs_prod: {
          "gke_node_pool_label_update": Dag(dt.timedelta(minutes=30)),
          "gke_node_pool_status": Dag(dt.timedelta(minutes=30)),
          "jobset_rollback_ttr": Dag(dt.timedelta(minutes=90)),
          "jobset_ttr_node_pool_resize": Dag(dt.timedelta(minutes=90)),
          "jobset_ttr_pod_delete": Dag(dt.timedelta(minutes=90)),
          "multi-host-availability-rollback": Dag(dt.timedelta(minutes=30)),
          "node_pool_ttr_disk_size": Dag(dt.timedelta(minutes=90)),
          "node_pool_ttr_update_label": Dag(dt.timedelta(minutes=90)),
          "tpu_info_format_validation_dag": Dag(dt.timedelta(minutes=30)),
          "tpu_sdk_monitoring_validation": Dag(dt.timedelta(minutes=30)),
          "jobset_ttr_kill_process": Dag(dt.timedelta(minutes=90)),
      },
  }

  @classmethod
  def arrange_schedule_time(
      cls,
      cluster: Cluster,
      target_dag_id: str,
      day_of_week: DayOfWeek = DayOfWeek.ALL,
  ) -> str:
    if cluster not in cls.registry:
      raise ValueError(f"Cluster {cluster} is not registered.")

    cluster_dags = cls.registry[cluster]
    if target_dag_id not in cluster_dags:
      raise ValueError(f"DAG '{target_dag_id}' not found in registry.")

    anchor = cls.DEFAULT_ANCHOR
    offset = dt.timedelta(0)

    for current_dag_id, dag in cluster_dags.items():
      if target_dag_id == current_dag_id:
        schedule_time = anchor + offset
        return (
            f"{schedule_time.minute} {schedule_time.hour} * * "
            f"{day_of_week.value}"
        )

      offset += dag.dag_run_timeout + cls.DEFAULT_MARGIN

      if offset >= dt.timedelta(hours=24):
        raise ValueError(f"Schedule exceeds 24h window at {current_dag_id}.")

    return None
