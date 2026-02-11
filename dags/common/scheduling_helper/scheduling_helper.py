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

from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags.common.vm_resource import TpuVersion, Zone


class DayOfWeek(enum.Enum):
  ALL = "*"
  WEEK_DAY = "1-5"
  WEEKEND = "0,6"


# Mock cluster to group TPU Observability DAGs
TPU_OBS_MOCK_CLUSTER = XpkClusterConfig(
    name="tpu-observability-automation-prod",
    device_version=TpuVersion.TRILLIUM,
    core_count=16,
    project="cienet-cmcs",
    zone=Zone.US_CENTRAL1_B.value,
)


class SchedulingHelper:
  """Manages DAG scheduling across different clusters."""

  DEFAULT_MARGIN = dt.timedelta(minutes=15)
  DEFAULT_ANCHOR = dt.datetime(2000, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)

  dag_to_timeout: dict[str, dict[str, dt.timedelta]] = {
      TPU_OBS_MOCK_CLUSTER.name: {
          "gke_node_pool_label_update": dt.timedelta(minutes=30),
          "gke_node_pool_status": dt.timedelta(minutes=30),
          "jobset_rollback_ttr": dt.timedelta(minutes=90),
          "jobset_ttr_node_pool_resize": dt.timedelta(minutes=90),
          "jobset_ttr_pod_delete": dt.timedelta(minutes=90),
          "multi-host-availability-rollback": dt.timedelta(minutes=30),
          "node_pool_ttr_disk_size": dt.timedelta(minutes=90),
          "node_pool_ttr_update_label": dt.timedelta(minutes=90),
          "tpu_info_format_validation_dag": dt.timedelta(minutes=30),
          "tpu_sdk_monitoring_validation": dt.timedelta(minutes=30),
          "jobset_ttr_kill_process": dt.timedelta(minutes=90),
      },
  }

  @classmethod
  def arrange_schedule_time(
      cls,
      dag_id: str,
      day_of_week: DayOfWeek = DayOfWeek.ALL,
  ) -> str:
    """Calculates a cron schedule by stacking timeouts and margins."""
    found_cluster_name = None
    for cluster_name, dags in cls.dag_to_timeout.items():
      if dag_id in dags:
        found_cluster_name = cluster_name
        break

    cluster_dags = cls.dag_to_timeout[found_cluster_name]
    anchor = cls.DEFAULT_ANCHOR
    offset = dt.timedelta(0)

    for current_dag_id, timeout in cluster_dags.items():
      if current_dag_id == dag_id:
        schedule_time = anchor + offset
        return f"{schedule_time.minute} {schedule_time.hour} * * {day_of_week.value}"

      offset += timeout + cls.DEFAULT_MARGIN

      if offset >= dt.timedelta(hours=24):
        raise ValueError(
            f"Schedule exceeds 24h window at {dag_id} in {found_cluster_name}."
        )
    return None

  @classmethod
  def get_dag_timeout(cls, target_dag_id: str) -> dt.timedelta:
    """Searches the registry and returns the specific timeout for a DAG."""
    for dags in cls.dag_to_timeout.values():
      if target_dag_id in dags:
        return dags[target_dag_id]
    raise ValueError(f"DAG '{target_dag_id}' not found in registry.")


if __name__ == "__main__":
  # Verification
  TEST_DAG = "jobset_ttr_pod_delete"
  print(
      f"Schedule for {TEST_DAG}: {SchedulingHelper.arrange_schedule_time(TEST_DAG)}"
  )
  print(f"Timeout for {TEST_DAG}: {SchedulingHelper.get_dag_timeout(TEST_DAG)}")
