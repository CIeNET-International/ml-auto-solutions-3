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

from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags.common.vm_resource import XpkClusters


class Dag():
  """
  The metadata of a DAG.
  Attributes:
    dag_id: The DAG ID.
    dag_run_timeout: The maximum allowed runtime of a DAG run.
  """
  dag_id: str
  dag_run_timeout: dt.timedelta

  def __init__(
      self,
      dag_id: str,
      dag_run_timeout: dt.timedelta=dt.timedelta(minutes=60)
  ):
    self.dag_id = dag_id
    self.dag_run_timeout = dag_run_timeout


class DayOfWeek(enum.Enum):
  ALL = "*"
  WEEK_DAY = "1-5"
  WEEKEND = "0,6"


class SchedulingHelper():
  """
  A helper class to arrange schedule time for XPK cluster DAGs.
  Attributes:
    DEFAULT_MARGIN: The default margin time between DAG runs.
    DEFAULT_ANCHOR: The default anchor time to start scheduling.
    registry: A mapping from XpkClusterConfig to a list of DAGs associated
      with the cluster.
  """

  DEFAULT_MARGIN = dt.timedelta(minutes=15)
  DEFAULT_ANCHOR = dt.datetime(2000, 1, 1, 13, 0, 0, tzinfo=dt.timezone.utc)

  registry: dict[XpkClusterConfig, list[Dag]] = {
      XpkClusters.TPU_V5P_128_CLUSTER: [
          Dag("maxtext_emc_orbax_res_gcs"),
          Dag("maxtext_emc_orbax_res_local"),
          Dag("maxtext_emc_resume_from_gcs"),
          Dag("maxtext_emc_save_gcs"),
          Dag("maxtext_emc_and_mtc_orbax_save_local"),
          Dag("maxtext_mtc_orbax_res_local"),
          Dag("maxtext_mtc_resume_from_gcs"),
          Dag("maxtext_mtc_orbax_save_gcs"),
          Dag("maxtext_regular_restore_with_node_disruption"),
          Dag("maxtext_regular_restore_with_resumed_workload"),
          Dag("maxtext_regular_save"),
      ],
  }

  @classmethod
  def ArrangeScheduleTime(
      cls,
      cluster: XpkClusterConfig,
      dag_id: str,
      day_of_week: DayOfWeek = DayOfWeek.ALL,
  ) -> str:
    """
    """

    if not any(dag.dag_id == dag_id for dag in cls.registry[cluster]):
        raise ValueError(f"{dag_id} is not found in the registry")

    anchor = cls.DEFAULT_ANCHOR
    offset = dt.timedelta(0)
    for dag in cls.registry[cluster]:
      if dag_id == dag.dag_id:
        schedule = anchor + offset
        return f"{schedule.minute} {schedule.hour} * * {day_of_week.value}"

      offset += dag.dag_run_timeout + cls.DEFAULT_MARGIN
      if offset >= dt.timedelta(hours=24):
        raise ValueError(f"Schedule exceeds 24 hours window; offset={offset}")


if __name__ == "__main__":
    test_case = SchedulingHelper.ArrangeScheduleTime(XpkClusters.TPU_V5P_128_CLUSTER, "maxtext_regular_save")
    print(f"Test case schedule: {test_case}")
