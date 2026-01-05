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

"""SchedulingHelper module for arranging DAG schedules in XPK clusters."""

from dataclasses import dataclass
import datetime as dt
import enum

from airflow.models.dagbag import DagBag
from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags.common.vm_resource import XpkClusters

@dataclass
class Project:
  """
  The metadata of a project.
  Attributes:
    project_id: The GCP project ID.
    cluster: The XPK cluster configuration.
  """
  project_path: str
  cluster: XpkClusterConfig

@dataclass
class Dag:
  """
  The metadata of a DAG.
  Attributes:
    dag_id: The DAG ID.
    last_scheduled_time: The last scheduled time of the DAG.
  """
  dag_id: str
  dagrun_timeout: dt.datetime



class DayOfWeek(enum.Enum):
  ALL = "*"
  WEEK_DAY = "1-5"
  WEEKEND = "0,6"


class SchedulingHelper:
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

  registry : dict[str, Project] = {
      "orbax" : Project(
          "dags/orbax/",
          XpkClusters.TPU_V5P_128_CLUSTER
      ),
  }

  @classmethod
  def GetAllDag(cls, project: Project) -> list[Dag]:
    """Get all DAG id from all Python files under target project(folder)."""
    if not project.project_path.endswith("/"):
      raise ValueError("folder_path must end with '/'")

    dagbags = DagBag(
        dag_folder=project.project_path,
        include_examples=False,
    )
    if len(dagbags.dags.keys()) == 0 :
      raise ValueError(f"{project.project_path} doesn't exist or has no dag.")

    dag_list: list[Dag] = []
    for dag_id, dag in dagbags.dags.items():
      rundag_timeout = getattr(dag, "dagrun_timeout", None)
      if not rundag_timeout:
        raise ValueError(
            f"Dag rundag_timeout parameter is necessary : {dag_id}"
        )
      dag_list.append(Dag(dag_id, rundag_timeout))
      dag_list.sort(key=lambda d: d.dag_id)

    return dag_list

  @classmethod
  def ArrangeScheduleTime(
      cls,
      project: Project,
      dag_id: str,
      day_of_week: DayOfWeek = DayOfWeek.ALL,
  ) -> str:
    """
    Compute a cron schedule for a DAG based on its run timeout and a
    default margin.
    Args:
        project (Project): Project containing the DAGs.
        dag_id (str): DAG ID to schedule.
        day_of_week (DayOfWeek, optional): Day of the week. Defaults to
        all days.
    Returns:
        str: Cron string in "minute hour * * day_of_week" format.
    Raises:
        ValueError: If DAG not found or schedule exceeds 24 hours.
    """
    dag_list = cls.GetAllDag(project)
    offset = dt.timedelta(0)
    anchor = cls.DEFAULT_ANCHOR

    fail_log_lines = [
        f"{'Dag':<50} "
        f"{'Dagrun_timeout':<15} "
        f"{'Margin':<10} "
        f"{'Accumulation time':<15}"
    ]

    for dag in dag_list :
      # 1. return "start time" of target dag
      # 2. make sure "end time" of target dag wouldn't be over 24 hours
      if dag.dag_id == dag_id:
        schedule = anchor + offset
        end_time = schedule + dag.dagrun_timeout + cls.DEFAULT_MARGIN
        if end_time - anchor > dt.timedelta(hours=24):
          fail_log_lines.append(
              f"{dag.dag_id:<50} "
              f"{str(dag.dagrun_timeout):<15} "
              f"{str(cls.DEFAULT_MARGIN):<10} "
              f"{str(end_time - anchor):<15}"
          )
          print("\n".join(fail_log_lines) + "\n")
          raise ValueError(
              "Schedule exceeds 24 hours window;"
              "adjust the DEFAULT_MARGIN or dagrun_timeout accordingly. "
              f"dag: {dag.dag_id}, end_time: {end_time.time()}"
          )
        return f"{schedule.minute} {schedule.hour} * * {day_of_week.value}"

      offset += dag.dagrun_timeout + cls.DEFAULT_MARGIN
      fail_log_lines.append(
          f"{dag.dag_id:<50} "
          f"{str(dag.dagrun_timeout):<15} "
          f"{str(cls.DEFAULT_MARGIN):<10} "
          f"{str(offset):<15}"
      )
      if offset >= dt.timedelta(hours=24):
        print(
            "\n".join(fail_log_lines) +
            "\n"
        )

        raise ValueError(
            "Schedule exceeds 24 hours window;"
            "adjust the DEFAULT_MARGIN or dagrun_timeout accordingly. "
            f"dag: {dag.dag_id}, accumulation time: {offset}"
        )


    raise ValueError("Dag doesn't exist")


if __name__ == "__main__" :
  s = SchedulingHelper.ArrangeScheduleTime(
      SchedulingHelper.registry["orbax"],
      "maxtext_regular_restore_with_node_disruption"
  )
