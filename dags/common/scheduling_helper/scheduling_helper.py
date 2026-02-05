import logging
import os
import yaml
import enum
import datetime as dt
from dataclasses import dataclass

from airflow.models.dagbag import DagBag


@dataclass
class Project:
  project_path: str
  cluster: any


@dataclass
class Dag:
  dag_id: str
  dagrun_timeout: dt.datetime


class DayOfWeek(enum.Enum):
  ALL = "*"
  WEEK_DAY = "1-5"
  WEEKEND = "0,6"


class SchedulingHelper:
  """Helper class for managing DAG scheduling and registry configuration.

  This class provides utilities for:
  - Loading and parsing schedule registry configuration from YAML files
  - Discovering DAGs from disk and validating their registration status
  - Calculating cron-based schedule times with automatic offset management
  - Exporting all schedules to a generated YAML file

  The scheduler uses a default anchor time
  (2000-01-01 08:00:00 UTC) and arranges
  DAGs sequentially with configurable margins to prevent overlap.

  Class Attributes:
    DEFAULT_MARGIN (dt.timedelta): Time buffer between
      consecutive DAG runs (15 minutes).
    DEFAULT_ANCHOR (dt.datetime): Starting time reference
      for schedule calculations.
    YAML_PATH (str): Path to the schedule registry configuration file.
    OUTPUT_YAML_PATH (str): Path where generated schedules will be exported.
  """

  DEFAULT_MARGIN = dt.timedelta(minutes=15)
  DEFAULT_ANCHOR = dt.datetime(2000, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
  YAML_PATH = os.path.join(os.path.dirname(__file__), "schedule_register.yaml")
  OUTPUT_YAML_PATH = os.path.join(
      os.path.dirname(__file__), "generated_schedules.yaml"
  )

  @classmethod
  def load_registry_config(cls) -> dict:
    if not os.path.exists(cls.YAML_PATH):
      raise FileNotFoundError(f"Registry file not found at: {cls.YAML_PATH}")
    with open(cls.YAML_PATH, "r", encoding="utf-8") as f:
      return yaml.safe_load(f)

  @classmethod
  def get_all_dags_from_disk(cls, project: Project) -> list[Dag]:
    """Required for CI Consistency tests."""

    dagbag = DagBag(dag_folder=project.project_path, include_examples=False)
    dag_list = []
    for dag_id, dag_obj in dagbag.dags.items():
      timeout = getattr(dag_obj, "dagrun_timeout", None)
      dag_list.append(Dag(dag_id=dag_id, dagrun_timeout=timeout))
    return dag_list

  @classmethod
  def discover_actual_dags(cls, project_path: str) -> set[str]:
    """Scans the disk for real DAG objects to catch unregistered files."""
    # Suppress airflow logs for a cleaner CI output
    logging.getLogger("airflow.models.dagbag.DagBag").setLevel(logging.ERROR)

    dagbag = DagBag(dag_folder=project_path, include_examples=False)
    return set(dagbag.dags.keys())

  @classmethod
  def arrange_schedule_time(
      cls,
      project_key: str,
      target_dag_id: str,
      day_of_week: DayOfWeek = DayOfWeek.ALL,
  ) -> str:
    """Calculates cron string using ONLY the registry file."""
    if not os.path.exists(cls.YAML_PATH):
      return None

    try:
      with open(cls.YAML_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
      return None

    project_data = config.get(project_key)
    if not project_data:
      return None

    if target_dag_id in project_data.get("no_scheduling_required", []):
      return None

    scheduled_list = project_data.get("require_scheduling", [])
    offset = dt.timedelta(0)
    found = False

    for entry in scheduled_list:
      curr_id = entry["id"] if isinstance(entry, dict) else entry
      if curr_id == target_dag_id:
        found = True
        break

      timeout_str = (
          entry.get("timeout", "01:00:00")
          if isinstance(entry, dict)
          else "01:00:00"
      )
      try:
        h, m, s = map(int, timeout_str.split(":"))
        duration = dt.timedelta(hours=h, minutes=m, seconds=s)
      except ValueError:
        duration = dt.timedelta(hours=1)
      offset += duration + cls.DEFAULT_MARGIN

    if not found:
      return None

    scheduled_time = cls.DEFAULT_ANCHOR + offset
    # Ensure we use .value for the enum
    dow_val = (
        day_of_week.value if isinstance(day_of_week, DayOfWeek) else day_of_week
    )
    return f"{scheduled_time.minute} {scheduled_time.hour} * * {dow_val}"

  @classmethod
  def export_all_schedules(cls):
    config = cls.load_registry_config()
    final_output = {}
    for project_key, data in config.items():
      final_output[project_key] = {
          "schedule_name": data.get("schedule_name", project_key),
          "project_path": data.get("project_path"),
          "schedules": {},
      }
      for entry in data.get("require_scheduling", []):
        current_dag_id = entry["id"] if isinstance(entry, dict) else entry
        final_output[project_key]["schedules"][
            current_dag_id
        ] = cls.arrange_schedule_time(project_key, current_dag_id)
      for current_dag_id in data.get("no_scheduling_required", []):
        final_output[project_key]["schedules"][current_dag_id] = None

    with open(cls.OUTPUT_YAML_PATH, "w", encoding="utf-8") as f:
      yaml.dump(final_output, f, sort_keys=False, allow_unicode=True)
    return cls.OUTPUT_YAML_PATH


# --- Internal Testing & Generation ---
if __name__ == "__main__":
  target_key = "tpu_observability"

  print(f"=== Starting Internal SchedulingHelper Test ({target_key}) ===\n")
  try:
    # 1. Run Export Logic
    output_file = SchedulingHelper.export_all_schedules()
    print(f"All schedules calculated and saved to: {output_file}")

    # 2. Preview Output
    with open(output_file, "r", encoding="utf-8") as f:
      review_data = yaml.safe_load(f)
      if target_key in review_data:
        project_schedules = review_data[target_key]["schedules"]
        print("\n--- Current Schedule Preview ---")
        for dag_id, cron in project_schedules.items():
          print(f"{dag_id:<50} -> {cron}")
      else:
        print(f"Key '{target_key}' not found in generated output.")

    print("\n=== Test Completed Successfully ===")
  except (FileNotFoundError, yaml.YAMLError, OSError, KeyError) as e:
    import traceback

    print(f"\nInternal Test Failed: {e}")
    traceback.print_exc()
