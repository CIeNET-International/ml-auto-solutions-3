import logging
import os
import yaml
import enum
import datetime as dt
from dataclasses import dataclass

from airflow.models.dagbag import DagBag


@dataclass
class Project:
  """
  Represents a project configuration for scheduling.

  Attributes:
    project_path (str): The file system path to the project directory.
    cluster (any): The cluster configuration or reference
      where the project will be executed.
  """

  project_path: str
  cluster: any


@dataclass
class Dag:
  """
  Represents a Directed Acyclic Graph (DAG) configuration.

  Attributes:
    dag_id (str): Unique identifier for the DAG.
    dagrun_timeout (dt.datetime): Maximum time allowed for
      a DAG run to complete before timing out.
  """

  dag_id: str
  dagrun_timeout: dt.datetime


class DayOfWeek(enum.Enum):
  """Enum representing days of the week for scheduling purposes.

  This enum provides convenient constants for specifying which days of the week
  a scheduled task should run.

  Attributes:
    ALL: Represents all days of the week (every day). Value: "*"
    WEEK_DAY: Represents weekdays (Monday through Friday). Value: "1-5"
    WEEKEND: Represents weekend days (Sunday and Saturday). Value: "0,6"

  Note:
    The values follow cron expression syntax where:
    - 0 represents Sunday
    - 1-5 represents Monday through Friday
    - 6 represents Saturday
    - "*" represents all days
  """

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
    """Load and parse the registry configuration from a YAML file.

    This class method reads the YAML configuration file located at the path
    specified by cls.YAML_PATH and returns its contents as a dictionary.

    Returns:
      dict: The parsed YAML configuration data as a dictionary.

    Raises:
      FileNotFoundError: If the YAML file does not exist at the specified path.

    Note:
      The file is read with UTF-8 encoding to ensure proper handling of
      special characters.
    """
    if not os.path.exists(cls.YAML_PATH):
      raise FileNotFoundError(f"Registry file not found at: {cls.YAML_PATH}")
    with open(cls.YAML_PATH, "r", encoding="utf-8") as f:
      return yaml.safe_load(f)

  @classmethod
  def get_settings(cls, config: dict, project_key: str):
    """
    Resolves settings with the following priority:
    Project Level > Global Level > System Default
    """
    global_cfg = config.get("global_settings", {})
    project_cfg = config.get(project_key, {})

    # 1. Resolve Anchor Time
    anchor_str = project_cfg.get("anchor_time") or global_cfg.get(
        "anchor_time", "08:00:00"
    )
    h, m, s = map(int, anchor_str.split(":"))
    anchor = dt.datetime(2000, 1, 1, h, m, s, tzinfo=dt.timezone.utc)

    # 2. Resolve Margin
    margin_min = project_cfg.get("margin_minutes") or global_cfg.get(
        "margin_minutes", 15
    )
    margin = dt.timedelta(minutes=int(margin_min))

    return anchor, margin

  @classmethod
  def discover_actual_dags(cls, project_path: str) -> set[str]:
    """
    Discover and return the set of actual DAG IDs found in the
    specified project path.

    This method scans the filesystem for DAG files and extracts
    the actual DAG objects that are defined in them. It suppresses
    Airflow's DagBag logging to reduce noise during execution.

    Args:
      project_path (str): The filesystem path to the folder
        containing DAG files. This is typically the root directory
        where Airflow DAG definitions are stored.

    Returns:
      set[str]: A set of DAG IDs (strings) for all DAGs
        discovered in the project path. Each ID corresponds to
        a DAG object's `dag_id` attribute.

    Note:
      - This method temporarily suppresses Airflow DagBag logging
        at ERROR level to provide cleaner output during CI/testing.
      - Example DAGs are explicitly excluded from the discovery
        process.
      - This is useful for validation and ensuring all DAGs are
        properly registered.
    """
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
    """
    Calculates and returns a cron schedule
    string for a target DAG based on project configuration.

    This method reads a YAML configuration
    file to determine the scheduling requirements
    for a specific DAG within a project.
    It calculates the scheduled time by accumulating
    timeout durations of all DAGs that should run
    before the target DAG, starting from
    a default anchor time.

    Args:
      project_key: The key identifying the project
        in the YAML configuration file.
      target_dag_id: The identifier of the DAG for which
        to calculate the schedule.
      day_of_week: The day(s) of the week when the DAG should run.
        Defaults to DayOfWeek.ALL.

    Returns:
      A cron schedule string in the format
      "minute hour * * day_of_week" if the DAG
      requires scheduling, or None if:
      - The YAML configuration file doesn't exist
      - The YAML file cannot be parsed
      - The project_key is not found in the configuration
      - The target_dag_id is listed in "no_scheduling_required"
      - The target_dag_id is not found in "require_scheduling"

    The schedule time is calculated by:
    1. Starting from DEFAULT_ANCHOR time
    2. Adding the timeout duration of each DAG that comes before the target DAG
        in the "require_scheduling" list
    3. Adding DEFAULT_MARGIN between each DAG execution
    4. Formatting the result as a cron expression with the specified day_of_week

    Example:
      If DEFAULT_ANCHOR is 00:00:00, and there are two DAGs before the target
      with timeouts of "01:30:00" and "02:00:00",
      with DEFAULT_MARGIN of 10 minutes,
      the calculated time would be 03:50:00 (1:30 + 0:10 + 2:00 + 0:10).
    """
    if not os.path.exists(cls.YAML_PATH):
      return None

    try:
      config = cls.load_registry_config()
      anchor, margin = cls.get_settings(config, project_key)
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
      offset += duration + margin
    if not found:
      return None

    scheduled_time = anchor + offset
    dow_val = (
        day_of_week.value if isinstance(day_of_week, DayOfWeek) else day_of_week
    )
    return f"{scheduled_time.minute} {scheduled_time.hour} * * {dow_val}"

  @classmethod
  def export_all_schedules(cls):
    """
    Export all schedules from the registry configuration to a YAML file.

    This method reads the registry configuration,
    processes scheduling information for all projects,
    and writes the compiled schedule data to an output YAML file.

    For each project in the registry:
    - Extracts project metadata (schedule_name, project_path)
    - Processes DAGs that require scheduling by calling arrange_schedule_time()
    - Includes DAGs that don't require scheduling with None values

    Returns:
      str: The path to the generated YAML file containing all schedules.

    The output YAML structure:
      {
        "project_key": {
          "schedule_name": str,
          "project_path": str,
          "schedules": {
            "dag_id": schedule_time or None,
            ...
        },
        ...
    """
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
