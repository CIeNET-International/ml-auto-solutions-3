import logging
import os
import yaml
import datetime as dt
from dataclasses import dataclass

from airflow.models.dagbag import DagBag


@dataclass
class Project:
  """
  Represents a project configuration for scheduling.
  """

  project_path: str
  cluster: any


@dataclass
class Dag:
  """
  Represents a Directed Acyclic Graph (DAG) configuration.
  """

  dag_id: str
  dagrun_timeout: dt.datetime


class SchedulingHelper:
  """Helper class for managing DAG scheduling and registry configuration."""

  YAML_PATH = os.path.join(os.path.dirname(__file__), "schedule_register.yaml")
  OUTPUT_YAML_PATH = os.path.join(
      os.path.dirname(__file__), "generated_schedules.yaml"
  )

  @classmethod
  def load_registry_config(cls) -> dict:
    """Load and parse the registry configuration from a YAML file."""
    if not os.path.exists(cls.YAML_PATH):
      raise FileNotFoundError(f"Registry file not found at: {cls.YAML_PATH}")
    with open(cls.YAML_PATH, "r", encoding="utf-8") as f:
      return yaml.safe_load(f)

  @classmethod
  def get_settings(
      cls, config: dict, project_key: str
  ) -> tuple[dt.datetime, dt.timedelta]:
    """
    Resolves settings with the following priority:
    Project Level > Global Level > System Default
    """
    default_cfg = config.get("default_settings", {})
    project_cfg = config.get(project_key, {})

    # 1. Resolve Anchor Time
    anchor_str = project_cfg.get("anchor_time") or default_cfg.get(
        "anchor_time", "08:00:00"
    )
    h, m, s = map(int, anchor_str.split(":"))
    anchor = dt.datetime(2000, 1, 1, h, m, s, tzinfo=dt.timezone.utc)

    # 2. Resolve Margin
    margin_min = project_cfg.get("margin_minutes") or default_cfg.get(
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
  def get_dagrun_timeout(
      cls, project_key: str, target_dag_id: str
  ) -> dt.timedelta:
    """Retrieves the timeout for a specific DAG from the registry.

    This method loads the registry configuration, looks up the
    project-specific scheduling requirements, and returns the DAG run
    timeout as a timedelta object.

    Args:
      project_key (str): The key identifying the project in the
        registry configuration.
      target_dag_id (str): The ID of the target DAG whose timeout
        should be retrieved.

    Returns:
      dt.timedelta: The timeout duration for the specified DAG,
        converted from the string format "HH:MM:SS" to a timedelta
        object.

    Raises:
      KeyError: If the project_key or target_dag_id is not found in
        the registry.
      ValueError: If the timeout string format is invalid or cannot be
        parsed.
    """
    config = cls.load_registry_config()
    project_data = config.get(project_key, {})
    scheduling_dict = project_data.get("require_scheduling", {})
    print(f"Scheduling Dict for project '{project_key}': {scheduling_dict}")
    timeout = scheduling_dict[target_dag_id]

    h, m, s = map(int, timeout.split(":"))
    return dt.timedelta(hours=h, minutes=m, seconds=s)

  @classmethod
  def arrange_schedule_time(
      cls,
      project_key: str,
      target_dag_id: str,
  ) -> str:
    """Calculates a cron schedule string for a target DAG based on
    sequential offsets.

    This method determines the scheduled execution time for a given DAG by:
    1. Loading the scheduling registry configuration
    2. Finding the anchor time and margin settings for the project
    3. Iterating through DAGs in the project's scheduling dictionary
       in order
    4. Accumulating time offsets (dag timeout + margin) until the
       target DAG is found
    5. Computing the final scheduled time and returning it as a
       cron string

    Args:
      project_key: The project identifier used to look up
        scheduling configuration
      target_dag_id: The DAG ID for which to calculate the
        schedule time

    Returns:
      A cron string in the format "minute hour * * *" representing
      the scheduled time, or None if the target_dag_id is not found
      in the project's scheduling dictionary
    """
    config = cls.load_registry_config()
    anchor, margin = cls.get_settings(config, project_key)

    project_data = config.get(project_key, {})
    scheduling_dict = project_data.get("require_scheduling", {})

    offset = dt.timedelta(0)
    found = False

    # Iterate through the dictionary in order
    for dag_id, timeout in scheduling_dict.items():
      if dag_id == target_dag_id:
        found = True
        break

      h, m, s = map(int, timeout.split(":"))
      offset += dt.timedelta(hours=h, minutes=m, seconds=s) + margin

    if not found:
      return None

    scheduled_time = anchor + offset
    return f"{scheduled_time.minute} {scheduled_time.hour} * * *"

  @classmethod
  def export_all_schedules(cls) -> str:
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

  @classmethod
  def run_scheduling_test(cls, project_key: str) -> None:
    """
    Run an internal test for the SchedulingHelper by exporting all schedules
    and previewing a specific project's schedule configuration.

    This method performs the following steps:
    1. Exports all schedules using the SchedulingHelper.export_all_schedules() method
    2. Loads the generated YAML file and extracts schedules for the specified project
    3. Prints a formatted preview of DAG IDs and their corresponding cron schedules

    Args:
      project_key (str): The project identifier to retrieve and display schedules for.
        This key should correspond to a top-level key in the exported YAML file.

    Returns:
      None: This method prints results to stdout and does not return a value.

    Raises:
      FileNotFoundError: If the output file from
      export_all_schedules() cannot be found.
      yaml.YAMLError: If the exported file is not valid YAML.
    """
    print(f"=== Starting Internal SchedulingHelper Test ({project_key}) ===\n")
    # 1. Run Export Logic
    output_file = SchedulingHelper.export_all_schedules()
    print(f"All schedules calculated and saved to: {output_file}")

    # 2. Preview Output
    with open(output_file, "r", encoding="utf-8") as f:
      review_data = yaml.safe_load(f)
      if project_key in review_data:
        project_schedules = review_data[project_key]["schedules"]
        print("\n--- Current Schedule Preview ---")
        for dag_id, cron in project_schedules.items():
          print(f"{dag_id:<50} -> {cron}")
      else:
        print(f"Key '{project_key}' not found in generated output.")

    print("\n=== Test Completed Successfully ===")


if __name__ == "__main__":
  # --- Internal Testing & Generation ---
  target_key = "tpu_observability"  # Select the project key
  SchedulingHelper.run_scheduling_test(project_key=target_key)
