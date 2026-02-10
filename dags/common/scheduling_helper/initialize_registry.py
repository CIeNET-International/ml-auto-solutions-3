import os
import datetime as dt
from airflow.models.dagbag import DagBag


def format_timeout(timeout_obj) -> str:
  """Converts a timedelta to HH:MM:SS string or returns None."""
  if not isinstance(timeout_obj, dt.timedelta):
    return None
  total_seconds = int(timeout_obj.total_seconds())
  hours, remainder = divmod(total_seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f"{hours:02}:{minutes:02}:{seconds:02}"


def generate_initial_registry(project_path: str, project_key: str) -> None:
  """
  Generates a YAML template. Automatically categorizes DAGs based on
  whether they have a defined timeout.
  """
  if not os.path.exists(project_path):
    print(f"Error: Path '{project_path}' does not exist.")
    return

  # Load DAGs using Airflow's DagBag
  dagbag = DagBag(dag_folder=project_path, include_examples=False)

  if not dagbag.dags:
    print("No DAGs found in the specified directory.")
    return

  # Categorize DAGs
  scheduled_dags = []
  unscheduled_dags = []

  for dag_id in sorted(dagbag.dags.keys()):
    dag_obj = dagbag.dags[dag_id]
    timeout_str = format_timeout(getattr(dag_obj, "dagrun_timeout", None))

    if timeout_str:
      scheduled_dags.append(f'    {dag_id}: "{timeout_str}"')
    else:
      unscheduled_dags.append(f"    {dag_id}: null")

  output = []

  # 1. Project Level Settings
  output.append(f"{project_key}:")
  output.append(f"  schedule_name: \"{project_key.replace('_', ' ').title()}\"")
  output.append(f'  project_path: "{project_path}"')
  output.append('  cluster_name: "TPU_V5P_128_CLUSTER"')
  output.append('  anchor_time: "08:00:00"')
  output.append("  margin_minutes: 15")

  # 2. Section: require_scheduling (Only those with timeouts)
  output.append("  require_scheduling:")
  if scheduled_dags:
    output.extend(scheduled_dags)
  else:
    output.append("    # No DAGs with timeouts found")

  # 3. Section: no_scheduling_required (Those without timeouts)
  output.append("  no_scheduling_required:")
  if unscheduled_dags:
    output.extend(unscheduled_dags)
  else:
    output.append("    # All DAGs have timeouts")

  print("\n--- Project YAML Template Generated (Auto-Categorized) ---\n")
  print("\n".join(output))
  print("\n---------------------------------------------------------")


if __name__ == "__main__":
  TARGET_PATH = "dags/tpu_observability/"
  TARGET_PROJECT_KEY = "tpu_observability"
  generate_initial_registry(TARGET_PATH, TARGET_PROJECT_KEY)
