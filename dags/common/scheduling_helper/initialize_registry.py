import os

from airflow.models.dagbag import DagBag


def generate_initial_registry(project_path, project_key):
  """
  Scans the directory and prints a YAML-compatible structure.
  All discovered DAGs are placed under 'require_scheduling' by default.
  """
  if not os.path.exists(project_path):
    print(f"Error: Path '{project_path}' does not exist.")
    return

  print(f"Scanning {project_path} for DAGs...")

  # Load DAGs using Airflow's DagBag
  dagbag = DagBag(dag_folder=project_path, include_examples=False)

  if not dagbag.dags:
    print("No DAGs found in the specified directory.")
    return

  # Prepare YAML output
  output = []
  output.append(f"{project_key}:")
  output.append(f"  schedule_name: \"{project_key.replace('_', ' ').title()}\"")
  output.append(f'  project_path: "{project_path}"')
  output.append(
      '  cluster_name: "please specify cluster name, '
      'e.g., TPU_V5P_128_CLUSTER"'
  )
  output.append("  require_scheduling:")

  # Sort alphabetically for a clean start
  for dag_id in sorted(dagbag.dags.keys()):
    dag_obj = dagbag.dags[dag_id]
    timeout = getattr(dag_obj, "dagrun_timeout", "MISSING")

    output.append(f'    - "{dag_id}"  # timeout: {timeout}')

  output.append("  no_scheduling_required: []")

  print("\n--- Copy the content below into schedule_register.yaml ---\n")
  print("\n".join(output))
  print("\n--------------------------------------------------------")
  print(
      "Edit the no_scheduling_required list and "
      "adjust DAG order as needed before running the scheduling helper."
  )


if __name__ == "__main__":
  # Define your project path here
  TARGET_PATH = "dags/tpu_observability/"
  TARGET_PROJECT_KEY = "tpu_observability"
  generate_initial_registry(TARGET_PATH, TARGET_PROJECT_KEY)
