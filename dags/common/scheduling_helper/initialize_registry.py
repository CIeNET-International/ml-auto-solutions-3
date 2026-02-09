import os

from airflow.models.dagbag import DagBag


def generate_initial_registry(project_path, project_key):
  """
  Generate an initial registry configuration for a project's DAGs in
  YAML format.

  This function scans a specified project path for Airflow DAGs,
  extracts their metadata, and generates a YAML-formatted
  configuration template that can be copied into a
  schedule_register.yaml file. The output includes the project key,
  schedule name, project path, cluster name placeholder, and a list
  of all discovered DAGs with their timeout values.

  Args:
    project_path (str): The file system path to the project directory
      containing DAG files.
    project_key (str): A unique identifier for the project, used as
      the top-level key in the YAML output. This will also be
      converted to a human-readable schedule name.

  Returns:
    None: The function prints the generated YAML configuration to
      stdout and does not return a value.

  Side Effects:
    - Prints error messages if the project path doesn't exist
    - Prints the generated YAML configuration to stdout
    - Prints instructions for manual editing of the configuration

  Example:
    >>> generate_initial_registry("/path/to/dags", "my_project")
    Scanning /path/to/dags for DAGs...

    --- Copy the content below into schedule_register.yaml ---

    my_project:
      schedule_name: "My Project"
      project_path: "/path/to/dags"
      cluster_name: "please specify cluster name, e.g., TPU_V5P_128_CLUSTER"
      require_scheduling:
      - "dag_1"  # timeout: 0:30:00
      - "dag_2"  # timeout: 1:00:00
      no_scheduling_required: []

    --------------------------------------------------------
    Edit the no_scheduling_required list and adjust DAG order as
    needed before running the scheduling helper.

  Note:
    - DAGs are sorted alphabetically in the output
    - The cluster_name field requires manual specification
    - Users must manually categorize DAGs into require_scheduling or
      no_scheduling_required lists
    - Requires Airflow's DagBag to be available for DAG discovery
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
