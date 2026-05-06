"""Utility functions for automating Jupyter notebooks in Airflow."""

import dataclasses
import datetime
import inspect
import logging
import textwrap
from airflow.models.taskmixin import DAGNode
from airflow.models.baseoperator import chain
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import (
    Project,
    RuntimeVersion,
    TpuVersion,
    V6E_GCE_NETWORK,
    V6E_GCE_SUBNETWORK,
    Zone,
)
from dags.post_training.util import test_config_util
from xlml.apis import gcp_config, gcs, metric_config, task, test_config

NOTEBOOK_CONFIG_GCS_PATH = (
    "gs://ml-auto-solutions-dag-configs/post-training/notebook_dag_configs.yaml"
)


def build_maxtext_setup_script() -> str:
  """Builds the shell script for setting up the MaxText environment on TPU VM.

  Returns:
      A shell script string that clones MaxText, installs dependencies, and
      sets up the virtual environment.
  """
  return textwrap.dedent(
      """
      set -e
      set -x

      # =======================================================================
      # Environment Setup
      # =======================================================================

      if [ ! -d "maxtext" ]; then
        git clone https://github.com/AI-Hypercomputer/maxtext.git
      fi
      cd maxtext

      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"

      uv venv --python 3.12 --seed --clear maxtext_venv
      source maxtext_venv/bin/activate

      # =======================================================================
      # MaxText Installation
      # =======================================================================

      uv pip install -e .[tpu-post-train] --resolution=lowest
      install_tpu_post_train_extra_deps

      # =======================================================================
      # Notebook Automation Tools
      # =======================================================================

      uv pip install nbconvert ipykernel papermill

      echo "Environment setup completed"
      """
  )


def _run_parameter_injection(
    notebook_path, output_path, parameters, env_params
):
  """
  Injects literal values or environment lookups into a notebook's code cells.

  This function searches for lines matching `KEY = VALUE` in the notebook and
  replaces the assignment with either a literal repr of the provided value
  or an `os.getenv` call.

  Args:
      notebook_path: Path to the source .ipynb file.
      output_path: Path where the modified .ipynb will be saved.
      parameters: Dict of {key: value} for literal injection.
      env_params: List of keys to be injected as `os.getenv("KEY")`.
  """
  import json
  import re

  with open(notebook_path, encoding="utf-8") as f:
    nb = json.load(f)

  all_keys_to_match = set(parameters.keys()) | set(env_params)
  found_keys = set()

  for cell in (c for c in nb["cells"] if c["cell_type"] == "code"):
    source = cell.get("source", [])
    if isinstance(source, str):
      lines = source.splitlines(keepends=True)
    else:
      lines = source

    new_lines = []
    for line in lines:
      # Match KEY=VALUE assignments with leading spaces and trailing comments.
      # Allow empty values and don't anchor to $ for robustness.
      match = re.match(r"^(\s*)(\w+)(\s*=\s*)([^#\n]*)(.*)", line)
      if match and (key := match.group(2)) in all_keys_to_match:
        found_keys.add(key)
        val = (
            repr(parameters[key])
            if key in parameters
            else f"os.getenv({key!r})"
        )
        # Preserve original indentation and comments
        line = f"{match.group(1)}{key}{match.group(3)}{val}{match.group(5)}\n"
        new_lines.append(line)
        continue

      new_lines.append(line)
    cell["source"] = new_lines

  injected_source = []
  if env_params:
    injected_source.append("import os\n")

  if missing := all_keys_to_match - found_keys:
    if injected_source:
      injected_source.append("\n")
    injected_source.append("# Injected missing parameters (fallback)\n")
    for key in sorted(list(missing)):
      val = (
          repr(parameters[key]) if key in parameters else f"os.getenv({key!r})"
      )
      injected_source.append(f"{key} = {val}\n")

  if injected_source:
    nb["cells"].insert(
        0,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": ["injected-parameters"]},
            "outputs": [],
            "source": injected_source,
        },
    )

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

  print(f"Prepared notebook: {output_path}")


def build_notebook_execution_command(
    notebook_path: str,
    parameters: dict,
    maxtext_path: str = "$(pwd)",
    venv_path: str | None = None,
    env_params: dict[str, any] | None = None,
) -> str:
  """
  Builds a shell command to execute a notebook with injected parameters.

  Args:
      notebook_path: Path to the input notebook file on the TPU VM.
      parameters: Parameters to inject literally (e.g., {"BATCH_SIZE": 32}).
      maxtext_path: Root directory for execution (defaults to current dir).
      venv_path: Path to a virtualenv to activate (relative to maxtext_path).
      env_params: Parameters to pass as environment variables. The notebook
          will be modified to read these via `os.getenv`.

  Returns:
      A shell command string that sets up the env and runs the notebook.
  """
  env_params = env_params or {}

  # Construct the shell command for environment setup and notebook run
  exports = " && ".join(f"export {k}={v}" for k, v in env_params.items())
  export_prefix = f"{exports} && " if exports else ""

  venv_cmd = f"source {venv_path}/bin/activate" if venv_path else "true"
  output_nb = "/tmp/notebook_with_params.ipynb"

  env_setup_script = textwrap.dedent(
      f"""
      cd {maxtext_path}
      {venv_cmd}
      """
  )

  # Bash heredoc containing Python code to inject notebook parameters
  func_body = textwrap.dedent(inspect.getsource(_run_parameter_injection))
  call_func = textwrap.dedent(
      f"""
      _run_parameter_injection(
          {notebook_path!r},
          {output_nb!r},
          {parameters!r},
          {list(env_params.keys())!r}
      )
      """
  )

  injection_script = f"""python << 'PYEOF'\n{func_body}\n{call_func}\nPYEOF"""

  notebook_run_script = (
      f"{export_prefix}papermill {output_nb} {output_nb} --log-output"
  )

  # Verify the success message exists in the notebook's output.
  # We 'grep -v "print("' to ignore the Python source code line
  # and ensure we are matching an actual execution result.
  expected_completed_message = "Training Completed Successfully!"
  verification_script = textwrap.dedent(
      f"""
      set -o pipefail
      if ! grep "{expected_completed_message}" {output_nb} | grep -vq "print("; then
        echo "Error: Notebook did not report '{expected_completed_message}'."
        exit 1
      fi
      """
  )

  template = textwrap.dedent(
      """
      set -ex
      set -o pipefail

      # 1. Environment Setup
      {env_setup_script}

      # 2. Parameter Injection
      {injection_script}

      # 3. Notebook Execution
      {notebook_run_script}

      # 4. Success Verification
      {verification_script}

      echo "Notebook execution completed successfully"
      """
  )

  return template.format(
      env_setup_script=env_setup_script,
      injection_script=injection_script,
      notebook_run_script=notebook_run_script,
      verification_script=verification_script,
  )


def initialize_notebook_test(
    test_name: str,
    dag_name: str,
    notebook_path: str,
    set_up_script: str,
    parameters: dict[str, any],
    task_owner: str,
    tpu_version: TpuVersion,
) -> test_config.TpuVmTest:
  """Creates a TpuVmTest configuration for notebook execution."""
  notebook_execution = build_notebook_execution_command(
      notebook_path=notebook_path,
      parameters=parameters,
      maxtext_path="maxtext",
      venv_path="maxtext_venv",
  )
  return test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=8,
          runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
          reserved=False,
          network=V6E_GCE_NETWORK,
          subnetwork=V6E_GCE_SUBNETWORK,
      ),
      test_name=test_name,
      set_up_cmds=[set_up_script],
      run_model_cmds=[notebook_execution],
      timeout=datetime.timedelta(minutes=180),
      task_owner=task_owner,
      num_slices=1,
      gcs_subfolder=f"{test_config_util.DEFAULT_BUCKET}/{dag_name}",
  )


@dataclasses.dataclass
class NotebookConfig:
  tpu_version: str
  zone: str


def load_notebook_config_from_gcs_yaml(
    gcs_path: str, dag_name: str
) -> NotebookConfig:
  """Loads and parses TPU version and zone configs from GCS yaml config."""
  config = gcs.load_yaml_from_gcs(gcs_path)
  dag_cfg = config.get("dag", {}).get(dag_name, {})

  tpu_version = dag_cfg.get("tpu_version")
  zone = dag_cfg.get("zone")

  return NotebookConfig(tpu_version=tpu_version, zone=zone)


def run_training(
    config: test_config.TpuVmTest, hf_token: str, zone: str | None = None
) -> DAGNode:
  target_zone = zone if zone is not None else Zone.EUROPE_WEST4_B.value
  return task.run_queued_resource_test(
      task_test_config=config,
      task_gcp_config=gcp_config.GCPConfig(
          project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
          zone=target_zone,
          dataset_name=metric_config.DatasetOption.XLML_DATASET,
      ),
      skip_post_process=True,
      custom_env={"HF_TOKEN": hf_token},
  )


def create_branched_notebook_tasks(
    dag_name: str,
    task_id_prefix: str,
    notebook_path: str,
    set_up_script: str,
    parameters: dict[str, any],
    task_owner: str,
    hf_token: str,
    config: NotebookConfig,
    previous_tasks: list[DAGNode] | None = None,
) -> list[DAGNode]:
  """Creates and chains branched notebook tasks for all TPU versions.

  Args:
      dag_name: Name of the DAG.
      task_id_prefix: Prefix for task and operator IDs (e.g. "rl_grpo" or "sft").
      notebook_path: Path to the notebook to run.
      set_up_script: Setup script for MaxText environment.
      parameters: Dict of parameters to inject in the notebook.
      task_owner: Owner of the task.
      hf_token: HuggingFace access token.
      config: Loaded NotebookConfig holding active tpu_version and zone.
      previous_tasks: Optional list of tasks/DAGNodes to chain *before* the branches.

  Returns:
      A list of terminal DAGNode tasks ([run_task, skipped_task]) from the end
      of this branch loop, which can be chained into subsequent tasks.
  """

  tpu_versions = [TpuVersion.V5E, TpuVersion.TRILLIUM]

  def choose_tpu_branch(
      tpu_version_value: str, task_group_id: str, skipped_task_id: str
  ):
    selected = config.tpu_version
    logging.info(
        f"[Branch tpu_version Decision] DAG: {dag_name}, "
        f"Task ID Prefix: {task_id_prefix}. "
        f"Configured active TPU version: '{selected}', "
        f"Current loop TPU version: '{tpu_version_value}'."
    )
    if selected == "all" or selected == tpu_version_value:
      logging.info(
          f"[Branch tpu_version Decision] MATCH! Routing"
          f"execution to active task group: '{task_group_id}'."
      )
      return task_group_id
    logging.info(
        f"[Branch tpu_version Decision] MISMATCH! Routing"
        f"execution to skipped placeholder task: '{skipped_task_id}'."
    )
    return skipped_task_id

  previous_version_tasks = previous_tasks or []
  for tpu_version in tpu_versions:
    # 1. Initialize the test config
    notebook_test = initialize_notebook_test(
        test_name=f"{dag_name}_{task_id_prefix}",
        dag_name=dag_name,
        notebook_path=notebook_path,
        set_up_script=set_up_script,
        parameters=parameters,
        task_owner=task_owner,
        tpu_version=tpu_version,
    )

    # 2. Create run training task group/task
    run_task = run_training(notebook_test, hf_token, zone=config.zone)

    # 3. Create skipped empty operator task
    skipped = EmptyOperator(
        task_id=f"skipped_{task_id_prefix}_{tpu_version.value}"
    )

    # 4. Create BranchPythonOperator
    branch = BranchPythonOperator(
        task_id=f"branch_{task_id_prefix}_{tpu_version.value}",
        python_callable=choose_tpu_branch,
        op_args=[tpu_version.value, run_task.group_id, skipped.task_id],
    )

    # 5. Chain previous tasks to the branch operator if exist
    if previous_version_tasks:
      chain(previous_version_tasks, branch)
      branch.trigger_rule = TriggerRule.ALL_DONE

    # 6. Connect branch to the run task and skipped empty operator
    branch >> [run_task, skipped]

    previous_version_tasks = [run_task, skipped]

  return previous_version_tasks
