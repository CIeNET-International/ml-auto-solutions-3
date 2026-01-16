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

"""Utilities for executing Python commands within TPU worker pods."""

import os
import tempfile
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import subprocess_util as subprocess


def execute_tpu_python_command(
    info, pod_name: str, python_code: str, namespace: str = "default"
) -> str:
  """Executes a Python command inside a specific TPU pod via kubectl exec.

  This utility centrally manages the 'tpumonitoring' SDK import and executes
  the provided Python snippet. It assumes the environment is ready.

  Args:
      info: Node pool and cluster information.
      pod_name: The name of the target pod.
      python_code: The Python snippet to run (without the SDK import).
      namespace: Kubernetes namespace.

  Returns:
      The standard output of the executed command.
  """
  # Centrally manage the common import as requested by the member
  full_python_code = f"from libtpu.sdk import tpumonitoring; {python_code}"

  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    # Build command: Get credentials and execute the command in pod
    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f'kubectl exec {pod_name} -n {namespace} -- python3 -c "{full_python_code}"',
    ])
    return subprocess.run_exec(cmd, env=env)
