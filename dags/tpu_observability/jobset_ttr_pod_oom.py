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

"""A DAG to validate JobSet Time-to-Recover (TTR) metrics
by injecting Out-of-Memory (OOM) faults into TPU worker pods."""

import datetime
import tempfile
import os
import random
import time
from typing import List

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout

DAG_ID = "jobset_ttr_pod_oom"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)


@task
def trigger_oom_failure(info, pod_name: str, namespace: str):
  """
  Injects an OOM fault by running a memory-intensive loop inside a pod.

  This task waits for the target pod to stabilize, then executes a Python
    one-liner via 'kubectl exec' that rapidly allocates memory. The script
    uses bytearray allocation combined with an immediate write operation
    (a[-1][0] = 1) to ensure physical RAM is committed by the OS, effectively
    triggering a SIGKILL from the OOM Killer (Exit Code 137).

  Args:
    info: An object containing cluster and node pool credentials/metadata.
    pod_name: The name of the target TPU worker pod to inject the fault.
    namespace: The Kubernetes namespace where the pod is running.

  Raises:
    Exception: If the subprocess execution fails for reasons other than
    the expected connection loss during an OOM event.
  """

  wait_time = 60
  print(
      f"Waiting {wait_time}s for Pod {pod_name} to stabilize before OOM Test..."
  )
  time.sleep(wait_time)

  python_logic = (
      "import time\n"
      "a = []\n"
      "print('Starting memory stress test...')\n"
      "while True:\n"
      "    a.append(bytearray(1024**3))\n"
      "    a[-1][0] = 1\n"
      "    time.sleep(0.01)"
  )

  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    credentials_cmd = jobset.Command.get_credentials_command(info)
    exec_cmd = (
        f'kubectl exec {pod_name} -n {namespace} -- python3 -c "{python_logic}"'
    )

    full_command = f"{credentials_cmd} && {exec_cmd}"

    try:
      print(f"Blasting {pod_name} memory now...")
      subprocess.run_exec(full_command, env=env)
    except Exception as e:
      if "137" in str(e):
        print(f"Success: Pod {pod_name} was OOMKilled (Exit Code 137).")
      else:
        print(f"Connection lost (Possible OOM): {e}")


@task
def pick_random_pod(active_pods: List[str]) -> str:
  """
  Randomly selects one pod from a list of available JobSet pods.

  This ensures that the fault injection is performed on a single
  unit of the TPU slice, allowing the test to validate how the
  JobSet controller handles partial failures within a replica.

  Args:
    pod_names (List[str]): List of active pod names in the JobSet.

  Returns:
    str: The name of the randomly selected pod.
  """
  if not active_pods:
    raise ValueError("No pods found to attack!")
  chosen_pod = random.choice(active_pods)
  print(f"Randomly selected pod for OOM test: {chosen_pod}")
  return chosen_pod


with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 1, 15),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "tpu-observability",
        "pod_oom",
        "TPU",
        "v6e-16",
        "Validation",
    ],
    description=(
        "This DAG tests the JobSet time-to-recover metric by injecting "
        "an OOM event into a random pod to trigger a recovery, "
        "then polls Cloud Monitoring to verify the metric is updated."
    ),
    doc_md="""
        ### JobSet Time-To-Recover (TTR) Test Using Random Pod OOM Injection
        ### Description
        This DAG verifies that JobSet can recover from a single pod failure caused by
        an Out-Of-Memory (OOM) event. It launches a JobSet, injects a memory-intensive
        Python stressor into a running pod, and uses a sensor to confirm that the
        JobSet controller triggers a recovery and reports the recovery duration (TTR).
        ### Prerequisites
        This test requires an existing cluster and the ability to execute commands
        within the pod via `kubectl exec`.
        ### Procedures
        First, the node pool is created. A JobSet YAML is then launched on the cluster
        and given time for all pods to reach a `Running` state. After stabilization,
        a random pod is selected and an OOM event is triggered via `kubectl exec`
        using a Python-based memory allocator that forces physical RAM commitment
        until the process is terminated (expecting Exit Code 137). A sensor is
        finally run which will poll Cloud Monitoring to detect that the JobSet
        Time-To-Recover (TTR) metric has been updated, resulting in a success,
        or timeout, and fail.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      selector = jobset.generate_node_pool_selector("jobset-ttr-pod-oom")

      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name=DAG_ID,
          node_pool_selector=selector,
      )

      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name=DAG_ID,
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
          node_pool_selector=selector,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
      )

      start_workload = jobset.run_workload.override(task_id="start_workload")(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      ensure_all_pods_running = jobset.wait_for_all_pods_running.override(
          task_id="ensure_all_pods_running"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      select_random_pod = pick_random_pod.override(task_id="select_random_pod")(
          active_pods=ensure_all_pods_running
      )

      trigger_oom_killed = trigger_oom_failure.override(
          task_id="trigger_oom_killed"
      )(info=cluster_info, pod_name=select_random_pod, namespace="default")

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_jobset_ttr_to_be_found"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info, jobset_config=jobset_config).as_teardown(
          setups=start_workload
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          cluster_info,
          create_node_pool,
          start_workload,
          ensure_all_pods_running,
          select_random_pod,
          trigger_oom_killed,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
