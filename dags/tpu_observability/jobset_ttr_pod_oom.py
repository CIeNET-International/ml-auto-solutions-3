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

"""A DAG to validate the `tpumonitoring` SDK, ensuring help() and
list_supported_metrics() are functional inside TPU worker pods."""

import datetime
import tempfile
import os
import random
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
from dags.tpu_observability.utils.jobset_util import JobSet
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH


@task
def trigger_oom_failure(info, pod_name: str):
  """
  Triggers an OOM (Out-of-Memory) event on a specified TPU pod.

  This task connects to the specified pod via kubectl exec and runs a
  memory-intensive Python script (oomkill.py). The script is designed
  to exhaust the container's memory limit, forcing a SIGKILL (Exit Code 137).
  A connection error is expected and caught when the pod is killed.

  Args:
      info: Node pool and cluster information.
      pod_name (str): The name of the target pod to be OOMKilled.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- python3 oomkill.py",
    ])

    try:
      print(f"Executing OOM script in {pod_name}...")
      subprocess.run_exec(cmd, env=env)
    except RuntimeError as e:
      print(f"Expectation: Connection closed due to OOMKilled. Info: {e}")


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
    dag_id="jobset_ttr_pod_oom",
    start_date=datetime.datetime(2026, 1, 15),
    schedule="0 18 * * *" if composer_env.is_prod_env() else None,
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
        an Out-Of-Memory (OOM) event. It launches a JobSet with memory limits,
        injects a memory-intensive workload into a running pod, and uses a sensor
        to confirm that the JobSet controller triggers a recovery and reports the
        recovery duration.
        ### Prerequisites
        This test requires an existing cluster and a container image containing
        the `oomkill.py` script to run.
        ### Procedures
        First, the node pool is created. A JobSet YAML with specific memory constraints
        is then launched on the cluster and given time for all pods to reach a
        `Running` state. After this, a random pod is selected and an OOM event is
        triggered via `kubectl exec` to interrupt the JobSet (expecting Exit Code 137).
        A sensor is finally run which will poll Cloud Monitoring to detect that the
        JobSet Time-To-Recover (TTR) metric has been updated, resulting in a success,
        or timeout, and fail.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    jobset_config = JobSet(
        jobset_name="jobset-ttr-pod-oom-v6e-workload",
        namespace="default",
        max_restarts=5,
        replicated_job_name="tpu-job-slice",
        replicas=1,
        backoff_limit=0,
        completions=4,
        parallelism=4,
        tpu_accelerator_type="tpu-v6e-slice",
        tpu_topology="4x4",
        container_name="jax-tpu-worker",
        image="us-docker.pkg.dev/cienet-cmcs/emma-tpu-test-repo/oom-test:v1",
        tpu_cores_per_pod=4,
    )
    raw_yaml = jobset_config.generate_yaml(workload_script="sleep infinity")
    custom_yaml = raw_yaml.replace(
        "google.com/tpu: 4",
        'google.com/tpu: 4\n                  memory: "4Gi"',
    )
  # Keyword arguments are generated dynamically at runtime (pylint does not
  # know this signature).
  with TaskGroup(  # pylint: disable=unexpected-keyword-arg
      group_id=f"v{config.tpu_version.value}"
  ):
    cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
        task_id="build_node_pool_info_from_gcs_yaml"
    )(
        gcs_path=GCS_CONFIG_PATH,
        dag_name="jobset_ttr_pod_oom",
        is_prod=composer_env.is_prod_env(),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    create_node_pool = node_pool.create.override(task_id="create_node_pool")(
        node_pool=cluster_info,
    )

    start_workload = jobset.run_workload.override(task_id="start_workload")(
        node_pool=cluster_info,
        yaml_config=custom_yaml,
        namespace=jobset_config.namespace,
    )

    ensure_all_pods_running = jobset.wait_for_all_pods_running.override(
        task_id="ensure_all_pods_running"
    )(
        num_pods=(jobset_config.replicas * jobset_config.parallelism),
        node_pool=cluster_info,
    )

    found_pods = jobset.list_pod_names.override(task_id="list_pod_names")(
        node_pool=cluster_info,
        namespace=jobset_config.namespace,
    )

    select_random_pod = pick_random_pod.override(task_id="select_random_pod")(
        active_pods=found_pods
    )

    trigger_oom_killed = trigger_oom_failure.override(
        task_id="trigger_oom_killed"
    )(info=cluster_info, pod_name=select_random_pod)

    wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
        task_id="wait_for_jobset_ttr_to_be_found"
    )(
        node_pool=cluster_info,
        jobset_name=jobset_config.jobset_name,
    )

    cleanup_workload = jobset.end_workload.override(
        task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
    )(
        node_pool=cluster_info,
        jobset_name=jobset_config.jobset_name,
        namespace=jobset_config.namespace,
    ).as_teardown(
        setups=start_workload
    )

    cleanup_node_pool = node_pool.delete.override(
        task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
    )(node_pool=cluster_info).as_teardown(
        setups=create_node_pool,
    )

    chain(
        create_node_pool,
        start_workload,
        ensure_all_pods_running,
        found_pods,
        select_random_pod,
        trigger_oom_killed,
        wait_for_metric_upload,
        cleanup_workload,
        cleanup_node_pool,
    )
