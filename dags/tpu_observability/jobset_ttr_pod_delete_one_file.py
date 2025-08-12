# Copyright 2026 Google LLC
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

"""A DAG to test jobset time-to-recover metric using a jobset pod delete in a self-contained file."""

import dataclasses
import datetime
from datetime import timedelta
import json
import logging
import os
import random
import tempfile
from typing import Final

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.sensors.base import PokeReturnValue
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common.vm_resource import DockerImage
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.utils.node_pool_util import Info
from dags.tpu_observability.configs.common import MachineConfigMap
from dags.common.scheduling_helper.scheduling_helper import (
    SchedulingHelper,
    get_dag_timeout,
)
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.gcp_util import list_time_series
from dags.tpu_observability.utils.subprocess_util import run_exec


@task
def build_jobset_from_gcs_yaml(
    image: str,
) -> JobSet:
  """Builds a JobSet instance with all configuration parameters expanded in-code."""
  namespace = "default"
  max_restarts = 5
  replicated_job_name = "tpu-job-slice"
  replicas = 1
  container_name = "jax-tpu-worker"
  backoff_limit = 0
  tpu_cores_per_pod = 4
  tpu_accelerator_type = "tpu-v6e-slice"
  tpu_topology = "4x4"
  completions = 4
  parallelism = 4

  return JobSet(
      namespace=namespace,
      max_restarts=max_restarts,
      replicated_job_name=replicated_job_name,
      replicas=replicas,
      backoff_limit=backoff_limit,
      completions=completions,
      parallelism=parallelism,
      tpu_accelerator_type=tpu_accelerator_type,
      tpu_topology=tpu_topology,
      container_name=container_name,
      image=image,
      tpu_cores_per_pod=tpu_cores_per_pod,
      delay_recovery=True,
  )


@task
def build_node_pool_info_from_gcs_yaml(
    is_prod: bool,
    machine_type: str,
    tpu_topology: str,
) -> Info:
  """Builds a GKE node pool Info instance with all configuration parameters expanded in-code."""
  project_id = "cienet-cmcs"
  if is_prod:
    cluster_name = "yuna-automation"
    location = "us-central1"
    region = "us-central1"
    node_locations = "us-central1-b"
    zone = "us-central1-b"
    reservation = "cloudtpu-20260121053000-1746002734"
    num_nodes = 4
  else:
    cluster_name = "yuna-automation"
    location = "us-central1"
    region = "us-central1"
    node_locations = "us-central1-b"
    zone = "us-central1-b"
    reservation = "cloudtpu-20260121053000-1746002734"
    num_nodes = 4

  # Explicitly configured node pool name for existing tests
  node_pool_name = "jobset-ttr-pod-delete-exist-v6e"

  return Info(
      project_id=project_id,
      cluster_name=cluster_name,
      node_pool_name=node_pool_name,
      location=location,
      region=region,
      node_locations=node_locations,
      zone=zone,
      reservation=reservation,
      num_nodes=num_nodes,
      machine_type=machine_type,
      tpu_topology=tpu_topology,
  )


@task
def delete_one_random_pod(
    node_pool: Info,
    jobset_config: JobSet,
    jobset_name: str,
):
  """Randomly selects and deletes one pod that is currently in the 'running' state."""
  running_pods = jobset.get_running_pods(
      node_pool=node_pool,
      jobset_name=jobset_name,
      namespace=jobset_config.namespace,
  )
  if not running_pods:
    logging.error(
        "No running pods found in namespace: %s", jobset_config.namespace
    )
    raise AirflowFailException(
        f"No running pods found in namespace: {jobset_config.namespace}"
    )

  target_pod = random.choice(running_pods)
  logging.info("Targeting pod for deletion: %s", target_pod)

  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(node_pool),
        jobset.Command.k8s_delete_pod_command(
            temp_config_file.name, target_pod, jobset_config.namespace
        ),
    ])

    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    run_exec(cmd, env=env)
    logging.info("Successfully initiated deletion for pod: %s", target_pod)

    return TimeUtil.from_datetime(current_time_utc)


@task.sensor(poke_interval=30, timeout=600, mode="poke")
def wait_for_jobset_recovered(
    node_pool: Info, jobset_config: JobSet, jobset_name: str
) -> PokeReturnValue:
  """Executes the GKE event command and extracts the LAST_SEEN timestamp for recovery."""
  with tempfile.TemporaryDirectory() as tmpdir:
    env = os.environ.copy()
    env["KUBECONFIG"] = os.path.join(tmpdir, "kubeconfig")

    cmd = " && ".join([
        jobset.Command.get_credentials_command(node_pool),
        # kubectl get events -n <namespace> --field-selector
        # involvedObject.kind=JobSet,involvedObject.name=<jobset_name> ...
        jobset.Command.k8s_get_jobset_events_command(
            jobset_name,
            jobset_config.namespace,
        ),
    ])

    stdout = run_exec(cmd, env=env)

    logging.info(stdout)
    lines = stdout.strip().splitlines()
    logging.info(lines)
    if not lines:
      logging.warning("No events found for JobSet %s", jobset_name)
      return PokeReturnValue(is_done=False)

    first_line = lines[0].split()
    if first_line:
      end_time = TimeUtil.from_iso_string(first_line[0])
      logging.info("Detected JobSet restart time: %s", end_time)
      return PokeReturnValue(is_done=True, xcom_value=end_time)


@task
def verify_recovery_duration(start_time: TimeUtil, end_time: TimeUtil):
  """Checks if the time elapsed since start_timestamp is > 60 seconds."""
  duration = end_time.time - start_time.time

  logging.info(f"Start Time: {start_time.to_iso_string()}")
  logging.info(f"End Time: {end_time.to_iso_string()}")
  logging.info(f"Recovery Duration: {duration} seconds")

  if duration < 60:
    raise AirflowFailException(
        f"Recovery too fast ({duration}s < 60s). "
        "The 'jobset_time_to_recover' metric requires > 60s to be recorded. "
        "Failing fast to avoid waiting for a missing metric."
    )


@task.sensor(poke_interval=60, timeout=3900, mode="poke")
def wait_for_jobset_ttr_to_be_found(
    node_pool: Info,
    jobset_name: str,
    start_time: TimeUtil = None,
) -> bool:
  """Polls the jobset time-to-recover metric in Cloud Monitoring."""
  query_start = (
      start_time if start_time else TimeUtil.now() - timedelta(minutes=60)
  )

  time_series = list_time_series(
      project_id=node_pool.project_id,
      filter_str=(
          'metric.type="kubernetes.io/jobset/times_to_recover" '
          f'resource.labels.cluster_name="{node_pool.cluster_name}" '
          f'resource.labels.entity_name="{jobset_name}"'
      ),
      start_time=query_start,
      end_time=TimeUtil.now(),
  )

  logging.info("Time series: %s", time_series)
  return len(time_series) > 0


DAG_ID = "jobset_ttr_pod_delete_one_file"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

# Keyword arguments are generated dynamically at runtime
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 5, 18),
    schedule=SCHEDULE if composer_env.is_prod_env() else "15 */2 * * *",
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu-observability",
        "delete_pod",
        "TPU",
        "v6e-16",
        "ttr-fix",
    ],
    description=(
        "This DAG tests the JobSet time-to-recover metric by deleting a random "
        "pod to trigger a recovery, then polls the metric to check if it is"
        " updated."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Random Pod Deletion

      ### Description
      This DAG verifies that JobSet can recover from a single pod failure.
      It launches a JobSet, deletes one running pod, and then uses a sensor
      to confirm that the JobSet controller triggers a restart.

      ### Prerequisites
      This test requires an existing cluster to run.

      ### Procedures
      First the node-pool is created, a jobset yaml is then launched on the
      cluster and given a short period of time to initialize. After this a
      random pod deletion is triggered to interrupt the jobset. A sensor is
      finally run which will poll Cloud Monitoring to detect that the jobset
      time-to-recover (TTR) metric has been updated, resulting in a success,
      or timeout, and fail.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime
    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      cluster_info = build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      jobset_config = build_jobset_from_gcs_yaml(
          image=DockerImage.TPU_OBS_LIBTPU_STABLE.value,
      )

      selector = jobset.generate_node_pool_selector(DAG_ID)
      jobset_name = jobset.generate_jobset_name("ttr-pod-delete")

      start_workload = jobset.run_workload.override(task_id="start_workload")(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
          node_pool_selector=selector,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      ensure_all_pods_running = jobset.wait_for_all_pods_running.override(
          task_id="ensure_all_pods_running"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      deletion_start_time = delete_one_random_pod.override(
          task_id="delete_random_pod"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      wait_for_recovery = wait_for_jobset_recovered.override(
          task_id="wait_for_recovery"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      verify_duration = verify_recovery_duration.override(
          task_id="verify_recovery_duration"
      )(
          start_time=deletion_start_time,
          end_time=wait_for_recovery,
      )

      wait_for_metric_upload = wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_jobset_ttr_to_be_found",
      )(
          node_pool=cluster_info,
          jobset_name=jobset_name,
          start_time=deletion_start_time,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      chain(
          selector,
          jobset_name,
          jobset_config,
          cluster_info,
          start_workload,
          ensure_all_pods_running,
          deletion_start_time,
          wait_for_recovery,
          verify_duration,
          wait_for_metric_upload,
          cleanup_workload,
      )
