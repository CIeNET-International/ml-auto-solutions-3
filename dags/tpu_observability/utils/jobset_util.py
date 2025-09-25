"""Utilities for managing JobSets in GKE clusters for TPU observability."""

import datetime
import logging
import os
import random
import subprocess
from typing import Final

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_generator import JobSet
from dags.tpu_observability.utils.monitoring import query_time_series
from dags.tpu_observability.utils.time_util import TimeUtil


def _get_credentials_command(node_pool: node_pool.Info):
  for attr_name in ["cluster_name", "region", "project_id"]:
    if not getattr(node_pool, attr_name):
      raise ValueError(f"{attr_name} must be set in the Info object.")

  return " ".join([
      "gcloud container clusters",
      f"get-credentials {node_pool.cluster_name}",
      f"--region={node_pool.region}",
      f"--project={node_pool.project_id}",
  ])


def _k8s_apply_jobset_command(
    kubeconfig: str, yaml_content: str, namespace: str
):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} apply",
      f"-f - -n {namespace} <<EOF\n",
      f"{yaml_content}\nEOF",
  ])


def _k8s_delete_jobset_command(kubeconfig: str, namespace: str):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} delete jobsets --all",
      f"-n {namespace} --timeout=60s --ignore-not-found=true",
  ])


def _k8s_get_pod_name_command(kubeconfig: str, namespace: str):
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} get pods",
      f"-n {namespace} -o jsonpath={{.items[*].metadata.name}}",
  ])


@task
def run_workload(
    info: node_pool.Info, kubeconfig: str, yaml_config: str, namespace: str
) -> TimeUtil:
  """Applies the specified YAML file to the GKE cluster.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    yaml_config: The JobSet object containing YAML configuration.
    namespace: The Kubernetes namespace to apply the JobSet.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  result = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_apply_jobset_command(kubeconfig, yaml_config, namespace),
      ]),
      shell=True,
      check=False,
      env=env,
      capture_output=True,
      text=True,
  )
  logging.info(
      "Command Execute:\n %s",
      " && ".join([
          _get_credentials_command(info),
          _k8s_apply_jobset_command(kubeconfig, yaml_config, namespace),
      ]),
  )
  if result.returncode != 0:
    raise AirflowFailException(
        f"Command failed with exit code {result.returncode}.\n ,STDERR"
        f" message: {result.stderr}"
    )
  logging.info("STDOUT message: %s", result.stdout)

  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  return current_time_utc


@task
def end_workload(info: node_pool.Info, kubeconfig: str, namespace: str):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    namespace: The YamlConfig object containing namespace information.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  result = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_delete_jobset_command(kubeconfig, namespace),
      ]),
      shell=True,
      check=False,
      env=env,
      capture_output=True,
      text=True,
  )
  if result.returncode != 0:
    logging.info("Command failed with exit code %s.", result.returncode)
    logging.info("STDERR message: %s", result.stderr)
  logging.info("STDOUT message: %s", result.stdout)


@task
def get_active_pods(info: node_pool.Info, kubeconfig: str, namespace: str):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    namespace: The YamlConfig object containing namespace information.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  process = subprocess.run(
      " && ".join([
          _get_credentials_command(info),
          _k8s_get_pod_name_command(kubeconfig, namespace),
      ]),
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )

  if not process or not process.stdout.strip():
    logging.warning("Received empty pod list from bash task.")
    raise AirflowFailException("Received empty pod list from bash task.")

  pod_list = process.stdout.strip().split()
  return pod_list


@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def wait_for_jobset_started(
    info: node_pool.Info, pod_name_list: str, job_apply_time: datetime.datetime
) -> bool:
  """Waits for the jobset to start by polling Cloud Logging for positive tensorcore utilization metrics.

  This task polls Cloud Logging for a specific log pattern that appears
  shortly after the TPU job begins execution within the specified container.
  It times out if no such log is found within a defined period.

  Args:
    info: An Info dataclass instance containing project and cluster details.
    pod_name_list: A list of pod names.
    job_apply_time: The datetime object of the time the job was applied.
  """

  end_time_datatime = job_apply_time + datetime.timedelta(minutes=10)
  start_time = TimeUtil.from_datetime(job_apply_time)
  end_time = TimeUtil.from_datetime(end_time_datatime)

  if not pod_name_list:
    raise AirflowFailException("pod_name_list is empty, sensor cannot proceed.")

  pod_name = random.choice(pod_name_list)
  metric_name = "kubernetes.io/container/accelerator/tensorcore_utilization"
  filter_string = [
      f'metric.type = "{metric_name}"',
      f'resource.labels.cluster_name = "{info.cluster_name}"',
      f'resource.labels.pod_name = "{pod_name}"',
  ]
  time_series_data = query_time_series(
      project_id=info.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
      view="FULL",
  )

  # The value of this metric means percentage of tensorcore utilization,
  # any positive values can represent that the jobset has started.
  threshold_value: Final[float] = 0.0

  # The minimum number of consecutive initial data points that must all exceed
  # 'threshold_value' to confirm that the jobset has successfully started and
  # is active.
  threshold_records_count: Final[int] = 3

  if (
      not time_series_data
      or len(time_series_data[0].points) < threshold_records_count
  ):
    return False
  last_n_data_points = [
      round(point.value.double_value, 2)
      for point in time_series_data[0].points[0:threshold_records_count]
  ]

  return all(p > threshold_value for p in last_n_data_points)

