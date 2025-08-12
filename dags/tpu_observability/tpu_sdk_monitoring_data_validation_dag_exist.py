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

"""A DAG to validate the `tpumonitoring` SDK metric data against tpu-info output."""

import datetime
import logging
import tempfile
import os

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task
from airflow.exceptions import AirflowException

from dags import composer_env
from dags.common import test_owner
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import tpu_monitoring_sdk_util as sdk
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.tpu_observability.tpu_info_metric import ALL_METRIC_STRATEGIES, BaseMetricStrategy
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout

DAG_ID = "tpu_sdk_monitoring_data_validation_exist"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)


@task
def get_tpu_info_metric_from_pod(
    node_pool: node_pool.Info,
    pod_name: str,
    jobset_config: jobset,
    metric_name: str,
) -> str:
  """Executes the 'tpu-info' command in the specified pod and returns its output."""
  with tempfile.TemporaryDirectory() as tmpdir:
    kube_dir = tmpdir + "/kubeconfig"
    env = os.environ.copy()
    env["KUBECONFIG"] = kube_dir

    cmd = " && ".join([
        jobset.Command.get_credentials_command(node_pool),
        (
            f"kubectl --kubeconfig={kube_dir} "
            f"exec {pod_name} -n {jobset_config.namespace} "
            f"-- tpu-info --metric {metric_name}"
        ),
    ])

    return subprocess.run_exec(cmd=cmd, env=env)


@task
def get_sdk_metric_data(
    node_pool: node_pool.Info, pod_name: str, metric_name: str
) -> str:
  """Gets raw metric data using libtpu sdk."""
  script = sdk.TpuMonitoringScript.GET_METRIC_DATA.format(
      metric_name=metric_name
  )
  return sdk.execute_sdk_command(node_pool, pod_name, script)


@task
def get_sdk_metric_description(
    node_pool: node_pool.Info, pod_name: str, metric_name: str
) -> str:
  """Gets metric description using libtpu sdk and logs it."""
  script = sdk.TpuMonitoringScript.GET_METRIC_DESCRIPTION.format(
      metric_name=metric_name
  )
  output = sdk.execute_sdk_command(node_pool, pod_name, script)
  logging.info("Metric %s description: %s", metric_name, output)
  return output


@task
def prepare_comparison_list(active_pods, tpu_info_tables, sdk_data):
  """Consolidates TPU diagnostic tables and SDK metric collections for each
  active pod.

  This function aligns hardware-level HBM tables with software-level SDK
  metrics on a per-pod basis. Each pod is mapped to a single Table object and
  a corresponding list of raw SDK strings. It handles potential telemetry gaps
  to ensure stable downstream parallel processing.

  Args:
      active_pods (list[str]): A list of active pod names (e.g., 8 pods).
      tpu_info_tables (list[Table]): A list of Table class instances
          containing HBM usage details for each pod.
      sdk_data (list[list[str]]): A nested list where each element is a
          list of raw SDK metric strings for all devices in that pod.

  Returns:
      list[dict]: A list of task-ready dictionaries. Each dictionary contains:
          - 'pod_name' (str): The name of the pod.
          - 'tpu_info' (Table | None): The TPU info table for the pod.
          - 'sdk_data_list' (list[str] | None): The list of SDK metrics for
          the pod.
  """
  comparison_data_list = []

  for i in range(len(active_pods)):
    tpu_table = tpu_info_tables[i] if i < len(tpu_info_tables) else None
    sdk_list = sdk_data[i] if i < len(sdk_data) else None
    package = {
        "pod_name": active_pods[i],
        "tpu_info": tpu_table,
        "sdk_data_list": sdk_list,
    }
    comparison_data_list.append(package)

  return comparison_data_list


@task
def compare_sdk_vs_tpu_info(
    strategy: BaseMetricStrategy,
    comparison_data: tuple[str, str, str],
):
  """Compares parsed tpu-info data with parsed SDK data."""

  pod_name = comparison_data["pod_name"]
  tpu_info_output = comparison_data["tpu_info"]
  sdk_output = comparison_data["sdk_data_list"]

  logging.info("Verifying %s for pod %s", strategy.dag_id_suffix, pod_name)

  tpu_info_values = strategy.parse_from_tpu_info(tpu_info_output)
  sdk_values = strategy.parse_from_sdk(sdk_output)

  if not tpu_info_values or not sdk_values:
    logging.warning(
        "Empty data parsed. tpu-info: %s, sdk: %s",
        tpu_info_values,
        sdk_values,
    )
    return

  if len(tpu_info_values) != len(sdk_values):
    raise AirflowException(
        f"Data count mismatch for {strategy.dag_id_suffix}. "
        f"tpu-info: {len(tpu_info_values)}, sdk: {len(sdk_values)}"
    )

  tolerance = strategy.tolerance_percent
  all_passed = True

  logging.info("--- Comparison (Tolerance: %s%%) ---", tolerance)
  logging.info(
      "%-15s | %-15s | %-10s | %-10s", "TPU-Info", "SDK", "Diff", "Result"
  )

  for v1, v2 in zip(tpu_info_values, sdk_values):
    diff = abs(v1 - v2)
    allowed_diff = v2 * (tolerance / 100.0)
    passed = diff <= allowed_diff or diff < 1e-6  # small epsilon

    status = "PASS" if passed else "FAIL"
    if not passed:
      all_passed = False

    logging.info("%-15.2f | %-15.2f | %-10.2f | %-10s", v1, v2, diff, status)

  if not all_passed:
    raise AirflowException(f"Validation failed for {strategy.dag_id_suffix}")


with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 1, 13),
    default_args={"retries": 0},
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "tpu-observability",
        "TPU",
        "v6e-16",
        "tpu-monitoring-sdk",
    ],
    description=(
        "Validates tpumonitoring SDK data consistency against tpu-info output."
    ),
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    @task
    def generate_second_node_pool_name(
        node_pool_info: node_pool.Info,
    ) -> str:
      """Generates a second node pool name."""
      return f"{node_pool_info.node_pool_name}-2"

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="tpu_sdk_monitoring_data_validation",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name="tpu_sdk_monitoring_data_validation",
      )

      selector = jobset.generate_node_pool_selector(DAG_ID)
      jobset_name = jobset.generate_jobset_name(jobset_config.dag_id_prefix)

      cluster_info_2 = node_pool.copy_node_pool_info_with_override(
          info=cluster_info,
          node_pool_name=generate_second_node_pool_name(cluster_info),
      )

      # with TaskGroup(group_id="create_node_pool") as create_node_pool:
      #   create_first_node_pool = node_pool.create.override(
      #       task_id="node_pool_1",
      #   )(
      #       node_pool=cluster_info,
      #   )

      #   create_second_node_pool = node_pool.create.override(
      #       task_id="node_pool_2",
      #   )(
      #       node_pool=cluster_info_2,
      #   )

      #   _ = [create_first_node_pool, create_second_node_pool]

      apply_time = jobset.run_workload.override(task_id="run_workload")(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      pod_names = jobset.list_pod_names.override(
          task_id="list_pod_names",
          retries=5,
          retry_delay=datetime.timedelta(seconds=10),
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      wait_for_jobset_started = jobset.wait_for_jobset_started.override(
          task_id="wait_for_jobset_started"
      )(
          node_pool=cluster_info,
          pod_name_list=pod_names,
          job_apply_time=apply_time,
      )

      verification_results = {}
      all_verification_groups = []

      for strategy in ALL_METRIC_STRATEGIES:
        group_id = f"verify_{strategy.dag_id_suffix}"

        with TaskGroup(group_id=group_id) as verification_group:
          tpu_info_metric_outputs = (
              get_tpu_info_metric_from_pod.override(
                  task_id="get_tpu_info_metric_table"
              )
              .partial(
                  node_pool=cluster_info,
                  jobset_config=jobset_config,
                  metric_name=strategy.tpu_info_metric_name,
              )
              .expand(pod_name=pod_names)
          )

          tpu_info_tables = (
              tpu_info.parse_tpu_info_output.override(task_id="parse_tpu_info")
              .partial()
              .expand(output=tpu_info_metric_outputs)
          )

          sdk_data_raw = (
              get_sdk_metric_data.override(task_id="get_sdk_data")
              .partial(
                  node_pool=cluster_info,
                  metric_name=strategy.tpu_sdk_metric_name,
              )
              .expand(pod_name=pod_names)
          )

          get_sdk_metric_description.override(
              task_id="log_description"
          ).partial(
              node_pool=cluster_info, metric_name=strategy.tpu_sdk_metric_name
          ).expand(
              pod_name=pod_names
          )

          package_data = prepare_comparison_list.override(
              task_id="package_data"
          )(pod_names, tpu_info_tables, sdk_data_raw)

          compare_sdk_vs_tpu_info.override(task_id="compare").partial(
              strategy=strategy
          ).expand(comparison_data=package_data)

        all_verification_groups.append(verification_group)

      clean_up_workload = jobset.end_workload.override(
          task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      # with TaskGroup(group_id="cleanup_node_pool") as cleanup_node_pool:
      #   cleanup_first_node_pool = node_pool.delete.override(
      #       task_id="cleanup_node_pool_1",
      #       trigger_rule=TriggerRule.ALL_DONE,
      #       retries=2,
      #   )(node_pool=cluster_info)

      #   cleanup_second_node_pool = node_pool.delete.override(
      #       task_id="cleanup_node_pool_2",
      #       trigger_rule=TriggerRule.ALL_DONE,
      #       retries=2,
      #   )(node_pool=cluster_info_2)

      #   chain(cleanup_first_node_pool, cleanup_second_node_pool)

      chain(
          apply_time,
          pod_names,
          wait_for_jobset_started,
          all_verification_groups,
          clean_up_workload,
      )
