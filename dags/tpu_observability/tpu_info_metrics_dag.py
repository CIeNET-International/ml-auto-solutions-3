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

"""
This script uses a factory pattern to dynamically generate an Airflow DAG for
each metric verification strategy.
"""

from dataclasses import replace
import datetime
import logging
import os
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common.vm_resource import Zone, Region
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH
from dags.tpu_observability.tpu_info_metric import ALL_METRIC_STRATEGIES
from dags.tpu_observability.tpu_info_metric import BaseMetricStrategy
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.node_pool_util import Info
from dags.tpu_observability.utils.time_util import TimeUtil


SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None


def compare_metric_values(
    cmd_values: list[float],
    monitoring_values: list[float],
    pod_name: str,
    metric_display_name: str,
    tolerance_percent: float,
):
  """Compares two lists of metric values and checks if they are within a tolerance range."""
  if len(cmd_values) != len(monitoring_values):
    raise AirflowException(
        f"For pod {pod_name} ({metric_display_name}), data count mismatch. "
        f"TPU-Info has {len(cmd_values)} values, Monitoring has "
        f"{len(monitoring_values)}."
    )

  logging.info(
      "--- Comparison Results for pod: %s, Metric: %s ---",
      pod_name,
      metric_display_name,
  )
  logging.info(
      "%-12s%-15s%-17s%-12s%-15s%-10s",
      "Device",
      "TPU-Info Val",
      "Monitoring Val",
      "Difference",
      "Allowed Diff",
      "Result",
  )
  logging.info("-" * 85)

  all_passed = True
  for i, (log_val, mon_val) in enumerate(zip(cmd_values, monitoring_values)):
    diff = abs(log_val - mon_val)
    allowed_diff = mon_val * (tolerance_percent / 100.0)
    passed = diff <= allowed_diff
    if not passed:
      all_passed = False
    logging.info(
        "%-12s%-15.2f%-17.2f%-12.2f%-15.2f%-10s",
        f"Device {i}",
        log_val,
        mon_val,
        diff,
        allowed_diff,
        "PASS" if passed else "FAIL",
    )
  logging.info("-" * 70)

  if not all_passed:
    raise AirflowException(
        f"Overall Result for Pod {pod_name} ({metric_display_name}): FAIL - "
        f"Values do not match within {tolerance_percent}% tolerance."
    )
  logging.info(
      "Overall Result for Pod %s (%s): PASS", pod_name, metric_display_name
  )


@task
def get_tpu_info_metric_from_pod(
    node_pool: node_pool.Info, pod_name: str, namespace: str, metric_name: str
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
            f"exec {pod_name} -n {namespace} "
            f"-- tpu-info --metric {metric_name}"
        ),
    ])

    result = subprocess.run_exec(
        cmd,
        env=env,
        log_command=True,
        log_output=True,
    )

    return result


@task
def run_metric_verification(
    node_pool: Info,
    job_apply_time: TimeUtil,
    metric_strategy: BaseMetricStrategy,
    comparison_data: tuple[str, list[tpu_info.Table]],
):
  """A generic task that uses a strategy object to verify a metric."""
  pod_name, tpu_info_output = comparison_data
  metric_name = metric_strategy.metric_name
  logging.info("Verifying metric '%s' for pod: %s...", metric_name, pod_name)

  start_time = job_apply_time
  end_time = job_apply_time + datetime.timedelta(minutes=10)

  time_series_data = metric_strategy.list_or_query_metric(
      project_id=node_pool.project_id,
      cluster_name=node_pool.cluster_name,
      pod_name=pod_name,
      start_time=start_time,
      end_time=end_time,
  )

  monitoring_values = metric_strategy.parse_from_monitoring(time_series_data)
  cmd_values = metric_strategy.parse_from_tpu_info(tpu_info_output)

  tolerance_for_metric = metric_strategy.tolerance_percent
  logging.info(
      "Using a tolerance of %.2f%% for metric '%s' comparison.",
      tolerance_for_metric,
      metric_strategy.dag_id_suffix,
  )

  compare_metric_values(
      cmd_values,
      monitoring_values,
      pod_name,
      metric_display_name=metric_strategy.dag_id_suffix,
      tolerance_percent=tolerance_for_metric,
  )

  return True


@task
def summarize_results(
    verification_results_dict: dict[str, list[bool]], active_pods: list[str]
):
  """
  Summarizes the results of metric verifications for all pods.

  """
  if not active_pods:
    raise AirflowException("No active nodes were found. Grand Result: SKIPPED")

  num_expected_pods = len(active_pods)
  overall_success = True
  failure_summary = []

  logging.info("--- Overall Verification Summary ---")
  logging.info("Total pods scheduled for verification: %s", num_expected_pods)
  logging.info("-" * 70)
  logging.info("%-35s | %-10s | %-20s", "Metric Name", "Result", "Details")
  logging.info("-" * 70)

  for metric_name, results in verification_results_dict.items():
    num_passes = len(results)  # Only successed task return result

    if num_passes < num_expected_pods:
      status = "FAIL"
      details = f"Passed {num_passes} of {num_expected_pods} pods."
      overall_success = False
      failure_summary.append(f"- {metric_name}: {details}")
    else:
      status = "PASS"
      details = f"All {num_expected_pods} pods passed."

    logging.info("%-35s | %-10s | %-20s", metric_name, status, details)

  logging.info("-" * 70)

  if not overall_success:
    error_message = (
        "Grand Result: FAILURE - One or more metric verifications failed.\n"
        "Failure Details:\n" + "\n".join(failure_summary)
    )
    raise AirflowException(error_message)

  logging.info(
      "Grand Result: SUCCESS - All metric verifications passed for all pods."
  )


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(
    dag_id="tpu_info_metrics_verification",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=SCHEDULE,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info", "unified"],
    description=(
        "Verifies multiple TPU metrics in a single DAG using TaskGroups."
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

    jobset_config = jobset.JobSet(
        jobset_name="tpu-info-v6e-workload",
        namespace="default",
        max_restarts=5,
        replicated_job_name="tpu-job-slice",
        replicas=2,
        backoff_limit=0,
        completions=4,
        parallelism=4,
        tpu_accelerator_type="tpu-v6e-slice",
        tpu_topology="4x4",
        container_name="jax-tpu-worker",
        image="asia-northeast1-docker.pkg.dev/cienet-cmcs/yuna-docker/tpu-info:v0.5.1",
        tpu_cores_per_pod=4,
    )

    workload_script = jobset.Workload.JAX_TPU_BENCHMARK

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="tpu_info_format_validation_dag",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      cluster_info_2 = node_pool.copy_node_pool_info_with_override(
          info=cluster_info,
          node_pool_name=generate_second_node_pool_name(cluster_info),
      )

      with TaskGroup(group_id="create_node_pool") as create_node_pool:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1",
            retries=2,
        )(
            node_pool=cluster_info,
        )

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
            retries=2,
        )(
            node_pool=cluster_info_2,
        )
      apply_time = jobset.run_workload(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=workload_script
          ),
          namespace=jobset_config.namespace,
      )

      active_pods = jobset.list_pod_names.override(task_id="get_active_pod")(
          node_pool=cluster_info,
          namespace=jobset_config.namespace,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=active_pods, job_apply_time=apply_time)

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
                  namespace=jobset_config.namespace,
                  metric_name=strategy.tpu_info_metric_name,
              )
              .expand(pod_name=active_pods)
          )

          tpu_info_metric_output = (
              tpu_info.parse_tpu_info_output.override(
                  task_id="get_each_metric_table"
              )
              .partial()
              .expand(output=tpu_info_metric_outputs)
          )

          verify_metric = (
              run_metric_verification.override(task_id="run_verification")
              .partial(
                  node_pool=cluster_info,
                  job_apply_time=apply_time,
                  metric_strategy=strategy,
              )
              .expand(comparison_data=active_pods.zip(tpu_info_metric_output))
          )

        all_verification_groups.append(verification_group)

        verification_results[strategy.dag_id_suffix] = verify_metric

      summary = summarize_results.override(
          task_id="summarize_results", trigger_rule=TriggerRule.ALL_DONE
      )(
          verification_results_dict=verification_results,
          active_pods=active_pods,
      )

      clean_up_workload = jobset.end_workload.override(
          task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          namespace=jobset_config.namespace,
      ).as_teardown(
          setups=apply_time
      )

      with TaskGroup(group_id="cleanup_node_pool") as cleanup_node_pool:
        cleanup_first_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_1",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info).as_teardown(
            setups=create_node_pool,
        )

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info_2).as_teardown(
            setups=create_node_pool,
        )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      [create_first_node_pool, create_second_node_pool]
      chain(cleanup_first_node_pool, cleanup_second_node_pool)

      chain(
          create_node_pool,
          apply_time,
          active_pods,
          wait_for_job_start,
          all_verification_groups,
          summary,
          clean_up_workload,
          cleanup_node_pool,
      )
      # pylint: enable=pointless-statement
