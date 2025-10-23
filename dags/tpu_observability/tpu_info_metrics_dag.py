"""
This script uses a factory pattern to dynamically generate an Airflow DAG
for each metric verification strategy.
"""
import datetime
import logging
import os
import re
import subprocess

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from google.cloud.monitoring_v3 import types

from dags.common.vm_resource import MachineVersion
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.jobset_util import JobSet
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.utils.monitoring import query_time_series
from dags.tpu_observability.utils.node_pool_util import Info
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.metric_strategies import BaseMetricStrategy
from dags.tpu_observability.metric_strategies import ALL_METRIC_STRATEGIES


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
        f"For pod {pod_name} ({metric_display_name}), data count mismatch. TPU-Info has"
        f" {len(cmd_values)} values, Monitoring has {len(monitoring_values)}."
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
        "Values do not match within {tolerance_percent}% tolerance."
    )
  logging.info(
      "Overall Result for Pod %s (%s): PASS", pod_name, metric_display_name
  )


@task
def get_tpu_info_metric_from_pod(
    kubeconfig: str, pod_name: str, namespace: str, metric_name: str
) -> str:
  """Executes the 'tpu-info' command in the specified pod and returns its output."""
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  result = subprocess.run(
      (
          f"kubectl --kubeconfig={kubeconfig} "
          f"exec {pod_name} -n {namespace} "
          f"-- tpu-info --metric {metric_name}"
      ),
      shell=True,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )
  logging.info("STDOUT: %s", result.stdout)
  return result.stdout


@task
def run_metric_verification(
    node_pool: Info,
    job_apply_time: datetime.datetime,
    metric_strategy: BaseMetricStrategy,
    comparison_data: tuple[str, list[tpu_info.Table]],
):
  """A generic task that uses a strategy object to verify a metric."""
  pod_name, tpu_info_output = comparison_data
  metric_name = metric_strategy.metric_name
  logging.info("Verifying metric '%s' for pod: %s...", metric_name, pod_name)

  end_time_datetime = job_apply_time + datetime.timedelta(minutes=10)
  start_time = TimeUtil.from_datetime(job_apply_time)
  end_time = TimeUtil.from_datetime(end_time_datetime)

  filter_string = [
      f'metric.type = "{metric_name}"',
      f'resource.labels.cluster_name = "{node_pool.cluster_name}"',
      f'resource.labels.pod_name = "{pod_name}"',
  ]
  time_series_data = query_time_series(
      project_id=node_pool.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
      view=types.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  monitoring_values = metric_strategy.parse_from_monitoring(time_series_data)
  util_values = metric_strategy.parse_from_tpu_info(tpu_info_output)

  tolerance_for_metric = metric_strategy.tolerance_percent
  logging.info(
      "Using a tolerance of %.2f%% for metric '%s' comparison.",
      tolerance_for_metric,
      metric_strategy.dag_id_suffix,
  )

  compare_metric_values(
      util_values,
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


with models.DAG(
    dag_id="tpu_info_metrics_verification",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=constants.Schedule.WEEKDAY_PST_6_30PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info", "unified"],
    description="Verifies multiple TPU metrics in a single DAG using TaskGroups.",
) as dag:
  cluster_info = node_pool.Info(
      project_id=models.Variable.get(
          "TCU_PROJECT_ID", default_var="cienet-cmcs"
      ),
      cluster_name=models.Variable.get(
          "TCU_CLUSTER_NAME", default_var="yuna-xpk-v6e-ew4"
      ),
      node_pool_name=models.Variable.get(
          "TCU_NODE_POOL_NAME", default_var="yuna-xpk-v6e-ew4-np-0"
      ),
      region=models.Variable.get("TCU_REGION", default_var="europe-west4"),
      location=models.Variable.get("TCU_LOCATION", default_var="europe-west4"),
      node_locations=models.Variable.get(
          "TCU_NODE_LOCATIONS", default_var="europe-west4-a"
      ),
      num_nodes=models.Variable.get("TCU_NUM_NODES", default_var=4),
      machine_type=models.Variable.get(
          "TCU_MACHINE_TYPE", default_var=MachineVersion.CT6E_STAND_4T.value
      ),
      tpu_topology=models.Variable.get("TCU_TPU_TOPOLOGY", default_var="4x4"),
  )

  kubeconfig_path = "/tmp/kubeconfig"
  jobset_config = JobSet(
      jobset_name=f"tpu-info-v6e-workload",
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

  workload_script = Workload.JAX_TPU_BENCHMARK
  apply_time = jobset.run_workload(
      node_pool=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=jobset_config.generate_yaml(workload_script=workload_script),
      namespace=jobset_config.namespace,
  )

  active_pods = jobset.get_active_pods.override(task_id="get_active_pod")(
      node_pool=cluster_info,
      kubeconfig=kubeconfig_path,
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
              kubeconfig=kubeconfig_path,
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
      kubeconfig=kubeconfig_path,
      jobset_name=jobset_config.jobset_name,
      namespace=jobset_config.namespace,
  ).as_teardown(
      setups=apply_time
  )

  (
      apply_time
      >> active_pods
      >> wait_for_job_start
      >> all_verification_groups
      >> summary
      >> clean_up_workload
  )
