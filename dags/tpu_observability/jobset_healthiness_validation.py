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

"""A DAG to test "Jobset Suspended Healthiness" metric."""

import datetime

from airflow import models
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from airflow.models.baseoperator import chain

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import Workload, ReplicatedJobStatus
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout


DAG_ID = "jobset_healthiness_validation"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

FAIL_WORKLOAD = "python3 -c 'import logging; import sys; logging.error(\"Simulating Failure\"); sys.exit(1)'"

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 8, 10),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "healthiness",
        "tpu-obervability",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the 'Suspended' status of jobset healthiness by "
        "comparing the number of 'suspended' replicas before and after "
        "a jobset is running."
    ),
    doc_md="""
      # JobSet Healthiness Test For the "Suspended" Status
      ### Description
      This DAG automates the process of creating node-pools, ensuring the
      correct number of "Suspended" replicas appear, then launching a jobset on
      multiple replicas to ensure the correct number begin running.
      ### Prerequisites
      This test requires an existing cluster to run.
      ### Procedures
      First a node-pool is created. The validation test is then run to
      check if the number of "Suspended" replicas is 0. Once the jobset is
      running the jobs should quickly enter the "Ready" state. Then using
      command to suspend entire jobset. The number of found replicas is
      tested against the number of replicas which should be "Suspended".
      If they match the DAG is a success.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      selector = jobset.generate_node_pool_selector(
          "jobset-healthiness-validation"
      )

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

      startup = jobset.create_jobset_startup_group(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      with TaskGroup(group_id="validate_running_metrics") as validate_running:
        running_metrics = [
            (ReplicatedJobStatus.SPECIFIED, "USE_CONFIG_REPLICAS"),
            (ReplicatedJobStatus.ACTIVE, "USE_CONFIG_REPLICAS"),
            (ReplicatedJobStatus.READY, "USE_CONFIG_REPLICAS"),
            (ReplicatedJobStatus.FAILED, 0),
            (ReplicatedJobStatus.SUCCEEDED, 0),
            (ReplicatedJobStatus.SUSPENDED, 0),
        ]
        for status, expected in running_metrics:
          jobset.wait_for_jobset_metrics.override(
              task_id=f"wait_{status.value}"
          )(
              metric_name=status,
              expected_value=expected,
              node_pool=cluster_info,
              jobset_config=jobset_config,
          )

      suspend_action = jobset.suspended_jobset.override(
          task_id="suspend_jobset"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      with TaskGroup(
          group_id="validate_suspended_metrics"
      ) as validate_suspended:
        suspended_metrics = [
            (ReplicatedJobStatus.ACTIVE, 0),
            (ReplicatedJobStatus.SUSPENDED, "USE_CONFIG_REPLICAS"),
        ]
        for status, expected in suspended_metrics:
          jobset.wait_for_jobset_metrics.override(
              task_id=f"wait_after_suspend_{status.value}"
          )(
              metric_name=status,
              expected_value=expected,
              node_pool=cluster_info,
              jobset_config=jobset_config,
          )

      resume_action = jobset.resume_jobset.override(task_id="resume_jobset")(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      with TaskGroup(group_id="inject_and_validate_failure") as failure_test:
        cleanup_for_failure = jobset.end_workload.override(
            task_id="cleanup_before_failure_injection"
        )(
            node_pool=cluster_info,
            jobset_config=jobset_config,
        )

        start_fail_job = jobset.run_workload.override(task_id="start_fail_job")(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            workload_type=FAIL_WORKLOAD,
        )

        validate_failed_metric = jobset.wait_for_jobset_metrics.override(
            task_id="wait_for_failed_count"
        )(
            metric_name=ReplicatedJobStatus.FAILED,
            expected_value="USE_CONFIG_REPLICAS",
            node_pool=cluster_info,
            jobset_config=jobset_config,
        )

        chain(cleanup_for_failure, start_fail_job, validate_failed_metric)

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      ).as_teardown(
          setups=startup.jobset_start_time
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          selector,
          jobset_config,
          cluster_info,
          create_node_pool,
          startup.task_group,
          validate_running,
          suspend_action,
          validate_suspended,
          resume_action,
          failure_test,
          cleanup_workload,
          cleanup_node_pool,
      )
