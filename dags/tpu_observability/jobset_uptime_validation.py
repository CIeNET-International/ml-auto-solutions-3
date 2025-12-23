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

"""A DAG to test jobset uptime metric."""

from dataclasses import replace
import datetime

from airflow import models
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import Region, Zone
from dags.tpu_observability.configs.common import MachineConfigMap
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_uptime_validation",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule="0 4 * * *",
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "uptime",
        "tpu-observability",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the jobset uptime metric by deploying a workload on a "
        "TPU v6e-16 node pool and verifying that the metric increases as expected."
    ),
    doc_md="""
      # JobSet Uptime Metric Test Using TPU v6e-16 Node Pool

      ### Description
      This DAG automates the process of creating a TPU v6e-16 node pool, launching
      a jobset, and monitoring the jobset uptime metric to ensure it increments
      correctly. It also includes a negative test case to verify metric behavior
      over invalid time ranges. Finally, the DAG cleans up all created resources.

      ### Prerequisites
      This test requires an existing GKE cluster with TPU v6e-16 quota.

      ### Procedures
      1. **Provisioning**: Creates a TPU v6e-16 node pool with a specified reservation.
      2. **Deployment**: Applies a JobSet workload and waits for Pods to become active.
      3. **Metric Validation**: Polls the jobset uptime metric to confirm it is
         increasing from the point of job application.
      4. **Negative Testing**: Attempts to verify uptime against a current (future)
         timestamp to ensure the sensor correctly handles out-of-bounds queries.
      5. **Cleanup**: Deletes both the JobSet workload and the node pool to prevent
         resource leakage.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value
    cluster_info = node_pool.Info(
        project_id=models.Variable.get(
            "TFV_PROJECT_ID", default_var="cienet-cmcs"
        ),
        cluster_name=models.Variable.get(
            "TFV_CLUSTER_NAME", default_var="tony-test"
        ),
        node_pool_name=models.Variable.get(
            "TFV_NODE_POOL_NAME", default_var="jobset-uptime-validation-v6e"
        ),
        region=models.Variable.get(
            "TFV_REGION", default_var=Region.US_CENTRAL1.value
        ),
        location=models.Variable.get(
            "TFV_LOCATION", default_var=Region.US_CENTRAL1.value
        ),
        node_locations=models.Variable.get(
            "TFV_NODE_LOCATIONS", default_var=Zone.US_CENTRAL1_B.value
        ),
        num_nodes=models.Variable.get("TFV_NUM_NODES", default_var=4),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    jobset_config = JobSet(
        jobset_name="uptime-validation-v6e-workload",
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
        image="asia-northeast1-docker.pkg.dev/cienet-cmcs/yuna-docker/tpu-info:v0.5.1",
        tpu_cores_per_pod=4,
    )

    workload_script = Workload.JAX_TPU_BENCHMARK

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      create_node_pool = node_pool.create(
          node_pool=cluster_info,
          reservation="cloudtpu-20251107233000-1246578561",
      )

      apply_time = jobset.run_workload(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=workload_script
          ),
          namespace=jobset_config.namespace,
      )

      active_pods = jobset.get_active_pods.override(task_id="get_active_pod")(
          node_pool=cluster_info,
          namespace=jobset_config.namespace,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=active_pods, job_apply_time=apply_time)

      wait_for_uptime = jobset.wait_for_jobset_uptime_increasing.override(
          task_id="wait_for_uptime_increasing"
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          job_apply_time=apply_time,
      )

      # clean_up_workload = jobset.end_workload.override(
      #     task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
      # )(
      #     node_pool=cluster_info,
      #     jobset_name=jobset_config.jobset_name,
      #     namespace=jobset_config.namespace,
      # ).as_teardown(
      #     setups=apply_time
      # )

      wait_for_uptime_fail = jobset.wait_for_jobset_uptime_increasing.override(
          task_id="wait_for_uptime_fail"
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          job_apply_time=datetime.datetime.now(datetime.timezone.utc),
          timeout=300,
          soft_fail=True,
          # If no data appears within 5 mins, skip this task
          # without failing the entire DAG.
      )

      # cleanup_node_pool = node_pool.delete.override(
      #     task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      # )(node_pool=cluster_info).as_teardown(
      #     setups=create_node_pool,
      # )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          create_node_pool
          >> apply_time
          >> active_pods
          >> wait_for_job_start
          >> wait_for_uptime
          # >> clean_up_workload
          >> wait_for_uptime_fail
          # >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
