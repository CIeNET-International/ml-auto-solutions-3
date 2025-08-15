"""A DAG to ensure a rollback effects the availablility of a mult-host GKE node pool as expected."""

import datetime
import logging

from airflow import models
from airflow.decorators import task
from airflow.models import Variable
from google.cloud import monitoring_v3

from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observablity.utils import node_pool_util as node_pool


@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def wait_node_pool_availability(
    node_pool: node_pool.Info,
    availability: bool,
    **context,
) -> bool:
  """Check current multi-host nodepool availability.

  This is a sensor task which runs every 30s for 900s. The task takes
  the current list of the multi_host availability outputs for the last 5
  minutes aggregated to 1 minute intervals. The results are listed, and
  the most recent result is checked to determine if it matches
  specified result, True or False.

  Args:
      node_pool: An instance of the Info class that encapsulates
        the configuration and metadata of a GKE node pool.
      availability(bool): True if the function is checking for the
        nodepool to become available, False if the function is checking for
        it to become unavailble.
      context: The Airflow context dictionary, which includes task metadata.

  """
  # Since the time must stay fixed we have to take it from the context "ti",
  # which contains the timestamp of when the task is first called, meaning.
  # it will not change with each sensor call
  task_instance = context["ti"]
  seconds = int(task_instance.start_date.timestamp())
  logging.info("start_date: %s", seconds)

  api_client = monitoring_v3.MetricServiceClient()

  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{node_pool.project_id}",
      filter=(
          'metric.type="kubernetes.io/node_pool/multi_host/available" '
          f'resource.labels.project_id = "{node_pool.project_id}" '
          f'resource.labels.cluster_name="{node_pool.cluster_name}" '
          f'resource.labels.node_pool_name="{node_pool.node_pool_name}"'
      ),
      interval=monitoring_v3.TimeInterval({
          "end_time": {"seconds": seconds},
          # Metrics are sampled every 60s and stored in the GCP backend,
          # but it may take up to 2 minute for the metric data to become
          # available on the client side.
          # Therefore, a longer time interval is necessary.
          # A 5-minute window is an arbitrary but sufficient choice to
          # ensure we can retrieve the latest metric data.
          "start_time": {"seconds": seconds - 300},
      }),
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  page_result = api_client.list_time_series(request=request)

  # We only want the most recent point, so we record all points in all
  # time series in a dictionary with their corresponding bool values to
  # ensure no overlapping time series can interfere.
  records = []
  for time_series in page_result:
    for point in time_series.points:
      end_ts_dt = point.interval.end_time
      pb = monitoring_v3.TypedValue.pb
      if pb(point.value).WhichOneof("value") == "bool_value":
        records.append((end_ts_dt, point.value.bool_value))

  if not records:
    logging.info("No records returned")
    return False

  _, state = max(records, key=lambda x: x[0])

  timeout = context["task"].timeout
  logging.info(
      "Waiting for node pool '%s' to become '%s' within %s seconds...",
      node_pool.node_pool_name,
      availability,
      timeout,
  )
  return availability == state


with models.DAG(
    dag_id="multi-host-availability-rollback",
    start_date=datetime.datetime(2025, 8, 10),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "multi-host-availability",
        "tpu_obervability",
        "rollback",
    ],
    description=(
        "This DAG tests the use of a node-pool rollback to interrupt a "
        "multi-host node-pool and ensures the node-pool is interrupted and "
        "then recovers"
    ),
    doc_md="""
    # Multi-host Node-Pool Availability Test Using Node-Pool Rollback

    ### Description
    This DAG automates the process of creating a multi-host node-pool, then
    using a node-pool rollback to interrupt the node-pool, while checking if
    the availability is correct at each step. Finally the DAG cleans up the
    node-pool which was created.

    ### Prerequisites
    This test requires an existing cluster.

    ### Procedures
    First the node-pool is created, if it found to be available the rollback
    is run. Once the rollback is finished the node-pool availability is
    tested to make sure the interruption was recorded. Afterwards, a final
    measurement is taken to ensure that the node-pool recovers from the
    inerrupt. If all of these tasks succeed than the test is successful.
    """,
) as dag:
  node_pool_info = node_pool.Info(
      project_id=Project.TPU_PROD_ENV_ONE_VM.value,
      cluster_name=Variable.get(
          "CLUSTER_NAME", default_var="qmcgarry-auto-test"
      ),
      node_pool_name=Variable.get(
          "NODE_POOL_NAME", default_var="nodepool-auto"
      ),
      location=Variable.get(
          "LOCATION", default_var=Region.ASIA_NORTHEAST1.value
      ),
      node_locations=Variable.get(
          "NODE_LOCATIONS", default_var=Zone.ASIA_NORTHEAST1_B.value
      ),
      num_nodes=Variable.get("NUM_NODES", default_var=4),
      machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t"),
      tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="4x4"),
  )

  task_id = "create_node_pool"
  create_node_pool = node_pool.create.override(task_id=task_id)(
      node_pool=node_pool_info
  )

  wait_node_pool_available = wait_node_pool_availability(
      node_pool=node_pool_info, availability=True
  )

  task_id = "rollback"
  rollback_node_pool = node_pool.rollback.override(task_id=task_id)(
      node_pool=node_pool_info
  )

  wait_node_pool_unavailable = wait_node_pool_availability(
      node_pool=node_pool_info, availability=False
  )

  # A successful rollback means the availability will return.
  # The end of the "rollback" task marks the start the availability
  # so this task should see the update and return True.
  wait_node_pool_recovered = wait_node_pool_availability(
      node_pool=node_pool_info, availability=True
  )

  task_id = "cleanup_node_pool"
  cleanup_node_pool = node_pool.delete.override(
      task_id=task_id, trigger_rule="all_done"
  )(node_pool=node_pool_info).as_teardown(
      setups=create_node_pool,
  )

  (
      create_node_pool
      >> wait_node_pool_available
      >> rollback_node_pool
      >> wait_node_pool_unavailable
      >> wait_node_pool_recovered
      >> cleanup_node_pool
  )
