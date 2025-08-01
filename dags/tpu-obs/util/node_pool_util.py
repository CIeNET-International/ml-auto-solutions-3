"""
Manages the lifecycle of a GKE node pool and verifies its status as an Airflow DAG.
"""

import dataclasses
import enum
import logging
import random
import re
import subprocess
import time
from typing import List

from airflow.decorators import task
from google import auth
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types
from googleapiclient import discovery


dataclass = dataclasses.dataclass
logger = logging.getLogger(__name__)


class Status(enum.Enum):
  """Enum for GKE node pool status."""
  RUNNING = enum.auto()
  PROVISIONING = enum.auto()
  STOPPING = enum.auto()
  RECONCILING = enum.auto()
  ERROR = enum.auto()
  UNKNOWN = enum.auto()

  @staticmethod
  def from_str(s: str) -> "Status":
      """Converts a string to a Status enum member."""
      status = Status.__members__.get(s)
      if status is None:
          logging.warning("Unknown status: %s", s)
          return Status.UNKNOWN
      return status


@dataclasses.dataclass
class Info():
  """Class to hold GKE node pool configuration parameters."""
  project_id: str
  cluster_name: str
  node_pool_name: str
  location: str
  node_locations: str
  machine_type: str
  num_nodes: int
  tpu_topology: str


@task
def create(node_pool: Info, ignore_failure: bool = False) -> None:
  """Creates the GKE node pool using gcloud command."""
  command_suffix = " 2>&1 || true" if ignore_failure else ""

  command = f"""
                gcloud container node-pools create {node_pool.node_pool_name} \\
                --project={node_pool.project_id} \\
                --cluster={node_pool.cluster_name} \\
                --location={node_pool.location} \\
                --node-locations {node_pool.node_locations} \\
                --num-nodes={node_pool.num_nodes} \\
                --machine-type={node_pool.machine_type} \\
                --tpu-topology={node_pool.tpu_topology}{command_suffix}
        """
  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
  logger.debug("STDOUT message: %s", process.stdout)
  logger.debug("STDERR message: %s", process.stderr)


@task
def delete(node_pool: Info) -> None:
  """Deletes the GKE node pool using gcloud command."""
  command = f"""
                gcloud container node-pools delete {node_pool.node_pool_name} \\
                --project {node_pool.project_id} \\
                --cluster {node_pool.cluster_name} \\
                --location {node_pool.location} \\
                --quiet
        """

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
  logger.debug("STDOUT message: %s", process.stdout)
  logger.debug("STDERR message: %s", process.stderr)


def list_nodes(node_pool: Info) -> List[str]:
  """Lists all VM instances (nodes) within the specified GKE node pool.

  This method queries the Google Cloud Container API and Compute API
  to retrieve details about the nodes belonging to the configured
  node pool. It parses instance group URLs to extract node names and zones.
  Args:
      node_pool (Info): An instance of the Info class containing GKE node pool
                   configuration parameters.
  Returns:
      A dictionary where keys are node names (str) and values are
            the zones (str) where the nodes are located. Returns an empty
            dictionary if no nodes are found, if GCP clients are not
            initialized, or in case of a 404 HttpError (node pool not found).
  Raises:
      RuntimeError: If no instance groups or zone are found for the node pool.
  """
  credentials, _ = auth.default()
  container_client = discovery.build(
      "container", "v1", credentials=credentials, cache_discovery=False
  )
  compute_client = discovery.build(
      "compute", "v1", credentials=credentials, cache_discovery=False
  )

  nodepool_path = (
      f"projects/{node_pool.project_id}/locations/{node_pool.location}"
      f"/clusters/{node_pool.cluster_name}/nodePools/{node_pool.node_pool_name}"
  )
  nodepool = (
      container_client.projects().locations().clusters().nodePools()
      .get(name=nodepool_path)
      .execute()
  )

  instance_group = nodepool.get("instanceGroupUrls", [])
  if not instance_group:
    raise RuntimeError(
        f"No instance groups found for node pool {node_pool.node_pool_name}."
    )

  node_names = []
  zone = None
  for url in instance_group:
    # URLs will be in the format:
    # https://www.googleapis.com/compute/v1/projects/{project_id}/zones/{zone}/instanceGroupManagers/{instance_group_name}
    match = re.search(
        r"zones/([\w-]+)/instanceGroupManagers/([\w-]+)", url
    )
    if not match:
      logging.warning("Could not parse instance group URL: %s", url)
      continue

    zone = match.group(1)
    ig_name = match.group(2)

    instances = (
        compute_client.instanceGroups()
        .listInstances(
            project=node_pool.project_id,
            zone=zone,
            instanceGroup=ig_name,
            body={"instanceState": "ALL"},
        ).execute()
    )

    for instance_item in instances.get("items", []):
      instance_url = instance_item["instance"]
      # Regex refined to match GKE node names
      # (e.g., gke-cluster-node-xxxx)
      node_name = re.search(r"gke[\w-]+", instance_url).group()
      if node_name:
        node_names.append(node_name)
      else:
        logging.warning(
            "Could not extract node name from URL: %s", instance_url
        )
  if zone is None:
    raise RuntimeError(
        f"No zone found for node pool {node_pool.node_pool_name}."
    )
  return node_names, zone


@task
def delete_one_random_node(node_pool: Info) -> None:
  """Defines an Airflow task to delete a random node from the GKE node pool.

  This function uses Airflow's `@task` decorator to create a Python callable
  that will be executed as an Airflow task. The callable itself performs
  the node listing, selection, and deletion using `gcloud` commands.

  Args:
      node_pool (Info): An instance of the Info class containing GKE node pool
                   configuration parameters.

  Returns:
      The decorated Airflow task object, ready to be included in a task flow.
  
  Raises:
      None
  """

  nodes_list, zone = list_nodes(node_pool)
  if not nodes_list:
    raise ValueError(
        f"No nodes found in node pool '{node_pool.node_pool_name}'. "
        "Cannot proceed with node deletion."
    )

  node_to_delete = random.choice(nodes_list)
  logging.info(
      "Randomly selected node for deletion: %s",
      node_to_delete,
  )

  command = f"""
      gcloud compute instances delete {node_to_delete} \\
          --project={node_pool.project_id} \\
          --zone={zone} \\
          --quiet
      """

  process = subprocess.run(
    command, shell=True, check=True, capture_output=True, text=True
)

  logger.debug("STDOUT message: %s", process.stdout)
  logger.debug("STDERR message: %s", process.stderr)


def _query_status_metric(node_pool: Info, poke_interval: int) -> Status:
  """Queries the latest status of a given node pool via the Google Cloud Monitoring API.

  This function constructs a request to read the "status" metric for a GKE node pool.
  It fetches time series data points from the last 5 minutes and returns the status
  from the most recent data point.

  Args:
      node_pool: An object containing node pool information (project ID, cluster name, etc.).
      poke_interval: The retry interval in seconds, used for logging when no data is found.

  Returns:
      A Status Enum object representing the latest status of the node pool.
  """
  project_name = f"projects/{node_pool.project_id}"
  now = int(time.time())
  request = {
      "name": project_name,
      "filter": (
          'metric.type="kubernetes.io/node_pool/status" '
          f'resource.labels.project_id = "{node_pool.project_id}" '
          f'resource.labels.cluster_name = "{node_pool.cluster_name}" '
          f'resource.labels.node_pool_name = "{node_pool.node_pool_name}"'
      ),
      "interval": types.TimeInterval({
          "end_time": {"seconds": now},
          "start_time": {"seconds": now - 300},
      }),
      "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  }

  monitoring_client = monitoring_v3.MetricServiceClient()
  time_series_data = monitoring_client.list_time_series(request)
  records = []
  for series in time_series_data:
    np_status = series.metric.labels.get("status", "unknown").upper()
    for point in series.points:
      end_ts_dt = point.interval.end_time
      records.append((end_ts_dt, np_status))
  if not records:
    logging.info("No records found yet. Retrying in %s seconds...", poke_interval)
    return Status.UNKNOWN

  _, latest_status = max(records, key=lambda r: r[0])

  return Status.from_str(latest_status)


@task.sensor(poke_interval=60, timeout=600, mode="reschedule")
def wait_for_status(
    node_pool: Info,
    status: Status,
    **context,
) -> bool:
  """Waits for the node pool to enter the target status."""
  # Consistent with Airflow's default timeout for sensor tasks.
  poke_interval = context["task"].poke_interval
  timeout = context["task"].timeout
  logging.info(
      "Waiting for node pool '%s' status to become '%s' within %s"
      " seconds...",
      node_pool.node_pool_name,
      status.name,
      timeout,
  )

  latest_status = _query_status_metric(node_pool, poke_interval)
  return latest_status == status


