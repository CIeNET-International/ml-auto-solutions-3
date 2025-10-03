"""A DAG to test the jobset time-to-recover metric from a node pool rollback."""

import dataclasses
import datetime
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from google.auth import default
from google.auth.transport.requests import Request
from google.cloud import monitoring_v3
from kubernetes import client

from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_yaml_generator import create_jobset_yaml


# Will be moved to node_pool_utils
@dataclasses.dataclass
class YamlConfig:
  """A data structure to store dynamic parameters for the JobSet YAML.

  This class centralizes all configurable parts of the YAML, making DAGs
  cleaner and more maintainable.
  """

  # Metadata
  jobset_name: str
  namespace: str

  # Failure Policy
  max_restarts: int

  # ReplicatedJob Spec
  replicated_job_name: str
  replicas: int

  # Job Template Spec
  backoff_limit: int
  completions: int
  parallelism: int

  # Pod Template Spec
  node_selector: Optional[Dict[str, str]]

  # Container Spec
  container_name: str
  image: str
  tpu_cores_per_pod: int
  command: Optional[List[str]]
  command_args: Optional[List[str]]

  # Volume Spec
  volume_name: Optional[str]
  config_map_name: Optional[str]


def get_replica_num(replica_type: str, job_name: str) -> int:
  """Get the number of a certain type of replicas from a running jobset.

  This uses the Kubernetes API to connect to a desired cluster and returns
  the number of replicas in a certain status.

  Args:
    replica_type(str): The type of replica being searched for.
    job_name(str): The name of the job replica which is run from the jobset.
  Returns:
    The number of replicas of the specific type in the jobset.
  """
  creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
  creds.refresh(Request())

  configuration = client.Configuration()
  configuration.host = "https://34.34.8.85"  # cmcs europe-west4
  configuration.verify_ssl = True
  configuration.ssl_ca_cert = "/home/airflow/gcs/data/europe-gke-ca.pem"
  configuration.api_key = {"authorization": "Bearer " + creds.token}
  client.Configuration.set_default(configuration)

  api = client.CustomObjectsApi()
  jobsets = api.list_namespaced_custom_object(
      group="jobset.x-k8s.io",
      version="v1alpha2",
      namespace="default",
      plural="jobsets",
  )
  if not jobsets["items"]:
    raise AirflowException("No items found" f"JobSet output: {jobsets}")
  jobset = jobsets["items"][0]
  if (
      jobset.get("status")
      and jobset["status"].get("replicatedJobsStatus")
      and jobset["status"]["replicatedJobsStatus"][0].get("name") == job_name
      and jobset["status"]["replicatedJobsStatus"][0].get(replica_type)
      is not None
  ):
    number_replicas = jobset["status"]["replicatedJobsStatus"][0][replica_type]
    logging.info("Found %s replicas", number_replicas)
  return number_replicas


# Will be moved to a util file
@task
def run_jobset_workload(info: node_pool.Info, yaml_config: YamlConfig):
  """Generates and runs a JobSet manifest.

  Args:
    info(Info): An instance of the Info class that encapsulates
      the configuration and metadata of a GKE node pool and workload.
      yaml_config(YamlConfig): All parameters needing to generate the
      JobSet file which will be run by this function.
  """
  params = dataclasses.asdict(yaml_config)

  base_job_name = yaml_config.jobset_name

  logging.info("Generating YAML content for JobSet: %s", base_job_name)
  yaml_content = create_jobset_yaml(**params)

  output_path = f"/tmp/{base_job_name}.yaml"
  with open(output_path, "w") as f:
    f.write(yaml_content)

  env = os.environ.copy()
  env["KUBECONFIG"] = "/tmp/kubeconfig"

  gcloud_cmd = (
      f"gcloud container clusters get-credentials {info.cluster_name} "
      f"--region={info.location} "
      f"--project={info.project_id} "
  )

  kubectl_cmd = (
      f"kubectl --kubeconfig=/tmp/kubeconfig apply -f {output_path} "
      "-n default"
  )

  subprocess.run(
      " && ".join([
          gcloud_cmd, kubectl_cmd
      ]), shell=True, check=True, capture_output=True, text=True, env=env
  )
  logging.info("Successfully applied YAML to the cluster.")


@task.sensor(poke_interval=30, timeout=600, mode="reschedule")
def wait_for_jobset_status(replica_type: str, job_name: str):
  """A sensor which checks if are any jobset replicas in a status type.

  Args:
    replica_type(str): The type of status being checked for.
    job_name(str): The name of the job replica which is run from the jobset.
  """
  ready_replicas = get_replica_num(replica_type=replica_type, job_name=job_name)
  if ready_replicas > 0:
    return True


@task
def end_workload(info: node_pool.Info):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    info(Info): Configuration object with cluster details.
  """
  command = (
      "export KUBECONFIG=/tmp/kubeconfig && "
      f"gcloud container clusters get-credentials {info.cluster_name} "
      f"--region {info.location} --project {info.project_id} && "
      "kubectl delete jobsets --all -n default --timeout=60s"
  )

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )

  logging.info("STDOUT message: %s", process.stdout)
  logging.info("STDERR message: %s", process.stderr)


@task.sensor(poke_interval=60, timeout=3600, mode="reschedule")
def wait_for_jobset_ttr(info: node_pool.Info) -> bool:
  """Polls the jobset time_between_interruptions metric.

  A sensor task which polls the jobset time_between_interruptions metric
  every 60 seconds for 60 minutes.

  Args:
    info(Info): An instance of the Info class that encapsulates
    the configuration and metadata of a GKE node pool and workload.
  """
  now = int(time.time())
  api_client = monitoring_v3.MetricServiceClient()
  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{info.project_id}",
      filter=(
          'metric.type="kubernetes.io/jobset/times_to_recover" '
          f'resource.labels.cluster_name="{info.cluster_name}" '
      ),
      interval=monitoring_v3.TimeInterval({
          # This particular metric takes a long time to update
          # to GCP, typically around 20-30 minutes.
          # This means that the sensor must be long running and
          # have a long search period to detect it.
          "end_time": {"seconds": now},
          "start_time": {"seconds": now - 3600},
      }),
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )
  page_result = api_client.list_time_series(request=request)

  # We just need to know that the event happened at all
  if page_result.time_series:
    logging.info("Event detected at %s", now)
    return True
  else:
    logging.info("No time series found at %s. Continuing...", now)
  return False


with models.DAG(
    dag_id="jobset_rollback_ttr-new",
    start_date=datetime.datetime(2025, 8, 10),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu-obervability",
        "rollback",
    ],
    description=(
        "This DAG tests the use of a node-pool rollback to interrupt a "
        "jobset, then polls the jobset time-to-recover metric to check "
        "if it is updated."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Node-Pool Rollback

      ### Description
      This DAG automates the process of creating a node-pool, launching a jobset
      then using a node-pool rollback to interrupt the node-pool, and afterwards
      monitors if the jobset TTR metric gets updated. Finally the DAG cleans up
      the jobset and node-pool which were created.

      ### Prerequisites
      This test requires an existing cluster to run.

      ### Procedures
      First the node-pool is created, a jobset yaml is then launched on the
      cluster and given a short period of time to initialize. After this a
      rollback is run on the previously created node-pool to interrupt it.
      A sensor is finally run which will either detect that the jobset
      time-to-recover metric has been updated, resulting in a success, or
      timeout, and fail.
      """,
) as dag:
  cluster_info = node_pool.Info(
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

  yaml_config_instance = YamlConfig(
      jobset_name="tpu-info-v6e-workload",
      namespace="default",
      max_restarts=5,
      replicated_job_name="tpu-job-slice",
      replicas=1,
      backoff_limit=0,
      completions=4,
      parallelism=4,
      image="python:3.10",
      container_name="jax-tpu-job",
      tpu_cores_per_pod=4,
      node_selector={
          "cloud.google.com/gke-tpu-accelerator": "tpu-v6e-slice",
          "cloud.google.com/gke-tpu-topology": "4x4",
      },
      command=["bash", "-c"],
      command_args=[
          """
        pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        python -c '
        import jax
        import jax.numpy as jnp
        import time
        import os
        from jax.sharding import Mesh, NamedSharding
        from jax.experimental.pjit import pjit

        os.environ.setdefault("JAX_USE_PJIT", "true")
        jax.distributed.initialize()

        global_devices = jax.devices()
        print(f"[Host {jax.process_index()}] Got {len(global_devices)} global devices")
        mesh = Mesh(global_devices, ("x",))

        size = 32768
        x_global = jnp.ones((size, size), dtype=jnp.float32)
        y_global = jnp.ones((size, size), dtype=jnp.float32)

        sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("x", None))
        x = jax.device_put(x_global, sharding)
        y = jax.device_put(y_global, sharding)

        @pjit
        def matmul_ultra_heavy(x, y):
            tmp1 = jnp.dot(x, y)
            tmp2 = jnp.dot(tmp1, y.T)
            tmp3 = jnp.dot(tmp2, x.T)
            tmp4 = jnp.dot(tmp3, x)
            tmp5 = jnp.dot(tmp4, y)
            return tmp5

        matmul_ultra_heavy(x, y).block_until_ready()

        print(f"[Host {jax.process_index()}] Starting benchmark...")

        start = time.time()
        for i in range(1_000_000):
            result = matmul_ultra_heavy(x, y)
        result.block_until_ready()
        end = time.time()

        if jax.process_index() == 0:
            print(f"Total time: {end - start:.2f} seconds (on full v6e-16)")
        '
        """
      ],
      volume_name=None,
      config_map_name=None,
  )

  create_node_pool = node_pool.create(node_pool=cluster_info)

  start_workload = run_jobset_workload(
      info=cluster_info, yaml_config=yaml_config_instance
  )

  wait_for_jobset_ready = wait_for_jobset_status(
      replica_type="ready", job_name=yaml_config_instance.replicated_job_name
  )

  rollback_node_pool = node_pool.rollback(node_pool=cluster_info)

  wait_for_metric_upload = wait_for_jobset_ttr(info=cluster_info)

  cleanup_workload = end_workload.override(trigger_rule=TriggerRule.ALL_DONE)(
      info=cluster_info
  ).as_teardown(
      setups=start_workload,
  )

  cleanup_node_pool = node_pool.delete.override(
      trigger_rule=TriggerRule.ALL_DONE
  )(node_pool=cluster_info).as_teardown(
      setups=create_node_pool,
  )

  (
      create_node_pool
      >> start_workload
      >> wait_for_jobset_ready
      >> rollback_node_pool
      >> wait_for_metric_upload
      >> cleanup_workload
      >> cleanup_node_pool
  )
