import time
from datetime import datetime, timezone
import os
import subprocess

from kubernetes import client as k8s_client, config as k8s_config
from google.cloud.container_v1 import ClusterManagerClient
from google.cloud import container_v1
from google.cloud import monitoring_v3
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.decorators import task, dag
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow import models
from dags.tpu_observability.utils import node_pool_util as node_pool

QUERY_WINDOW_DURATION_SECONDS = 3600


def _to_iso(ts):
  if ts is None:
    return None
  if isinstance(ts, str):
    return ts
  for attr in ("ToDatetime", "to_datetime"):
    fn = getattr(ts, attr, None)
    if callable(fn):
      try:
        return fn(tzinfo=timezone.utc).isoformat()
      except TypeError:
        return fn().astimezone(timezone.utc).isoformat()
  return getattr(ts, "isoformat", lambda: str(ts))()


def _iso_or_now(ts):
  return _to_iso(ts) or datetime.now(timezone.utc).isoformat()


def _nodepool_name(info):
  return (
      f"projects/{info.project_id}/locations/{info.location}/"
      f"clusters/{info.cluster_name}/nodePools/{info.node_pool_name}"
  )


def _operation_name(project_id, location, op_id_or_name):
  return (
      op_id_or_name
      if str(op_id_or_name).startswith("projects/")
      else f"projects/{project_id}/locations/{location}/operations/{op_id_or_name}"
  )


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_nodepool_metrics_event(
    filter_query: str,
    project_id: str,
    anchor_seconds: int | None = None,
    window_seconds: int = QUERY_WINDOW_DURATION_SECONDS,
    give_up_grace_seconds: int = 7200,  # 2h total grace
) -> bool:
  """Poll Cloud Monitoring for node-pool recovery metrics."""
  from google.cloud import monitoring_v3
  import time

  anchor = int(anchor_seconds or time.time())
  start_s = anchor - window_seconds
  end_s = max(anchor + window_seconds, int(time.time()))

  interval = monitoring_v3.TimeInterval({
      "start_time": {"seconds": start_s},
      "end_time": {"seconds": end_s},
  })

  client = monitoring_v3.MetricServiceClient()
  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{project_id}",
      filter=filter_query,
      interval=interval,
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  try:
    series = list(client.list_time_series(request=request))
    if series:
      print(f"[poll] Metric detected ({len(series)} series).")
      return True

    if time.time() > (anchor + window_seconds + give_up_grace_seconds):
      print("[poll] Grace period exceeded — ending sensor.")
      return True

    print("[poll] No metrics yet, repoking…")
    return False

  except Exception as e:
    print(f"[poll] ERROR during metric check: {e}")
    return False


@task(task_id="check_for_negative")
def check_for_negative(cluster: str, region: str, project: str) -> None:
  desc = subprocess.run(
      [
          "gcloud",
          "container",
          "clusters",
          "describe",
          cluster,
          "--region",
          region,
          "--project",
          project,
          "--format",
          "value(locationType)",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  if desc.returncode != 0:
    print("FAILED: gcloud container clusters describe")
    print("--- STDOUT ---")
    print(desc.stdout)
    print("--- STDERR ---")
    print(desc.stderr)
    raise RuntimeError("Cluster describe failed")

  kubeconfig = "/tmp/kubeconfig"
  try:
    env = os.environ.copy()
    env["KUBECONFIG"] = kubeconfig
    subprocess.run(
        [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            cluster,
            "--region",
            region,
            "--project",
            project,
            "--verbosity",
            "debug",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
  except subprocess.CalledProcessError as e:
    print("FAILED: gcloud container clusters get-credentials")
    print("--- STDOUT ---")
    print(e.stdout)
    print("--- STDERR ---")
    print(e.stderr)
    raise

  k8s_config.load_kube_config(config_file=kubeconfig)
  v1 = k8s_client.CoreV1Api()

  all_ready = True
  for node in v1.list_node().items:  # V1NodeList
    name = node.metadata.name
    ready = next(
        (c for c in (node.status.conditions or []) if c.type == "Ready"), None
    )
    status = ready.status if ready else "Unknown"
    if status == "True":
      print(f"Node {name} is Ready")
    else:
      print(f"Node {name} is not Ready")
      all_ready = False

  if all_ready:
    print("Restart shorter than 150 seconds. This may cause a false negative")
  else:
    print("Restart longer than 150 seconds. False negative should not occur")


@task.sensor(poke_interval=30, timeout=3600, mode="reschedule")
def wait_for_update_to_complete(op_full: str, **ctx) -> bool:
  """
  Waits until the given GKE node-pool update operation reaches DONE.
  Prints status on every poke. XCom-pushes 'op_end' (ISO string) when finished.
  """
  client = ClusterManagerClient()
  op = client.get_operation(name=op_full)
  status = getattr(op.status, "name", str(op.status))

  print(f"[wait] status={status}")

  if status == "DONE":
    end_str = _iso_or_now(getattr(op, "end_time", None))
    ctx["ti"].xcom_push(key="op_end", value=end_str)
    print(
        f"[wait] DONE start={_to_iso(getattr(op, 'start_time', None))} end={end_str}"
    )
    if op.error and (op.error.message or op.error.details):
      raise RuntimeError(f"GKE operation finished with error: {op.error}")
    return True

  print("[wait] …repoking")
  return False


@task
def note_down_duration(update_result: dict, **ctx) -> int:
  """
  Uses update_result['start_ts'] and XCom 'op_end' to compute duration.
  Logs the 150s rule and returns anchor_seconds (midpoint) for metric polling.
  """
  start_str = update_result["start_ts"]
  end_str = ctx["ti"].xcom_pull(
      task_ids="wait_for_update_to_complete", key="op_end"
  )

  def _parse(ts):
    return (
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts
        else datetime.now(timezone.utc)
    )

  t0 = _parse(start_str)
  t1 = _parse(end_str)
  seconds = max(0, int((t1 - t0).total_seconds()))
  print(
      f"[duration] start={t0.isoformat()} end={t1.isoformat()} duration={seconds}s"
  )
  if seconds < 150:
    print("Restart shorter than 150 seconds. This may cause a false negative")
  else:
    print("Restart longer than 150 seconds. False negative should not occur")

  anchor = int((t0.timestamp() + t1.timestamp()) / 2)
  print(f"[anchor] anchor_seconds={anchor}")
  return anchor


@task
def update_nodepool_disksize(info: node_pool.Info, new_size_gb: int) -> dict:
  """
  Calls GKE UpdateNodePool via API and returns:
    {"op_full": <projects/.../operations/operation-...>, "start_ts": <ISO>}
  """
  cm = container_v1.ClusterManagerClient()
  req = container_v1.UpdateNodePoolRequest(name=_nodepool_name(info))
  req.disk_size_gb = int(new_size_gb)

  op = cm.update_node_pool(request=req)
  op_full = _operation_name(info.project_id, info.location, op.name)
  start_str = _iso_or_now(getattr(op, "start_time", None))

  print(f"[update] op_id={op.name}")
  print(f"[update] op_full={op_full}")
  print(f"[update] start_ts={start_str}")
  return {"op_full": op_full, "start_ts": start_str}


with models.DAG(
    dag_id="nodepool_disk_size_ttr",
    start_date=datetime(2025, 6, 26),
    schedule="00 02 * * *",
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "nodepool_disk_size_ttr",
        "tpu_obervability",
        "time_to_recover",
    ],
    description=(
        "Tests GKE node-pool recovery by resizing disks, waiting ≥150s, "
        "checking node readiness, and polling Cloud Monitoring for recovery events. "
        "Cleans up the node pool afterward."
    ),
    doc_md="""
    # Node-Pool Availability Test (Disk Resize)

    ### Purpose
    This DAG tests whether a GKE node pool remains observable and recovers
    as expected when a disk resize operation forces node restarts.

    ### Expected Outcome
    - Nodes restart and return to Ready state.
    - A recovery event is recorded in Cloud Monitoring.

    """,
) as dag:
  node_pool_info = node_pool.Info(
      project_id="cienet-cmcs",
      cluster_name=Variable.get(
          "CLUSTER_NAME", default_var="athielee-auto-test-4"
      ),
      node_pool_name=Variable.get(
          "NODE_POOL_NAME", default_var="athie-nodepool-auto"
      ),
      location=Variable.get("LOCATION", default_var="europe-west4"),
      node_locations=Variable.get(
          "NODE_LOCATIONS", default_var="europe-west4-a"
      ),
      num_nodes=Variable.get("NUM_NODES", default_var=2),
      machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t"),
      tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="2x4"),
  )

  UPTIME_FILTER_QRY = (
      'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
      f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
  )

  update_result = update_nodepool_disksize(node_pool_info, new_size_gb=150)

  wait_done = wait_for_update_to_complete(update_result["op_full"])

  anchor = note_down_duration(update_result)

  poll_nodepool = wait_for_nodepool_metrics_event(
      filter_query=UPTIME_FILTER_QRY,
      project_id=node_pool_info.project_id,
      anchor_seconds=anchor,
  )

  create_nodepool = node_pool.create(node_pool=node_pool_info)

  check_for_negative_task = check_for_negative(
      cluster=node_pool_info.cluster_name,
      region=node_pool_info.location,
      project=node_pool_info.project_id,
  )

  cleanup_node_pool = node_pool.delete.override(trigger_rule="all_done")(
      node_pool=node_pool_info
  ).as_teardown(
      setups=create_nodepool,
  )

  (
      create_nodepool
      >> update_result
      >> wait_done
      >> anchor
      >> check_for_negative_task
      >> poll_nodepool
      >> cleanup_node_pool
  )
