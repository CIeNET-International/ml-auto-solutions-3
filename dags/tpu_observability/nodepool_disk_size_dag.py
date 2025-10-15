import time
from datetime import datetime, timezone
import os
import subprocess

from kubernetes import client as k8s_client, config as k8s_config
from google.cloud.container_v1 import ClusterManagerClient
from google.cloud import container_v1
from google.cloud import monitoring_v3
from airflow.providers.standard.operators.bash import BashOperator
from airflow.decorators import task, dag
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow import models
from airflow.models import Variable

# from dags.common.vm_resource import Project, Region, Zone
# from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import node_pool_util as node_pool


QUERY_WINDOW_DURATION_SECONDS = 600




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
    # always return full resource name
    return (
        op_id_or_name if str(op_id_or_name).startswith("projects/")
        else f"projects/{project_id}/locations/{location}/operations/{op_id_or_name}"
    )


@task.sensor(poke_interval=60, timeout=2700, mode="reschedule")
def wait_for_nodepool_metrics_event(
    filter_query: str,
    project_id: str,
    anchor_seconds: int | None = None,
    window_seconds: int = QUERY_WINDOW_DURATION_SECONDS,
) -> bool:
    """Poll a fixed Monitoring window around the (fixed) event time."""
    anchor = int(anchor_seconds if anchor_seconds is not None else time.time())
    interval = monitoring_v3.TimeInterval({
        "start_time": {"seconds": anchor - window_seconds},
        "end_time": {"seconds": anchor + window_seconds},
    })

    request = monitoring_v3.ListTimeSeriesRequest(
        name=f"projects/{project_id}",
        filter=filter_query,
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )
    client = monitoring_v3.MetricServiceClient()
    try:
        resp = client.list_time_series(request=request)

        print("******Poking for uptime******")
        print({"start_seconds": anchor - window_seconds,
               "end_seconds": anchor + window_seconds})

        if resp.time_series:
            print("Event detected")
            return True
        else:
            print("No time series found. Retrying...")
            return False
    except Exception as e:
        print(f"ERROR during metric check: {e}. Will repoke.")
        return False


@task(task_id="check_for_negative")
def check_for_negative(cluster: str, region: str, project: str) -> None:
    desc = subprocess.run(
        [
            "gcloud", "container", "clusters", "describe", cluster,
            "--region", region,
            "--project", project,
            "--format", "value(locationType)",
        ],
        capture_output=True, text=True, check=False
    )
    if desc.returncode != 0:
        print("FAILED: gcloud container clusters describe")
        print("--- STDOUT ---")
        print(desc.stdout)
        print("--- STDERR ---")
        print(desc.stderr)
        raise RuntimeError("Cluster describe failed")
    
    # 1) fetch kubeconfig with gcloud
    kubeconfig = "/tmp/kubeconfig"
    try:
        env = os.environ.copy()
        env["KUBECONFIG"] = kubeconfig
        subprocess.run(
            [
                "gcloud", "container", "clusters", "get-credentials", cluster,
                "--region", region,
                "--project", project,
                "--verbosity", "debug",
            ],
            env=env,
            capture_output=True,  # <— so e.stdout/e.stderr are populated
            text=True,
            check=True,           # <— raises on non-zero
        )
    except subprocess.CalledProcessError as e:
        print("FAILED: gcloud container clusters get-credentials")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise

    # 2) use Kubernetes Python client to read node readiness
    k8s_config.load_kube_config(config_file=kubeconfig)
    v1 = k8s_client.CoreV1Api()

    all_ready = True
    for node in v1.list_node().items:  # V1NodeList
        name = node.metadata.name
        ready = next((c for c in (node.status.conditions or []) if c.type == "Ready"), None)
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

    # Log current status each poke (no deduping)
    print(f"[wait] status={status}")

    if status == "DONE":
        end_str = _iso_or_now(getattr(op, "end_time", None))
        ctx["ti"].xcom_push(key="op_end", value=end_str)
        print(f"[wait] DONE start={_to_iso(getattr(op,'start_time', None))} end={end_str}")
        if op.error and (op.error.message or op.error.details):
            raise RuntimeError(f"GKE operation finished with error: {op.error}")
        return True

    # Not done yet → ask Airflow to repoke later
    print("[wait] …repoking")
    return False

@task
def note_down_duration(update_result: dict, **ctx) -> int:
    """
    Uses update_result['start_ts'] and XCom 'op_end' to compute duration.
    Logs the 150s rule and returns anchor_seconds (midpoint) for metric polling.
    """
    start_str = update_result["start_ts"]
    end_str = ctx["ti"].xcom_pull(task_ids="wait_for_update_to_complete", key="op_end")

    def _parse(ts):
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)

    t0 = _parse(start_str)
    t1 = _parse(end_str)
    seconds = max(0, int((t1 - t0).total_seconds()))
    print(f"[duration] start={t0.isoformat()} end={t1.isoformat()} duration={seconds}s")
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
    schedule="00 02 * * *",  # constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "nodepool_disk_size_ttr",
        "tpu_obervability",
        "time_to_recover",
    ],
    description=(
        "Validates that Cloud Monitoring metrics correctly appear after a GKE "
        "node-pool disk size update. The DAG performs a controlled node-pool "
        "update and verifies metric visibility in the Monitoring Explorer. "
        "A secondary sub-task (check_for_negative) logs whether the node restart "
        "duration is shorter or longer than 150 seconds. "
        "Note: This test must be executed on a multi-host cluster (minimum topology 2x4)."
    ),
    doc_md="""
    # Node-Pool Disk Resize Metric Test

    ### Objective
    Validate that Cloud Monitoring metrics appear after performing a **GKE node-pool disk size update**.  
    This simulates a controlled restart event to confirm metric visibility and recovery behavior.

    ---

    ### Expected Outcome
    Metrics show up in **Monitoring Explorer** after the disk size update.  
    Sub-task check_for_negative logs whether the restart duration is **shorter or longer than 150s**, helping identify possible false negatives.

    ---

    ### Notes
    Must be tested on a **multi-host cluster** (minimum topology **2×4**).  
    Task flow:  
    create_nodepool → update_nodepool_disksize → wait_for_update_to_complete → note_down_duration → check_for_negative → poll_nodepool → cleanup_nodepool
    """,
) as dag:

    node_pool_info = node_pool.Info(
        project_id="cienet-cmcs", # cienet-cmcs # tpu-prod-env-one-vm
        cluster_name=Variable.get("CLUSTER_NAME", default_var="athielee-auto-test-4"), # athielee-auto-test-4 # tpu-observability-automation
        node_pool_name=Variable.get("NODE_POOL_NAME", default_var="athie-nodepool-auto"),
        location=Variable.get("LOCATION", default_var="europe-west4"), # europe-west4 # us-east5
        node_locations=Variable.get("NODE_LOCATIONS", default_var="europe-west4-a"), # europe-west4-a # us-east5-b
        num_nodes=Variable.get("NUM_NODES", default_var=1),
        machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-1t"), # ct6e-standard-4t
        tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="1x1"), # 2x2
    )

    UPTIME_FILTER_QRY = (
        'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
        f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
    )

    # Kick off API update (returns op_full + start_ts)
    update_result = update_nodepool_disksize(node_pool_info, new_size_gb=150)

    # Wait for that exact operation to finish
    wait_done = wait_for_update_to_complete(update_result["op_full"])

    # Compute duration + anchor after update completes
    anchor = note_down_duration(update_result)

    poll_nodepool = wait_for_nodepool_metrics_event(
        filter_query=UPTIME_FILTER_QRY,
        project_id=node_pool_info.project_id,
        anchor_seconds=anchor,
    )

    # create_nodepool = node_pool.create(node_pool=node_pool_info, reservation="cloudtpu-20250131131310-2118578099")
    # create_nodepool = BashOperator(
    #     task_id="create_nodepool",
    #     bash_command=f"""
    #     if gcloud container node-pools describe {node_pool_info.node_pool_name} \\
    #     --cluster tpu-observability-automation --project tpu-prod-env-one-vm \\
    #     --region us-east5 &> /dev/null; then
    #     echo "Nodepool {node_pool_info.node_pool_name} already exists."
    #     else
    #     gcloud container node-pools create {node_pool_info.node_pool_name} \\
    #     --project tpu-prod-env-one-vm \\
    #     --cluster tpu-observability-automation \\
    #     --location us-east5 \\
    #     --node-locations us-east5-b \\
    #     --num-nodes="1" \\
    #     --machine-type "ct6e-standard-4t" \\
    #     --tpu-topology "2x2" \\
    #     --enable-gvnic \\
    #     --scopes "https://www.googleapis.com/auth/cloud-platform" \\
    #     --reservation-affinity=specific \\
    #     --reservation=cloudtpu-20250131131310-2118578099
    #     fi
    #     """,
    #     retries=3,
    #     )

    
    create_nodepool = BashOperator(
        task_id="create_nodepool",
        bash_command=f"""
        if gcloud container node-pools describe {node_pool_info.node_pool_name} \\
        --cluster {node_pool_info.cluster_name} --project {node_pool_info.project_id} \\
        --region {node_pool_info.location} &> /dev/null; then
        echo "Nodepool {node_pool_info.node_pool_name} already exists."
        else
        gcloud container node-pools create {node_pool_info.node_pool_name} \\
        --project {node_pool_info.project_id} \\
        --cluster {node_pool_info.cluster_name} \\
        --location {node_pool_info.location} \\
        --node-locations {node_pool_info.node_locations} \\
        --num-nodes="1" \\
        --machine-type "ct6e-standard-1t" \\
        --enable-gvnic \\
        --scopes "https://www.googleapis.com/auth/cloud-platform" 
        fi
        """,
        retries=3,
        )
    #         --tpu-topology "1x1" \\

    # disk_size_change = BashOperator(
    #     task_id="disk_size_change",
    #     bash_command=(
    #         f'gcloud container node-pools update {node_pool_info.node_pool_name} '
    #         f'--project={node_pool_info.project_id} '
    #         f'--cluster={node_pool_info.cluster_name} '
    #         f'--region {node_pool_info.location} '
    #         f'--disk-size=150 --quiet'
    #     ),
    #     retries=3,
    # )

    # wait_update = wait_nodepool_update_done(
    #     op_name=disk_size_change.output,    # XCom from BashOperator
    #     project_id=node_pool_info.project_id,
    #     location=node_pool_info.location,   # use --region for regional clusters; --zone for zonal
    # )


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


    # (
    #     create_nodepool >> disk_size_change >> wait_update,
    #     wait_update >> check_for_negative_task,
    #     [wait_update, check_for_negative_task] >> poll_nodepool >> cleanup_node_pool,

    # )

    (
        create_nodepool
        >> update_result
        >> wait_done
        >> anchor
        >> check_for_negative_task
        >> poll_nodepool
        >> cleanup_node_pool
    )