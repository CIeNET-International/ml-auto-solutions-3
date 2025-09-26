import time
import datetime
import os
import subprocess

from kubernetes import client as k8s_client, config as k8s_config
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


with models.DAG(
    dag_id="nodepool_disk_size_ttr",
    start_date=datetime.datetime(2025, 6, 26),
    schedule="00 02 * * *",  # constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "nodepool_disk_size_ttr",
        "tpu_obervability",
        "time_to_recover",
    ],
    description=(
        "Creates a test GKE cluster/node pool (if absent), resizes the node pool disk, "
        "waits ≥150s to avoid false negatives, checks node readiness, then polls Cloud "
        "Monitoring (kubernetes.io/node_pool/accelerator/times_to_recover) until an event "
        "is detected or timeout. Always attempts to delete the node pool at the end."
    ),
    doc_md="""
    # Multi-host Node-Pool Availability Test Using Disk Resize
    ### Description
    This DAG provisions (or reuses) a GKE cluster and creates a multi-host node pool,
    performs a disk size update to intentionally trigger a node restart, enforces a
    ≥150s guard window to detect false negatives, checks node readiness, then polls
    Cloud Monitoring for the accelerator recovery metric. The DAG always attempts to
    clean up the node pool at the end.

    ### Prerequisites
    This test requires an existing cluster.

    ### Procedures
    1. Create cluster if absent (create_cluster), then create node pool if absent (create_nodepool).
    2. In parallel:
       - Update node pool disk size to 200GiB to trigger a restart (disk_size_change).
       - Wait at least 150 seconds (wait_time TimeDeltaSensor) to avoid counting very short restarts.
    3. After the wait, check node readiness with kubectl and log potential false negatives (check_for_negative).
    4. Build a time interval and poll Cloud Monitoring for
       `kubernetes.io/node_pool/accelerator/times_to_recover` using a sensor that reschedules
       between pokes (wait_for_nodepool), up to a 45-minute timeout.
    5. Always attempt to delete the node pool (delete_nodepool, ALL_DONE trigger rule).
    """,
) as dag:

    node_pool_info = node_pool.Info(
        project_id="tpu-prod-env-one-vm", # cienet-cmcs
        cluster_name=Variable.get("CLUSTER_NAME", default_var="tpu-observability-automation"), # athielee-auto-test-4
        node_pool_name=Variable.get("NODE_POOL_NAME", default_var="athie-nodepool-auto"),
        location=Variable.get("LOCATION", default_var="us-east5"), # europe-west4
        node_locations=Variable.get("NODE_LOCATIONS", default_var="us-east5-b"), # europe-west4-a
        num_nodes=Variable.get("NUM_NODES", default_var=1),
        machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t"),
        tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="2x2"),
    )

    UPTIME_FILTER_QRY = (
        'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
        f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
    )

    poll_nodepool = wait_for_nodepool_metrics_event(
        UPTIME_FILTER_QRY,
        node_pool_info.project_id,
    )

    #create_nodepool = node_pool.create(node_pool=node_pool_info)
    create_nodepool = BashOperator(
        task_id="create_nodepool",
        bash_command=f"""
        if gcloud container node-pools describe {node_pool_info.node_pool_name} \\
        --cluster tpu-observability-automation --project tpu-prod-env-one-vm \\
        --region us-east5 &> /dev/null; then
        echo "Nodepool {node_pool_info.node_pool_name} already exists."
        else
        gcloud container node-pools create {node_pool_info.node_pool_name} \\
        --project tpu-prod-env-one-vm \\
        --cluster tpu-observability-automation \\
        --location us-east5 \\
        --node-locations us-east5-b \\
        --num-nodes="1" \\
        --machine-type "ct6e-standard-4t" \\
        --tpu-topology "2x2" \\
        --enable-gvnic \\
        --scopes "https://www.googleapis.com/auth/cloud-platform" \\
        --reservation-affinity=specific \\
        --reservation=cloudtpu-20250131131310-2118578099
        fi
        """,
        retries=3,
        )
    

    disk_size_change = BashOperator(
        task_id="disk_size_change",
        bash_command=(
            f'gcloud container node-pools update {node_pool_info.node_pool_name} '
            f'--project={node_pool_info.project_id} '
            f'--cluster={node_pool_info.cluster_name} '
            f'--region {node_pool_info.location} '
            f'--disk-size=150 --quiet'
        ),
        retries=3,
    )

    wait_time = TimeDeltaSensor(
        task_id="wait_time",
        delta=datetime.timedelta(seconds=150),
    )

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
        create_nodepool >> [disk_size_change, wait_time],
        wait_time >> check_for_negative_task,
        [disk_size_change, check_for_negative_task] >> poll_nodepool >> cleanup_node_pool,
    )
