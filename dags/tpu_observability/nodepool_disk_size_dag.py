
from google.cloud import monitoring_v3
import time
import datetime
from airflow.providers.standard.operators.bash import BashOperator
from airflow.decorators import task, dag
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.time_delta import TimeDeltaSensor

from airflow import models
from airflow.models import Variable

# ################################################
# # create cluster related (will be deleted after)
# from google.cloud import container_v1
# from airflow.providers.google.cloud.operators.kubernetes_engine import GKECreateClusterOperator
# from google.cloud.container_v1.types import Cluster, PrivateClusterConfig, IPAllocationPolicy
# from google.api_core.exceptions import NotFound
# from airflow.operators.python import BranchPythonOperator
# from airflow.operators.empty import EmptyOperator
# ################################################

from dags.tpu_observability.utils import node_pool_util as node_pool



QUERY_WINDOW_DURATION_SECONDS = 600

@task.sensor(poke_interval=60, timeout=2700, mode="reschedule")
def wait_for_nodepool(
        filter_query: str,
        time_interval: dict[str, int],
        project_id: str,
) -> bool:
    """A sensor task which polls the jobset time_between_interruptions metric
    every 60 seconds for 1 hour.

    Args:
        filter_query(str): The filter query to use for the metric request.
        time_interval(dict[str:int]): A dictionary containing the start and 
        end time of theinterval to check.

    Returns:
        bool:True if an event is detected. False if 1 hour passes.
    """
    fixed_time_interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": (time_interval["end_seconds"])},
            "start_time": {"seconds": (time_interval["start_seconds"])},
        }
    )

    mon_client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    try:
        results = mon_client.list_time_series(
            request={
                "name": project_name,
                "filter": filter_query,
                "interval": fixed_time_interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )
        print("******Poking for uptime******")
        print(time_interval)
        if results.time_series:
            print("Event detected")
            return True
        else:
            print("No time series found. Retrying...")
            return False

    except Exception as e:
        print(f"ERROR during metric check: {e}. Retrying...")
        return False

# Return a time interval to search
@task
def create_time_interval():
    """Creates a time interval to search for the metric. DAGs are ephemeral 
    so alltimes must be synchronized intentinoally using this time frame.

    Returns:
        dict[str, int]: A dictionary containing the start and end time of the
        interval.
    """
    start = time.time()
    seconds = int(start)

    interval_dict = {
        "end_seconds": seconds + QUERY_WINDOW_DURATION_SECONDS,
        "start_seconds": seconds - QUERY_WINDOW_DURATION_SECONDS
    }

    print(f"Returning time interval parameters: {interval_dict}")
    return interval_dict

# # create cluster related
# def decide_create_cluster(project_id: str, location: str, cluster_name: str, **_):
    
#     """Return the next task_id: 'skip_cluster' if exists, else 'create_cluster'."""
#     client = container_v1.ClusterManagerClient()
#     name = f"projects/{project_id}/locations/{location}/clusters/{cluster_name}"
#     try:
#         client.get_cluster(name=name)
#         return "skip_cluster"
#     except NotFound:
#         return "create_cluster"
#     except Exception:
#         # If the API is temporarily flaky, try creating — downstream will error if it truly exists.
#         return "create_cluster"



with models.DAG(
    dag_id="nodepool_disk_size_ttr",
    start_date=datetime.datetime(2025, 6, 26),
    schedule="00 02 * * *", # constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY
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
    doc_md = """
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
      project_id="cienet-cmcs", # Project.TPU_PROD_ENV_ONE_VM.value
      cluster_name=Variable.get(
          "CLUSTER_NAME", default_var="athielee-auto-test-4"
      ),
      node_pool_name=Variable.get(
          "NODE_POOL_NAME", default_var="nodepool-auto"
      ),
      location=Variable.get(
          "LOCATION", default_var="europe-west4" # Region.ASIA_NORTHEAST1.value
      ),
      node_locations=Variable.get(
          "NODE_LOCATIONS", default_var="europe-west4-a" # Zone.ASIA_NORTHEAST1_B.value
      ),
      num_nodes=Variable.get("NUM_NODES", default_var=1),
      machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t"),
      tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="2x2"),
  )
  
  UPTIME_FILTER_QRY = (
        'metric.type="kubernetes.io/node_pool/accelerator/times_to_recover" '
        f'AND resource.labels.cluster_name="{node_pool_info.cluster_name}"'
  )

  time_interval = create_time_interval()
  poll_nodepool = wait_for_nodepool(UPTIME_FILTER_QRY, time_interval, node_pool_info.project_id)    
    

#   # Branch: decide whether to create the cluster
#   check_on_cluster = BranchPythonOperator(
#     task_id="check_on_cluster",
#     python_callable=decide_create_cluster,
#     op_kwargs=dict(
#         project_id=node_pool_info.project_id,
#         location=node_pool_info.location,
#         cluster_name=node_pool_info.cluster_name,
#       ),
#   )

#   skip = EmptyOperator(task_id="skip_cluster")

#   # Merge point so downstream continues whether we created or skipped
#   join_after_branch = EmptyOperator(
#     task_id="join_after_branch",
#     trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
#   )

  
#   create_cluster = GKECreateClusterOperator(
#     task_id="create_cluster",
#     project_id=node_pool_info.project_id,
#     location=node_pool_info.location,      # us-central1
#     body=Cluster(
#         name=f"{node_pool_info.cluster_name}",
#         network="athie-cmcs-vpc",
#         subnetwork="athie-subnet-euw4",
#         initial_node_count=2,
#         node_config={"machine_type": "e2-standard-8"},
#         ip_allocation_policy=IPAllocationPolicy(use_ip_aliases=True),
#         release_channel={"channel": "REGULAR"},
#     ),
#     retries=0,
#   )

  create_nodepool = node_pool.create(node_pool=node_pool_info)

  disk_size_change = BashOperator(
        task_id="disk_size_change",
        bash_command=f'gcloud container node-pools update {node_pool_info.node_pool_name} \
                --project={node_pool_info.project_id} --cluster={node_pool_info.cluster_name} --region {node_pool_info.location} \
                --disk-size=150 --quiet',
        retries=3,
  )

  # Ensures the restart time takes at least 150 seconds
  wait_time = TimeDeltaSensor(
        task_id="wait_time",
        delta=datetime.timedelta(seconds=150),
  )
    
  # Checks if nodes are in "Ready" status or not. If they all are, the restart is complete
  check_for_negative = BashOperator(
        task_id="check_for_negative",
        bash_command="""
        export KUBECONFIG=/tmp/kubeconfig

        gcloud container clusters get-credentials {{ params.cluster }} \
            --region {{ params.region }} \
            --project {{ params.project }}
            --kubeconfig $KUBECONFIG

        all_ready=true
        
        for node in $(kubectl get nodes --no-headers | awk '{print $1}'); do
            status=$(kubectl get node "$node" \
            -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
            if [[ "$status" != "True" ]]; then
                echo "Node $node is not Ready"
                all_ready=false
            else
                echo "Node $node is Ready"
            fi
        done
        
        if [ "$all_ready" = true ]; then
            echo  "Restart shorter than 150 seconds. This may cause a false negative"
        else
            echo "Restart longer than 150 seconds. False negative should not occur"
        fi
        """,
        params={'cluster': node_pool_info.cluster_name,
                'region': node_pool_info.location,
                'project': node_pool_info.project_id}
  )

  cleanup_node_pool = node_pool.delete.override(trigger_rule="all_done")(
      node_pool=node_pool_info
  ).as_teardown(
      setups=create_nodepool,
  )

  

#   (
#     check_on_cluster >> [create_cluster, skip],
#     [create_cluster, skip] >> join_after_branch,
#     join_after_branch >> create_nodepool >> [disk_size_change, wait_time],
#     wait_time >> check_for_negative ,
#     [disk_size_change, check_for_negative] >> time_interval >> poll_nodepool >> cleanup_node_pool,
#   )

  (
    create_nodepool >> [disk_size_change, wait_time],
    wait_time >> check_for_negative ,
    [disk_size_change, check_for_negative] >> time_interval >> poll_nodepool >> cleanup_node_pool,
  )