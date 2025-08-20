#!/usr/local/bin/python
#
# Perform multiple BashOperator with different rates to automate tpu-info streaming.

from typing import Final
from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.exceptions import AirflowFailException
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from google.cloud import monitoring_v3

# Download JAX code from GS or paste-server (can't use paste-server because of gcert)
JAX_CODE_URL: Final[str] = "https://paste.googleplex.com/4829159660191744?raw"
JAX_CODE_YAML: Final[str] = "jax-tpu-benchmark-code.yaml"
JAX_CODE_YAML_PATH: Final[str] = f"/tmp/{JAX_CODE_YAML}"

# Download v6e workload from GS or paste-server (can't use paste-server because of gcert)
TPU_INFO_WORKLOAD_URL: Final[
    str
] = "https://paste.googleplex.com/6544496703307776?raw"
TPU_INFO_WORKLOAD_YAML: Final[str] = "tpu-info-v6e-workload.yaml"
TPU_INFO_WORKLOAD_YAML_PATH: Final[str] = f"/tmp/{TPU_INFO_WORKLOAD_YAML}"

# Download tpu-info output parser scripts from GS.
# TODO: convert them to BashOperator and Python task.
CAPTURE_TPU_INFO_SCRIPT: Final[str] = "capture-tpu-info.sh"
CAPTURE_TPU_INFO_PATH: Final[str] = f"/tmp/{CAPTURE_TPU_INFO_SCRIPT}"
DIFF_TPU_INFO_SCRIPT: Final[str] = "diff-tpu-info.sh"
DIFF_TPU_INFO_PATH: Final[str] = f"/tmp/{DIFF_TPU_INFO_SCRIPT}"

#
# User, TPU-type, cluster, and nodepool are likely to be changed
#
DEF_USER: Final[str] = "vincentlau"
DEF_PROJECT: Final[str] = "tpu-prod-env-one-vm"
DEF_REGION: Final[str] = "asia-northeast1"
DEF_ZONE: Final[str] = f"{DEF_REGION}-b"
DEF_TPU_TYPE: Final[str] = "v6e-16"
DEF_CLUSTER_NAME: Final[str] = f"{DEF_USER}-v6e"

NUM_SLICES: Final[int] = 2
CLUSTER_VERSION: Final[str] = "1.33.2-gke.1283000"
NODE_VERSION: Final[str] = f"{CLUSTER_VERSION}"

# Use these versions for tpu-info streaming
LIBTPU_VERSION: Final[str] = "0.0.19.dev20250710+nightly"
TPU_INFO_VERSION: Final[str] = "0.4.0"

# Local temporary kube config file
KUBECONFIG: Final[str] = "/tmp/kubeconfig"

# Run at 5 minutes after midnight every day.
# params seems to be read-only
with DAG(
    dag_id="tpu-info-streaming",
    start_date=datetime(2025, 7, 21),
    schedule="5 0 * * *",
    catchup=False,
    params={
        "USER": f"{DEF_USER}",
        "CLUSTER_NAME": f"{DEF_CLUSTER_NAME}",
        "PROJECT": f"{DEF_PROJECT}",
        "REGION": f"{DEF_REGION}",
        "ZONE": f"{DEF_ZONE}",
        "TPU_TYPE": f"{DEF_TPU_TYPE}",
    },
) as dag:
  USER = dag.params["USER"]
  CLUSTER_NAME = dag.params["CLUSTER_NAME"]
  PROJECT = dag.params["PROJECT"]
  REGION = dag.params["REGION"]
  ZONE = dag.params["ZONE"]
  TPU_TYPE = dag.params["TPU_TYPE"]

  # Network/router/firewall settings; not likely be changed
  NETWORK_NAME = f"{CLUSTER_NAME}-mtu9k"
  NETWORK_NAME_1 = f"{NETWORK_NAME}-1-{ZONE}"
  NETWORK_FW_NAME_1 = f"{NETWORK_NAME}-fw-1-{ZONE}"
  NETWORK_NAME_2 = f"{CLUSTER_NAME}-privatenetwork-2-{ZONE}"
  SUBNET_NAME_2 = f"{CLUSTER_NAME}-privatesubnet-2-{ZONE}"
  FIREWALL_RULE_NAME = f"{CLUSTER_NAME}-privatefirewall-2-{ZONE}"
  ROUTER_NAME = f"{CLUSTER_NAME}-network-2-{ZONE}"
  NAT_CONFIG = f"{CLUSTER_NAME}-natconfig-2-{ZONE}"

  # Create a bash operator to capture tpu-info stream at a rate
  # for a duration in seconds, and report the effective rate.
  # TODO: haven't tested if the "script" command would work in Cloud Composer environment.
  # because it does not have login shell.
  def tpu_info_streaming_op(rate, duration) -> BashOperator:
    return BashOperator(
        task_id=f"tpu_info_streaming_{rate}s",
        bash_command=f"""
                gsutil cp gs://cienet-tpu-observability-tpu-info/workloads/{CAPTURE_TPU_INFO_SCRIPT} {CAPTURE_TPU_INFO_PATH}
                gsutil cp gs://cienet-tpu-observability-tpu-info/workloads/{DIFF_TPU_INFO_SCRIPT} {DIFF_TPU_INFO_PATH}
                chmod 755 {CAPTURE_TPU_INFO_PATH} {DIFF_TPU_INFO_PATH}

                TFILE="/tmp/tpu-info-str-{rate}s-$$.txt"
                {CAPTURE_TPU_INFO_PATH} -d {duration} -o $TFILE {rate};
                {DIFF_TPU_INFO_PATH} -s $TFILE;
                rm -f $TFILE
            """,
    )

  # Note: pull the podname pushed by "get_first_pod" task via XCom.
  # But there is no way to save in dag.params or global variable.
  def _pull_first_pod(**kwargs) -> [str]:
    ti = kwargs["ti"]
    podname = ti.xcom_pull(task_ids="get_first_pod")
    return podname

  @task
  def abort_flow():
    raise AirflowFailException("Task aborted.")

  #
  # TODO: for unknown reason, it did not detect the log message.
  #
  def wait_for_log_msg_op(msg) -> BashOperator:
    return BashOperator(
        task_id="wait_for_log_msg",
        execution_timeout=timedelta(seconds=1200),
        bash_command=f"""
                export KUBECONFIG={KUBECONFIG}
                gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT}

                PODNAME=$(kubectl -n default --kubeconfig {KUBECONFIG} get pods | awk 'NR==2 {{ print $1 }}')
                echo "podname=$PODNAME, podname2={{ ti.xcom_pull(taskids='get_first_pod') }}"
                if [[ -z "$PODNAME" ]]; then
                    exit 1
                fi

                kubectl -n default --kubeconfig {KUBECONFIG} logs -f --timestamps $PODNAME | \\
                    awk '{{ print $0; if ($0 ~ "{msg}") {{ exit 0 }} }}'
            """,
    )

  def install_tpu_utils_op() -> BashOperator:
    return BashOperator(
        task_id="install_tpu_utils",
        bash_command=f"""
                export KUBECONFIG={KUBECONFIG}
                gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT}

                PODNAME=$(kubectl -n default --kubeconfig {KUBECONFIG} get pods | awk 'NR==2 {{ print $1 }}')
                echo "podname=$PODNAME"
                if [[ -z "$PODNAME" ]]; then
                    exit 1
                fi

                kubectl -n default exec $PODNAME -- /bin/bash -c \\
                "pip -q show tpu-info | grep -s 'Version: {TPU_INFO_VERSION}' || \\
                 ( gsutil cp gs://vidisha-sethi-tpu-cli-1/tpu_info-{TPU_INFO_VERSION}-py3-none-any.whl /tmp/ && \\
                   pip install /tmp/tpu_info-{TPU_INFO_VERSION}-py3-none-any.whl --force-reinstall ); \\
                 pip -q show jax || pip install jax; \\
                 pip show libtpu | grep -s 'Version: {LIBTPU_VERSION}' >/dev/null || \\
                 pip install libtpu=={LIBTPU_VERSION} -f https://storage.googleapis.com/libtpu-wheels/index.html"
            """,
    )

  #
  # Define all the tasks.
  #

  # Pre-requisite: these files must be uploaded to GS first from a host.
  # gsutil mb gs://cienet-tpu-observability-tpu-info
  # gsutil cp {CAPTURE_TPU_INFO_SCRIPT} gs://cienet-tpu-observability-tpu-info/workloads/{CAPTURE_TPU_INFO_SCRIPT}
  # gsutil cp {DIFF_TPU_INFO_SCRIPT} gs://cienet-tpu-observability-tpu-info/workloads/{DIFF_TPU_INFO_SCRIPT}
  # gsutil cp {JAX_CODE_YAML} gs://cienet-tpu-observability-tpu-info/workloads/{JAX_CODE_YAML}
  # gsutil cp {TPU_INFO_WORKLOAD_YAML} gs://cienet-tpu-observability-tpu-info/workloads/{TPU_INFO_WORKLOAD_YAML}
  start_task = BashOperator(
      task_id="start_tpu_info_streaming_test",
      bash_command="""
            whoami
            env
        """,
  )

  # Use xpk to create cluster and nodepool.
  # Assume that the networks have been created using gcloud.
  create_cluster_nodepool_task = BashOperator(
      task_id="create_cluster_nodepool",
      bash_command=f"""
            gcloud container clusters describe {CLUSTER_NAME} --project {PROJECT} --region {REGION} &>/dev/null
            if [[ $? -eq 0 ]]; then
              echo "GKE cluster {CLUSTER_NAME} already exists."
            else
              export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias \\
                --enable-multi-networking --network={NETWORK_NAME_1} \\
                --subnetwork={NETWORK_NAME_1} --cluster-version={CLUSTER_VERSION}"
              export NODE_POOL_ARGUMENTS="--node-version={NODE_VERSION} \\
                --additional-node-network network={NETWORK_NAME_2},subnetwork={SUBNET_NAME_2}"
              xpk cluster create --cluster {CLUSTER_NAME} \\
                --cluster-cpu-machine-type=n2-standard-32 \\
                --num-slices={NUM_SLICES} \\
                --tpu-type={TPU_TYPE} \\
                --project={PROJECT} \\
                --zone={ZONE} \\
                --demand \\
                --custom-cluster-arguments="$CLUSTER_ARGUMENTS" \\
                --custom-nodepool-arguments="$NODE_POOL_ARGUMENTS"
            fi
        """,
      retries=3,
  )

  # Use xpk to delete cluster (and nodepool?)
  delete_cluster_nodepool_task = BashOperator(
      task_id="delete_cluster_nodepool",
      trigger_rule=TriggerRule.ALL_DONE,
      bash_command=f"""
            gcloud container clusters describe {CLUSTER_NAME} --project {PROJECT} --region {REGION} &>/dev/null
            if [[ $? -ne 0 ]]; then
                echo "GKE cluster {CLUSTER_NAME} does not exist."
            else
                # xpk cluster delete --cluster {CLUSTER_NAME} --zone={ZONE} --project={PROJECT}
                pass
            fi
        """,
  )

  #
  # TODO: there are no errors.  But it is not clear if the workload
  # was actually success because the wait_for_log_msg task would time out.
  #
  launch_workload_task = BashOperator(
      task_id="launch_workload",
      bash_command=f"""
            gsutil cp gs://cienet-tpu-observability-tpu-info/workloads/{JAX_CODE_YAML} {JAX_CODE_YAML_PATH}
            gsutil cp gs://cienet-tpu-observability-tpu-info/workloads/{TPU_INFO_WORKLOAD_YAML} {TPU_INFO_WORKLOAD_YAML_PATH}

            export KUBECONFIG={KUBECONFIG}
            gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT}

            # Cloud Composer does not set any namespaces to kubenete contexts.
            # Create a namespace for this Composer, or force the namespace to default
            #kubectl create namespace $COMPOSER_VERSIONED_NAMESPACE

            kubectl -n default --kubeconfig {KUBECONFIG} apply -f {JAX_CODE_YAML_PATH}
            kubectl -n default --kubeconfig {KUBECONFIG} apply -f {TPU_INFO_WORKLOAD_YAML_PATH}
        """,
  )

  timeout = 240
  sleep_4m_task = BashOperator(
      task_id=f"sleep_{timeout}s", bash_command=f"sleep {timeout}"
  )

  #
  # Push the first pod name via Xcom
  #
  # get_first_pod_task = BashOperator(
  #    task_id="get_first_pod",
  #    do_xcom_push=True,
  #    bash_command=f"""
  #        export KUBECONFIG={KUBECONFIG}
  #        gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT}
  #
  #        kubectl -n default --kubeconfig {KUBECONFIG} get pods | awk 'NR==2 {{ print $1 }}'
  #    """)

  #
  # Pull the pod name via Xcom and push to Xcom automatically again.
  #
  # set_podname_task = PythonOperator(
  #    task_id="set_podname",
  #    python_callable=_pull_first_pod,
  #    provide_context=True)

  tpu_info_streaming_0_1_task = tpu_info_streaming_op(0.1, 20)
  tpu_info_streaming_0_5_task = tpu_info_streaming_op(0.5, 20)
  tpu_info_streaming_1_0_task = tpu_info_streaming_op(1.0, 20)
  tpu_info_streaming_5_0_task = tpu_info_streaming_op(5.0, 20)

  #
  # This is not necesasry because the /tmp will be cleared in each task
  #
  remove_yamls_scripts_task = BashOperator(
      task_id="remove_yamls_scripts",
      depends_on_past=False,
      trigger_rule=TriggerRule.ALL_DONE,
      bash_command=f"""
            rm -f {JAX_CODE_YAML_PATH} {TPU_INFO_WORKLOAD_YAML_PATH};
            rm -f {CAPTURE_TPU_INFO_PATH} {DIFF_TPU_INFO_PATH}
        """,
  )

  stop_workload_task = BashOperator(
      task_id="stop_workload",
      depends_on_past=False,
      trigger_rule=TriggerRule.ALL_DONE,
      bash_command=f"""
            gsutil cp gs://cienet-tpu-observability-tpu-info/workloads/{JAX_CODE_YAML} {JAX_CODE_YAML_PATH}
            gsutil cp gs://cienet-tpu-observability-tpu-info/workloads/{TPU_INFO_WORKLOAD_YAML} {TPU_INFO_WORKLOAD_YAML_PATH}

            export KUBECONFIG={KUBECONFIG}
            gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT}

            kubectl -n default --kubeconfig {KUBECONFIG} delete -f {JAX_CODE_YAML_PATH}
            kubectl -n default --kubeconfig {KUBECONFIG} delete -f {TPU_INFO_WORKLOAD_YAML_PATH}
        """,
  )

  perform_tests_task = DummyOperator(task_id="perform_tests")

  end_task = DummyOperator(task_id="end_tpu_info_streaming_validation")

  # For debug purpose.
  # abort_task = abort_flow()

  {
      start_task
      >> create_cluster_nodepool_task
      >> launch_workload_task
      >> sleep_4m_task
      >> wait_for_log_msg_op("Starting benchmark")
      >> install_tpu_utils_op()
      >> perform_tests_task
  }

  {
      perform_tests_task
      >> [
          tpu_info_streaming_0_1_task,
          tpu_info_streaming_0_5_task,
          tpu_info_streaming_1_0_task,
          tpu_info_streaming_5_0_task,
      ]
      >> stop_workload_task
      >> remove_yamls_scripts_task
      >> end_task
  }
