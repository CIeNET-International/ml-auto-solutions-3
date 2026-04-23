from airflow import DAG
import dataclasses
from airflow.decorators import task
import os
import datetime
import string
import textwrap
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.utils.jobset_util import Command
from dags.tpu_observability.utils.node_pool_util import Info as node_pool_info
from dags.tpu_observability.utils.time_util import TimeUtil
import tempfile

_TEMPLATE = string.Template(
    textwrap.dedent(
        """
        apiVersion: jobset.x-k8s.io/v1alpha2
        kind: JobSet
        metadata:
          generation: 2
          labels:
            kueue.x-k8s.io/queue-name: multislice-queue
            xpk.google.com/workload: $jobset_name
          name: $jobset_name
          namespace: default
        spec:
          coordinator:
            replicatedJob: pathways-head
          failurePolicy:
            maxRestarts: $max_restarts
            restartStrategy: Recreate
          network:
            enableDNSHostnames: true
            publishNotReadyAddresses: true
          replicatedJobs:
          - name: pathways-head
            replicas: 1
            template:
              metadata:
                annotations:
                  alpha.jobset.sigs.k8s.io/exclusive-topology: kubernetes.io/hostname
              spec:
                backoffLimit: 0
                completionMode: Indexed
                completions: 1
                parallelism: 1
                template:
                  metadata:
                    annotations:
                    labels:
                      kueue.x-k8s.io/podset: pathways-head
                  spec:
                    containers:
                    - command:
                      - bash
                      - -c
                      - |
                        echo XPK Start: $$(date);
                        _sigterm() (kill -SIGTERM $$! 2>/dev/null;);
                        trap _sigterm SIGTERM;

                        (export TPU_STDERR_LOG_LEVEL=0 && export TPU_MIN_LOG_LEVEL=0 && export TF_CPP_MIN_LOG_LEVEL=0 && export TPU_VMODULE=real_program_continuator=1 &&    export ENABLE_PATHWAYS_PERSISTENCE=1 && export JAX_PLATFORMS=proxy && export ENABLE_PJRT_COMPATIBILITY=true && export MAXTEXT_ASSETS_ROOT=/deps/src/MaxText/assets MAXTEXT_PKG_DIR=/deps/src/MaxText MAXTEXT_REPO_ROOT=/deps && python3 -m MaxText.train src/MaxText/configs/base.yml per_device_batch_size=1 ici_tensor_parallelism=4 ici_fsdp_parallelism=-1 quantization='int8' remat_policy=full max_target_length=1024 attention=flash gcs_metrics=True use_iota_embed=True dataset_path=gs://max-datasets-rogue dataset_type=$dataset_type reuse_example_batch=1 enable_checkpointing=False sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048 checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False enable_pathways_goodput=True enable_goodput_recording=True enable_single_controller=True metrics_file=metrics.txt goodput_upload_interval_seconds=30 monitor_goodput=True  steps=$steps model_name=$model base_output_directory=gs://lidanny-southamerica-west1/dorah-pathways_2_slice_v6e-256_llama3-8b-8192/  run_name=dor-v6e-256x2-1 gcs_metrics=true ) & PID=$$!;
                        while kill -0 $$PID 2>/dev/null;
                            do sleep 5;
                        done;kubectl
                        wait $$PID;
                        EXIT_CODE=$$?;

                        echo XPK End: $$(date);
                        echo EXIT_CODE=$$EXIT_CODE;


                        exit $$EXIT_CODE
                      env:
                      - name: PATHWAYS_HEAD
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                      - name: JAX_PLATFORMS
                        value: proxy
                      - name: XCLOUD_ENVIRONMENT
                        value: GCP
                      - name: JAX_BACKEND_TARGET
                        value: grpc://$jobset_name-pathways-head-0-0.$jobset_name:29000
                      image: gcr.io/cloud-tpu-multipod-dev/chzheng_grain_081_1_latest@sha256:680157a37cf0be9b0bcdc0d98efd209bbf2b4eef12de57277071e1df44fd4401
                      imagePullPolicy: Always
                      name: jax-tpu
                      resources:
                        limits:
                          cpu: "24"
                          memory: 100G
                      securityContext:
                        privileged: true
                      volumeMounts:
                      - mountPath: /tmp
                        name: shared-tmp
                    dnsPolicy: ClusterFirstWithHostNet
                    hostNetwork: true
                    initContainers:
                    - args:
                      - --server_port=29001
                      - --gcs_scratch_location=gs://lidanny-southamerica-west1/dorah-pathways_2_slice_v6e-256_llama3-8b-8192/
                      - --node_type=resource_manager
                      - --instance_count=$total_replicas
                      - --instance_type=tpuv6e:$tpu_topology
                      - --xla_tpu_use_enhanced_launch_barrier=true
                      env:
                      - name: REPLICATED_JOB_NAME
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']
                      - name: JOBSET_NAME
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.annotations['jobset.sigs.k8s.io/jobset-name']
                      - name: HOST_ADDRESS
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                      - name: TPU_SKIP_MDS_QUERY
                        value: "true"
                      image: us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server@sha256:73ba12efe8750ea977772e92b050da8f769e9ae0de97926a3ff11fd523196fcf
                      imagePullPolicy: Always
                      name: pathways-rm
                      ports:
                      - containerPort: 29001
                        protocol: TCP
                      - containerPort: 29002
                        protocol: TCP
                      resources:
                        limits:
                          cpu: "8"
                          memory: 16G
                      restartPolicy: Always
                    - args:
                      - --server_port=29000
                      - --resource_manager_address=$jobset_name-pathways-head-0-0.$jobset_name:29001
                      - --gcs_scratch_location=gs://lidanny-southamerica-west1/dorah-pathways_2_slice_v6e-256_llama3-8b-8192/
                      - --num_elastic_slices=$hardware_slice
                      - --virtual_slices=$virtual_slices_str
                      - --sidecar_name=external
                      - --xla_tpu_scoped_vmem_limit_kib=98304
                      - --xla_tpu_enable_async_collective_fusion=true
                      - --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
                      - --xla_tpu_enable_async_collective_fusion_multiple_steps=true
                      - --xla_tpu_overlap_compute_collective_tc=true
                      - --xla_enable_async_all_gather=true
                      - --xla_tpu_use_enhanced_launch_barrier=true
                      env:
                      - name: PATHWAYS_HEAD
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                      image: us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server@sha256:4f3edcc2d28278e5da5cebd6ed36feec650656c173001b7fe80a1074424b7622
                      imagePullPolicy: Always
                      name: pathways-proxy
                      ports:
                      - containerPort: 29000
                        protocol: TCP
                      resources:
                        limits:
                          cpu: "16"
                          memory: 100G
                      restartPolicy: Always
                    nodeSelector:
                      cloud.google.com/gke-nodepool: cpu-np
                    restartPolicy: Never
                    volumes:
                    - hostPath:
                        path: /tmp
                        type: DirectoryOrCreate
                      name: shared-tmp
          - name: worker
            replicas: $total_replicas
            template:
              metadata: {}
              spec:
                backoffLimit: 256
                completionMode: Indexed
                completions: $completions
                parallelism: $parallelism
                template:
                  metadata:
                    annotations:
                      alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
                    labels:
                      kueue.x-k8s.io/podset: worker
                  spec:
                    containers:
                    - args:
                      - --server_port=29005
                      - --resource_manager_address=$jobset_name-pathways-head-0-0.$jobset_name:29001
                      - --gcs_scratch_location=gs://lidanny-southamerica-west1/dorah-pathways_2_slice_v6e-256_llama3-8b-8192/
                      - --xla_tpu_use_enhanced_launch_barrier=true
                      env:
                      - name: TPU_MIN_LOG_LEVEL
                        value: "0"
                      - name: TF_CPP_MIN_LOG_LEVEL
                        value: "0"
                      - name: XCLOUD_ENVIRONMENT
                        value: GCP
                      - name: MEGASCALE_GRPC_ENABLE_XOR_TRACER
                        value: "false"
                      - name: MEGASCALE_NUM_SLICES
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                      - name: JOBSET_NAME
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.annotations['jobset.sigs.k8s.io/jobset-name']
                      - name: REPLICATED_JOB_NAME
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']
                      - name: MEGASCALE_SLICE_ID
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.labels['jobset.sigs.k8s.io/job-index']
                      - name: PATHWAYS_HEAD
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                      - name: MEGASCALE_COORDINATOR_ADDRESS
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                      image: us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server@sha256:73ba12efe8750ea977772e92b050da8f769e9ae0de97926a3ff11fd523196fcf
                      imagePullPolicy: Always
                      name: pathways-worker
                      ports:
                      - containerPort: 29005
                        protocol: TCP
                      - containerPort: 29006
                        protocol: TCP
                      - containerPort: 8471
                        protocol: TCP
                      - containerPort: 8080
                        protocol: TCP
                      resources:
                        limits:
                          google.com/tpu: $tpu_cores_per_pod
                      volumeMounts:
                      - mountPath: /tmp
                        name: shared-tmp
                    dnsPolicy: ClusterFirstWithHostNet
                    hostNetwork: true
                    initContainers:
                    - env:
                      - name: GRPC_SERVER_ADDRESS
                        value: '''0.0.0.0:50051'''
                      image: gcr.io/cloud-tpu-multipod-dev/chzheng_grain_081_1_latest@sha256:c521a815f6dc41dd9ac78ee532fb782727b2fbf8de290500081b2e9e3eb3c7ec
                      imagePullPolicy: Always
                      name: colocated-python-sidecar
                      ports:
                      - containerPort: 50051
                        protocol: TCP
                      resources: {}
                      restartPolicy: Always
                      volumeMounts:
                      - mountPath: /tmp
                        name: shared-tmp
                    nodeSelector:
                      cloud.google.com/gke-tpu-accelerator: $tpu_accelerator_type
                      cloud.google.com/gke-tpu-topology: $tpu_topology
                    priorityClassName: very-high
                    restartPolicy: OnFailure
                    terminationGracePeriodSeconds: 300
                    volumes:
                    - hostPath:
                        path: /tmp
                        type: DirectoryOrCreate
                      name: shared-tmp
          startupPolicy:
            startupPolicyOrder: InOrder
          successPolicy:
            operator: All
            targetReplicatedJobs:
            - pathways-head
          suspend: false
        """
    )
)


@dataclasses.dataclass
class JobSet:
  """
  Generates YAML configurations for Kubernetes JobSets.

  This class helps in creating JobSet YAMLs by providing a template and allowing
  customization of various parameters like jobset name, replicas, TPU
  configuration, and the workload script to be executed.

  Attributes:
    jobset_name: The name of the JobSet.
    namespace: The Kubernetes namespace for the JobSet.
    max_restarts: The maximum number of restarts for the JobSet.
    replicated_job_name: The name for the replicated Job within the JobSet.
    replicas: The number of replicas for the replicated Job.
    backoff_limit: The number of failed pods to tolerate before marking the
      Job as failed.
    completions: The number of pods that must complete successfully.
    parallelism: The number of pods to run in parallel.
    tpu_accelerator_type: The type of TPU accelerator (e.g.,
      "tpu-v6e-slice").
    tpu_topology: The TPU topology (e.g., "4x4").
    container_name: The name of the container in the pod.
    image: The container image to use.
    tpu_cores_per_pod: The number of TPU cores requested per pod.
  """

  jobset_name: str
  namespace: str
  model: str
  num_chips: int
  dataset_type: str
  hardware_slice: int
  spare_slices: int
  max_restarts: int
  steps: int
  replicated_job_name: str
  backoff_limit: int
  completions: int
  parallelism: int
  tpu_accelerator_type: str
  tpu_topology: str
  container_name: str
  image: str
  tpu_cores_per_pod: int

  def generate_yaml(self, workload_script: Workload) -> str:
    """Generates the final JobSet YAML content.

    Args:
        workload_script: A pre-formatted, JSON-escaped string from the Workload
          class.

    Returns:
        A string containing the complete JobSet YAML.
    """
    params = dataclasses.asdict(self)
    params["total_replicas"] = self.hardware_slice + self.spare_slices
    slice_fragment = f"tpuv6e:{self.tpu_topology}"
    params["virtual_slices_str"] = ",".join([slice_fragment] * self.hardware_slice)
    params["command"] = ["bash", "-c"]
    params["args"] = workload_script

    return _TEMPLATE.substitute(params)


@task
def run_workload(
    node_pool: node_pool_info, yaml_config: str, namespace: str
) -> TimeUtil:
  """
  Applies the specified YAML file to the GKE cluster.

  Args:
    node_pool: Configuration object with cluster details.
    yaml_config: The JobSet object containing YAML configuration.
    namespace: The Kubernetes namespace to apply the JobSet.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_apply_jobset_command(
            temp_config_file.name, yaml_config, namespace
        ),
    ])

    subprocess.run_exec(cmd, env=env)

    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    return TimeUtil.from_datetime(current_time_utc)

@task
def end_workload(node_pool: node_pool_info, jobset_name: str, namespace: str):
  """
  Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    node_pool: Configuration object with cluster details.
    jobset_name: The name of the JobSet to delete.
    namespace: The Kubernetes namespace to delete the JobSet from.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_delete_jobset_command(
            temp_config_file.name, jobset_name, namespace
        ),
    ])

    subprocess.run_exec(cmd, env=env)
