"""Generates YAML configurations for Kubernetes JobSets for TPU workloads.

This script provides classes to define TPU workloads and generate the
corresponding JobSet YAML files, allowing for customization of various
parameters like TPU topology, container image, and the script to be executed.
"""
import dataclasses
import json
import string
import textwrap

import yaml
from typing import List, Optional


class Workload:
  """A library of predefined workload scripts for JobSet.

  Each workload is a JSON-escaped string, ready to be used as a shell argument.
  """
  JAX_TPU_BENCHMARK = json.dumps(textwrap.dedent("""
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

      print(f"[Host {jax.process_index()}] Allocating data...")
      size = 32768
      x_global = jnp.ones((size, size), dtype=jnp.float32)
      y_global = jnp.ones((size, size), dtype=jnp.float32)

      print(f"[Host {jax.process_index()}] Sharding data...")
      sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("x", None))
      x = jax.device_put(x_global, sharding)
      y = jax.device_put(y_global, sharding)
      print(f"[Host {jax.process_index()}] Data on device")

      # ========= Define heavy workload =========
      @pjit
      def matmul_ultra_heavy(x, y):
          tmp1 = jnp.dot(x, y)
          tmp2 = jnp.dot(tmp1, y.T)
          tmp3 = jnp.dot(tmp2, x.T)
          tmp4 = jnp.dot(tmp3, x)
          tmp5 = jnp.dot(tmp4, y)
          return tmp5

      print(f"[Host {jax.process_index()}] Warming up...")
      matmul_ultra_heavy(x, y).block_until_ready()

      # ========= Benchmark =========
      print(f"[Host {jax.process_index()}] Starting benchmark...")

      start = time.time()
      for i in range(1_000_000): # Remember to control loop time to control experiment time
          result = matmul_ultra_heavy(x, y)
      result.block_until_ready()
      end = time.time()

      if jax.process_index() == 0:
          print(f"Total time: {end - start:.2f} seconds (on full v6e-16)")
      ' &&
      echo "Workload finished, sleeping now..." &&
      sleep 10000
      """), ensure_ascii=False)


_TEMPLATE = string.Template("""
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: $jobset_name
  annotations:
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
  namespace: $namespace
spec:
  failurePolicy:
    maxRestarts: $max_restarts
  replicatedJobs:
  - name: $replicated_job_name
    replicas: $replicas
    template:
      spec:
        backoffLimit: $backoff_limit
        completions: $completions
        parallelism: $parallelism
        template:
          spec:
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: $tpu_accelerator_type
              cloud.google.com/gke-tpu-topology: $tpu_topology
            containers:
            - name: $container_name
              image: $image
              command: $command
              args:
                - $args
              stdin: true
              tty: true
              resources:
                requests:
                  google.com/tpu: $tpu_cores_per_pod
                limits:
                  google.com/tpu: $tpu_cores_per_pod
""")


@dataclasses.dataclass
class JobSet:
  """Generates YAML configurations for Kubernetes JobSets.

  This class helps in creating JobSet YAMLs by providing a template and allowing
  customization of various parameters like jobset name, replicas, TPU
  configuration, and the workload script to be executed.
  """

  def __init__(
      self,
      jobset_name: str,
      namespace: str = "default",
      max_restarts: int = 5,
      replicated_job_name: str = "tpu-slice",
      replicas: int = 2,
      backoff_limit: int = 0,
      completions: int = 4,
      parallelism: int = 4,
      tpu_accelerator_type: str = "tpu-v6e-slice",
      tpu_topology: str = "4x4",
      container_name: str = "jax-tpu-worker",
      image: str = "python:3.10",
      command: Optional[List[str]] = None,
      tpu_cores_per_pod: int = 4):
    """Initializes a JobSet configuration object.

    Args:
        jobset_name: The name of the JobSet.
        namespace: The Kubernetes namespace for the JobSet. Defaults to
          "default".
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
        command: The command to run in the container.
        tpu_cores_per_pod: The number of TPU cores requested per pod.
    """
    self.params = {
        "jobset_name": jobset_name,
        "namespace": namespace,
        "max_restarts": max_restarts,
        "replicated_job_name": replicated_job_name,
        "replicas": replicas,
        "backoff_limit": backoff_limit,
        "completions": completions,
        "parallelism": parallelism,
        "tpu_accelerator_type": tpu_accelerator_type,
        "tpu_topology": tpu_topology,
        "container_name": container_name,
        "image": image,
        "command": command if command is not None else ["bash", "-c"],
        "tpu_cores_per_pod": tpu_cores_per_pod,
    }

  def generate_yaml(self, workload_script: Workload) -> str:
    """Generates the final JobSet YAML content.

    Args:
        workload_script: A pre-formatted, JSON-escaped string from the Workload
          class.

    Returns:
        A string containing the complete JobSet YAML.
    """

    final_params = self.params.copy()
    final_params["args"] = workload_script

    yaml_content = _TEMPLATE.substitute(final_params)
    output_file_path = (f"/tmp/{final_params['jobset_name']}.yaml")
    with open(output_file_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    return output_file_path

if __name__ == "__main__":
  my_jobset = JobSet(
      jobset_name="tpu-info-v6e-workload",
      namespace="default",
      max_restarts=5,
      replicated_job_name="tpu-job-slice",
      replicas=2,
      backoff_limit=0,
      completions=4,
      parallelism=4,
      tpu_accelerator_type="tpu-v6e-slice",
      tpu_topology="4x4",
      container_name="jax-tpu-job",
      image="asia-northeast1-docker.pkg.dev/cienet-cmcs/yuna-docker/tpu-info:v0.4.0",
      command=["bash", "-c"],
  )

  script_to_run = Workload.JAX_TPU_BENCHMARK
  yaml_path = my_jobset.generate_yaml(workload_script=script_to_run)
