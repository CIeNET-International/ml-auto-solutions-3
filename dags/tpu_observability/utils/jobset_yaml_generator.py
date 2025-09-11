"""Utility to generate YAML for a Kubernetes JobSet for TPU workloads."""

import dataclasses
import textwrap
from typing import Dict, List, Optional
import yaml


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

  # Container Spec
  container_name: str
  image: str
  tpu_cores_per_pod: int
  command: Optional[List[str]] = None
  command_args: Optional[List[str]] = None

  # Pod Template Spec
  node_selector: Optional[Dict[str, str]] = None

  # Volume Spec
  volume_name: Optional[str] = None
  config_map_name: Optional[str] = None


# --- Custom String Class for Multi-line Args ---
class LiteralScalarString(str):
  pass


def literal_presenter(dumper, data):
  """This presenter tells PyYAML to render instances of LiteralScalarString.

  Args:
    dumper: The PyYAML dumper object.
    data: The FlowList instance to represent.

  Returns:
    The represented YAML sequence in flow style.
  """
  return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralScalarString, literal_presenter)


# --- Custom List Class for Flow Style Command ---
class FlowList(list):
  pass


def flow_list_presenter(dumper, data):
  """This presenter tells PyYAML to render instances of FlowList.

  This uses the flow style (e.g., ["item1", "item2"]).

  Args:
    dumper: The PyYAML dumper object.
    data: The FlowList instance to represent.

  Returns:
    The represented YAML sequence in flow style.
  """
  return dumper.represent_sequence(
      "tag:yaml.org,2002:seq", data, flow_style=True
  )


yaml.add_representer(FlowList, flow_list_presenter)


def create_jobset_yaml(
    jobset_name: str = "tpu-info-v6e-workload",
    namespace: str = "default",
    max_restarts: int = 5,
    replicated_job_name: str = "tpu-job-slice",
    replicas: int = 2,
    backoff_limit: int = 0,
    completions: int = 4,
    parallelism: int = 4,
    image: str = "python:3.10",
    container_name: str = "jax-tpu-job",
    tpu_cores_per_pod: int = 4,
    node_selector: Optional[Dict[str, str]] = None,
    command: Optional[List[str]] = None,
    command_args: Optional[List[str]] = None,
    volume_name: Optional[str] = None,
    config_map_name: Optional[str] = None,
) -> str:
  """Dynamically generates the YAML configuration for a Kubernetes JobSet.

  Args:
      jobset_name (str): The name of the JobSet.
      namespace (str): The namespace where the JobSet will be created.
      max_restarts (int): The maximum number of restarts for the JobSet's
        failure policy.
      replicated_job_name (str): The name of the ReplicatedJob.
      replicas (int): The number of replicas for the ReplicatedJob.
      backoff_limit (int): The backoff limit for the job template.
      completions (int): The number of completions for each job.
      parallelism (int): The parallelism for each job.
      image (str): The container image to use.
      container_name (str): The name of the container within the pod.
      tpu_cores_per_pod (int): The number of TPU cores requested per pod.
      node_selector (Optional[Dict[str, str]]): The node selector for pod
        scheduling.
      command (Optional[List[str]]): The main command to run in the container.
      command_args (Optional[List[str]]): The arguments for the command.
      volume_name (Optional[str]): The name of the volume to mount.
      config_map_name (Optional[str]): The name of the ConfigMap to use. If
        None, no volume is mounted.

  Returns:
      str: The formatted YAML string.
  """
  if node_selector is None:
    node_selector = {
        "cloud.google.com/gke-tpu-accelerator": "tpu-v6e-slice",
        "cloud.google.com/gke-tpu-topology": "4x4",
    }

  if command_args is None:
    command_args = [
        """
            echo "sleep..."
            sleep 10000
            """
    ]
  processed_args = []
  for arg in command_args:
    dedented_arg = textwrap.dedent(arg).strip()
    processed_args.append(LiteralScalarString(dedented_arg))

  processed_command = FlowList(command) if command is not None else None

  jobset_dict = {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {
          "name": jobset_name,
          "annotations": {
              "alpha.jobset.sigs.k8s.io/exclusive-topology": (
                  "cloud.google.com/gke-nodepool"
              )
          },
          "namespace": namespace,
      },
      "spec": {
          "failurePolicy": {
              "restartStrategy": "BlockingRecreate",
              "maxRestarts": max_restarts,
          },
          "replicatedJobs": [{
              "name": replicated_job_name,
              "replicas": replicas,
              "template": {
                  "spec": {
                      "backoffLimit": backoff_limit,
                      "completions": completions,
                      "parallelism": parallelism,
                      "completionMode": "Indexed",
                      "template": {
                          "spec": {
                              "hostNetwork": True,
                              "dnsPolicy": "ClusterFirstWithHostNet",
                              "subdomain": "headless-svc",
                              "restartPolicy": "Never",
                              "nodeSelector": node_selector,
                              "containers": [{
                                  "name": container_name,
                                  "image": image,
                                  "ports": [
                                      {"containerPort": 8471},
                                      {"containerPort": 8080},
                                      {"containerPort": 8431},
                                  ],
                                  "securityContext": {"privileged": True},
                                  "command": processed_command,
                                  "args": processed_args,
                                  "stdin": True,
                                  "tty": True,
                                  "resources": {
                                      "requests": {
                                          "google.com/tpu": tpu_cores_per_pod
                                      },
                                      "limits": {
                                          "google.com/tpu": tpu_cores_per_pod
                                      },
                                  },
                              }],
                          }
                      },
                  }
              },
          }],
      },
  }

  if config_map_name:
    pod_spec = jobset_dict["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    pod_spec["containers"][0]["volumeMounts"] = [{
        "name": volume_name,
        "mountPath": "/app",
    }]
    pod_spec["volumes"] = [{
        "name": volume_name,
        "configMap": {"name": config_map_name},
    }]

  return yaml.dump(jobset_dict, sort_keys=False)
