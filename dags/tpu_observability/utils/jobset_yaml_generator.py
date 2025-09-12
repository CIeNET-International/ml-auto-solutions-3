"""Generates a JobSet YAML file for running TPU workloads on GKE.

This script defines a `JobSetParams` dataclass to configure the JobSet
and uses a YAML template to generate the final JobSet YAML. It includes
custom handling for command arguments to allow multi-line strings.
"""

import dataclasses
from dataclasses import asdict
import textwrap
from typing import List, Optional

import yaml


dataclass = dataclasses.dataclass
TEMPLATE = """
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {jobset_name}
  annotations:
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
  namespace: {namespace}
spec:
  failurePolicy:
    maxRestarts: {max_restarts}
  replicatedJobs:
  - name: {replicated_job_name}
    replicas: {replicas}
    template:
      spec:
        backoffLimit: {backoff_limit}
        completions: {completions}
        parallelism: {parallelism}
        template:
          spec:
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: {tpu_accelerator_type}
              cloud.google.com/gke-tpu-topology: {tpu_topology}
            containers:
            - name: {container_name}
              image: {image}
              __COMMAND_ARGS_PLACEHOLDER__
              stdin: true
              tty: true
              resources:
                requests:
                  google.com/tpu: {tpu_cores_per_pod}
                limits:
                  google.com/tpu: {tpu_cores_per_pod}
"""


@dataclass
class YamlConfig:
  """A class to hold all parameters for the JobSet YAML template."""

  # --- Metadata ---
  jobset_name: str

  namespace: str = "default"

  # --- JobSet Specification ---
  max_restarts: int = 5
  replicated_job_name: str = "tpu-slice"
  replicas: int = 2

  # --- Pod Specification ---
  backoff_limit: int = 0
  completions: int = 4
  parallelism: int = 4

  # --- Node Selection (Crucial for TPUs) ---
  tpu_accelerator_type: str = "tpu-v6e-slice"
  tpu_topology: str = "4x4"

  # --- Container Specification ---
  container_name: str = "jax-tpu-worker"
  image: str = (
      "asia-northeast1-docker.pkg.dev/cienet-cmcs/yuna-docker/tpu-info:v0.4.0"
  )
  command: Optional[List[str]] = (["bash", "-c"],)
  command_args: Optional[List[str]] = None

  # --- Resource Allocation ---
  tpu_cores_per_pod: int = 4


class LiteralScalarString(str):
  pass


def literal_presenter(dumper, data):
  return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

yaml.add_representer(LiteralScalarString, literal_presenter)


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


def create_jobset_yaml(jobset_config):
  """Generates and saves a JobSet YAML file from the given JobSetParams.

  This function takes a JobSetParams object, formats a base YAML template,
  and then uses PyYAML to add custom elements like multi-line command arguments
  with literal style. The resulting YAML is saved to a file named after
  the jobset_name.

  Args:
    jobset_config: An instance of JobSetParams containing all configuration
      parameters for the JobSet.
  """
  params_dict = asdict(jobset_config)

  final_yaml = TEMPLATE.format(**params_dict)
  intermediate_yaml_str = final_yaml.replace(
      "__COMMAND_ARGS_PLACEHOLDER__", "command_args_placeholder: true"
  )
  data = yaml.safe_load(intermediate_yaml_str)
  container_spec = data["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]["containers"][0]
  del container_spec["command_args_placeholder"]

  container_spec["command"] = jobset_config.command

  if jobset_config.command_args:
    processed_args = [
        LiteralScalarString(textwrap.dedent(arg).strip())
        for arg in jobset_config.command_args
    ]
    container_spec["args"] = processed_args
  output_file_path = f"/tmp/{jobset_config.jobset_name}.yaml"

  with open(output_file_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, sort_keys=False, indent=2)
  return output_file_path
