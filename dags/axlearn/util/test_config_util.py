"""Test Configuration Class utility for orbax testcases"""

import posixpath
from typing import Optional
from dataclasses import dataclass

from airflow.exceptions import AirflowFailException
from absl import logging
import math
import re

from dags import gcs_bucket
from xlml.utils.gke import zone_to_region
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.orbax.util import checkpoint_util
from dags.multipod.configs.common import SetupMode


DEFAULT_BUCKET = gcs_bucket.AXLEARN_AUTOMATION_BUCKET

# Only one version of Axlearn is used at the moment with Jax 0.5.3.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.AXLEARN_CUSTOM,
)]

# Valid models and sizes for current Maxtext Repository.
MODELS = {
    "deepseek2",
    "deepseek3",
    "gemma",
    "gemma2",
    "gemma3",
    "gpt",
    "gpt3",
    "llama2",
    "llama3",
    "llama3.1",
    "llama3.3",
    "llama4",
    "mistral",
    "qwen3",
}


@dataclass
class Checkpointing:
  """Represents the information of a checkpointing mechanism.

  Attributes:
    name: A unique name for the checkpointing configuration.
    en: Indicates whether a replicator is enabled.
  """

  name: str
  enable_multi_tier_checkpointing: bool


@dataclass
class TestConfig:
  """Holds the general configuration for a checkpointing test."""

  cluster: XpkClusters
  run_name: str
  slices: list[int]
  instance_type: str
  mesh_type: str
  module: str
  short_id: str
  step: int
  checkpoint_step: int
  model_config: str
  trainer_dir: str
  data_dir: str
  train_batch_size: int
  fsdp: int
  data: int


  def generate_step_to_validate(self) -> list[int]:
    total_steps = self.step
    k = self.checkpoint_step
    last_step = self.step
    return [*range(self.checkpoint_step, total_steps, k), last_step]
