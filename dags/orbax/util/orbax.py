import posixpath
from typing import Optional
from dataclasses import dataclass
from enum import Enum


from dags import gcs_bucket
from xlml.utils.gke import zone_to_region
from dags.common.vm_resource import XpkClusters
from dags.orbax.util import checkpoint_util

DEFAULT_BUCKET = gcs_bucket.MTC_AUTOMATION_BUCKET
DEFAULT_RAM_DISK = "/local"


class CheckpointingMode(Enum):
  """Enum for different checkpointing modes."""
  REG = "regular"  # Regular checkpointing
  ECM = "emergency"  # Emergency checkpointing
  MTC = "multi_tier"  # Multi-tier checkpointing

  @property
  def short_name(self) -> str:
    """Get the short 3-letter name for the mode."""
    return self.name.lower()  # "reg", "ecm", "mtc"

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
  machine_type: str
  accelerator: str
  slices: list[int]
  model_name: str
  short_id: str
  step: int
  checkpoint_step: int
  base_dir: str
  mode: CheckpointingMode

  # Optional parameters for emergency/multi-tier modes
  replicator_backup_time: Optional[int] = None
  local_checkpoint_step: Optional[int] = None
  ram_disk_size: Optional[str] = None
  cpc_config: Optional[checkpoint_util.CheckpointConfiguration] = None

  def __init__(
      self,
      cluster: XpkClusters,
      machine_type: str,
      accelerator: str,
      slices: list[int],
      model_name: str,
      short_id: str,
      step: int,
      checkpoint_step: int,
      base_dir: str,
      mode: CheckpointingMode = CheckpointingMode.REG,
      replicator_backup_time: Optional[int] = None,
      local_checkpoint_step: Optional[int] = None,
      ram_disk_size_in_mi: str = "100Gi",
  ):
    """Initializes the test configurations.

    Args:
      cluster: The specified cluster to be used for the test.
      machine_type: The type of machine (e.g., GPU, TPU).
      accelerator: The type of accelerator (e.g., GPU, TPU) to use.
      slices: The number of slices to be used.
      model_name: The name of the model being tested.
      short_id: A short identifier for the test run.
      step: The current step of the training process.
      checkpoint_step: The step interval for regular checkpoints saved to GCS.
      mode: The checkpointing mode (regular, emergency, or multi_tier).
      replicator_backup_time: Time for replicator backup (emergency/multi_tier only).
      local_checkpoint_step: Step interval for local checkpoints (emergency/multi_tier only).
      ram_disk_size_in_mi: RAM disk size (emergency/multi_tier only).
      base_dir: Base directory for outputs.
    """

    self.cluster = cluster
    self.machine_type = machine_type
    self.accelerator = accelerator
    self.slices = slices
    self.model_name = model_name
    self.short_id = short_id
    self.step = step
    self.checkpoint_step = checkpoint_step
    self.mode = mode
    self.replicator_backup_time = replicator_backup_time
    self.local_checkpoint_step = local_checkpoint_step
    self.ram_disk_size = ram_disk_size_in_mi
    self.base_dir = base_dir

    # Only create CPC config for emergency/multi_tier modes
    if self.mode in [CheckpointingMode.ECM, CheckpointingMode.MTC]:
      self.cpc_config = checkpoint_util.CheckpointConfiguration(
          project_id=self.cluster.project,
          region=zone_to_region(self.cluster.zone),
          cluster_name=self.cluster.name,
          gcs_bucket=gcs_bucket.MTC_AUTOMATION_BUCKET.removeprefix("gs://"),
          ramdisk_memory_in_mi=self.ram_disk_size,
          machine_type=self.machine_type,
      )
    else:
      self.cpc_config = None

  def generate_steps_to_validate(
      self, checkpoint_period: Optional[int] = None
  ) -> list[int]:
    """Generate list of steps to validate based on checkpoint period.

    Args:
      checkpoint_period: Step interval for checkpoints. If None, uses checkpoint_step.
    """
    total_steps = self.step

    # Use provided checkpoint_period or default to GCS checkpoint period
    k = (
        checkpoint_period
        if checkpoint_period is not None
        else self.checkpoint_step
    )

    last_step = self.step - 1
    return [*range(0, total_steps, k), last_step]

  def generate_workload_command(
      self,
      run_name: str,
      out_folder: str,
      checkpoint_dir: Optional[str] = None,
      enable_multi_tier_checkp: Optional[bool] = False,
  ) -> tuple[str, ...]:
    """Generate workload command based on checkpointing mode."""

    base_command = (
        "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
        "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
        "python3 -m MaxText.train MaxText/configs/base.yml "
        "remat_policy=full "
        "global_parameter_scale=1 "
        f"base_output_directory={posixpath.join(self.base_dir, out_folder)} "
        "dataset_type=synthetic "
        f"steps={self.step} "
        "per_device_batch_size=2 "
        "max_target_length=256 "
        f"model_name={self.model_name} "
        "reuse_example_batch=1 "
        f"checkpoint_period={self.checkpoint_step} "
        f"run_name={run_name}"
    )

    # Add mode-specific parameters
    if self.mode == CheckpointingMode.REG:
      # Regular checkpointing - basic command
      return (base_command,)

    elif self.mode == CheckpointingMode.ECM:
      # Emergency checkpointing
      return (
          base_command
          + " "
          + (
              "enable_emergency_checkpoint=true "
              f"local_checkpoint_directory={checkpoint_dir or DEFAULT_RAM_DISK} "
              f"local_checkpoint_period={self.local_checkpoint_step} "
              "enable_multi_tier_checkpointing={enable_multi_tier_checkp}"
          ),
      )

    elif self.mode == CheckpointingMode.MTC:
      # Multi-tier checkpointing
      return (
          base_command
          + " "
          + (
              "enable_emergency_checkpoint=true "
              f"local_checkpoint_directory={checkpoint_dir or DEFAULT_RAM_DISK} "
              f"local_checkpoint_period={self.local_checkpoint_step} "
              "enable_multi_tier_checkpointing=true "
              f"multi_tier_checkpointing_backup_interval_minutes={self.replicator_backup_time}"
          ),
      )

    else:
      raise ValueError(f"Unsupported checkpointing mode: {self.mode}")
