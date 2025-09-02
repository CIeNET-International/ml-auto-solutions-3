"""
A DAG to run MaxText multi-tier checkpointing tests.

This DAG performs a series of tests to save and validate checkpoints
for the MaxText model. It runs tests in two modes: one with the replicator
service enabled (Multi-tier Checkpointing). The tests are executed on a TPU
multi-pod cluster.
"""

import datetime
from dataclasses import dataclass
from typing import Optional

from airflow import models
from airflow.utils.task_group import TaskGroup

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import validation_util
from dags.orbax.util import checkpoint_util
from xlml.utils import xpk
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_ecm_restore_from_gcs"
BASE_OUTPUT_DIR = gcs_bucket.MTC_AUTOMATION_BUCKET

# Only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
)]
RAM_DISK = "/local"


@dataclass
class Checkpointing:
  """
  Represents the information of a checkpointing mechanism.

  Attributes:
    name: A unique name for the checkpointing configuration.
    use_replicator: Indicates whether a replicator is enabled.
  """

  name: str
  use_replicator: bool


@dataclass
class TestConfig:
  """Holds the general configuration for a checkpointing test."""

  cluster: XpkClusters
  machine_type: str
  accelerator: str
  slices: list[int]
  model_name: str
  short_id: str
  replicator_backup_time: int
  step: int
  local_checkpoint_step: int
  checkpoint_step: int
  checkpoint_period: int
  ram_disk_size: str
  cpc_config: checkpoint_util.CheckpointConfiguration

  def __init__(
      self,
      cluster: XpkClusters,
      machine_type: str,
      accelerator: str,
      slices: list[int],
      model_name: str,
      short_id: str,
      replicator_backup_time: int,
      step: int,
      local_checkpoint_step: int,
      checkpoint_period: int,
      ram_disk_size_in_mi: str,
      checkpoint_step: Optional[int] = None,
  ):
    """
    Initializes the test configurations.

    Args:
      cluster: The specified cluster to be used for the test.
      machine_type: The type of machine (e.g., GPU, TPU).
      accelerator: The type of accelerator (e.g., GPU, TPU) to use.
      slices: The number of slices to be used.
      model_name: The name of the model being tested.
      short_id: A short identifier for the test run.
      replicator_backup_time: The allowed time for replicator takes to backup
        and store checkpoint to bucket
      step: The current step of the training process.
      local_checkpoint_step: The step interval for local checkpoints.
      checkpoint_period: The step interval for regular checkpoints to GCS.
      checkpoint_step: The step interval for the checkpoints store in the
        bucket.
      ram_disk_size_in_mi: The size in mebibytes (Mi) about the RAM disk in the
        CSI driver. The unit is in mebibytes (Mi) but the value should be passed
        as a string with the unit, e.g., "2G" or "2048M". Defaults to "100G"".
    """

    self.cluster = cluster
    self.machine_type = machine_type
    self.accelerator = accelerator
    self.slices = slices
    self.model_name = model_name
    self.short_id = short_id
    self.replicator_backup_time = replicator_backup_time
    self.step = step
    self.local_checkpoint_step = local_checkpoint_step
    self.checkpoint_period = checkpoint_period
    self.checkpoint_step = checkpoint_step
    self.ram_disk_size = ram_disk_size_in_mi
    self.cpc_config = checkpoint_util.CheckpointConfiguration(
        project_id=self.cluster.project,
        region=zone_to_region(self.cluster.zone),
        cluster_name=self.cluster.name,
        gcs_bucket=BASE_OUTPUT_DIR.removeprefix("gs://"),
        ramdisk_memory_in_mi=self.ram_disk_size,
        machine_type=self.machine_type,
    )

  def generate_workload_command(
      self,
      cp: Checkpointing,
      checkpoint_dir: str,
      run_name: str,
      custom_step: Optional[int] = None,
  ) -> str:
    steps = custom_step if custom_step is not None else self.step
    return (
        "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
        "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
        "python3 -m MaxText.train MaxText/configs/base.yml "
        "remat_policy=full "
        "global_parameter_scale=1 "
        f"base_output_directory={BASE_OUTPUT_DIR} "
        "dataset_type=synthetic "
        f"steps={steps} "
        "per_device_batch_size=1 "
        "max_target_length=256 "
        f"model_name={self.model_name} "
        "per_device_batch_size=2 "
        "reuse_example_batch=1 "
        "enable_emergency_checkpoint=true "
        f"local_checkpoint_directory={checkpoint_dir} "
        f"local_checkpoint_period={self.local_checkpoint_step} "
        f"checkpoint_period={self.checkpoint_period} "
        f"use_replicator_service={cp.use_replicator} "
        f"replicator_backup_interval_minutes={self.replicator_backup_time} "
        f"run_name={run_name}",
    )


with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 12),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "emergency_checkpoint_manager",
        "multitier_checkpointing",
        "nightly",
        "orbax",
    ],
    description="A DAG to test MaxText Emergency Checkpoint Manager GCS restore functionality.",
    doc_md="""
      # Emergency Checkpoint Manager GCS Restore Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      that the **Emergency Checkpoint Manager (ECM)** can successfully save
      checkpoints to GCS and restore from them when local checkpoints are unavailable.

      ### Test Scenario
      The DAG tests the critical scenario where:
      1. Training runs normally with emergency checkpoints saved to GCS
      2. Local checkpoints are deleted (simulating failure/preemption)
      3. Training resumes and successfully restores from GCS emergency checkpoints

      ### Prerequisites
      - An existing cluster with Multi-tier Checkpointing configuration enabled
      - A GCS bucket with HNS (Hierarchical Namespace) enabled
      - Emergency Checkpoint Manager functionality enabled

      ### Test Flow
      1. **Apply Configuration:** Deploy Checkpoint Configuration YAML to enable MTC features
      2. **Initial Training (0-100 steps):** Run MaxText with emergency checkpointing enabled
         - Saves regular checkpoints to GCS every 25 steps (0, 25, 50, 75, 99)
         - Saves local checkpoints to RAM disk for faster access
      3. **RAM Disk Cleanup:** Remove all local checkpoints to force GCS restoration
      4. **Resume Training (100-200 steps):** Continue training from where it left off
         - Emergency Checkpoint Manager restores from latest GCS checkpoint
         - Training continues and saves additional checkpoints (100, 125, 150, 175, 199)
      5. **Final Cleanup:** Clean up RAM disk resources
      6. **Comprehensive Validation:**
         - **Log Validation:** Verify checkpoint save/restore events in logs
         - **GCS Restore Validation:** Confirm restoration from GCS occurred
         - **File Validation:** Verify all expected checkpoint files exist in GCS

      ### Key Parameters
      - **checkpoint_period=25:** Regular checkpoint interval for GCS saves
      - **local_checkpoint_step=20:** Local checkpoint interval for RAM disk
      - **Initial training:** 100 steps, **Resume training:** 200 steps total
      - **Emergency checkpointing:** Enabled throughout the test

      ### Success Criteria
      The test passes when:
      1. All expected checkpoints are saved to GCS during initial training
      2. Local checkpoints are successfully removed during cleanup
      3. Training successfully resumes from GCS checkpoints (not from scratch)
      4. All checkpoint files are verified to exist in the expected GCS locations
      5. Log validation confirms proper save and restore events occurred
    """,
    concurrency=2,
) as dag:
  # Only one set of test configurations (e.g., v5p-128) is supported at the moment.
  # Other configurations (e.g., v5e and/or v6e) may be introduced later.
  test_configs = [
      TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER_ORBAX,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-ecm-res-gcs",
          replicator_backup_time=30,
          step=100,
          local_checkpoint_step=20,
          checkpoint_period=25,
          ram_disk_size_in_mi="800000Mi",
      ),
  ]

  checkpointing = Checkpointing(
      name="ecm",  # Emergency Checkpointing Manager
      use_replicator=False,
  )
  
  with TaskGroup(
      group_id=f"maxtext_{checkpointing.name}_restore_from_gcs",
  ) as group:
    for mode, image in DOCKER_IMAGES:
      for test_config in test_configs:
        for slice_num in test_config.slices:
          # We conditionally set the trigger_rule on the first task.
          # If first task group failed the next one can execute.
          wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
              trigger_rule="all_done"
          )(test_config.cpc_config)
          apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)
          
          # Generate consistent run name for both training phases
          run_name = checkpoint_util.generate_run_name(
              test_config.short_id,
              checkpointing.name,
              slice_num,
              test_config.accelerator,
          )

          # First training phase - train to step 100
          initial_workload_command = test_config.generate_workload_command(
              cp=checkpointing,
              checkpoint_dir=RAM_DISK,
              slice_number=slice_num,
              run_name=run_name,
          )

          start_time = validation_util.generate_timestamp()
          initial_training_run = gke_config.get_gke_config(
              num_slices=slice_num,
              cluster=test_config.cluster,
              time_out_in_min=60,
              test_name=f"{test_config.short_id}",
              run_model_cmds=initial_workload_command,
              docker_image=image.value,
              test_owner=test_owner.JACKY_F,
          ).run(
              ramdisk_directory=RAM_DISK,
              mtc_enabled=True,
              xpk_branch=BRANCH_ABHINAV_MTC,
              skip_post_process=True,
          )

          cleanup_command = (f"rm -rf {RAM_DISK}/*",)
          ram_disk_cleanup_restore = gke_config.get_gke_config(
              num_slices=slice_num,
              cluster=test_config.cluster,
              time_out_in_min=60,
              test_name=f"{test_config.short_id}-cleanup-restore",
              run_model_cmds=cleanup_command,
              docker_image=image.value,
              test_owner=test_owner.JACKY_F,
          ).run(
              ramdisk_directory=RAM_DISK,
              mtc_enabled=True,
              xpk_branch=BRANCH_ABHINAV_MTC,
              skip_post_process=True,
          )

          # Second training phase - continue from checkpoint and reach step 200
          resume_workload_command = test_config.generate_workload_command(
              cp=checkpointing,
              checkpoint_dir=RAM_DISK,
              slice_number=slice_num,
              run_name=run_name,
              custom_step=200,
          )

          resume_training_run = gke_config.get_gke_config(
              num_slices=slice_num,
              cluster=test_config.cluster,
              time_out_in_min=60,
              test_name=f"{test_config.short_id}-restore",
              run_model_cmds=resume_workload_command,
              docker_image=image.value,
              test_owner=test_owner.JACKY_F,
          ).run(
              ramdisk_directory=RAM_DISK,
              mtc_enabled=True,
              xpk_branch=BRANCH_ABHINAV_MTC,
              skip_post_process=True,
          )

          # Final cleanup after resume training
          ram_disk_cleanup = gke_config.get_gke_config(
              num_slices=slice_num,
              cluster=test_config.cluster,
              time_out_in_min=60,
              test_name=f"{test_config.short_id}-cleanup",
              run_model_cmds=cleanup_command,
              docker_image=image.value,
              test_owner=test_owner.JACKY_F,
          ).run(
              ramdisk_directory=RAM_DISK,
              mtc_enabled=True,
              xpk_branch=BRANCH_ABHINAV_MTC,
              skip_post_process=True,
          )

          end_time = validation_util.generate_timestamp()
          
          # Validation steps for entire training process (0 to 200)
          first_end_step = test_config.step - 1  # 99
          final_step = 200 - 1  # 199
          vali_step_list = [0]  # Start with step 0
          vali_step_list.extend([
              i for i in range(test_config.checkpoint_period, final_step, test_config.checkpoint_period)  # Every checkpoint_period steps
          ])
          vali_step_list.append(first_end_step)  # First training end (99)
          vali_step_list.append(final_step)  # Final step (199)
          vali_step_list = sorted(list(set(vali_step_list)))  # Remove duplicates and sort

          validate_log = validation_util.validate_log_with_gcs_save(
              project_id=test_config.cluster.project,
              location=zone_to_region(test_config.cluster.zone),
              cluster_name=test_config.cluster.name,
              pod_pattern="max.*-job-0-0",
              text_filter="(blocking + background).",
              start_time=start_time,
              end_time=end_time,
              vali_step_list=vali_step_list,
          )

          # Validate that GCS restore happened during the second training run
          validate_gcs_restore = validation_util.validate_gcs_restore_log(
              project_id=test_config.cluster.project,
              location=zone_to_region(test_config.cluster.zone),
              cluster_name=test_config.cluster.name,
              pod_pattern="max.*-job-0-0", 
              start_time=start_time,
              end_time=end_time,
          )

          # Validate that checkpoint files exist in GCS bucket
          validate_gcs_files = validation_util.validate_gcs_checkpoint_files(
              bucket_path=f"{BASE_OUTPUT_DIR}/{run_name}/checkpoints",
              vali_step_list=vali_step_list,
          )

          (
              wait_delete_cpc
              >> apply_cpc
              >> run_name
              >> start_time
              >> initial_training_run
              >> ram_disk_cleanup_restore
              >> resume_training_run
              >> ram_disk_cleanup
              >> end_time
              >> [validate_log, validate_gcs_restore, validate_gcs_files]
          )
