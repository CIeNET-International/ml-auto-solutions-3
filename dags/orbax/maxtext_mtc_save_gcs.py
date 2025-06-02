"""
A DAG to run MaxText multi-tier checkpointing with replicator enabled
validates the local checkpoints are replicated (copy) to bucket
with HNS (Hierarchical Namespace)
"""

import datetime
from airflow import models
from dataclasses import dataclass
from typing import Optional

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import validation_util
from dags.orbax.util import checkpoint_util
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region


# Global variables across test configurations.
SCHEDULE = "0 12 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_mtc_orbax_save_gcs"
DEFAULT_BUCKET = gcs_bucket.MTC_AUTOMATION_BUCKET

# Only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_ORBAX_HEAD,
)]
RAM_DISK = "/local"
USE_REPLICATOR = True


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
    self.checkpoint_step = checkpoint_step
    self.ram_disk_size = ram_disk_size_in_mi
    self.cpc_config = checkpoint_util.CheckpointConfiguration(
        project_id=self.cluster.project,
        region=zone_to_region(self.cluster.zone),
        cluster_name=self.cluster.name,
        gcs_bucket=gcs_bucket.MTC_AUTOMATION_BUCKET.removeprefix("gs://"),
        ramdisk_memory_in_mi=self.ram_disk_size,
        machine_type=self.machine_type,
    )

  def generate_workload_command(
      self,
      checkpoint_dir: str,
      out_folder: str,
      slice_number: int,
  ) -> str:
    run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{self.short_id}-mtc-{slice_number}x-{self.accelerator}-{run_time}"
    return (
        "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
        "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
        "python3 -m MaxText.train MaxText/configs/base.yml "
        "remat_policy=full "
        "global_parameter_scale=1 "
        f"base_output_directory={out_folder} "
        "dataset_type=synthetic "
        f"steps={self.step} "
        "per_device_batch_size=1 "
        "max_target_length=256 "
        f"model_name={self.model_name} "
        "per_device_batch_size=2 "
        "reuse_example_batch=1 "
        "enable_emergency_checkpoint=true "
        f"local_checkpoint_directory={checkpoint_dir} "
        f"local_checkpoint_period={self.local_checkpoint_step} "
        f"use_replicator_service=true "
        f"replicator_backup_interval_minutes={self.replicator_backup_time} "
        f"run_name={run_name}",
    )


with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 30),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "multitier_checkpointing",
        "nightly",
        "orbax",
    ],
    description="DAG that verifies the orbax multi-tier checkpointing saving functionality with replicator to GCS bucket",
    doc_md="",
    concurrency=2,
) as dag:
  test_configs = [
      TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER_ORBAX,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-sv-gcs",
          replicator_backup_time=1,
          step=200,
          local_checkpoint_step=20,
          checkpoint_step=15,
          ram_disk_size_in_mi="800000Mi",
      ),
  ]
  output_folder = f"maxtext_mtc_orbax_save_gcs"
  for mode, image in DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        # We conditionally set the trigger_rule on the first task.
        # If first task group failed the next one can execute.
        wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule="all_done"
        )(test_config.cpc_config)
        apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)
        workload_command = test_config.generate_workload_command(
            checkpoint_dir=RAM_DISK,
            out_folder=output_folder,
            slice_number=slice_num,
        )
        start_time = validation_util.generate_timestamp()
        maxtext_chkpt_run_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-mtc",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.CAMILO_Q,
        ).run(
            ramdisk_directory=RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )
        cleanup_command = (f"rm -rf {RAM_DISK}/*",)
        ram_disk_cleanup = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-cleanup",
            run_model_cmds=cleanup_command,
            docker_image=image.value,
            test_owner=test_owner.CAMILO_Q,
        ).run(
            ramdisk_directory=RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )


        end_time = validation_util.generate_timestamp()

        # Here we are looking for the string '(blocking + background)'.
        # We will compare expected steps with the ones we found when query this
        # regex. Should be the same
        view_logs_saved_steps = validation_util.factory_create_validation_config(
            test_config=test_config,
            namespace="default",
            text_filter="Finished async_save \(blocking \+ background\)\.",
            type_payload="json_payload"
        )

        # Actual Task that would check if local stored steps
        # Need to pass start_time and end_time as variables here since there is a deserialization issue.
        validate_steps_task = validation_util.validate_log_with_step(view_logs_saved_steps=view_logs_saved_steps, start_time=start_time, end_time=end_time)

        # Return a string with the correct bucket path name used in the actual
        # training
        view_pods_check_dir_logs = validation_util.factory_create_validation_config(
            test_config=test_config,
            namespace="default",
            text_filter="Config param checkpoint_dir",
            type_payload="text_payload"
        )

        # We need to get logs from replicator_worker from inside mtc driver.
        # Here we are looking for the string 'Successful: backup for step'. This will tell us that the
        # # checkpoint were backup succesfully. Since all replicator workers need to aggre before the backup
        # We only need logs from one pod.
        view_repl_save_bucket_logs = validation_util.factory_create_validation_config(
            test_config=test_config,
            text_filter="Successful: backup for step",
            namespace="gke-managed-checkpointing",
            type_payload="text_payload"
        )
        validate_gcs_bucket_task = validation_util.validate_log_with_gcs(view_repl_save_bucket_logs=view_repl_save_bucket_logs, view_pods_check_dir_logs=view_pods_check_dir_logs,bucket_name=DEFAULT_BUCKET, start_time=start_time, end_time=end_time)

        (
            wait_delete_cpc
            >> apply_cpc
            >> start_time
            >> maxtext_chkpt_run_test
            >> ram_disk_cleanup
            >> end_time
            >> validate_steps_task
            >> validate_gcs_bucket_task
        )
