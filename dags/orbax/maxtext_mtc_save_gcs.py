"""
A DAG to run MaxText multi-tier checkpointing with replicator enabled
validates the local checkpoints are replicated (copy) to bucket
with HNS (Hierarchical Namespace)
"""

import datetime
from airflow import models
from dataclasses import dataclass

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import logging_mtc as log
from dags.orbax.util import multi_tier_checkpoint_util as mtc
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region


# Global variables across test configurations.
SCHEDULE = "0 12 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_mtc_orbax_save_gcs"
BASE_OUTPUT_DIRECTORY = f"{gcs_bucket.MTC_AUTOMATION_BUCKET}/{DAG_TEST_NAME}"
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
)]
RAM_DISK = "/local"
USE_REPLICATOR = True


@dataclass
class Testconfig:
  """Holds the general configuration for a checkpointing test.

  Attributes:
    cluster: The specified cluster to be used for the test.
    machine_type: The type of machine (e.g., GPU, TPU).
    accelerator: The type of accelerator (e.g., GPU, TPU) to use.
    slices: The number of slices to be used.
    model_name: The name of the model being tested.
    short_id (str): A short id to be used for naming the test run.
    replicator_min: The time the replicator takes to backup checkpoints to bucket.
    step: The current step of the training process.
    local_checkpoint_step: The step interval for local checkpoints.
    checkpoint_step: The step interval for the checkpoints store in the bucket.
    ram_disk_mi: The size about the RAM disk in the CSI driver, in Mi.
  """

  cluster: XpkClusters
  machine_type: str
  accelerator: str
  slices: list[int]
  model_name: str
  short_id: str
  replicator_min: int
  step: int
  local_checkpoint_step: int
  checkpoint_step: int
  ram_disk_mi: str


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
      Testconfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER_ORBAX,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-sv",
          replicator_min=30,
          step=100,
          local_checkpoint_step=20,
          checkpoint_step=25,
          ram_disk_mi="800000Mi",
      ),
  ]
  for mode, image in DOCKER_IMAGES:
    for test_config in test_configs:
      cpc_conf = mtc.CheckpointConfiguration(
          project_id=test_config.cluster.project,
          region=zone_to_region(test_config.cluster.zone),
          cluster_name=test_config.cluster.name,
          gcs_bucket=gcs_bucket.MTC_AUTOMATION_BUCKET.removeprefix("gs://"),
          ramdisk_memory_in_mi=test_config.ram_disk_mi,
          machine_type=test_config.machine_type,
      )
      for slice_num in test_config.slices:
        # We conditionally set the trigger_rule on the first task.
        # If first task group failed the next one can execute.
        init_delete_cpc = mtc.initiate_cpc_deletion(cpc_conf)
        wait_delete_cpc = mtc.wait_for_cpc_deletion.override(
            trigger_rule="all_done"
        )(cpc_conf)
        apply_cpc = mtc.apply_cpc(cpc_conf)
        run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        run_name = (
            f"{test_config.short_id}-mtc-{slice_num}x-{test_config.accelerator}-{run_time}"
        )
        workload_command = (
            "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
            "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
            "python3 -m MaxText.train MaxText/configs/base.yml remat_policy=full "
            f"global_parameter_scale=1 base_output_directory={BASE_OUTPUT_DIRECTORY} "
            f"dataset_type=synthetic steps={test_config.step} per_device_batch_size=1 "
            f"max_target_length=256 model_name={test_config.model_name} per_device_batch_size=2 "
            f"reuse_example_batch=1 enable_emergency_checkpoint=true checkpoint_period={test_config.checkpoint_step} "
            f"local_checkpoint_directory={RAM_DISK} local_checkpoint_period={test_config.local_checkpoint_step} "
            f"use_replicator_service={USE_REPLICATOR} replicator_backup_interval_minutes={test_config.replicator_min} "
            f"run_name={run_name}",
        )

        start_time = log.generate_timestamp()

        maxtext_chkpt_run_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-mtc",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.CAMILO,
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
            test_owner=test_owner.CAMILO,
        ).run(
            ramdisk_directory=RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )

        vali_step = test_config.step- 1
        vali_step_list = [
            i
            for i in range(0, vali_step, test_config.local_checkpoint_step)
        ]
        vali_step_list.append(vali_step)

        end_time = log.generate_timestamp()

        # Here we are looking for the string '(blocking + background)'.
        # We will compare expected steps with the ones we found when query this
        # regex. Should be the same
        validate_steps = log.validate_log_with_step(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter="(blocking + background).",
            start_time=start_time,
            end_time=end_time,
            vali_step_list=vali_step_list,
        )

        # We need to get logs from replicator_worker from inside mtc driver.
        # Here we are looking for the string 'Successful: backup for step'. This will tell us that the
        # # checkpoint were backup succesfully. Since all replicator workers need to aggre before the backup
        # We only need logs from one pod.
        validate_gcs_bucket = log.validate_log_with_gcs(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter="Successful: backup for step",
            namespace="gke-managed-checkpointing",
            container_name="replication-worker",
            pod_pattern="multitier-driver",
            start_time=start_time,
            end_time=end_time,
        )
        (
            init_delete_cpc
            >> wait_delete_cpc
            >> apply_cpc
            >> start_time
            >> maxtext_chkpt_run_test
            >> ram_disk_cleanup
            >> end_time
            >> validate_steps
            >> validate_gcs_bucket
        )
