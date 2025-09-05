"""
A DAG to run MaxText regular checkpointing tests.

This DAG performs a series of tests to save and validate checkpoints
for the MaxText model using the regular checkpointer.
The tests are executed on a TPU multi-pod cluster.
"""

import datetime
from dataclasses import dataclass
import posixpath
from typing import Optional

from airflow import models

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import validation_util
from dags.orbax.util import checkpoint_util
from xlml.utils.xpk import MAIN_BRANCH
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_regular_save"
BASE_OUTPUT_DIR = gcs_bucket.MTC_AUTOMATION_BUCKET

# Only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [
    (
        SetupMode.NIGHTLY,
        DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
    )
]


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

    def __init__(
        self,
        cluster: XpkClusters,
        machine_type: str,
        accelerator: str,
        slices: list[int],
        model_name: str,
        short_id: str,
        step: int,
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
          step: The current step of the training process.
          checkpoint_step: The step interval for regular checkpoints saved to GCS.
        """

        self.cluster = cluster
        self.machine_type = machine_type
        self.accelerator = accelerator
        self.slices = slices
        self.model_name = model_name
        self.short_id = short_id
        self.step = step
        self.checkpoint_step = checkpoint_step

    def generate_workload_command(
        self,
        run_name: str,
        out_folder: str,
    ) -> str:
        return (
            "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
            "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
            "python3 -m MaxText.train MaxText/configs/base.yml "
            "remat_policy=full "
            "global_parameter_scale=1 "
            f"base_output_directory={posixpath.join(BASE_OUTPUT_DIR, out_folder)} "
            "dataset_type=synthetic "
            f"steps={self.step} "
            "per_device_batch_size=1 "
            "max_target_length=256 "
            f"checkpoint_period={self.checkpoint_step} "
            f"model_name={self.model_name} "
            "per_device_batch_size=2 "
            "reuse_example_batch=1 "
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
        "regular_checkpointing",
        "nightly",
        "orbax",
    ],
    description="A DAG to test MaxText regular checkpoint saving functionality.",
    doc_md="""
      # MaxText Regular Checkpointing Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      regular checkpoint saving for the MaxText model. The DAG runs a single
      MaxText training job and validates that checkpoints are saved correctly
      at specified intervals.

      ### Test Scenario
      The DAG tests the standard checkpointing scenario where:
      1. Training runs normally with regular checkpoints saved to GCS at defined intervals
      2. Checkpoint logs are validated to ensure proper save events
      3. GCS bucket is validated to ensure checkpoint files exist

      ### Prerequisites
      - An existing TPU cluster configured for MaxText training
      - Access to a GCS bucket for checkpoint storage

      ### Test Flow
      1. **Start Training:** Run MaxText training job for 100 steps
         - Saves regular checkpoints to GCS every 25 steps (0, 25, 50, 75, 99)
         - No local checkpoints or emergency features are used
      2. **Log Validation:** Verify checkpoint save events in logs
         - Looks for 'Finished async_save (blocking + background)' messages
         - Validates that saves occurred at expected steps
      3. **File Validation:** Verify checkpoint files exist in GCS bucket
         - Checks that actual checkpoint files are present for each expected step

      ### Key Parameters
      - **checkpoint_step=25:** Regular checkpoint interval for GCS saves
      - **step=100:** Total training steps
      - **Model:** llama2-7b on v5p-128 TPU slices

      ### Success Criteria
      The test passes when:
      1. All expected checkpoint save logs are found at steps 0, 25, 50, 75, 99
      2. All corresponding checkpoint files exist in the GCS bucket
      3. No emergency checkpointing or multi-tier features are involved
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
            short_id="max-reg-save",
            step=100,
            checkpoint_step=25,
        ),
    ]

    checkpointing_type = "reg"  # Regular Checkpointing

    for mode, image in DOCKER_IMAGES:
        for test_config in test_configs:
            for slice_num in test_config.slices:
                
                run_name = validation_util.generate_run_name(
                    short_id=test_config.short_id,
                    checkpointing_type=checkpointing_type,
                    slice_number=slice_num,
                    accelerator=test_config.accelerator,
                )

                workload_command = test_config.generate_workload_command(
                    run_name=run_name,
                    out_folder=DAG_TEST_NAME,
                )

                start_time = validation_util.generate_timestamp()

                maxtext_chkpt_run_test = gke_config.get_gke_config(
                    num_slices=slice_num,
                    cluster=test_config.cluster,
                    time_out_in_min=60,
                    test_name=f"{test_config.short_id}-{checkpointing_type}",
                    run_model_cmds=workload_command,
                    docker_image=image.value,
                    test_owner=test_owner.JACKY_F,
                ).run(
                    xpk_branch=MAIN_BRANCH,
                    skip_post_process=True,
                )

                end_time = validation_util.generate_timestamp()

                total_steps = test_config.step
                checkpoint_period = test_config.checkpoint_step
                last_step = test_config.step - 1
                gcs_steps_to_validate = [
                    *range(0, total_steps, checkpoint_period),
                    last_step,
                ]

                validate_log = validation_util.validate_checkpoint_saves(
                    project_id=test_config.cluster.project,
                    location=zone_to_region(test_config.cluster.zone),
                    cluster_name=test_config.cluster.name,
                    steps_to_validate=gcs_steps_to_validate,
                    pod_pattern="max.*-job-0-0",
                    start_time=start_time,
                    end_time=end_time,
                )

                validate_bucket = validation_util.validate_gcs_checkpoint_files(
                    bucket_path=f"{BASE_OUTPUT_DIR}/{DAG_TEST_NAME}/{run_name}/checkpoints",
                    steps_to_validate=gcs_steps_to_validate,
                )

                (
                    start_time
                    >> maxtext_chkpt_run_test
                    >> end_time
                    >> validate_log
                    >> validate_bucket
                )
