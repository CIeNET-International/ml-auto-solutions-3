# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A DAG to run MXLA MaxText tests.
"""
import datetime
import subprocess
import re
from airflow import models
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from dags import composer_env
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config


def add_egress_ip_to_gke(ti, cluster_name, project_id, region):
  """
  Adds the Airflow Egress IP (fetched via XCom) to the Master Authorized
  Networks of the specified GKE cluster.

  The new IP is added as a /32 CIDR block. Existing authorized networks are
  preserved.
  """

  airflow_ip = ti.xcom_pull(task_ids='get_airflow_egress_ip')
  if not airflow_ip:
    raise ValueError("cannot get Airflow Egress IP from XCom")

  new_cidr = f"{airflow_ip}/32"
  print(f"Airflow Egress IP (New CIDR): {new_cidr}")

  describe_cmd = [
      "gcloud",
      "container",
      "clusters",
      "describe",
      cluster_name,
      f"--region={region}",
      f"--project={project_id}",
      "--format=value(masterAuthorizedNetworksConfig.cidrBlocks)",
  ]

  try:
    result = subprocess.run(
        describe_cmd, capture_output=True, text=True, check=True
    )
    existing_networks_str = result.stdout.strip()
  except subprocess.CalledProcessError as e:
    print(f"describe GKE failed: {e.stderr}")
    raise

  cidr_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}\b"
  existing_networks = re.findall(cidr_pattern, existing_networks_str)

  all_networks = set(existing_networks)
  all_networks.add(new_cidr)

  master_networks = ",".join(sorted(list(all_networks)))

  print(f"Complete Master Authorized Networks List: {master_networks}")

  update_cmd = [
      "gcloud",
      "container",
      "clusters",
      "update",
      cluster_name,
      f"--region={region}",
      f"--project={project_id}",
      "--enable-master-authorized-networks",
      f"--master-authorized-networks={master_networks}",
  ]

  print(f"Executing update command: {' '.join(update_cmd)}")
  try:
    subprocess.run(update_cmd, check=True)
    print("GKE networks update success.")
  except subprocess.CalledProcessError as e:
    print(f"Update GKE cluster failed: {e.stderr}")
    raise


def remove_egress_ip_from_gke(ti, cluster_name, project_id, region):
  """
  Removes the Airflow Egress IP (fetched via XCom) from the Master Authorized
  Networks of the specified GKE cluster.

  The function first describes the cluster, finds the Airflow IP CIDR block,
  removes it from the list, and updates the cluster configuration.
  """

  airflow_ip = ti.xcom_pull(task_ids="get_airflow_egress_ip")

  if not airflow_ip:
    print("Warning: Airflow IP not found in XCom. Skipping removal.")
    return

  target_cidr = f"{airflow_ip}/32"
  print(f"Target CIDR for removal: {target_cidr}")

  describe_cmd = [
      "gcloud",
      "container",
      "clusters",
      "describe",
      cluster_name,
      f"--region={region}",
      f"--project={project_id}",
      "--format=value(masterAuthorizedNetworksConfig.cidrBlocks)",
  ]

  try:
    result = subprocess.run(
        describe_cmd, capture_output=True, text=True, check=True
    )
    existing_networks_str = result.stdout.strip()
  except subprocess.CalledProcessError as e:
    print(f"describe GKE cluster failed: {e.stderr}")
    return

  cidr_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}\b"
  # existing_networks ['IP/32', 'IP/32', ...] list
  existing_networks_clean = re.findall(cidr_pattern, existing_networks_str)

  all_networks = set(existing_networks_clean)
  all_networks.discard("")

  if target_cidr in all_networks:
    all_networks.remove(target_cidr)
    print(f"Successfully removed {target_cidr} from the list.")
  else:
    print(f"Target CIDR {target_cidr} not found. Skipping removal.")
    return

  master_networks = ",".join(sorted(list(all_networks)))

  print(f"Complete Networks List after removal (Cleaned): {master_networks}")

  update_cmd = [
      "gcloud",
      "container",
      "clusters",
      "update",
      cluster_name,
      f"--region={region}",
      f"--project={project_id}",
      "--enable-master-authorized-networks",
      f"--master-authorized-networks={master_networks}",
  ]

  print(f"Executing update command: {' '.join(update_cmd)}")
  try:
    subprocess.run(update_cmd, check=True)
    print("Remove GKE network success！")
  except subprocess.CalledProcessError as e:
    print(f"remove GKE network failed, or only one ip left: {e.stderr}")


SCHEDULED_TIME = "None" if composer_env.is_prod_env() else None
V6E_CLUSTER_NAME = "bodaborg-v6e-8-yucmhab-c"
V6E_PROJECT_ID = "tpu-prod-env-one-vm"
V6E_REGION = "us-east5"

with models.DAG(
    dag_id="mxla_maxtext_nightly_gke",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "gke",
        "nightly",
        "mlscale_perfx",
    ],
    start_date=datetime.datetime(2024, 3, 12),
    catchup=False,
) as dag:
  jax_nightly_image = DockerImage.MAXTEXT_TPU_JAX_NIGHTLY
  DEFAULT_TEST_NAME = "mxla-maxtext-nightly-gke"

  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  get_airflow_egress_ip = BashOperator(
      task_id="get_airflow_egress_ip",
      bash_command="curl https://ifconfig.me",
  )

  add_ip_to_gke_auth_networks = PythonOperator(
      task_id="add_ip_to_gke_auth_networks",
      python_callable=add_egress_ip_to_gke,
      op_kwargs={
          "cluster_name": V6E_CLUSTER_NAME,
          "project_id": V6E_PROJECT_ID,
          "region": V6E_REGION,
      },
  )

  remove_ip_from_gke_auth_networks = PythonOperator(
      task_id="remove_ip_from_gke_auth_networks",
      python_callable=remove_egress_ip_from_gke,
      op_kwargs={
          "cluster_name": V6E_CLUSTER_NAME,
          "project_id": V6E_PROJECT_ID,
          "region": V6E_REGION,
      },
      trigger_rule=TriggerRule.ALL_DONE,
  )

  # --- v5p tests  ---
  maxtext_nightly_1slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_2slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=2,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_4slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=4,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_8slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=8,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  # --- v6e tests  ---
  maxtext_nightly_1slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_2slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=2,
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_4slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=4,
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_8slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=8,
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=DEFAULT_TEST_NAME,
      docker_image=jax_nightly_image.value,
      test_owner="RAYMOND_Z",
  ).run_with_quarantine(quarantine_task_group)

  v5p_test_chain = (
      maxtext_nightly_1slice_v5p_8
      >> maxtext_nightly_2slice_v5p_8
      >> maxtext_nightly_4slice_v5p_8
      >> maxtext_nightly_8slice_v5p_8
  )

  v6e_test_chain = (
      maxtext_nightly_1slice_v6e_8
      >> maxtext_nightly_2slice_v6e_8
      >> maxtext_nightly_4slice_v6e_8
      >> maxtext_nightly_8slice_v6e_8
  )

  ip_setup_chain = get_airflow_egress_ip >> add_ip_to_gke_auth_networks

  _ = ip_setup_chain >> v5p_test_chain

  _ = ip_setup_chain >> v6e_test_chain

  _ = [v5p_test_chain, v6e_test_chain] >> remove_ip_from_gke_auth_networks
