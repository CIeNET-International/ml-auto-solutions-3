"""Utility functions for managing Multi-tier Cluster Configuration.

This module provides tasks for creating, applying, and deleting
Multi-tier Driver cluster Configurations for enable Multi Tier Checkpointing.

A `CheckpointConfiguration` (CPC) is cluster-scoped and every orbax DAG uses
the same name on the same cluster, so concurrent DAG runs would otherwise
race: `create` fails with 409 when another run got there first, and deleting
the CPC tears down the Multi-tier Checkpointing driver and RAM disks under a
workload that is still using it. The tasks here therefore treat the CPC as a
lease owned by one DAG run at a time:

* `apply_cpc` creates the CPC tagged with the owning DAG run, and waits while
  another run holds it.
* `wait_for_cpc_deletion` deletes the CPC only if this run owns it (or the
  lease is stale), and otherwise waits for the holder to release it.
"""

import datetime
import os
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Optional

from absl import logging
import yaml

from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import DagRun
from airflow.operators.python import get_current_context
from airflow.utils.session import create_session
from airflow.utils.state import DagRunState

from dags import composer_env
from xlml.utils import gke

# Annotations recording which DAG run created the CPC.
OWNER_ENV_ANNOTATION = "ml-auto-solutions/owner-env"
OWNER_DAG_ID_ANNOTATION = "ml-auto-solutions/owner-dag-id"
OWNER_RUN_ID_ANNOTATION = "ml-auto-solutions/owner-run-id"

# A CPC without owner annotations (created manually or by code predating the
# lease) is treated as abandoned once it is older than this. The longest a DAG
# run holds the CPC is the two-phase resume tests: two 60-minute workloads plus
# node provisioning and validation, well under four hours.
LEGACY_CPC_MAX_AGE = datetime.timedelta(hours=4)

# How often the lease sensors poll, and how long a run waits for another run
# to release the CPC before failing. The sensors run in reschedule mode so a
# queued run does not hold a worker slot while it waits.
CPC_POKE_INTERVAL_SECONDS = 30
CPC_WAIT_TIMEOUT = datetime.timedelta(hours=6)


@dataclass
class CheckpointConfiguration:
  """
  A dataclass to hold attributes of a Cloud Public Compute (CPC) instance.
  Initializes the CheckpointConfiguration.
  """

  project_id: str
  region: str
  cluster_name: str
  gcs_bucket: str
  machine_type: str
  ramdisk_memory_in_mi: str
  toleration_key: str = "google.com/tpu"

  def load_yaml_and_parse_body(
      self,
  ) -> tuple[str, str, str, str, dict[str, any]]:
    """
    Loads a YAML template, fills it from the class attributes and parses it.

    This method constructs a CheckpointConfiguration YAML manifest as a string,
    using class attributes such as `self.gcs_bucket`, `self.machine_type`,
    `self.toleration_key`, and `self.ramdisk_memory` to fill in the
    placeholders. It then uses `yaml.safe_load` to convert this YAML string
    into a Python dictionary.

    Finally, it extracts key fields—group, version, plural, and name—from the
    loaded dictionary for use in API requests or other operations.

    Returns:
      tuple[str, str, str, str, dict[str, any]]: A tuple containing the
      extracted API group, API version, plural name, resource name, and the
      full parsed YAML body as a dictionary.
    """

    cpc_yaml_template = f"""
      apiVersion: checkpointing.gke.io/v1
      kind: CheckpointConfiguration
      metadata:
        name: my-checkpointconfiguration # This name will be used for deletion
      spec:
        cloudStorageBucketName: {self.gcs_bucket}
        nodeSelector:
          node.kubernetes.io/instance-type: {self.machine_type}
        tolerations:
        - key: {self.toleration_key}
          operator: Exists
          effect: NoSchedule
        inMemoryVolumeSize: {self.ramdisk_memory_in_mi}
    """
    logging.info(f"CPC ==> {cpc_yaml_template}")
    cpc_body = yaml.safe_load(cpc_yaml_template)
    group, version = cpc_body.get("apiVersion").split("/", 1)
    plural = cpc_body.get("kind").lower() + "s"
    name = cpc_body.get("metadata", {}).get("name")

    return group, version, plural, name, cpc_body

  def create_custom_objects_api_client(self) -> k8s_client.CustomObjectsApi:
    """Create a CustomObjectsApi client for the given cluster."""
    return k8s_client.CustomObjectsApi(
        gke.get_authenticated_client(
            self.project_id, self.region, self.cluster_name
        )
    )


@dataclass(frozen=True)
class CpcOwner:
  """Identity of the DAG run that holds the CheckpointConfiguration lease."""

  env: str
  dag_id: str
  run_id: str

  def __str__(self) -> str:
    return f"{self.env}/{self.dag_id}/{self.run_id}"

  def annotations(self) -> dict[str, str]:
    return {
        OWNER_ENV_ANNOTATION: self.env,
        OWNER_DAG_ID_ANNOTATION: self.dag_id,
        OWNER_RUN_ID_ANNOTATION: self.run_id,
    }

  @classmethod
  def from_object(cls, cpc_object: dict[str, Any]) -> Optional["CpcOwner"]:
    """Returns the owner recorded on a CPC object, or None if untagged."""
    annotations = cpc_object.get("metadata", {}).get("annotations") or {}
    try:
      return cls(
          env=annotations[OWNER_ENV_ANNOTATION],
          dag_id=annotations[OWNER_DAG_ID_ANNOTATION],
          run_id=annotations[OWNER_RUN_ID_ANNOTATION],
      )
    except KeyError:
      return None


def _current_owner() -> CpcOwner:
  """Identifies the DAG run executing the current task."""
  dag_run = get_current_context()["dag_run"]
  return CpcOwner(
      env=os.environ.get(composer_env.COMPOSER_ENVIRONMENT, "local"),
      dag_id=dag_run.dag_id,
      run_id=dag_run.run_id,
  )


def _dag_run_is_finished(owner: CpcOwner) -> bool:
  """Whether the owning DAG run in this Airflow environment is over."""
  with create_session() as session:
    runs = DagRun.find(
        dag_id=owner.dag_id, run_id=owner.run_id, session=session
    )
    if not runs:
      # The run was deleted from the metadata DB; nothing will release it.
      return True
    return runs[0].state in (DagRunState.SUCCESS, DagRunState.FAILED)


def _creation_time(cpc_object: dict[str, Any]) -> Optional[datetime.datetime]:
  timestamp = cpc_object.get("metadata", {}).get("creationTimestamp")
  if not timestamp:
    return None
  return datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _is_stale(cpc_object: dict[str, Any], me: CpcOwner) -> bool:
  """Whether the CPC is held by a lease nobody is going to release.

  A lease is stale when its owner is a finished DAG run in this Airflow
  environment (e.g. the run was cleared or killed before its teardown ran),
  or when the CPC carries no owner at all and is older than
  `LEGACY_CPC_MAX_AGE`. A CPC owned by a run in another environment is never
  considered stale, because its state cannot be checked from here.
  """
  owner = CpcOwner.from_object(cpc_object)
  if owner is None:
    created = _creation_time(cpc_object)
    if created is None:
      return False
    age = datetime.datetime.now(datetime.timezone.utc) - created
    return age > LEGACY_CPC_MAX_AGE
  if owner.env != me.env:
    return False
  return _dag_run_is_finished(owner)


def _describe_holder(cpc_object: dict[str, Any]) -> str:
  owner = CpcOwner.from_object(cpc_object)
  created = cpc_object.get("metadata", {}).get("creationTimestamp")
  holder = str(owner) if owner else "an untagged creator"
  deleting = cpc_object.get("metadata", {}).get("deletionTimestamp")
  suffix = " (deletion in progress)" if deleting else ""
  return f"{holder}, created {created}{suffix}"


class _CpcClient:
  """Thin wrapper around CustomObjectsApi for one CheckpointConfiguration."""

  def __init__(self, cpc: CheckpointConfiguration):
    self._api = cpc.create_custom_objects_api_client()
    group, version, plural, name, body = cpc.load_yaml_and_parse_body()
    if not name:
      raise AirflowFailException(
          "Could not determine the CheckpointConfiguration name."
      )
    self._group = group
    self._version = version
    self._plural = plural
    self.name = name
    self.body = body

  def get(self) -> Optional[dict[str, Any]]:
    """Returns the CPC object, or None if it does not exist."""
    try:
      return self._api.get_cluster_custom_object(
          group=self._group,
          version=self._version,
          plural=self._plural,
          name=self.name,
      )
    except ApiException as e:
      if e.status == HTTPStatus.NOT_FOUND:
        return None
      raise

  def create(self, owner: CpcOwner) -> bool:
    """Creates the CPC tagged with `owner`. Returns False on a 409 race."""
    body = dict(self.body)
    body["metadata"] = dict(self.body.get("metadata", {}))
    body["metadata"]["annotations"] = {
        **body["metadata"].get("annotations", {}),
        **owner.annotations(),
    }
    try:
      self._api.create_cluster_custom_object(
          group=self._group,
          version=self._version,
          plural=self._plural,
          body=body,
      )
      return True
    except ApiException as e:
      if e.status == HTTPStatus.CONFLICT:
        return False
      raise

  def delete(self) -> None:
    """Sends a foreground delete; a 404 means it is already gone."""
    try:
      self._api.delete_cluster_custom_object(
          group=self._group,
          version=self._version,
          plural=self._plural,
          name=self.name,
          body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"),
      )
    except ApiException as e:
      if e.status != HTTPStatus.NOT_FOUND:
        raise


def _acquire_cpc(cpc: CheckpointConfiguration, me: CpcOwner) -> bool:
  """One attempt to own the CPC. Returns True once this run holds it."""
  client = _CpcClient(cpc)
  existing = client.get()

  if existing is not None:
    if CpcOwner.from_object(existing) == me:
      logging.info(f"CheckpointConfiguration '{client.name}' is held by us.")
      return True
    if _is_stale(existing, me):
      logging.info(
          f"CheckpointConfiguration '{client.name}' is held by a stale lease"
          f" ({_describe_holder(existing)}); deleting it."
      )
      client.delete()
      return False
    logging.info(
        f"CheckpointConfiguration '{client.name}' is held by"
        f" {_describe_holder(existing)}; waiting for it to be released."
    )
    return False

  logging.info(
      f"Attempting to create CheckpointConfiguration '{client.name}' as {me}..."
  )
  if client.create(me):
    logging.info(f"Created CheckpointConfiguration '{client.name}'.")
    return True
  logging.info(
      f"CheckpointConfiguration '{client.name}' was created by another run"
      " first; waiting for it to be released."
  )
  return False


def _release_cpc(cpc: CheckpointConfiguration, me: CpcOwner) -> bool:
  """One attempt to get rid of the CPC. Returns True once it is gone.

  Only a CPC owned by this run (or a stale lease) is deleted; one held by
  another run is left alone until that run releases it.
  """
  client = _CpcClient(cpc)
  existing = client.get()
  if existing is None:
    logging.info(f"CheckpointConfiguration '{client.name}' is gone.")
    return True

  if CpcOwner.from_object(existing) == me or _is_stale(existing, me):
    logging.info(
        f"Deleting CheckpointConfiguration '{client.name}'"
        f" ({_describe_holder(existing)}); waiting for it to disappear."
    )
    client.delete()
  else:
    logging.info(
        f"CheckpointConfiguration '{client.name}' is held by"
        f" {_describe_holder(existing)}; waiting for it to be released."
    )
  return False


@task.sensor(
    poke_interval=CPC_POKE_INTERVAL_SECONDS,
    timeout=CPC_WAIT_TIMEOUT.total_seconds(),
    mode="reschedule",
)
def apply_cpc(cpc: CheckpointConfiguration) -> bool:
  """Creates the CheckpointConfiguration on behalf of this DAG run.

  Succeeds once the CPC exists and is owned by this run. While another run
  holds it, the sensor keeps waiting; a stale lease is deleted first.
  """
  return _acquire_cpc(cpc, _current_owner())


@task.sensor(
    poke_interval=CPC_POKE_INTERVAL_SECONDS,
    timeout=CPC_WAIT_TIMEOUT.total_seconds(),
    mode="reschedule",
)
def wait_for_cpc_deletion(cpc: CheckpointConfiguration) -> bool:
  """Deletes this run's CheckpointConfiguration and waits until it is gone.

  Used both before `apply_cpc` (to wait for the cluster to be free, deleting
  leftovers from this run or stale leases) and as the teardown that releases
  the CPC for the next run. A CPC held by another live run is not touched.
  """
  return _release_cpc(cpc, _current_owner())
