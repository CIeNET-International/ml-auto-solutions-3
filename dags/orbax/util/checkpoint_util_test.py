"""Tests for the CheckpointConfiguration lease in checkpoint_util."""

import datetime
import unittest
from unittest import mock

from kubernetes.client.rest import ApiException

from dags.orbax.util import checkpoint_util

ME = checkpoint_util.CpcOwner(env="liuryan", dag_id="dag_a", run_id="run_1")
OTHER_SAME_ENV = checkpoint_util.CpcOwner(
    env="liuryan", dag_id="dag_b", run_id="run_2"
)
OTHER_ENV = checkpoint_util.CpcOwner(
    env="ml-automation-solutions", dag_id="dag_a", run_id="run_9"
)


def _cpc_object(owner=None, age=datetime.timedelta(minutes=5), deleting=False):
  created = datetime.datetime.now(datetime.timezone.utc) - age
  metadata = {
      "name": "my-checkpointconfiguration",
      "creationTimestamp": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
  }
  if owner is not None:
    metadata["annotations"] = owner.annotations()
  if deleting:
    metadata["deletionTimestamp"] = metadata["creationTimestamp"]
  return {"metadata": metadata, "spec": {}}


def _api_exception(status):
  return ApiException(status=status, reason=str(status))


class CheckpointConfigurationLeaseTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.api = mock.Mock()
    self.cpc = mock.Mock(spec=checkpoint_util.CheckpointConfiguration)
    self.cpc.create_custom_objects_api_client.return_value = self.api
    self.cpc.load_yaml_and_parse_body.return_value = (
        "checkpointing.gke.io",
        "v1",
        "checkpointconfigurations",
        "my-checkpointconfiguration",
        {
            "apiVersion": "checkpointing.gke.io/v1",
            "kind": "CheckpointConfiguration",
            "metadata": {"name": "my-checkpointconfiguration"},
            "spec": {"cloudStorageBucketName": "bucket"},
        },
    )
    self.run_finished = mock.patch.object(
        checkpoint_util, "_dag_run_is_finished", return_value=False
    ).start()
    self.addCleanup(mock.patch.stopall)

  def _existing(self, cpc_object):
    """Makes the API report `cpc_object` (None for 404) on get."""
    if cpc_object is None:
      self.api.get_cluster_custom_object.side_effect = _api_exception(404)
    else:
      self.api.get_cluster_custom_object.return_value = cpc_object

  # --- apply_cpc / _acquire_cpc -------------------------------------------

  def test_acquire_creates_cpc_tagged_with_owner(self):
    self._existing(None)

    self.assertTrue(checkpoint_util._acquire_cpc(self.cpc, ME))

    body = self.api.create_cluster_custom_object.call_args.kwargs["body"]
    self.assertEqual(body["metadata"]["annotations"], ME.annotations())
    self.assertEqual(body["metadata"]["name"], "my-checkpointconfiguration")
    self.assertEqual(body["spec"], {"cloudStorageBucketName": "bucket"})
    self.api.delete_cluster_custom_object.assert_not_called()

  def test_acquire_does_not_mutate_template_body(self):
    self._existing(None)
    checkpoint_util._acquire_cpc(self.cpc, ME)
    template = self.cpc.load_yaml_and_parse_body.return_value[4]
    self.assertNotIn("annotations", template["metadata"])

  def test_acquire_waits_when_create_loses_race(self):
    self._existing(None)
    self.api.create_cluster_custom_object.side_effect = _api_exception(409)

    self.assertFalse(checkpoint_util._acquire_cpc(self.cpc, ME))

  def test_acquire_succeeds_when_already_owned_by_this_run(self):
    self._existing(_cpc_object(owner=ME))

    self.assertTrue(checkpoint_util._acquire_cpc(self.cpc, ME))

    self.api.create_cluster_custom_object.assert_not_called()
    self.api.delete_cluster_custom_object.assert_not_called()

  def test_acquire_waits_for_live_run_in_same_env(self):
    self._existing(_cpc_object(owner=OTHER_SAME_ENV))
    self.run_finished.return_value = False

    self.assertFalse(checkpoint_util._acquire_cpc(self.cpc, ME))

    self.api.create_cluster_custom_object.assert_not_called()
    self.api.delete_cluster_custom_object.assert_not_called()

  def test_acquire_deletes_lease_of_finished_run_in_same_env(self):
    self._existing(_cpc_object(owner=OTHER_SAME_ENV))
    self.run_finished.return_value = True

    self.assertFalse(checkpoint_util._acquire_cpc(self.cpc, ME))

    self.run_finished.assert_called_once_with(OTHER_SAME_ENV)
    self.api.delete_cluster_custom_object.assert_called_once()
    self.api.create_cluster_custom_object.assert_not_called()

  def test_acquire_never_deletes_lease_from_other_env(self):
    self._existing(_cpc_object(owner=OTHER_ENV))
    self.run_finished.return_value = True

    self.assertFalse(checkpoint_util._acquire_cpc(self.cpc, ME))

    self.run_finished.assert_not_called()
    self.api.delete_cluster_custom_object.assert_not_called()

  def test_acquire_waits_for_recent_untagged_cpc(self):
    self._existing(_cpc_object(owner=None, age=datetime.timedelta(hours=3)))

    self.assertFalse(checkpoint_util._acquire_cpc(self.cpc, ME))

    self.api.delete_cluster_custom_object.assert_not_called()

  def test_acquire_deletes_old_untagged_cpc(self):
    self._existing(_cpc_object(owner=None, age=datetime.timedelta(hours=5)))

    self.assertFalse(checkpoint_util._acquire_cpc(self.cpc, ME))

    self.api.delete_cluster_custom_object.assert_called_once()

  def test_acquire_propagates_unexpected_get_errors(self):
    self.api.get_cluster_custom_object.side_effect = _api_exception(403)

    with self.assertRaises(ApiException):
      checkpoint_util._acquire_cpc(self.cpc, ME)

  def test_acquire_propagates_unexpected_create_errors(self):
    self._existing(None)
    self.api.create_cluster_custom_object.side_effect = _api_exception(422)

    with self.assertRaises(ApiException):
      checkpoint_util._acquire_cpc(self.cpc, ME)

  # --- wait_for_cpc_deletion / _release_cpc --------------------------------

  def test_release_is_done_when_cpc_is_absent(self):
    self._existing(None)

    self.assertTrue(checkpoint_util._release_cpc(self.cpc, ME))

    self.api.delete_cluster_custom_object.assert_not_called()

  def test_release_deletes_own_cpc_and_keeps_waiting(self):
    self._existing(_cpc_object(owner=ME))

    self.assertFalse(checkpoint_util._release_cpc(self.cpc, ME))

    kwargs = self.api.delete_cluster_custom_object.call_args.kwargs
    self.assertEqual(kwargs["name"], "my-checkpointconfiguration")
    self.assertEqual(kwargs["body"].propagation_policy, "Foreground")

  def test_release_keeps_deleting_own_cpc_while_deletion_in_progress(self):
    self._existing(_cpc_object(owner=ME, deleting=True))

    self.assertFalse(checkpoint_util._release_cpc(self.cpc, ME))

  def test_release_leaves_live_run_of_other_dag_alone(self):
    self._existing(_cpc_object(owner=OTHER_SAME_ENV))
    self.run_finished.return_value = False

    self.assertFalse(checkpoint_util._release_cpc(self.cpc, ME))

    self.api.delete_cluster_custom_object.assert_not_called()

  def test_release_deletes_stale_lease(self):
    self._existing(_cpc_object(owner=OTHER_SAME_ENV))
    self.run_finished.return_value = True

    self.assertFalse(checkpoint_util._release_cpc(self.cpc, ME))

    self.api.delete_cluster_custom_object.assert_called_once()

  def test_release_tolerates_404_on_delete(self):
    self._existing(_cpc_object(owner=ME))
    self.api.delete_cluster_custom_object.side_effect = _api_exception(404)

    self.assertFalse(checkpoint_util._release_cpc(self.cpc, ME))

  # --- helpers --------------------------------------------------------------

  def test_owner_from_untagged_object_is_none(self):
    self.assertIsNone(checkpoint_util.CpcOwner.from_object(_cpc_object()))
    self.assertIsNone(checkpoint_util.CpcOwner.from_object({}))

  def test_owner_round_trips_through_annotations(self):
    self.assertEqual(
        checkpoint_util.CpcOwner.from_object(_cpc_object(owner=ME)), ME
    )

  def test_untagged_cpc_without_timestamp_is_not_stale(self):
    self.assertFalse(checkpoint_util._is_stale({"metadata": {}}, ME))

  @mock.patch.object(checkpoint_util, "create_session")
  @mock.patch.object(checkpoint_util.DagRun, "find")
  def test_dag_run_is_finished(self, find, create_session):
    mock.patch.stopall()  # use the real _dag_run_is_finished here
    create_session.return_value.__enter__.return_value = mock.sentinel.session

    find.return_value = []
    self.assertTrue(checkpoint_util._dag_run_is_finished(OTHER_SAME_ENV))

    find.return_value = [mock.Mock(state="running")]
    self.assertFalse(checkpoint_util._dag_run_is_finished(OTHER_SAME_ENV))

    find.return_value = [mock.Mock(state="failed")]
    self.assertTrue(checkpoint_util._dag_run_is_finished(OTHER_SAME_ENV))

    find.assert_called_with(
        dag_id="dag_b", run_id="run_2", session=mock.sentinel.session
    )

  @mock.patch.dict("os.environ", {"COMPOSER_ENVIRONMENT": "liuryan"})
  @mock.patch.object(checkpoint_util, "get_current_context")
  def test_current_owner_uses_dag_run_from_context(self, get_current_context):
    get_current_context.return_value = {
        "dag_run": mock.Mock(dag_id="dag_a", run_id="run_1")
    }
    self.assertEqual(checkpoint_util._current_owner(), ME)


if __name__ == "__main__":
  unittest.main()
