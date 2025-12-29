# This test file should be run from the project root directory using:
# python -m unittest discover dags/orbax/tests -p "test_scheduling_helper_utils.py"
#
# Other methods that might work:
# 1. Specific test: python -m unittest dags.orbax.tests.test_scheduling_helper_utils.TestSampleSchedulingHelper.test_scheduling_helper_success

from absl.testing import absltest
from dags.orbax.tests.temp_scheduling_helper import TempSchedulingHelper
from dags.common.vm_resource import XpkClusters


class TestSampleSchedulingHelper(absltest.TestCase):
  """
  Test cases for the SchedulingHelper class.
  """

  def test_scheduling_helper_success(self):
    """
    Test scheduling helper ArrangeScheduleTime with success result.
    """

    schedule = TempSchedulingHelper.ArrangeScheduleTime(
        XpkClusters.TPU_V5P_128_CLUSTER,
        "test_dag_2",
    )

    self.assertEqual(schedule, "15 19 * * *")

  def test_scheduling_helper_with_unregistered_cluster(self):
    """
    Test scheduling helper with nonexistent cluster.
    """
    with self.assertRaises(ValueError) as ctx:
      TempSchedulingHelper.ArrangeScheduleTime(
          XpkClusters.TPU_V5E_256_CLUSTER,
          "test_dag_6",
      )

    self.assertIn(
        f"{XpkClusters.TPU_V5E_256_CLUSTER.name} is not found in the registry",
        str(ctx.exception),
    )

  def test_scheduling_helper_with_unregistered_dag_id(self):
    """
    Test scheduling helper with nonexistent dag_id.
    """
    with self.assertRaises(ValueError) as ctx:
      TempSchedulingHelper.ArrangeScheduleTime(
          XpkClusters.TPU_V5P_128_CLUSTER,
          "nonexistent_dag_id",
      )

    self.assertIn(
        "nonexistent_dag_id is not found in the registry",
        str(ctx.exception),
    )

  def test_scheduling_helper_with_overtime_dag(self):
    """
    Test ArrangeScheduleTime raises ValueError if total DAG schedule
    exceeds 24 hours.
    """
    with self.assertRaises(ValueError) as ctx:
      TempSchedulingHelper.ArrangeScheduleTime(
          XpkClusters.TPU_V5P_128_CLUSTER,
          "test_dag_5",
      )

    self.assertIn("Schedule exceeds 24 hours window", str(ctx.exception))
    self.assertIn("dag_id=test_dag_4", str(ctx.exception))
