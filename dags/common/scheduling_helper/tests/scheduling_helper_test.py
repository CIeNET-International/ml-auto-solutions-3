# This test file should be run from the project root directory using:
# python -m unittest discover dags/common/scheduling_helper/tests -p "scheduling_helper_test.py"
#
# Other methods that might work:
# 1. Specific test: python -m unittest dags.common.scheduling_helper.tests.scheduling_helper_test.TestSampleSchedulingHelper.test_arrangescheduletime_is_correct

"""The test file of scheduling helper"""

from absl.testing import absltest
from dags.common.vm_resource import XpkClusters
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper
from dags.common.scheduling_helper.scheduling_helper import Project


class MockSchedulingHelper(SchedulingHelper):
  """
  Mock class of SchedulingHelper with fake registry
  """

  registry: dict[str, Project] = {
      "normal_dag": Project(
          "dags/common/scheduling_helper/tests/normal_dag/",
          XpkClusters.TPU_V5P_128_CLUSTER,
      ),
      "overtime_dag": Project(
          "dags/common/scheduling_helper/tests/overtime_dag/",
          XpkClusters.TPU_V5E_256_CLUSTER,
      ),
      "no_timeout_dag": Project(
          "dags/common/scheduling_helper/tests/no_timeout_dag/",
          XpkClusters.TPU_V4_16_CLUSTER,
      ),
  }


class TestSampleSchedulingHelper(absltest.TestCase):
  """Test cases for the SchedulingHelper class."""

  # unit test
  def test_arrangescheduletime_is_correct(self):
    expected_result = "0 13 * * *"
    try:
      schedule = MockSchedulingHelper.ArrangeScheduleTime(
          MockSchedulingHelper.registry["normal_dag"], "normal_dag_1"
      )
    except ValueError:
      schedule = None

    self.assertEqual(
        schedule,
        expected_result,
        msg="test_arrangescheduletime_is_correct faild",
    )

  def test_nonexist_dag(self):
    with self.assertRaises(ValueError) as ctx:
      MockSchedulingHelper.ArrangeScheduleTime(
          MockSchedulingHelper.registry["normal_dag"],
          "nonexist_dag",
      )
    self.assertIn("Dag doesn't exist", str(ctx.exception))

  def test_overtime_dag(self):
    with self.assertRaises(ValueError) as ctx:
      MockSchedulingHelper.ArrangeScheduleTime(
          MockSchedulingHelper.registry["overtime_dag"], "overtime_dag_2"
      )
    self.assertIn(
        "Schedule exceeds 24 hours window;adjust the DEFAULT_MARGIN "
        "or dagrun_timeout accordingly.",
        str(ctx.exception),
    )

  def test_dag_without_timeout(self):
    with self.assertRaises(ValueError) as ctx:
      MockSchedulingHelper.ArrangeScheduleTime(
          MockSchedulingHelper.registry["no_timeout_dag"],
          "no_timeout_dag",
      )
    self.assertIn(
        "Dag rundag_timeout parameter is necessary", str(ctx.exception)
    )

  # CI test
  def test_all_dags_correctly_scheduled(self):
    """
    Test all dags under folder are scheduled
    """

    dag_ids = ["normal_dag_1", "normal_dag_2"]
    expected_results = ["0 13 * * *", "15 14 * * *"]

    for dag_id, expected_result in zip(dag_ids, expected_results):
      try:
        schedule = MockSchedulingHelper.ArrangeScheduleTime(
            MockSchedulingHelper.registry["normal_dag"],
            dag_id,
        )
      except ValueError:
        schedule = None

      self.assertEqual(
          schedule,
          expected_result,
          msg=f"DAG {dag_id} scheduled incorrectly",
      )

  def test_schedule_replicated(self):
    """
    Ensure no duplicate clusters are registered in SchedulingHelper.registry.
    """
    cluster_names = []
    seen = set()
    first_duplicate = None

    for project in MockSchedulingHelper.registry.values():
      name = project.cluster.name
      if name in seen and first_duplicate is None:
        first_duplicate = name
      seen.add(name)
      cluster_names.append(name)

    self.assertIsNone(
        first_duplicate,
        f"Duplicate cluster found in registry: {first_duplicate}. ",
    )


if __name__ == "__main__":
  absltest.main()
