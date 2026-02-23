# # Copyright 2025 Google LLC
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
"""The test file of scheduling helper using absltest."""

"""Unit tests for scheduling_helper.py."""

import datetime as dt
from unittest.mock import patch
from absl.testing import absltest, parameterized
import datetime as dt
from absl.testing import absltest
from absl.testing import parameterized
from airflow.models import DagBag

from dags.common.scheduling_helper import scheduling_helper


class TestSchedulingHelperBase(parameterized.TestCase):
  """Base class for SchedulingHelper tests with shared mock data."""

  def setUp(self):
    super().setUp()
    # Mock data with non-round numbers to ensure precise calculation
    self.mock_registry = {
        "cluster_a": {
            # Start: 08:00
            "dag_1": dt.timedelta(minutes=12),
            # Start: 08:00 + 12m + 15m = 08:27
            "dag_2": dt.timedelta(minutes=33),
            # Start: 08:27 + 33m + 15m = 09:15
            "dag_3": dt.timedelta(seconds=45),
            # Start: 09:15 + 45s + 15m = 09:30:45 -> 09:30
            "dag_4": dt.timedelta(minutes=20),
            # Start: 09:30:45 + 20m + 15m = 10:05:45 -> 10:05
            "dag_5": dt.timedelta(minutes=10),
            # Start: 10:05:45 + 10m + 15m = 10:30:45 -> 10:30
            "dag_6": dt.timedelta(minutes=5),
        },
        "cluster_b": {
            "dag_x": dt.timedelta(minutes=5),
            # Start: 08:00 + 5m + 15m = 08:20
            "dag_y": dt.timedelta(minutes=10),
            # Start: 08:20 + 10m + 15m = 08:45
        },
    }
    self.expected_schedules = {
        "dag_1": "0 8 * * *",
        "dag_2": "27 8 * * *",
        "dag_3": "15 9 * * *",
        "dag_4": "30 9 * * *",
        "dag_5": "5 10 * * *",
        "dag_6": "30 10 * * *",
        "dag_x": "0 8 * * *",
        "dag_y": "20 8 * * *",
    }


class TestBaseSchedulingFeature(TestSchedulingHelperBase):
  """Validates the cron string generation and stacking logic."""

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_stacking_logic_sequence(self, mock_registered):
    """Verifies the cumulative offset for the entire sequence."""

    mock_registered.items.return_value = self.mock_registry.items()
    for dag_id, expected_cron in self.expected_schedules.items():
      with self.subTest(dag_id=dag_id):
        actual = scheduling_helper.SchedulingHelper.arrange_schedule_time(
            dag_id
        )
        self.assertEqual(actual, expected_cron)

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_output_is_invariant(self, mock_registered):
    """Ensures deterministic output across multiple identical calls."""

    mock_registered.items.return_value = self.mock_registry.items()
    for dag_id, expected_cron in self.expected_schedules.items():
      with self.subTest(dag_id=dag_id):
        res1 = scheduling_helper.SchedulingHelper.arrange_schedule_time(dag_id)
        res2 = scheduling_helper.SchedulingHelper.arrange_schedule_time(dag_id)
        self.assertEqual(res1, expected_cron)
        self.assertEqual(res1, res2)

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_alignment_with_anchor(self, mock_registered):
    """Validates that all schedules align with the anchor time."""

    mock_registered.items.return_value = self.mock_registry.items()
    schedule = scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_1")
    self.assertEqual(schedule, self.expected_schedules["dag_1"])

  @parameterized.named_parameters(
      ("all", scheduling_helper.DayOfWeek.ALL, "*"),
      ("weekday", scheduling_helper.DayOfWeek.WEEK_DAY, "1-5"),
      ("weekend", scheduling_helper.DayOfWeek.WEEKEND, "0,6"),
  )
  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_day_of_week_options(
      self, day_enum, expected_suffix, mock_registered
  ):
    """Checks that the correct day-of-week field is set in the cron string."""

    mock_registered.items.return_value = self.mock_registry.items()
    schedule = scheduling_helper.SchedulingHelper.arrange_schedule_time(
        "dag_6", day_of_week=day_enum
    )
    self.assertTrue(
        schedule.endswith(expected_suffix),
        f"Schedule {schedule} does not end with {expected_suffix}",
    )


class TestUnexpectedCases(TestSchedulingHelperBase):
  """Validates boundary conditions and registration checks."""

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_unregistered_dag(self, mock_registered):
    """
    Ensures that requesting a schedule for an unregistered DAG
    raises the correct error.
    """

    mock_registered.items.return_value = self.mock_registry.items()
    with self.assertRaises(scheduling_helper.UnregisteredDagError):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("ghost_dag")

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_24hours_window_cumulative(self, mock_registered):
    """Validates that the cumulative schedule does not exceed 24 hours."""

    long_dags = {f"d{i}": dt.timedelta(hours=5) for i in range(6)}
    mock_registered.items.return_value = {"c1": long_dags}.items()
    with self.assertRaises(scheduling_helper.ScheduleWindowError):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("d5")

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_24hours_window_single_dag(self, mock_registered):
    """
    Ensures that a single DAG with a duration exceeding 24 hours is rejected.
    """

    mock_registered.items.return_value = {
        "c1": {"huge_dag": dt.timedelta(hours=25)}
    }.items()
    with self.assertRaises(scheduling_helper.ScheduleWindowError) as cm:
      scheduling_helper.SchedulingHelper.arrange_schedule_time("huge_dag")
    self.assertIn("Schedule exceeds 24h window", str(cm.exception))


class TestFormatIntegrity(TestSchedulingHelperBase):
  """Ensures output is valid and deterministic."""

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_output_is_valid_cron(self, mock_registered):
    """Validates that the generated cron string adheres to expected format."""

    mock_registered.items.return_value = self.mock_registry.items()
    # Minute field: matches 0-59 to ensure valid minute range.
    minute = r"[0-5]?\d"
    # Hour field: matches 0-23 to ensure valid hour range.
    hour = r"[0-1]?\d|2[0-3]"
    # SchedulingHelper currently only supports a once-a-day schedule.
    fixed = r"\*"
    # Corresponds to the DayOfWeek Enum.
    week = r"\*|1-5|0,6"
    # Final assembled expected cron pattern for validation.
    pattern = rf"^{minute} {hour} {fixed} {fixed} {week}$"
    res = scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_1")
    self.assertRegex(res, pattern)


if __name__ == "__main__":
  absltest.main()
def run_registration_check(dag_folder: str) -> bool:
  dagbag = DagBag(dag_folder=dag_folder, include_examples=False)
class TestSchedulingHelper(parameterized.TestCase):
  """Test cases for the SchedulingHelper class logic."""

  def setUp(self):
    super().setUp()
    self.dag_folder = "dags/tpu_observability"
    # Mock data to simulate the stacking logic
    # Offset calculation: Start = 08:00 + Sum(Previous Timeouts + 15m Margin)
    self.fake_registered_dags = {
        "fake_cluster": {
            "dag_1": dt.timedelta(minutes=30),  # Start: 08:00
            "dag_2": dt.timedelta(minutes=30),  # Start: 08:00 + 30 + 15 = 08:45
            "dag_3": dt.timedelta(minutes=60),  # Start: 08:45 + 30 + 15 = 09:30
        },
        "overtime_cluster": {
            "extreme_dag": dt.timedelta(hours=25),
        },
    }
    # Patch the global REGISTERED_DAGS in the module
    self.patcher = absltest.mock.patch(
        "dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS",
        self.fake_registered_dags,
    )
    self.patcher.start()

  def tearDown(self):
    self.patcher.stop()
    super().tearDown()

  # --- Unit Tests ---

  def test_get_dag_timeout_is_correct(self):
    """Verifies that get_dag_timeout retrieves the correct timedelta."""
    timeout = scheduling_helper.get_dag_timeout("dag_2")
    self.assertEqual(timeout, dt.timedelta(minutes=30))

  def test_arrange_schedule_time_logic(self):
    """Tests the stacking logic (Anchor + Offset + Margin)."""
    # 1st DAG should be at the anchor (08:00)
    self.assertEqual(
        scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_1"),
        "0 8 * * *",
    )
    # 2nd DAG = 08:00 + 30m (timeout) + 15m (margin) = 08:45
    self.assertEqual(
        scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_2"),
        "45 8 * * *",
    )
    # 3rd DAG = 08:45 + 30m (timeout) + 15m (margin) = 09:30
    self.assertEqual(
        scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_3"),
        "30 9 * * *",
    )

  def test_day_of_week_options(self):
    """Verifies that DayOfWeek enum correctly applies to the Cron string."""
    dag_id = "dag_1"
    # Weekend mode
    schedule = scheduling_helper.SchedulingHelper.arrange_schedule_time(
        dag_id, scheduling_helper.DayOfWeek.WEEKEND
    )
    self.assertEqual(schedule, "0 8 * * 0,6")

  # --- Error Handling Tests ---

  def test_nonexist_dag(self):
    """Tests that a ValueError is raised for unregistered DAGs."""
    with self.assertRaisesRegex(ValueError, "is not registered"):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("ghost_dag")

  def test_overtime_error(self):
    """Tests that schedules exceeding 24 hours trigger a ValueError."""
    with self.assertRaisesRegex(ValueError, "Schedule exceeds 24h window"):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("extreme_dag")

  # --- CI Test (Production Data Check) ---

  def test_registration_check(self):
    """
    CI Test: Ensures every DAG file in the folder is registered in the helper.
    """
    self.patcher.stop()

    try:
      dagbag = DagBag(dag_folder=self.dag_folder, include_examples=False)
      actual_ids = set(dagbag.dag_ids)
      registered_ids = set()
      for dags_dict in scheduling_helper.REGISTERED_DAGS.values():
        registered_ids.update(dags_dict.keys())
      missing = actual_ids - registered_ids
      self.assertEmpty(
          missing,
          msg=(
              f"The following DAGs exist in {self.dag_folder} but are NOT "
              f"registered in scheduling_helper.py: {missing}. "
              "Please add them to REGISTERED_DAGS to ensure they are scheduled."
          ),
      )
      extra = registered_ids - actual_ids
      if extra:
        print(f"\n[WARNING]: DAGs registered but not found in folder: {extra}")

    finally:
      self.patcher.start()


if __name__ == "__main__":
  absltest.main()
