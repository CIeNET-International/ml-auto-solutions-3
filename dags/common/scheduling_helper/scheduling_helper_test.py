# Copyright 2025 Google LLC
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

"""Unit tests for scheduling_helper.py."""

import datetime as dt
from unittest.mock import patch
from absl.testing import absltest, parameterized

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
    mock_registered.items.return_value = self.mock_registry.items()
    with self.assertRaises(scheduling_helper.UnregisteredDagError):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("ghost_dag")

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_24hours_window_cumulative(self, mock_registered):
    long_dags = {f"d{i}": dt.timedelta(hours=5) for i in range(6)}
    mock_registered.items.return_value = {"c1": long_dags}.items()
    with self.assertRaises(scheduling_helper.ScheduleWindowError):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("d5")

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_24hours_window_single_dag(self, mock_registered):
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
    mock_registered.items.return_value = self.mock_registry.items()
    cron_pattern = r"^([0-5]?\d) ([0-1]?\d|2[0-3]) \* \* (\*|1-5|0,6)$"
    res = scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_1")
    self.assertRegex(res, cron_pattern)


if __name__ == "__main__":
  absltest.main()
