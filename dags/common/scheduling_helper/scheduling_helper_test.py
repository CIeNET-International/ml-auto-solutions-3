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
            "dag_1": dt.timedelta(minutes=12),  # Start: 08:00
            "dag_2": dt.timedelta(
                minutes=33
            ),  # Start: 08:00 + 12m + 15m = 08:27
            "dag_3": dt.timedelta(
                seconds=45
            ),  # Start: 08:27 + 33m + 15m = 09:15
            "dag_4": dt.timedelta(
                minutes=20
            ),  # Start: 09:15 + 45s + 15m = 09:35
            "dag_5": dt.timedelta(
                minutes=10
            ),  # Start: 09:35 + 20m + 15m = 09:70 (10:10)
            "dag_6": dt.timedelta(
                minutes=5
            ),  # Start: 10:10 + 10m + 15m = 10:35
        },
        "cluster_b": {
            "dag_x": dt.timedelta(minutes=5),
            "dag_y": dt.timedelta(minutes=10),
        },
    }


class TestSchedulingLogic(TestSchedulingHelperBase):
  """Validates the cron string generation and stacking logic."""

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_alignment_with_anchor(self, mock_registered):
    mock_registered.items.return_value = self.mock_registry.items()
    # The first DAG should always align with DEFAULT_ANCHOR (08:00 UTC)
    schedule = scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_1")
    self.assertEqual(schedule, "0 8 * * *")

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_complex_calculation(self, mock_registered):
    mock_registered.items.return_value = self.mock_registry.items()
    # Testing the 'stacking' effect with non-standard durations
    schedule = scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_2")
    self.assertEqual(schedule, "27 8 * * *")

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
        "dag_1", day_of_week=day_enum
    )
    self.assertTrue(schedule.endswith(expected_suffix))


class TestErrorHandling(TestSchedulingHelperBase):
  """Validates boundary conditions and registration checks."""

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_unregistered_dag(self, mock_registered):
    mock_registered.items.return_value = self.mock_registry.items()
    with self.assertRaisesRegex(ValueError, "is not registered"):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("ghost_dag")

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_24hours_window_single_dag(self, mock_registered):
    mock_registered.items.return_value = {
        "c1": {"huge_dag": dt.timedelta(hours=25)}
    }.items()
    with self.assertRaisesRegex(ValueError, "Schedule exceeds 24h window"):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("huge_dag")

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_24hours_window_cumulative(self, mock_registered):
    # 5 DAGs @ 5 hours each = 25 hours. The 6th DAG should trigger the error.
    long_dags = {f"d{i}": dt.timedelta(hours=5) for i in range(6)}
    mock_registered.items.return_value = {"c1": long_dags}.items()
    with self.assertRaisesRegex(ValueError, "Schedule exceeds 24h window"):
      scheduling_helper.SchedulingHelper.arrange_schedule_time("d5")


class TestFormatConsistency(TestSchedulingHelperBase):
  """Ensures output is valid and deterministic."""

  @patch("dags.common.scheduling_helper.scheduling_helper.REGISTERED_DAGS")
  def test_output_is_valid_cron(self, mock_registered):
    mock_registered.items.return_value = self.mock_registry.items()
    cron_pattern = r"^\d{1,2} \d{1,2} \* \* (\*|1-5|0,6)$"
    res = scheduling_helper.SchedulingHelper.arrange_schedule_time("dag_1")
    self.assertRegex(res, cron_pattern)


if __name__ == "__main__":
  absltest.main()
