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

import datetime as dt
from absl.testing import absltest
from absl.testing import parameterized
from airflow.models import DagBag

from dags.common.scheduling_helper import scheduling_helper


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
