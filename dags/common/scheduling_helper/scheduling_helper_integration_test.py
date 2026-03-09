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

"""
Integration test to ensure all DAGs in the folder are registered in the helper.
"""


from absl.testing import absltest
from airflow.models import DagBag

from dags.common.scheduling_helper import scheduling_helper


class TestSchedulingHelperIntegration(absltest.TestCase):
  """Integration tests for the scheduling helper module."""

  CHECKED_DAG_FOLDERS = [
      "dags/tpu_observability",
  ]

  def test_registration_check(self):
    """
    Ensures every DAG file in the folder is registered in the helper.
    """
    actual_ids = set()
    for folder in self.CHECKED_DAG_FOLDERS:
      dagbag = DagBag(dag_folder=folder, include_examples=False)
      actual_ids.update(dagbag.dag_ids)

    registered_ids = set()
    for dags_dict in scheduling_helper.REGISTERED_DAGS.values():
      registered_ids.update(dags_dict.keys())

    missing = actual_ids - registered_ids
    self.assertEmpty(
        missing,
        msg=(
            f"The following DAGs exist in {self.CHECKED_DAG_FOLDERS} "
            f"but are not registered in scheduling_helper.py: {missing}."
        ),
    )

    extra = registered_ids - actual_ids
    if extra:
      raise scheduling_helper.StaleRegistrationError(
          f"The following DAG IDs are registered in scheduling_helper.py but "
          f"were not found in {self.CHECKED_DAG_FOLDERS}: {extra}. "
          "Please remove these stale entries."
      )


if __name__ == "__main__":
  absltest.main()
