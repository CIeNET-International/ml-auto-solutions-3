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

  def test_registration_check(self):
    """
    Ensures every DAG file in the folder is registered in the helper.
    """
    dagbag = DagBag(dag_folder="dags/tpu_observability", include_examples=False)
    actual_ids = set(dagbag.dag_ids)
    registered_ids = set()
    for dags_dict in scheduling_helper.REGISTERED_DAGS.values():
      registered_ids.update(dags_dict.keys())
    missing = actual_ids - registered_ids
    self.assertEmpty(
        missing,
        msg=(
            f"The following DAGs exist in dags/tpu_observability but are NOT "
            f"registered in scheduling_helper.py: {missing}. "
            "Please add them to REGISTERED_DAGS to ensure they are scheduled."
        ),
    )
    extra = registered_ids - actual_ids
    if extra:
      print(f"\n[WARNING]: DAGs registered but not found in folder: {extra}")


if __name__ == "__main__":
  absltest.main()
