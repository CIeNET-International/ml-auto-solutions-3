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

from airflow.models import DagBag

from dags.common.scheduling_helper import scheduling_helper


def run_registration_check(dag_folder: str) -> bool:
  dagbag = DagBag(dag_folder=dag_folder, include_examples=False)

  registered_ids = set()
  for dags_dict in scheduling_helper.REGISTERED_DAGS.values():
    registered_ids.update(dags_dict.keys())

  actual_ids = set(dagbag.dag_ids)

  missing = actual_ids - registered_ids
  extra = registered_ids - actual_ids

  success = True
  if missing:
    print(f"error: missing DAGs in REGISTERED_DAGS: {missing}")
    success = False

  if extra:
    print(f"warning: DAGs in REGISTERED_DAGS but not found: {extra}")

  if success:
    print("success: all DAGs are properly registered.")

  return success


if __name__ == "__main__":
  result = run_registration_check("dags/tpu_observability")
