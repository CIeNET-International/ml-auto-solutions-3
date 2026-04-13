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

"""Scheduling helper package for DAG scheduling and timeout management.

This package provides:
- SchedulingHelper: Manages DAG scheduling across different clusters.
- TaskGroupWithTimeout: A TaskGroup that enforces a shared deadline timeout.
- get_dag_timeout: Returns the registered timeout for a specific DAG.

Usage:
  from dags.common.scheduling_helper import (
      SchedulingHelper,
      TaskGroupWithTimeout,
      get_dag_timeout,
  )
"""

from dags.common.scheduling_helper.scheduling_helper import (
    DagIdToTimeout,
    DayOfWeek,
    REGISTERED_DAGS,
    ScheduleWindowError,
    SchedulingError,
    SchedulingHelper,
    TPU_OBS_MOCK_CLUSTER,
    UnregisteredDagError,
    get_dag_timeout,
)
from dags.common.scheduling_helper.timeout import TaskGroupWithTimeout
