# Copyright 2026 Google LLC
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

"""Environments and limits for the daily monitor."""

import dataclasses

# Every environment reports to the same Chat space, "[Daily Monitoring]"
# (spaces/AAQA57164nY), through this webhook.
WEBHOOK_SECRET = "orbax-daily-monitor-chat-webhook-erniechang-test"

MONITORED_FOLDER = "dags/orbax"
# The retired monitor DAG. Skipped if an environment still has it deployed.
SELF_DAG_ID = "orbax_daily_monitor"

# Collection limits.
HISTORY_RUNS = 7
MAX_RUNS_IN_WINDOW = 100
AIRFLOW_ERROR_LINES_PER_TASK = 50
POD_LINES_PER_WORKLOAD = 300
POD_TAIL_LINES = 100
TAIL_POD_PATTERN = "slice-job-0-0"


@dataclasses.dataclass(frozen=True)
class Environment:
  """One Composer environment the monitor reports on."""

  name: str  # Short name, the ENV value.
  composer: str  # Composer environment name.
  project: str  # Project of Composer, Vertex AI and the webhook secret.
  airflow_url: str  # Airflow web server URL, no trailing slash.
  # Where the JSON files go. Always the erniechang-test Composer bucket: the
  # monitor never writes to the Composer environment it reads.
  bucket: str  # Bucket name, without gs://.
  prefix: str  # Folder for the JSON files inside the bucket.
  title: str  # First line of the Chat report.
  # Secret Manager secret with the Chat webhook URL; the URL decides the space.
  webhook_secret: str = WEBHOOK_SECRET


WRITE_BUCKET = "us-east1-erniechang-test-72c040a8-bucket"

ENVIRONMENTS = {
    "test": Environment(
        name="test",
        composer="erniechang-test",
        project="cloud-ml-auto-solutions",
        airflow_url=(
            "https://3776ea3159a646e4ae176f7424dea099-dot-us-east1"
            ".composer.googleusercontent.com"
        ),
        bucket=WRITE_BUCKET,
        prefix="data/dag_monitor/orbax",
        title="Orbax DAG daily report",
    ),
    # Reads the prod Composer (Airflow REST API GETs and Cloud Logging only);
    # writes to the erniechang-test bucket. Never posts: posting only works in
    # dag_monitor.AUTO_POST_ENVS (erniechang-test).
    "prod": Environment(
        name="prod",
        composer="ml-automation-solutions",
        project="cloud-ml-auto-solutions",
        airflow_url=(
            "https://efdb2a1d6c2c435b8b7de77690c286a9-dot-us-central1"
            ".composer.googleusercontent.com"
        ),
        bucket=WRITE_BUCKET,
        prefix="data/dag_monitor/orbax-prod",
        title="Orbax DAG daily report",
    ),
}
