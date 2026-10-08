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

"""Orbax daily monitor that runs outside Airflow (Cloud Run Job).

Reads the Orbax DAG runs of one Composer environment through the Airflow REST
API and Cloud Logging, builds the same digest as the orbax_daily_monitor DAG
with xlml.utils.dag_monitor, triages it with Gemini, and writes the digest,
triage and Chat draft to the Composer bucket. Entry point: main.py.
"""
