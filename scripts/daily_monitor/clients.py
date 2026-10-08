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

"""REST clients: Airflow API, Cloud Logging, GCS, Secret Manager.

All calls go through one authorized session (Application Default
Credentials: the Cloud Run job's service account, or your
`gcloud auth application-default login` locally). Only the Airflow API and
Cloud Logging are read; GCS is the only place this code writes.
"""

import base64
import json
import os
import time
from typing import Any, Optional
from urllib import parse

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
RETRY_STATUS = (429, 500, 502, 503, 504)
MIN_QUERY_GAP_S = 1.1  # Stay under 60 Cloud Logging read requests/min.
LOGGING_PAGE_MAX = 1000


def authorized_session() -> Any:
  """A requests session that adds an OAuth token to every call."""
  # pylint: disable=import-outside-toplevel
  import google.auth
  from google.auth.transport.requests import AuthorizedSession

  credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
  return AuthorizedSession(credentials)


def _request(session: Any, method: str, url: str, **kwargs) -> Any:
  """One HTTP call with retries on quota and server errors."""
  for attempt in range(4):
    response = session.request(method, url, timeout=60, **kwargs)
    if response.status_code in RETRY_STATUS and attempt < 3:
      time.sleep(
          15 * (attempt + 1) if response.status_code == 429 else 2**attempt
      )
      continue
    return response
  return response


class AirflowClient:
  """GET-only Airflow 2 stable REST API client."""

  def __init__(self, session: Any, airflow_url: str):
    self._session = session
    self._api = airflow_url.rstrip("/") + "/api/v1"
    self.calls = 0

  def get(self, path: str, **params) -> dict[str, Any]:
    query = parse.urlencode({k: v for k, v in params.items() if v is not None})
    self.calls += 1
    response = _request(
        self._session,
        "GET",
        self._api + path + (f"?{query}" if query else ""),
        headers={"Accept": "application/json"},
    )
    if response.status_code != 200:
      raise RuntimeError(
          f"Airflow API GET {path}: HTTP {response.status_code}:"
          f" {response.text[:300]}"
      )
    return response.json()


class LogClient:
  """Cloud Logging entries:list, newest entries first, rate limited."""

  URL = "https://logging.googleapis.com/v2/entries:list"

  def __init__(self, session: Any):
    self._session = session
    self._last = 0.0
    self.stats = {"log_queries": 0, "log_lines": 0}

  def read(self, project: str, log_filter: str, limit: int) -> list[dict]:
    """Returns up to `limit` of the newest matching entries, oldest first."""
    entries, token = [], None
    while len(entries) < limit:
      wait = MIN_QUERY_GAP_S - (time.monotonic() - self._last)
      if wait > 0:
        time.sleep(wait)
      self._last = time.monotonic()
      self.stats["log_queries"] += 1
      body = {
          "resourceNames": [f"projects/{project}"],
          "filter": log_filter,
          "orderBy": "timestamp desc",
          "pageSize": min(limit - len(entries), LOGGING_PAGE_MAX),
      }
      if token:
        body["pageToken"] = token
      response = _request(self._session, "POST", self.URL, json=body)
      if response.status_code != 200:
        raise RuntimeError(
            f"Cloud Logging: HTTP {response.status_code}:"
            f" {response.text[:300]}"
        )
      reply = response.json()
      entries += reply.get("entries") or []
      token = reply.get("nextPageToken")
      if not token:
        break
    entries = entries[:limit]
    self.stats["log_lines"] += len(entries)
    return entries[::-1]


def entry_text(entry: dict[str, Any]) -> str:
  """The message of a Cloud Logging entry (text or JSON payload)."""
  if entry.get("textPayload"):
    return str(entry["textPayload"]).rstrip()
  payload = entry.get("jsonPayload") or {}
  if isinstance(payload, dict) and payload.get("message"):
    return str(payload["message"]).rstrip()
  return json.dumps(payload)[:500] if payload else ""


def entry_pod(entry: dict[str, Any]) -> str:
  return ((entry.get("resource") or {}).get("labels") or {}).get("pod_name", "")


class GcsStore:
  """JSON files under gs://<bucket>/<prefix>/."""

  def __init__(self, session: Any, bucket: str, prefix: str):
    self._session = session
    self._bucket = bucket
    self._prefix = prefix.strip("/")
    self.location = f"gs://{bucket}/{self._prefix}"

  def _object(self, name: str) -> str:
    return parse.quote(f"{self._prefix}/{name}", safe="")

  def exists(self, name: str) -> bool:
    response = _request(
        self._session,
        "GET",
        "https://storage.googleapis.com/storage/v1/b/"
        f"{self._bucket}/o/{self._object(name)}",
    )
    if response.status_code == 404:
      return False
    if response.status_code != 200:
      raise RuntimeError(f"GCS stat {name}: HTTP {response.status_code}")
    return True

  def write_json(self, name: str, value: Any) -> str:
    response = _request(
        self._session,
        "POST",
        f"https://storage.googleapis.com/upload/storage/v1/b/{self._bucket}/o",
        params={"uploadType": "media", "name": f"{self._prefix}/{name}"},
        data=(json.dumps(value, indent=2) + "\n").encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    if response.status_code != 200:
      raise RuntimeError(
          f"GCS write {name}: HTTP {response.status_code}:"
          f" {response.text[:300]}"
      )
    return f"{self.location}/{name}"


class LocalStore:
  """JSON files in a local folder (OUTPUT=local:<dir>), for local runs."""

  def __init__(self, folder: str):
    self.location = os.path.abspath(os.path.expanduser(folder))

  def exists(self, name: str) -> bool:
    return os.path.exists(os.path.join(self.location, name))

  def write_json(self, name: str, value: Any) -> str:
    os.makedirs(self.location, exist_ok=True)
    path = os.path.join(self.location, name)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(value, f, indent=2)
      f.write("\n")
    return path


def read_secret(session: Any, project: str, secret: str) -> str:
  """The latest version of a Secret Manager secret. Never log it."""
  response = _request(
      session,
      "GET",
      "https://secretmanager.googleapis.com/v1/"
      f"projects/{project}/secrets/{secret}/versions/latest:access",
  )
  if response.status_code != 200:
    raise RuntimeError(f"Secret {secret}: HTTP {response.status_code}")
  data = response.json()["payload"]["data"]
  return base64.b64decode(data).decode("utf-8").strip()


def open_store(session: Any, output: Optional[str], bucket: str, prefix: str):
  """OUTPUT=local:<dir> -> LocalStore; else GCS (OUTPUT=gs://b/p overrides)."""
  if output and output.startswith("local:"):
    return LocalStore(output[len("local:") :])
  if output and output.startswith("gs://"):
    bucket, _, prefix = output[len("gs://") :].partition("/")
  return GcsStore(session, bucket, prefix)
