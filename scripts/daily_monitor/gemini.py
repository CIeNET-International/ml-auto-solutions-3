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

"""Gemini triage on Vertex AI, moved from the orbax_daily_monitor DAG.

Gemini only returns triage JSON. dag_monitor.validate_triage checks every
quoted evidence line against the collected logs, and code does all writes
and posting.
"""

import logging
import time
from typing import Any, Callable, Optional

from scripts.daily_monitor import clients
from xlml.utils import dag_monitor

GEMINI_MODEL = "gemini-3.5-flash"
# Gemini 3.x is served only from the global Vertex endpoint, not us-east1.
GEMINI_LOCATION = "global"
GEMINI_TIMEOUT_S = 120
GEMINI_ATTEMPTS = 3  # The first call plus 2 retries.
# Total time for all search conversations of a run. DAGs left when it runs
# out use the single-call triage instead.
SEARCH_BUDGET_S = 20 * 60

Generate = Callable[[dict[str, Any]], dict[str, Any]]


def vertex_generator(session: Any, project: str) -> Generate:
  """Returns a function that POSTs one generateContent request body."""
  url = (
      "https://aiplatform.googleapis.com/v1/"
      f"projects/{project}/locations/{GEMINI_LOCATION}/"
      f"publishers/google/models/{GEMINI_MODEL}:generateContent"
  )

  def generate(body: dict[str, Any]) -> dict[str, Any]:
    error = None
    for attempt in range(GEMINI_ATTEMPTS):
      if attempt:
        time.sleep(5 * attempt)
      try:
        response = session.post(url, json=body, timeout=GEMINI_TIMEOUT_S)
      except Exception as e:  # pylint: disable=broad-except
        error = repr(e)[:300]
        continue
      if response.status_code != 200:
        error = f"HTTP {response.status_code}: {response.text[:300]}"
        continue
      return response.json()
    raise RuntimeError(f"Vertex AI {GEMINI_MODEL} failed: {error}")

  return generate


def single_call(generate: Generate, prompt: str) -> str:
  """One single-call triage request; returns the reply text."""
  response = generate(
      {
          "contents": [{"role": "user", "parts": [{"text": prompt}]}],
          "generationConfig": {
              "temperature": 0,
              "responseMimeType": "application/json",
              "responseSchema": dag_monitor.TRIAGE_SCHEMA,
          },
      }
  )
  candidates = response.get("candidates") or []
  parts = (
      (candidates[0].get("content") or {}).get("parts") if candidates else []
  )
  return "".join(p.get("text", "") for p in parts or [] if not p.get("thought"))


def search_triage(
    digest: dict[str, Any], generate: Generate, logs: clients.LogClient
) -> tuple[list[Any], dict[str, list[str]], dict[str, Any], list[str]]:
  """One search_logs conversation per latest failure.

  Returns:
    (items, extra evidence texts per DAG, per-DAG search record, DAG IDs that
    still need the single-call triage).
  """

  def read_logs(project: str, log_filter: str, limit: int) -> list[str]:
    return [
        clients.entry_text(e) for e in logs.read(project, log_filter, limit)
    ]

  items, extra, record, pending = [], {}, {}, []
  deadline = time.monotonic() + SEARCH_BUDGET_S
  for dag, run in dag_monitor.latest_failures(digest):
    dag_id = dag["dag_id"]
    if time.monotonic() > deadline:
      record[dag_id] = {"mode": "single", "error": "search time budget used"}
      pending.append(dag_id)
      continue
    try:
      result = dag_monitor.run_search_triage(dag, run, generate, read_logs)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Search triage failed for %s", dag_id)
      record[dag_id] = {"mode": "single", "error": repr(e)[:300]}
      pending.append(dag_id)
      continue
    items += result["reply"]["items"]
    extra[dag_id] = result["texts"]
    record[dag_id] = {
        "mode": "search",
        "turns": result["turns"],
        "queries": result["queries"],
    }
    logging.info(
        "Search triage for %s: %d turn(s), %d search(es)",
        dag_id,
        result["turns"],
        len(result["queries"]),
    )
  return items, extra, record, pending


def triage(
    digest: dict[str, Any],
    generate: Generate,
    logs: clients.LogClient,
    search: bool,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
  """Triages each latest failure; falls back to the digest on errors.

  Returns:
    (triage, search file content or None when search was off).
  """
  prompt = dag_monitor.build_triage_prompt(digest)
  error = None
  items, extra, record = [], {}, None
  if prompt is not None and search:
    items, extra, record, pending = search_triage(digest, generate, logs)
    # Only DAGs whose search failed still need the single call.
    prompt = dag_monitor.build_triage_prompt(digest, pending)
  if prompt is None and not items:
    result = dag_monitor.fallback_triage(digest)  # No failure to triage.
  else:
    if prompt is not None:
      try:
        reply = dag_monitor.parse_reply(single_call(generate, prompt))
        done = {i.get("dag_id") for i in items if isinstance(i, dict)}
        items += [
            i
            for i in reply.get("items") or []
            if isinstance(i, dict) and i.get("dag_id") not in done
        ]
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Gemini triage failed; using the digest only")
        error = repr(e)[:500]
    result = dag_monitor.validate_triage({"items": items}, digest, extra)
  result.update({"model": GEMINI_MODEL, "error": error})
  if record is None:
    return result, None
  result["search"] = record
  return result, {"queries": record, "lines": extra}
