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

"""Entry point of the daily monitor Cloud Run Job.

  python -m scripts.daily_monitor.main

Settings come from environment variables (set on the job, or overridden with
`gcloud run jobs execute ... --update-env-vars`):
  ENV            Environment in config.ENVIRONMENTS (default: test).
  REPORT_DATE    UTC day YYYY-MM-DD (default: yesterday, UTC).
  GEMINI_SEARCH  true: let Gemini run read-only log searches (default false).
  POST           true: post the draft through the Chat webhook (default
                 false). Only works in a dag_monitor.AUTO_POST_ENVS
                 environment.
  FORCE_POST     true: post even if <date>.posted.json exists.
  OUTPUT         gs://bucket/prefix or local:<dir> (default: the
                 environment's Composer bucket folder).

Writes <date>.json, <date>.triage.json, <date>.draft.json, and
<date>.search.json with GEMINI_SEARCH.
"""

import dataclasses
import datetime
import logging
import os
import sys
from typing import Any, Callable, Mapping, Optional

from scripts.daily_monitor import chat
from scripts.daily_monitor import clients
from scripts.daily_monitor import collector
from scripts.daily_monitor import config
from scripts.daily_monitor import gemini
from xlml.utils import dag_monitor

TRUE = ("1", "true", "yes")


@dataclasses.dataclass(frozen=True)
class Settings:
  env: config.Environment
  report_date: str
  search: bool
  post: bool
  force_post: bool
  output: Optional[str]


def settings_from(environ: Mapping[str, str]) -> Settings:
  """Parses and checks the environment variables."""
  name = environ.get("ENV", "test")
  if name not in config.ENVIRONMENTS:
    raise ValueError(
        f"ENV={name!r}: expected one of {list(config.ENVIRONMENTS)}"
    )
  report_date = (
      environ.get("REPORT_DATE")
      or (
          datetime.datetime.now(datetime.timezone.utc).date()
          - datetime.timedelta(days=1)
      ).isoformat()
  )
  datetime.date.fromisoformat(report_date)  # Raises on a bad date.

  def flag(key: str) -> bool:
    return environ.get(key, "").strip().lower() in TRUE

  return Settings(
      env=config.ENVIRONMENTS[name],
      report_date=report_date,
      search=flag("GEMINI_SEARCH"),
      post=flag("POST"),
      force_post=flag("FORCE_POST"),
      output=environ.get("OUTPUT") or None,
  )


@dataclasses.dataclass
class Deps:
  """Everything that talks to the outside, so tests can pass fakes."""

  airflow: clients.AirflowClient
  logs: clients.LogClient
  store: Any  # clients.GcsStore or clients.LocalStore.
  generate: gemini.Generate
  read_webhook_url: Callable[[], str]
  http_post: Callable[..., Any]


def run(settings: Settings, deps: Deps) -> dict[str, Any]:
  """Collects, triages, renders, writes and (optionally) posts one day.

  Returns:
    {file kind: path} of what was written.
  """
  env, day, store = settings.env, settings.report_date, deps.store
  logging.info(
      "Report %s for %s (%s), search=%s, post=%s",
      day,
      env.composer,
      store.location,
      settings.search,
      settings.post,
  )
  written = {}
  digest = collector.Collector(env, deps.airflow, deps.logs).collect(day)
  written["digest"] = store.write_json(f"{day}.json", digest)
  logging.info(
      "Wrote %s: summary=%s, errors=%d",
      written["digest"],
      digest["summary"],
      len(digest.get("collector_errors") or []),
  )
  triage, search = gemini.triage(
      digest, deps.generate, deps.logs, settings.search
  )
  if search is not None:
    written["search"] = store.write_json(f"{day}.search.json", search)
  written["triage"] = store.write_json(f"{day}.triage.json", triage)
  logging.info(
      "Wrote %s: source=%s, items=%d, dropped=%d",
      written["triage"],
      triage["source"],
      len(triage["items"]),
      triage["dropped"],
  )
  messages = dag_monitor.render_report(digest, triage, env.title)
  written["draft"] = store.write_json(
      f"{day}.draft.json", {"messages": messages}
  )
  for i, message in enumerate(messages, 1):
    logging.info("Draft message %d (%d chars):\n%s", i, len(message), message)
  if not dag_monitor.auto_post_enabled(env.composer, {"post": settings.post}):
    logging.info("Posting is off (POST=%s).", settings.post)
    return written
  marker = f"{day}.posted.json"
  if store.exists(marker) and not settings.force_post:
    logging.info("Already posted: %s/%s", store.location, marker)
    return written
  posted = chat.post_messages(
      deps.read_webhook_url(), messages, day, deps.http_post
  )
  written["posted"] = store.write_json(
      marker,
      {
          "messages": posted,
          "posted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "draft": written["draft"],
      },
  )
  logging.info("Posted %d message(s): %s", len(posted), posted)
  return written


def real_deps(settings: Settings) -> Deps:
  """Deps that call the real Google Cloud APIs as the default credentials."""
  import requests  # pylint: disable=import-outside-toplevel

  env = settings.env
  session = clients.authorized_session()
  return Deps(
      airflow=clients.AirflowClient(session, env.airflow_url),
      logs=clients.LogClient(session),
      store=clients.open_store(
          session, settings.output, env.bucket, env.prefix
      ),
      generate=gemini.vertex_generator(session, env.project),
      read_webhook_url=lambda: clients.read_secret(
          session,
          env.project,
          env.webhook_secret,
      ),
      http_post=requests.post,
  )


def main() -> int:
  logging.basicConfig(
      level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
  )
  settings = settings_from(os.environ)
  written = run(settings, real_deps(settings))
  logging.info("Done: %s", written)
  return 0


if __name__ == "__main__":
  sys.exit(main())
