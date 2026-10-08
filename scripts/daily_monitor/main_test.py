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

"""Tests for the daily monitor job, with fake Airflow, Logging and Vertex.

Run from the repo root:
  python3 -m unittest discover scripts/daily_monitor "*_test.py" -t .
"""

import base64
import json
import os
import tempfile
import unittest
from urllib import parse

from scripts.daily_monitor import chat
from scripts.daily_monitor import clients
from scripts.daily_monitor import config
from scripts.daily_monitor import main
from xlml.utils import dag_monitor

DAY = "2026-10-08"
DAG_ID = "maxtext_regular_restore"
RUN_ID = "scheduled__2026-10-08T01:00:00+00:00"
WORKLOAD = "mtx-restore-abc123"
EXCEPTION = "AssertionError: restore event not found for step 39"
CLUSTER = "TPU_V5P_BODABORG_NAP_CLUSTER"
ENV = config.ENVIRONMENTS["test"]


def _dag(dag_id, fileloc, tags=(), token=None):
  return {
      "dag_id": dag_id,
      "fileloc": fileloc,
      "file_token": token or dag_id,
      "owners": ["multipod", "airflow"],
      "tags": [{"name": t} for t in tags],
      "schedule_interval": {"__type": "CronExpression", "value": "0 1 * * *"},
  }


class FakeAirflow:
  """Routes GET paths to canned Airflow REST API replies."""

  def __init__(self):
    self.calls = 0
    run_q = parse.quote(RUN_ID, safe="")
    base = f"/dags/{DAG_ID}/dagRuns"
    failed_run = {
        "dag_run_id": RUN_ID,
        "state": "failed",
        "start_date": "2026-10-08T01:00:05+00:00",
        "end_date": "2026-10-08T02:10:00+00:00",
    }
    self.routes = {
        "/dags": {
            "dags": [
                _dag(
                    DAG_ID, "/home/airflow/gcs/dags/orbax/restore.py", ["orbax"]
                ),
                _dag(
                    config.SELF_DAG_ID,
                    "/home/airflow/gcs/dags/orbax/orbax_daily_monitor.py",
                ),
                _dag("other_dag", "/home/airflow/gcs/dags/other/x.py"),
            ]
        },
        f"/dagSources/{DAG_ID}": {
            "content": f"cluster = GkeClusters.{CLUSTER}\n"
        },
        base: {"dag_runs": [failed_run]},
        f"{base}/{run_q}/taskInstances": {
            "task_instances": [
                {
                    "task_id": "restore.generate_workload_id",
                    "state": "success",
                    "start_date": "2026-10-08T01:00:10+00:00",
                },
                {
                    "task_id": "restore.validate_restore",
                    "state": "failed",
                    "try_number": 1,
                    "start_date": "2026-10-08T02:00:00+00:00",
                    "end_date": "2026-10-08T02:05:00+00:00",
                },
            ]
        },
        f"{base}/{run_q}/taskInstances/restore.generate_workload_id"
        "/xcomEntries/return_value": {"value": f"'{WORKLOAD}'"},
    }
    self.paths = []

  def get(self, path, **params):
    del params
    self.calls += 1
    self.paths.append(path)
    if path not in self.routes:
      raise RuntimeError(f"unexpected path {path}")
    return self.routes[path]


class FakeLogs:
  """Returns canned entries by the kind of filter."""

  def __init__(self):
    self.stats = {"log_queries": 0, "log_lines": 0}
    self.filters = []
    events_extra = dag_monitor.checkpoint_events_extra(config.TAIL_POD_PATTERN)
    self.kinds = [
        ("severity>=ERROR", [EXCEPTION]),
        ("airflow-worker", ["task started", EXCEPTION]),
        (events_extra, ["Saved checkpoint at step 39", "Restored step 39"]),
        (
            dag_monitor.message_matches(dag_monitor.POD_ERROR_PATTERN),
            ["ERROR: restore metric missing"],
        ),
        (config.TAIL_POD_PATTERN, ["step 40 done"]),
    ]

  def read(self, project, log_filter, limit):
    del project, limit
    self.stats["log_queries"] += 1
    self.filters.append(log_filter)
    for needle, lines in self.kinds:
      if needle in log_filter:
        return [
            {
                "textPayload": line,
                "timestamp": "2026-10-08T02:00:00Z",
                "resource": {"labels": {"pod_name": "slice-job-0-0-xyz"}},
            }
            for line in lines
        ]
    return []


def _reply(evidence=EXCEPTION):
  return {
      "dag_id": DAG_ID,
      "run_id": RUN_ID,
      "category": "checkpoint_validation",
      "summary": "The restore check failed although step 39 was restored.",
      "evidence": [evidence],
  }


class FakeGemini:
  """Single call: a JSON reply. Search: submit_triage on the first turn."""

  def __init__(self, fail=False):
    self.bodies = []
    self.fail = fail

  def __call__(self, body):
    self.bodies.append(body)
    if self.fail:
      raise RuntimeError("Vertex AI down")
    if "tools" in body:
      part = {"functionCall": {"name": "submit_triage", "args": _reply()}}
    else:
      part = {"text": json.dumps({"items": [_reply()]})}
    return {"candidates": [{"content": {"role": "model", "parts": [part]}}]}


class FakeResponse:

  def __init__(self, status=200, body=None):
    self.status_code = status
    self._body = body or {}
    self.text = json.dumps(self._body)

  def json(self):
    return self._body


class JobTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.store = clients.LocalStore(self.tmp.name)
    self.posts = []

  def _deps(self, gemini=None):
    def http_post(url, **kwargs):
      self.posts.append((url, kwargs))
      return FakeResponse(200, {"name": "m/1", "thread": {"name": "t/1"}})

    return main.Deps(
        airflow=FakeAirflow(),
        logs=FakeLogs(),
        store=self.store,
        generate=gemini or FakeGemini(),
        read_webhook_url=lambda: "https://chat.example/webhook?key=SECRET",
        http_post=http_post,
    )

  def _settings(self, **env):
    return main.settings_from({"REPORT_DATE": DAY, **env})

  def _load(self, name):
    with open(os.path.join(self.tmp.name, name), encoding="utf-8") as f:
      return json.load(f)

  def test_writes_digest_triage_and_draft(self):
    written = main.run(self._settings(), self._deps())
    self.assertEqual(set(written), {"digest", "triage", "draft"})
    digest = self._load(f"{DAY}.json")
    self.assertEqual(digest["environment"], "erniechang-test")
    self.assertEqual([d["dag_id"] for d in digest["dags"]], [DAG_ID])
    dag = digest["dags"][0]
    self.assertEqual(dag["tags"], ["orbax"])
    self.assertEqual(dag["owners"], ["multipod"])
    self.assertEqual(dag["file"], "dags/orbax/restore.py")
    run = dag["runs"][0]
    self.assertEqual(run["failed_tasks"][0]["exception"], EXCEPTION)
    self.assertEqual(run["failed_tasks"][0]["workload"], WORKLOAD)
    self.assertIn(EXCEPTION, run["failed_tasks"][0]["log_tail"])
    self.assertEqual(
        run["probes"]["checkpoint_events"][0]["lines"],
        ["Saved checkpoint at step 39", "Restored step 39"],
    )
    self.assertEqual(run["pod_tail"]["lines"], ["step 40 done"])
    triage = self._load(f"{DAY}.triage.json")
    self.assertEqual(triage["source"], "gemini")
    self.assertEqual(triage["dropped"], 0)
    self.assertEqual(triage["model"], "gemini-3.5-flash")
    messages = self._load(f"{DAY}.draft.json")["messages"]
    self.assertTrue(messages)
    self.assertIn(ENV.title, messages[0])
    self.assertEqual(self.posts, [])

  def test_made_up_evidence_is_dropped(self):
    def generate(body):
      del body
      text = json.dumps({"items": [_reply("restore event logged at step 39")]})
      return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    main.run(self._settings(), self._deps(generate))
    triage = self._load(f"{DAY}.triage.json")
    self.assertEqual(triage["dropped"], 1)
    self.assertNotEqual(triage["source"], "gemini")

  def test_gemini_failure_falls_back(self):
    main.run(self._settings(), self._deps(FakeGemini(fail=True)))
    triage = self._load(f"{DAY}.triage.json")
    self.assertIn("Vertex AI down", triage["error"])
    self.assertTrue(self._load(f"{DAY}.draft.json")["messages"])

  def test_search_writes_search_file(self):
    gemini = FakeGemini()
    written = main.run(self._settings(GEMINI_SEARCH="true"), self._deps(gemini))
    self.assertIn("search", written)
    search = self._load(f"{DAY}.search.json")
    self.assertEqual(search["queries"][DAG_ID]["mode"], "search")
    self.assertEqual(search["queries"][DAG_ID]["turns"], 1)
    self.assertEqual(len(gemini.bodies), 1)  # No single call needed.
    self.assertEqual(self._load(f"{DAY}.triage.json")["source"], "gemini")

  def test_post_once_then_skip_unless_forced(self):
    main.run(self._settings(POST="true"), self._deps())
    sent = len(self.posts)
    self.assertGreater(sent, 0)
    url, kwargs = self.posts[0]
    self.assertIn("key=SECRET", url)
    self.assertEqual(
        kwargs["params"]["threadKey"], f"orbax-daily-monitor-{DAY}"
    )
    self.assertEqual(
        self._load(f"{DAY}.posted.json")["messages"][0]["name"], "m/1"
    )
    main.run(self._settings(POST="true"), self._deps())
    self.assertEqual(len(self.posts), sent)
    main.run(self._settings(POST="true", FORCE_POST="1"), self._deps())
    self.assertEqual(len(self.posts), 2 * sent)

  def test_prod_never_posts(self):
    settings = self._settings(ENV="prod", POST="true")
    main.run(settings, self._deps())
    self.assertEqual(self.posts, [])
    self.assertFalse(self.store.exists(f"{DAY}.posted.json"))
    digest = self._load(f"{DAY}.json")
    self.assertEqual(digest["environment"], "ml-automation-solutions")


class SettingsTest(unittest.TestCase):

  def test_defaults(self):
    settings = main.settings_from({})
    self.assertEqual(settings.env.composer, "erniechang-test")
    self.assertFalse(settings.search or settings.post or settings.force_post)
    self.assertRegex(settings.report_date, r"^\d{4}-\d{2}-\d{2}$")

  def test_bad_values(self):
    with self.assertRaises(ValueError):
      main.settings_from({"ENV": "staging"})
    with self.assertRaises(ValueError):
      main.settings_from({"REPORT_DATE": "2026-13-01"})


class ProdTest(unittest.TestCase):

  def test_reads_prod_and_writes_to_the_test_bucket(self):
    prod = main.settings_from({"ENV": "prod"}).env
    self.assertEqual(prod.composer, "ml-automation-solutions")
    self.assertIn("efdb2a1d6c2c435b8b7de77690c286a9", prod.airflow_url)
    self.assertEqual(prod.bucket, "us-east1-erniechang-test-72c040a8-bucket")
    self.assertEqual(prod.prefix, "data/dag_monitor/orbax-prod")

  def test_every_environment_writes_to_the_test_bucket(self):
    for env in config.ENVIRONMENTS.values():
      self.assertEqual(env.bucket, config.ENVIRONMENTS["test"].bucket)
    prefixes = [env.prefix for env in config.ENVIRONMENTS.values()]
    self.assertEqual(len(prefixes), len(set(prefixes)))

  def test_every_environment_uses_the_same_chat_space(self):
    secrets = {env.webhook_secret for env in config.ENVIRONMENTS.values()}
    self.assertEqual(
        secrets, {"orbax-daily-monitor-chat-webhook-erniechang-test"}
    )


class FakeSession:
  """Records requests; replies from a list."""

  def __init__(self, replies):
    self.replies = list(replies)
    self.requests = []

  def request(self, method, url, timeout=None, **kwargs):
    del timeout
    self.requests.append((method, url, kwargs))
    return self.replies.pop(0)


class ClientsTest(unittest.TestCase):

  def test_logs_pages_and_returns_oldest_first(self):
    session = FakeSession(
        [
            FakeResponse(
                200, {"entries": [{"n": 3}, {"n": 2}], "nextPageToken": "p"}
            ),
            FakeResponse(200, {"entries": [{"n": 1}, {"n": 0}]}),
        ]
    )
    logs = clients.LogClient(session)
    gap = clients.MIN_QUERY_GAP_S
    clients.MIN_QUERY_GAP_S = 0
    self.addCleanup(setattr, clients, "MIN_QUERY_GAP_S", gap)
    self.assertEqual(logs.read("p1", "f", 3), [{"n": 1}, {"n": 2}, {"n": 3}])
    first, second = (
        session.requests[0][2]["json"],
        session.requests[1][2]["json"],
    )
    self.assertEqual(first["resourceNames"], ["projects/p1"])
    self.assertEqual((first["pageSize"], second["pageSize"]), (3, 1))
    self.assertEqual(second["pageToken"], "p")

  def test_airflow_error_raises(self):
    client = clients.AirflowClient(
        FakeSession([FakeResponse(403, {"detail": "no"})]), "https://af/"
    )
    with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
      client.get("/dags", limit=1)

  def test_gcs_write_and_exists(self):
    session = FakeSession([FakeResponse(200), FakeResponse(404)])
    store = clients.GcsStore(session, "bkt", "data/dag_monitor/orbax/")
    self.assertEqual(
        store.write_json("a.json", {"x": 1}),
        "gs://bkt/data/dag_monitor/orbax/a.json",
    )
    self.assertEqual(
        session.requests[0][2]["params"]["name"],
        "data/dag_monitor/orbax/a.json",
    )
    self.assertFalse(store.exists("a.posted.json"))
    self.assertIn(
        "data%2Fdag_monitor%2Forbax%2Fa.posted.json", session.requests[1][1]
    )

  def test_open_store(self):
    self.assertIsInstance(
        clients.open_store(None, "local:/tmp/x", "b", "p"), clients.LocalStore
    )
    store = clients.open_store(None, "gs://other/some/prefix", "b", "p")
    self.assertEqual(store.location, "gs://other/some/prefix")
    self.assertEqual(
        clients.open_store(None, None, "b", "p").location, "gs://b/p"
    )

  def test_read_secret(self):
    data = base64.b64encode(b"https://hook\n").decode()
    session = FakeSession([FakeResponse(200, {"payload": {"data": data}})])
    self.assertEqual(clients.read_secret(session, "p", "s"), "https://hook")

  def test_entry_text(self):
    self.assertEqual(clients.entry_text({"textPayload": "a \n"}), "a")
    self.assertEqual(clients.entry_text({"jsonPayload": {"message": "b"}}), "b")
    self.assertEqual(clients.entry_text({}), "")


class ChatTest(unittest.TestCase):

  def test_error_hides_url(self):
    def post(url, **kwargs):
      raise ConnectionError(f"failed {url}")

    with self.assertRaises(RuntimeError) as e:
      chat.post_messages("https://hook?key=SECRET", ["m"], DAY, post)
    self.assertNotIn("SECRET", str(e.exception))


if __name__ == "__main__":
  unittest.main()
