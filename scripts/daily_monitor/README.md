# Orbax daily monitor (Cloud Run Job)

Builds the daily Orbax DAG health report **outside Airflow**. Every day at
00:30 UTC, Cloud Scheduler starts a Cloud Run Job. The job reads the
`dags/orbax` runs of the Composer environment `erniechang-test` through the
Airflow REST API and Cloud Logging. It builds the same digest as the
`orbax_daily_monitor` DAG with `xlml/utils/dag_monitor.py`, triages failures
with Gemini (`gemini-3.5-flash`, global endpoint), and writes the files to the
Composer bucket. The Jetski `orbax-daily-monitor` skill reads them from there.

```
Cloud Scheduler (30 0 * * * UTC) ──► Cloud Run Job orbax-daily-monitor
                                     (runs as ml-auto-solutions-dev@)
  ├─ Airflow REST API (GET only) ── DAGs, runs, task instances, XCom, sources
  ├─ Cloud Logging (read only) ──── worker ERROR lines, pod logs, checkpoint events
  ├─ Vertex AI ──────────────────── triage (+ optional read-only search_logs)
  └─ GCS ────────────────────────── gs://us-east1-erniechang-test-72c040a8-bucket/
                                    data/dag_monitor/orbax/<date>.{json,triage.json,draft.json[,search.json]}
```

Posting to Chat is off by default (`POST=false`): new Chat webhooks need a
policy exception (go/webhook-security-decision). Until then, review and post
the draft with the Jetski skill.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point; settings from environment variables; writes the files |
| `collector.py` | Digest from the Airflow REST API + Cloud Logging |
| `gemini.py` | Vertex AI triage (single call, or the search_logs loop) |
| `chat.py` | Webhook post (only with `POST=true`) |
| `clients.py` | REST clients: Airflow, Cloud Logging, GCS, Secret Manager |
| `config.py` | Environments and limits |
| `main_test.py` | Unit tests with fakes |
| `Dockerfile`, `requirements.txt`, `deploy.sh` | Image and deployment |

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ENV` | `test` | Environment in `config.ENVIRONMENTS` |
| `REPORT_DATE` | yesterday (UTC) | Day to report, `YYYY-MM-DD` |
| `GEMINI_SEARCH` | `false` | Let Gemini run up to 5 read-only log searches per failing DAG |
| `POST` | `false` | Post the draft through the Chat webhook |
| `FORCE_POST` | `false` | Post even if `<date>.posted.json` exists |
| `OUTPUT` | Composer bucket folder | `gs://bucket/prefix`, or `local:<dir>` for local runs |

## Test

From the repo root:

```
PYTHONPATH=. python3 -m unittest discover scripts/daily_monitor "*_test.py" -t .
pyink scripts/daily_monitor --pyink-indentation=2 --pyink-use-majority-quotes --line-length=80 --check
pylint --rcfile=.pylintrc scripts/daily_monitor/*.py --disable=E1123
```

CI only checks `dags/` and `xlml/`, so run these locally.

## Run locally (read-only, writes local files)

```
gcloud auth application-default login
PYTHONPATH=. REPORT_DATE=2026-10-08 OUTPUT=local:/tmp/orbax-monitor \
  python3 -m scripts.daily_monitor.main
```

This uses your credentials and calls Vertex AI (a few cents).

## Deploy (project cloud-ml-auto-solutions, region us-east1)

Needs the Cloud Run and Cloud Scheduler APIs (enabled 2026-10-12). From the
repo root:

```
scripts/daily_monitor/deploy.sh build      # Image gcr.io/cloud-ml-auto-solutions/daily-monitoring
scripts/daily_monitor/deploy.sh job        # Cloud Run Job orbax-daily-monitor
scripts/daily_monitor/deploy.sh execute REPORT_DATE=2026-10-08,OUTPUT=gs://us-east1-erniechang-test-72c040a8-bucket/data/dag_monitor/orbax-cloudrun-test
scripts/daily_monitor/deploy.sh schedule   # Daily trigger, 00:30 UTC
```

Logs: Cloud Run → Jobs → `orbax-daily-monitor` → Executions, or
`gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="orbax-daily-monitor"' --project cloud-ml-auto-solutions --limit 100`.

If the Airflow API returns HTTP 403, `ml-auto-solutions-dev@` needs an Airflow
role in `erniechang-test` (Airflow UI → Security → List Users).

## Prod environment

`ENV=prod` reads the prod Composer `ml-automation-solutions` (us-central1)
with Airflow REST API GETs and Cloud Logging reads only, and writes to the
erniechang-test bucket under `data/dag_monitor/orbax-prod/`. It never posts.
It runs as its own job, `orbax-daily-monitor-prod`:

```
MONITOR_ENV=prod scripts/daily_monitor/deploy.sh job
MONITOR_ENV=prod scripts/daily_monitor/deploy.sh execute REPORT_DATE=2026-10-08
MONITOR_ENV=prod scripts/daily_monitor/deploy.sh schedule
```

If the first run gets HTTP 403 from the Airflow API, `ml-auto-solutions-dev@`
needs an Airflow role (Viewer is enough) in `ml-automation-solutions`.

## Pause or roll back

```
gcloud scheduler jobs pause orbax-daily-monitor --location us-east1 --project cloud-ml-auto-solutions
```

The `orbax_daily_monitor` DAG still exists and writes the same files; trigger
it from the Airflow UI as before.
