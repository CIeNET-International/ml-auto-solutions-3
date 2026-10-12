# Orbax daily monitor (Cloud Run Job)

Builds the daily Orbax DAG health report **outside Airflow**. Every day at
00:30 UTC, Cloud Scheduler starts one Cloud Run Job per environment. Each job
reads the `dags/orbax` runs of one Composer environment through the Airflow
REST API and Cloud Logging. It builds the digest with
`xlml/utils/dag_monitor.py`, triages failures with Gemini (`gemini-3.5-flash`,
global endpoint), and writes the files to the erniechang-test bucket. The
Jetski `orbax-daily-monitor` skill reads them from there.

## Environments

| `ENV` | Reads (Composer) | Cloud Run Job | Writes to `gs://us-east1-erniechang-test-72c040a8-bucket/` |
|---|---|---|---|
| `test` | `erniechang-test` (us-east1) | `orbax-daily-monitor` | `data/dag_monitor/orbax/` |
| `prod` | `ml-automation-solutions` (us-central1) | `orbax-daily-monitor-prod` | `data/dag_monitor/orbax-prod/` |

- Both jobs run in `cloud-ml-auto-solutions` / `us-east1` as
  `ml-auto-solutions-dev@`, with Airflow REST API GETs and Cloud Logging reads
  only.
- The monitor never writes to the Composer environment it reads; every
  environment writes to the erniechang-test bucket. This is temporary: we
  don't have permission yet to create a dedicated bucket, so the monitor uses
  the erniechang-test bucket until we do.
- Both report to the same Chat space, "[Daily Monitoring]"
  (`spaces/AAQA57164nY`), with the same title.
- Prod never auto-posts: posting only works in `dag_monitor.AUTO_POST_ENVS`
  (erniechang-test).

```
Cloud Scheduler (30 0 * * * UTC) ──► Cloud Run Job orbax-daily-monitor[-prod]
                                     (runs as ml-auto-solutions-dev@)
  ├─ Airflow REST API (GET only) ── DAGs, runs, task instances, XCom, sources
  ├─ Cloud Logging (read only) ──── worker ERROR lines, pod logs, checkpoint events
  ├─ Vertex AI ──────────────────── triage (+ optional read-only search_logs)
  └─ GCS ────────────────────────── gs://us-east1-erniechang-test-72c040a8-bucket/
                                    data/dag_monitor/orbax[-prod]/<date>.{json,triage.json,draft.json[,search.json]}
```

## Chat report

`<date>.draft.json` holds the messages, posted in one thread:

- **Main message:** title, totals, and one bullet per failure cause. DAGs with
  the same cause share a bullet: `- *Label* (N DAGs): a, b. summary`. No links.
- **One reply per folder** (for example `*orbax/*`): one line per failed DAG
  with its Airflow link, `` - `dag`: Airflow ``. A reply continues in another
  message only past about 4,000 characters (about 12 failures).

Posting to Chat is off by default (`POST=false`): new Chat webhooks need a
policy exception (go/webhook-security-decision). Until then, the Jetski skill
drafts the report (`orbax report [date] [--env test|prod]`) and posts it only
after an explicit "post". The skill can also draft one `[CIeNET-Sheriff]` bug
for a DAG that failed 2 or more runs in a row (`orbax report bug <date>`); it
files the bug only after an explicit "file".

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
| `OUTPUT` | The environment's folder (see Environments) | `gs://bucket/prefix`, or `local:<dir>` for local runs |

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
PYTHONPATH=. ENV=test REPORT_DATE=2026-10-08 OUTPUT=local:/tmp/orbax-monitor \
  python3 -m scripts.daily_monitor.main
```

This uses your credentials and calls Vertex AI (a few cents). Use `ENV=prod`
for the prod Composer.

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

### Prod

`MONITOR_ENV=prod` makes `deploy.sh` target `orbax-daily-monitor-prod`. The
image is shared, so `build` is needed only once:

```
MONITOR_ENV=prod scripts/daily_monitor/deploy.sh job
MONITOR_ENV=prod scripts/daily_monitor/deploy.sh execute REPORT_DATE=2026-10-08
MONITOR_ENV=prod scripts/daily_monitor/deploy.sh schedule
```

`ml-auto-solutions-dev@` can read the prod Airflow API (verified by the
2026-10-12 executions). If that changes and the job gets HTTP 403, it needs an
Airflow role (Viewer is enough) in `ml-automation-solutions`.

## Pause or roll back

```
gcloud scheduler jobs pause orbax-daily-monitor --location us-east1 --project cloud-ml-auto-solutions
gcloud scheduler jobs pause orbax-daily-monitor-prod --location us-east1 --project cloud-ml-auto-solutions
```

Pausing stops the daily reports; run `deploy.sh execute` by hand when needed.
The `orbax_daily_monitor` DAG it replaced was removed on 2026-10-12.
