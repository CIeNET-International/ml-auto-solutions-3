#!/bin/bash
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

# Builds, deploys, schedules and runs the daily monitor Cloud Run Job.
#
# Usage (from the repo root):
#   scripts/daily_monitor/deploy.sh build      # Image -> gcr.io/cloud-ml-auto-solutions/daily-monitoring.
#   scripts/daily_monitor/deploy.sh job        # Create/update the job.
#   scripts/daily_monitor/deploy.sh schedule   # Create/update the 00:30 UTC trigger.
#   scripts/daily_monitor/deploy.sh execute [KEY=VALUE,...]
#       One manual run, e.g. execute REPORT_DATE=2026-10-08,GEMINI_SEARCH=true
#
# MONITOR_ENV=prod (default test) targets the prod job orbax-daily-monitor-prod,
# which reads ml-automation-solutions and writes to the erniechang-test bucket.
#
# Every command is printed before it runs.

set -euo pipefail

PROJECT="${PROJECT:-cloud-ml-auto-solutions}"
REGION="${REGION:-us-east1}"
MONITOR_ENV="${MONITOR_ENV:-test}"
if [[ "${MONITOR_ENV}" == test ]]; then
  JOB="${JOB:-orbax-daily-monitor}"
else
  JOB="${JOB:-orbax-daily-monitor-${MONITOR_ENV}}"
fi
SA="${SA:-ml-auto-solutions-dev@cloud-ml-auto-solutions.iam.gserviceaccount.com}"
SCHEDULE="${SCHEDULE:-30 0 * * *}"
IMAGE="${IMAGE:-gcr.io/${PROJECT}/daily-monitoring}"

run() {
  echo "+ $*" >&2
  "$@"
}

build() {
  # Stage only what the image needs.
  local ctx tag
  ctx="$(mktemp -d)"
  trap 'rm -rf "${ctx}"' RETURN
  for f in __init__.py xlml/__init__.py xlml/utils/__init__.py \
      xlml/utils/dag_monitor.py xlml/apis/__init__.py \
      xlml/apis/gke_cluster_config.py dags/__init__.py \
      dags/common/__init__.py dags/common/vm_resource.py; do
    mkdir -p "${ctx}/$(dirname "${f}")"
    cp "${f}" "${ctx}/${f}"
  done
  mkdir -p "${ctx}/scripts"
  cp -r scripts/daily_monitor "${ctx}/scripts/daily_monitor"
  find "${ctx}" -name '__pycache__' -prune -exec rm -rf {} +
  rm -f "${ctx}/scripts/daily_monitor/"*_test.py
  cp scripts/daily_monitor/Dockerfile "${ctx}/Dockerfile"
  tag="$(git rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M)"
  # --async: streaming build logs needs project Viewer, so poll the status.
  local id status
  id="$(run gcloud builds submit "${ctx}" --project="${PROJECT}" \
    --region="${REGION}" --tag="${IMAGE}:${tag}" --async --format='value(id)')"
  echo "Build ${id}: https://console.cloud.google.com/cloud-build/builds;region=${REGION}/${id}?project=${PROJECT}" >&2
  while :; do
    status="$(gcloud builds describe "${id}" --project="${PROJECT}" \
      --region="${REGION}" --format='value(status)')"
    case "${status}" in
      SUCCESS) break ;;
      QUEUED|PENDING|WORKING) sleep 5 ;;
      *) echo "Build ${id} ended with ${status}" >&2; return 1 ;;
    esac
  done
  run gcloud container images add-tag "${IMAGE}:${tag}" "${IMAGE}:latest" \
    --quiet
}

job() {
  run gcloud run jobs deploy "${JOB}" --project="${PROJECT}" \
    --region="${REGION}" --image="${IMAGE}:latest" \
    --service-account="${SA}" --task-timeout=45m --max-retries=0 \
    --cpu=1 --memory=1Gi \
    --set-env-vars="ENV=${MONITOR_ENV},GEMINI_SEARCH=false,POST=false"
}

schedule() {
  local verb=create
  if gcloud scheduler jobs describe "${JOB}" --project="${PROJECT}" \
      --location="${REGION}" >/dev/null 2>&1; then
    verb=update
  fi
  run gcloud scheduler jobs "${verb}" http "${JOB}" --project="${PROJECT}" \
    --location="${REGION}" --schedule="${SCHEDULE}" --time-zone=UTC \
    --http-method=POST \
    --uri="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${JOB}:run" \
    --oauth-service-account-email="${SA}"
}

execute() {
  local args=()
  if [[ -n "${1:-}" ]]; then
    args+=("--update-env-vars=$1")
  fi
  run gcloud run jobs execute "${JOB}" --project="${PROJECT}" \
    --region="${REGION}" "${args[@]}"
}

case "${1:-}" in
  build) build ;;
  job) job ;;
  schedule) schedule ;;
  execute) execute "${2:-}" ;;
  *)
    sed -n '/^# Usage/,/^# Every/p' "$0" >&2
    exit 2
    ;;
esac
