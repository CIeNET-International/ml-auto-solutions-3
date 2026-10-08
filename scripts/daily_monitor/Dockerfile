# Image for the daily monitor Cloud Run Job. deploy.sh builds it from a
# staged context that holds only the files below, not the whole repo.
FROM python:3.11-slim

WORKDIR /app
COPY scripts/daily_monitor/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --create-home monitor

# xlml/utils/dag_monitor.py, dags/common/vm_resource.py (+ the cluster config
# it imports) and scripts/daily_monitor/.
COPY . /app
RUN chmod -R a+rX /app  # Context files keep the cloudtop's 640/750 modes.
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
USER monitor
CMD ["python", "-m", "scripts.daily_monitor.main"]
