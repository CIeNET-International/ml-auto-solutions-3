FROM python:3.11

RUN apt-get update && apt-get install -y vim nano && nano --version && apt-get install -y curl gpg

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

RUN pip install jax libtpu

RUN pip install --no-cache-dir libtpu==0.0.20.dev20250722+nightly -f https://storage.googleapis.com/libtpu-wheels/index.html

RUN pip install tpu-info
