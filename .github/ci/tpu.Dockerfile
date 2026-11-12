# The image the TPU jobs run in: Python, jax[tpu] at the pin, pytest and git
# (scripts/mlir_snapshot.py records the commit it lowers). The tree under
# test is mounted read-only at /src and imported from there, so no code of a
# pull request runs while the image builds, the one step with a network.
# tpu_job.sh builds it from main's copy of this file.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ARG JAX_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "jax[tpu]==${JAX_VERSION}" "pytest>=8"

LABEL flywheel-tpu-ci=1
ENV PYTHONPATH=/src \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_OPTIONAL_LOCKS=0
WORKDIR /src
