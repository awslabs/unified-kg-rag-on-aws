# syntax=docker/dockerfile:1
# aws-graphrag data-plane image: the same image runs every Step Functions phase
# (run-ingestion) and ad-hoc run-rag / run-eval. Built with uv for reproducible,
# locked dependency resolution.
FROM public.ecr.aws/docker/library/python:3.12-slim AS base

# uv (static binary) for fast, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Runtime deps: the AWS CLI is used by the entrypoint to sync the source corpus
# from S3 (the loader only reads a local Path). libgomp1 is needed by the
# scientific stack (graspologic/numpy) used in community detection.
RUN apt-get update \
    && apt-get install -y --no-install-recommends awscli libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# GIT_PYTHON_REFRESH=quiet: GitPython (pulled in transitively by ragas) probes
# for a git binary at import time; the data plane never uses git, so silence the
# probe rather than ship git in the image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Install dependencies first (cached layer) using only the lock + metadata.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now the source, then install the project itself.
COPY aws_graphrag ./aws_graphrag
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Baked-in config (deployed endpoints come from injected env vars; see
# docker/config.yaml) and the S3-sync entrypoint.
COPY docker/config.yaml /app/config.yaml
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}"

# The entrypoint syncs an s3:// source dir locally (if any) then exec's the
# command supplied by Step Functions (run-ingestion ...).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["run-ingestion", "--help"]
