# syntax=docker/dockerfile:1
# unified-kg-rag-on-aws data-plane image: the same image runs every Step Functions phase
# (run-ingestion) and ad-hoc run-rag / run-eval. Built with uv for reproducible,
# locked dependency resolution.
# Base image pinned by tag and digest (multi-arch index) so a re-pushed tag
# cannot change the build. Bump both together, and re-check the apt pins below.
FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f AS base

# uv (static binary) for fast, locked installs. Pinned by tag and digest so a
# new uv release cannot change the build; bump both together.
COPY --from=ghcr.io/astral-sh/uv:0.12.6@sha256:88bc6eb1ccd4b82efd0e1b530caffabddf50dc2bf612e66c14ea25b8ee8a4d3d /uv /usr/local/bin/uv

# Runtime deps: the AWS CLI is used by the entrypoint to sync the source corpus
# from S3 (the loader only reads a local Path). libgomp1 is needed by the
# scientific stack (graspologic/numpy) used in community detection.
# Versions are pinned (reproducible builds + satisfies SAST). Bump these when
# the python:3.12-slim base advances and the pinned version is dropped from the
# apt repo (check with `apt-cache policy awscli libgomp1` in the base image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        awscli=2.23.6-1 \
        libgomp1=14.2.0-19 \
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

# Optional extras. The default image parses .pdf/.txt/.csv/.json only. Markdown
# and HTML need the `unstructured` extra, which adds ~30 packages (spaCy, thinc,
# pypdfium2, ...) and about 200 MB, so it is opt-in:
#   docker build --build-arg UV_EXTRAS="--extra unstructured" -f docker/Dockerfile .
ARG UV_EXTRAS=""

# Install dependencies first (cached layer) using only the lock + metadata.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev ${UV_EXTRAS}

# Now the source, then install the project itself.
COPY unified_kg_rag ./unified_kg_rag
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${UV_EXTRAS}

# Baked-in, endpoint-free config (deployed endpoints come from env vars the
# compute stack injects; see docker/config.yaml) and the S3-sync entrypoint.
COPY docker/config.yaml /app/config.yaml
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}"

# Run as a non-root user (defense in depth; satisfies SAST "missing USER").
# The user owns /app so the entrypoint can write under it; the entrypoint also
# syncs the source corpus to /tmp, which is world-writable. HOME is set so AWS
# CLI / boto3 can resolve a cache/config dir.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
ENV HOME=/home/appuser
USER appuser

# The entrypoint syncs an s3:// source dir locally (if any) then exec's the
# command supplied by Step Functions (run-ingestion ...).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["run-ingestion", "--help"]
