# syntax=docker/dockerfile:1
# unified-kg-rag-on-aws data-plane image: the same image runs the Step Functions ingestion task
# (run-ingestion) and ad-hoc run-rag / run-eval (the CDK QueryTask). run-eval's
# default evaluators are LLM-free and need no extra; the opt-in ragas judge
# needs the `eval` extra (about 390 MB: ragas, datasets, pandas, pyarrow,
# langchain-community), so it is opt-in here too (UV_EXTRAS below).
# Built with uv for reproducible, locked dependency resolution.
# Base image pinned by tag and digest (multi-arch index) so a re-pushed tag
# cannot change the build. Bump both together, and re-check the apt pins below.
# BASE_IMAGE only lets a build pull the same digest from another registry (CI
# falls back to Docker Hub when public.ecr.aws rate-limits); keep the digest.
# The image builds for linux/amd64 and linux/arm64 (Graviton, the CDK stack's
# fargate_cpu_architecture=arm64) alike: the base and uv digests are
# multi-arch indexes, the apt pins exist for both architectures, and every
# locked package has an aarch64 wheel or is pure Python. Keep a new pin so.
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54
# checkov:skip=CKV_DOCKER_7: BASE_IMAGE defaults to a tag+digest pin; Checkov cannot resolve ARGs
FROM ${BASE_IMAGE} AS build

# uv (static binary) for fast, locked installs, in the build stage only: the
# runtime image gets the virtual environment, not uv. Pinned by tag and
# digest so a new uv release cannot change the build; bump both together.
COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Optional extras (none by default). The ragas judge of run-eval needs the
# `eval` extra (--extra eval). The default image parses
# .pdf/.txt/.csv/.json/.md/.html; Word (.docx) needs the small `docx` extra
# (--extra docx). The `unstructured` extra, the richer Markdown and HTML
# parser, adds ~30 packages (spaCy, thinc, pypdfium2, ...) and about 200 MB,
# so it is opt-in:
#   docker build --build-arg UV_EXTRAS="--extra unstructured" -f docker/Dockerfile .
# With it the build also installs the spaCy model unstructured tokenizes with
# (en_core_web_sm, ~13 MB, below): unstructured would otherwise download it
# from GitHub at the first Markdown/HTML file, which a private VPC (no egress)
# cannot, and the file would wait 120 s and fail.
# The image renders no visualization (run-visualization runs where its HTML is
# read), so it leaves the `viz` extra out.
ARG UV_EXTRAS=""

# Install dependencies first (cached layer) using only the lock + metadata.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev ${UV_EXTRAS}

# Now the source, then the project itself, installed into the venv (not
# editable), so the runtime stage needs only /app/.venv.
COPY unified_kg_rag ./unified_kg_rag
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${UV_EXTRAS}

# A wheel directory for the spaCy model install below: empty unless a build
# passes one (`--build-context spacy-wheels=<dir>`), which replaces this stage.
FROM scratch AS spacy-wheels

FROM ${BASE_IMAGE} AS runtime

# libgomp1 is the system OpenMP runtime that numba's `omp` threading layer
# links (numba comes with the `unstructured` extra); the core and eval wheels
# bundle their own libgomp (scikit-network) or none, so a default image does
# not load it. The corpus is synced from S3 by docker/s3_sync.py (boto3, in
# the venv), so there is no AWS CLI. Versions are pinned (reproducible builds
# + satisfies SAST). Bump the pin when the python:3.12-slim base advances and
# it is dropped from the apt repo (check with `apt-cache policy libgomp1` in
# the base image). zlib1g is upgraded to the Debian security release that fixes
# CVE-2026-76844 and CVE-2026-85091 (HIGH): the base image still carries the
# vulnerable build, so the image scan would fail; drop the pin once the base
# digest ships the fixed version.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1=14.2.0-19 \
        zlib1g=1:1.3.dfsg+really1.3.1-1+deb13u1 \
    && rm -rf /var/lib/apt/lists/*

# GIT_PYTHON_REFRESH=quiet: GitPython (pulled in transitively by ragas, in an
# image built with the eval extra) probes for a git binary at import time;
# the data plane never uses git, so silence the probe rather than ship git in
# the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app
COPY --from=build /app/.venv /app/.venv

# The spaCy model, installed at build time into the venv: the version, URL
# and SHA-256 unstructured pins, downloaded with retries by
# scripts/install_spacy_model.py (bind-mounted, not copied). A wheel in the
# `spacy-wheels` build context is used instead of a download: the empty stage
# above by default, CI's cached wheel with
# `--build-context spacy-wheels=<dir>`. The checks fail the build if
# unstructured cannot tokenize with it.
ARG UV_EXTRAS=""
RUN --mount=type=bind,from=spacy-wheels,target=/var/cache/spacy-wheels,rw \
    --mount=type=bind,source=scripts/install_spacy_model.py,target=/tmp/install_spacy_model.py \
    case "${UV_EXTRAS}" in \
      *unstructured*) \
        /app/.venv/bin/python /tmp/install_spacy_model.py --cache-dir /var/cache/spacy-wheels \
        && /app/.venv/bin/python -c "from unstructured.nlp.tokenize import sent_tokenize; sent_tokenize('Ready.')" \
        && /app/.venv/bin/python -c "import en_core_web_sm" ;; \
    esac

# Baked-in, endpoint-free config (deployed endpoints come from env vars the
# compute stack injects; see docker/config.yaml), the S3-sync entrypoint and
# its sync script.
COPY docker/config.yaml /app/config.yaml
COPY docker/s3_sync.py /usr/local/lib/unified-kg-rag/s3_sync.py
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}"

# Run as a non-root user (defense in depth; satisfies SAST "missing USER").
# The code, the venv and the config stay root-owned and read-only to it; it
# owns only the directories the CLIs write under /app (the stage cache,
# outputs and log files, by their config defaults). The entrypoint syncs the
# source corpus to /tmp, which is world-writable. HOME is set so boto3 and
# the libraries resolve a cache/config dir.
RUN useradd --create-home --uid 10001 appuser \
    && install -d -o appuser -g appuser /app/cache /app/outputs /app/logs
ENV HOME=/home/appuser
USER appuser

# Each task is a batch job that exits when done, with no server to probe; ECS
# and Step Functions track the process exit code instead.
HEALTHCHECK NONE

# The entrypoint syncs an s3:// source dir locally (if any) then exec's the
# command supplied by Step Functions (run-ingestion ...).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["run-ingestion", "--help"]
