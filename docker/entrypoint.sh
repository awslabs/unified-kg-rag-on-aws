#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Data-plane entrypoint. The ingestion loader only reads a local filesystem
# path, so if the source directory is an s3:// URI (set by Step Functions via
# UKR_SOURCE_DIRECTORY) we sync it to a local scratch dir first and rewrite
# the env var to that local path before exec'ing the requested command.
set -euo pipefail

LOCAL_SOURCE_DIR="/tmp/graphrag-source"

if [[ "${UKR_SOURCE_DIRECTORY:-}" == s3://* ]]; then
    echo "[entrypoint] syncing source corpus ${UKR_SOURCE_DIRECTORY} -> ${LOCAL_SOURCE_DIR}"
    mkdir -p "${LOCAL_SOURCE_DIR}"
    # aws s3 sync --only-show-errors, with boto3 (docker/s3_sync.py): the
    # image carries no AWS CLI. A failed download fails the task here.
    python /usr/local/lib/unified-kg-rag/s3_sync.py \
        "${UKR_SOURCE_DIRECTORY}" "${LOCAL_SOURCE_DIR}"
    # Every corpus is staged in the same local dir, so incremental deletion is
    # scoped to the S3 source rather than the staging path (unless set).
    # The URI is kept as typed: document ids derive from it, so
    # s3://b/corpus and s3://b/corpus/ give the same files different ids and
    # switching spelling indexes the corpus again (iac/README.md, "One
    # spelling per corpus"). Normalizing it would change existing ids.
    export UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE="${UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE:-${UKR_SOURCE_DIRECTORY}}"
    export UKR_SOURCE_DIRECTORY="${LOCAL_SOURCE_DIR}"
    echo "[entrypoint] synced $(find "${LOCAL_SOURCE_DIR}" -type f | wc -l) file(s)"
fi

exec "$@"
