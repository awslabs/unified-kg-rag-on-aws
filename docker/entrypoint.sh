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
    aws s3 sync "${UKR_SOURCE_DIRECTORY}" "${LOCAL_SOURCE_DIR}" --only-show-errors
    # Every corpus is staged in the same local dir, so incremental deletion is
    # scoped to the S3 source rather than the staging path (unless set).
    export UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE="${UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE:-${UKR_SOURCE_DIRECTORY}}"
    export UKR_SOURCE_DIRECTORY="${LOCAL_SOURCE_DIR}"
    echo "[entrypoint] synced $(find "${LOCAL_SOURCE_DIR}" -type f | wc -l) file(s)"
fi

exec "$@"
