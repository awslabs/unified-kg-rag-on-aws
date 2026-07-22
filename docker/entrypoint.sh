#!/usr/bin/env bash
# Data-plane entrypoint. The ingestion loader only reads a local filesystem
# path, so if the source directory is an s3:// URI (set by Step Functions via
# GRAPHRAG_SOURCE_DIRECTORY) we sync it to a local scratch dir first and rewrite
# the env var to that local path before exec'ing the requested command.
set -euo pipefail

LOCAL_SOURCE_DIR="/tmp/graphrag-source"

if [[ "${GRAPHRAG_SOURCE_DIRECTORY:-}" == s3://* ]]; then
    echo "[entrypoint] syncing source corpus ${GRAPHRAG_SOURCE_DIRECTORY} -> ${LOCAL_SOURCE_DIR}"
    mkdir -p "${LOCAL_SOURCE_DIR}"
    aws s3 sync "${GRAPHRAG_SOURCE_DIRECTORY}" "${LOCAL_SOURCE_DIR}" --only-show-errors
    export GRAPHRAG_SOURCE_DIRECTORY="${LOCAL_SOURCE_DIR}"
    echo "[entrypoint] synced $(find "${LOCAL_SOURCE_DIR}" -type f | wc -l) file(s)"
fi

exec "$@"
