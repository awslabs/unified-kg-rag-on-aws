# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``s3://bucket/key`` paths: the one parser of every S3 path a user writes
(``run-eval``'s, a schema or seed source's). Pure: no AWS SDK."""

from __future__ import annotations

import re
from pathlib import Path

S3_SCHEME = "s3://"
# S3 general-purpose bucket naming rules (3-63 chars; lowercase letters,
# digits, dots and hyphens; starts and ends with a letter or digit).
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def is_s3_uri(value: str | Path | None) -> bool:
    """True when ``value`` is an ``s3://`` URI, in any case of the scheme
    (URL schemes are case-insensitive); a ``Path`` never is one."""
    return isinstance(value, str) and value[: len(S3_SCHEME)].lower() == S3_SCHEME


def split_s3_uri(uri: str) -> tuple[str, str] | None:
    """The ``(bucket, key)`` of the ``s3://`` URI ``uri``, the key as
    written (``?`` and ``#`` are part of it, empty for the bucket root);
    None when what follows the scheme does not start with a valid bucket
    name (a URL with userinfo, say)."""
    bucket, _, key = uri[len(S3_SCHEME) :].partition("/")
    return (bucket, key) if _BUCKET.match(bucket) else None
