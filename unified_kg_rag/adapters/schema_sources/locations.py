# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a schema source's files live: a local path or an S3 location.

``s3://bucket/key`` names one object; ``s3://bucket/prefix/`` (a trailing
``/``) names the objects under the prefix, as a directory. Objects are read
with the run's session (:meth:`.SchemaSourceContext.client`), one client per
:class:`SourceReader`: the size is checked (HeadObject, or the listing)
before any byte is fetched, and a GetObject never reads more than the cap
allows, so an object that grew since it was sized is refused too.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from unified_kg_rag.shared.exceptions import SchemaSourceError

from .registry import SchemaSourceContext, redact_location

S3_SCHEME = "s3://"
# S3 general-purpose bucket naming rules (3-63 chars; lowercase letters,
# digits, dots and hyphens; starts and ends with a letter or digit).
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_MISSING = {"404", "NoSuchKey", "NotFound"}
_DENIED = {"403", "AccessDenied", "AccessDeniedException", "AllAccessDisabled"}


def is_s3(location: str) -> bool:
    """True when ``location`` is an ``s3://`` URL (any case of the scheme)."""
    return location[: len(S3_SCHEME)].lower() == S3_SCHEME


def split_s3(location: str) -> tuple[str, str]:
    """``s3://bucket/key`` as ``(bucket, key)``; the key is taken as
    written (``?`` and ``#`` are part of it), empty for the bucket root.

    Raises:
        SchemaSourceError: The bucket name is not a valid one (a URL with
            userinfo, say).
    """
    bucket, _, key = location[len(S3_SCHEME) :].partition("/")
    if not _BUCKET.match(bucket):
        raise SchemaSourceError(
            f"Schema source '{redact_location(location)}' does not name an S3 "
            "bucket; write s3://bucket/key (or s3://bucket/prefix/)"
        )
    return bucket, key


class SourceFile(BaseModel):
    """One file of a schema source, sized but not read yet."""

    model_config = ConfigDict(frozen=True)

    location: str  # a local path, or an s3:// URL as written
    size: int

    @property
    def name(self) -> str:
        """The location as logs and errors show it."""
        return redact_location(self.location)

    @property
    def suffix(self) -> str:
        """The lower-case suffix of the file name (``.ttl``), or ``""``."""
        return Path(self.location.rpartition("/")[2]).suffix.lower()

    @property
    def base(self) -> str:
        """The URI the file's relative references resolve against: the
        ``s3://`` URL, or the local file's ``file://`` URI."""
        if is_s3(self.location):
            return self.location
        return Path(self.location).resolve().as_uri()


class SourceReader:
    """Sizes and reads the S3 objects of a schema source, with one client
    built on the first call."""

    def __init__(self, context: SchemaSourceContext) -> None:
        self._context = context
        self._client: Any = None

    def _s3(self) -> Any:
        if self._client is None:
            self._client = self._context.client("s3")
        return self._client

    def head(self, location: str, *, prefix_hint: bool = False) -> SourceFile:
        """The S3 object ``location`` names, sized (HeadObject).
        ``prefix_hint``: a missing object's error says how to read the
        prefix as a directory (for a source that reads one).

        Raises:
            SchemaSourceError: The object is missing, access is denied, or
                S3 cannot be reached.
        """
        bucket, key = split_s3(location)
        try:
            response = self._s3().head_object(Bucket=bucket, Key=key)
        except _aws_errors() as e:
            raise _s3_error(e, location, "HeadObject", prefix_hint) from e
        return SourceFile(location=location, size=int(response["ContentLength"]))

    def list_prefix(self, location: str) -> list[tuple[str, SourceFile]]:
        """The objects under the prefix ``location`` names, each with its
        key relative to the prefix, sorted by key; folder markers (keys
        ending in ``/``) are left out. Every page of the listing is read.

        Raises:
            SchemaSourceError: Access is denied, the bucket is missing, or
                S3 cannot be reached.
        """
        bucket, prefix = split_s3(location)
        found: list[tuple[str, SourceFile]] = []
        try:
            paginator = self._s3().get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if key.endswith("/"):
                        continue
                    found.append(
                        (
                            key[len(prefix) :],
                            SourceFile(
                                location=f"{S3_SCHEME}{bucket}/{key}",
                                size=int(item["Size"]),
                            ),
                        )
                    )
        except _aws_errors() as e:
            raise _s3_error(e, location, "ListObjectsV2") from e
        return sorted(found, key=lambda pair: pair[0])

    def read(self, file: SourceFile, *, max_bytes: int) -> bytes:
        """The bytes of the S3 object ``file`` names, at most ``max_bytes``
        of them (GetObject).

        Raises:
            SchemaSourceError: The object cannot be read, or holds more than
                ``max_bytes`` (it may have grown since it was sized).
        """
        bucket, key = split_s3(file.location)
        try:
            body = self._s3().get_object(Bucket=bucket, Key=key)["Body"]
            try:
                data: bytes = body.read(max_bytes + 1)
            finally:
                body.close()
        except _aws_errors() as e:
            raise _s3_error(e, file.location, "GetObject") from e
        if len(data) > max_bytes:
            raise _too_large(file.name, max_bytes)
        return data


def read_source_bytes(
    context: SchemaSourceContext, location: str, *, max_bytes: int
) -> bytes:
    """The bytes of the S3 object ``location`` names: sized with
    HeadObject, refused above ``max_bytes`` before it is fetched, then read
    with GetObject.

    Raises:
        SchemaSourceError: The URL names a prefix or no valid bucket, the
            object is missing, access is denied, S3 cannot be reached, or
            the object holds more than ``max_bytes``.
    """
    _, key = split_s3(location)
    if not key or key.endswith("/"):
        raise SchemaSourceError(
            f"Schema source '{redact_location(location)}' names an S3 prefix; "
            "name one object (s3://bucket/key)"
        )
    reader = SourceReader(context)
    file = reader.head(location)
    if file.size > max_bytes:
        raise _too_large(file.name, max_bytes, file.size)
    return reader.read(file, max_bytes=max_bytes)


def _too_large(name: str, max_bytes: int, size: int | None = None) -> SchemaSourceError:
    if size is None:
        return SchemaSourceError(
            f"Schema source file '{name}' holds more than {max_bytes} bytes "
            "(it grew since it was sized)"
        )
    return SchemaSourceError(
        f"Schema source file '{name}' holds {size} bytes, more than the "
        f"{max_bytes} it may hold"
    )


def _aws_errors() -> tuple[type[Exception], ...]:
    """What an S3 call raises (botocore is imported on the first call)."""
    from botocore.exceptions import BotoCoreError, ClientError

    return (BotoCoreError, ClientError)


def _s3_error(
    error: Exception, location: str, action: str, prefix_hint: bool = False
) -> SchemaSourceError:
    """``error`` of ``action`` on ``location`` as a ``SchemaSourceError``
    saying what to fix; the URL is shown without credentials, the error by
    its code (no request ids or signatures)."""
    from botocore.exceptions import ClientError

    where = redact_location(location)
    if isinstance(error, ClientError):
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code == "NoSuchBucket":
            return SchemaSourceError(
                f"Schema source '{where}': the S3 bucket does not exist"
            )
        if code in _MISSING:
            hint = (
                "; end the path with '/' to read the prefix as a directory"
                if prefix_hint
                else ""
            )
            return SchemaSourceError(
                f"Schema source '{where}' not found (no such S3 object){hint}"
            )
        if code in _DENIED or status == 403:
            return SchemaSourceError(
                f"Access to schema source '{where}' denied ({action}); the "
                "run's role needs s3:GetObject on the objects and "
                "s3:ListBucket on the bucket (without it a missing object is "
                "reported as denied too)"
            )
        return SchemaSourceError(
            f"Cannot read schema source '{where}' from S3 ({action}): "
            f"{code or status or 'unknown error'}"
        )
    # A BotoCoreError: no credentials, no connection, ...
    return SchemaSourceError(
        f"Cannot read schema source '{where}' from S3 ({action}): "
        f"{type(error).__name__}: {error}"
    )
