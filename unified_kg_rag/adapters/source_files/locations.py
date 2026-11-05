# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a source's files live: a local path or an S3 location.

``s3://bucket/key`` names one object; ``s3://bucket/prefix/`` (a trailing
``/``) names the objects under the prefix, as a directory. Objects are read
with the run's session (:meth:`SourceClientProvider.client`, which
:class:`.SchemaSourceContext` implements), one client per
:class:`SourceReader`: the size is checked (HeadObject, or the listing)
before any byte is fetched, and a GetObject never reads more than the cap
allows, so an object that grew since it was sized is refused too.

The errors are the :class:`SourceTerms` a caller passes: what they raise
(``SchemaSourceError`` by default) and how they name the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit
from urllib.request import url2pathname

from pydantic import BaseModel, ConfigDict

from unified_kg_rag.shared.exceptions import ConfigurationError, SchemaSourceError
from unified_kg_rag.shared.utils.s3_uri import S3_SCHEME, is_s3_uri, split_s3_uri

_MISSING = {"404", "NoSuchKey", "NotFound"}
_DENIED = {"403", "AccessDenied", "AccessDeniedException", "AllAccessDisabled"}


class SourceClientProvider(Protocol):
    """What reads a source's files from AWS: the run's clients and the
    source's location as logs and errors may show it."""

    @property
    def location(self) -> str:
        """The source's path without credentials (:func:`redact_location`)."""
        ...

    def client(self, service_name: str, **kwargs: Any) -> Any:
        """A client of the run's session for ``service_name``."""
        ...


@dataclass(frozen=True)
class SourceTerms:
    """How the errors of a source's reader name it, and what they raise.

    Attributes:
        noun: What a message calls the source, capitalised (``Schema
            source``, ``Ontology``); lower-cased inside a sentence.
        options: The config key of the source's options, for messages that
            name one (``schema_source.options``).
        max_bytes: The config key of the source's size cap, for the size
            hint (``schema_source.options.max_bytes``).
        kind: The source's kind, for messages that name it (``owl``).
        purpose: What the files serve, for the size hint (``the schema``).
        error: The exception raised; a ``ConfigurationError`` (exit 2).
    """

    noun: str = "Schema source"
    options: str = "schema_source.options"
    max_bytes: str = "schema_source.options.max_bytes"
    kind: str = ""
    purpose: str = "the schema"
    error: type[ConfigurationError] = SchemaSourceError

    def __call__(self, message: str) -> ConfigurationError:
        """``message`` as :attr:`error`."""
        return self.error(message)


SCHEMA_SOURCE_TERMS = SourceTerms()


def redact_location(path: str | None) -> str:
    """``path`` without the parts of a URL that may carry a credential: the
    userinfo (``user:token@``), the query and the fragment. A plain file path
    is returned as is."""
    if path is None:
        return "<no path>"
    if "://" not in path:
        return path
    try:
        parts = urlsplit(path)
        host = parts.netloc.rpartition("@")[2]
    except ValueError:
        return "<unparseable URL>"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def split_s3(
    location: str, *, terms: SourceTerms = SCHEMA_SOURCE_TERMS
) -> tuple[str, str]:
    """``s3://bucket/key`` as ``(bucket, key)``; the key is taken as
    written (``?`` and ``#`` are part of it), empty for the bucket root.

    Raises:
        terms.error: The bucket name is not a valid one (a URL with
            userinfo, say).
    """
    split = split_s3_uri(location)
    if split is None:
        raise terms(
            f"{terms.noun} '{redact_location(location)}' does not name an S3 "
            "bucket; write s3://bucket/key (or s3://bucket/prefix/)"
        )
    return split


def resolve_source_path(
    path: str, *, refusal: str, terms: SourceTerms = SCHEMA_SOURCE_TERMS
) -> Path | str:
    """The local path ``path`` names, or the ``s3://`` URL as is.

    A plain path is a local one, and so is a ``file://`` URL without a host
    (or with ``localhost``); an ``s3://`` URL must name a valid bucket. Every
    source kind that reads files resolves its ``path`` here, so they accept
    the same locations.

    Raises:
        terms.error: ``path`` is a URL of another scheme (or a ``file://``
            URL naming another host), with ``refusal`` as the message; or
            an ``s3://`` URL naming no valid bucket.
    """
    if "://" not in path:
        return Path(path)
    if is_s3_uri(path):
        split_s3(path, terms=terms)
        return path
    parts = urlsplit(path)
    if parts.scheme.lower() != "file" or parts.netloc not in ("", "localhost"):
        raise terms(refusal)
    return Path(url2pathname(parts.path))


class SourceFile(BaseModel):
    """One file of a source, sized but not read yet."""

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
        if is_s3_uri(self.location):
            return self.location
        return Path(self.location).resolve().as_uri()


class SourceReader:
    """Sizes and reads the S3 objects of a source, with one client built on
    the first call; its errors are ``terms``."""

    def __init__(
        self,
        context: SourceClientProvider,
        *,
        terms: SourceTerms = SCHEMA_SOURCE_TERMS,
    ) -> None:
        self._context = context
        self._client: Any = None
        self.terms = terms

    def _s3(self) -> Any:
        if self._client is None:
            self._client = self._context.client("s3")
        return self._client

    def head(self, location: str, *, prefix_hint: bool = False) -> SourceFile:
        """The S3 object ``location`` names, sized (HeadObject).
        ``prefix_hint``: a missing object's error says how to read the
        prefix as a directory (for a source that reads one).

        Raises:
            terms.error: The object is missing, access is denied, or S3
                cannot be reached.
        """
        bucket, key = split_s3(location, terms=self.terms)
        try:
            response = self._s3().head_object(Bucket=bucket, Key=key)
        except _aws_errors() as e:
            raise _s3_error(e, location, "HeadObject", self.terms, prefix_hint) from e
        return SourceFile(location=location, size=int(response["ContentLength"]))

    def list_prefix(self, location: str) -> list[tuple[str, SourceFile]]:
        """The objects under the prefix ``location`` names, each with its
        key relative to the prefix, sorted by key; folder markers (keys
        ending in ``/``) are left out. Every page of the listing is read.

        Raises:
            terms.error: Access is denied, the bucket is missing, or S3
                cannot be reached.
        """
        bucket, prefix = split_s3(location, terms=self.terms)
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
            raise _s3_error(e, location, "ListObjectsV2", self.terms) from e
        return sorted(found, key=lambda pair: pair[0])

    def read(self, file: SourceFile, *, max_bytes: int) -> bytes:
        """The bytes of the S3 object ``file`` names, at most ``max_bytes``
        of them (GetObject).

        Raises:
            terms.error: The object cannot be read, or holds more than
                ``max_bytes`` (it may have grown since it was sized).
        """
        bucket, key = split_s3(file.location, terms=self.terms)
        try:
            body = self._s3().get_object(Bucket=bucket, Key=key)["Body"]
            try:
                data: bytes = body.read(max_bytes + 1)
            finally:
                body.close()
        except _aws_errors() as e:
            raise _s3_error(e, file.location, "GetObject", self.terms) from e
        if len(data) > max_bytes:
            raise _too_large(file.name, max_bytes, self.terms)
        return data


def read_source_bytes(
    context: SourceClientProvider,
    location: str,
    *,
    max_bytes: int,
    terms: SourceTerms = SCHEMA_SOURCE_TERMS,
) -> bytes:
    """The bytes of the S3 object ``location`` names: sized with
    HeadObject, refused above ``max_bytes`` before it is fetched, then read
    with GetObject.

    Raises:
        terms.error: The URL names a prefix or no valid bucket, the object
            is missing, access is denied, S3 cannot be reached, or the
            object holds more than ``max_bytes``.
    """
    _, key = split_s3(location, terms=terms)
    if not key or key.endswith("/"):
        raise terms(
            f"{terms.noun} '{redact_location(location)}' names an S3 prefix; "
            "name one object (s3://bucket/key)"
        )
    reader = SourceReader(context, terms=terms)
    file = reader.head(location)
    if file.size > max_bytes:
        raise _too_large(file.name, max_bytes, terms, file.size)
    return reader.read(file, max_bytes=max_bytes)


def _too_large(
    name: str, max_bytes: int, terms: SourceTerms, size: int | None = None
) -> ConfigurationError:
    if size is None:
        return terms(
            f"{terms.noun} file '{name}' holds more than {max_bytes} bytes "
            "(it grew since it was sized)"
        )
    return terms(
        f"{terms.noun} file '{name}' holds {size} bytes, more than the "
        f"{max_bytes} it may hold"
    )


def _aws_errors() -> tuple[type[Exception], ...]:
    """What an S3 call raises (botocore is imported on the first call)."""
    from botocore.exceptions import BotoCoreError, ClientError

    return (BotoCoreError, ClientError)


def _s3_error(
    error: Exception,
    location: str,
    action: str,
    terms: SourceTerms,
    prefix_hint: bool = False,
) -> ConfigurationError:
    """``error`` of ``action`` on ``location`` as a ``terms.error`` saying
    what to fix; the URL is shown without credentials, the error by its code
    (no request ids or signatures)."""
    from botocore.exceptions import ClientError

    where = redact_location(location)
    noun = terms.noun
    if isinstance(error, ClientError):
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code == "NoSuchBucket":
            return terms(f"{noun} '{where}': the S3 bucket does not exist")
        if code in _MISSING:
            hint = (
                "; end the path with '/' to read the prefix as a directory"
                if prefix_hint
                else ""
            )
            return terms(f"{noun} '{where}' not found (no such S3 object){hint}")
        if code in _DENIED or status == 403:
            return terms(
                f"Access to {noun.lower()} '{where}' denied ({action}); the "
                "run's role needs s3:GetObject on the objects and "
                "s3:ListBucket on the bucket (without it a missing object is "
                "reported as denied too)"
            )
        return terms(
            f"Cannot read {noun.lower()} '{where}' from S3 ({action}): "
            f"{code or status or 'unknown error'}"
        )
    # A BotoCoreError: no credentials, no connection, ...
    return terms(
        f"Cannot read {noun.lower()} '{where}' from S3 ({action}): "
        f"{type(error).__name__}: {error}"
    )
