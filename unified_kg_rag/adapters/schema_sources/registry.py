# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry of extraction-schema sources, keyed by ``schema_source.kind``.

A source turns ``processing.graph_extraction.schema_source`` into a
``SchemaDocument``: the inline schema keys it fills (entity types,
relationship types, patterns). Start-up loads it once and writes the keys
into a copy of the config (``application.bootstrap.resolve_extraction_schema``),
so extraction, prompts and cache keys only ever read the inline keys. A new
format or service registers its own kind, with no edit to the config model
or the extractor::

    @register_schema_source("my_format")
    def _load(context: SchemaSourceContext) -> SchemaDocument:
        return SchemaDocument(relationship_types=read_types(context.source.path))

and is selected with ``schema_source.kind: my_format``. The loader gets a
:class:`SchemaSourceContext`: the source settings (``path``, ``options``,
``secret_arn``), the run's config and an AWS session built on first use.
It must return the types in the same order for the same content (sort what
an unordered source, a set or an API page, yields): the order is part of
the prompt and of the cache keys, so a reordering re-extracts the corpus.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlsplit, urlunsplit

from unified_kg_rag.domain.models.config import (
    INLINE_SCHEMA_SOURCE,
    Config,
    SchemaSourceConfig,
)
from unified_kg_rag.domain.models.schema import SchemaDocument
from unified_kg_rag.shared.exceptions import SchemaSourceError

if TYPE_CHECKING:
    import boto3


def _default_session(config: Config) -> boto3.Session:
    import boto3

    return boto3.Session(
        profile_name=config.aws.profile_name, region_name=config.aws.region_name
    )


class SchemaSourceContext:
    """What a schema loader is given.

    Attributes:
        source: ``processing.graph_extraction.schema_source`` (``kind``,
            ``path``, ``options``, ``secret_arn``). Never log ``path`` as
            is (:attr:`location` is the printable form) or the secret.
        config: The run's config (``aws.region_name``, ``aws.profile_name``
            ...); its schema keys are not loaded yet.
    """

    def __init__(
        self,
        source: SchemaSourceConfig,
        config: Config,
        *,
        session_factory: Callable[[], boto3.Session] | None = None,
    ) -> None:
        """``session_factory`` builds the AWS session, once, on the first
        :meth:`boto_session` (by default from ``aws.profile_name`` and
        ``aws.region_name``), so a loader that makes no AWS call builds
        none."""
        self.source = source
        self.config = config
        self._session_factory = session_factory or (lambda: _default_session(config))
        self._session: boto3.Session | None = None
        self._lock = threading.Lock()

    @property
    def location(self) -> str:
        """``source.path`` as logs and errors may show it
        (:func:`redact_location`)."""
        return redact_location(self.source.path)

    def boto_session(self) -> boto3.Session:
        """The run's AWS session, built on the first call."""
        with self._lock:
            if self._session is None:
                self._session = self._session_factory()
            return self._session

    def client(self, service_name: str, **kwargs: Any) -> Any:
        """A client of the run's session in the region the config sends
        ``service_name`` to: ``aws.bedrock_region`` for the Bedrock
        services, ``aws.region_name`` for every other one."""
        region = (
            self.config.aws.bedrock_region
            if service_name.startswith("bedrock")
            else self.config.aws.region_name
        )
        kwargs.setdefault("region_name", region)
        # Any: the service name is only known at run time.
        session: Any = self.boto_session()
        return session.client(service_name, **kwargs)


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


SchemaLoader = Callable[[SchemaSourceContext], SchemaDocument]
LoaderT = TypeVar("LoaderT", bound=SchemaLoader)

_REGISTRY: dict[str, SchemaLoader] = {}
# The form schema_source.kind takes (SchemaSourceConfig.kind).
_KIND = re.compile(r"^[a-z][a-z0-9_]*$")


def register_schema_source(kind: str) -> Callable[[LoaderT], LoaderT]:
    """Register the decorated loader under ``kind``.

    The loader is called as ``loader(context)`` with a
    :class:`SchemaSourceContext` and returns a ``SchemaDocument``, its types
    in a deterministic order; it raises ``SchemaSourceError`` for a source it
    cannot read. A later registration of the same kind replaces the earlier
    one, so a deployment can override a built-in.

    Raises:
        ValueError: ``kind`` is ``inline`` (the inline keys themselves) or
            not of the form ``schema_source.kind`` takes.
    """
    if kind == INLINE_SCHEMA_SOURCE or not _KIND.match(kind):
        raise ValueError(
            f"Cannot register a schema source as {kind!r}: a kind is "
            f"lower-case letters, digits and '_', and not "
            f"'{INLINE_SCHEMA_SOURCE}'"
        )

    def decorator(loader: LoaderT) -> LoaderT:
        _REGISTRY[kind] = loader
        return loader

    return decorator


def get_schema_source(kind: str) -> SchemaLoader:
    """The loader registered under ``kind``.

    Raises:
        SchemaSourceError: Nothing is registered under ``kind``.
    """
    loader = _REGISTRY.get(kind)
    if loader is None:
        raise SchemaSourceError(
            f"Unknown processing.graph_extraction.schema_source.kind '{kind}'; "
            f"registered: {', '.join([INLINE_SCHEMA_SOURCE, *registered_schema_sources()])}"
        )
    return loader


def registered_schema_sources() -> list[str]:
    """The registered kinds, sorted (``inline`` is built in, not listed)."""
    return sorted(_REGISTRY)
