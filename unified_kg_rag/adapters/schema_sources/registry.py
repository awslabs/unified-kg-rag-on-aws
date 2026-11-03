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

from collections.abc import Callable
from typing import TypeVar

from unified_kg_rag.adapters.source_files.context import (
    KIND_PATTERN,
    KindRegistry,
    SourceContextBase,
)
from unified_kg_rag.domain.models.config import INLINE_SCHEMA_SOURCE, SchemaSourceConfig
from unified_kg_rag.domain.models.schema import SchemaDocument
from unified_kg_rag.shared.exceptions import SchemaSourceError


class SchemaSourceContext(SourceContextBase[SchemaSourceConfig]):
    """What a schema loader is given; a
    :class:`.source_files.locations.SourceClientProvider`, so the shared
    location readers read its S3 objects.

    Attributes:
        source: ``processing.graph_extraction.schema_source`` (``kind``,
            ``path``, ``options``, ``secret_arn``). Never log ``path`` as
            is (:attr:`location` is the printable form) or the secret.
        config: The run's config (``aws.region_name``, ``aws.profile_name``
            ...); its schema keys are not loaded yet.
    """


SchemaLoader = Callable[[SchemaSourceContext], SchemaDocument]
LoaderT = TypeVar("LoaderT", bound=SchemaLoader)

_KINDS: KindRegistry[SchemaLoader] = KindRegistry()
_REGISTRY = _KINDS.loaders
# Kind -> (module, how to install it) of the optional dependency its loader
# imports.
_REQUIRES = _KINDS.requires


def register_schema_source(
    kind: str, *, requires: tuple[str, str] | None = None
) -> Callable[[LoaderT], LoaderT]:
    """Register the decorated loader under ``kind``.

    The loader is called as ``loader(context)`` with a
    :class:`SchemaSourceContext` and returns a ``SchemaDocument``, its types
    in a deterministic order; it raises ``SchemaSourceError`` for a source it
    cannot read. ``requires`` is the ``(module, how to install it)`` of the
    optional dependency the loader imports, or None: the CLIs check it is
    installed before any model call. A later registration of the same kind
    replaces the earlier one, requirement included, so a deployment can
    override a built-in.

    Raises:
        ValueError: ``kind`` is ``inline`` (the inline keys themselves) or
            not of the form ``schema_source.kind`` takes.
    """
    if kind == INLINE_SCHEMA_SOURCE or not KIND_PATTERN.match(kind):
        raise ValueError(
            f"Cannot register a schema source as {kind!r}: a kind is "
            f"lower-case letters, digits and '_', and not "
            f"'{INLINE_SCHEMA_SOURCE}'"
        )

    def decorator(loader: LoaderT) -> LoaderT:
        _KINDS.register(kind, loader, requires)
        return loader

    return decorator


def get_schema_source(kind: str) -> SchemaLoader:
    """The loader registered under ``kind``.

    Raises:
        SchemaSourceError: Nothing is registered under ``kind``.
    """
    loader = _KINDS.get(kind)
    if loader is None:
        raise SchemaSourceError(
            f"Unknown processing.graph_extraction.schema_source.kind '{kind}'; "
            f"registered: {', '.join([INLINE_SCHEMA_SOURCE, *registered_schema_sources()])}"
        )
    return loader


def schema_source_requirement(kind: str) -> tuple[str, str] | None:
    """The ``(module, how to install it)`` ``kind`` declared, or None."""
    return _KINDS.requirement(kind)


def registered_schema_sources() -> list[str]:
    """The registered kinds, sorted (``inline`` is built in, not listed)."""
    return _KINDS.kinds()
