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
    def _load(source: SchemaSourceConfig) -> SchemaDocument:
        return SchemaDocument(relationship_types=read_types(source.path))

and is selected with ``schema_source.kind: my_format``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeVar

from unified_kg_rag.domain.models.config import INLINE_SCHEMA_SOURCE, SchemaSourceConfig
from unified_kg_rag.domain.models.schema import SchemaDocument
from unified_kg_rag.shared.exceptions import SchemaSourceError

SchemaLoader = Callable[[SchemaSourceConfig], SchemaDocument]
LoaderT = TypeVar("LoaderT", bound=SchemaLoader)

_REGISTRY: dict[str, SchemaLoader] = {}
# The form schema_source.kind takes (SchemaSourceConfig.kind).
_KIND = re.compile(r"^[a-z][a-z0-9_]*$")


def register_schema_source(kind: str) -> Callable[[LoaderT], LoaderT]:
    """Register the decorated loader under ``kind``.

    A later registration of the same kind replaces the earlier one, so a
    deployment can override a built-in.

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
