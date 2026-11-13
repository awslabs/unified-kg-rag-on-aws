# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical names of the per-suffix stores.

Indexers write and retrievers read the same OpenSearch aliases and Neptune
vertex labels, so both sides derive the names here.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from unified_kg_rag.domain.models import Config, Constants, validate_safe_name
from unified_kg_rag.shared import ConfigurationError


def namespace_suffix(suffix: str | None) -> str:
    """The suffix a store name ends in for ``suffix``: itself, else
    ``Constants.DEFAULT_SUFFIX``. For an item or a store request, which name
    their suffix themselves (an item without one was written by a run
    without ``index_value``); a query or a run resolves its own
    with :func:`resolve_namespace`."""
    return suffix or Constants.DEFAULT_SUFFIX.value


def resolve_namespace(config: Config, suffix: str | None) -> str:
    """The index suffix a query or a run under ``config`` addresses:
    ``suffix``, else ``processing.document_parsing.index_value`` (the
    namespace ingestion with the same config writes), else
    ``Constants.DEFAULT_SUFFIX``. The one rule of every entry point (the
    chain, the CLIs, the MCP server, the console, ingestion), so a query
    without a suffix reads what ingestion with its config wrote.

    Raises:
        ConfigurationError: ``suffix`` is not a safe name (lowercase
            letters, digits, ``-``, ``_``): a wildcard or a comma would
            widen the store target to other namespaces.
    """
    if suffix:
        try:
            validate_safe_name(suffix, "index suffix")
        except ValueError as e:
            raise ConfigurationError(str(e)) from e
    return namespace_suffix(suffix or config.processing.document_parsing.index_value)


def index_suffix(value: Any) -> str | None:
    """The index suffix an item's ``index`` value names: the string, or the
    first of a list; ``None`` when it names none (the stores then use
    ``Constants.DEFAULT_SUFFIX``). The one rule the indexers and the
    registry namespace share, so a document's registry record and its
    artifacts are kept under the same namespace.

    Raises:
        ConfigurationError: The value is not a safe name: no store accepts
            it as part of an index name or a label.
    """
    if isinstance(value, str) and value:
        suffix = value
    elif isinstance(value, list | tuple) and value:
        suffix = str(value[0])
    else:
        return None
    try:
        return validate_safe_name(suffix, "index suffix")
    except ValueError as e:
        raise ConfigurationError(str(e)) from e


def store_name(
    base: str,
    suffix: str | None,
    additional_suffix: str | None = None,
    *,
    timestamp: bool = False,
) -> str:
    """``<base>-<suffix>[-<additional_suffix>][-<timestamp>]``.

    ``suffix`` defaults to ``Constants.DEFAULT_SUFFIX``; ``additional_suffix``
    is ``indexing.additional_suffix``. ``timestamp`` names a concrete
    OpenSearch index behind an alias, to the millisecond
    (``YYYYmmddHHMMSSfff``); the indexer moves a name that is already
    taken (``OpenSearchIndexer._new_index_name``).
    """
    name = f"{base}-{namespace_suffix(suffix)}"
    if additional_suffix:
        name = f"{name}-{additional_suffix}"
    if timestamp:
        now = datetime.now()
        name = f"{name}-{now:%Y%m%d%H%M%S}{now.microsecond // 1000:03d}"
    return name


def suffixes_of_aliases(
    aliases: Iterable[str], base: str, additional_suffix: str | None = None
) -> list[str]:
    """The suffixes ``aliases`` name under ``base`` (``store_name`` read
    back: ``<base>-<suffix>[-<additional_suffix>]``), sorted: the
    namespaces a store holds an index of ``base`` for."""
    prefix = f"{base}-"
    suffixes: set[str] = set()
    for name in aliases:
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if additional_suffix:
            if not rest.endswith(f"-{additional_suffix}"):
                continue
            rest = rest[: -len(additional_suffix) - 1]
        if rest:
            suffixes.add(rest)
    return sorted(suffixes)


def graph_label(
    prefix: str, suffix: str | None, additional_suffix: str | None = None
) -> str:
    """Neptune vertex label: the capitalized label prefix as the base."""
    return store_name(prefix.capitalize(), suffix, additional_suffix)
