# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical names of the per-suffix stores.

Indexers write and retrievers read the same OpenSearch aliases and Neptune
vertex labels, so both sides derive the names here.
"""

from datetime import datetime
from typing import Any

from unified_kg_rag.domain.models import Constants, validate_safe_name
from unified_kg_rag.shared import ConfigurationError


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
    name = f"{base}-{suffix or Constants.DEFAULT_SUFFIX.value}"
    if additional_suffix:
        name = f"{name}-{additional_suffix}"
    if timestamp:
        now = datetime.now()
        name = f"{name}-{now:%Y%m%d%H%M%S}{now.microsecond // 1000:03d}"
    return name


def graph_label(
    prefix: str, suffix: str | None, additional_suffix: str | None = None
) -> str:
    """Neptune vertex label: the capitalized label prefix as the base."""
    return store_name(prefix.capitalize(), suffix, additional_suffix)
