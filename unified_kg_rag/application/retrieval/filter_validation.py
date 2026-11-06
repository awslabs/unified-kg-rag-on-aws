# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filter-key validation of ``GraphRAGChain``: every query refuses, before
any model call, a filter key no index its strategy reads declares, one the
strategy scopes itself, or a schema annotation the configured extraction
schema writes on no item (``auto`` routes only among the strategies that
apply every key).

The functions take the chain they validate for: it holds the config and
the strategy instances (``_get_strategy_instance``), and it keeps
``check_filter_keys`` and ``_validate_filter_keys`` as delegates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from unified_kg_rag.adapters.retrieval.base import (
    SCOPED_FILTER_KEYS,
    BaseGraphRAGRetriever,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import routable_strategies
from unified_kg_rag.adapters.storage.filter_schema import (
    SCHEMA_ANNOTATION_FIELDS,
    check_filter_values,
    typed_filter_values,
    union_filter_fields,
)
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.domain.ingestion.schema_annotation import SchemaAnnotator
from unified_kg_rag.domain.models import SearchStrategy
from unified_kg_rag.shared import InvalidFilterError, get_logger
from unified_kg_rag.shared.exceptions import SchemaSourceError

if TYPE_CHECKING:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput

logger = get_logger(__name__)


def check_filter_keys(
    chain: GraphRAGChain, filters: dict[str, Any] | None, strategy: SearchStrategy
) -> None:
    """Raise ``InvalidFilterError`` if ``filters`` names a key no index
    ``strategy`` reads declares, as each of its queries would (for
    ``auto``: if no routable strategy applies every key, since ``auto``
    routes only among those that do), or a value is malformed
    (``check_filter_values``). For callers that keep filters across
    queries (an interactive session)."""
    check_filter_values(typed_filter_values(filters))
    if strategy is SearchStrategy.AUTO:
        auto_candidates(chain, filters)
    else:
        check_strategy_filter_keys(chain, filters, strategy)


def check_strategy_filter_keys(
    chain: GraphRAGChain, filters: dict[str, Any] | None, strategy: SearchStrategy
) -> None:
    _check_schema_filter_keys(chain, filters)
    instance = chain._get_strategy_instance(strategy)
    reserved = sorted(set(filters or {}) & instance.reserved_filter_keys(strategy))
    if reserved:
        raise InvalidFilterError(
            f"Filter key(s) {', '.join(reserved)}: {strategy.value} search "
            "scopes what it reads by them itself, with values a filter "
            "cannot be intersected with (a range, a browse from the "
            "roots, a list on a multi-valued field), so it refuses them."
        )
    # Under every strategy, those that never scope these keys (naive,
    # global) too: a filter is valid or not whatever the strategy, and
    # auto routes it among the same candidates.
    ranges = sorted(
        key for key in SCOPED_FILTER_KEYS if isinstance((filters or {}).get(key), dict)
    )
    if ranges:
        raise InvalidFilterError(
            f"Filter key(s) {', '.join(ranges)}: a range cannot be "
            "combined with the ids a search scopes these keys to; give a "
            "value or a list of values."
        )
    validate_filter_keys(
        filters,
        instance.filtered_retrievers(strategy),
        instance.filtered_index_prefixes(strategy),
    )


def auto_candidates(
    chain: GraphRAGChain, filters: dict[str, Any] | None
) -> tuple[SearchStrategy, ...]:
    """The routable strategies that apply every key of ``filters``.

    ``auto`` routes only among these, so a filter that one of them
    applies never fails by route. Raises ``InvalidFilterError`` when
    none applies them all.
    """
    routable = routable_strategies(chain.config)
    if not filters:
        return routable
    # Once, not per candidate: every strategy would refuse it alike.
    _check_schema_filter_keys(chain, filters)
    candidates: list[SearchStrategy] = []
    refusals: list[str] = []
    for routed in routable:
        try:
            check_strategy_filter_keys(chain, filters, routed)
        except InvalidFilterError as e:
            refusals.append(f"{routed.value}: {e}")
        else:
            candidates.append(routed)
    if not candidates:
        raise InvalidFilterError(
            "No strategy auto may route to applies every filter key. "
            + " ".join(refusals)
        )
    if refusals:
        logger.info(
            "auto routes only among %s: the filters leave out %s",
            ", ".join(s.value for s in candidates),
            ", ".join(s.value for s in routable if s not in candidates),
        )
    return tuple(candidates)


def _check_schema_filter_keys(
    chain: GraphRAGChain, filters: dict[str, Any] | None
) -> None:
    """Refuse a filter on a schema annotation (``in_schema``,
    ``type_iri``, ``predicate_iri``) the configured extraction schema
    writes on no item.

    Every index declares the field, so the filter would pass the key
    check, match nothing on each index that declares it and empty those
    streams of every strategy (all relationships under ``in_schema:
    true`` with entity types alone): an empty answer, not an error.
    ``in_schema`` is written where the schema applies (``strict``, or
    ``align`` with relationship types), an IRI where a declared type of
    its kind has one and the mode is not ``off``.
    """
    if not filters or not (wanted := SCHEMA_ANNOTATION_FIELDS & set(filters)):
        return
    written = _schema_annotation_fields(chain)
    if written is None:
        return
    missing = sorted(wanted - written)
    if not missing:
        return
    reasons = {
        "in_schema": "in_schema is stored under schema_mode strict, or "
        "align with relationship_types declared",
        "type_iri": "type_iri is stored once an entity type declares an "
        "iri and schema_mode is not off",
        "predicate_iri": "predicate_iri is stored once a relationship type "
        "declares an iri and schema_mode is not off",
    }
    mode = chain.config.processing.graph_extraction.schema_mode
    raise InvalidFilterError(
        f"Filter key(s) {', '.join(missing)}: the configured extraction "
        f"schema (schema_mode {mode}) stores no such field on any entity "
        "or relationship, so the filter would match nothing ("
        + "; ".join(reasons[key] for key in missing)
        + "). Change processing.graph_extraction and re-run ingestion "
        "first."
    )


def _schema_annotation_fields(chain: GraphRAGChain) -> frozenset[str] | None:
    """The annotation fields the configured extraction schema writes
    (``SchemaAnnotator.fields``), or ``None`` when its schema source
    cannot be loaded (the check is then skipped, logged once). The
    source is loaded once, with the chain's session: a failed load is
    not retried on each query."""
    fields: frozenset[str] | None = getattr(chain, "_schema_fields", None)
    if fields is not None or getattr(chain, "_schema_fields_unreadable", False):
        return fields
    try:
        schema = resolve_extraction_schema(
            chain.config, providers=getattr(chain, "providers", None)
        ).processing.graph_extraction.extraction_schema()
    except SchemaSourceError as e:
        chain._schema_fields_unreadable = True
        logger.warning(
            "Not checking schema filter keys against the extraction schema: %s",
            e,
        )
        return None
    # A loader that breaks its contract (raises something else) must not
    # fail, and reload on, every filtered query: the check is advisory.
    except Exception as e:  # noqa: BLE001 - the check is advisory
        chain._schema_fields_unreadable = True
        logger.warning(
            "Not checking schema filter keys against the extraction schema: "
            "loading the schema source failed (%s: %s)",
            type(e).__name__,
            e,
            exc_info=True,
        )
        return None
    fields = SchemaAnnotator(schema).fields
    chain._schema_fields = fields
    return fields


def validate_filter_keys(
    filters: dict[str, Any] | None,
    retrievers: list[BaseGraphRAGRetriever],
    prefixes: list[str] | None = None,
) -> None:
    """Reject caller filter keys that no retriever of the strategy declares
    (on a document retriever, on the indexes ``prefixes`` the strategy
    reads with the caller's filters, when given).

    Each retriever applies a key only to the indexes / labels that declare
    it, so an undeclared key would be dropped everywhere and the query would
    silently run unfiltered. A retriever without a declared schema accepts
    every key.
    """
    if not filters:
        return
    schemas = [
        r.filter_fields() if prefixes is None else r.filter_fields_for(prefixes)
        for r in retrievers
    ]
    if any(schema is None for schema in schemas):
        return
    declared = union_filter_fields(s for s in schemas if s is not None)
    unknown = sorted(key for key in filters if not declared.declares(key))
    if unknown:
        raise InvalidFilterError(
            f"Unknown filter key(s): {', '.join(unknown)}. No index this "
            "search strategy reads has these fields, so the filter would "
            f"be ignored. Filterable keys: {', '.join(declared.describe())}."
        )


def check_requested_filter_keys(chain: GraphRAGChain, rag_input: RAGInput) -> None:
    """Refuse filter keys an explicit strategy refuses, before any model
    call (a follow-up rewrite, query processing); ``auto`` checks them
    before routing (``_resolve_strategy``), but for the schema
    annotation keys, which no strategy applies on a schema that writes
    none (``_check_schema_filter_keys``)."""
    if not rag_input.filters:
        return
    if rag_input.search_strategy is SearchStrategy.AUTO:
        _check_schema_filter_keys(chain, rag_input.filters)
    else:
        check_strategy_filter_keys(chain, rag_input.filters, rag_input.search_strategy)
