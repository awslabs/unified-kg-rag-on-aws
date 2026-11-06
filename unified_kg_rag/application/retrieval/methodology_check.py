# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What methodologies the queried index namespace holds, and the refusal of
a strategy that reads another (docs/design.md#46-methodology-profiles).

``GraphRAGChain`` reads the namespace's build record through its document
retriever (``aread_build_record``) before any model call of a query, keeps
what it holds per suffix for ``HELD_TTL_SECONDS`` and reads it again before
refusing, so a methodology added meanwhile is found. The functions take the
chain they check for, like ``filter_validation``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from unified_kg_rag.adapters.retrieval.strategy_registry import (
    get_strategy_spec,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    strategy_methodology as registered_methodology,
)
from unified_kg_rag.domain.ingestion.methodologies import held_methodologies
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    Methodology,
    RetrieverRole,
    SearchStrategy,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import MethodologyNotIndexedError

if TYPE_CHECKING:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

logger = get_logger(__name__)

# How long a suffix's held methodologies are reused before the build record
# is read again (a refusal reads it again first).
HELD_TTL_SECONDS = 300.0


def _namespace(chain: GraphRAGChain, suffix: str | None) -> str:
    return (
        suffix
        or chain.config.processing.document_parsing.index_value
        or Constants.DEFAULT_SUFFIX.value
    )


async def namespace_methodologies(
    chain: GraphRAGChain, suffix: str | None, *, refresh: bool = False
) -> tuple[Methodology, ...]:
    """The methodologies the index namespace of ``suffix`` holds: from its
    build record, else (no record, one from before methodology profiles, a
    backend without index metadata, an unreadable one) the configured
    effective methodologies."""
    key = suffix or ""
    now = time.monotonic()
    with chain._cache_lock:
        cached = chain._held_methodologies.get(key)
    if cached is not None and not refresh and now - cached[0] < HELD_TTL_SECONDS:
        return cached[1]
    record = None
    try:
        retriever = chain._get_retriever(RetrieverRole.DOCUMENT)
        record = await retriever.aread_build_record(suffix)
    except Exception as e:  # noqa: BLE001 - the record is optional
        # The query meets a store error itself; the config stands in here.
        logger.debug("Reading the build record of '%s' failed: %s", key, e)
    held = held_methodologies(record, chain.config)
    with chain._cache_lock:
        chain._held_methodologies[key] = (now, held)
    return held


def strategy_methodology(
    chain: GraphRAGChain, strategy: SearchStrategy
) -> Methodology | None:
    """The methodology ``strategy`` reads: its instance's, which a wrapper
    sets from the strategy it wraps (``None``: it declares none). When the
    instance cannot be built (its stores fail), its class's: the query meets
    that failure itself, under ``processing.ignore_errors`` too."""
    if strategy is SearchStrategy.AUTO:
        return None
    try:
        methodology = chain._get_strategy_instance(strategy).methodology
    except Exception as e:  # noqa: BLE001 - the query reports it
        logger.debug("Building the %s strategy failed: %s", strategy.value, e)
        methodology = None
    return methodology or registered_methodology(strategy, chain.config)


async def check_strategy_methodology(
    chain: GraphRAGChain, strategy: SearchStrategy, suffix: str | None
) -> tuple[Methodology, ...]:
    """The methodologies the namespace of ``suffix`` holds, raising
    ``MethodologyNotIndexedError`` if ``strategy`` reads another one (the
    record is read again first).

    For ``auto`` nothing is refused here: it routes among the strategies of
    the held methodologies (``strategy_registry.routable_strategies``).
    """
    held = await namespace_methodologies(chain, suffix)
    methodology = strategy_methodology(chain, strategy)
    if methodology is None or methodology in held:
        return held
    held = await namespace_methodologies(chain, suffix, refresh=True)
    if methodology in held:
        return held
    raise MethodologyNotIndexedError(
        not_indexed_message(
            strategy, methodology, held, _namespace(chain, suffix), chain.config
        )
    )


def not_indexed_message(
    strategy: SearchStrategy,
    methodology: Methodology,
    held: tuple[Methodology, ...],
    namespace: str,
    config: Config,
) -> str:
    """Why ``strategy`` is refused, what the namespace serves instead and how
    to add ``methodology``."""
    serving = [
        s.value
        for s in SearchStrategy
        if s is not SearchStrategy.AUTO and _reads(s, held, config)
    ]
    holds = ", ".join(m.value for m in held) or "no methodology"
    instead = f"; its strategies: {', '.join(serving)}, or auto" if serving else ""
    return (
        f"Search strategy '{strategy.value}' reads the {methodology.value} "
        f"methodology, which the index namespace '{namespace}' does not hold "
        f"(it holds {holds}{instead}). To add {methodology.value}, include it "
        "in indexing.methodologies and re-run ingestion: a namespace with "
        "the doc-status registry is rebuilt once from its stored "
        "extractions, so only the model calls that build what "
        f"{methodology.value} adds are new."
    )


def _reads(
    strategy: SearchStrategy, held: tuple[Methodology, ...], config: Config
) -> bool:
    try:
        get_strategy_spec(strategy)
    except ValueError:
        return False
    methodology = registered_methodology(strategy, config)
    return methodology is not None and methodology in held
