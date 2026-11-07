# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What methodologies the queried index namespace holds, and the refusal of
a strategy that reads another (docs/design.md#46-methodology-profiles).

``GraphRAGChain`` reads the namespace's build record through its document
retriever (``aread_build_record``) before any model call of a query, keeps
what it holds per suffix for ``HELD_TTL_SECONDS`` (a failed read is not
kept) and reads it again before refusing, so a methodology added meanwhile
is found. The functions take the
chain they check for, like ``filter_validation``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from unified_kg_rag.adapters.retrieval.strategy_registry import (
    get_strategy_spec,
    serves,
    strategy_artifacts,
    strategy_methodology,
)
from unified_kg_rag.domain.ingestion.methodologies import held_artifacts
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    IndexArtifact,
    Methodology,
    SearchStrategy,
)
from unified_kg_rag.domain.models.methodology import (
    METHODOLOGY_PROFILES,
    artifact_switches,
    methodologies_holding,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import MethodologyNotIndexedError

if TYPE_CHECKING:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

logger = get_logger(__name__)

# How long a suffix's held artifacts are reused before the build record
# is read again (a refusal reads it again first).
HELD_TTL_SECONDS = 300.0


def _namespace(chain: GraphRAGChain, suffix: str | None) -> str:
    return (
        suffix
        or chain.config.processing.document_parsing.index_value
        or Constants.DEFAULT_SUFFIX.value
    )


async def namespace_artifacts(
    chain: GraphRAGChain, suffix: str | None, *, refresh: bool = False
) -> frozenset[IndexArtifact]:
    """The index artifacts the namespace of ``suffix`` holds: from its build
    record, else (no record, one from before methodology profiles, a backend
    without index metadata, an unreadable one) those a run of the config
    builds (``held_artifacts``)."""
    key = suffix or ""
    now = time.monotonic()
    with chain._cache_lock:
        cached = chain._held_artifacts.get(key)
    if cached is not None and not refresh and now - cached[0] < HELD_TTL_SECONDS:
        return cached[1]
    try:
        record = await chain.aread_build_record(suffix)
    except Exception as e:  # noqa: BLE001 - the record is optional
        # The query meets a store error itself; the config stands in here,
        # for this query only: a cached fallback would refuse (or serve)
        # by the config for HELD_TTL_SECONDS after the store recovered.
        logger.debug("Reading the build record of '%s' failed: %s", key, e)
        return held_artifacts(None, chain.config)
    held = held_artifacts(record, chain.config)
    with chain._cache_lock:
        chain._held_artifacts[key] = (now, held)
    return held


async def check_strategy_methodology(
    chain: GraphRAGChain, strategy: SearchStrategy, suffix: str | None
) -> tuple[Methodology, ...]:
    """The methodologies the namespace of ``suffix`` holds, raising
    ``MethodologyNotIndexedError`` if it does not serve ``strategy`` (the
    record is read again first): it lacks an artifact the strategy declares
    it reads (``StrategySpec.required_artifacts``), or, for a strategy that
    declares none, the strategy's methodology.

    For ``auto`` nothing is refused here: it routes among the strategies of
    the held methodologies (``strategy_registry.routable_strategies``).
    """
    for refresh in (False, True):
        artifacts = await namespace_artifacts(chain, suffix, refresh=refresh)
        held = methodologies_holding(artifacts)
        if _served(chain, strategy, artifacts, held):
            return held
    namespace = _namespace(chain, suffix)
    needed = strategy_artifacts(strategy, chain.config)
    methodology = strategy_methodology(strategy, chain.config)
    if needed is not None and not (
        # One that reads its methodology's artifacts and more (global and
        # drift: graphrag and the community reports) on a namespace without
        # that methodology is refused as a strategy that declares none.
        methodology is not None
        and methodology not in held
        and METHODOLOGY_PROFILES[methodology].required <= needed
    ):
        missing = needed - artifacts
        switches = artifact_switches(missing)
        how = (
            "ingest with "
            + " and ".join(f"{key}: true" for key in switches)
            + ": a namespace with the doc-status registry is rebuilt once from "
            "its stored extractions, so only the model calls that build them "
            "are new."
            if switches
            else "run ingestion with indexing.methodologies naming any methodology."
        )
        raise MethodologyNotIndexedError(
            f"Search strategy '{strategy.value}' reads the "
            f"{', '.join(sorted(a.value for a in missing))} of the index "
            f"namespace '{namespace}', which it does not hold; {how}"
        )
    assert methodology is not None  # _served serves a strategy without one
    raise MethodologyNotIndexedError(
        not_indexed_message(strategy, methodology, artifacts, namespace, chain.config)
    )


def _served(
    chain: GraphRAGChain,
    strategy: SearchStrategy,
    artifacts: frozenset[IndexArtifact],
    held: tuple[Methodology, ...],
) -> bool:
    if strategy is SearchStrategy.AUTO:
        return True
    needed = strategy_artifacts(strategy, chain.config)
    if needed is not None:
        return needed <= artifacts
    methodology = strategy_methodology(strategy, chain.config)
    return methodology is None or methodology in held


def not_indexed_message(
    strategy: SearchStrategy,
    methodology: Methodology,
    artifacts: frozenset[IndexArtifact],
    namespace: str,
    config: Config,
) -> str:
    """Why ``strategy`` is refused, what the namespace (holding
    ``artifacts``) serves instead and how to add ``methodology``."""
    held = methodologies_holding(artifacts)
    serving = [
        s.value
        for s in SearchStrategy
        if s is not SearchStrategy.AUTO and _reads(s, artifacts, config)
    ]
    holds = ", ".join(m.value for m in held) or "no methodology"
    instead = f"; its strategies: {', '.join(serving)}, or auto" if serving else ""
    if methodology is Methodology.RAPTOR:
        # The tree is built from the stored text units: no rebuild.
        how = (
            "re-run ingestion: the summary tree is built from the stored text "
            "units, with no re-extraction."
        )
    elif set(held) <= {Methodology.RAPTOR}:
        # Nothing extracted yet: the graph costs a first build.
        how = (
            "re-run ingestion: the namespace holds no graph, so every text "
            "unit is extracted (the cost of a first build)."
        )
    else:
        how = (
            "re-run ingestion: a namespace with the doc-status registry is "
            "rebuilt once from its stored extractions, so only the model "
            f"calls that build what {methodology.value} adds are new."
        )
    return (
        f"Search strategy '{strategy.value}' reads the {methodology.value} "
        f"methodology, which the index namespace '{namespace}' does not hold "
        f"(it holds {holds}{instead}). To add {methodology.value}, include it "
        f"in indexing.methodologies and {how}"
    )


def _reads(
    strategy: SearchStrategy, artifacts: frozenset[IndexArtifact], config: Config
) -> bool:
    try:
        get_strategy_spec(strategy)
    except ValueError:
        return False
    if strategy_artifacts(strategy, config) is not None:
        return serves(strategy, artifacts, config)
    methodology = strategy_methodology(strategy, config)
    return methodology is not None and methodology in methodologies_holding(artifacts)


def servable_strategies(
    artifacts: frozenset[IndexArtifact], config: Config
) -> list[SearchStrategy]:
    """The strategies a namespace holding ``artifacts`` serves, ``auto``
    first when any other is (it routes among them)."""
    served = [
        s
        for s in SearchStrategy
        if s is not SearchStrategy.AUTO and _reads(s, artifacts, config)
    ]
    return [SearchStrategy.AUTO, *served] if served else []
