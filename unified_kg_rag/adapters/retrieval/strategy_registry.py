# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for search strategies.

Each strategy class registers itself under its ``SearchStrategy`` member with
``@register_strategy``, declaring the retriever roles it needs and the
query-side extractions it reads (the methodology whose artifacts it reads is
the class's ``methodology``); :mod:`unified_kg_rag.application.retrieval.rag_chain` builds and
dispatches strategies from these records alone. The keys are the closed
``SearchStrategy`` enum, so a new strategy is an in-tree change
(docs/design.md#15-extension-guide). Like the provider and renderer
registries it lives with what it registers: the strategies are
``BaseSearchStrategy`` adapters, so the registry is an adapter module too.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    RetrieverRole,
    SearchStrategy,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import MethodologyNotIndexedError

if TYPE_CHECKING:
    from unified_kg_rag.adapters.retrieval.base import BaseSearchStrategy

logger = get_logger(__name__)


class QueryInput(str, Enum):
    """Query-side LLM outputs a strategy reads from ``SearchQuery``.

    The chain runs only the extractions the resolved strategy declares.
    """

    ENTITIES = "entities"  # ``entity_focus``
    DUAL_KEYWORDS = "dual_keywords"  # ``hl_keywords`` / ``ll_keywords``


@dataclass(frozen=True)
class StrategySpec:
    """Registration record for a search strategy.

    Attributes:
        strategy_class: The ``BaseSearchStrategy`` subclass implementing the mode.
        required_roles: Retriever ROLES the strategy needs injected (GRAPH /
            DOCUMENT), not concrete backends. The composition root binds each
            role to an adapter, so strategies stay backend-agnostic.
        query_inputs: Query-side extractions the strategy reads.
        needs_community_reports: Whether the strategy answers from community
            reports, which only a namespace holding ``graphrag`` has (a
            strategy declaring ``methodology`` graphrag needs no flag).
        wraps: For a strategy that runs another one's search (``iterative``),
            the strategy it wraps under a config. The chain then extracts
            what that one reads and packs the context as that one does
            (``wrapped_strategy``), and the wrapper reads its methodology
            (``strategy_methodology``).
    """

    strategy_class: type[BaseSearchStrategy]
    required_roles: tuple[RetrieverRole, ...] = field(
        default=(RetrieverRole.DOCUMENT, RetrieverRole.GRAPH)
    )
    query_inputs: frozenset[QueryInput] = frozenset()
    needs_community_reports: bool = False
    wraps: Callable[[Config], SearchStrategy] | None = None

    @property
    def methodology(self) -> Methodology | None:
        """The methodology the strategy class declares (``None``: one a
        wrapping instance sets)."""
        return self.strategy_class.methodology


_REGISTRY: dict[SearchStrategy, StrategySpec] = {}

StrategyT = TypeVar("StrategyT", bound="type[BaseSearchStrategy]")


def register_strategy(
    strategy: SearchStrategy,
    *,
    required_roles: tuple[RetrieverRole, ...] = (
        RetrieverRole.DOCUMENT,
        RetrieverRole.GRAPH,
    ),
    query_inputs: frozenset[QueryInput] = frozenset(),
    needs_community_reports: bool = False,
    wraps: Callable[[Config], SearchStrategy] | None = None,
) -> Callable[[StrategyT], StrategyT]:
    """Class decorator that registers a search strategy under ``strategy``.

    Args:
        strategy: The ``SearchStrategy`` enum value this class implements.
        required_roles: Retriever roles to inject when instantiating it.
        query_inputs: Query-side extractions the strategy reads; the chain
            skips the LLM calls for the rest.
        needs_community_reports: The strategy answers from community reports;
            ``auto`` does not route to it on a namespace without
            ``graphrag``.
        wraps: The strategy it runs the search of, under a config
            (``StrategySpec.wraps``).

    Raises:
        ValueError: If ``strategy`` is already registered to another class.
    """

    def decorator(cls: StrategyT) -> StrategyT:
        if strategy in _REGISTRY and _REGISTRY[strategy].strategy_class is not cls:
            raise ValueError(
                f"Search strategy '{strategy.value}' is already registered to "
                f"'{_REGISTRY[strategy].strategy_class.__name__}'; cannot re-register "
                f"to '{cls.__name__}'."
            )
        _REGISTRY[strategy] = StrategySpec(
            strategy_class=cls,
            required_roles=tuple(required_roles),
            query_inputs=frozenset(query_inputs),
            needs_community_reports=needs_community_reports,
            wraps=wraps,
        )
        logger.debug(
            "Registered search strategy '%s' -> %s", strategy.value, cls.__name__
        )
        return cls

    return decorator


def get_strategy_spec(strategy: SearchStrategy) -> StrategySpec:
    """Look up the registration record for ``strategy``.

    Raises:
        ValueError: If no strategy is registered for ``strategy``.
    """
    try:
        return _REGISTRY[strategy]
    except KeyError:
        available = ", ".join(sorted(s.value for s in _REGISTRY))
        raise ValueError(
            f"No search strategy registered for '{strategy.value}'. "
            f"Available: {available or '(none)'}."
        ) from None


def wrapped_strategy(strategy: SearchStrategy, config: Config) -> SearchStrategy:
    """The strategy whose query extraction and context packing ``strategy``
    uses under ``config``: the one it wraps (``StrategySpec.wraps``), else
    itself. A wrapper is never wrapped again."""
    wraps = get_strategy_spec(strategy).wraps
    return strategy if wraps is None else wraps(config)


def query_inputs(strategy: SearchStrategy, config: Config) -> frozenset[QueryInput]:
    """The query-side extractions ``strategy`` reads under ``config``: its
    own, or those of the strategy it wraps (``wrapped_strategy``)."""
    return get_strategy_spec(wrapped_strategy(strategy, config)).query_inputs


def strategy_methodology(
    strategy: SearchStrategy, config: Config
) -> Methodology | None:
    """The methodology ``strategy`` reads under ``config``: its class's, or,
    for a wrapper that declares none, that of the strategy it wraps
    (``wrapped_strategy``: ``iterative`` over ``mix`` reads lightrag)."""
    own = get_strategy_spec(strategy).methodology
    if own is not None:
        return own
    return get_strategy_spec(wrapped_strategy(strategy, config)).methodology


def serves(
    strategy: SearchStrategy, held: Collection[Methodology], config: Config
) -> bool:
    """Whether ``strategy`` reads one of the ``held`` methodologies under
    ``config`` (one that declares none even through what it wraps is kept),
    and one that needs community reports a namespace holding ``graphrag``."""
    spec = get_strategy_spec(strategy)
    if spec.needs_community_reports and Methodology.GRAPHRAG not in held:
        return False
    methodology = strategy_methodology(strategy, config)
    return methodology is None or methodology in held


# What auto falls back to when no configured strategy reads a held
# methodology: GraphRAG's and LightRAG's general-purpose strategies, then any.
_FALLBACKS = (SearchStrategy.LOCAL, SearchStrategy.MIX)


def routable_strategies(
    config: Config, held: Collection[Methodology] | None = None
) -> tuple[SearchStrategy, ...]:
    """The strategies ``auto`` may pick: ``search.auto_routable_strategies``
    without those whose methodology the queried namespace does not hold
    (``held``; by default the configured effective methodologies,
    ``Config.effective_methodologies``).

    Without communities a namespace holds no ``graphrag``, so routing a query
    to global or drift answered it from nothing. When no configured strategy
    is servable, the first of ``local``, ``mix`` and then any registered
    strategy that is.

    Raises:
        MethodologyNotIndexedError: No registered strategy reads a held
            methodology.
    """
    held = frozenset(config.effective_methodologies if held is None else held)
    configured = config.search.auto_routable_strategies
    routable = tuple(s for s in configured if serves(s, held, config))
    if routable:
        return routable
    candidates = [*_FALLBACKS, *(s for s in SearchStrategy if s not in _FALLBACKS)]
    fallback = next(
        (
            s
            for s in candidates
            if s in _REGISTRY and strategy_methodology(s, config) in held
        ),
        None,
    )
    if fallback is None:
        raise MethodologyNotIndexedError(
            "No search strategy reads what the index namespace holds ("
            + (", ".join(m.value for m in sorted(held)) or "nothing")
            + "): re-run ingestion with indexing.methodologies naming one of "
            + ", ".join(m.value for m in Methodology)
        )
    logger.warning(
        "No strategy in search.auto_routable_strategies (%s) reads what the "
        "index namespace holds (%s); auto uses %s",
        ", ".join(s.value for s in configured),
        ", ".join(m.value for m in sorted(held)),
        fallback.value,
    )
    return (fallback,)
