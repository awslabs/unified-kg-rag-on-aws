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

from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.models import (
    GENERAL_FALLBACKS,
    Config,
    IndexArtifact,
    Methodology,
    RetrieverRole,
    SearchStrategy,
)
from unified_kg_rag.domain.models.methodology import methodologies_holding
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import MethodologyNotIndexedError
from unified_kg_rag.shared.logging import warn_once

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
class RunRecord:
    """The record a strategy keeps of its run in ``SearchResult.metadata``.

    The chain copies it into the answer's metadata, screens its model-written
    text with the output guardrail (masked, or left out when refused, as a
    synthesized result is) and drops it from a refused answer; evaluation
    keeps it per answer (``EvaluationResult.strategy_records``).

    Attributes:
        key: Its ``SearchResult.metadata`` key.
        model_text: Dotted paths of the model-written text in it; a list met
            on the way is walked item by item, and every string under a
            path is screened (``"verdicts.next_queries"``: each verdict's
            follow-up queries).
    """

    key: str
    model_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategySpec:
    """Registration record for a search strategy.

    Attributes:
        strategy_class: The ``BaseSearchStrategy`` subclass implementing the mode.
        required_roles: Retriever ROLES the strategy needs injected (GRAPH /
            DOCUMENT), not concrete backends. The composition root binds each
            role to an adapter, so strategies stay backend-agnostic.
        query_inputs: Query-side extractions the strategy reads.
        wraps: For a strategy that runs another one's search (``iterative``),
            the strategy it wraps under a config. The chain then extracts
            what that one reads and packs the context as that one does
            (``wrapped_strategy``), and the wrapper reads its methodology
            (``strategy_methodology``).
        required_artifacts: The index artifacts the strategy reads, for one
            that reads other than its methodology requires (``naive`` reads
            the text units alone, ``global`` the community reports too): a
            namespace holding them serves it, whatever methodology built
            them. ``None``: its methodology's required artifacts.
        run_record: The record of its run it keeps in the search metadata
            (``RunRecord``), if any.
    """

    strategy_class: type[BaseSearchStrategy]
    required_roles: tuple[RetrieverRole, ...] = field(
        default=(RetrieverRole.DOCUMENT, RetrieverRole.GRAPH)
    )
    query_inputs: frozenset[QueryInput] = frozenset()
    wraps: Callable[[Config], SearchStrategy] | None = None
    required_artifacts: frozenset[IndexArtifact] | None = None
    run_record: RunRecord | None = None

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
    wraps: Callable[[Config], SearchStrategy] | None = None,
    required_artifacts: frozenset[IndexArtifact] | None = None,
    run_record: RunRecord | None = None,
) -> Callable[[StrategyT], StrategyT]:
    """Class decorator that registers a search strategy under ``strategy``.

    Args:
        strategy: The ``SearchStrategy`` enum value this class implements.
        required_roles: Retriever roles to inject when instantiating it.
        query_inputs: Query-side extractions the strategy reads; the chain
            skips the LLM calls for the rest.
        wraps: The strategy it runs the search of, under a config
            (``StrategySpec.wraps``).
        required_artifacts: The index artifacts it reads, when other than
            its methodology requires (``StrategySpec.required_artifacts``).
        run_record: The record of its run it keeps in the search metadata
            and where its model-written text is (``RunRecord``): the chain
            screens and clears it, evaluation keeps it. A strategy that
            writes model text into its metadata must declare it, or the
            text skips the output guardrail.

    Re-registering the same class with the same arguments keeps the record
    it has; with other arguments it raises, because replacing the record
    would change, for every caller in the process, what the strategy reads.

    Raises:
        ValueError: If ``strategy`` is already registered to another class,
            or to this class with other arguments.
    """

    def decorator(cls: StrategyT) -> StrategyT:
        spec = StrategySpec(
            strategy_class=cls,
            required_roles=tuple(required_roles),
            query_inputs=frozenset(query_inputs),
            wraps=wraps,
            required_artifacts=(
                None if required_artifacts is None else frozenset(required_artifacts)
            ),
            run_record=run_record,
        )
        if run_record is not None and any(
            other.run_record is not None
            and other.run_record.key == run_record.key
            and other.strategy_class is not cls
            for other in _REGISTRY.values()
        ):
            raise ValueError(
                f"Search strategy '{strategy.value}': another strategy keeps "
                f"its run record under '{run_record.key}'."
            )
        registered = _REGISTRY.get(strategy)
        if registered is not None:
            if registered.strategy_class is not cls:
                raise ValueError(
                    f"Search strategy '{strategy.value}' is already registered to "
                    f"'{registered.strategy_class.__name__}'; cannot re-register "
                    f"to '{cls.__name__}'."
                )
            if registered != spec:
                raise ValueError(
                    f"Search strategy '{strategy.value}' is already registered to "
                    f"'{cls.__name__}' with other arguments; cannot change them."
                )
            return cls
        _REGISTRY[strategy] = spec
        logger.debug(
            "Registered search strategy '%s' -> %s", strategy.value, cls.__name__
        )
        return cls

    return decorator


def _register_built_in_strategies() -> None:
    """Import the built-in strategies, which registers them (once per process).

    A lookup reads the registry, and the classes fill it when their package
    is imported: an entry point that imported none (``run-describe``)
    found no strategy at all. The import is deferred because that package
    imports this module.
    """
    import unified_kg_rag.adapters.search_strategies  # noqa: F401


def run_records() -> tuple[RunRecord, ...]:
    """The run record of every registered strategy that keeps one
    (``StrategySpec.run_record``), each key once."""
    _register_built_in_strategies()
    return tuple(
        spec.run_record for spec in _REGISTRY.values() if spec.run_record is not None
    )


def get_strategy_spec(strategy: SearchStrategy) -> StrategySpec:
    """Look up the registration record for ``strategy``.

    Raises:
        ValueError: If no strategy is registered for ``strategy``.
    """
    _register_built_in_strategies()
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


def citations_enabled(config: Config, strategy: SearchStrategy | str | None) -> bool:
    """Whether the answers of ``strategy`` (the one a query ran: ``auto``'s
    route) cite under ``config``: ``search.citations.strategies``' entry for
    it, else, for a wrapper (``iterative``), its base's, else
    ``search.citations.enabled``. A strategy that is unknown or not given
    (a result saved without one) reads ``enabled``."""
    settings = config.search.citations
    try:
        resolved = SearchStrategy(strategy) if strategy is not None else None
    except ValueError:
        resolved = None
    if resolved is None or resolved is SearchStrategy.AUTO:
        return settings.enabled
    base = wrapped_strategy(resolved, config)
    return settings.enabled_for(resolved, base if base is not resolved else None)


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


def strategy_artifacts(
    strategy: SearchStrategy, config: Config
) -> frozenset[IndexArtifact] | None:
    """The index artifacts ``strategy`` declares it reads under ``config``
    (``StrategySpec.required_artifacts``; for a wrapper, the strategy it
    wraps), or ``None``: its methodology's required ones."""
    own = get_strategy_spec(strategy).required_artifacts
    if own is not None:
        return own
    return get_strategy_spec(wrapped_strategy(strategy, config)).required_artifacts


def serves(
    strategy: SearchStrategy, artifacts: Collection[IndexArtifact], config: Config
) -> bool:
    """Whether a namespace holding ``artifacts`` serves ``strategy`` under
    ``config``: it holds the artifacts the strategy declares it reads
    (``strategy_artifacts``), else the strategy's methodology (one that
    declares none even through what it wraps is kept)."""
    held = frozenset(artifacts)
    needed = strategy_artifacts(strategy, config)
    if needed is not None:
        return needed <= held
    methodology = strategy_methodology(strategy, config)
    return methodology is None or methodology in methodologies_holding(held)


def routable_strategies(
    config: Config, artifacts: Collection[IndexArtifact] | None = None
) -> tuple[SearchStrategy, ...]:
    """The strategies ``auto`` may pick: ``search.auto_routable_strategies``
    without those the queried namespace does not serve (``serves`` the
    ``artifacts`` it holds; by default those a run of ``config`` builds,
    ``artifacts_to_build``).

    Without community reports a namespace serves no global or drift, so
    routing a query to them answered it from nothing. When no configured
    strategy is servable, the first of ``local``, ``mix`` and then any
    registered strategy that is.

    Raises:
        MethodologyNotIndexedError: No registered strategy reads a held
            methodology.
    """
    _register_built_in_strategies()
    holding = frozenset(artifacts_to_build(config) if artifacts is None else artifacts)
    held = methodologies_holding(holding)
    configured = config.search.auto_routable_strategies
    routable = tuple(s for s in configured if serves(s, holding, config))
    if routable:
        return routable
    # The general-purpose strategies first (GENERAL_FALLBACKS), then any.
    candidates = [
        *GENERAL_FALLBACKS,
        *(s for s in SearchStrategy if s not in GENERAL_FALLBACKS),
    ]
    fallback = next(
        (
            s
            for s in candidates
            if s in _REGISTRY
            and strategy_methodology(s, config) in held
            and serves(s, holding, config)
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
    # Once per configuration and namespace holdings, not on every query.
    warn_once(
        logger,
        ("auto_fallback", tuple(configured), tuple(sorted(held)), fallback),
        "No strategy in search.auto_routable_strategies (%s) reads what the "
        "index namespace holds (%s); auto uses %s",
        ", ".join(s.value for s in configured),
        ", ".join(m.value for m in sorted(held)),
        fallback.value,
        then_debug=True,
    )
    return (fallback,)
