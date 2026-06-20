# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Registry for search strategies.

Replaces the previous hardcoded ``strategy_map`` dict and ``if SIMPLE/else``
retriever wiring in :mod:`aws_graphrag.retrieval.rag_chain` with a declarative,
enum-keyed registry. New strategies register themselves with
``@register_strategy`` and declare which retrievers they need, so adding a
strategy no longer requires editing dispatch code.

This mirrors the declarative-factory pattern already used by
``ParserFactory._loader_configs`` and ``EvaluationManager.EVALUATOR_MAPPING``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from aws_graphrag.core import get_logger
from aws_graphrag.domain.models import RetrieverRole, SearchStrategy

if TYPE_CHECKING:
    from aws_graphrag.retrieval.base import BaseSearchStrategy

logger = get_logger(__name__)


@dataclass(frozen=True)
class StrategySpec:
    """Registration record for a search strategy.

    Attributes:
        strategy_class: The ``BaseSearchStrategy`` subclass implementing the mode.
        required_roles: Retriever ROLES the strategy needs injected (GRAPH /
            DOCUMENT), not concrete backends. The composition root binds each
            role to an adapter, so strategies stay backend-agnostic.
    """

    strategy_class: type[BaseSearchStrategy]
    required_roles: tuple[RetrieverRole, ...] = field(
        default=(RetrieverRole.DOCUMENT, RetrieverRole.GRAPH)
    )


_REGISTRY: dict[SearchStrategy, StrategySpec] = {}

StrategyT = TypeVar("StrategyT", bound="type[BaseSearchStrategy]")


def register_strategy(
    strategy: SearchStrategy,
    *,
    required_roles: tuple[RetrieverRole, ...] = (
        RetrieverRole.DOCUMENT,
        RetrieverRole.GRAPH,
    ),
) -> Callable[[StrategyT], StrategyT]:
    """Class decorator that registers a search strategy under ``strategy``.

    Args:
        strategy: The ``SearchStrategy`` enum value this class implements.
        required_roles: Retriever roles to inject when instantiating it.

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
            strategy_class=cls, required_roles=tuple(required_roles)
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


def registered_strategies() -> tuple[SearchStrategy, ...]:
    """Return the strategies currently registered (registration order)."""
    return tuple(_REGISTRY)
