# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start-up checks the CLIs run before the first paid model call.

A missing store endpoint otherwise surfaces only when a retriever or indexer
first connects: at query time for ``run-rag``/``run-eval``, and at the final
indexing stage for ``run-ingestion``, after every LLM stage has been paid for.
"""

from __future__ import annotations

from unified_kg_rag.domain.models import Config, RetrieverRole, SearchStrategy
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec

# role -> (config key, environment variable) of the endpoint the default
# adapter for that role connects to.
_ROLE_ENDPOINTS: dict[RetrieverRole, tuple[str, str]] = {
    RetrieverRole.GRAPH: ("aws.neptune.endpoint", "NEPTUNE_ENDPOINT"),
    RetrieverRole.DOCUMENT: ("aws.opensearch.endpoint", "OPENSEARCH_ENDPOINT"),
}


def _endpoint(config: Config, role: RetrieverRole) -> str | None:
    if role is RetrieverRole.GRAPH:
        return config.aws.neptune.endpoint
    return config.aws.opensearch.endpoint


def strategy_roles(config: Config, strategy: SearchStrategy) -> set[RetrieverRole]:
    """Retriever roles ``strategy`` can use; ``auto`` covers every routable one."""
    # Importing the package registers the built-in strategies.
    import unified_kg_rag.adapters.search_strategies  # noqa: F401

    strategies = (
        config.search.auto_routable_strategies
        if strategy is SearchStrategy.AUTO
        else [strategy]
    )
    return {role for s in strategies for role in get_strategy_spec(s).required_roles}


def missing_endpoints_error(
    config: Config, roles: set[RetrieverRole], purpose: str
) -> str | None:
    """Message naming each unset endpoint ``roles`` need, or None if all are set."""
    missing = [
        f"{key} (env {env})"
        for role, (key, env) in _ROLE_ENDPOINTS.items()
        if role in roles and not _endpoint(config, role)
    ]
    if not missing:
        return None
    return (
        f"Missing endpoint configuration for {purpose}: {', '.join(missing)}. "
        f"Set {'them' if len(missing) > 1 else 'it'} in the config file or "
        "the environment."
    )
