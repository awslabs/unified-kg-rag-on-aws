# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start-up checks the CLIs run before the first paid model call.

A missing store endpoint otherwise surfaces only when a retriever or indexer
first connects: at query time for ``run-rag``/``run-eval``, and at the final
indexing stage for ``run-ingestion``, after every LLM stage has been paid for.
A missing optional extra surfaces when the first model is built, inside a
stage or the chain, as an unexpected error.
"""

from __future__ import annotations

import importlib.util
from functools import reduce
from typing import Any

from unified_kg_rag.adapters.provider_registry import (
    RETRIEVER_PROVIDER_ROLES,
    ProviderRole,
    provider_endpoint,
    provider_kind,
    provider_requirement,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    get_strategy_spec,
    routable_strategies,
)
from unified_kg_rag.domain.models import Config, RetrieverRole, SearchStrategy


def _role_endpoint(config: Config, role: RetrieverRole) -> tuple[str, str] | None:
    """The endpoint setting of the store kind ``config`` selects for ``role``.

    None for a kind that declares no endpoint (an embedded store, or a custom
    kind that checks its own settings).
    """
    # Importing the module registers the built-in kinds and their endpoints.
    import unified_kg_rag.adapters.providers  # noqa: F401

    provider_role = RETRIEVER_PROVIDER_ROLES[role]
    return provider_endpoint(provider_role, provider_kind(config, provider_role))


def _setting(config: Config, key: str) -> Any:
    """The value at dotted ``key`` (``aws.neptune.endpoint``) of ``config``."""
    return reduce(getattr, key.split("."), config)


def strategy_roles(config: Config, strategy: SearchStrategy) -> set[RetrieverRole]:
    """Retriever roles ``strategy`` can use; ``auto`` covers every routable one."""
    # Importing the package registers the built-in strategies.
    import unified_kg_rag.adapters.search_strategies  # noqa: F401

    strategies = (
        routable_strategies(config) if strategy is SearchStrategy.AUTO else [strategy]
    )
    return {role for s in strategies for role in get_strategy_spec(s).required_roles}


def missing_endpoints_error(
    config: Config, roles: set[RetrieverRole], purpose: str
) -> str | None:
    """Message naming each unset endpoint ``roles`` need, or None if all are set.

    Only roles whose selected store kind declares an endpoint
    (``register_provider(..., endpoint=...)``) are checked.
    """
    missing: list[str] = []
    for role in RETRIEVER_PROVIDER_ROLES:
        endpoint = _role_endpoint(config, role) if role in roles else None
        if endpoint is not None and not _setting(config, endpoint[0]):
            missing.append(f"{endpoint[0]} (env {endpoint[1]})")
    if not missing:
        return None
    return (
        f"Missing endpoint configuration for {purpose}: {', '.join(missing)}. "
        f"Set {'them' if len(missing) > 1 else 'it'} in the config file or "
        "the environment."
    )


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_extras_error(config: Config) -> str | None:
    """Message naming each optional module the selected providers import
    that is not installed, or None if all are.

    Only kinds that declare one (``register_provider(..., requires=...)``)
    are checked.
    """
    # Importing the module registers the built-in kinds and their needs.
    import unified_kg_rag.adapters.providers  # noqa: F401

    # (module, how to install it) -> the settings that select it, each once
    # (two roles share a section: graph indexer and retriever, ...).
    missing: dict[tuple[str, str], dict[str, None]] = {}
    for role in ProviderRole:
        kind = provider_kind(config, role)
        requirement = provider_requirement(role, kind)
        if requirement is not None and not _installed(requirement[0]):
            setting = f"providers.{role.section}.kind '{kind}'"
            missing.setdefault(requirement, {})[setting] = None
    if not missing:
        return None
    modules = ", ".join(
        f"'{module}' (for {', '.join(settings)})"
        for (module, _), settings in missing.items()
    )
    installs = "; ".join(dict.fromkeys(install for _, install in missing))
    return f"Not installed: {modules}. Install {installs}."
