# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What an index namespace holds is computed by one rule
(``domain.ingestion.methodologies.namespace_holdings``) for the chain, the
console, ``run-describe`` and the MCP server; copies of it disagreed (the
console showed nothing held for a namespace without a build record, while
every strategy the config builds was served)."""

from __future__ import annotations

import pytest

from unified_kg_rag.application.retrieval.methodology_check import (
    summarize_namespace,
)
from unified_kg_rag.domain.ingestion.methodologies import (
    artifacts_to_build,
    namespace_holdings,
)
from unified_kg_rag.domain.models import Config, IndexArtifact, Methodology

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("record", [None, {"mode": "full"}, "not a record"])
def test_without_a_record_naming_artifacts_the_config_stands_in(record: object) -> None:
    config = Config()
    holdings = namespace_holdings(record, config)  # type: ignore[arg-type]
    assert holdings.source == "config"
    assert holdings.artifacts == artifacts_to_build(config)
    assert Methodology.GRAPHRAG in holdings.methodologies
    assert holdings.scopes is None


def test_a_record_naming_artifacts_decides() -> None:
    record = {
        "artifacts": [IndexArtifact.TEXT_UNITS.value, IndexArtifact.SUMMARY_TREE.value],
        "source_scopes": ["corpus-a", 3],
    }
    holdings = namespace_holdings(record, Config())
    assert holdings.source == "build_record"
    assert holdings.methodologies == (Methodology.RAPTOR,)
    assert holdings.scopes == ["corpus-a"]
    assert holdings.record == record


def test_the_summary_tells_served_strategies_from_auto_s_routes() -> None:
    # Every strategy the namespace serves can be named; auto routes only
    # among search.auto_routable_strategies it serves.
    config = Config()
    summary = summarize_namespace(namespace_holdings(None, config), config)
    assert summary["artifacts_from"] == "config"
    assert "auto" not in summary["strategies"]
    assert summary["auto_routes"] == [
        s.value for s in config.search.auto_routable_strategies
    ]
    assert set(summary["auto_routes"]) < set(summary["strategies"])
