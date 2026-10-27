# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller filters on the keys strategies scope their sub-searches by.

local, DRIFT, mix and hybrid fetch by ``id`` and walk by ``source_id`` /
``target_id``. The scope used to replace a caller value on those keys, so a
``source_id`` filter returned edges of other sources and an ``id`` filter
returned every expanded entity and text unit.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import PipelineStageStatus, SearchStrategy
from unified_kg_rag.shared.exceptions import InvalidFilterError

pytestmark = pytest.mark.integration

_SCOPING = [
    SearchStrategy.LOCAL,
    SearchStrategy.DRIFT,
    SearchStrategy.MIX,
    SearchStrategy.HYBRID,
]


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp_path = tmp_path_factory.mktemp("scoped-filter-keys")
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    (source / "freight.txt").write_text("Carrier moves gadgets for Depot.")
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    o = config.indexing.opensearch
    entities = {
        h["_source"]["name"]: h["_id"]
        for h in e2e._stored(config, providers, o.entities_index_prefix)
    }
    rag = GraphRAGChain(config, providers=providers)
    yield rag, entities
    rag.close()


def _search(chain: GraphRAGChain, strategy: SearchStrategy, filters: dict) -> list:
    return chain.search(
        RAGInput(
            query="Who stores the gadgets for Vendor?",
            search_strategy=strategy,
            filters=filters,
        )
    ).search_results.results


@pytest.mark.parametrize("strategy", _SCOPING, ids=lambda s: s.value)
def test_a_source_id_filter_keeps_only_that_sources_edges(
    stored: Any, strategy: SearchStrategy
) -> None:
    chain, entities = stored
    vendor = entities["Vendor"]
    results = _search(chain, strategy, {"source_id": vendor})
    sources = {(r.metadata or {}).get("source_id") for r in results}
    sources.discard(None)
    # Depot -> Carrier was reached by the walk from Depot.
    assert sources == {vendor}, strategy


@pytest.mark.parametrize("strategy", _SCOPING, ids=lambda s: s.value)
def test_an_id_filter_keeps_only_that_artifact(
    stored: Any, strategy: SearchStrategy
) -> None:
    chain, entities = stored
    vendor = entities["Vendor"]
    results = _search(chain, strategy, {"id": [vendor]})
    ids = {(r.metadata or {}).get("id") or r.chunk_id or r.source for r in results}
    assert ids == {vendor}, strategy


@pytest.mark.parametrize("key", ["id", "source_id", "target_id"])
@pytest.mark.parametrize("strategy", _SCOPING, ids=lambda s: s.value)
def test_a_range_on_a_scoped_key_is_refused(
    stored: Any, strategy: SearchStrategy, key: str
) -> None:
    chain, _ = stored
    with pytest.raises(InvalidFilterError, match=key):
        chain.check_filter_keys({key: {"gte": "a"}}, strategy)
