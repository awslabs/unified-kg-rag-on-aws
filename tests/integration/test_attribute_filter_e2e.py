# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document attribute filters on every strategy, on the embedded stores.

Relationships, claims and graph vertices carry no document attribute, so an
``attr_<key>`` filter used to leave them unfiltered: local, mix, hybrid, DRIFT
and simple returned relationships from documents the filter excludes, even for
a filter no document matches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.adapters.ingestion.chunker import BaseChunker
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import PipelineStageStatus, SearchStrategy

pytestmark = pytest.mark.integration

_STRATEGIES = [s for s in SearchStrategy if s is not SearchStrategy.AUTO]


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp_path = tmp_path_factory.mktemp("attribute-filters")
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")

    create = BaseChunker._create_text_unit

    def with_category(
        self: BaseChunker, chunk_text: str, document: Any, *a: Any
    ) -> Any:
        # A document's ``filters`` metadata (here its file's stem).
        document.metadata["filters"] = {"category": Path(document.file_name).stem}
        return create(self, chunk_text, document, *a)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(BaseChunker, "_create_text_unit", with_category)
        context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    rag = GraphRAGChain(config, providers=providers)
    yield rag
    rag.close()


def _ask(chain: GraphRAGChain, strategy: SearchStrategy, filters: dict) -> Any:
    return chain.invoke(
        RAGInput(
            query="Who stores the gadgets for Vendor?",
            search_strategy=strategy,
            filters=filters,
        )
    )


@pytest.mark.parametrize("key", ["attr_category", "attributes.filters.category"])
@pytest.mark.parametrize("strategy", _STRATEGIES, ids=lambda s: s.value)
def test_a_filter_no_document_matches_returns_nothing(
    chain: GraphRAGChain, strategy: SearchStrategy, key: str
) -> None:
    out = _ask(chain, strategy, {key: "nomatch"})
    assert out.sources == [], strategy


# Community reports cite the text units of all their members, so a report
# passes when one member's document matches (as an entity does), and the
# fake model's syntheses quote the canned answer: only the artifacts one
# document vouches for are checked.
_SINGLE_SOURCE = {"text", "entity", "relationship", "claim"}


def _depot(out: Any) -> list[str]:
    return [
        r.content
        for r in out.search_results.results
        if r.retriever_type in _SINGLE_SOURCE and "Depot" in r.content
    ]


@pytest.mark.parametrize("strategy", _STRATEGIES, ids=lambda s: s.value)
def test_a_filter_keeps_only_the_matching_documents_content(
    chain: GraphRAGChain, strategy: SearchStrategy
) -> None:
    # Depot occurs only in logistics.txt; Vendor occurs in both documents.
    filtered = _ask(chain, strategy, {"attr_category": "contract"})
    assert not _depot(filtered), strategy
    if strategy is not SearchStrategy.GLOBAL:
        assert _depot(_ask(chain, strategy, {})), strategy
        assert filtered.search_results.results, strategy


def _stored_relationship_filters(config: Any, providers: Any) -> dict[str, Any]:
    prefix = config.indexing.opensearch.relationships_index_prefix
    return {
        f"{h['_source']['source_name']}-{h['_source']['target_name']}": h[
            "_source"
        ].get("attr_category")
        for h in e2e._stored(config, providers, prefix)
    }


def test_incremental_runs_keep_relationship_filters_current(
    tmp_path: Path,
) -> None:
    # The relationship index carries its documents' filters, so the store
    # applies an attribute filter itself: an edge both documents state
    # carries both values, and one document's removal narrows them.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Vendor ships widgets for Buyer.")
    create = BaseChunker._create_text_unit

    def with_category(
        self: BaseChunker, chunk_text: str, document: Any, *a: Any
    ) -> Any:
        document.metadata["filters"] = {"category": Path(document.file_name).stem}
        return create(self, chunk_text, document, *a)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(BaseChunker, "_create_text_unit", with_category)
        context = e2e._ingest(config, providers, source, tmp_path / "run-1")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        assert _stored_relationship_filters(config, providers) == {
            "Vendor-Buyer": ["contract", "logistics"]
        }

        (source / "logistics.txt").unlink()
        context = e2e._ingest(config, providers, source, tmp_path / "run-2")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _stored_relationship_filters(config, providers) == {
        "Vendor-Buyer": "contract"
    }
    rag = GraphRAGChain(config, providers=providers)
    try:
        for strategy in (SearchStrategy.LOCAL, SearchStrategy.MIX):
            out = _ask(rag, strategy, {"attr_category": "logistics"})
            assert out.sources == [], strategy
    finally:
        rag.close()
