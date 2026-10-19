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


def _stored_filters(config: Any, providers: Any) -> dict[str, Any]:
    """Each stored entity, relationship and claim's document filters (both
    fields; a null field matches no filter, so it counts as absent)."""
    o = config.indexing.opensearch
    stored: dict[str, Any] = {}
    for kind, prefix, name in (
        ("entity", o.entities_index_prefix, ("name",)),
        ("relationship", o.relationships_index_prefix, ("source_name", "target_name")),
        ("claim", o.claims_index_prefix, ("subject_name", "object_name")),
    ):
        for hit in e2e._stored(config, providers, prefix):
            source = hit["_source"]
            label = "-".join(str(source.get(n)) for n in name)
            attr = {
                k: v
                for k, v in source.items()
                if k.startswith("attr_") and v is not None
            }
            nested = {
                k: v
                for k, v in (
                    (source.get("attributes") or {}).get("filters") or {}
                ).items()
                if v is not None
            }
            assert attr == {f"attr_{k}": v for k, v in nested.items()}, (kind, label)
            stored[f"{kind} {label}"] = nested
    return stored


# One relationship and one claim (Acme -> Buyer) both documents state.
_CONTRACT = "Acme supplies Buyer."
_LOGISTICS = "Acme ships to Buyer."


@pytest.mark.parametrize(
    ("kept", "after"),
    [
        ({"category": "contract"}, {"category": "contract"}),  # a key dropped
        ({}, {}),  # no filters left
    ],
    ids=["key-dropped", "no-filters-left"],
)
def test_incremental_runs_keep_relationship_filters_current(
    tmp_path: Path, kept: dict[str, Any], after: dict[str, Any]
) -> None:
    # The entity, relationship and claim indexes carry their documents'
    # filters, so the store applies an attribute filter itself: an item both
    # documents state carries both values, a document's removal narrows them
    # (a lineage update, which must clear the keys only it had), and a claim
    # merged with an unfiltered document's keeps the filtered one's.
    from tests.integration import test_claim_convergence as claims

    def respond(system: str, human: str) -> str:
        # Extraction with claims, as the claim convergence tests; query-time
        # prompts as the embedded end-to-end test.
        if "knowledge graph extraction expert" in system or (
            "claim extraction specialist" in system
        ):
            return claims._respond(system, human)
        return e2e._respond(system, human)

    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.claim_extraction.enabled = True
    server = FakeOllama(respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    per_file = {
        "contract": kept,
        "logistics": {"category": "logistics", "clearance": "secret"},
    }
    create = BaseChunker._create_text_unit

    def with_filters(self: BaseChunker, chunk_text: str, document: Any, *a: Any) -> Any:
        if filters := per_file[Path(document.file_name).stem]:
            document.metadata["filters"] = dict(filters)
        return create(self, chunk_text, document, *a)

    labels = (
        "entity Acme",
        "entity Buyer",
        "relationship Acme-Buyer",
        "claim Acme-Buyer",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(BaseChunker, "_create_text_unit", with_filters)
        (source / "logistics.txt").write_text(_LOGISTICS)
        context = e2e._ingest(config, providers, source, tmp_path / "run-1")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        # The other document joins: a claim merged into the stored one keeps
        # its filters (they were read back without them).
        (source / "contract.txt").write_text(_CONTRACT)
        context = e2e._ingest(config, providers, source, tmp_path / "run-2")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        both = {
            k: sorted(
                {*([v] if isinstance(v, str) else v), *([kept[k]] if k in kept else [])}
            )
            for k, v in per_file["logistics"].items()
        }
        both = {k: v[0] if len(v) == 1 else v for k, v in both.items()}
        stored = _stored_filters(config, providers)
        assert {label: stored[label] for label in labels} == dict.fromkeys(labels, both)

        (source / "logistics.txt").unlink()
        context = e2e._ingest(config, providers, source, tmp_path / "run-3")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    stored = _stored_filters(config, providers)
    assert {label: stored[label] for label in labels} == dict.fromkeys(labels, after)
    rag = GraphRAGChain(config, providers=providers)
    try:
        for strategy in (SearchStrategy.LOCAL, SearchStrategy.MIX):
            for filters in (
                {"attr_category": "logistics"},
                {"attr_clearance": "secret"},
            ):
                out = _ask(rag, strategy, filters)
                assert out.sources == [], (strategy, filters)
    finally:
        rag.close()


@pytest.mark.parametrize(
    "kept", [{"category": "contract"}, {}], ids=["other-filters", "no-filters"]
)
def test_a_removed_documents_filter_keys_stop_matching(
    tmp_path: Path, kept: dict[str, Any]
) -> None:
    # Regression: the lineage update after a removal is a partial update,
    # which kept the removed document's ``clearance`` on the entities and
    # the relationship, so a ``clearance:secret`` filter still returned the
    # remaining document's content.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Vendor ships widgets for Buyer.")
    per_file = {
        "contract": kept,
        "logistics": {"category": "logistics", "clearance": "secret"},
    }
    create = BaseChunker._create_text_unit

    def with_filters(self: BaseChunker, chunk_text: str, document: Any, *a: Any) -> Any:
        if filters := per_file[Path(document.file_name).stem]:
            document.metadata["filters"] = dict(filters)
        return create(self, chunk_text, document, *a)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(BaseChunker, "_create_text_unit", with_filters)
        context = e2e._ingest(config, providers, source, tmp_path / "run-1")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        (source / "logistics.txt").unlink()
        context = e2e._ingest(config, providers, source, tmp_path / "run-2")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    rag = GraphRAGChain(config, providers=providers)
    try:
        for strategy in (SearchStrategy.LOCAL, SearchStrategy.MIX):
            for key in ("attr_clearance", "attributes.filters.clearance"):
                assert _ask(rag, strategy, {key: "secret"}).sources == [], strategy
    finally:
        rag.close()
