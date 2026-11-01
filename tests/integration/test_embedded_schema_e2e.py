# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A strict extraction schema end to end on the embedded stores (no AWS).

The fake model answers a synthetic typed corpus with declared types, other
spellings of them, undeclared and empty types, an undeclared relationship
type and a reversed pattern, and gleaning proposes an entity of an
undeclared type. The full ingest stores only what the schema allows, under
the declared labels; an incremental run (a document added, one changed, one
deleted) leaves the graph a fresh full build of the final corpus produces.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import Config, PipelineStageStatus, SearchStrategy

pytestmark = pytest.mark.integration

# Name -> the type the model gives it: declared, another spelling of a
# declared one, undeclared, empty.
TYPES = {
    "Vendor": "ORGANIZATION",
    "Buyer": "organization",
    "Gadget": "Product",
    "Depot": "warehouse",
    "Carrier": "",
}
SCHEMA = {
    "schema_mode": "strict",
    "entity_types": ["ORGANIZATION: Companies", "PRODUCT: Goods"],
    "relationship_types": ["SUPPLIES: Sells to", "STOCKS: Holds goods"],
    "relationship_patterns": [
        "ORGANIZATION -SUPPLIES-> ORGANIZATION",
        "ORGANIZATION -STOCKS-> PRODUCT",
    ],
}


def _relationship(source: str, target: str, type_: str, text: str) -> str:
    return (
        f"<relationship><source>{source}</source><target>{target}</target>"
        f"<type>{type_}</type><description>{source} {type_} {target}"
        f"</description><strength>7</strength><source_text>{text}</source_text>"
        "</relationship>"
    )


def _extraction(text: str) -> str:
    names = [name for name in TYPES if name in text]
    entities = "".join(
        f"<entity><name>{n}</name><type>{TYPES[n]}</type>"
        f"<description>{n} in the supply chain</description>"
        f"<confidence>9</confidence><source_text>{n}</source_text></entity>"
        for n in names
    )
    relationships = []
    for source in names:
        for target in names:
            if source == target:
                continue
            if target == "Gadget":
                # Organization -> product, as declared: kept (unless the
                # source is dropped); product -> organization: reversed.
                relationships.append(_relationship(source, target, "stocks", text))
                relationships.append(_relationship(target, source, "STOCKS", text))
            elif "Gadget" not in (source, target):
                relationships.append(_relationship(source, target, "Supplies", text))
                relationships.append(_relationship(source, target, "mentions", text))
    return (
        f"<entities>{entities}</entities>"
        f"<relationships>{''.join(relationships)}</relationships>"
    )


_REFINEMENT = (
    "<refinement_plan><identified_issues><issue>"
    "<issue_type>MISSING_ENTITY</issue_type><details><name>Ledger</name>"
    "<type>document</type><description>A ledger</description></details>"
    "<text_evidence>Ledger</text_evidence></issue>"
    "<issue><issue_type>MISSING_RELATIONSHIP</issue_type><details>"
    "<source>Vendor</source><target>Ledger</target><type>SUPPLIES</type>"
    "<description>Vendor keeps a ledger</description><strength>3</strength>"
    "</details><text_evidence>Ledger</text_evidence></issue>"
    "</identified_issues></refinement_plan>"
)


class _Model:
    def __init__(self) -> None:
        self.extraction_prompts: list[str] = []

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            self.extraction_prompts.append(system)
            return _extraction(stages._source_text(human))
        if "graph refinement specialist" in system:
            return _REFINEMENT
        return e2e._respond(system, human)


def _config(
    stores: Path, viz: Path, change: Callable[[Config], None] | None = None
) -> Config:
    config = e2e._config(stores, viz)
    for key, value in SCHEMA.items():
        setattr(config.processing.graph_extraction, key, value)
    config.processing.gleaning.enabled = True
    config.processing.gleaning.max_rounds = 1
    if change is not None:
        change(config)
    return config


def _graph(providers: Providers) -> tuple[dict[str, str], set[tuple[str, str, str]]]:
    """The stored entities (name -> type) and edges (source, type, target)."""
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        label = graph._entity_label(None)
        with graph.store.read("test") as connection:
            edge_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM edges WHERE label = ?", (label,)
                )
            ]
        names = dict(graph.read_entity_names())
        entities = graph.read_entities(sorted(names))
        edges = graph.read_relationships(sorted(edge_ids))
        return (
            {e.name: e.type or "" for e in entities},
            {(r.source_name or "", r.type or "", r.target_name or "") for r in edges},
        )
    finally:
        graph.close()


def _run(
    tmp_path: Path,
    name: str,
    corpus: dict[str, str],
    change: Callable[[Config], None] | None = None,
):
    """A fresh store set with one full ingest of ``corpus`` (the schema
    config, then ``change``)."""
    config = _config(tmp_path / name / "stores", tmp_path / name / "viz", change)
    model = _Model()
    providers = e2e._providers(
        config, FakeOllama(model, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / name / "corpus"
    source.mkdir(parents=True)
    for file_name, text in corpus.items():
        (source / file_name).write_text(text)
    context = e2e._ingest(config, providers, source, tmp_path / name / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    return config, providers, model, source


FIRST = {
    "contract.txt": "Vendor supplies Buyer.",
    "logistics.txt": "Depot stocks Gadget for Vendor.",
}
FINAL = {
    "logistics.txt": "Vendor stocks Gadget and Ledger.",
    "freight.txt": "Carrier supplies Buyer and Vendor.",
}


def test_strict_schema_end_to_end_and_incremental_matches_full(
    tmp_path: Path, no_aws  # noqa: F811
) -> None:
    config, providers, model, source = _run(tmp_path, "inc", FIRST)
    # The prompt lists the declared types and patterns.
    assert model.extraction_prompts
    prompt = model.extraction_prompts[0]
    assert "## Relationship Types (STRICT - use only these types):" in prompt
    assert "- **SUPPLIES**: Sells to" in prompt
    assert "- (ORGANIZATION)-[STOCKS]->(PRODUCT)" in prompt
    assert "HIERARCHICAL" not in prompt

    entities, edges = _graph(providers)
    # Depot (warehouse) is off schema: dropped with its edges; types are the
    # merged (lowercased) declared ones.
    assert entities == {
        "Vendor": "organization",
        "Buyer": "organization",
        "Gadget": "product",
    }
    # Declared labels only; "mentions" and the reversed STOCKS are gone, and
    # gleaning's Ledger (document) took its edge with it.
    assert edges == {
        ("Vendor", "SUPPLIES", "Buyer"),
        ("Buyer", "SUPPLIES", "Vendor"),
        ("Vendor", "STOCKS", "Gadget"),
    }

    # Incremental: logistics changed, freight added, contract deleted.
    (source / "contract.txt").unlink()
    for file_name, text in FINAL.items():
        (source / file_name).write_text(text)
    context = e2e._ingest(config, providers, source, tmp_path / "inc" / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    delta = context.incremental_delta
    assert delta is not None
    assert (len(delta.new), len(delta.changed), len(delta.deleted)) == (1, 1, 1)
    incremental = _graph(providers)

    _, full_providers, _, _ = _run(tmp_path, "full", FINAL)
    assert incremental == _graph(full_providers)
    entities, edges = incremental
    # Carrier is untyped (off schema); Ledger an undeclared type.
    assert set(entities) == {"Vendor", "Buyer", "Gadget"}
    assert all(not re.search(r"[a-z]", rel_type) for _, rel_type, _ in edges)
    assert ("Vendor", "STOCKS", "Gadget") in edges


ORG_IRI = "https://example.org/onto#Organization"
SUPPLIES_IRI = "https://example.org/onto#supplies"


def test_align_stores_the_annotations_and_in_schema_filters_entities(
    tmp_path: Path, no_aws  # noqa: F811
) -> None:
    def align(config: Config) -> None:
        extraction = config.processing.graph_extraction
        extraction.schema_mode = "align"
        extraction.entity_types = [
            {"label": "ORGANIZATION", "description": "Companies", "iri": ORG_IRI},
            "PRODUCT: Goods",
        ]
        extraction.relationship_types = [
            {"label": "SUPPLIES", "description": "Sells to", "iri": SUPPLIES_IRI},
            "STOCKS: Holds goods",
        ]

    config, providers, _, _ = _run(tmp_path, "align", FIRST, change=align)
    graph = providers.build_graph_indexer()
    try:
        names = dict(graph.read_entity_names())
        stored = {
            e.name: (e.in_schema, e.type_iri)
            for e in graph.read_entities(sorted(names))
        }
    finally:
        graph.close()
    # align keeps everything: Depot (warehouse) and gleaning's Ledger
    # (document) are off schema.
    assert stored == {
        "Vendor": (True, ORG_IRI),
        "Buyer": (True, ORG_IRI),
        "Gadget": (True, None),
        "Depot": (False, None),
        "Ledger": (False, None),
    }
    documents = e2e._stored(
        config, providers, config.indexing.opensearch.entities_index_prefix
    )
    assert {h["_source"]["name"]: h["_source"].get("in_schema") for h in documents} == {
        name: in_schema for name, (in_schema, _) in stored.items()
    }

    off_schema = {name for name, (in_schema, _) in stored.items() if not in_schema}

    def search(strategy: SearchStrategy, filters: dict | None) -> list:
        return rag.search(
            RAGInput(
                query="Who stocks the gadgets?",
                search_strategy=strategy,
                filters=filters,
            )
        ).search_results.results

    rag = GraphRAGChain(config, providers=providers)
    try:
        for strategy in (SearchStrategy.LOCAL, SearchStrategy.MIX):
            unfiltered = {
                r.metadata.get("name")
                for r in search(strategy, None)
                if r.retriever_type == "entity"
            }
            assert unfiltered - off_schema, strategy  # in-schema entities too
            results = search(strategy, {"in_schema": False})
            entities = {
                r.metadata.get("name") for r in results if r.retriever_type == "entity"
            }
            # Only off-schema entities (vector search and graph expansion);
            # the relationships index applies the filter too.
            assert entities and entities <= off_schema, (strategy, entities)
            assert all(
                r.metadata.get("in_schema") is False
                for r in results
                if r.retriever_type == "relationship"
            ), strategy
    finally:
        rag.close()
