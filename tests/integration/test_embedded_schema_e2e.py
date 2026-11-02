# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A strict extraction schema end to end on the embedded stores (no AWS).

The fake model answers a synthetic typed corpus with declared types, other
spellings of them, undeclared and empty types, an undeclared relationship
type and a reversed pattern, and gleaning proposes an entity of an
undeclared type. The full ingest stores only what the schema allows, under
the declared labels; an incremental run (a document added, one changed, one
deleted) leaves the graph a fresh full build of the final corpus produces.
The schema filter test also runs on the local containers (``LOCAL_STORES=1``).
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.extras import requires_ontology
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.schema_annotation import schema_annotation_digest
from tests.fixtures.store_backends import clear_extraction_records, local_store_config
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.document_identity import (
    config_registry_namespace,
)
from unified_kg_rag.domain.ingestion.schema_annotation import REANNOTATING_DIGEST
from unified_kg_rag.domain.models import Config, PipelineStageStatus, SearchStrategy
from unified_kg_rag.shared.exceptions import InvalidFilterError

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
    stores: Path,
    viz: Path,
    change: Callable[[Config], None] | None = None,
    schema: dict[str, Any] = SCHEMA,
) -> Config:
    config = e2e._config(stores, viz)
    for key, value in schema.items():
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
    schema: dict[str, Any] = SCHEMA,
):
    """A fresh store set with one full ingest of ``corpus`` (the ``schema``
    settings, then ``change``)."""
    config = _config(
        tmp_path / name / "stores", tmp_path / name / "viz", change, schema
    )
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

# SCHEMA as an ontology: the same labels, descriptions and patterns, with
# IRIs.
ONTOLOGY = """\
@prefix ex: <https://example.org/onto#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:Organization a owl:Class ; rdfs:comment "Companies"@en .
ex:Product a owl:Class ; rdfs:comment "Goods"@en .
ex:supplies a owl:ObjectProperty ; rdfs:comment "Sells to"@en ;
    rdfs:domain ex:Organization ; rdfs:range ex:Organization .
ex:stocks a owl:ObjectProperty ; rdfs:comment "Holds goods"@en ;
    rdfs:domain ex:Organization ; rdfs:range ex:Product .
ex:name a owl:DatatypeProperty ; rdfs:domain ex:Organization .
"""


@requires_ontology
def test_an_owl_schema_source_end_to_end(tmp_path: Path, no_aws) -> None:  # noqa: F811
    ontology = tmp_path / "onto.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")

    schema = {
        "schema_mode": "strict",
        "schema_source": {"kind": "owl", "path": str(ontology)},
    }
    _, providers, model, _ = _run(tmp_path, "owl", FIRST, schema=schema)
    prompt = model.extraction_prompts[0]
    assert "- **SUPPLIES**: Sells to" in prompt
    assert "- (ORGANIZATION)-[STOCKS]->(PRODUCT)" in prompt
    entities, edges = _graph(providers)
    # What the inline strict schema stores.
    assert entities == {
        "Vendor": "organization",
        "Buyer": "organization",
        "Gadget": "product",
    }
    assert edges == {
        ("Vendor", "SUPPLIES", "Buyer"),
        ("Buyer", "SUPPLIES", "Vendor"),
        ("Vendor", "STOCKS", "Gadget"),
    }
    graph = providers.build_graph_indexer()
    try:
        names = dict(graph.read_entity_names())
        iris = {e.name: e.type_iri for e in graph.read_entities(sorted(names))}
    finally:
        graph.close()
    # With the ontology's IRIs.
    assert iris == {
        "Vendor": ORG_IRI,
        "Buyer": ORG_IRI,
        "Gadget": "https://example.org/onto#Product",
    }


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


def test_align_with_entity_types_alone_stores_declared_iris_only(
    tmp_path: Path, no_aws  # noqa: F811
) -> None:
    # No relationship type declared: extraction keeps the records as written
    # and no item gets in_schema, but a declared class IRI is still stored.
    def entity_types_only(config: Config) -> None:
        extraction = config.processing.graph_extraction
        extraction.schema_mode = "align"
        extraction.relationship_patterns = []
        extraction.relationship_types = []
        extraction.entity_types = [
            {"label": "ORGANIZATION", "description": "Companies", "iri": ORG_IRI},
            "PRODUCT: Goods",
        ]

    config, providers, _, _ = _run(tmp_path, "iri", FIRST, change=entity_types_only)
    graph = providers.build_graph_indexer()
    try:
        names = dict(graph.read_entity_names())
        stored = {
            e.name: (e.in_schema, e.type_iri)
            for e in graph.read_entities(sorted(names))
        }
        assert isinstance(graph, SQLiteGraphIndexer)
        label = graph._entity_label(None)
        with graph.store.read("test") as connection:
            edge_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM edges WHERE label = ?", (label,)
                )
            ]
        edges = graph.read_relationships(sorted(edge_ids))
    finally:
        graph.close()
    assert stored == {
        "Vendor": (None, ORG_IRI),
        "Buyer": (None, ORG_IRI),
        "Gadget": (None, None),
        "Depot": (None, None),
        "Ledger": (None, None),
    }
    assert edges and all((r.in_schema, r.predicate_iri) == (None, None) for r in edges)
    documents = e2e._stored(
        config, providers, config.indexing.opensearch.entities_index_prefix
    )
    by_name = {h["_source"]["name"]: h["_source"] for h in documents}
    assert {name: source.get("type_iri") for name, source in by_name.items()} == {
        name: iri for name, (_, iri) in stored.items()
    }
    assert not any("in_schema" in source for source in by_name.values())


def _on_containers(config: Config) -> None:
    """``config`` on the Gremlin and OpenSearch containers, in a namespace of
    its own."""
    local = local_store_config()
    config.providers.graph.kind = "gremlin"
    config.providers.vector.kind = "opensearch"
    config.aws.opensearch = local.aws.opensearch
    config.aws.neptune = local.aws.neptune
    config.indexing.neptune.vertex_ids = local.indexing.neptune.vertex_ids
    config.indexing.additional_suffix = f"sf{uuid.uuid4().hex[:8]}"


@pytest.mark.parametrize("stores", ["embedded", "container"])
def test_schema_filters_never_empty_a_stream_silently(
    tmp_path: Path, no_aws, stores: str  # noqa: F811
) -> None:
    # strict with entity types alone: relationships had no in_schema, so
    # in_schema: true dropped every relationship (mix, hybrid, local, drift,
    # auto); and no type declares an IRI, so an IRI filter emptied its kind.
    if stores == "container" and os.getenv("LOCAL_STORES") != "1":
        pytest.skip("set LOCAL_STORES=1 with docker/compose.local.yaml running")

    def entity_types_only(config: Config) -> None:
        extraction = config.processing.graph_extraction
        extraction.relationship_patterns = []
        extraction.relationship_types = []
        if stores == "container":
            _on_containers(config)

    config, providers, _, _ = _run(tmp_path, stores, FIRST, change=entity_types_only)
    rag = GraphRAGChain(config, providers=providers)

    def relationships(strategy: SearchStrategy, filters: dict | None) -> list:
        results = rag.search(
            RAGInput(
                query="Who supplies Buyer?", search_strategy=strategy, filters=filters
            )
        ).search_results.results
        return [r for r in results if r.retriever_type == "relationship"]

    try:
        for strategy in (
            SearchStrategy.LOCAL,
            SearchStrategy.MIX,
            SearchStrategy.HYBRID,
            SearchStrategy.DRIFT,
            SearchStrategy.AUTO,
        ):
            unfiltered = relationships(strategy, None)
            in_schema = relationships(strategy, {"in_schema": True})
            assert unfiltered, strategy
            # Every relationship strict kept is in schema.
            assert len(in_schema) == len(unfiltered), strategy
            assert all(r.metadata.get("in_schema") is True for r in in_schema)
            assert relationships(strategy, {"in_schema": False}) == [], strategy
            for key in ("type_iri", "predicate_iri"):
                with pytest.raises(InvalidFilterError, match=key):
                    relationships(strategy, {key: "https://example.org/onto#x"})
    finally:
        rag.close()
        if stores == "container":
            graph = providers.build_graph_indexer()
            vectors = providers.build_vector_indexer()
            try:
                graph.clear(["default"])
                vectors.clear(["default"])
                clear_extraction_records(vectors, ["default"])
            finally:
                graph.close()
                vectors.close()


def _annotations(config: Config, providers: Providers) -> dict[str, dict]:
    """Every stored item's schema annotations, by store, item kind and a
    readable key (an absent annotation reads as ``None``)."""
    graph = providers.build_graph_indexer()
    try:
        entities = graph.read_schema_annotations("entities") or {}
        edges = graph.read_schema_annotations("relationships") or {}
        names = dict(graph.read_entity_names())
        rels = {r.id: r for r in graph.read_relationships(sorted(edges))}
    finally:
        graph.close()
    prefixes = config.indexing.opensearch
    vector = {
        prefix: {
            hit["_id"]: hit["_source"]
            for hit in e2e._stored(config, providers, getattr(prefixes, prefix))
        }
        for prefix in ("entities_index_prefix", "relationships_index_prefix")
    }

    def edge(rel_id: str) -> tuple[str, str, str]:
        rel = rels[rel_id]
        return (rel.source_name or "", rel.type or "", rel.target_name or "")

    return {
        "graph entities": {names[i]: (a.in_schema, a.iri) for i, a in entities.items()},
        "graph relationships": {
            edge(i): (a.in_schema, a.iri) for i, a in edges.items()
        },
        "vector entities": {
            doc["name"]: (doc.get("in_schema"), doc.get("type_iri"))
            for doc in vector["entities_index_prefix"].values()
        },
        "vector relationships": {
            edge(i): (doc.get("in_schema"), doc.get("predicate_iri"))
            for i, doc in vector["relationships_index_prefix"].items()
        },
    }


COMPANY_IRI = "https://example.org/onto#Company"
STOCKS_IRI = "https://example.org/onto#stocks"


def _annotated(config: Config) -> None:
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


def _iris_and_forms_edited(config: Config) -> None:
    # Other IRIs (one moved, one dropped, one added) and other spellings of
    # the same items: nothing extraction sees.
    _annotated(config)
    extraction = config.processing.graph_extraction
    extraction.entity_types = [
        {"label": "ORGANIZATION", "description": "Companies", "iri": COMPANY_IRI},
        {"label": "PRODUCT", "description": "Goods"},
    ]
    extraction.relationship_types = [
        "SUPPLIES: Sells to",
        {"label": "STOCKS", "description": "Holds goods", "iri": STOCKS_IRI},
    ]
    extraction.relationship_patterns = [
        {
            "source_type": "ORGANIZATION",
            "relation": "SUPPLIES",
            "target_type": "ORGANIZATION",
        },
        "ORGANIZATION -STOCKS-> PRODUCT",
    ]


def _entity_types_off(config: Config) -> None:
    extraction = config.processing.graph_extraction
    extraction.schema_mode = "off"
    extraction.relationship_patterns = []
    extraction.relationship_types = []
    extraction.entity_types = [
        {"label": "ORGANIZATION", "description": "Companies", "iri": ORG_IRI},
        "PRODUCT: Goods",
    ]


def _entity_types_align(config: Config) -> None:
    _entity_types_off(config)
    config.processing.graph_extraction.schema_mode = "align"


@pytest.mark.parametrize(
    ("first", "edited"),
    [
        (_annotated, _iris_and_forms_edited),
        (_entity_types_off, _entity_types_align),
    ],
    ids=["iris-and-forms", "off-to-align-entity-types"],
)
def test_an_edit_extraction_does_not_see_rewrites_the_annotations_without_a_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_aws,  # noqa: F811
    first: Callable[[Config], None],
    edited: Callable[[Config], None],
) -> None:
    config, providers, _, source = _run(tmp_path, "inc", FIRST, change=first)
    before = _annotations(config, providers)

    changed = _config(tmp_path / "inc" / "stores", tmp_path / "inc" / "viz", edited)
    server = FakeOllama(_Model(), e2e.ANSWER, e2e.DIM)
    changed_providers = e2e._providers(changed, server, e2e.FakeCrossEncoder("r"))
    context = e2e._ingest(
        changed, changed_providers, source, tmp_path / "inc" / "run-2"
    )
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    # Nothing extracted again, nothing embedded again.
    assert context.incremental_delta is not None
    assert context.incremental_delta.is_empty
    assert server.chat_models == []
    assert server.embed_models == []
    after = _annotations(changed, changed_providers)
    assert after != before

    # What a full build of the edited config stores.
    _, full_providers, _, _ = _run(tmp_path, "full", FIRST, change=edited)
    full_config = _config(
        tmp_path / "full" / "stores", tmp_path / "full" / "viz", edited
    )
    assert after == _annotations(full_config, full_providers)

    # Recorded: the next run has nothing to rewrite.
    calls: list[object] = []
    monkeypatch.setattr(
        "unified_kg_rag.application.storage.indexing_manager.IndexingManager.reannotate",
        lambda self, suffixes: calls.append(suffixes),
    )
    context = e2e._ingest(
        changed, changed_providers, source, tmp_path / "inc" / "run-3"
    )
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls == []
    assert server.chat_models == []


def _ingest_into(
    tmp_path: Path,
    stores: str,
    source: Path,
    change: Callable[[Config], None],
    run: str,
    reset: bool = False,
) -> tuple[Config, Providers]:
    """One ingest of ``source`` into the store set ``stores``."""
    config = _config(tmp_path / stores / "stores", tmp_path / stores / "viz", change)
    config.indexing.reset = reset
    providers = e2e._providers(
        config, FakeOllama(_Model(), e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    context = e2e._ingest(config, providers, source, tmp_path / stores / run)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    return config, providers


def _recorded_digest(config: Config) -> str | None:
    registry = SQLiteDocStatusStore(config)
    try:
        return registry.get_annotation_digest(
            config_registry_namespace(config.indexing, None)
        )
    finally:
        registry.close()


def test_a_schema_reverted_on_another_corpus_rewrites_the_namespace(
    tmp_path: Path, no_aws  # noqa: F811
) -> None:
    # Two corpora (scopes) in one namespace. Corpus A's run annotates the
    # whole namespace by the edited IRIs; corpus B's next run, with the
    # IRIs B last wrote, finds the namespace holding A's and rewrites it.
    corpora = {"a": FIRST, "b": FINAL}
    sources = {}
    for name, corpus in corpora.items():
        sources[name] = tmp_path / f"corpus-{name}"
        sources[name].mkdir()
        for file_name, text in corpus.items():
            (sources[name] / file_name).write_text(text)
    _ingest_into(tmp_path, "inc", sources["a"], _annotated, "run-a1")
    _ingest_into(tmp_path, "inc", sources["b"], _annotated, "run-b1")
    edited, _ = _ingest_into(
        tmp_path, "inc", sources["a"], _iris_and_forms_edited, "run-a2"
    )
    assert _recorded_digest(edited) == schema_annotation_digest(edited)
    reverted, providers = _ingest_into(
        tmp_path, "inc", sources["b"], _annotated, "run-b2"
    )
    assert _recorded_digest(reverted) == schema_annotation_digest(reverted)

    _ingest_into(tmp_path, "full", sources["a"], _annotated, "run-a")
    full, full_providers = _ingest_into(
        tmp_path, "full", sources["b"], _annotated, "run-b"
    )
    assert _annotations(reverted, providers) == _annotations(full, full_providers)


def test_a_reset_records_the_annotations_once_it_wrote_every_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws  # noqa: F811
) -> None:
    config, _, _, source = _run(tmp_path, "inc", FIRST, change=_annotated)
    assert _recorded_digest(config) == schema_annotation_digest(config)
    edited, _ = _ingest_into(
        tmp_path, "inc", source, _iris_and_forms_edited, "run-2", reset=True
    )
    assert _recorded_digest(edited) == schema_annotation_digest(edited)
    assert schema_annotation_digest(edited) != schema_annotation_digest(config)

    # A rebuild that stops, or fails a write, may leave items of the earlier
    # schema: the namespace stays marked for the next run.
    manager = "unified_kg_rag.application.storage.indexing_manager.IndexingManager"
    index_all_data = IndexingManager.index_all_data

    def one_failed(self: IndexingManager, **kwargs: Any) -> Any:
        results = index_all_data(self, **kwargs)
        results["neptune_entities"].add_error("refused")
        return results

    monkeypatch.setattr(f"{manager}.index_all_data", one_failed)
    _ingest_into(tmp_path, "inc", source, _annotated, "run-3", reset=True)
    assert _recorded_digest(config) == REANNOTATING_DIGEST

    def stopped(self: IndexingManager, **kwargs: Any) -> Any:
        raise RuntimeError("stopped")

    _ingest_into(tmp_path, "inc", source, _iris_and_forms_edited, "run-4", reset=True)
    monkeypatch.setattr(f"{manager}.index_all_data", stopped)
    config = _config(tmp_path / "inc" / "stores", tmp_path / "inc" / "viz", _annotated)
    config.indexing.reset = True
    providers = e2e._providers(
        config, FakeOllama(_Model(), e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    context = e2e._ingest(config, providers, source, tmp_path / "inc" / "run-5")
    assert context.status is PipelineStageStatus.FAILED
    assert _recorded_digest(config) == REANNOTATING_DIGEST
