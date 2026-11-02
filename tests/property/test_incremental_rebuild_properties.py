# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental indexing converges to a fresh full build (AWS-free).

Random rounds add, modify and delete documents. After every round the graph an
incremental run leaves behind must equal the one a full build over the current
corpus produces: the same entities citing the same text units with the same
names, types, descriptions and ranks, and the same edges with the same text
units, weights, descriptions and ranks. Nothing the corpus no longer mentions
may survive, as an orphan or as a line of a description.

A document is a list of text units; a text unit lists a few synthetic entity
names, a set of weighted edges between them, and a set of edges gleaning adds
afterwards. An edge endpoint the unit does not list is named only inside the
edge, so an entity one document lists can be the endpoint of another
document's edge. "Extraction" is deterministic: each unit's record holds
its entities and edges (ids from the stable-id helpers the extractor uses),
the gleaned edges join the record by the gleaner's own rule, the graph is the
merge of the records (``merge_contributions``), and a text unit's id depends
on its document, position and content.

The strict-schema variant types every entity and edge (declared, undeclared
and empty types, spellings of declared ones) and builds each record with the
real extraction parse and gleaning code under ``schema_mode: strict``, so the
drops (off-schema entities and their edges, off-schema types, off-pattern
edges) and the ``off_schema`` rejections are those extraction makes.

The seed variants also add, edit, re-shard and remove seed graphs (their
node cards name document entities) in each round: the shard documents are
converted as bootstrap converts them, their units and records are the
pipeline's, and the stores must hold what a full build over the documents
and the configured seeds holds, with each item's IRI, source kinds and
confidence. The schema variant types the seeds too.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
import unified_kg_rag.adapters.ingestion.graph_extractor as extractor_module
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.fixtures.incremental_runs import edge_state, entity_state
from tests.fixtures.seed_graphs import (
    EX,
    Seed,
    random_seeds,
    seed_shard_extraction,
    seeded_corpus,
)
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    merge_contributions,
    record_keys,
    record_payload,
)
from unified_kg_rag.domain.ingestion.delta_detector import detect_delta, document_doc_id
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.ingestion.schema_annotation import SchemaAnnotator
from unified_kg_rag.domain.ingestion.seed_graph import annotate_sources
from unified_kg_rag.domain.models import (
    Config,
    Document,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.seed_graph import SeedEdge, SeedNode

pytestmark = pytest.mark.property

_PATHS = ["/corpus/a.txt", "/corpus/b.txt", "/corpus/c.txt"]
_NAMES = ["Vendor", "Vendor Inc", "Depot", "Carrier", "Buyer"]
_TYPE = "RELATED_TO"

# Edge = (source name, target name, strength). A text unit = (the entity
# names it lists, its extracted edges, the edges gleaning adds); an endpoint
# it does not list is named only inside the edge.
_edge = st.tuples(
    st.sampled_from(_NAMES), st.sampled_from(_NAMES), st.integers(1, 3)
).filter(lambda e: e[0] != e[1])
_edges = st.lists(_edge, max_size=3, unique_by=lambda e: (e[0], e[1]))
_listed = st.lists(st.sampled_from(_NAMES), max_size=2, unique=True)
_unit = st.tuples(_listed, _edges, _edges).filter(lambda u: u[0] or u[1] or u[2])
_content = st.lists(_unit, min_size=1, max_size=2)
# One round: each path's content, or None when the file is absent.
_round = st.fixed_dictionaries({path: st.none() | _content for path in _PATHS})

Edge = tuple[str, str, int]
Unit = tuple[list[str], list[Edge], list[Edge]]
Content = list[Unit]


def _unit_id(path: str, index: int, unit: Unit) -> str:
    listed, edges, gleaned = unit
    digest = hashlib.sha256(
        f"{path}|{index}|{sorted(listed)}|{sorted(edges)}|{sorted(gleaned)}".encode()
    ).hexdigest()
    # Lowercase hex, as the chunker's ids are: they sort before every seed
    # unit id ("sg-..."), so records merge documents first.
    return str(uuid.UUID(digest[:32]))


def _document(path: str, content: Content | Document) -> Document:
    # A seed graph's shard is its own (virtual) document.
    if isinstance(content, Document):
        return content
    return Document(
        page_content=repr(content),
        document_id=path,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


def _edge(
    source: str, target: str, weights: dict[str, float], description: str = ""
) -> Relationship:
    edge = Relationship(
        id=BaseProcessor._generate_relationship_id(source, target, _TYPE),
        source_id=BaseProcessor._generate_entity_id(source),
        target_id=BaseProcessor._generate_entity_id(target),
        source_name=source,
        target_name=target,
        type=_TYPE,
        description=description or None,
    )
    apply_text_unit_weights(edge, weights)
    return edge


def _extract(corpus: dict[str, Any], config: Config | None = None):
    """Text units, their records, and the graph a build over ``corpus``
    produces.

    Each unit's record lists its entities (a type that depends on the unit's
    position, so types are voted on) and its extracted edges, each with text
    naming the unit; the gleaned edges are added to the record by the
    gleaner's own rule, and the graph is the records merged and resolved by
    the full build's code. A seed shard's units and records are the
    pipeline's (:func:`seed_shard_extraction`).
    """
    units: list[TextUnit] = []
    records: list[TextUnitExtraction] = []
    for path, content in corpus.items():
        if isinstance(content, Document):
            seed_units, seed_records = seed_shard_extraction(content)
            units += seed_units
            records += seed_records
            continue
        for index, unit in enumerate(content):
            unit_id = _unit_id(path, index, unit)
            where = f"{path}#{index}"
            units.append(
                TextUnit(
                    id=unit_id,
                    text="...",
                    document_ids=[path],
                    attributes={RECORD_KEYS_ATTRIBUTE: {"graph": f"g-{unit_id}"}},
                )
            )
            listed, unit_edges, unit_gleaned = unit
            record = TextUnitExtraction(
                text_unit_id=unit_id,
                entities=[
                    Entity(
                        id=BaseProcessor._generate_entity_id(name),
                        name=name,
                        type="ORG" if index % 2 == 0 else "PERSON",
                        description=f"{name} in {where}",
                        confidence=0.6 if index % 2 == 0 else 0.7,
                        text_unit_ids=[unit_id],
                    )
                    for name in listed
                ],
                relationships=[
                    _edge(
                        source,
                        target,
                        {unit_id: float(strength)},
                        f"{source} -> {target} in {where}",
                    )
                    for source, target, strength in unit_edges
                ],
            )
            GraphGleaner._add_to_record(
                record,
                [],
                [
                    _edge(
                        source,
                        target,
                        {unit_id: float(strength)},
                        f"gleaned {source} -> {target} in {where}",
                    )
                    for source, target, strength in unit_gleaned
                ],
            )
            records.append(record)
    resolved, _ = GraphResolver(
        config or Config(), max_workers=1, use_process_pool=False
    ).resolve_graph(*merge_contributions(records))
    return units, records, resolved["entities"], resolved["relationships"]


def _graph_state(graph: FakeGraphStore):
    entities = graph.read_entities(sorted(graph.ids("entities")))
    edges = graph.read_relationships(sorted(graph.ids("relationships")))
    return (
        {e.id: entity_state(e) for e in entities},
        {r.id: edge_state(r) for r in edges},
    )


def _expected_state(
    corpus: dict[str, Any], extract: Callable[[dict[str, Any]], Any] = _extract
):
    _, _, entities, edges = extract(corpus)
    return (
        {e.id: entity_state(e) for e in entities},
        {r.id: edge_state(r) for r in edges},
    )


def _run(
    inc: IncrementalIndexer,
    corpus: dict[str, Any],
    extract: Callable[[dict[str, Any]], Any] = _extract,
) -> None:
    """One incremental run, wired like the indexing stage."""
    documents = [_document(path, content) for path, content in corpus.items()]
    delta, fingerprints = detect_delta(documents, inc.doc_status)
    assert inc.remove_changed_and_deleted(delta)
    to_extract = set(delta.new + delta.changed)
    extracted = [d for d in documents if document_doc_id(d) in to_extract]
    if not extracted:
        return
    units, records, entities, edges = extract(
        {d.file_path: corpus[d.file_path] for d in extracted}
    )
    # The extraction stages store the units' records before indexing.
    inc.indexing_manager.opensearch_indexer.write_extraction_records(
        {
            record_keys(unit)["graph"]: record_payload(record)
            for unit, record in zip(units, records, strict=True)
        }
    )
    lineages = build_document_lineage(documents=extracted, text_units=units)
    inc.commit(
        lineages=lineages,
        fingerprints=fingerprints,
        text_units=units,
        entities=entities,
        relationships=edges,
    )


@pytest.mark.slow
# 300 examples take seconds alone and over 120 s (the default timeout) on a
# loaded runner: the timeout leaves margin for that, not for a hang.
@pytest.mark.timeout(600)
@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rounds=st.lists(_round, min_size=1, max_size=5))
# a.txt lists Vendor; b.txt names it only inside its Vendor -> Depot edge.
# Deleting a.txt must keep Vendor (citing b.txt's unit) and b.txt's edge.
@example(
    rounds=[
        {
            _PATHS[0]: [(["Vendor", "Buyer"], [("Vendor", "Buyer", 1)], [])],
            _PATHS[1]: [(["Depot"], [("Vendor", "Depot", 2)], [])],
            _PATHS[2]: None,
        },
        {
            _PATHS[0]: None,
            _PATHS[1]: [(["Depot"], [("Vendor", "Depot", 2)], [])],
            _PATHS[2]: None,
        },
    ]
)
# The same, but gleaning (not extraction) adds b.txt's Vendor -> Depot edge.
@example(
    rounds=[
        {
            _PATHS[0]: [(["Vendor"], [], [])],
            _PATHS[1]: [(["Depot"], [], [("Vendor", "Depot", 2)])],
            _PATHS[2]: None,
        },
        {
            _PATHS[0]: None,
            _PATHS[1]: [(["Depot"], [], [("Vendor", "Depot", 2)])],
            _PATHS[2]: None,
        },
    ]
)
# "Vendor Inc", named only inside b.txt's edge, merged into the "Vendor" it
# lists; c.txt lists "Vendor Inc" with another type, so a full build keeps it
# apart, with b.txt's edge to it. The stored "Vendor" must split.
@example(
    rounds=[
        {
            _PATHS[0]: None,
            _PATHS[1]: [(["Vendor"], [], [("Depot", "Vendor Inc", 1)])],
            _PATHS[2]: None,
        },
        {
            _PATHS[0]: None,
            _PATHS[1]: [(["Vendor"], [], [("Depot", "Vendor Inc", 1)])],
            _PATHS[2]: [(["Vendor"], [], []), (["Vendor Inc"], [], [])],
        },
    ]
)
# The same split, where the delta's own merge of the two spellings gives the
# delta edge other per-unit strengths than a full build.
@example(
    rounds=[
        {
            _PATHS[0]: None,
            _PATHS[1]: [
                (["Vendor"], [], [("Depot", "Vendor Inc", 1), ("Vendor", "Depot", 2)])
            ],
            _PATHS[2]: None,
        },
        {
            _PATHS[0]: None,
            _PATHS[1]: [
                (["Vendor"], [], [("Depot", "Vendor Inc", 1), ("Vendor", "Depot", 2)])
            ],
            _PATHS[2]: [
                ([], [], [("Vendor", "Depot", 1), ("Vendor Inc", "Depot", 1)]),
                (["Vendor Inc"], [], []),
            ],
        },
    ]
)
def test_incremental_rounds_match_a_full_build(rounds) -> None:
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    inc = IncrementalIndexer(FakeDocStatusStore(), manager)

    for state in rounds:
        corpus = {path: content for path, content in state.items() if content}
        _run(inc, corpus)

        assert _graph_state(graph) == _expected_state(corpus)
        expected_entities, _ = _expected_state(corpus)
        assert {
            e.id: entity_state(e) for e in vector.data.get("entities", {}).values()
        } == expected_entities
        assert vector.ids("relationships") == set(_expected_state(corpus)[1])


# --- strict extraction schema ---------------------------------------------------


def _schema_config(mode: str) -> Config:
    return Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_mode": mode,
                    "entity_types": [
                        {"label": "ORGANIZATION", "iri": "https://example.org/o#Org"},
                        "PERSON",
                    ],
                    "relationship_types": [
                        {"label": "SUPPLIES", "iri": "https://example.org/o#supplies"},
                        "RELATED_TO",
                    ],
                    "relationship_patterns": ["ORGANIZATION -SUPPLIES-> PERSON"],
                }
            }
        }
    )


_STRICT = _schema_config("strict")
_CONFIGS = {"strict": _STRICT, "align": _schema_config("align")}
# Declared, a spelling of a declared one, undeclared and empty.
_ENTITY_TYPES = ["ORGANIZATION", "person", "vendor", ""]
_RELATIONSHIP_TYPES = ["SUPPLIES", "related to", "KNOWS"]

# A typed text unit: (its (name, type) statements, its edges, the edges
# gleaning proposes); an edge is (source, target, strength, type).
_typed_edge = st.tuples(
    st.sampled_from(_NAMES),
    st.sampled_from(_NAMES),
    st.integers(1, 3),
    st.sampled_from(_RELATIONSHIP_TYPES),
).filter(lambda e: e[0] != e[1])
_typed_edges = st.lists(_typed_edge, max_size=3, unique_by=lambda e: e[:2])
_statements = st.lists(
    st.tuples(st.sampled_from(_NAMES), st.sampled_from(_ENTITY_TYPES)),
    max_size=3,
    unique=True,
)
_typed_unit = st.tuples(_statements, _typed_edges, _typed_edges).filter(
    lambda u: u[0] or u[1] or u[2]
)
_typed_round = st.fixed_dictionaries(
    {path: st.none() | st.lists(_typed_unit, min_size=1, max_size=2) for path in _PATHS}
)


@cache
def _strict_processors(mode: str = "strict") -> tuple[GraphExtractor, GraphGleaner]:
    """The real extractor and gleaner under the schema in ``mode``, with no
    model behind them (each unit's answer is given)."""
    config = _CONFIGS[mode]
    with (
        mock.patch.object(bedrock_module, "BedrockLanguageModelFactory"),
        mock.patch.object(extractor_module, "create_robust_xml_output_parser"),
        mock.patch.object(extractor_module, "setup_chain"),
        mock.patch.object(gleaner_module, "create_robust_xml_output_parser"),
        mock.patch.object(gleaner_module, "setup_chain"),
    ):
        return GraphExtractor(config), GraphGleaner(config, show_progress=False)


def _typed_relationship(edge: tuple[str, str, int, str], where: str) -> dict:
    source, target, strength, type_ = edge
    return {
        "source": source,
        "target": target,
        "type": type_,
        "strength": str(strength),
        "description": f"{source} {type_} {target} in {where}",
    }


def _strict_extract(corpus: dict[str, Any], mode: str = "strict"):
    """:func:`_extract` with each record parsed and gleaned by the real code
    under the schema in ``mode`` (strict by default)."""
    extractor, gleaner = _strict_processors(mode)
    units: list[TextUnit] = []
    records: list[TextUnitExtraction] = []
    for path, content in corpus.items():
        if isinstance(content, Document):
            seed_units, seed_records = seed_shard_extraction(content)
            units += seed_units
            records += seed_records
            continue
        for index, unit in enumerate(content):
            unit_id = _unit_id(path, index, unit)
            where = f"{path}#{index}"
            text_unit = TextUnit(
                id=unit_id,
                text="...",
                document_ids=[path],
                attributes={RECORD_KEYS_ATTRIBUTE: {"graph": f"g-{unit_id}"}},
            )
            units.append(text_unit)
            statements, unit_edges, unit_gleaned = unit
            record = extractor._parse_extraction_result(
                {
                    "entities": [
                        {
                            "name": name,
                            "type": type_,
                            "description": f"{name} in {where}",
                        }
                        for name, type_ in statements
                    ],
                    "relationships": [
                        _typed_relationship(edge, where) for edge in unit_edges
                    ],
                },
                text_unit,
            )
            gleaner._apply_answer(
                {
                    "identified_issues": {
                        "issue": [
                            {
                                "issue_type": "MISSING_RELATIONSHIP",
                                "details": _typed_relationship(
                                    edge, f"gleaned {where}"
                                ),
                            }
                            for edge in unit_gleaned
                        ]
                    }
                },
                text_unit,
                record,
            )
            for rel in record.relationships:
                apply_text_unit_weights(
                    rel, {unit_id: rel.weight if rel.weight is not None else 1.0}
                )
            records.append(record)
    resolved, _ = GraphResolver(
        _CONFIGS[mode], max_workers=1, use_process_pool=False
    ).resolve_graph(*merge_contributions(records))
    return units, records, resolved["entities"], resolved["relationships"]


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rounds=st.lists(_typed_round, min_size=1, max_size=4))
# a.txt lists Vendor as a vendor (off schema: dropped, its edge with it);
# b.txt lists it as an ORGANIZATION supplying Buyer. Deleting b.txt must
# leave no Vendor, and the stored records keep the off_schema rejection.
@example(
    rounds=[
        {
            _PATHS[0]: [
                ([("Vendor", "vendor"), ("Buyer", "person")], [], []),
            ],
            _PATHS[1]: [
                (
                    [("Vendor", "ORGANIZATION"), ("Buyer", "person")],
                    [("Vendor", "Buyer", 2, "SUPPLIES")],
                    [("Buyer", "Vendor", 1, "SUPPLIES")],
                )
            ],
            _PATHS[2]: None,
        },
        {
            _PATHS[0]: [
                (
                    [("Vendor", "vendor"), ("Buyer", "person")],
                    [("Vendor", "Buyer", 1, "related to")],
                    [],
                ),
            ],
            _PATHS[1]: None,
            _PATHS[2]: None,
        },
    ]
)
def test_incremental_rounds_match_a_full_build_under_strict(rounds) -> None:
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=_STRICT.indexing.opensearch)
    manager = IndexingManager(
        config=_STRICT, vector_indexer=vector, graph_indexer=graph
    )
    inc = IncrementalIndexer(FakeDocStatusStore(), manager)

    for state in rounds:
        corpus = {path: content for path, content in state.items() if content}
        _run(inc, corpus, _strict_extract)

        expected_entities, expected_edges = _expected_state(corpus, _strict_extract)
        assert _graph_state(graph) == (expected_entities, expected_edges)
        assert {
            e.id: entity_state(e) for e in vector.data.get("entities", {}).values()
        } == expected_entities
        assert vector.ids("relationships") == set(expected_edges)
        # Every stored edge has a declared type, written as declared.
        assert {r.type for r in graph.read_relationships(sorted(expected_edges))} <= {
            "SUPPLIES",
            "RELATED_TO",
        }


def _annotations(
    entities: list[Entity], relationships: list[Relationship]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {e.id: (e.in_schema, e.type_iri) for e in entities},
        {r.id: (r.in_schema, r.predicate_iri) for r in relationships},
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rounds=st.lists(_typed_round, min_size=1, max_size=4))
# Buyer is an ORGANIZATION in a.txt and a vendor in b.txt and c.txt: the vote
# makes it off schema until c.txt goes, then in schema with the class IRI.
@example(
    rounds=[
        {
            _PATHS[0]: [([("Buyer", "ORGANIZATION")], [], [])],
            _PATHS[1]: [
                (
                    [("Buyer", "vendor"), ("Vendor", "person")],
                    [("Vendor", "Buyer", 2, "supplies")],
                    [],
                )
            ],
            _PATHS[2]: [([("Buyer", "vendor")], [("Buyer", "Depot", 1, "KNOWS")], [])],
        },
        {
            _PATHS[0]: [([("Buyer", "ORGANIZATION")], [], [])],
            _PATHS[1]: [
                (
                    [("Buyer", "vendor"), ("Vendor", "person")],
                    [("Vendor", "Buyer", 2, "supplies")],
                    [],
                )
            ],
            _PATHS[2]: None,
        },
    ]
)
@pytest.mark.parametrize("mode", ["align", "strict"])
def test_incremental_rounds_store_the_annotations_of_a_full_build(
    mode: str, rounds
) -> None:
    """The stored schema annotations are those a full build writes: each is a
    function of the item's merged type, whichever run wrote it."""
    config = _CONFIGS[mode]
    graph = FakeGraphStore(config)
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    inc = IncrementalIndexer(FakeDocStatusStore(), manager)
    annotator = SchemaAnnotator(config.processing.graph_extraction.extraction_schema())

    def extract(corpus: dict[str, Any]):
        return _strict_extract(corpus, mode)

    for state in rounds:
        corpus = {path: content for path, content in state.items() if content}
        _run(inc, corpus, extract)

        _, _, entities, edges = extract(corpus)
        expected = _annotations(
            annotator.entities(entities), annotator.relationships(edges)
        )
        assert _graph_state(graph) == _expected_state(corpus, extract)
        assert (
            _annotations(
                graph.read_entities(sorted(graph.ids("entities"))),
                graph.read_relationships(sorted(graph.ids("relationships"))),
            )
            == expected
        )
        assert (
            _annotations(
                list(vector.data.get("entities", {}).values()),
                list(vector.data.get("relationships", {}).values()),
            )
            == expected
        )
        # Every edge strict keeps has a declared type (an entity may not: one
        # named only as an endpoint has no type, so it is off schema).
        if mode == "strict":
            assert all(v[0] is True for v in expected[1].values())


# --- seed graphs -----------------------------------------------------------------

# Seed nodes reuse the documents' names (one entity with what a document
# states) and add names only a seed states.
_SEED_NODE_NAMES = [*_NAMES, "Harbor", "Mill"]
_SEED_GRAPHS = ["alpha", "beta"]


def _seeds(
    names: list[str] = _SEED_NODE_NAMES,
    entity_types: tuple[str | None, ...] = (None, "ORG", "PRODUCT"),
    relationship_types: tuple[str, ...] = (_TYPE, "SUPPLIES"),
) -> st.SearchStrategy[Seed]:
    return random_seeds(names, entity_types, relationship_types)


# One round: the documents' contents and each seed graph, or None when it is
# not configured (a seed that was is removed).
_seeded_round = st.tuples(
    _round,
    st.fixed_dictionaries({name: st.none() | _seeds() for name in _SEED_GRAPHS}),
)


def _seeded_corpus(
    state: dict[str, Any], seeds: dict[str, Seed | None], config: Config
) -> dict[str, Any]:
    return seeded_corpus(
        state, seeds, config.processing.graph_extraction.extraction_schema()
    )


def _seed_entity_state(entity: Entity, kinds: list[str] | None) -> tuple[Any, ...]:
    return (*entity_state(entity), entity.confidence, entity.iri, kinds)


def _seed_edge_state(rel: Relationship, kinds: list[str] | None) -> tuple[Any, ...]:
    return (*edge_state(rel), kinds)


def _stored_seed_state(graph: FakeGraphStore, vector: FakeVectorStore):
    """The stores' items with their confidence, IRI and source kinds, and
    the stored text units."""
    entities = graph.read_entities(sorted(graph.ids("entities")))
    edges = graph.read_relationships(sorted(graph.ids("relationships")))
    return {
        "graph_entities": {
            e.id: _seed_entity_state(e, e.source_kinds) for e in entities
        },
        "graph_edges": {r.id: _seed_edge_state(r, r.source_kinds) for r in edges},
        "vector_entities": {
            e.id: _seed_entity_state(e, e.source_kinds)
            for e in vector.data.get("entities", {}).values()
        },
        "vector_relationships": {
            r.id: _seed_edge_state(r, r.source_kinds)
            for r in vector.data.get("relationships", {}).values()
        },
        "vector_text_units": vector.ids("text_units"),
    }


def _expected_seed_state(
    corpus: dict[str, Any], extract: Callable[[dict[str, Any]], Any] = _extract
):
    """:func:`_stored_seed_state` of a fresh full build of ``corpus``: the
    indexing annotates each item's source kinds from its text units."""
    units, _, entities, edges = extract(corpus)
    entity_states = {e.id: _seed_entity_state(e, annotate_sources(e)) for e in entities}
    edge_states = {r.id: _seed_edge_state(r, annotate_sources(r)) for r in edges}
    return {
        "graph_entities": entity_states,
        "graph_edges": edge_states,
        "vector_entities": entity_states,
        "vector_relationships": edge_states,
        "vector_text_units": {u.id for u in units},
    }


def _seed_of(
    shards: int, nodes: Sequence[SeedNode], edges: Sequence[SeedEdge] = ()
) -> Seed:
    return shards, tuple(nodes), tuple(edges)


# Vendor, a document entity, is a seed node card with an edge to Harbor
# that only the seed states; Harbor and Mill sit in other shards.
_VENDOR_CARD = [
    SeedNode(id="vendor", name="Vendor", type="PRODUCT", iri=EX + "vendor"),
    SeedNode(id="harbor", name="Harbor", description="Harbor seed fact"),
    SeedNode(id="mill", name="Mill", type="ORG", iri=EX + "shared"),
]
_VENDOR_EDGES = [
    SeedEdge(source="vendor", target="harbor", type=_TYPE, weight=2.5),
    SeedEdge(source="mill", target="vendor", type="SUPPLIES"),
    SeedEdge(source="vendor", target="depot", type=_TYPE, description="seed"),
]
_VENDOR_DOCS = {
    _PATHS[0]: [(["Vendor", "Buyer"], [("Vendor", "Depot", 1)], [])],
    _PATHS[1]: None,
    _PATHS[2]: None,
}


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rounds=st.lists(_seeded_round, min_size=1, max_size=4))
# Re-shard the seed (every shard path changes), edit Vendor's card, then
# remove the seed: Vendor keeps only what a.txt states, Harbor and Mill go.
@example(
    rounds=[
        (
            _VENDOR_DOCS,
            {
                "alpha": _seed_of(
                    1, [*_VENDOR_CARD, SeedNode(id="depot", name="Depot")]
                ),
                "beta": None,
            },
        ),
        (
            _VENDOR_DOCS,
            {
                "alpha": _seed_of(
                    3,
                    [*_VENDOR_CARD, SeedNode(id="depot", name="Depot")],
                    _VENDOR_EDGES,
                ),
                "beta": None,
            },
        ),
        (
            _VENDOR_DOCS,
            {
                "alpha": _seed_of(
                    3,
                    [
                        _VENDOR_CARD[0].model_copy(update={"description": "edited"}),
                        *_VENDOR_CARD[1:],
                        SeedNode(id="depot", name="Depot"),
                    ],
                    _VENDOR_EDGES,
                ),
                "beta": _seed_of(2, _VENDOR_CARD[1:]),
            },
        ),
        (_VENDOR_DOCS, {"alpha": None, "beta": None}),
    ]
)
# The document goes, the seed stays: Vendor becomes seed-only (rank 0).
@example(
    rounds=[
        (
            _VENDOR_DOCS,
            {"alpha": _seed_of(2, _VENDOR_CARD, _VENDOR_EDGES[:2]), "beta": None},
        ),
        (
            dict.fromkeys(_PATHS),
            {"alpha": _seed_of(2, _VENDOR_CARD, _VENDOR_EDGES[:2]), "beta": None},
        ),
    ]
)
def test_incremental_seed_rounds_match_a_full_build(rounds) -> None:
    """Rounds also add, edit, re-shard and remove seed graphs whose node
    cards name document entities: the stores still hold what a full build
    over the documents and the configured seeds holds, IRIs, source kinds,
    confidences, ranks and weights included."""
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    inc = IncrementalIndexer(FakeDocStatusStore(), manager)

    for state, seeds in rounds:
        corpus = _seeded_corpus(state, seeds, config)
        _run(inc, corpus)

        assert _stored_seed_state(graph, vector) == _expected_seed_state(corpus)


_typed_seeded_round = st.tuples(
    _typed_round,
    st.fixed_dictionaries(
        {
            name: st.none()
            | _seeds(
                entity_types=("ORGANIZATION", "person", "vendor", None),
                relationship_types=tuple(_RELATIONSHIP_TYPES),
            )
            for name in _SEED_GRAPHS
        }
    ),
)


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rounds=st.lists(_typed_seeded_round, min_size=1, max_size=4))
# A seed types Vendor as a vendor (off schema: strict drops the node and its
# SUPPLIES edge) and Mill as an ORGANIZATION supplying Buyer; a.txt types
# Vendor as an ORGANIZATION. Re-sharded, then removed.
@example(
    rounds=[
        (
            {
                _PATHS[0]: [
                    ([("Vendor", "ORGANIZATION"), ("Buyer", "person")], [], [])
                ],
                _PATHS[1]: None,
                _PATHS[2]: None,
            },
            {
                "alpha": _seed_of(
                    1,
                    [
                        SeedNode(id="vendor", name="Vendor", type="vendor"),
                        SeedNode(id="mill", name="Mill", type="ORGANIZATION"),
                        SeedNode(id="buyer", name="Buyer", type="person"),
                    ],
                    [
                        SeedEdge(source="vendor", target="buyer", type="SUPPLIES"),
                        SeedEdge(source="mill", target="buyer", type="supplies"),
                        SeedEdge(source="mill", target="vendor", type="KNOWS"),
                    ],
                ),
                "beta": None,
            },
        ),
        (
            {
                _PATHS[0]: [
                    ([("Vendor", "ORGANIZATION"), ("Buyer", "person")], [], [])
                ],
                _PATHS[1]: None,
                _PATHS[2]: None,
            },
            {
                "alpha": _seed_of(
                    3,
                    [
                        SeedNode(id="mill", name="Mill", type="ORGANIZATION"),
                        SeedNode(id="buyer", name="Buyer", type="person"),
                    ],
                    [SeedEdge(source="mill", target="buyer", type="supplies")],
                ),
                "beta": None,
            },
        ),
        (
            {
                _PATHS[0]: [
                    ([("Vendor", "ORGANIZATION"), ("Buyer", "person")], [], [])
                ],
                _PATHS[1]: None,
                _PATHS[2]: None,
            },
            {"alpha": None, "beta": None},
        ),
    ]
)
@pytest.mark.parametrize("mode", ["align", "strict"])
def test_incremental_typed_seed_rounds_match_a_full_build(mode: str, rounds) -> None:
    """The strict variant with typed seeds (declared, respelled, undeclared
    and no types), converted under the run's schema: the stores hold what a
    full build holds, schema annotations included."""
    config = _CONFIGS[mode]
    graph = FakeGraphStore(config)
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    inc = IncrementalIndexer(FakeDocStatusStore(), manager)
    annotator = SchemaAnnotator(config.processing.graph_extraction.extraction_schema())

    def extract(corpus: dict[str, Any]):
        return _strict_extract(corpus, mode)

    for state, seeds in rounds:
        corpus = _seeded_corpus(state, seeds, config)
        _run(inc, corpus, extract)

        assert _stored_seed_state(graph, vector) == _expected_seed_state(
            corpus, extract
        )
        _, _, entities, edges = extract(corpus)
        expected = _annotations(
            annotator.entities(entities), annotator.relationships(edges)
        )
        assert (
            _annotations(
                graph.read_entities(sorted(graph.ids("entities"))),
                graph.read_relationships(sorted(graph.ids("relationships"))),
            )
            == expected
        )
        if mode == "strict":
            assert all(v[0] is True for v in expected[1].values())
