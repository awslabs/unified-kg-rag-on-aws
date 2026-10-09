# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental indexing converges to a fresh full build (AWS-free).

Random rounds add, modify and delete documents. After every round the graph an
incremental run leaves behind must equal the one a full build over the current
corpus produces: the same entities citing the same text units, and the same
edges with the same text units and weights. Nothing the corpus no longer
mentions may survive as an orphan.

A document is a list of text units; a text unit is a set of weighted edges
between a few synthetic entity names. "Extraction" is deterministic: ids come
from the same stable-id helpers the extractor uses, and a text unit's id
depends on its document, position and content.
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.delta_detector import document_doc_id
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
    sum_weights,
)
from unified_kg_rag.domain.models import (
    Config,
    Document,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.property

_PATHS = ["/corpus/a.txt", "/corpus/b.txt", "/corpus/c.txt"]
_NAMES = ["Vendor", "Depot", "Carrier", "Buyer"]
_TYPE = "RELATED_TO"

# Edge = (source name, target name, strength); a text unit holds a few edges.
_edge = st.tuples(
    st.sampled_from(_NAMES), st.sampled_from(_NAMES), st.integers(1, 3)
).filter(lambda e: e[0] != e[1])
_unit = st.lists(_edge, min_size=1, max_size=3, unique_by=lambda e: (e[0], e[1]))
_content = st.lists(_unit, min_size=1, max_size=2)
# One round: each path's content, or None when the file is absent.
_round = st.fixed_dictionaries({path: st.none() | _content for path in _PATHS})

Content = list[list[tuple[str, str, int]]]


def _unit_id(path: str, index: int, unit: list[tuple[str, str, int]]) -> str:
    digest = hashlib.sha256(f"{path}|{index}|{sorted(unit)}".encode()).hexdigest()
    return f"tu-{digest[:16]}"


def _document(path: str, content: Content) -> Document:
    return Document(
        page_content=repr(content),
        document_id=path,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _extract(corpus: dict[str, Content]):
    """Text units, entities and edges a build over ``corpus`` produces."""
    units: list[TextUnit] = []
    entity_units: dict[str, list[str]] = {}
    edge_weights: dict[tuple[str, str], list[dict[str, float]]] = {}
    for path, content in corpus.items():
        for index, unit in enumerate(content):
            unit_id = _unit_id(path, index, unit)
            units.append(TextUnit(id=unit_id, text="...", document_ids=[path]))
            for source, target, strength in unit:
                for name in (source, target):
                    cited = entity_units.setdefault(name, [])
                    if unit_id not in cited:
                        cited.append(unit_id)
                edge_weights.setdefault((source, target), []).append(
                    {unit_id: float(strength)}
                )
    entities = [
        Entity(
            id=BaseProcessor._generate_entity_id(name),
            name=name,
            type="ORG",
            text_unit_ids=cited,
        )
        for name, cited in entity_units.items()
    ]
    edges = []
    for (source, target), weights in edge_weights.items():
        edge = Relationship(
            id=BaseProcessor._generate_relationship_id(source, target, _TYPE),
            source_id=BaseProcessor._generate_entity_id(source),
            target_id=BaseProcessor._generate_entity_id(target),
            type=_TYPE,
        )
        apply_text_unit_weights(edge, sum_weights(weights))
        edges.append(edge)
    return units, entities, edges


def _graph_state(graph: FakeGraphStore):
    entities = graph.read_entities(sorted(graph.ids("entities")))
    edges = graph.read_relationships(sorted(graph.ids("relationships")))
    return (
        {e.id: frozenset(e.text_unit_ids or []) for e in entities},
        {
            r.id: (frozenset(r.text_unit_ids or []), round(r.weight or 0.0, 9))
            for r in edges
        },
    )


def _expected_state(corpus: dict[str, Content]):
    _, entities, edges = _extract(corpus)
    return (
        {e.id: frozenset(e.text_unit_ids or []) for e in entities},
        {
            r.id: (frozenset(r.text_unit_ids or []), round(r.weight or 0.0, 9))
            for r in edges
        },
    )


def _run(inc: IncrementalIndexer, corpus: dict[str, Content]) -> None:
    """One incremental run, wired like the indexing stage."""
    documents = [_document(path, content) for path, content in corpus.items()]
    delta, fingerprints = inc.plan(documents)
    assert inc.remove_changed_and_deleted(delta)
    to_extract = set(delta.new + delta.changed)
    extracted = [d for d in documents if document_doc_id(d) in to_extract]
    if not extracted:
        return
    units, entities, edges = _extract(
        {d.file_path: corpus[d.file_path] for d in extracted}
    )
    lineages = build_document_lineage(
        documents=extracted,
        text_units=units,
        entities=entities,
        relationships=edges,
        communities=[],
        claims=[],
    )
    inc.commit(
        lineages=lineages,
        fingerprints=fingerprints,
        text_units=units,
        entities=entities,
        relationships=edges,
    )


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rounds=st.lists(_round, min_size=1, max_size=5))
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
            e.id: frozenset(e.text_unit_ids or [])
            for e in vector.data.get("entities", {}).values()
        } == expected_entities
        assert vector.ids("relationships") == set(_expected_state(corpus)[1])
