# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relationships gleaning adds make their endpoints cite the gleaned unit.

a.txt extracts Vendor in tA; gleaning b.txt's tB adds Vendor -> Depot. The
document lineage attributes Vendor to b.txt through that edge, so unless
Vendor also cites tB, deleting a.txt keeps Vendor as shared but strips it to
no text units at all. Runs the real gleaning of the records (only the model
call is replaced), their merge and the incremental indexer over in-memory
stores.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import merge_contributions
from unified_kg_rag.domain.ingestion.delta_detector import detect_delta, document_doc_id
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Config,
    Document,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)

pytestmark = pytest.mark.unit

VENDOR = BaseProcessor._generate_entity_id("Vendor")
DEPOT = BaseProcessor._generate_entity_id("Depot")
EDGE = BaseProcessor._generate_relationship_id("Vendor", "Depot", "SHIPS_TO")


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config)


def _entity(name: str, unit: str) -> Entity:
    return Entity(
        id=BaseProcessor._generate_entity_id(name),
        name=name,
        type="ORG",
        text_unit_ids=[unit],
    )


def _gleaned_edge(unit: str) -> Relationship:
    edge = Relationship(
        id=EDGE,
        source_id=VENDOR,
        target_id=DEPOT,
        source_name="Vendor",
        target_name="Depot",
        type="SHIPS_TO",
    )
    apply_text_unit_weights(edge, {unit: 2.0})
    return edge


def _document(path: str) -> Document:
    return Document(
        page_content=path,
        document_id=path,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


_SHIPS = {
    "refinement_plan": {
        "identified_issues": {
            "issue": {
                "issue_type": "MISSING_RELATIONSHIP",
                "details": {
                    "source": "Vendor",
                    "target": "Depot",
                    "type": "SHIPS_TO",
                    "strength": "2",
                },
                "text_evidence": "Vendor ships to Depot",
            }
        }
    }
}
_NOTHING: dict = {"refinement_plan": {"identified_issues": {}}}


def _glean(gleaner: GraphGleaner, mocker, corpus: dict[str, list[str]]):
    """Extract each document's listed entities, then glean b.txt's edge."""
    units = [
        TextUnit(
            id=f"t-{path}",
            text="Vendor ships to Depot." if path == "/b.txt" else "Vendor.",
            document_ids=[path],
        )
        for path in corpus
    ]
    records = [
        TextUnitExtraction(
            text_unit_id=f"t-{path}",
            entities=[_entity(name, f"t-{path}") for name in names],
        )
        for path, names in corpus.items()
    ]
    gleaner.batch_processor = mocker.MagicMock()
    gleaner.batch_processor.execute_with_fallback.side_effect = (
        lambda items_to_process, **_: [
            _SHIPS if unit.id == "t-/b.txt" else _NOTHING for unit in items_to_process
        ]
    )
    records, _ = gleaner.glean(units, records)
    entities, edges = merge_contributions(records)
    return units, entities, edges


def _index(inc: IncrementalIndexer, gleaner, mocker, corpus) -> None:
    documents = [_document(path) for path in corpus]
    delta, fingerprints = detect_delta(documents, inc.doc_status)
    assert inc.remove_changed_and_deleted(delta)
    extracted = set(delta.new + delta.changed)
    batch = {
        p: names
        for p, names in corpus.items()
        if document_doc_id(_document(p)) in extracted
    }
    if not batch:
        return
    units, entities, edges = _glean(gleaner, mocker, batch)
    lineages = build_document_lineage(
        documents=[_document(p) for p in batch],
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


def test_gleaned_edge_makes_existing_endpoint_cite_its_unit(gleaner, mocker) -> None:
    _, entities, edges = _glean(
        gleaner, mocker, {"/a.txt": ["Vendor"], "/b.txt": ["Depot"]}
    )
    by_id = {e.id: e for e in entities}
    assert set(by_id[VENDOR].text_unit_ids) == {"t-/a.txt", "t-/b.txt"}
    assert by_id[DEPOT].text_unit_ids == ["t-/b.txt"]
    assert [e.id for e in edges] == [EDGE]


def test_gleaned_edge_to_an_unlisted_entity_adds_a_stub(gleaner, mocker) -> None:
    # Without a.txt in the batch Vendor is not extracted at all: the edge is
    # kept and Vendor added citing b.txt's unit, as first-pass extraction does,
    # so the result does not depend on which documents share the batch.
    _, entities, edges = _glean(gleaner, mocker, {"/b.txt": ["Depot"]})
    by_id = {e.id: e for e in entities}
    assert by_id[VENDOR].text_unit_ids == ["t-/b.txt"]
    assert [e.id for e in edges] == [EDGE]


def test_deleting_the_extracting_document_keeps_the_endpoint_cited(
    config: Config, gleaner, mocker
) -> None:
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    inc = IncrementalIndexer(FakeDocStatusStore(), manager)

    _index(inc, gleaner, mocker, {"/a.txt": ["Vendor"], "/b.txt": ["Depot"]})
    _index(inc, gleaner, mocker, {"/b.txt": ["Depot"]})

    (vendor,) = graph.read_entities([VENDOR])
    assert vendor.text_unit_ids == ["t-/b.txt"]
    assert vector.data["entities"][VENDOR].text_unit_ids == ["t-/b.txt"]
    assert graph.ids("relationships") == {EDGE}
    # The same graph a fresh build of b.txt alone produces.
    _, fresh_entities, fresh_edges = _glean(gleaner, mocker, {"/b.txt": ["Depot"]})
    stored = graph.read_entities(sorted(graph.ids("entities")))
    assert {e.id: set(e.text_unit_ids) for e in stored} == {
        e.id: set(e.text_unit_ids) for e in fresh_entities
    }
    assert graph.ids("relationships") == {e.id for e in fresh_edges}


def test_the_merge_cites_endpoints_another_unit_lists() -> None:
    # Vendor is listed by tA only; tB's edge names it. The merged graph's
    # Vendor cites both, whichever records share the batch.
    records = [
        TextUnitExtraction(text_unit_id="tA", entities=[_entity("Vendor", "tA")]),
        TextUnitExtraction(
            text_unit_id="tB",
            entities=[_entity("Depot", "tB")],
            relationships=[_gleaned_edge("tB")],
        ),
    ]
    entities, _ = merge_contributions(records)
    by_id = {e.id: e for e in entities}
    assert set(by_id[VENDOR].text_unit_ids) == {"tA", "tB"}
