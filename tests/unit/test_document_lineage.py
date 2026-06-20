# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for per-document lineage attribution (incremental indexing, AWS-free).

`build_document_lineage` is the pure logic that lets the production pipeline
record which artifacts each document produced, so a later run removes only a
document's *exclusive* artifacts.
"""

from __future__ import annotations

import pytest

from aws_graphrag.application.ingestion.incremental import build_document_lineage
from aws_graphrag.domain.ingestion.delta_detector import compute_doc_id
from aws_graphrag.domain.models import Document, Entity, Relationship, TextUnit

pytestmark = pytest.mark.unit


def _doc(doc_id: str, path: str) -> Document:
    return Document(
        page_content="x",
        document_id=doc_id,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def test_lineage_attributes_artifacts_to_source_document() -> None:
    docs = [_doc("d1", "/a.txt"), _doc("d2", "/b.txt")]
    text_units = [
        TextUnit(id="t1", text="...", document_ids=["d1"]),
        TextUnit(id="t2", text="...", document_ids=["d2"]),
    ]
    entities = [
        Entity(id="e1", name="A", text_unit_ids=["t1"]),
        Entity(id="e2", name="B", text_unit_ids=["t2"]),
    ]
    relationships = [
        Relationship(id="r1", source_id="e1", target_id="e1", text_unit_ids=["t1"]),
    ]

    lineages = build_document_lineage(docs, text_units, entities, relationships, [], [])
    by_doc = {ln.doc_id: ln for ln in lineages}

    a, b = compute_doc_id("/a.txt"), compute_doc_id("/b.txt")
    assert by_doc[a].entity_ids == ["e1"]
    assert by_doc[a].text_unit_ids == ["t1"]
    assert by_doc[a].relationship_ids == ["r1"]
    assert by_doc[b].entity_ids == ["e2"]
    assert by_doc[b].relationship_ids == []


def test_shared_artifact_attributed_to_both_documents() -> None:
    # An entity spanning text units from two documents is attributed to each;
    # the registry's exclusive-id computation later protects the shared id.
    docs = [_doc("d1", "/a.txt"), _doc("d2", "/b.txt")]
    text_units = [
        TextUnit(id="t1", text="...", document_ids=["d1"]),
        TextUnit(id="t2", text="...", document_ids=["d2"]),
    ]
    shared = Entity(id="shared", name="S", text_unit_ids=["t1", "t2"])

    lineages = build_document_lineage(docs, text_units, [shared], [], [], [])
    for ln in lineages:
        assert "shared" in ln.entity_ids


def test_lineage_uses_stable_doc_id_not_runtime_document_id() -> None:
    # Same file, different per-run document_id -> same stable doc_id.
    docs = [_doc("run-specific-uuid", "/a.txt")]
    text_units = [TextUnit(id="t1", text="...", document_ids=["run-specific-uuid"])]
    lineages = build_document_lineage(docs, text_units, [], [], [], [])
    assert lineages[0].doc_id == compute_doc_id("/a.txt")
