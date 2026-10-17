# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-document lineage attribution (incremental indexing, AWS-free).

`build_document_lineage` records each document's text units, by which a later
run finds what the document contributed in the stores; `artifact_owners`
attributes a run's artifacts to its documents through the text units they
cite (a failed write marks its owners FAILED).
"""

from __future__ import annotations

import pytest

from unified_kg_rag.application.ingestion.incremental import (
    artifact_owners,
    build_document_lineage,
)
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id
from unified_kg_rag.domain.models import (
    CommunityReport,
    Document,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.unit


def _doc(doc_id: str, path: str) -> Document:
    return Document(
        page_content="x",
        document_id=doc_id,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


_DOCS = [_doc("d1", "/a.txt"), _doc("d2", "/b.txt")]
_UNITS = [
    TextUnit(id="t1", text="...", document_ids=["d1"]),
    TextUnit(id="t2", text="...", document_ids=["d2"]),
    TextUnit(id="t3", text="...", document_ids=["d1"]),
]


def test_lineage_lists_each_documents_text_units() -> None:
    lineages = build_document_lineage(_DOCS, _UNITS, "tenant-a")
    by_doc = {ln.doc_id: ln for ln in lineages}

    a, b = compute_doc_id("/a.txt"), compute_doc_id("/b.txt")
    assert by_doc[a].text_unit_ids == ["t1", "t3"]
    assert by_doc[b].text_unit_ids == ["t2"]
    assert {ln.suffix for ln in lineages} == {"tenant-a"}
    assert by_doc[a].file_path == "/a.txt"


def test_lineage_uses_stable_doc_id_not_runtime_document_id() -> None:
    # Same file, different per-run document_id -> same stable doc_id.
    docs = [_doc("run-specific-uuid", "/a.txt")]
    text_units = [TextUnit(id="t1", text="...", document_ids=["run-specific-uuid"])]
    lineages = build_document_lineage(docs, text_units)
    assert lineages[0].doc_id == compute_doc_id("/a.txt")


def test_artifacts_are_owned_by_the_documents_of_the_units_they_cite() -> None:
    lineages = build_document_lineage(_DOCS, _UNITS)
    a, b = compute_doc_id("/a.txt"), compute_doc_id("/b.txt")
    owners = artifact_owners(
        lineages,
        [
            _UNITS[1],
            Entity(id="e1", name="A", text_unit_ids=["t1"]),
            Entity(id="shared", name="S", text_unit_ids=["t1", "t2"]),
            Relationship(id="r1", source_id="e1", target_id="e1", text_unit_ids=["t3"]),
            CommunityReport(
                id="cr1", name="r", community_id="c1", text_unit_ids=["t2"]
            ),
            Entity(id="stray", name="X", text_unit_ids=["t-other-run"]),
        ],
    )

    assert owners["t2"] == {b}
    assert owners["e1"] == owners["r1"] == {a}
    assert owners["shared"] == {a, b}
    assert owners["cr1"] == {b}
    # A text unit no document of the run holds owns nothing.
    assert "stray" not in owners
