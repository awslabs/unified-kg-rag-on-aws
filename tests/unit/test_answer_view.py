# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One view of an answer for every surface: the id lists decoded where a
result is made, one file name and title rule, one marker rule, and the
console, MCP and run-rag reading the same view."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import TypeAdapter

from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.cli.console.ask import source_view
from unified_kg_rag.application.cli.mcp_tools import _record
from unified_kg_rag.application.presentation.answer_view import AnswerView, SourceView
from unified_kg_rag.application.retrieval.citations import unit_from_metadata
from unified_kg_rag.application.retrieval.rag_chain import ProcessedQuery, RAGOutput
from unified_kg_rag.domain.models import (
    Citation,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    Source,
)
from unified_kg_rag.domain.retrieval import source_fields

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("stored", "decoded"),
    [
        ('["u1", "u2"]', ["u1", "u2"]),  # an edge's JSON list string
        ("u1", ["u1"]),  # a Gremlin one-element list read as its value
        (["u1", "", None], ["u1"]),
        ("[not json", ["[not json"]),
        ([], []),
    ],
)
def test_id_lists_are_decoded_where_a_result_is_made(stored: Any, decoded: list) -> None:  # type: ignore[type-arg]
    result = RetrievalResult(
        content="x",
        score=1.0,
        retriever_type="relationship",
        metadata={"text_unit_ids": stored, "document_ids": stored, "name": "kept"},
    )
    assert result.metadata["text_unit_ids"] == decoded
    assert result.metadata["document_ids"] == decoded
    assert result.metadata["name"] == "kept"
    source = TypeAdapter(Source).validate_python(
        {
            "content": "",
            "source": None,
            "score": None,
            "metadata": {"entity_ids": stored},
        }
    )
    assert source["metadata"]["entity_ids"] == decoded


@pytest.mark.parametrize(
    ("metadata", "name"),
    [
        ({"attributes": {"file_path": "/srv/corpus/secret/a.pdf"}}, "a.pdf"),
        ({"file_name": "C:\\docs\\x.pdf"}, "x.pdf"),
        ({"attributes": {"file_name": "sub/계약서.pdf"}}, "계약서.pdf"),
        ({"file_name": "top.pdf", "attributes": {"file_name": "attr.pdf"}}, "attr.pdf"),
        ({"attributes": {"file_path": "dir/p.pdf", "file_name": "n.pdf"}}, "n.pdf"),
        ({}, None),
    ],
)
def test_every_surface_names_the_same_file(
    metadata: dict[str, Any], name: str | None
) -> None:
    # Regression (rv7): seven helpers disagreed on these.
    source = Source(
        content="text",
        source="tu-1",
        score=None,
        metadata={"section_type": "text", **metadata},
    )
    view = SourceView.of_source(1, source)
    assert view.file_name == name
    assert source_view(view)["file_name"] == name
    assert _record(view, 2000).document_name == name
    assert run_rag_chain._hit_label(view) == (name or "tu-1")
    assert unit_from_metadata(source["metadata"], "tu-1").file_name == name
    assert source_fields.file_name(source["metadata"]) == name


def _output(**metadata: Any) -> RAGOutput:
    sources = [
        Source(
            content="Vendor shall deliver within 30 days.",
            source="tu-1",
            score=0.9,
            metadata={
                "section_type": "text",
                "chunk_id": "tu-1",
                "attributes": {"file_path": "/srv/x/contract.pdf"},
            },
        ),
        Source(
            content="Vendor supplies Buyer",
            source="rel-1",
            score=0.5,
            metadata={
                "section_type": "relationship",
                "source_name": "Vendor",
                "target_name": "구매자",
                "type": "SUPPLIES",
                "source_id": "e1",
                "target_id": "e2",
                "text_unit_ids": '["tu-1"]',
            },
        ),
        Source(
            content="again",
            source="tu-1",
            score=None,
            metadata={"section_type": "text"},
        ),
    ]
    written = RetrievalResult(
        content="Key point: Vendor delivers.",
        score=0.7,
        source="kp-1",
        retriever_type="general",
        metadata={"synthesized": True, "key_point_rank": 1},
    )
    return RAGOutput(
        answer="Within 30 days [1][3][4].",
        sources=sources,
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=[written],
            total_results=1,
            search_strategy="local",
            processing_time=0.0,
        ),
        citations=[
            Citation(marker=1, cited=True, section_type="text", source_id="tu-1"),
            Citation(
                marker=2, cited=False, section_type="relationship", source_id="rel-1"
            ),
            Citation(marker=3, cited=True, section_type="text", source_id="tu-1"),
            Citation(marker=4, cited=True, section_type="general", source_id="kp-1"),
            Citation(marker=5, cited=False, section_type="text", source_id="gone"),
        ],
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="q", final_query="q rewritten"),
        metadata={"search_strategy": "local", "abstained": False, **metadata},
    )


def test_one_view_marks_titles_and_lists_every_source() -> None:
    view = AnswerView.from_output(_output())
    text, relation, again = view.sources
    # A source shown twice takes the next citation of its id.
    assert [(s.marker, s.cited) for s in view.sources] == [
        (1, True),
        (2, False),
        (3, True),
    ]
    assert view.citation_ranks == {1: 1, 2: 2, 3: 3}
    assert (text.title, text.file_name, text.chunk_id) == (
        "contract.pdf",
        "contract.pdf",
        "tu-1",
    )
    assert relation.title == "Vendor -> 구매자" and relation.type == "SUPPLIES"
    assert [e.id for e in relation.entities] == ["e1", "e2"]
    assert relation.text_unit_ids == ["tu-1"]
    assert again.text == "again"
    # The model-written section the answer cites is no source, but is shown.
    (point,) = view.written
    assert (point.kind, point.marker, point.cited, point.synthesized) == (
        "key_point",
        4,
        True,
        True,
    )
    assert view.searched_query == "q rewritten" and view.strategy == "local"


def test_the_surfaces_map_the_same_view() -> None:
    view = AnswerView.from_output(_output())
    console = [source_view(s) for s in view.sources]
    mcp = [_record(s, 2000) for s in view.sources]
    assert [c["title"] for c in console][:2] == ["contract.pdf", "Vendor -> 구매자"]
    assert [m.name for m in mcp][:2] == [None, "Vendor -> 구매자"]
    assert [c["marker"] for c in console] == [m.citation for m in mcp] == [1, 2, 3]
    assert run_rag_chain._hit_label(view.sources[1]) == "Vendor -> 구매자"


def test_a_source_is_the_dict_it_was_with_its_lists_decoded() -> None:
    output = _output()
    # A plain dict: it serializes and compares as one (json.dumps of an
    # answer's sources keeps working), its id lists decoded.
    dumped = json.loads(json.dumps(output.sources))
    assert dumped[1]["metadata"]["text_unit_ids"] == ["tu-1"]
    assert output.sources[2] == {
        "content": "again",
        "source": "tu-1",
        "score": None,
        "metadata": {"section_type": "text"},
    }


def test_source_fields_take_one_reading_of_documents_and_entities() -> None:
    assert source_fields.document_ids({"attributes": {"document_id": "d1"}}) == ["d1"]
    assert source_fields.document_ids({"document_ids": ["d2"], "document_id": "x"}) == [
        "d2"
    ]
    refs = source_fields.entity_refs(
        "query_result", {"entity_ids": ["e1", "e2"], "entity_names": ["창고"]}, None
    )
    assert [(r.id, r.name) for r in refs] == [("e1", "창고"), ("e2", None)]
    assert (
        source_fields.title("community", {"name": "Level 0: 倉庫"}) == "Level 0: 倉庫"
    )
