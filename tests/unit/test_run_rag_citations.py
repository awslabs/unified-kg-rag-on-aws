# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-rag`` prints the citations an answer uses, and its JSON output
carries every citation (``RAGOutput.citations``)."""

from __future__ import annotations

import json

import pytest

from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.cli.conventions import print_json
from unified_kg_rag.application.retrieval.rag_chain import ProcessedQuery, RAGOutput
from unified_kg_rag.domain.models import (
    Citation,
    CitedDocument,
    SearchQuery,
    SearchResult,
)

pytestmark = pytest.mark.unit


def _output() -> RAGOutput:
    return RAGOutput(
        answer="창고는 부품을 보관합니다 [1][2].",
        sources=[],
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=[],
            total_results=0,
            search_strategy="local",
            processing_time=0.1,
        ),
        citations=[
            Citation(
                marker=1,
                cited=True,
                section_type="text",
                source_id="tu-1",
                file_name="계약서.pdf",
                file_path="legal/계약서.pdf",
                page=3,
                page_end=4,
                chunk_id="tu-1",
                text_unit_ids=["tu-1"],
            ),
            Citation(
                marker=2,
                cited=True,
                section_type="entity",
                source_id="e-1",
                title="창고",
                file_name="물류.txt",
                documents=[
                    CitedDocument(document_id="d-2", file_name="물류.txt"),
                    CitedDocument(document_id="d-3", file_name="[/red]x.txt"),
                ],
            ),
            Citation(
                marker=3,
                cited=False,
                section_type="entity",
                source_id="e-2",
                title="Vendor",
            ),
            Citation(
                marker=4,
                cited=True,
                section_type="entity",
                source_id="e-3",
                title="Depot",
                provenance={"kind": "seed_graph", "seed_graphs": ["warehouses"]},
            ),
        ],
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="q", final_query="q"),
    )


def test_run_rag_prints_the_cited_sources(capsys: pytest.CaptureFixture[str]) -> None:
    result = {"success": True, **_output().model_dump()}
    run_rag_chain.RAGChainRunner._print_result(result)
    printed = capsys.readouterr().out
    assert "Citations (3)" in printed
    assert "[1] legal/계약서.pdf, pp. 3-4" in printed
    assert "[2] 물류.txt (+1 more) - entity: 창고" in printed
    assert "[4] seed graph warehouses - entity: Depot" in printed
    # Uncited sections are not listed.
    assert "Vendor" not in printed


def test_one_page_prints_as_one(capsys: pytest.CaptureFixture[str]) -> None:
    line = run_rag_chain._citation_line(
        Citation(marker=5, section_type="text", file_name="a.pdf", page=2, page_end=2)
    )
    assert line == "[5] a.pdf, p. 2"


def test_a_passage_is_not_named_twice() -> None:
    # Regression (rv8): a passage's title is its file name, so the line read
    # "[2] supply-agreement.pdf, p. 2 - text: supply-agreement.pdf".
    line = run_rag_chain._citation_line(
        Citation(
            marker=2,
            section_type="text",
            file_name="supply-agreement.pdf",
            title="supply-agreement.pdf",
            page=2,
        )
    )
    assert line == "[2] supply-agreement.pdf, p. 2"
    # A graph section's own title is kept.
    entity = Citation(
        marker=3, section_type="entity", file_name="a.pdf", title="Vendor"
    )
    assert run_rag_chain._citation_line(entity) == "[3] a.pdf - entity: Vendor"


def test_the_json_output_carries_every_citation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_json({"success": True, **_output().model_dump()})
    printed = json.loads(capsys.readouterr().out)
    assert [c["marker"] for c in printed["citations"]] == [1, 2, 3, 4]
    assert printed["citations"][0]["file_path"] == "legal/계약서.pdf"
    assert printed["citations"][1]["documents"][1]["document_id"] == "d-3"
