# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What an answer was given without reaches every surface the same way.

Regression (rv8): the degradation ledger (``metadata["degraded"]``) never
reached the surfaces: ``AnswerView`` had no field for it, the console and
the MCP tools read ``error_class`` from the metadata themselves and showed
no partial answer, and ``run-rag`` printed it only in its verbose metrics
dump. ``citations_unresolved`` was ``True``, a second channel beside the
ledger. Now ``AnswerView`` (``QueryOutcome``) carries ``degraded``,
``degraded_error``, ``citations_unresolved`` and ``error_class``, and the
three surfaces map them.
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from unified_kg_rag.application.cli import mcp_tools, run_rag_chain
from unified_kg_rag.application.cli.console.ask import AskService
from unified_kg_rag.application.cli.mcp_tools import KnowledgeGraphTools
from unified_kg_rag.application.presentation.answer_view import (
    AnswerView,
    QueryOutcome,
)
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    ProcessedQuery,
    RAGInput,
    RAGOutput,
    SearchOutput,
)
from unified_kg_rag.domain.models import (
    Citation,
    CitedDocument,
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    Source,
)
from unified_kg_rag.domain.retrieval.degradation import (
    DegradationLedger,
    DegradationStep,
)

pytestmark = pytest.mark.unit

_KEY_POINT = RetrievalResult(
    content="Vendor is the main supplier of 구매자.",
    score=0.8,
    source="synthesized_global_1",
    retriever_type="general",
    metadata={"synthesized": True, "key_point_rank": 1},
)


def _degraded() -> dict[str, Any]:
    ledger = DegradationLedger()
    ledger.record(DegradationStep.GLOBAL_MAP, "1/3 reports unrated", "Throttling")
    ledger.record(DegradationStep.TRANSLATION, "searched untranslated", "Timeout")
    ledger.record_unresolved(
        DegradationStep.CITATION_LINEAGE,
        "graph citations name no document",
        "RuntimeError",
    )
    return ledger.metadata()


def _output(metadata: dict[str, Any], *, error: str | None = None) -> RAGOutput:
    passage = Source(
        content="Vendor ships parts to 구매자.",
        source="tu-1",
        score=0.9,
        metadata={
            "section_type": "text",
            "id": "tu-1",
            "attributes": {
                "file_name": "계약서.txt",
                "relative_path": "법무/계약서.txt",
            },
        },
    )
    return RAGOutput(
        answer=DEFAULT_ERROR_MESSAGE if error else "Vendor supplies 구매자 [1][2].",
        sources=[] if error else [passage],
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=[] if error else [_KEY_POINT],
            total_results=0 if error else 1,
            search_strategy="global",
            processing_time=0.1,
            metadata={"error": error} if error else {},
        ),
        citations=(
            []
            if error
            else [
                Citation(
                    marker=1,
                    cited=True,
                    section_type="key_point",
                    source_id=_KEY_POINT.source,
                    documents=[
                        CitedDocument(
                            document_id="d1",
                            file_name="계약서.txt",
                            text_unit_ids=["tu-1"],
                        )
                    ],
                ),
                Citation(
                    marker=2,
                    cited=True,
                    section_type="text",
                    source_id="tu-1",
                    file_name="계약서.txt",
                    title="계약서.txt",
                ),
            ]
        ),
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="q", final_query="q"),
        metadata={"search_strategy": "global", **metadata},
    )


def _error(error_class: str) -> RAGOutput:
    return _output(
        {"error": True, "error_type": "SomeError", "error_class": error_class},
        error="it failed",
    )


# Each surface's (degraded steps, unresolved steps, error class).
Reported = tuple[list[str], list[str], str | None]


async def _run_rag(output: RAGOutput) -> Reported:
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["-q", "hi", "--output-format", "json"]
    )
    runner.metrics_sink = MagicMock()
    runner.rag_chain = MagicMock()
    runner.rag_chain.ainvoke = AsyncMock(return_value=output)
    document = runner._json_document(await runner._run_query(RAGInput(query="hi")))
    printed = json.loads(json.dumps(document, default=str))
    return (
        [d["step"] for d in printed["degraded"]],
        [d["step"] for d in printed["citations_unresolved"]],
        printed["error_class"],
    )


async def _mcp(output: RAGOutput) -> Reported:
    class Chain:
        config = Config()

        async def ainvoke(self, rag_input: RAGInput) -> RAGOutput:
            return output

    tools = KnowledgeGraphTools(Chain())  # type: ignore[arg-type]
    try:
        result = await tools.answer("hi")
    except mcp_tools._AbsorbedQueryError as e:
        # The error answer is raised, with its class.
        return [], [], e.error_class.value
    return (
        [d.step for d in result.degraded],
        [d.step for d in result.citations_unresolved],
        None,
    )


async def _console(output: RAGOutput) -> Reported:
    result = AskService._result(output, Config(), 0.1)
    json.dumps(result)  # the event the page reads
    return (
        [d["step"] for d in result["degraded"]],
        [d["step"] for d in result["citations_unresolved"]],
        result["error_kind"],
    )


_SURFACES = {"run-rag": _run_rag, "mcp": _mcp, "console": _console}


@pytest.mark.parametrize("surface", sorted(_SURFACES))
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            _output(_degraded()),
            (["global_map", "translation"], ["citation_lineage"], None),
        ),
        (_output({}), ([], [], None)),
        (_error("timeout"), ([], [], "timeout")),
        (_error("model_unreachable"), ([], [], "model_unreachable")),
        (_error("not_a_class"), ([], [], "failure")),
    ],
    ids=["degraded", "clean", "timeout", "model_unreachable", "unknown_class"],
)
async def test_every_surface_reports_the_same_degraded_steps_and_error_class(
    surface: str, output: RAGOutput, expected: Reported
) -> None:
    assert await _SURFACES[surface](output) == expected
    view = AnswerView.from_output(output)
    assert (
        [d.step for d in view.degraded],
        [d.step for d in view.citations_unresolved],
        view.error_class,
    ) == expected


def test_the_view_reads_the_ledger_once() -> None:
    view = AnswerView.from_output(_output(_degraded()))
    assert view.partial and view.degraded_error == "Throttling"
    assert view.citations_unresolved[0].error_type == "RuntimeError"
    assert not AnswerView.from_output(_error("timeout")).partial


def _printed(document: dict[str, Any], *, verbose: bool = False) -> str:
    out = io.StringIO()
    original = run_rag_chain.console
    run_rag_chain.console = Console(file=out, width=300, color_system=None)
    try:
        run_rag_chain.RAGChainRunner._print_result(document, verbose=verbose)
    finally:
        run_rag_chain.console = original
    return out.getvalue()


async def test_run_rag_prints_a_partial_answer_and_its_written_sections() -> None:
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["-q", "hi"]
    )
    runner.metrics_sink = MagicMock()
    runner.rag_chain = MagicMock()
    runner.rag_chain.ainvoke = AsyncMock(return_value=_output(_degraded()))
    document = await runner._run_query(RAGInput(query="hi"))
    text = _printed(document)
    assert (
        "Partial answer: given without global_map (1/3 reports unrated); "
        "translation (searched untranslated)" in text
    )
    assert "Some citations name no document: citation_lineage failed" in text
    # Regression (rv8): a passage's citation named its file twice.
    assert "[2] 계약서.txt" in text and "[2] 계약서.txt -" not in text
    # A key point names the document its citation resolved.
    assert "[1] 계약서.txt - key_point" in text
    # Regression (rv8): the text output never showed the model-written
    # sections the answer cites; the JSON had none either.
    verbose = _printed(document, verbose=True)
    assert "Written Sections (1)" in verbose
    assert "[key_point] synthesized_global_1" in verbose
    assert document["written"][0]["kind"] == "key_point"
    assert document["written"][0]["documents"] == ["계약서.txt"]


def test_the_console_shows_a_written_section_as_a_source() -> None:
    result = AskService._result(_output({}), Config(), 0.1)
    by_rank = {s["rank"]: s for s in result["sources"]}
    assert [s["kind"] for s in result["sources"]] == ["text", "key_point"]
    written = by_rank[2]
    assert written["synthesized"] and written["documents"] == ["계약서.txt"]
    assert written["excerpt"].startswith("Vendor is the main supplier")
    # Regression (rv8): the cited key point named no source to open.
    ranks = {c["marker"]: c["source_rank"] for c in result["citations"]}
    assert ranks == {1: 2, 2: 1}
    assert by_rank[1]["synthesized"] is False


def test_an_empty_query_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Regression (rv8): "   " ran the chain and ended in a traceback, exit 1.
    for query in ("", "   ", "　"):
        monkeypatch.setattr("sys.argv", ["run-rag", "-q", query])
        with pytest.raises(SystemExit) as exited:
            run_rag_chain.CommandLineInterface().parse_args()
        assert exited.value.code == 2
        assert "--query: The query is empty." in capsys.readouterr().err


def test_a_surface_builds_its_query_as_an_invalid_argument() -> None:
    from unified_kg_rag.shared.exceptions import (
        ErrorClass,
        ToolArgumentError,
        classify,
    )

    with pytest.raises(ToolArgumentError) as refused:
        RAGInput.checked(query=" \n ")
    assert classify(refused.value) is ErrorClass.INVALID_ARGUMENT
    assert str(refused.value) == "query: The query is empty."
    assert RAGInput.checked(query=" 공급사는? ").query == "공급사는?"


def test_the_outcome_carries_the_error_answers_type_and_message() -> None:
    # Regression (rv9 guards LM4): the MCP tools and run-rag read the error
    # message and type from the metadata themselves; the outcome carries
    # them, for a search without an answer too.
    output = SearchOutput(
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=[],
            total_results=0,
            search_strategy="local",
            processing_time=0.0,
            metadata={"error": "No indices found for 'tenant-b'"},
        ),
        processed_query=ProcessedQuery(original_query="q", final_query="q"),
        metadata={"error": True, "error_type": "IndexNotFoundError"},
    )
    outcome = QueryOutcome.of_output(output)
    assert (outcome.error_type, outcome.error_message) == (
        "IndexNotFoundError",
        "No indices found for 'tenant-b'",
    )
    # A surface's JSON document reports its failure fields itself.
    assert "error_message" not in outcome.outcome_fields()
    # No error answer: no message, whatever the search metadata holds.
    output.metadata = {}
    assert QueryOutcome.of_output(output).error_message is None


def test_an_answer_without_an_error_reports_no_error_type() -> None:
    # Low (rv10 query): error_type was read without the error flag.
    outcome = QueryOutcome.of({"error_type": "ValueError"})
    assert outcome.error_type is None
    assert QueryOutcome.of({"error": True, "error_type": "ValueError"}).error_type == (
        "ValueError"
    )
