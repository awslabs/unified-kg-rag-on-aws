# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Citation precision and recall (``evaluation.citation_scoring``) in the
retrieval evaluator, and the cited files the manager keeps per result."""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
)
from unified_kg_rag.evaluation import EvaluationManager, RetrievalEvaluator
from unified_kg_rag.evaluation.answer_match_evaluator import AnswerMatchEvaluator
from unified_kg_rag.evaluation.citation_scoring import cited_source_ids

pytestmark = pytest.mark.unit


def _scores(
    config: Config, cited: list[list[str]] | None, refs: list[str]
) -> tuple[dict[str, float], dict[str, str]]:
    result = EvaluationResult(
        query_id="q",
        question="?",
        generated_answer="a",
        ground_truth="",
        retrieved_source_ids=[["a.txt"]],
        retrieved_source_types=["text"],
        cited_source_ids=cited,
        metadata={"reference_sources": refs},
    )
    report = RetrievalEvaluator(config).evaluate_single(
        EvaluationQuery(query_id="q", question="?"), result, ""
    )
    values = {m.metric_type.value: m.value for m in report.metrics}
    return values, report.metadata.get("skipped_metrics", {})


def test_citations_are_scored_against_the_references(config: Config) -> None:
    values, _ = _scores(
        config,
        # A passage of a.txt, an entity over b.txt and 계약서.pdf, one of
        # c.txt, and a seed-only fact naming no file.
        [["a.txt"], ["b.txt", "계약서.pdf"], ["c.txt"], []],
        ["a.txt", "계약서", "d.txt"],
    )
    assert values["citation_precision"] == pytest.approx(2 / 3)
    assert values["citation_recall"] == pytest.approx(2 / 3)


def test_an_answer_citing_nothing_recalls_nothing(config: Config) -> None:
    values, skipped = _scores(config, [], ["a.txt"])
    assert values["citation_recall"] == 0.0
    assert "citation_precision" not in values
    assert skipped["citation_precision"] == "no_citations"


@pytest.mark.parametrize(
    ("cited", "refs", "enabled", "reason"),
    [
        ([["a.txt"]], [], True, "no_reference_sources"),
        (None, ["a.txt"], True, "no_citation_data"),
        ([["a.txt"]], ["a.txt"], False, "citations_disabled"),
    ],
)
def test_citation_metrics_are_skipped_without_what_they_need(
    config: Config,
    cited: list[list[str]] | None,
    refs: list[str],
    enabled: bool,
    reason: str,
) -> None:
    config.search.citations.enabled = enabled
    values, skipped = _scores(config, cited, refs)
    assert "citation_precision" not in values and "citation_recall" not in values
    assert skipped["citation_precision"] == skipped["citation_recall"] == reason


def test_cited_files_are_read_from_the_cited_citations() -> None:
    assert cited_source_ids({"citations": []}) == []
    assert cited_source_ids({"answer": "x"}) is None
    assert cited_source_ids(
        {
            "citations": [
                {"cited": True, "file_name": "a.txt", "documents": []},
                {"cited": False, "file_name": "z.txt"},
                {
                    "cited": True,
                    "documents": [{"file_name": "b.txt"}, {"file_name": "c.txt"}],
                },
            ]
        }
    ) == [["a.txt"], ["b.txt", "c.txt"]]


class _Store:
    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        if ids := (query.filters or {}).get("id"):
            return [
                RetrievalResult(
                    content="",
                    score=0.0,
                    source=i,
                    retriever_type="text",
                    metadata={
                        "id": i,
                        "document_ids": ["doc-2"],
                        "attributes": {"file_name": "supply.txt"},
                    },
                )
                for i in ids
            ]
        return [
            RetrievalResult(
                content="Vendor supplies parts to Buyer.",
                score=0.9,
                source="tu-0",
                retriever_type="text",
                metadata={
                    "id": "tu-0",
                    "document_ids": ["doc-1"],
                    "attributes": {"file_name": "contract.txt"},
                },
            ),
            RetrievalResult(
                content="Vendor: a supplier.",
                score=0.8,
                source="e-1",
                retriever_type="entity",
                metadata={"id": "e-1", "text_unit_ids": ["tu-1"]},
            ),
        ]


async def test_the_manager_scores_what_the_answer_cites(config: Config) -> None:
    config.providers.rerank.kind = "none"
    chain = GraphRAGChain(
        config=config,
        model_factory=ScriptedLLMFactory(lambda _s, _h: "Vendor supplies Buyer [2]."),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: _Store()},
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())  # type: ignore[method-assign]
    config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
    manager = EvaluationManager(config, rag_chain=chain)
    results, _, summary = await manager.evaluate_dataset(
        [
            EvaluationQuery(
                query_id="q1",
                question="Who does Vendor supply?",
                metadata={
                    "search_strategy": "simple",
                    "enable_query_processing": False,
                },
            )
        ],
        [
            EvaluationGroundTruth(
                query_id="q1", ground_truth="", reference_sources=["supply.txt"]
            )
        ],
        show_progress=False,
    )
    await chain.aclose()
    # The answer cites the entity [2], resolved through its lineage.
    assert results[0].cited_source_ids == [["supply.txt"]]
    stats = summary.metric_statistics
    assert stats["citation_precision"]["mean"] == 1.0
    assert stats["citation_recall"]["mean"] == 1.0


@pytest.mark.parametrize(
    ("answer", "enabled", "exact"),
    [
        ("Depot [1].", True, 1.0),
        ("창고 ［１］【2】", True, 1.0),
        ("Depot [1].", False, 0.0),
    ],
)
def test_answer_match_scores_the_words_without_the_markers(
    config: Config, answer: str, enabled: bool, exact: float
) -> None:
    config.search.citations.enabled = enabled
    reference = "창고" if "창고" in answer else "Depot"
    result = EvaluationResult(
        query_id="q",
        question="?",
        generated_answer=answer,
        ground_truth=reference,
        metadata={},
    )
    report = AnswerMatchEvaluator(config).evaluate_single(
        EvaluationQuery(query_id="q", question="?"), result, reference
    )
    values = {m.metric_type.value: m.value for m in report.metrics}
    assert values["exact_match"] == exact
