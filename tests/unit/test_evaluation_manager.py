# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EvaluationManager (AWS-free).

``load_data`` is a pure static parser. ``_evaluate_results`` threads
ground-truth expectations onto each result's metadata and dispatches to the
configured evaluators. The GRAPH_AWARE evaluator is AWS-free, so the manager is
configured to use only it; LangChain/Ragas (which build Bedrock clients) are
never constructed.
"""

from __future__ import annotations

import json

import pytest

from aws_graphrag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
)
from aws_graphrag.evaluation import EvaluationManager
from aws_graphrag.evaluation.evaluation_manager import (
    GraphAwareEvaluator,
    LangChainEvaluator,
    RagasEvaluator,
)

pytestmark = pytest.mark.unit


def _graph_aware_manager(config: Config) -> EvaluationManager:
    config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
    return EvaluationManager(config, rag_chain=object())


class TestLoadData:
    def test_requires_path(self) -> None:
        with pytest.raises(ValueError):
            EvaluationManager.load_data("")

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            EvaluationManager.load_data(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            EvaluationManager.load_data(path)

    def test_parses_question_answer_and_ids(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "x1",
                        "question": "Q1?",
                        "answer": "A1",
                        "category": "cat",
                        "difficulty": "hard",
                    }
                ]
            ),
            encoding="utf-8",
        )
        queries, gts = EvaluationManager.load_data(path)
        assert len(queries) == 1 and len(gts) == 1
        q = queries[0]
        assert q.query_id == "x1"
        assert q.question == "Q1?"
        assert q.category == "cat"
        assert q.difficulty == "hard"
        assert gts[0].ground_truth == "A1"

    def test_query_id_prefers_query_id_then_id_then_index(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(
                [
                    {"query_id": "qq", "question": "a", "answer": "x"},
                    {"id": "ii", "question": "b", "answer": "x"},
                    {"question": "c", "answer": "x"},  # falls back to q_2
                ]
            ),
            encoding="utf-8",
        )
        queries, _ = EvaluationManager.load_data(path)
        assert [q.query_id for q in queries] == ["qq", "ii", "q_2"]

    def test_skips_items_without_question(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(
                [
                    {"question": "ok", "answer": "a"},
                    {"answer": "no question"},  # skipped
                    "not a dict",  # skipped
                ]
            ),
            encoding="utf-8",
        )
        queries, gts = EvaluationManager.load_data(path)
        assert len(queries) == 1
        assert queries[0].question == "ok"

    def test_ground_truth_built_from_expected_only(self, tmp_path) -> None:
        # No textual answer, but expected_entities present -> still build a GT.
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "question": "Q?",
                        "expected_entities": ["Alice"],
                        "expected_relationships": ["works at"],
                        "reference_sources": ["doc1"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        queries, gts = EvaluationManager.load_data(path)
        assert len(gts) == 1
        gt = gts[0]
        assert gt.ground_truth == ""  # no answer
        assert gt.expected_entities == ["Alice"]
        assert gt.expected_relationships == ["works at"]
        assert gt.reference_sources == ["doc1"]

    def test_no_ground_truth_signal_skips_gt_but_keeps_query(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"question": "Q only"}]), encoding="utf-8")
        queries, gts = EvaluationManager.load_data(path)
        assert len(queries) == 1
        assert gts == []

    def test_base_metadata_merged_and_none_stripped(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(
                [{"question": "Q?", "answer": "A", "metadata": {"item_key": 1}}]
            ),
            encoding="utf-8",
        )
        queries, _ = EvaluationManager.load_data(
            path, base_metadata={"shared": "v", "dropme": None}
        )
        md = queries[0].metadata
        assert md["shared"] == "v"
        assert md["item_key"] == 1
        assert "dropme" not in md  # None values stripped from base metadata

    def test_item_metadata_overrides_base(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps([{"question": "Q?", "answer": "A", "metadata": {"k": "item"}}]),
            encoding="utf-8",
        )
        queries, _ = EvaluationManager.load_data(path, base_metadata={"k": "base"})
        assert queries[0].metadata["k"] == "item"


class TestInitialization:
    def test_requires_rag_chain(self, config: Config) -> None:
        from aws_graphrag.shared import EvaluationException

        with pytest.raises(EvaluationException):
            EvaluationManager(config, rag_chain=None)

    def test_evaluator_mapping_covers_all_types(self) -> None:
        assert EvaluationManager.EVALUATOR_MAPPING == {
            EvaluatorType.LANGCHAIN: LangChainEvaluator,
            EvaluatorType.RAGAS: RagasEvaluator,
            EvaluatorType.GRAPH_AWARE: GraphAwareEvaluator,
        }

    def test_only_enabled_evaluators_initialized(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        assert set(manager.evaluators) == {EvaluatorType.GRAPH_AWARE}

    def test_unknown_evaluator_type_skipped(self, config: Config, mocker) -> None:
        # An evaluator type absent from EVALUATOR_MAPPING is skipped, not fatal.
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
        mocker.patch.dict(EvaluationManager.EVALUATOR_MAPPING, {}, clear=True)
        manager = EvaluationManager(config, rag_chain=object())
        assert manager.evaluators == {}

    def test_init_failure_of_one_evaluator_does_not_crash(
        self, config: Config, mocker
    ) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("init failed")

        mocker.patch.dict(
            EvaluationManager.EVALUATOR_MAPPING,
            {EvaluatorType.GRAPH_AWARE: _Boom},
        )
        manager = EvaluationManager(config, rag_chain=object())
        assert manager.evaluators == {}


class TestEvaluateResults:
    async def test_threads_expectations_as_copy(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        query = EvaluationQuery(query_id="q1", question="?")
        result = EvaluationResult(
            query_id="q1",
            question="?",
            generated_answer="Alice works at Acme",
            ground_truth="",
        )
        expected_entities = ["Alice", "Acme"]
        gt = EvaluationGroundTruth(
            query_id="q1",
            ground_truth="ref",
            expected_entities=expected_entities,
            expected_relationships=["works at"],
        )
        await manager._evaluate_results([query], [result], [gt])

        # Ground truth string threaded onto the result.
        assert result.ground_truth == "ref"
        # Expectations copied onto metadata.
        assert result.metadata["expected_entities"] == ["Alice", "Acme"]
        # It must be a COPY, not the same list object as the GT's (so an
        # in-place mutation of the result does not corrupt shared GT lists).
        assert result.metadata["expected_entities"] is not gt.expected_entities
        result.metadata["expected_entities"].append("Mutant")
        assert gt.expected_entities == ["Alice", "Acme"]

    async def test_no_matching_gt_leaves_metadata_clean(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        query = EvaluationQuery(query_id="q1", question="?")
        result = EvaluationResult(
            query_id="q1", question="?", generated_answer="x", ground_truth=""
        )
        gt = EvaluationGroundTruth(query_id="OTHER", ground_truth="ref")
        reports = await manager._evaluate_results([query], [result], [gt])
        assert "expected_entities" not in result.metadata
        assert result.ground_truth == ""  # no GT for this id
        assert reports  # still produced a report

    async def test_evaluator_failure_isolated(self, config: Config, mocker) -> None:
        manager = _graph_aware_manager(config)

        async def _boom(*a, **k):
            raise RuntimeError("eval down")

        manager.evaluators[EvaluatorType.GRAPH_AWARE].aevaluate_batch = _boom
        query = EvaluationQuery(query_id="q1", question="?")
        result = EvaluationResult(
            query_id="q1", question="?", generated_answer="x", ground_truth=""
        )
        gt = EvaluationGroundTruth(query_id="q1", ground_truth="ref")
        # Failure is caught and logged; returns no reports rather than raising.
        reports = await manager._evaluate_results([query], [result], [gt])
        assert reports == []


class TestExtractFromResult:
    def test_dict_answer_extracted(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        assert manager._extract_from_result({"answer": "hi"}, "answer", "") == "hi"

    def test_non_dict_non_ragoutput_answer_stringified(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        # For "answer" key, an unknown raw type is stringified.
        assert manager._extract_from_result(42, "answer") == "42"

    def test_non_answer_key_returns_default(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        assert manager._extract_from_result(42, "metadata", {}) == {}


class TestLeanContextStrings:
    def test_desired_fields_extracted(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        out = manager.create_lean_context_strings(
            [{"description": "d", "name": "n", "irrelevant": "z"}]
        )
        assert len(out) == 1
        assert "description" in out[0] and "name" in out[0]
        assert "irrelevant" not in out[0]

    def test_minimal_info_fallback_when_no_desired_fields(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        out = manager.create_lean_context_strings([{"source": "s1", "score": 0.5}])
        assert "s1" in out[0]
