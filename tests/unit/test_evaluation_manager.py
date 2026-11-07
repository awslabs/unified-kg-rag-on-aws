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

import asyncio
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.extras import requires_eval
from unified_kg_rag.adapters.llm.langchain import BATCH_ITEM_FAILED
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    NO_CONTEXT_ANSWER,
    ProcessedQuery,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    EvaluatorType,
    ModelRole,
    SearchQuery,
    SearchResult,
)
from unified_kg_rag.domain.models.config import EvaluationConfig
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.evaluation.evaluation_manager import (
    _WARM_UP_QUESTION,
    GraphAwareEvaluator,
)
from unified_kg_rag.shared import ConfigurationError, EvaluationException
from unified_kg_rag.shared.usage import record_llm_call

pytestmark = pytest.mark.unit


def _graph_aware_manager(config: Config) -> EvaluationManager:
    config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
    return EvaluationManager(config, rag_chain=object())


class TestLoadData:
    def test_requires_path(self) -> None:
        # A ValueError too, as before.
        with pytest.raises(ConfigurationError, match="data path is required"):
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

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"questions": []}, "must be a JSON array"),
            ([], "contains no queries"),
            ([{"question": "ok"}, "not a dict"], "index 1: expected an object"),
            ([{"id": "a1", "answer": "x"}], "index 0 \\(query_id 'a1'\\).*question"),
            ([{"question": "  "}], "question"),
            ([{"question": "q", "metadata": []}], "'metadata' must be an object"),
            (
                [{"question": "q"}, {"id": "q_0", "question": "r"}],
                "duplicate query_id 'q_0'",
            ),
            (
                [{"id": "s1", "question": "q", "metadata": {"search_strategy": "x"}}],
                "(?s)query_id 's1'.*search_strategy",
            ),
            (
                [{"id": "t1", "question": "q", "expected_entities": "Vendor"}],
                "(?s)query_id 't1'.*expected_entities",
            ),
            (
                [{"id": "al", "question": "q", "answer_aliases": 5}],
                "query_id 'al'.*'answer_aliases' must be a string or a list",
            ),
            (
                [{"question": "q", "metadata": {"answer_aliases": ["x", 2]}}],
                "'answer_aliases' must be a string or a list of strings",
            ),
            # A filter the chain would refuse on every query (#445).
            (
                [{"id": "f1", "question": "q", "metadata": {"filters": {"id": []}}}],
                "(?s)query_id 'f1'.*'id' is an empty list",
            ),
            (
                [{"id": "f2", "question": "q", "metadata": {"filters": {"g": None}}}],
                "(?s)query_id 'f2'.*'g' has no value",
            ),
            # A string its field's type cannot take (a "false" is typed).
            (
                [
                    {
                        "id": "f3",
                        "question": "q",
                        "metadata": {"filters": {"in_schema": "maybe"}},
                    }
                ],
                "(?s)query_id 'f3'.*'in_schema' is a boolean field",
            ),
        ],
    )
    def test_invalid_dataset_fails_fast_with_location(
        self, tmp_path, payload, match
    ) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(EvaluationException, match=match):
            EvaluationManager.load_data(path)

    def test_base_metadata_is_validated_too(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"question": "q"}]), encoding="utf-8")
        with pytest.raises(EvaluationException, match="top_k"):
            EvaluationManager.load_data(path, base_metadata={"top_k": "many"})

    def test_non_rag_metadata_keys_pass_through(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps([{"question": "q", "metadata": {"answer_aliases": ["x"]}}]),
            encoding="utf-8",
        )
        queries, _ = EvaluationManager.load_data(path)
        assert queries[0].metadata["answer_aliases"] == ["x"]

    def test_top_level_answer_aliases_accepted(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(
                [
                    {"question": "q", "answer": "a", "answer_aliases": ["x"]},
                    {
                        "question": "r",
                        "answer_aliases": ["top"],
                        "metadata": {"answer_aliases": ["nested"]},
                    },
                ]
            ),
            encoding="utf-8",
        )
        queries, _ = EvaluationManager.load_data(path)
        assert queries[0].metadata["answer_aliases"] == ["x"]
        assert queries[1].metadata["answer_aliases"] == ["nested"]

    def test_numeric_zero_answer_is_a_ground_truth(self, tmp_path) -> None:
        path = tmp_path / "eval.json"
        path.write_text(json.dumps([{"question": "How many?", "answer": 0}]))
        _, gts = EvaluationManager.load_data(path)
        assert [gt.ground_truth for gt in gts] == ["0"]

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
        from unified_kg_rag.shared import EvaluationException

        with pytest.raises(EvaluationException):
            EvaluationManager(config, rag_chain=None)

    @requires_eval
    def test_resolver_covers_all_types(self) -> None:
        # The lazy resolver returns a class for every evaluator type (ragas
        # is imported on demand to avoid a circular import at module load).
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.RAGAS).__name__
            == "RagasEvaluator"
        )
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.GRAPH_AWARE)
            is GraphAwareEvaluator
        )

    def test_only_the_deterministic_evaluators_are_enabled_by_default(self) -> None:
        # The LLM judges are opt-in: ragas costs about ten judge calls per
        # answer and needs the eval extra (a core install exited 2).
        assert Config().evaluation.enabled_evaluators == [
            EvaluatorType.ANSWER_MATCH,
            EvaluatorType.RETRIEVAL,
            EvaluatorType.GRAPH_AWARE,
        ]

    def test_the_removed_langchain_evaluator_is_dropped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            evaluation = EvaluationConfig(
                enabled_evaluators=["langchain", "answer_match"]
            )
        assert evaluation.enabled_evaluators == [EvaluatorType.ANSWER_MATCH]
        assert "'langchain' evaluator is removed" in caplog.text
        assert "answer_correctness" in caplog.text

    def test_the_removed_langchain_metrics_key_loads_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from unified_kg_rag.application.config import get_config

        path = tmp_path / "config.yaml"
        path.write_text(
            "evaluation:\n  langchain_metrics: [correctness]\n"
            "  enabled_evaluators: [langchain, retrieval]\n"
        )
        with caplog.at_level(logging.WARNING):
            config = get_config(path)
        assert config.evaluation.enabled_evaluators == [EvaluatorType.RETRIEVAL]
        assert "evaluation.langchain_metrics" in caplog.text
        assert "answer_correctness" in caplog.text

    def test_a_removed_evaluator_built_around_validation_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="langchain.*removed"):
            EvaluationManager._resolve_evaluator_class(EvaluatorType.LANGCHAIN)

    def test_only_enabled_evaluators_initialized(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        assert set(manager.evaluators) == {EvaluatorType.GRAPH_AWARE}

    def test_unknown_evaluator_type_skipped(self, config: Config, mocker) -> None:
        # A type the resolver returns None for is skipped, not fatal.
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
        mocker.patch.object(
            EvaluationManager, "_resolve_evaluator_class", return_value=None
        )
        manager = EvaluationManager(config, rag_chain=object())
        assert manager.evaluators == {}

    def test_init_failure_aborts_by_default(self, config: Config, mocker) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("init failed")

        mocker.patch.object(
            EvaluationManager, "_resolve_evaluator_class", return_value=_Boom
        )
        with pytest.raises(
            EvaluationException, match="graph_aware.*init failed"
        ) as raised:
            EvaluationManager(config, rag_chain=object())
        # The evaluator's own error is the cause, with its traceback.
        assert isinstance(raised.value.__cause__, RuntimeError)

    def test_invalid_config_aborts_by_default(self, config: Config, mocker) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
        mocker.patch.object(GraphAwareEvaluator, "validate_config", return_value=False)
        with pytest.raises(EvaluationException, match="invalid configuration"):
            EvaluationManager(config, rag_chain=object())

    def test_init_failure_dropped_and_recorded_with_ignore_errors(
        self, config: Config, mocker
    ) -> None:
        config.processing.ignore_errors = True
        config.evaluation.enabled_evaluators = [
            EvaluatorType.GRAPH_AWARE,
            EvaluatorType.ANSWER_MATCH,
        ]
        mocker.patch.object(
            GraphAwareEvaluator, "__init__", side_effect=RuntimeError("init failed")
        )
        manager = EvaluationManager(config, rag_chain=object())
        assert set(manager.evaluators) == {EvaluatorType.ANSWER_MATCH}
        assert manager.dropped_evaluators == {
            "graph_aware": "initialization failed: init failed"
        }
        manifest = manager.build_run_manifest()
        assert manifest["dropped_evaluators"] == manager.dropped_evaluators
        assert manifest["enabled_evaluators"] == ["answer_match"]


def _without_modules(monkeypatch: pytest.MonkeyPatch, *packages: str) -> None:
    """Make each package unimportable, as on an install without it."""
    for package in packages:
        for name in [
            m for m in sys.modules if m == package or m.startswith(f"{package}.")
        ]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, package, None)


class TestWithoutTheEvalExtra:
    @pytest.mark.parametrize(
        ("evaluator", "package"),
        [
            (EvaluatorType.RAGAS, "ragas"),
            (EvaluatorType.RAGAS, "datasets"),
        ],
    )
    @pytest.mark.parametrize("ignore_errors", [False, True])
    def test_an_enabled_judge_fails_naming_the_extra(
        self,
        config: Config,
        monkeypatch: pytest.MonkeyPatch,
        evaluator: EvaluatorType,
        package: str,
        ignore_errors: bool,
    ) -> None:
        # Not dropped under ignore_errors: the summary would lack the metrics
        # the config asks for, and installing the extra is the fix.
        _without_modules(monkeypatch, package)
        config.processing.ignore_errors = ignore_errors
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE, evaluator]
        with pytest.raises(
            ConfigurationError, match=rf"'{evaluator.value}'.*'{package}'.*--extra eval"
        ):
            EvaluationManager(config, rag_chain=object())

    def test_the_llm_free_evaluators_need_no_extra(
        self, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _without_modules(monkeypatch, "ragas", "datasets", "pandas")
        llm_free = [
            EvaluatorType.ANSWER_MATCH,
            EvaluatorType.RETRIEVAL,
            EvaluatorType.GRAPH_AWARE,
        ]
        config.evaluation.enabled_evaluators = llm_free
        manager = EvaluationManager(config, rag_chain=object())
        assert set(manager.evaluators) == set(llm_free)


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

    async def test_per_query_error_recorded_not_raised(self, config: Config) -> None:
        # Scoring runs after every answer is generated: an evaluator error is a
        # failed metric (counted by --max-failure-rate), even with the default
        # ignore_errors=False, so the generated answers are never discarded.
        assert config.processing.ignore_errors is False
        manager = _graph_aware_manager(config)
        evaluator = manager.evaluators[EvaluatorType.GRAPH_AWARE]

        async def _boom(*a, **k):
            raise RuntimeError("judge down")

        evaluator.aevaluate_single = _boom
        query = EvaluationQuery(query_id="q1", question="?")
        result = EvaluationResult(
            query_id="q1", question="?", generated_answer="x", ground_truth=""
        )
        reports = await evaluator.aevaluate_batch([query], [result], [""])
        assert reports[0].metadata["evaluation_failed"] is True

    async def test_evaluator_crash_recorded_not_raised(self, config: Config) -> None:
        assert config.processing.ignore_errors is False
        manager = _graph_aware_manager(config)

        async def _boom(*a, **k):
            raise RuntimeError("eval down")

        manager.evaluators[EvaluatorType.GRAPH_AWARE].aevaluate_batch = _boom
        query = EvaluationQuery(query_id="q1", question="?")
        result = EvaluationResult(
            query_id="q1", question="?", generated_answer="x", ground_truth=""
        )
        gt = EvaluationGroundTruth(query_id="q1", ground_truth="ref")
        # Failure is caught and logged (not raised) and recorded per query as a
        # failed report, so the summary can count the metrics as failed.
        reports = await manager._evaluate_results([query], [result], [gt])
        assert len(reports) == 1
        assert reports[0].metrics == []
        assert reports[0].metadata["evaluation_failed"] is True
        assert set(reports[0].metadata["failed_metrics"]) == {
            "entity_coverage",
            "relationship_coverage",
            "retrieved_entity_coverage",
            "retrieved_relationship_coverage",
            "answer_length",
        }

    async def test_errored_result_is_not_scored(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        queries = [
            EvaluationQuery(query_id="ok", question="?"),
            EvaluationQuery(query_id="bad", question="?"),
        ]
        results = [
            EvaluationResult(
                query_id="ok", question="?", generated_answer="Alice", ground_truth=""
            ),
            EvaluationResult(
                query_id="bad",
                question="?",
                generated_answer=DEFAULT_ERROR_MESSAGE,
                ground_truth="",
                error=True,
            ),
        ]
        gts = [
            EvaluationGroundTruth(
                query_id=qid, ground_truth="", expected_entities=["Alice"]
            )
            for qid in ("ok", "bad")
        ]
        reports = await manager._evaluate_results(queries, results, gts)
        assert [r.query_id for r in reports] == ["ok"]


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

    def test_failed_item_yields_default(self, config: Config) -> None:
        manager = _graph_aware_manager(config)
        assert manager._extract_from_result(BATCH_ITEM_FAILED, "answer", "") == ""


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

    def test_content_is_kept_for_vector_results(self, config: Config) -> None:
        # A vector-retriever result carries its text in `content` and nothing else. It
        # must NOT degrade to the id+score minimal-info fallback, or the top-scoring
        # results reach RAGAS / Recall@k with no text to match against.
        out = _graph_aware_manager(config).create_lean_context_strings(
            [
                {
                    "content": "Miquette Giraudy is a keyboardist.",
                    "source": "s1",
                    "score": 0.9,
                }
            ]
        )
        assert "Miquette Giraudy" in out[0]
        assert "s1" not in out[0]


def _rag_output(answer: str, metadata: dict, error: str | None = None) -> RAGOutput:
    return RAGOutput(
        answer=answer,
        sources=[],
        search_results=SearchResult(
            query=SearchQuery(query="?"),
            results=[],
            total_results=0,
            search_strategy="error" if error else "local",
            processing_time=0.1,
            metadata={"error": error} if error else {},
        ),
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="?", final_query="?"),
        metadata=metadata,
    )


class _FakeChain:
    """Returns canned RAG outputs keyed by question (AWS-free)."""

    def __init__(self, outputs: dict) -> None:
        self.outputs = outputs

    async def ainvoke(self, inputs, config=None):
        return self.outputs[inputs["query"]]


class TestAnswerGenerationInputs:
    async def test_question_wins_over_query_metadata(self, config: Config) -> None:
        # load_data validates {**rag_fields, "query": question}; the chain must
        # receive the same precedence, not a stale metadata.query.
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        seen: list[dict] = []

        class _RecordingChain:
            async def ainvoke(self, inputs, config=None):
                seen.append(inputs)
                return _rag_output("Vendor ships.", {})

        manager = EvaluationManager(config, rag_chain=_RecordingChain())
        query = EvaluationQuery(
            query_id="q1",
            question="Who ships?",
            metadata={"query": "Who pays?", "search_strategy": "local"},
        )
        await manager.evaluate_dataset(
            [query],
            [EvaluationGroundTruth(query_id="q1", ground_truth="Vendor")],
            show_progress=False,
        )
        assert [i["query"] for i in seen] == ["Who ships?"]
        assert seen[0]["search_strategy"] == "local"


class TestAnswerSpend:
    """A failed or retried answer keeps the usage and time it spent."""

    @staticmethod
    def _chain(fail_attempts: int, sleep: float = 0.0) -> Any:
        class _Chain:
            attempts = 0

            async def ainvoke(self, inputs, config=None):
                # One model call per attempt, as the RAG chain's would record.
                record_llm_call("query", "m-a", input_tokens=100, output_tokens=5)
                await asyncio.sleep(sleep)
                type(self).attempts += 1
                if type(self).attempts <= fail_attempts:
                    raise TimeoutError("query timed out")
                usage = {"llm_calls": 1, "input_tokens": 100, "output_tokens": 5}
                return _rag_output(
                    "Vendor ships.", {"processing_time": 0.001, "usage": usage}
                )

        return _Chain()

    async def _answer(self, config: Config, chain: Any) -> EvaluationResult:
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(config, rag_chain=chain)
        results, _, summary = await manager.evaluate_dataset(
            [EvaluationQuery(query_id="q1", question="Who ships?")],
            [EvaluationGroundTruth(query_id="q1", ground_truth="Vendor")],
            show_progress=False,
        )
        assert summary.usage["answer_generation"].llm_calls == (
            results[0].usage.llm_calls if results[0].usage else 0
        )
        return results[0]

    async def test_a_failed_answer_keeps_its_partial_usage_and_time(
        self, config: Config
    ) -> None:
        # Before, it had usage None and response_time None, so a strategy
        # that timed out looked cheaper and faster.
        result = await self._answer(config, self._chain(fail_attempts=1, sleep=0.05))
        assert result.error is True
        assert result.usage is not None
        assert (result.usage.llm_calls, result.usage.input_tokens) == (1, 100)
        assert result.response_time is not None and result.response_time >= 0.05

    async def test_a_retried_answer_counts_every_attempt(self, config: Config) -> None:
        chain = self._chain(fail_attempts=1, sleep=0.02)
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(config, rag_chain=chain)
        manager.batch_processor.max_attempts = 2
        manager.batch_processor.retry_multiplier = 0
        results, _, _ = await manager.evaluate_dataset(
            [EvaluationQuery(query_id="q1", question="Who ships?")],
            [EvaluationGroundTruth(query_id="q1", ground_truth="Vendor")],
            show_progress=False,
        )
        (result,) = results
        assert result.error is False
        # The chain's usage of the answering attempt plus the one that raised.
        assert result.usage is not None and result.usage.llm_calls == 2
        # Not the answering attempt's processing_time alone.
        assert result.response_time is not None and result.response_time >= 0.04

    async def test_an_answer_at_the_first_attempt_is_the_chains(
        self, config: Config
    ) -> None:
        result = await self._answer(config, self._chain(fail_attempts=0))
        assert result.response_time == 0.001
        assert result.usage is not None and result.usage.llm_calls == 1


class TestWarmUp:
    async def test_warm_up_answers_a_synthetic_query_with_the_run_settings(
        self, config: Config, caplog
    ) -> None:
        seen: list[dict] = []

        class _Chain:
            async def ainvoke(self, inputs, config=None):
                seen.append(inputs)
                raise RuntimeError("store unreachable")

        # Judge-free: the core install has no LLM-judge evaluator.
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(config, rag_chain=_Chain())
        query = EvaluationQuery(
            query_id="q1",
            question="Who ships?",
            metadata={
                "search_strategy": "mix",
                "top_k": 3,
                "conversation_id": "c1",
                "answerable": False,
            },
        )
        await manager.warm_up(query)  # a failure is logged, not raised
        (inputs,) = seen
        assert inputs["query"] == _WARM_UP_QUESTION != query.question
        assert (inputs["search_strategy"], inputs["top_k"]) == ("mix", 3)
        # No conversation turn and no dataset field.
        assert "conversation_id" not in inputs and "answerable" not in inputs
        assert inputs["use_memory"] is False
        assert "warm-up query failed" in caplog.text

    async def test_a_strategy_the_namespace_does_not_serve_fails_fast(
        self, config: Config
    ) -> None:
        # Every query would fail as the warm-up did: raised, not logged.
        from unified_kg_rag.shared import MethodologyNotIndexedError

        class _Chain:
            async def ainvoke(self, inputs, config=None):
                raise MethodologyNotIndexedError("reads the raptor methodology")

        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(config, rag_chain=_Chain())
        query = EvaluationQuery(query_id="q1", question="Who ships?")
        with pytest.raises(MethodologyNotIndexedError, match="raptor"):
            await manager.warm_up(query)

    def test_the_manifest_records_dense_script_and_the_usage_schema(
        self, config: Config
    ) -> None:
        from unified_kg_rag.domain.models import EvaluationGroundTruth

        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(config, rag_chain=object())
        english = [EvaluationQuery(query_id="q1", question="Who ships?")]
        truths = [EvaluationGroundTruth(query_id="q1", ground_truth="Vendor")]
        manifest = manager.build_run_manifest(queries=english, ground_truths=truths)
        assert manifest["dataset"]["dense_script"] is False
        assert manifest["usage_schema"] == 3
        for question, answer, aliases in (
            ("누가 배송하나?", "Vendor", None),
            ("Who ships?", "운송사", None),
            ("Who ships?", "Vendor", ["運送会社"]),
        ):
            query = EvaluationQuery(
                query_id="q1",
                question=question,
                metadata={"answer_aliases": aliases} if aliases else {},
            )
            truth = EvaluationGroundTruth(query_id="q1", ground_truth=answer)
            built = manager.build_run_manifest(queries=[query], ground_truths=[truth])
            assert built["dataset"]["dense_script"] is True, (question, answer)


class TestErroredQueries:
    async def test_failed_query_is_called_once(self, config: Config) -> None:
        # max_attempts=1 is one call in total: the chain retries transient
        # errors itself, and a deterministic failure must not run twice.
        config.processing.max_concurrency = 3
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]

        class _FailingChain:
            calls = 0

            async def ainvoke(self, inputs, config=None):
                type(self).calls += 1
                raise ValueError("invalid filter")  # deterministic, not transient

        manager = EvaluationManager(config, rag_chain=_FailingChain())
        assert manager.batch_processor.max_attempts == 1
        assert manager.batch_processor.max_concurrency == 3
        results, _, _ = await manager.evaluate_dataset(
            [EvaluationQuery(query_id="q1", question="Who ships?")],
            [EvaluationGroundTruth(query_id="q1", ground_truth="Vendor")],
            show_progress=False,
        )
        assert _FailingChain.calls == 1
        assert results[0].error is True

    async def test_judge_error_keeps_answers_and_breaches_budget(
        self, config: Config
    ) -> None:
        from unified_kg_rag.application.cli.run_evaluation import (
            failure_budget_breaches,
        )

        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(
            config, rag_chain=_FakeChain({"Who ships?": _rag_output("Vendor", {})})
        )

        async def _boom(*a, **k):
            raise RuntimeError("judge down")

        manager.evaluators[EvaluatorType.ANSWER_MATCH].aevaluate_single = _boom
        results, reports, summary = await manager.evaluate_dataset(
            [EvaluationQuery(query_id="q1", question="Who ships?")],
            [EvaluationGroundTruth(query_id="q1", ground_truth="Vendor")],
            show_progress=False,
        )
        # The generated answer survives for save_results ...
        assert results[0].generated_answer == "Vendor"
        assert reports[0].metadata["evaluation_failed"] is True
        # ... and the failed metrics decide the exit code.
        assert summary.metric_outcomes["answer_match"]["answer_contains"] == {
            "scored": 0,
            "failed": 1,
            "skipped": 0,
        }
        assert failure_budget_breaches(summary, max_failure_rate=1.0)

    async def test_rag_error_fallback_flagged_counted_failed_and_not_scored(
        self, config: Config
    ) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
        chain = _FakeChain(
            {
                "Who founded Acme?": _rag_output(
                    "Alice founded Acme.", {"processing_time": 1.0}
                ),
                "Who audits Acme?": _rag_output(
                    DEFAULT_ERROR_MESSAGE,
                    {"error": True, "processing_time": 9.0},
                    error="upstream timeout",
                ),
            }
        )
        manager = EvaluationManager(config, rag_chain=chain)
        queries = [
            EvaluationQuery(query_id="q1", question="Who founded Acme?"),
            EvaluationQuery(query_id="q2", question="Who audits Acme?"),
        ]
        gts = [
            EvaluationGroundTruth(
                query_id=q.query_id, ground_truth="", expected_entities=["Alice"]
            )
            for q in queries
        ]
        results, reports, summary = await manager.evaluate_dataset(
            queries, gts, show_progress=False
        )

        errored = {r.query_id: r for r in results}["q2"]
        assert errored.error is True
        assert errored.error_message == "upstream timeout"
        assert {r.query_id: r for r in results}["q1"].error is False

        # The apology text is not scored by any evaluator.
        assert [r.query_id for r in reports] == ["q1"]
        assert summary.successful_evaluations == 1
        assert summary.failed_evaluations == 1
        # Errored response time is not averaged in.
        assert summary.average_response_time == pytest.approx(1.0)
        assert summary.metric_statistics["entity_coverage"]["count"] == 1
        assert summary.metric_outcomes["graph_aware"]["entity_coverage"] == {
            "scored": 1,
            "failed": 0,
            "skipped": 1,
        }

    def test_failed_item_is_an_error(self) -> None:
        assert (
            EvaluationManager._detect_generation_error(BATCH_ITEM_FAILED, {})
            is not None
        )
        assert EvaluationManager._detect_generation_error(None, {}) is not None
        assert EvaluationManager._detect_generation_error({"answer": "a"}, {}) is None

    def test_error_flag_without_detail_has_generic_message(self) -> None:
        msg = EvaluationManager._detect_generation_error(
            {"answer": "x", "metadata": {"error": True}}, {"error": True}
        )
        assert msg == "RAG chain returned an error response"


class TestMetricOutcomes:
    def _report(self, metrics=(), metadata=None) -> EvaluationReport:
        return EvaluationReport(
            query_id="q",
            evaluator_type=EvaluatorType.RAGAS,
            metrics=[
                EvaluationMetric(
                    metric_type=EvaluationMetricType.ANSWER_CORRECTNESS, value=v
                )
                for v in metrics
            ],
            metadata=metadata or {},
        )

    def test_failed_and_skipped_excluded_from_mean_and_counted(
        self, config: Config
    ) -> None:
        manager = _graph_aware_manager(config)
        manager.evaluators = {}  # count from reports only
        reports = [
            self._report([0.8]),
            self._report(metadata={"failed_metrics": {"answer_correctness": "boom"}}),
            self._report(metadata={"skipped_metrics": {"answer_correctness": "empty"}}),
        ]
        stats = manager._calculate_metric_statistics(reports)
        assert stats["answer_correctness"]["mean"] == pytest.approx(0.8)
        assert stats["answer_correctness"]["count"] == 1
        outcomes = manager._calculate_metric_outcomes([], reports)
        assert outcomes == {
            "ragas": {"answer_correctness": {"scored": 1, "failed": 1, "skipped": 1}}
        }


class TestSummaryBackwardCompat:
    def test_old_summary_json_still_loads(self) -> None:
        old = {
            "total_queries": 2,
            "successful_evaluations": 2,
            "failed_evaluations": 0,
            "average_response_time": 1.5,
            "metric_statistics": {"correctness": {"mean": 0.5, "count": 2}},
            "evaluation_start_time": datetime(2026, 1, 1).isoformat(),
            "evaluation_end_time": datetime(2026, 1, 1, 0, 1).isoformat(),
            "configuration": {},
        }
        summary = EvaluationSummary.model_validate(json.loads(json.dumps(old)))
        assert summary.successful_evaluations == 2
        assert summary.metric_statistics["correctness"]["mean"] == 0.5
        assert summary.metric_outcomes == {}

    def test_old_result_json_still_loads(self) -> None:
        result = EvaluationResult.model_validate(
            {
                "query_id": "q",
                "question": "?",
                "generated_answer": "a",
                "ground_truth": "",
            }
        )
        assert result.error is False and result.error_message is None

    def test_truncated_source_uses_only_the_content_the_model_saw(
        self, config: Config
    ) -> None:
        # A section cut by the token budget is reported with its truncated
        # `content`, but its metadata may still hold the full description /
        # full_content. Evaluation must see only what the answer model saw.
        out = _graph_aware_manager(config).create_lean_context_strings(
            [
                {
                    "content": "Vendor ships parts…",
                    "source": "r1",
                    "score": 0.7,
                    "metadata": {
                        "truncated": True,
                        "description": "Vendor ships parts to Buyer every month.",
                        "full_content": "Full report: Vendor ships parts monthly.",
                    },
                }
            ]
        )
        assert out == ["Vendor ships parts…"]

    def test_truncated_flag_in_metadata_only_is_honoured(self, config: Config) -> None:
        out = _graph_aware_manager(config).create_lean_context_strings(
            [
                {
                    "content": "short",
                    "metadata": {"truncated": True, "summary": "long summary"},
                }
            ]
        )
        assert out == ["short"]

    def test_untruncated_source_uses_content_without_duplication(
        self, config: Config
    ) -> None:
        # `content` already embeds the report's name/summary; re-adding the
        # metadata fields would hand RAGAS the same text twice.
        content = "Report: Vendor network\nSummary: Vendor supplies Buyer."
        out = _graph_aware_manager(config).create_lean_context_strings(
            [
                {
                    "content": content,
                    "metadata": {
                        "truncated": False,
                        "name": "Vendor network",
                        "summary": "Vendor supplies Buyer.",
                        "full_content": "Vendor supplies Buyer.",
                    },
                }
            ]
        )
        assert out == [content]

    def test_empty_content_falls_back_to_metadata_fields(self, config: Config) -> None:
        out = _graph_aware_manager(config).create_lean_context_strings(
            [{"content": "", "metadata": {"description": "d"}}]
        )
        assert "description" in out[0]

    def test_truncated_without_content_is_minimal_info(self, config: Config) -> None:
        out = _graph_aware_manager(config).create_lean_context_strings(
            [{"source": "r1", "metadata": {"truncated": True, "summary": "long"}}]
        )
        assert out == [str({"source": "r1"})]


class TestComparability:
    async def _run(
        self, config: Config, strategies: dict[str, str], requested: str | None = None
    ):
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        chain = _FakeChain(
            {
                q: _rag_output("Vendor", {"search_strategy": s, "processing_time": 1})
                for q, s in strategies.items()
            }
        )
        manager = EvaluationManager(config, rag_chain=chain)
        queries = [
            EvaluationQuery(
                query_id=f"q{i}",
                question=q,
                category="lookup" if i % 2 == 0 else None,
                difficulty="easy",
                metadata={"search_strategy": requested} if requested else {},
            )
            for i, q in enumerate(strategies)
        ]
        gts = [
            EvaluationGroundTruth(query_id="q0", ground_truth="Vendor"),
            EvaluationGroundTruth(query_id="q1", ground_truth="Buyer"),
        ]
        return manager, *await manager.evaluate_dataset(
            queries, gts, show_progress=False
        )

    async def test_grouped_by_actual_strategy_category_difficulty(
        self, config: Config
    ) -> None:
        _, _, _, summary = await self._run(config, {"A?": "local", "B?": "global"})
        grouped = summary.grouped_statistics
        assert grouped["search_strategy"]["local"]["exact_match"]["mean"] == 1.0
        assert grouped["search_strategy"]["global"]["exact_match"]["mean"] == 0.0
        assert set(grouped["category"]) == {"lookup"}  # None is not a group
        assert grouped["difficulty"]["easy"]["exact_match"]["count"] == 2

    async def test_filenames_carry_single_strategy(self, config, tmp_path) -> None:
        manager, results, reports, summary = await self._run(
            config, {"A?": "local", "B?": "local"}
        )
        manager.save_results(results, reports, summary, tmp_path / "one")
        assert all("_local_" in f.name for f in (tmp_path / "one").iterdir())

        manager, results, reports, summary = await self._run(
            config, {"A?": "local", "B?": "global"}
        )
        manager.save_results(results, reports, summary, tmp_path / "mixed")
        names = [f.name for f in (tmp_path / "mixed").iterdir()]
        assert names and not any("local" in n or "global" in n for n in names)

    async def test_filenames_carry_the_requested_strategy(
        self, config, tmp_path
    ) -> None:
        # An auto run named after where it routed was taken for a local run.
        manager, results, reports, summary = await self._run(
            config, {"A?": "local", "B?": "local"}, requested="auto"
        )
        manager.save_results(results, reports, summary, tmp_path)
        names = [f.name for f in tmp_path.iterdir()]
        assert names and all("_auto_" in n and "local" not in n for n in names)

    async def test_runs_in_the_same_second_do_not_overwrite(
        self, config, tmp_path, mocker
    ) -> None:
        manager, results, reports, summary = await self._run(
            config, {"A?": "local", "B?": "local"}
        )
        frozen = datetime(2026, 1, 1, 12, 0, 0)
        mocker.patch(
            "unified_kg_rag.evaluation.evaluation_manager.datetime",
            wraps=datetime,
            **{"now.return_value": frozen},
        )
        manager.save_results(results, reports, summary, tmp_path)
        manager.save_results(results, reports, summary, tmp_path)
        assert len(list(tmp_path.glob("evaluation_summary_*.json"))) == 2

    def test_run_manifest_records_judge_self_preference(self, config: Config) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
        manager = EvaluationManager(config, rag_chain=object())
        manager.evaluators = {}
        manager.config.evaluation.enabled_evaluators = [EvaluatorType.RAGAS]
        models = manager.build_run_manifest()["models"]
        # The default judge is another model than the answer model.
        assert models["judge_is_answer_model"] is False
        # Pinned to the answer model's tier, it grades its own answers.
        manager.config.models.roles.evaluation = "default"
        assert manager.build_run_manifest()["models"]["judge_is_answer_model"] is True
        manager.config.models.roles.evaluation = "judge.other-model-v1"
        assert manager.build_run_manifest()["models"]["judge_is_answer_model"] is False

    def test_run_manifest_counts_the_pairwise_judge(self, config: Config) -> None:
        # Before, a deterministic-only comparison with --pairwise-judge
        # recorded no judge, hiding that it judged its own answers.
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        config.models.roles.evaluation = "default"
        manager = EvaluationManager(config, rag_chain=object())
        one = {"pairwise_judge": True, "search_strategy": ["local"]}
        two = {"pairwise_judge": True, "search_strategy": ["local", "mix"]}
        assert (
            manager.build_run_manifest(cli_args=one)["models"]["judge_is_answer_model"]
            is None
        )
        # One strategy against --baseline-results is judged too.
        against = {**one, "baseline_results": ["ci/previous"]}
        standard = {"overall": False, "mode": "standard", "overlap_model": None}
        assert (
            manager.build_run_manifest(cli_args=against)["pairwise_judge"] == standard
        )
        manifest = manager.build_run_manifest(cli_args=two)
        assert manifest["models"]["judge_is_answer_model"] is True
        assert manifest["models"]["evaluation_judge"] == config.model_id(
            ModelRole.EVALUATION
        )
        assert manifest["pairwise_judge"] == standard
        # The differential mode records its overlap model too.
        config.evaluation.pairwise_mode = "differential"
        assert manager.build_run_manifest(cli_args=two)["pairwise_judge"] == {
            "overall": False,
            "mode": "differential",
            "overlap_model": config.model_id(ModelRole.PAIRWISE_OVERLAP),
        }

    def test_run_manifest_records_the_answer_span(self, config: Config) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(config, rag_chain=object())
        manifest = manager.build_run_manifest()
        assert manifest["answer_scoring"] == {
            "answer_span": "first_line",
            "answer_normalization": "default",
            "token_f1_tokens": "script_aware_v2",
        }
        assert manifest["models"]["answer_span"] is None
        manager.config.evaluation.answer_span = "extracted"
        assert manager.build_run_manifest()["models"]["answer_span"] == (
            config.model_id(ModelRole.ANSWER_SPAN)
        )
        manager.evaluators = {}
        manager.config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
        assert manager.build_run_manifest()["answer_scoring"] is None

    def test_summary_counts_answer_span_fallbacks(self, config: Config) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        manager = EvaluationManager(config, rag_chain=object())

        def report(qid: str, **meta: Any) -> EvaluationReport:
            return EvaluationReport(
                query_id=qid,
                evaluator_type=EvaluatorType.ANSWER_MATCH,
                metrics=[
                    EvaluationMetric(
                        metric_type=EvaluationMetricType.TOKEN_F1, value=1.0
                    )
                ],
                metadata=meta,
            )

        stats = manager._calculate_answer_span_statistics(
            [report("a"), report("b", answer_span_fallback="throttled")]
        )
        assert stats == {"answer_span": "first_line", "scored": 2, "fallbacks": 1}
        manager.evaluators = {}
        assert manager._calculate_answer_span_statistics([report("a")]) == {}

    @requires_eval
    def test_run_manifest(self, config: Config, tmp_path) -> None:
        import hashlib

        data = tmp_path / "eval.json"
        data.write_bytes(b'[{"question": "q"}]')
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
        manager = EvaluationManager(config, rag_chain=object())
        manifest = manager.build_run_manifest(
            data, {"eval_data_path": data, "top_k": 5}
        )
        assert (
            manifest["dataset"]["sha256"]
            == hashlib.sha256(data.read_bytes()).hexdigest()
        )
        assert manifest["cli_args"] == {"eval_data_path": str(data), "top_k": 5}
        # A provider base URL may carry a key in its userinfo or query. The
        # credential is a fake one this test checks gets stripped.
        url = "https://user:secret@llm.example.com:8443/v1?key=k#f"  # gitleaks:allow
        manifest = manager.build_run_manifest(data, {"model_base_url": url})
        assert manifest["cli_args"] == {
            "model_base_url": "https://llm.example.com:8443/v1"
        }
        assert manifest["models"]["answer_generation"] == (
            config.model_id(ModelRole.ANSWER_GENERATION)
        )
        assert manifest["models"]["evaluation_judge"] is None  # no LLM judge
        assert manifest["models"]["judge_is_answer_model"] is None
        assert manifest["enabled_evaluators"] == ["graph_aware"]
        assert manifest["package_version"] and manifest["created_at"]
        assert len(manifest["retrieval_config_sha256"]) == 64
        assert manifest["index"] == {}
        assert manifest["library_versions"]["ragas"]
        assert manifest["library_versions"]["langchain-core"]
        assert "git_sha" in manifest
        assert "git_dirty" in manifest
        json.dumps(manifest)  # serializable as-is

    def test_config_hash_tracks_resolved_config(self, config: Config) -> None:
        config.evaluation.enabled_evaluators = []
        manager = EvaluationManager(config, rag_chain=object())
        before = manager.build_run_manifest()["retrieval_config_sha256"]
        config.models.roles.answer_generation = "another-model"
        assert manager.build_run_manifest()["retrieval_config_sha256"] != before

    def test_config_hash_ignores_the_environment(self, config: Config) -> None:
        # The same evaluation in another account/region/bucket hashes the same.
        config.evaluation.enabled_evaluators = []
        manager = EvaluationManager(config, rag_chain=object())
        before = manager.build_run_manifest()["retrieval_config_sha256"]
        config.aws.region_name = "eu-west-1"
        config.aws.bedrock.region_name = "eu-west-1"
        config.aws.s3.bucket_name = "another-bucket"
        config.providers.llm.base_url = "http://localhost:11434"
        config.evaluation.outputs_directory = "elsewhere"
        assert manager.build_run_manifest()["retrieval_config_sha256"] == before
        config.models.embedding.dimension = 256
        assert manager.build_run_manifest()["retrieval_config_sha256"] != before

    @pytest.mark.parametrize(
        "update",
        [
            # Retrieval behaviour the old hand-kept section list left out.
            {"graph": {"community_detection": {"enabled": False}}},
            {"processing": {"claim_extraction": {"enabled": True}}},
            {"processing": {"ignore_errors": True}},
            {"indexing": {"neptune": {"max_hops": 1}}},
            {"indexing": {"neptune": {"traversal_fetch_multiplier": 7}}},
            {"aws": {"bedrock": {"guardrail": {"check_output": False}}}},
        ],
    )
    def test_config_hash_covers_settings_that_shape_answers(self, update: dict) -> None:
        def _hash(config: Config) -> str:
            config.evaluation.enabled_evaluators = []
            manager = EvaluationManager(config, rag_chain=object())
            return manager.build_run_manifest()["retrieval_config_sha256"]

        assert _hash(Config.model_validate(update)) != _hash(Config())

    def test_dataset_content_hash_ignores_cli_metadata(
        self, config: Config, tmp_path
    ) -> None:
        # A comparison sets each run's strategy on every query; the dataset
        # is still the same file.
        data = tmp_path / "eval.json"
        data.write_text('[{"question": "Who ships?", "answer": "Vendor"}]')
        config.evaluation.enabled_evaluators = []
        manager = EvaluationManager(config, rag_chain=object())

        def _content_hash(strategy: str) -> str:
            queries, truths = manager.load_data(data, {"search_strategy": strategy})
            manifest = manager.build_run_manifest(
                data, queries=queries, ground_truths=truths
            )
            return manifest["dataset"]["content_sha256"]

        assert _content_hash("local") == _content_hash("mix")
        data.write_text('[{"question": "Who ships?",\n "answer": "Vendor"}]')
        assert _content_hash("local") == _content_hash("mix")  # formatting only

    def test_config_hash_tracks_the_guardrail_but_not_its_id(
        self, config: Config
    ) -> None:
        config.evaluation.enabled_evaluators = []
        manager = EvaluationManager(config, rag_chain=object())

        def digest() -> str:
            return manager.build_run_manifest()["retrieval_config_sha256"]

        unguarded = digest()
        guardrail = config.aws.bedrock.guardrail
        guardrail.identifier = "gr-account-a"
        guarded = digest()
        assert guarded != unguarded
        # Another account's guardrail with the same settings: the same run.
        guardrail.identifier = "gr-account-b"
        guardrail.version = "7"
        assert digest() == guarded
        guardrail.check_output = False
        assert digest() != guarded

    async def test_index_build_records_read_per_suffix(self, config: Config) -> None:
        from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

        record = {"pipeline_id": "run-1", "mode": "full"}

        class _Retriever:
            async def aread_build_record(self, suffix):
                if suffix == "broken":
                    raise RuntimeError("unreachable")
                return record if suffix == "default" else None

        chain = GraphRAGChain.__new__(GraphRAGChain)
        object.__setattr__(chain, "_get_retriever", lambda role: _Retriever())
        config.evaluation.enabled_evaluators = []
        manager = EvaluationManager(config, rag_chain=chain)
        queries = [
            EvaluationQuery(query_id="a", question="?"),
            EvaluationQuery(query_id="b", question="?", metadata={"suffix": "v2"}),
            EvaluationQuery(query_id="c", question="?", metadata={"suffix": "broken"}),
        ]
        builds = await manager._read_index_builds(queries)
        assert builds == {"broken": None, "default": record, "v2": None}
        assert manager.build_run_manifest(index_builds=builds)["index"] == builds
        assert (
            await EvaluationManager(config, rag_chain=object())._read_index_builds(
                queries
            )
            == {}
        )

    def test_git_state_none_without_git(self, mocker) -> None:
        mocker.patch(
            "unified_kg_rag.evaluation.evaluation_manager.shutil.which",
            return_value=None,
        )
        assert EvaluationManager._git_state() == (None, None)

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
    def test_git_state_ignores_a_host_repo_that_does_not_track_the_package(
        self, tmp_path, monkeypatch
    ) -> None:
        # e.g. the package installed into a host project's .venv: the host
        # repo's HEAD says nothing about the code that ran.
        import subprocess

        from unified_kg_rag.evaluation import evaluation_manager as em

        package_dir = tmp_path / ".venv" / "unified_kg_rag"
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("")
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        (tmp_path / "app.py").write_text("")
        subprocess.run(["git", "-C", str(tmp_path), "add", "app.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "commit",
                "-qm",
                "host",
            ],
            check=True,
        )
        monkeypatch.setattr(em, "_PACKAGE_DIR", package_dir)
        assert EvaluationManager._git_state() == (None, None)

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
    def test_git_state_reports_checkout_sha_and_dirty_flag(
        self, tmp_path, monkeypatch
    ) -> None:
        import subprocess

        from unified_kg_rag.evaluation import evaluation_manager as em

        package_dir = tmp_path / "unified_kg_rag"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("")
        git = ["git", "-C", str(tmp_path)]
        ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        subprocess.run([*git, "add", "."], check=True)
        subprocess.run([*git, *ident, "commit", "-qm", "pkg"], check=True)
        head = subprocess.run(
            [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        monkeypatch.setattr(em, "_PACKAGE_DIR", package_dir)
        assert EvaluationManager._git_state() == (head, False)
        (package_dir / "__init__.py").write_text("changed = True\n")
        assert EvaluationManager._git_state() == (head, True)

    async def test_evaluate_dataset_attaches_manifest(self, config: Config) -> None:
        manager, results, reports, summary = await self._run(
            config, {"A?": "local", "B?": "local"}
        )
        manifest = summary.run_manifest
        assert manifest["dataset"]["num_queries"] == 2
        assert len(manifest["dataset"]["content_sha256"]) == 64
        assert "path" not in manifest["dataset"]  # library call: no file


class TestAbstention:
    async def test_the_summary_and_manifest_name_the_benchmark(
        self, config: Config
    ) -> None:
        # Before, only the per-row metadata said which benchmark was scored:
        # a musique-full-answerable run summarized like a MuSiQue-Ans one.
        label = {"benchmark": "musique-full-answerable"}
        _, _, summary = await self._run(
            config,
            [("A?", "Vendor.", {}, label), ("B?", "Vendor.", {}, label)],
        )
        assert summary.benchmarks == {"musique-full-answerable": 2}
        assert summary.run_manifest["dataset"]["benchmarks"] == {
            "musique-full-answerable": 2
        }
        _, _, summary = await self._run(config, [("A?", "Vendor.", {}, {})])
        assert summary.benchmarks == {}

    async def _run(self, config: Config, items: list[tuple[str, str, dict, dict]]):
        """items: (question, answer, chain metadata, query metadata)."""
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        chain = _FakeChain({q: _rag_output(a, md) for q, a, md, _ in items})
        manager = EvaluationManager(config, rag_chain=chain)
        queries = [
            EvaluationQuery(query_id=f"q{i}", question=q, metadata=qmd)
            for i, (q, _, _, qmd) in enumerate(items)
        ]
        gts = [
            EvaluationGroundTruth(query_id=f"q{i}", ground_truth="Vendor")
            for i, (_, _, _, qmd) in enumerate(items)
            if qmd.get("answerable") is not False
        ]
        return await manager.evaluate_dataset(queries, gts, show_progress=False)

    async def test_abstention_rate_overall_and_per_strategy(
        self, config: Config
    ) -> None:
        local = {"search_strategy": "local"}
        results, reports, summary = await self._run(
            config,
            [
                ("A?", "Vendor ships.", local, {}),
                ("B?", NO_CONTEXT_ANSWER, {**local, "abstained": True}, {}),
                # The flag decides, in any language; the text never does.
                (
                    "C?",
                    "관련 정보를 찾지 못했습니다.",
                    {"search_strategy": "global", "abstained": True},
                    {},
                ),
                ("D?", NO_CONTEXT_ANSWER, {"search_strategy": "global"}, {}),
            ],
        )
        assert [r.abstained for r in results] == [False, True, True, False]
        stats = summary.abstention_statistics
        # completed = abstained + answered: "answered" means not declined,
        # as in answer_match's coverage.
        assert stats["completed"] == 4
        assert stats["abstained"] == 2 and stats["answered"] == 2
        assert stats["abstention_rate"] == pytest.approx(2 / 4)
        assert stats["per_strategy"]["local"]["abstention_rate"] == 0.5
        assert stats["per_strategy"]["global"]["abstention_rate"] == 0.5
        assert "unanswerable" not in stats
        # An abstention on an answerable item is still graded (a miss).
        assert summary.metric_statistics["answer_contains"]["count"] == 4

    async def test_saved_results_keep_decline_and_sufficiency_fields(
        self, config: Config, tmp_path
    ) -> None:
        # Before, a saved result kept only `abstained`: why it declined, the
        # strategy's sufficiency verdict and the iterative run record were
        # dropped, so a run could not be analysed from its files.
        iterative = {
            "base": "mix",
            "mode": "loop",
            "rounds": 2,
            "stop_reason": "max_rounds",
            "verdicts": [{"round": 1, "sufficient": False, "bridges": ["Depot"]}],
        }
        insufficient = {
            "search_strategy": "iterative",
            "abstained": True,
            "abstention_reason": "insufficient_context",
            "context_sufficient": False,
            "insufficiency_reason": "max_rounds",
            "iterative": iterative,
        }
        results, reports, summary = await self._run(
            config,
            [
                ("A?", "Vendor ships.", {"search_strategy": "local"}, {}),
                ("B?", "倉庫の情報はありません。", insufficient, {}),
            ],
        )
        manager = EvaluationManager(config, rag_chain=_FakeChain({}))
        manager.save_results(results, reports, summary, tmp_path)
        saved = {
            row["query_id"]: row
            for path in tmp_path.glob("evaluation_results_*.json")
            for row in json.loads(path.read_text(encoding="utf-8"))
        }
        assert saved["q0"]["abstention_reason"] is None
        assert saved["q0"]["context_sufficient"] is None
        assert saved["q0"]["insufficiency_reason"] is None
        assert saved["q0"]["iterative"] is None
        assert saved["q1"]["abstained"] is True
        assert saved["q1"]["abstention_reason"] == "insufficient_context"
        assert saved["q1"]["context_sufficient"] is False
        assert saved["q1"]["insufficiency_reason"] == "max_rounds"
        assert saved["q1"]["iterative"] == iterative
        # A saved file reads back (run-eval --from-results).
        assert EvaluationResult.model_validate(saved["q1"]) == results[1]

    async def test_guardrail_blocks_are_counted_per_blocked_text(
        self, config: Config
    ) -> None:
        def block(source: str) -> dict:
            record = {"source": source, "action": "blocked", "policies": []}
            return {"guardrail": [record]}

        masked = {"guardrail": [{"source": "input", "action": "masked"}]}
        results, _, summary = await self._run(
            config,
            [
                ("A?", "Vendor ships.", masked, {}),
                ("B?", "Sorry.", {**block("input"), "blocked": True}, {}),
                ("C?", "Sorry.", block("output"), {}),
                ("D?", "Sorry.", block("output"), {}),
            ],
        )
        assert [r.guardrail_blocked for r in results] == [
            None,
            "input",
            "output",
            "output",
        ]
        assert summary.guardrail_blocked_counts == {"input": 1, "output": 2}

    async def test_unanswerable_items_scored_on_abstention_only(
        self, config: Config
    ) -> None:
        no = {"answerable": False}
        results, reports, summary = await self._run(
            config,
            [
                ("A?", "Vendor ships.", {}, {}),
                ("B?", NO_CONTEXT_ANSWER, {"abstained": True}, no),
                ("C?", "A confident guess.", {}, no),
            ],
        )
        assert [r.query_id for r in reports] == ["q0"]
        # Same shape as the overall block: no "accuracy", since a decline in
        # the model's own words ("A confident guess." could be one) is not
        # detected and the rate is only a lower bound.
        assert summary.abstention_statistics["unanswerable"] == {
            "completed": 2,
            "abstained": 1,
            "answered": 1,
            "failed": 0,
            "abstention_rate": 0.5,
        }
        assert summary.metric_outcomes["answer_match"]["answer_contains"] == {
            "scored": 1,
            "failed": 0,
            "skipped": 2,
        }

    async def test_declines_are_scored_against_answerability(
        self, config: Config
    ) -> None:
        no = {"answerable": False}
        marked = {"abstained": True, "abstention_reason": "marker"}
        results, reports, summary = await self._run(
            config,
            [
                ("A?", "Vendor ships.", {}, {}),
                # A decline that names the gold entity is still not an answer.
                ("B?", "The context names Vendor but not the depot.", marked, {}),
                ("C?", "Buyer ships.", {}, {}),
                ("D?", "정보가 없습니다.", marked, no),
                ("E?", "A confident guess.", {}, no),
            ],
        )
        assert [r.abstained for r in results] == [False, True, False, True, False]
        by_id = {r.query_id: r for r in reports}
        assert by_id["q1"].metadata["abstained"] is True
        assert {m.value for m in by_id["q1"].metrics} == {0.0}
        stats = summary.abstention_statistics
        assert stats["unanswerable"]["abstained"] == 1
        assert stats["answerable"] == {
            "completed": 3,
            "abstained": 1,
            "answered": 2,
            "failed": 0,
            "abstention_rate": pytest.approx(1 / 3),
        }
        assert stats["precision"] == 0.5 and stats["recall"] == 0.5
        assert stats["f1"] == 0.5
        contains = stats["answer_match"]["answer_contains"]
        assert contains["coverage"] == pytest.approx(2 / 3)
        # The same "answered" share in both blocks.
        answerable = stats["answerable"]
        assert answerable["answered"] / answerable["completed"] == pytest.approx(
            contains["coverage"]
        )
        assert contains["accuracy"] == pytest.approx(1 / 3)
        assert contains["selective_accuracy"] == 0.5
        # A, plus the correct decline of D, over all five.
        assert contains["overall_accuracy"] == pytest.approx(2 / 5)

    def test_failed_answers_count_as_not_declined(self) -> None:
        """Regression: failed answers were dropped, so a strategy that failed
        on half the unanswerable items and declined the rest had recall 1.0
        (one that declined 8 of 10 had 0.8), and its failed answerable items
        left accuracy untouched."""

        def result(i: int, *, answerable: bool, **fields: Any) -> EvaluationResult:
            return EvaluationResult(
                query_id=f"{'a' if answerable else 'u'}{i}",
                question="q",
                generated_answer="" if fields.get("error") else "x",
                ground_truth="Vendor" if answerable else "",
                search_strategy="s",
                metadata={} if answerable else {"answerable": False},
                **fields,
            )

        declines = [result(i, answerable=False, abstained=i < 8) for i in range(10)]
        failures = [
            result(i, answerable=False, error=i < 5, abstained=i >= 5)
            for i in range(10)
        ]
        stats = EvaluationManager._calculate_abstention_statistics
        assert stats(declines)["recall"] == pytest.approx(0.8)
        failing = stats(failures)
        assert failing["recall"] == pytest.approx(0.5)
        assert failing["unanswerable"] == {
            "completed": 5,
            "abstained": 5,
            "answered": 0,
            "failed": 5,
            "abstention_rate": 0.5,
        }
        # A failed answerable item scores 0 in accuracy, as a wrong answer.
        answerable = [
            result(0, answerable=True),
            result(1, answerable=True, error=True),
        ]
        reports = [
            EvaluationReport(
                query_id="a0",
                evaluator_type=EvaluatorType.ANSWER_MATCH,
                metrics=[
                    EvaluationMetric(
                        metric_type=EvaluationMetricType.ANSWER_CONTAINS, value=1.0
                    )
                ],
            )
        ]
        block = stats(answerable, reports)["answer_match"]["answer_contains"]
        assert block == {
            "coverage": 1.0,
            "accuracy": 0.5,
            "selective_accuracy": 0.5,
        }

    async def test_no_decline_means_no_precision(self, config: Config) -> None:
        _, _, summary = await self._run(
            config,
            [
                ("A?", "Vendor ships.", {}, {}),
                ("B?", "Vendor.", {}, {"answerable": False}),
            ],
        )
        stats = summary.abstention_statistics
        assert stats["precision"] is None
        assert stats["recall"] == 0.0 and stats["f1"] == 0.0
        # Without unanswerable items: coverage and accuracy only.
        _, _, summary = await self._run(config, [("A?", "Vendor ships.", {}, {})])
        stats = summary.abstention_statistics
        assert "precision" not in stats
        assert stats["answer_match"]["answer_contains"] == {
            "coverage": 1.0,
            "accuracy": 1.0,
            "selective_accuracy": 1.0,
        }

    def test_answerable_must_be_boolean(self, tmp_path) -> None:
        path = tmp_path / "eval.json"
        path.write_text(
            json.dumps([{"question": "q", "metadata": {"answerable": "no"}}]),
            encoding="utf-8",
        )
        with pytest.raises(EvaluationException, match="answerable"):
            EvaluationManager.load_data(path)


def test_the_manager_loads_the_schema_source(tmp_path) -> None:
    # The run manifest hashes the loaded schema, not the default one.
    path = tmp_path / "schema.yaml"
    path.write_text('relationship_types: ["WORKS_FOR"]\n', encoding="utf-8")
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {"kind": "file", "path": str(path)}
                }
            }
        }
    )
    manager = _graph_aware_manager(config)
    extraction = manager.config.processing.graph_extraction
    assert extraction.schema_source.kind == "inline"
    assert extraction.relationship_types == ["WORKS_FOR"]
    assert [s.label for s in extraction.extraction_schema().relationship_types] == [
        "WORKS_FOR"
    ]
