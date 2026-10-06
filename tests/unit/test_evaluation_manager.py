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
from datetime import datetime

import pytest

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
    SearchQuery,
    SearchResult,
)
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.evaluation.evaluation_manager import GraphAwareEvaluator
from unified_kg_rag.shared import EvaluationException

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

    def test_resolver_covers_all_types(self) -> None:
        # The lazy resolver returns a class for every evaluator type (langchain/
        # ragas are imported on demand to avoid a circular import at module load).
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.LANGCHAIN).__name__
            == "LangChainEvaluator"
        )
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.RAGAS).__name__
            == "RagasEvaluator"
        )
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.GRAPH_AWARE)
            is GraphAwareEvaluator
        )

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

    def test_init_failure_of_one_evaluator_does_not_crash(
        self, config: Config, mocker
    ) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("init failed")

        mocker.patch.object(
            EvaluationManager, "_resolve_evaluator_class", return_value=_Boom
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
        # Failure is caught and logged (not raised) and recorded per query as a
        # failed report, so the summary can count the metrics as failed.
        reports = await manager._evaluate_results([query], [result], [gt])
        assert len(reports) == 1
        assert reports[0].metrics == []
        assert reports[0].metadata["evaluation_failed"] is True
        assert set(reports[0].metadata["failed_metrics"]) == {
            "entity_coverage",
            "relationship_coverage",
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

    async def abatch(self, inputs, config=None, return_exceptions=False):
        return [self.outputs[i["query"]] for i in inputs]


class TestErroredQueries:
    async def test_batch_failure_is_not_retried_with_backoff(
        self, config: Config
    ) -> None:
        config.processing.max_concurrency = 3
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]

        class _FailingBatchChain:
            batch_calls = 0

            async def abatch(self, inputs, config=None, return_exceptions=False):
                type(self).batch_calls += 1
                raise ValueError("invalid filter")  # deterministic, not transient

            async def ainvoke(self, inputs, config=None):
                return _rag_output("Vendor ships.", {"processing_time": 0.1})

        manager = EvaluationManager(config, rag_chain=_FailingBatchChain())
        assert manager.batch_processor.max_retries == 1
        assert manager.batch_processor.max_concurrency == 3
        results, _, _ = await manager.evaluate_dataset(
            [EvaluationQuery(query_id="q1", question="Who ships?")],
            [EvaluationGroundTruth(query_id="q1", ground_truth="Vendor")],
            show_progress=False,
        )
        # One batch attempt, then the per-item sequential fallback answers it.
        assert _FailingBatchChain.batch_calls == 1
        assert results[0].generated_answer == "Vendor ships."

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

    def test_empty_sentinel_is_an_error(self) -> None:
        assert EvaluationManager._detect_generation_error({}, {}) is not None
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
            evaluator_type=EvaluatorType.LANGCHAIN,
            metrics=[
                EvaluationMetric(metric_type=EvaluationMetricType.CORRECTNESS, value=v)
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
            self._report(metadata={"failed_metrics": {"correctness": "boom"}}),
            self._report(metadata={"skipped_metrics": {"correctness": "empty"}}),
        ]
        stats = manager._calculate_metric_statistics(reports)
        assert stats["correctness"]["mean"] == pytest.approx(0.8)
        assert stats["correctness"]["count"] == 1
        outcomes = manager._calculate_metric_outcomes([], reports)
        assert outcomes == {
            "langchain": {"correctness": {"scored": 1, "failed": 1, "skipped": 1}}
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
    async def _run(self, config: Config, strategies: dict[str, str]):
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
        assert manifest["models"]["answer_generation"] == (
            config.search.answer_generation_model_id
        )
        assert manifest["models"]["evaluation_judge"] is None  # no LLM judge
        assert manifest["enabled_evaluators"] == ["graph_aware"]
        assert manifest["package_version"] and manifest["created_at"]
        assert len(manifest["config_sha256"]) == 64
        assert manifest["library_versions"]["ragas"]
        assert manifest["library_versions"]["langchain-core"]
        assert "git_sha" in manifest
        json.dumps(manifest)  # serializable as-is

    def test_config_hash_tracks_resolved_config(self, config: Config) -> None:
        config.evaluation.enabled_evaluators = []
        manager = EvaluationManager(config, rag_chain=object())
        before = manager.build_run_manifest()["config_sha256"]
        config.search.answer_generation_model_id = "another-model"
        assert manager.build_run_manifest()["config_sha256"] != before

    def test_git_sha_none_without_git(self, mocker) -> None:
        mocker.patch(
            "unified_kg_rag.evaluation.evaluation_manager.shutil.which",
            return_value=None,
        )
        assert EvaluationManager._git_sha() is None

    async def test_evaluate_dataset_attaches_manifest(self, config: Config) -> None:
        manager, results, reports, summary = await self._run(
            config, {"A?": "local", "B?": "local"}
        )
        manifest = summary.run_manifest
        assert manifest["dataset"]["num_queries"] == 2
        assert len(manifest["dataset"]["content_sha256"]) == 64
        assert "path" not in manifest["dataset"]  # library call: no file


class TestAbstention:
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
                # Detected from the exact text when the flag is absent.
                ("C?", NO_CONTEXT_ANSWER, {"search_strategy": "global"}, {}),
            ],
        )
        assert [r.abstained for r in results] == [False, True, True]
        stats = summary.abstention_statistics
        assert stats["abstained"] == 2 and stats["answered"] == 3
        assert stats["abstention_rate"] == pytest.approx(2 / 3)
        assert stats["per_strategy"]["local"]["abstention_rate"] == 0.5
        assert stats["per_strategy"]["global"]["abstention_rate"] == 1.0
        assert "unanswerable" not in stats
        # An abstention on an answerable item is still graded (a miss).
        assert summary.metric_statistics["answer_contains"]["count"] == 3

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
        assert summary.abstention_statistics["unanswerable"] == {
            "total": 2,
            "correct_abstentions": 1,
            "accuracy": 0.5,
        }
        assert summary.metric_outcomes["answer_match"]["answer_contains"] == {
            "scored": 1,
            "failed": 0,
            "skipped": 2,
        }

    def test_answerable_must_be_boolean(self, tmp_path) -> None:
        path = tmp_path / "eval.json"
        path.write_text(
            json.dumps([{"question": "q", "metadata": {"answerable": "no"}}]),
            encoding="utf-8",
        )
        with pytest.raises(EvaluationException, match="answerable"):
            EvaluationManager.load_data(path)
