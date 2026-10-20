# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired strategy comparison, pairwise judge and benchmark preparation."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import random
from datetime import datetime
from pathlib import Path

import pytest

from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.domain.prompts import PairwiseComparisonPrompt
from unified_kg_rag.evaluation import pairwise
from unified_kg_rag.evaluation.answer_match_evaluator import AnswerMatchEvaluator
from unified_kg_rag.evaluation.benchmarks import musique
from unified_kg_rag.evaluation.comparison import bootstrap_ci, paired_comparison
from unified_kg_rag.evaluation.text_matching import normalize_answer_squad
from unified_kg_rag.shared import EvaluationException

pytestmark = pytest.mark.unit


def _run(scores: dict[str, float], strategy: str = "local"):
    reports = [
        EvaluationReport(
            query_id=qid,
            evaluator_type=EvaluatorType.ANSWER_MATCH,
            metrics=[
                EvaluationMetric(metric_type=EvaluationMetricType.TOKEN_F1, value=v)
            ],
            evaluation_time=datetime(2026, 1, 1),
        )
        for qid, v in scores.items()
    ]
    results = [
        EvaluationResult(
            query_id=qid,
            question="?",
            generated_answer="a",
            ground_truth="",
            search_strategy=strategy,
        )
        for qid in scores
    ]
    return reports, results


def _result(qid: str, answer: str, error: bool = False) -> EvaluationResult:
    return EvaluationResult(
        query_id=qid,
        question="q?",
        generated_answer=answer,
        ground_truth="",
        error=error,
    )


class TestPairedComparison:
    def test_pairs_only_queries_both_scored(self) -> None:
        out = paired_comparison(
            {
                "local": [_run({"a": 0.2, "b": 0.4, "c": 1.0})],
                "mix": [_run({"a": 0.6, "b": 0.4}, "mix")],
            }
        )
        delta = out["deltas_vs_baseline"]["mix"]["answer_match/token_f1"]
        assert delta["pairs"] == 2
        assert delta["mean_delta"] == pytest.approx(0.2)
        assert (delta["wins"], delta["ties"], delta["losses"]) == (1, 1, 0)
        assert delta["per_query"] == {"a": pytest.approx(0.4), "b": 0.0}

    def test_a_failed_answer_scores_zero_and_is_reported(self) -> None:
        # Before, "mix" failing on the query it would lose dropped that query
        # from the comparison and "mix" looked as good as "local" (survivor
        # bias).
        reports, results = _run({"easy": 1.0}, "mix")
        results.append(_result("hard", "", error=True))
        reports.append(
            EvaluationReport(
                query_id="easy",
                evaluator_type=EvaluatorType.GRAPH_AWARE,
                metrics=[
                    EvaluationMetric(
                        metric_type=EvaluationMetricType.ANSWER_LENGTH, value=9.0
                    )
                ],
                evaluation_time=datetime(2026, 1, 1),
            )
        )
        local = _run({"easy": 1.0, "hard": 1.0})
        local[0].extend(
            reports[-1].model_copy(update={"query_id": qid}) for qid in ("easy", "hard")
        )
        out = paired_comparison({"local": [local], "mix": [(reports, results)]})
        mix = out["strategies"]["mix"]
        assert mix["metrics"]["answer_match/token_f1"]["mean"] == 0.5
        assert (mix["failed_answers"], mix["failed_queries"]) == (1, ["hard"])
        assert out["strategies"]["local"]["failed_answers"] == 0
        delta = out["deltas_vs_baseline"]["mix"]["answer_match/token_f1"]
        assert (delta["pairs"], delta["unpaired"]) == (2, 0)
        assert delta["mean_delta"] == -0.5
        # answer_length is a diagnostic: a failed answer gets no value for it.
        length = out["deltas_vs_baseline"]["mix"]["graph_aware/answer_length"]
        assert (length["pairs"], length["unpaired"]) == (1, 1)
        assert out["warnings"] and "local 0, mix 1" in out["warnings"][0]

    def test_blocked_answers_are_counted_and_warned(self) -> None:
        # Before, a strategy whose answers the guardrail blocked showed no
        # trace of it in the comparison.
        reports, results = _run({"a": 1.0, "b": 0.0}, "mix")
        results[1].guardrail_blocked = "output"
        out = paired_comparison(
            {"local": [_run({"a": 1.0, "b": 1.0})], "mix": [(reports, results)]}
        )
        assert out["strategies"]["mix"]["guardrail_blocked_counts"] == {"output": 1}
        assert out["strategies"]["local"]["guardrail_blocked_counts"] == {}
        (warning,) = [w for w in out["warnings"] if "guardrail" in w]
        assert "local 0, mix 1" in warning
        # Not "scored on the blocked message" for every metric: the retrieval
        # and context metrics of an output block are not scored at all.
        assert "no retrieval or context score (unpaired)" in warning

    def test_equal_failure_counts_raise_no_warning(self) -> None:
        out = paired_comparison(
            {"local": [_run({"a": 1.0})], "mix": [_run({"a": 0.0})]}
        )
        assert out["warnings"] == []

    def test_repeats_are_averaged_per_query(self) -> None:
        out = paired_comparison(
            {
                "local": [_run({"a": 0.0}), _run({"a": 1.0})],
                "mix": [_run({"a": 1.0}, "mix"), _run({"a": 1.0}, "mix")],
            }
        )
        local = out["strategies"]["local"]["metrics"]["answer_match/token_f1"]
        assert local["mean"] == 0.5
        assert local["repeat_stdev"] == pytest.approx(0.7071, abs=1e-4)
        assert out["deltas_vs_baseline"]["mix"]["answer_match/token_f1"][
            "mean_delta"
        ] == pytest.approx(0.5)

    def test_bootstrap_ci_covers_a_known_mean_and_is_seeded(self) -> None:
        rng = random.Random(1)
        deltas = [0.1 + rng.gauss(0, 0.3) for _ in range(200)]
        low, high = bootstrap_ci(deltas, resamples=2000, confidence=0.95, seed=0)
        assert low < sum(deltas) / len(deltas) < high
        assert low < 0.1 < high
        assert (low, high) == bootstrap_ci(
            deltas, resamples=2000, confidence=0.95, seed=0
        )
        # A pure-noise difference: the CI straddles 0.
        noise = [rng.gauss(0, 0.3) for _ in range(200)]
        low, high = bootstrap_ci(noise, resamples=2000, confidence=0.95, seed=0)
        assert low < 0 < high


class _FakeChain:
    """Judge that always prefers the answer shown first (position bias),
    except that it prefers "detailed" on comprehensiveness wherever shown."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, inputs):
        self.calls.append(inputs)
        first_detailed = inputs["answer1"] == "detailed"
        verdict = {c: {"winner": "1"} for c in pairwise.CRITERIA}
        verdict["comprehensiveness"] = {"winner": "1" if first_detailed else "2"}
        return verdict


class TestPairwiseJudge:
    def _judge(self, mocker) -> pairwise.PairwiseJudge:
        mocker.patch.object(pairwise, "setup_chain", return_value=_FakeChain())
        mocker.patch.object(pairwise.Providers, "resolve")
        return pairwise.PairwiseJudge(Config())

    def test_position_bias_is_a_tie_and_a_stable_verdict_wins(self, mocker) -> None:
        judge = self._judge(mocker)
        outcome = asyncio.run(judge.ajudge("q?", "short", "detailed"))
        assert outcome == {
            "comprehensiveness": "candidate",
            "diversity": "tie",
            "empowerment": "tie",
            "directness": "tie",
        }
        # Both orders were asked.
        assert [c["answer1"] for c in judge.chain.calls] == ["short", "detailed"]

    def test_a_failed_answer_loses_the_pair_unjudged(self, mocker) -> None:
        # Before, a candidate that failed on a query was left out of the
        # pairs, so failing on hard queries could only raise its win rate.
        judge = self._judge(mocker)
        out = asyncio.run(
            judge.acompare(
                [_result("a", "short"), _result("b", "short"), _result("c", "x", True)],
                [
                    _result("a", "detailed"),
                    _result("b", "x", error=True),
                    _result("c", "x", error=True),
                ],
            )
        )
        assert (out["pairs"], out["judged"], out["failed"]) == (3, 1, 0)
        assert (out["candidate_errors"], out["baseline_errors"]) == (1, 0)
        assert out["both_errors"] == 1
        assert len(judge.chain.calls) == 2  # only pair "a", in both orders
        assert out["per_query"]["b"] == dict.fromkeys(pairwise.CRITERIA, "baseline")
        assert out["criteria"]["comprehensiveness"]["candidate_win_rate"] == 0.5
        assert out["criteria"]["directness"]["candidate_win_rate"] == 0.25

    def test_a_malformed_verdict_fails_the_pair(self, mocker) -> None:
        # Before, a missing criterion became a silent tie.
        judge = self._judge(mocker)
        judge.chain.ainvoke = mocker.AsyncMock(
            return_value={"verdicts": {"comprehensiveness": {"winner": "2"}}}
        )
        out = asyncio.run(judge.acompare([_result("a", "x")], [_result("a", "y")]))
        assert (out["judged"], out["failed"]) == (0, 1)
        assert out["criteria"]["comprehensiveness"]["candidate_win_rate"] is None

    def test_judge_calls_nest_under_one_span(self, mocker) -> None:
        judge = self._judge(mocker)
        span = mocker.patch.object(pairwise.observability, "span")
        asyncio.run(judge.acompare([_result("a", "short")], [_result("a", "long")]))
        span.assert_called_once_with(
            "kg_rag.evaluate", evaluator="pairwise_judge", items=1
        )

    def test_judge_failures_count_against_the_failure_budget(self, mocker) -> None:
        from unified_kg_rag.application.cli.run_evaluation import (
            pairwise_budget_breaches,
        )

        judge = self._judge(mocker)
        judge.chain.ainvoke = mocker.AsyncMock(side_effect=RuntimeError("throttled"))
        out = asyncio.run(judge.acompare([_result("a", "x")], [_result("a", "y")]))
        assert (out["judged"], out["failed"]) == (0, 1)
        # Every judge call failed: a breach even at the default rate of 1.0.
        assert pairwise_budget_breaches({"mix": out}, 1.0) == [
            "pairwise judge mix: 1/1 failed"
        ]
        assert pairwise_budget_breaches({"mix": {**out, "judged": 3}}, 0.5) == []

    def test_prompt_templates_declare_their_variables(self) -> None:
        resolved = PairwiseComparisonPrompt.resolve()
        for name in PairwiseComparisonPrompt.input_variables:
            assert "{" + name + "}" in resolved.human_prompt_template


class TestOfficialSquadNormalization:
    def test_matches_the_official_script(self) -> None:
        assert normalize_answer_squad("The  Eiffel-Tower!") == "eiffeltower"
        assert normalize_answer_squad("an Apple, a day") == "apple day"
        # No Unicode folding, unlike the default normalization.
        assert normalize_answer_squad("Ｒome") == "ｒome"

    @pytest.mark.parametrize(
        ("mode", "em"), [("default", 1.0), ("official_squad", 0.0)]
    )
    def test_answer_match_uses_the_configured_normalization(
        self, mode: str, em: float
    ) -> None:
        # The default keeps the minus sign and splits the dashed range; the
        # official script deletes both dashes.
        config = Config()
        config.evaluation.answer_normalization = mode
        result = EvaluationResult(
            query_id="q", question="?", generated_answer="1990 1995", ground_truth=""
        )
        report = AnswerMatchEvaluator(config).evaluate_single(
            EvaluationQuery(query_id="q", question="?"), result, "1990-1995"
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["exact_match"] == em


def _musique_line(qid: str, supporting: list[int], titles: list[str]) -> str:
    return json.dumps(
        {
            "id": qid,
            "question": f"Who supplies the buyer in {qid}?",
            "answer": "Vendor",
            "answer_aliases": ["The Vendor"],
            "answerable": True,
            "paragraphs": [
                {
                    "idx": i,
                    "title": title,
                    "paragraph_text": f"{title} paragraph {i}.",
                    "is_supporting": i in supporting,
                }
                for i, title in enumerate(titles)
            ],
        }
    )


class TestMusique:
    def _source(self, tmp_path: Path) -> Path:
        source = tmp_path / "in" / "synthetic.jsonl"
        source.parent.mkdir()
        lines = [
            _musique_line("2hop__1_2", [0, 1], ["Vendor", "Buyer", "Depot"]),
            # The same title with other text gets its own file; a/b is unsafe.
            _musique_line("3hop1__3_4_5", [2], ["Carrier", "Vendor", "A/B"]),
            json.dumps({"id": "x", "answerable": False, "paragraphs": []}),
        ]
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return source

    def test_prepare_writes_corpus_and_dataset(self, tmp_path: Path) -> None:
        out = tmp_path / "bench"
        counts = musique.prepare(self._source(tmp_path), out)
        assert counts == {
            "benchmark": "musique-full-answerable",
            "questions": 2,
            "paragraphs": 6,
        }
        files = sorted(p.name for p in (out / "corpus").iterdir())
        assert files == [
            "A B.txt",
            "Buyer.txt",
            "Carrier.txt",
            "Depot.txt",
            "Vendor (2).txt",
            "Vendor.txt",
        ]
        assert (out / "corpus" / "Vendor.txt").read_text() == (
            "Vendor\n\nVendor paragraph 0.\n"
        )
        data = json.loads((out / "eval_data.json").read_text())
        assert data[0]["reference_sources"] == ["Vendor.txt", "Buyer.txt"]
        assert data[1]["reference_sources"] == ["A B.txt"]
        assert [d["category"] for d in data] == ["2hop", "3hop"]
        assert data[0]["answer_aliases"] == ["The Vendor"]
        # Loads as a run-eval dataset.
        from unified_kg_rag.evaluation import EvaluationManager

        queries, truths = EvaluationManager.load_data(out / "eval_data.json")
        assert (
            len(queries) == 2
            and truths[0].reference_sources == data[0]["reference_sources"]
        )

    def test_a_title_with_a_leading_dot_is_not_a_hidden_file(self) -> None:
        # Before, ".hack" became ".hack.txt", which the parser skips.
        taken: set[str] = set()
        assert musique._file_name(".hack", taken) == "hack.txt"
        assert musique._file_name(". . .", taken) == "paragraph.txt"

    def test_file_names_differ_under_reference_matching(self) -> None:
        # Before, "A_B" and "A B" got files the retrieval evaluator reads as
        # the same reference, so retrieving one counted as the other.
        taken: set[str] = set()
        assert musique._file_name("A_B", taken) == "A_B.txt"
        assert musique._file_name("A B", taken) == "A B (2).txt"
        assert musique._file_name("ＡＢ", taken) == "ＡＢ.txt"
        assert musique._file_name("ab", taken) == "ab (2).txt"

    def test_the_full_file_pairs_an_id_with_its_unanswerable_twin(
        self, tmp_path: Path, caplog
    ) -> None:
        # Before, musique_full_v1.0_dev.jsonl was refused as repeating every
        # id: it carries each question once answerable, once not.
        source = tmp_path / "full.jsonl"
        answerable = _musique_line("2hop__1_2", [0], ["Vendor", "Buyer"])
        twin = json.loads(answerable)
        twin["answerable"] = False
        source.write_text(f"{answerable}\n{json.dumps(twin)}\n", encoding="utf-8")
        counts = musique.prepare(source, tmp_path / "out")
        assert counts["questions"] == 1
        assert counts["benchmark"] == "musique-full-answerable"
        # Its answerable half is not MuSiQue-Ans: labelled apart (before,
        # "musique"), with a WARNING that the scores are not comparable.
        data = json.loads((tmp_path / "out" / "eval_data.json").read_text())
        assert data[0]["metadata"]["benchmark"] == "musique-full-answerable"
        assert "not comparable to MuSiQue-Ans" in caplog.text

        # A MuSiQue-Ans file (answerable items only) keeps its name.
        source.write_text(f"{answerable}\n", encoding="utf-8")
        musique.prepare(source, tmp_path / "ans")
        data = json.loads((tmp_path / "ans" / "eval_data.json").read_text())
        assert data[0]["metadata"]["benchmark"] == "musique"

    def test_limit_samples_with_the_seed_and_refuses_a_used_corpus(
        self, tmp_path: Path
    ) -> None:
        source = self._source(tmp_path)
        musique.prepare(source, tmp_path / "one", limit=1, seed=3)
        musique.prepare(source, tmp_path / "two", limit=1, seed=3)
        assert (tmp_path / "one" / "eval_data.json").read_text() == (
            tmp_path / "two" / "eval_data.json"
        ).read_text()
        with pytest.raises(EvaluationException, match="not empty"):
            musique.prepare(source, tmp_path / "one")

    def test_download_checks_the_pinned_digest(self, tmp_path: Path, mocker) -> None:
        body = b'{"id": "2hop__1_2"}\n'
        mocker.patch.object(
            musique.urllib.request, "urlopen", return_value=io.BytesIO(body)
        )
        with pytest.raises(EvaluationException, match="SHA-256"):
            musique.download_dev(tmp_path / "dev.jsonl")
        assert not (tmp_path / "dev.jsonl").exists()
        mocker.patch.object(musique, "DEV_SHA256", hashlib.sha256(body).hexdigest())
        mocker.patch.object(
            musique.urllib.request, "urlopen", return_value=io.BytesIO(body)
        )
        assert musique.download_dev(tmp_path / "dev.jsonl").read_bytes() == body


def test_pairwise_prompt_override_must_keep_both_answers() -> None:
    with pytest.raises(ValueError, match="answer2"):
        Config(custom_prompts={"pairwise_comparison_human": "{query} {answer1}"})


def test_prepare_benchmark_cli(tmp_path: Path, capsys) -> None:
    from unified_kg_rag.application.cli import prepare_benchmark

    source = TestMusique()._source(tmp_path)
    prepare_benchmark.main(
        [
            "musique",
            "--output-directory",
            str(tmp_path / "b"),
            "--data-path",
            str(source),
            "--output-format",
            "json",
        ]
    )
    assert (tmp_path / "b" / "eval_data.json").is_file()
    # The label the rows carry, on stdout (--json) and on the console.
    out = capsys.readouterr()
    assert json.loads(out.out)["benchmark"] == "musique-full-answerable"
    assert "labelled 'musique-full-answerable'" in " ".join(out.err.split())
    with pytest.raises(SystemExit) as exit_info:  # corpus/ is no longer empty
        prepare_benchmark.main(
            [
                "musique",
                "--output-directory",
                str(tmp_path / "b"),
                "--data-path",
                str(source),
            ]
        )
    assert exit_info.value.code == 2  # a usage error


def test_prepare_benchmark_refuses_a_used_directory_before_downloading(
    tmp_path: Path, mocker
) -> None:
    from unified_kg_rag.application.cli import prepare_benchmark

    (tmp_path / "b" / "corpus").mkdir(parents=True)
    (tmp_path / "b" / "corpus" / "old.txt").write_text("x")
    download = mocker.patch.object(musique, "download_dev")
    with pytest.raises(SystemExit) as exit_info:
        prepare_benchmark.main(["musique", "--output-directory", str(tmp_path / "b")])
    assert exit_info.value.code == 2
    download.assert_not_called()
