# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Evaluate page's per-question view of a saved run (no app)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unified_kg_rag.application.cli.console import evaluations

pytestmark = pytest.mark.unit


def _save(
    directory: Path,
    strategy: str,
    stamp: str,
    scores: dict[str, float | None],
    *,
    metric: str = "answer_contains",
) -> None:
    """One strategy's saved run: ``scores`` per query (None: the answer
    failed)."""
    results = [
        {
            "query_id": qid,
            "question": f"Question {qid}?",
            "generated_answer": (
                f"{strategy} answers {qid}" if value is not None else ""
            ),
            "ground_truth": "Vendor",
            "error": value is None,
            "error_message": "timed out" if value is None else None,
        }
        for qid, value in scores.items()
    ]
    reports = [
        {
            "query_id": qid,
            "evaluator_type": "answer_match",
            "metrics": [{"metric_type": metric, "value": value}],
        }
        for qid, value in scores.items()
        if value is not None
    ]
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{strategy}_{stamp}_0123abcd.json"
    (directory / f"evaluation_results_{name}").write_text(json.dumps(results))
    (directory / f"evaluation_reports_{name}").write_text(json.dumps(reports))


def test_each_question_has_every_strategy_score_and_answer(tmp_path: Path) -> None:
    _save(tmp_path, "local", "20261101_100000", {"q1": 1.0, "q2": 0.0})
    _save(tmp_path, "mix", "20261101_100001", {"q1": 0.0, "q2": None})
    view = evaluations.question_scores(tmp_path, None)
    assert view["metric"] == "answer_match/answer_contains"
    assert view["strategies"] == ["local", "mix"]
    by_id = {q["id"]: q for q in view["questions"]}
    assert by_id["q1"]["question"] == "Question q1?"
    assert by_id["q1"]["scores"] == {"local": 1.0, "mix": 0.0}
    assert by_id["q1"]["answers"]["mix"]["text"] == "mix answers q1"
    # A failed answer has no score and shows its error.
    assert by_id["q2"]["scores"] == {"local": 0.0}
    assert by_id["q2"]["answers"]["mix"]["error"] == "timed out"


def test_the_comparison_primary_metric_is_the_one_shown(tmp_path: Path) -> None:
    _save(tmp_path, "local", "20261101_100000", {"q1": 0.5}, metric="token_f1")
    _save(tmp_path, "mix", "20261101_100001", {"q1": 0.7}, metric="token_f1")
    comparison = {"multiplicity": {"primary_metric": "answer_match/token_f1"}}
    view = evaluations.question_scores(tmp_path, comparison)
    assert view["metric"] == "answer_match/token_f1"
    assert view["questions"][0]["scores"] == {"local": 0.5, "mix": 0.7}


def test_a_run_without_per_question_results_has_no_questions(tmp_path: Path) -> None:
    assert evaluations.question_scores(tmp_path, None) == {
        "metric": None,
        "strategies": [],
        "questions": [],
    }


def test_a_question_marked_unanswerable_says_so(tmp_path: Path) -> None:
    _save(tmp_path, "local", "20261101_100000", {"q1": 1.0, "q2": 0.0})
    path = next(tmp_path.glob("evaluation_results_*.json"))
    results = json.loads(path.read_text())
    results[1]["metadata"] = {"answerable": False}
    results[1]["abstained"] = True
    path.write_text(json.dumps(results))
    by_id = {
        q["id"]: q for q in evaluations.question_scores(tmp_path, None)["questions"]
    }
    assert by_id["q1"]["unanswerable"] is False
    assert by_id["q2"]["unanswerable"] is True
    assert by_id["q2"]["answers"]["local"]["declined"] == 1
    assert by_id["q2"]["answers"]["local"]["repeats"] == 1


def test_declines_are_counted_over_every_repeat(tmp_path: Path) -> None:
    """The decline badge reads every repeat, as the scores beside it do,
    not the first repeat's answer alone."""
    for repeat, declined in enumerate((False, True, True)):
        stamp = f"20261101_10000{repeat}"
        _save(tmp_path, "local", stamp, {"q1": 0.0})
        path = next(tmp_path.glob(f"evaluation_results_local_{stamp}_*.json"))
        results = json.loads(path.read_text())
        results[0]["metadata"] = {"answerable": False}
        results[0]["abstained"] = declined
        path.write_text(json.dumps(results))
    _save(tmp_path, "local", "20261101_100009", {"q1": None})  # failed: not read
    (question,) = evaluations.question_scores(tmp_path, None)["questions"]
    assert question["answers"]["local"]["declined"] == 2
    assert question["answers"]["local"]["repeats"] == 3
