# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What run-eval reports of the iterative strategy's runs: the rounds it
took, why it stopped and how often the round cap ended it (the setting an
iterative experiment tunes)."""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.domain.models import Config, EvaluationQuery, EvaluationResult
from unified_kg_rag.evaluation.comparison import paired_comparison
from unified_kg_rag.evaluation.iterative_statistics import iterative_statistics

pytestmark = pytest.mark.unit


def _result(qid: str, record: dict[str, Any] | None) -> EvaluationResult:
    return EvaluationResult(
        query_id=qid,
        question="?",
        generated_answer="a",
        ground_truth="",
        search_strategy="iterative",
        iterative=record,
    )


_RECORDS = [
    {"rounds": 0, "stop_reason": "sufficient", "degraded": False},
    {"rounds": 2, "stop_reason": "max_rounds", "degraded": False},
    {"rounds": 2, "stop_reason": "max_rounds", "degraded": False},
    {
        "rounds": 1,
        "stop_reason": "controller_unparsed",
        "degraded": "controller_unparsed",
    },
]


def test_rounds_stop_reasons_and_the_share_at_max_rounds() -> None:
    results = [_result(f"q{i}", r) for i, r in enumerate(_RECORDS)]
    stats = iterative_statistics([*results, _result("plain", None)])
    assert stats == {
        "queries": 4,
        "rounds": {"0": 1, "1": 1, "2": 2},
        "mean_rounds": 1.25,
        "stop_reasons": {"controller_unparsed": 1, "max_rounds": 2, "sufficient": 1},
        "max_rounds_rate": 0.5,
        "degraded": 1,
    }
    assert iterative_statistics([_result("plain", None)]) == {}


def test_the_summary_and_the_comparison_carry_it() -> None:
    from datetime import datetime

    from unified_kg_rag.domain.models import EvaluatorType
    from unified_kg_rag.evaluation import EvaluationManager

    results = [_result(f"q{i}", r) for i, r in enumerate(_RECORDS)]
    queries = [EvaluationQuery(query_id=r.query_id, question="?") for r in results]
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    manager = EvaluationManager(config, rag_chain=object())  # type: ignore[arg-type]
    now = datetime(2026, 1, 1)
    summary = manager._generate_summary(queries, results, [], now, now)
    assert summary.iterative_statistics["max_rounds_rate"] == 0.5
    comparison = paired_comparison(
        {"local": [([], [_result("q0", None)])], "iterative": [([], results)]}
    )
    assert comparison["strategies"]["iterative"]["iterative_statistics"]["queries"] == 4
    assert comparison["strategies"]["local"]["iterative_statistics"] == {}
