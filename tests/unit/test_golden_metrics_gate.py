# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The golden-metrics gate's comparison and its committed baseline
(``tests/fixtures/golden_metrics.py``; the gate itself is
``tests/integration/test_golden_metrics.py``, run by the ``golden-metrics``
CI job)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.fixtures import golden_metrics as gm
from unified_kg_rag.domain.models import SearchType

pytestmark = pytest.mark.unit


def _result() -> dict[str, Any]:
    run = {"queries": 2, "failed": 0, "metrics": {"mrr": 0.5, "token_f1": 0.25}}
    return {
        "schema_version": gm.SCHEMA_VERSION,
        "cases": {
            "split": {
                "tune": {"hybrid/local": copy.deepcopy(run)},
                "test": {"hybrid/local": copy.deepcopy(run)},
            },
            "whole": {"all": {"vector/naive": copy.deepcopy(run)}},
        },
    }


def test_equal_results_differ_nowhere() -> None:
    assert gm.compare(_result(), _result()) == ([], [])


def test_the_test_half_and_an_unsplit_dataset_are_gated_exactly() -> None:
    measured = _result()
    # Any difference counts, however small: the run is deterministic.
    measured["cases"]["split"]["test"]["hybrid/local"]["metrics"]["mrr"] = 0.5 + 1e-12
    measured["cases"]["whole"]["all"]["vector/naive"]["failed"] = 1
    gated, reported = gm.compare(_result(), measured)
    assert gated == [
        f"split [test] hybrid/local mrr: 0.5 -> {0.5 + 1e-12!r}",
        "whole [all] vector/naive failed: 0 -> 1",
    ]
    assert reported == []


def test_the_tune_half_is_reported_not_gated() -> None:
    measured = _result()
    measured["cases"]["split"]["tune"]["hybrid/local"]["metrics"]["token_f1"] = 0.0
    gated, reported = gm.compare(_result(), measured)
    assert gated == []
    assert reported == ["split [tune] hybrid/local token_f1: 0.25 -> 0.0"]


def test_a_missing_run_metric_or_case_is_gated() -> None:
    measured = _result()
    del measured["cases"]["whole"]["all"]["vector/naive"]["metrics"]["mrr"]
    measured["cases"]["split"]["test"]["lexical/local"] = measured["cases"]["split"][
        "test"
    ]["hybrid/local"]
    measured["cases"]["extra"] = {"all": {}}
    gated, _ = gm.compare(_result(), measured)
    assert gated == [
        "extra: only in the run",
        "split [test] lexical/local: only in the run",
        "whole [all] vector/naive mrr: only in the baseline",
    ]


def test_another_schema_version_asks_for_a_new_baseline() -> None:
    measured = _result()
    measured["schema_version"] = gm.SCHEMA_VERSION + 1
    gated, _ = gm.compare(_result(), measured)
    assert len(gated) == 1 and "regenerate the baseline" in gated[0]


def test_select_keeps_one_case_and_search_type() -> None:
    selected = gm.select(_result(), "split", SearchType.HYBRID)
    assert selected["cases"] == {
        "split": {
            "tune": {
                "hybrid/local": _result()["cases"]["split"]["tune"]["hybrid/local"]
            },
            "test": {
                "hybrid/local": _result()["cases"]["split"]["test"]["hybrid/local"]
            },
        }
    }
    assert gm.select(_result(), "split", SearchType.VECTOR)["cases"] == {
        "split": {"tune": {}, "test": {}}
    }


def test_the_baseline_covers_every_case_strategy_and_search_type() -> None:
    baseline = gm.read_baseline()
    assert baseline["schema_version"] == gm.SCHEMA_VERSION
    assert set(baseline["cases"]) == set(gm.CASES)
    runs = {f"{t.value}/{s.value}" for t in gm.SEARCH_TYPES for s in gm.STRATEGIES}
    for case, splits in baseline["cases"].items():
        # The structural benchmark is split (tune for development, test
        # gated); the bridge datasets are not.
        expected = {"tune", "test"} if case == "structural" else {"all"}
        assert set(splits) == expected, case
        for split, scored in splits.items():
            assert set(scored) == runs, (case, split)
            for run, score in scored.items():
                # A baseline with failed queries would gate a broken run.
                assert score["failed"] == 0 and score["queries"] > 0, (case, run)
                assert {"hit_at_k", "mrr", "ndcg_at_k", "token_f1"} <= set(
                    score["metrics"]
                ), (case, run)
