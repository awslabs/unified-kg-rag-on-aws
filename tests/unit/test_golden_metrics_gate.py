# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The golden-metrics gate's comparison and its committed baseline
(``tests/fixtures/golden_metrics.py``; the gate itself is
``tests/integration/test_golden_metrics.py``, run by the ``golden-metrics``
CI job)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures import golden_metrics as gm
from unified_kg_rag.domain.models import SearchType
from unified_kg_rag.domain.prompts.retrieval import (
    CommunityRelevancePrompt,
    DriftActionPrompt,
    DriftPrimerPrompt,
    IterativeControllerPrompt,
    KeywordsExtractionPrompt,
)

pytestmark = pytest.mark.unit


def _result() -> dict[str, Any]:
    run = {
        "queries": 2,
        "failed": 0,
        "metrics": {"mrr": 0.5, "token_f1": 0.25},
        "answers": {"q1": "aaaaaaaaaa-bbbbbbbbbb", "q2": "cccccccccc-dddddddddd"},
    }
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


def test_an_answer_that_moves_no_mean_is_gated() -> None:
    # A saturated metric (mrr 1.0 either way) hides a changed ranking; the
    # answer's fingerprint does not.
    measured = _result()
    answers = measured["cases"]["split"]["test"]["hybrid/local"]["answers"]
    answers["q1"] = "eeeeeeeeee-bbbbbbbbbb"
    answers["q2"] = "cccccccccc-ffffffffff"
    measured["cases"]["split"]["tune"]["hybrid/local"]["answers"][
        "q1"
    ] = "aaaaaaaaaa-ffffffffff"
    del measured["cases"]["whole"]["all"]["vector/naive"]["answers"]["q2"]
    gated, reported = gm.compare(_result(), measured)
    assert gated == [
        "split [test] hybrid/local answer q1: the ranked sources changed",
        "split [test] hybrid/local answer q2: the context or the answer changed",
        "whole [all] vector/naive answer q2: only in the baseline",
    ]
    assert reported == [
        "split [tune] hybrid/local answer q1: the context or the answer changed"
    ]


def _answer(**fields: Any) -> dict[str, Any]:
    return {
        "query_id": "q1",
        "retrieved_source_ids": [["a.txt"], ["b.txt"]],
        "retrieved_source_types": ["text", "text"],
        "cited_source_ids": [["a.txt"]],
        "cited_section_types": ["text"],
        "retrieved_contexts": ["A.", "B."],
        "retrieved_entities": ["A"],
        "retrieved_relationships": [],
        "generated_answer": "A. [1]",
        "abstained": False,
        "error": False,
        "response_time": 0.5,
        "usage": {"llm_calls": 2},
        **fields,
    }


def test_the_fingerprint_reads_the_ranking_and_the_answer_not_the_timing() -> None:
    ranked, read = gm.fingerprint(_answer()).split("-")
    assert gm.fingerprint(_answer(response_time=9.0, usage={})) == f"{ranked}-{read}"
    # The same sources in another order are another ranking.
    swapped = gm.fingerprint(
        _answer(retrieved_source_ids=[["b.txt"], ["a.txt"]])
    ).split("-")
    assert swapped[0] != ranked and swapped[1] == read
    # The same ranking with another context or answer.
    for change in (
        {"retrieved_contexts": ["A.", "C."]},
        {"retrieved_entities": ["B"]},
        {"generated_answer": "B. [2]"},
        {"abstained": True},
    ):
        moved = gm.fingerprint(_answer(**change)).split("-")
        assert moved[0] == ranked and moved[1] != read, change


def test_another_schema_version_asks_for_a_new_baseline() -> None:
    measured = _result()
    measured["schema_version"] = gm.SCHEMA_VERSION + 1
    gated, _ = gm.compare(_result(), measured)
    assert len(gated) == 1 and "regenerate the baseline" in gated[0]


def test_select_keeps_one_split() -> None:
    selected = gm.select(_result(), "split", SearchType.HYBRID, "test")
    assert selected["cases"] == {
        "split": {
            "test": {
                "hybrid/local": _result()["cases"]["split"]["test"]["hybrid/local"]
            }
        }
    }


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
    for case, splits in baseline["cases"].items():
        types = gm.CASE_SEARCH_TYPES[case]
        runs = {f"{t.value}/{s.value}" for t in types for s in gm.STRATEGIES}
        if case == "structural-deep":
            runs |= {
                f"{t.value}/{strategy.value}@{arm}"
                for t in types
                for arm, (strategy, _) in gm.ARMS.items()
            }
        # The structural benchmarks are split (tune for development, test
        # gated); the bridge datasets are not.
        split = case.startswith("structural")
        assert set(splits) == ({"tune", "test"} if split else {"all"}), case
        for name, scored in splits.items():
            assert set(scored) == runs, (case, name)
            for run, score in scored.items():
                # A baseline with failed queries would gate a broken run.
                assert score["failed"] == 0 and score["queries"] > 0, (case, run)
                assert {"hit_at_k", "mrr", "ndcg_at_k", "token_f1"} <= set(
                    score["metrics"]
                ), (case, run)
                assert len(score["answers"]) == score["queries"], (case, run)


def test_the_gate_runs_every_case_search_type_and_split() -> None:
    assert {(case, t) for case, t, _ in gm.RUNS} == {
        (case, t) for case in gm.CASES for t in gm.CASE_SEARCH_TYPES[case]
    }
    assert {split for case, _, split in gm.RUNS if case.startswith("structural")} == {
        "tune",
        "test",
    }


def test_the_deep_case_keeps_the_first_questions_of_each_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def question(n: int, template: str, split: str) -> dict[str, Any]:
        return {"query_id": f"q{n}", "metadata": {"template": template, "split": split}}

    path = tmp_path / "eval_data.json"
    questions = [
        question(0, "count", "test"),
        question(1, "count", "tune"),
        question(2, "count", "test"),
        question(3, "top", "test"),
    ]
    path.write_text(json.dumps(questions), encoding="utf-8")
    # Per template and split, in the file's order.
    monkeypatch.setattr(gm, "DEEP_QUESTIONS_PER_TEMPLATE", 1)
    assert [q["query_id"] for q in gm._first_of_each_template(path)] == [
        "q0",
        "q1",
        "q3",
    ]
    monkeypatch.setattr(gm, "DEEP_QUESTIONS_PER_TEMPLATE", 2)
    assert len(gm._first_of_each_template(path)) == 4


# --- the fake responder reads the question ---------------------------------


@pytest.fixture
def responder() -> Any:
    return gm._responder(
        {"Acme": "ORGANIZATION", "Bolt": "ORGANIZATION", "Corr": "LOCATION"}
    )


def test_the_keywords_follow_the_question(responder: Any) -> None:
    assert responder.keywords("Which buyers does Acme supply?") == {
        "high_level_keywords": ["buyers", "supply"],
        "low_level_keywords": ["Acme"],
    }
    assert responder.keywords("Where is Bolt based?")["high_level_keywords"] == [
        "based"
    ]


def test_the_controller_asks_for_the_bridge_until_it_was_searched(
    responder: Any,
) -> None:
    context = "Acme supplies Bolt.\nCorr is far away."

    def verdict(searched: str) -> dict[str, Any]:
        human = (
            'QUESTION: "Who does Acme supply?"\n\nALREADY SEARCHED:\n'
            f"<searched>\n{searched}\n</searched>\n\nDATA:\n"
            f"<context>\n{context}\n</context>"
        )
        return dict(json.loads(responder.controller(human)))

    first = verdict("")
    assert first["sufficient"] is False
    assert first["bridge_entities"] == first["next_queries"] == ["Bolt"]
    assert verdict("Bolt") == {
        "sufficient": True,
        "known_facts": ["Acme supplies Bolt."],
        "bridge_entities": [],
        "next_queries": [],
    }


def test_a_drift_step_follows_up_the_next_entity(responder: Any) -> None:
    text = "Acme supplies Bolt.\nCorr is far away."
    step = responder.drift_step("Who does Acme supply?", text)
    assert step["follow_up_queries"] == ["Bolt"]
    # 20 per question token of the best sentence: "acme" ("supplies" is
    # another token than "supply").
    assert step["score"] == 20
    assert (
        responder.drift_step("Who does Acme supply?", text, "Bolt")["follow_up_queries"]
        == []
    )


def _rendered(prompt: Any, **values: str) -> tuple[str, str]:
    """``prompt``'s system and human messages, as the strategies send them."""
    filled = {name: values.get(name, "English") for name in prompt.input_variables}
    return (
        prompt.system_prompt_template.format(**filled),
        prompt.human_prompt_template.format(**filled),
    )


def test_the_responder_reads_the_question_from_the_real_prompts(
    responder: Any,
) -> None:
    # A prompt whose wording moves away from what the responder looks for
    # fails here, not silently in the gate (a step that no longer reads the
    # question pins nothing).
    question = "Who does Acme supply?"
    context = "Acme supplies Bolt.\nCorr is far away."

    def respond(prompt: Any, **values: str) -> dict[str, Any]:
        reply, _ = responder.respond(
            *_rendered(prompt, **{"query": question, **values})
        )
        return dict(json.loads(reply))

    assert respond(KeywordsExtractionPrompt) == {
        "high_level_keywords": ["supply"],
        "low_level_keywords": ["Acme"],
    }
    assert respond(CommunityRelevancePrompt, community_summary=context)["rating"] == 1
    controller = respond(
        IterativeControllerPrompt, context=context, searched="", max_next_queries="2"
    )
    assert controller["sufficient"] is False
    assert controller["next_queries"] == ["Bolt"]
    primer = respond(DriftPrimerPrompt, community_reports=context, num_follow_ups="3")
    assert primer["follow_up_queries"] == ["Bolt"]
    assert primer["intermediate_answer"].startswith("Acme supplies Bolt.")
    action = respond(
        DriftActionPrompt,
        query="Bolt",
        global_query=question,
        context=context,
        num_follow_ups="3",
    )
    # The follow-up names Bolt already: nothing further to follow.
    assert action["follow_up_queries"] == [] and action["score"] == 20
