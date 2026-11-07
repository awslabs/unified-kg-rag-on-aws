# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The synthetic structural benchmark (``prepare-benchmark structural``;
docs/design.md §9.4) and the structured statistics run-eval reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unified_kg_rag.application.bootstrap import load_seed_graphs
from unified_kg_rag.application.cli import prepare_benchmark
from unified_kg_rag.application.config import get_config
from unified_kg_rag.domain.models import EvaluationResult
from unified_kg_rag.domain.models.graph_query import GraphQueryPlan
from unified_kg_rag.evaluation.benchmarks import structural
from unified_kg_rag.evaluation.structured_statistics import structured_statistics
from unified_kg_rag.shared import EvaluationException

pytestmark = pytest.mark.unit


def _dataset(path: Path) -> list[dict]:
    return json.loads((path / "eval_data.json").read_text("utf-8"))


def test_a_seed_generates_one_world(tmp_path: Path) -> None:
    first = structural.prepare(tmp_path / "a", seed=1, size="small")
    again = structural.prepare(tmp_path / "b", seed=1, size="small")
    other = structural.prepare(tmp_path / "c", seed=2, size="small")
    assert first == again and other["seed"] == 2
    for name in ("eval_data.json", "seed/structural.jsonl"):
        assert (tmp_path / "a" / name).read_text() == (
            tmp_path / "b" / name
        ).read_text()
    assert (tmp_path / "a" / "seed/structural.jsonl").read_text() != (
        tmp_path / "c" / "seed/structural.jsonl"
    ).read_text()
    assert first["questions"] == len(_dataset(tmp_path / "a")) > 20
    assert set(first["templates"]) == set(structural.TEMPLATES)


def test_gold_answers_are_what_the_world_states() -> None:
    world = structural.generate_world(seed=4, size="medium")
    found = structural.questions(world, seed=4, size="medium")
    by_name = {node.name: node for node in world.nodes.values()}
    checked = 0
    for q in found:
        if q.template == "count_suppliers":
            buyer = by_name[
                q.question.removeprefix("How many vendors supply ").rstrip("?")
            ]
            assert int(q.answer) == len(set(world.sources(buyer.id, "SUPPLIES")))
        elif q.template == "supplies_yes_no":
            vendor, buyer = (
                q.question.removeprefix("Does ").rstrip("?").split(" supply ")
            )
            supplied = by_name[buyer].id in world.targets(
                by_name[vendor].id, "SUPPLIES"
            )
            assert q.answer == ("yes" if supplied else "no")
        elif q.template == "shared_buyer":
            pair = q.question.removeprefix("Which buyer do both ").removesuffix(
                " supply?"
            )
            a, b = (by_name[n].id for n in pair.split(" and "))
            shared = set(world.targets(a, "SUPPLIES")) & set(
                world.targets(b, "SUPPLIES")
            )
            assert [world.nodes[s].name for s in shared] == [q.answer]
        elif q.template == "max_employees":
            country = q.question.rsplit(" in ", 1)[1].rstrip("?")
            values = [
                n.properties["employees"]
                for n in world.of_type("VENDOR")
                if n.properties["country"] == country
            ]
            want = max(values) if "largest" in q.question else min(values)
            assert int(q.answer) == want
        elif q.template == "text_only_ceo":
            assert q.answer in world.ceo.values() and q.plan is None
            continue
        else:
            continue
        checked += 1
        GraphQueryPlan.model_validate(q.plan)
    assert checked > 50
    # Yes and no are balanced; numbers carry their word form.
    yes_no = [q.answer for q in found if q.template == "supplies_yes_no"]
    assert yes_no.count("yes") == yes_no.count("no")
    small = next(
        q for q in found if q.template == "count_suppliers" and int(q.answer) < 20
    )
    assert len(small.aliases) == 2


def test_the_outputs_ingest_as_a_seed_graph_and_a_dataset(tmp_path: Path) -> None:
    counts = structural.prepare(tmp_path, seed=0, size="small")
    config = get_config(str(tmp_path / "structural-config.yaml"))
    (seed,) = config.processing.seed_graphs
    assert seed.name == "structural" and Path(seed.path).is_file()
    (conversion,) = load_seed_graphs(config)
    assert (conversion.stats.nodes, conversion.stats.edges) == (
        counts["nodes"],
        counts["edges"],
    )
    rows = _dataset(tmp_path)
    assert {r["metadata"]["benchmark"] for r in rows} == {"structural"}
    assert all(r["category"] == r["metadata"]["template"] for r in rows)
    controls = [r for r in rows if r["category"] == "text_only_ceo"]
    assert controls and all(r["reference_sources"] for r in controls)
    assert all(
        (tmp_path / "corpus" / r["reference_sources"][0]).is_file() for r in controls
    )
    assert all("gold_plan" in r["metadata"] for r in rows if r not in controls)
    # Each control's answer is in its document, and nowhere in the seed.
    seed_text = (tmp_path / "seed" / "structural.jsonl").read_text()
    for row in controls:
        assert (
            row["answer"]
            in (tmp_path / "corpus" / row["reference_sources"][0]).read_text()
        )
        assert row["answer"] not in seed_text


def test_a_used_directory_is_refused(tmp_path: Path) -> None:
    structural.prepare(tmp_path, seed=0)
    with pytest.raises(EvaluationException, match="not empty"):
        structural.prepare(tmp_path, seed=0)


def test_the_cli_writes_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "bench"
    argv = [
        "structural",
        "--output-directory",
        str(out),
        "--seed",
        "5",
        "--output-format",
        "json",
    ]
    prepare_benchmark.main(argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["benchmark"] == "structural" and printed["seed"] == 5
    assert {p.name for p in out.iterdir()} == {
        "corpus",
        "seed",
        "eval_data.json",
        "structural-config.yaml",
    }
    with pytest.raises(SystemExit) as exit_info:
        prepare_benchmark.main(argv)
    assert exit_info.value.code == 2


def _result(structured: dict | None) -> EvaluationResult:
    return EvaluationResult(
        query_id="q",
        question="?",
        generated_answer="a",
        ground_truth="",
        structured=structured,
    )


def test_run_eval_reports_what_the_structured_arm_did() -> None:
    results = [
        _result(
            {"status": "answered", "repairs": 1, "exact": True, "truncated": False}
        ),
        _result(
            {"status": "answered", "repairs": 0, "exact": False, "truncated": True}
        ),
        _result({"status": "fallback", "reason": "invalid_plan", "repairs": 1}),
        _result({"status": "fallback", "reason": "empty_result"}),
        _result(None),
    ]
    stats = structured_statistics(results)
    assert stats == {
        "queries": 4,
        "status": {"answered": 2, "fallback": 2},
        "fallback_reasons": {"empty_result": 1, "invalid_plan": 1},
        "valid_plan_rate": 0.75,
        "repair_rate": 0.5,
        "truncated_rate": 0.5,
        "exact_rate": 0.5,
    }
    assert structured_statistics([_result(None)]) == {}


def test_absent_entities_and_other_languages_are_asked() -> None:
    world = structural.generate_world(seed=2, size="medium")
    found = structural.questions(world, seed=2, size="medium")
    names = {node.name for node in world.nodes.values()}
    absent = [q for q in found if q.template == "absent_entity"]
    assert absent and all(q.answer in ("no", "0") for q in absent)
    for q in absent:
        vendor = q.plan["nodes"][0]["mention"]
        assert vendor not in names and vendor in q.question
    korean = [q for q in found if q.template == "multilingual_ko"]
    japanese = [q for q in found if q.template == "multilingual_ja"]
    assert korean and japanese
    assert {q.answer for q in korean if not q.answer.isdigit()} <= {"예", "아니요"}
    assert {q.answer for q in japanese if not q.answer.isdigit()} <= {"はい", "いいえ"}


def _score(item: dict, answer: str, *, abstained: bool = False) -> float:
    """The answer_contains an item of the benchmark's dataset scores."""
    from unified_kg_rag.domain.models import Config, EvaluationQuery
    from unified_kg_rag.evaluation import AnswerMatchEvaluator

    evaluator = AnswerMatchEvaluator(Config())
    metadata = {**item["metadata"], "answer_aliases": item["answer_aliases"]}
    query = EvaluationQuery(
        query_id=item["query_id"], question=item["question"], metadata=metadata
    )
    result = EvaluationResult(
        query_id=item["query_id"],
        question=item["question"],
        generated_answer=answer,
        ground_truth=item["answer"],
        metadata=metadata,
        abstained=abstained,
    )
    report = evaluator.evaluate_single(query, result, item["answer"])
    return {m.metric_type.value: m.value for m in report.metrics}["answer_contains"]


def _item(tmp_path: Path, template: str, counts: range | None = None) -> dict:
    """An item of ``template`` (with an answer in ``counts``) of a world."""
    structural.prepare(tmp_path, seed=3, size="medium")
    return next(
        item
        for item in _dataset(tmp_path)
        if item["metadata"]["template"] == template
        and (
            counts is None
            or (item["answer"].isdigit() and int(item["answer"]) in counts)
        )
    )


_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
_KO = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_JA = "〇一二三四五六七八九"


class TestStructuralScoring:
    """Gold answers of the structural templates are scored by structure: a
    count by the number the answer states first (in Korean and Japanese
    too), the top three as a set, and a decline as the right answer to a
    question about an absent vendor."""

    def test_a_count_is_the_leading_number(self, tmp_path: Path) -> None:
        item = _item(tmp_path, "count_suppliers", range(2, 9))
        n = int(item["answer"])
        assert item["metadata"]["answer_type"] == "number"
        assert _score(item, f"{_WORDS[n].title()} vendors supply it.") == 1.0
        assert _score(item, f"It is supplied by {n} vendors.") == 1.0
        # Any small number in the prose used to match the gold.
        assert _score(item, f"{_WORDS[n + 1]} vendors. {_WORDS[n]} are local.") == 0
        assert _score(item, f"It has {n + 1} vendors (one with {n} products).") == 0

    def test_korean_and_japanese_counts(self, tmp_path: Path) -> None:
        item = _item(tmp_path, "multilingual_ko", range(1, 9))
        n = int(item["answer"])
        assert f"{_KO[n]} 곳" in item["answer_aliases"]
        assert _score(item, f"공급하는 벤더는 {n}곳입니다.") == 1.0
        assert _score(item, f"{_KO[n]} 곳의 벤더가 공급합니다.") == 1.0
        assert (
            _score(item, f"벤더는 총 {n + 1}곳이며, 그중 {n}곳은 Westria에 있습니다.")
            == 0
        )
        ja = _item(tmp_path / "ja", "multilingual_ja", range(1, 9))
        m = int(ja["answer"])
        assert f"{_JA[m]}社" in ja["answer_aliases"]
        assert _score(ja, f"供給しているベンダーは{_JA[m]}社です。") == 1.0
        assert _score(ja, f"ベンダーは{m}社です。") == 1.0
        assert _score(ja, f"ベンダーは{m + 1}社です。") == 0

    def test_the_top_three_is_a_set(self, tmp_path: Path) -> None:
        item = _item(tmp_path, "top_three")
        names = item["metadata"]["answer_items"]
        assert item["metadata"]["answer_type"] == "set" and len(names) == 3
        listed = "\n".join(f"{i}. {n} ({5 - i} buyers)" for i, n in enumerate(names))
        assert _score(item, listed) == 1.0
        assert (
            _score(item, f"The top three are {names[2]}, {names[0]} and {names[1]}.")
            == 1.0
        )
        assert _score(item, f"{names[0]} and {names[1]} supply the most.") == 0

    def test_a_decline_answers_an_absent_vendor(self, tmp_path: Path) -> None:
        structural.prepare(tmp_path, seed=3, size="medium")
        absent = [
            i
            for i in _dataset(tmp_path)
            if i["metadata"]["template"] == "absent_entity"
        ]
        assert absent and all(i["metadata"]["decline_is_correct"] for i in absent)
        assert all(_score(i, "No answer found.", abstained=True) == 1.0 for i in absent)
        count = next(i for i in absent if i["answer"] == "0")
        assert _score(count, "It supplies no buyers: no such vendor exists.") == 1.0
        assert _score(count, "It supplies 2 buyers.") == 0
        # Elsewhere a decline is still a wrong answer.
        other = _item(tmp_path / "other", "count_suppliers")
        assert _score(other, "No answer found.", abstained=True) == 0
