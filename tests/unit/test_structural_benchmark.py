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
