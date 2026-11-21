# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Held-out evaluation: the tune/test halves of a prepared benchmark.

Defaults chosen and reported on one sample are biased upward by the
selection; ``prepare-benchmark`` labels every row with a deterministic half
(``metadata.split``), ``run-eval --split`` answers one half, and the run
manifest records the half and how the dataset was prepared
(``prepared.json``). All synthetic data, no AWS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.domain.models import Config, EvaluationQuery, EvaluatorType
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.evaluation.benchmarks import (
    PREPARED_FILE,
    SPLITS,
    content_sha256,
    musique,
    split_of,
    structural,
)
from unified_kg_rag.evaluation.saved_results import dataset_warnings
from unified_kg_rag.shared import EvaluationException


def _line(qid: str, *, answerable: bool = True) -> str:
    titles = [f"Vendor {qid}", f"Buyer {qid}"]
    return json.dumps(
        {
            "id": qid,
            "question": f"Who supplies the buyer in {qid}?",
            "answer": "Vendor",
            "answer_aliases": [],
            "answerable": answerable,
            "paragraphs": [
                {
                    "idx": i,
                    "title": title,
                    "paragraph_text": f"{title} paragraph {i}{'' if answerable else ' x'}.",
                    "is_supporting": i == 0,
                }
                for i, title in enumerate(titles)
            ],
        }
    )


def _source(tmp_path: Path, ids: list[str], *, full: bool = False) -> Path:
    path = tmp_path / "in" / "musique.jsonl"
    path.parent.mkdir(exist_ok=True)
    lines = [_line(q) for q in ids]
    if full:
        lines += [_line(q, answerable=False) for q in ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_IDS = [f"2hop__{i}_{i + 1}" for i in range(40)]


def test_the_split_is_the_parity_of_the_ids_sha256() -> None:
    for qid in _IDS:
        number = int(hashlib.sha256(qid.encode("utf-8")).hexdigest(), 16)
        assert split_of(qid) == ("tune" if number % 2 == 0 else "test")
    halves = {split_of(q) for q in _IDS}
    assert halves == set(SPLITS)


def test_every_row_names_its_half_and_prepared_json_records_the_draw(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, _IDS)
    counts = musique.prepare(source, tmp_path / "all", limit=10, seed=3)
    rows = json.loads((tmp_path / "all" / "eval_data.json").read_text())
    assert all(r["metadata"]["split"] == split_of(r["query_id"]) for r in rows)
    assert sum(counts["splits"].values()) == 10
    prepared = json.loads((tmp_path / "all" / PREPARED_FILE).read_text())
    assert prepared["seed"] == 3 and prepared["limit"] == 10
    assert prepared["split"] is None and prepared["unanswerable"] is None
    assert prepared["sources"] == [
        {
            "file": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]


@pytest.mark.parametrize("half", SPLITS)
def test_a_split_draws_only_from_its_half(tmp_path: Path, half: str) -> None:
    source = _source(tmp_path, _IDS)
    musique.prepare(source, tmp_path / half, limit=5, split=half)
    rows = json.loads((tmp_path / half / "eval_data.json").read_text())
    assert len(rows) == 5
    assert {split_of(r["query_id"]) for r in rows} == {half}
    prepared = json.loads((tmp_path / half / PREPARED_FILE).read_text())
    assert prepared["split"] == half


def test_an_unanswerable_question_shares_its_twins_half(tmp_path: Path) -> None:
    source = _source(tmp_path, _IDS, full=True)
    musique.prepare(source, tmp_path / "out", limit=4, unanswerable=4, split="test")
    rows = json.loads((tmp_path / "out" / "eval_data.json").read_text())
    declined = [r for r in rows if r["metadata"].get("answerable") is False]
    assert declined and all(r["metadata"]["split"] == "test" for r in declined)
    assert all(
        split_of(r["query_id"].removesuffix(musique.UNANSWERABLE_ID_SUFFIX)) == "test"
        for r in declined
    )


def test_the_structural_rows_are_labelled(tmp_path: Path) -> None:
    structural.prepare(tmp_path / "s", seed=1)
    rows = json.loads((tmp_path / "s" / "eval_data.json").read_text())
    assert all(r["metadata"]["split"] == split_of(r["query_id"]) for r in rows)
    assert (tmp_path / "s" / PREPARED_FILE).is_file()


def _dataset(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "eval_data.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_load_data_keeps_one_half(tmp_path: Path) -> None:
    rows = [
        {
            "query_id": q,
            "question": f"Who ships {q}?",
            "answer": "Vendor",
            "metadata": {"split": split_of(q)},
        }
        for q in _IDS[:10]
    ]
    path = _dataset(tmp_path, rows)
    queries, truths = EvaluationManager.load_data(path, split="test")
    expected = [q for q in _IDS[:10] if split_of(q) == "test"]
    assert [q.query_id for q in queries] == expected
    assert [t.query_id for t in truths] == expected
    every, _ = EvaluationManager.load_data(path)
    assert len(every) == 10


def test_load_data_refuses_an_unlabelled_dataset(tmp_path: Path) -> None:
    path = _dataset(tmp_path, [{"query_id": "q1", "question": "Who?", "answer": "A"}])
    with pytest.raises(EvaluationException, match="without metadata.split"):
        EvaluationManager.load_data(path, split="tune")
    labelled = _dataset(
        tmp_path,
        [{"query_id": "q1", "question": "Who?", "metadata": {"split": "tune"}}],
    )
    with pytest.raises(EvaluationException, match="no item with metadata.split 'test'"):
        EvaluationManager.load_data(labelled, split="test")


def test_the_manifest_records_the_half_the_draw_and_the_settings(
    tmp_path: Path,
) -> None:
    out = tmp_path / "bench"
    musique.prepare(_source(tmp_path, _IDS), out, limit=6, seed=2)
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    manager = EvaluationManager(config, rag_chain=object())
    data = out / "eval_data.json"
    queries, truths = EvaluationManager.load_data(data, split="tune")
    manifest = manager.build_run_manifest(
        data, {"split": "tune"}, queries=queries, ground_truths=truths
    )
    dataset = manifest["dataset"]
    assert dataset["split"] == "tune"
    assert dataset["splits"] == {"tune": len(queries)}
    assert dataset["prepared"]["seed"] == 2 and dataset["prepared"]["limit"] == 6
    settings = manifest["answer_settings"]
    assert manifest["retrieval_config_sha256"] == content_sha256(settings)
    assert settings["evaluation.retrieval_k"] == 5


def test_prepared_json_is_recorded_only_for_the_dataset_it_describes(
    tmp_path: Path,
) -> None:
    # RV4-4: any dataset beside a prepared.json got its provenance (seed,
    # sizes, split) in its manifest, a 2-item file beside a 42-question one.
    out = tmp_path / "bench"
    structural.prepare(out, seed=0)
    manager = EvaluationManager(Config(), rag_chain=object())

    def prepared(path: Path) -> Any:
        return manager.build_run_manifest(path, {})["dataset"].get("prepared")

    data = out / "eval_data.json"
    assert prepared(data)["benchmark"] == structural.BENCHMARK_NAME
    rows = json.loads(data.read_text())
    # Reformatted, it is the same dataset.
    data.write_text(json.dumps(rows), encoding="utf-8")
    assert prepared(data) is not None
    mine = out / "mine.json"
    mine.write_text(json.dumps(rows[:2]), encoding="utf-8")
    assert prepared(mine) is None
    # Edited since: no longer the one prepared.
    data.write_text(json.dumps(rows[1:]), encoding="utf-8")
    assert prepared(data) is None


def test_runs_of_different_halves_are_different_datasets() -> None:
    def manifest(split: str | None) -> dict[str, Any]:
        dataset: dict[str, Any] = {"content_sha256": "a" * 64}
        if split:
            dataset["split"] = split
        return {"dataset": dataset}

    assert dataset_warnings([("local", manifest(None)), ("mix", manifest(None))]) == []
    warned = dataset_warnings([("local", manifest("tune")), ("mix", manifest("test"))])
    assert warned and "split tune" in warned[0] and "split test" in warned[0]


def test_query_metadata_split_is_not_a_chain_input() -> None:
    from unified_kg_rag.evaluation.evaluation_manager import _rag_input_fields

    query = EvaluationQuery(query_id="q", question="?", metadata={"split": "tune"})
    assert "split" not in _rag_input_fields(query.metadata)


class _Chain:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def ainvoke(self, inputs: dict[str, Any], config: Any = None) -> Any:
        self.asked.append(inputs["query"])
        meta = {"search_strategy": "local", "processing_time": 0.1}
        return {"answer": "Vendor", "sources": [], "metadata": meta}


async def test_run_eval_answers_one_half(tmp_path: Path, mocker) -> None:
    from unified_kg_rag.application.cli import run_evaluation

    mocker.patch.object(run_evaluation, "display_ascii_art")
    rows = [
        {
            "query_id": q,
            "question": f"Who ships {q}?",
            "answer": "Vendor",
            "metadata": {"split": split_of(q)},
        }
        for q in _IDS[:12]
    ]
    data = _dataset(tmp_path, rows)
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    out = tmp_path / "out"
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--data-path", str(data), "--split", "test", "--output-directory", str(out)]
    )
    chain = _Chain()
    assert await run_evaluation.EvaluationRunner(args, chain, config).run() == 0
    # The warm-up query aside, only the test half is answered.
    asked = [q for q in chain.asked if q.startswith("Who ships")]
    assert asked == [r["question"] for r in rows if split_of(r["query_id"]) == "test"]
    (summary,) = out.rglob("evaluation_summary_*.json")
    manifest = json.loads(summary.read_text())["run_manifest"]
    assert manifest["dataset"]["split"] == "test"


def test_run_eval_split_needs_a_dataset_to_answer(tmp_path: Path, mocker) -> None:
    from unified_kg_rag.application.cli import run_evaluation

    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--from-results", str(tmp_path), "--split", "tune"]
    )
    with pytest.raises(SystemExit) as exit_info:
        run_evaluation.EvaluationRunner(args, None, Config())._validate_args()
    assert exit_info.value.code == 2
