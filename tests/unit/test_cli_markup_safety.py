# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data printed by the CLIs is never read as rich markup (AWS-free).

Regression: an entity name, a community title, a query or an error holding
``[/red]`` or ``[/x]`` went into a markup string, and rich raised
``MarkupError`` (a closing tag no open tag matches), crashing the CLI after
the run or while printing its error.
"""

from __future__ import annotations

import argparse
import asyncio

import pytest

from unified_kg_rag.application.cli import (
    prepare_benchmark,
    run_evaluation,
    run_ingestion_pipeline,
    run_rag_chain,
)
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    Config,
    Entity,
    Relationship,
)
from unified_kg_rag.domain.models.evaluation import EvaluationSummary
from unified_kg_rag.evaluation.benchmarks import musique
from unified_kg_rag.shared.utils import display

pytestmark = pytest.mark.unit

_NAME = "[/red]Vendor[bold]"
_QUERY = "[/x] what does Vendor ship?"


def test_sample_tables_and_communities_print_markup_in_data_literally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    entity = Entity(id="e1", name=_NAME, type="[/x]", description="[/red] desc")
    display.display_sample_entities([entity])
    display.display_sample_relationships(
        [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                source_name=_NAME,
                target_name="[/y]Buyer",
                description="[/red] ships",
            )
        ]
    )
    display.display_sample_claims(
        [
            Claim(
                id="c1",
                subject_id="e1",
                subject_name=_NAME,
                object_name="[/z]",
                type="[/red]",
                description="[/red] pays",
            )
        ]
    )
    display.display_communities(
        [
            Community(
                id="c1",
                name="[/red] Widget supply",
                level="0",
                parent="",
                children=[],
                entity_ids=["e1"],
                size=1,
            )
        ],
        None,
        [entity],
        None,
    )

    out = capsys.readouterr().out
    assert "[/red]Vendor[bold]" in out
    assert "[/y]Buyer" in out
    assert "[/red] Widget supply" in out


def test_run_rag_banner_prints_the_query_literally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = argparse.Namespace(
        mode="rag",
        search_strategy="local",
        search_type="hybrid",
        top_k=5,
        retrieval_multiplier=2,
        use_memory=False,
        interactive=False,
        disable_query_processing=False,
        verbose=False,
        suffix="[/x]",
        filters=["k:[/red]"],
        conversation_id="[/c]",
        query=_QUERY,
    )

    runner._display_run_info()

    out = capsys.readouterr().out
    assert _QUERY in out
    assert "k:[/red]" in out


def test_run_ingestion_prints_an_unexpected_error_literally(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"boom on {_NAME}")

    monkeypatch.setattr(run_ingestion_pipeline, "IngestionPipelineRunner", fail)
    monkeypatch.setattr("sys.argv", ["run-ingestion", "--source-directory", "."])

    with pytest.raises(SystemExit) as exit_info:
        run_ingestion_pipeline.main()

    assert exit_info.value.code == 1
    assert f"boom on {_NAME}" in capsys.readouterr().out


def test_run_eval_prints_an_unexpected_error_literally(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"boom on {_QUERY}")

    monkeypatch.setattr(run_evaluation, "load_cli_config", fail)
    monkeypatch.setattr("sys.argv", ["run-eval", "--data-path", "eval.json"])

    with pytest.raises(SystemExit) as exit_info:
        run_evaluation.main()

    assert exit_info.value.code == 1
    assert f"boom on {_QUERY}" in capsys.readouterr().out


def test_run_rag_prints_metrics_filters_and_init_errors_literally(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_rag_chain.RAGChainRunner._print_result(
        {
            "success": True,
            "answer": "ok",
            "search_results": {"note": "[/red] note"},
            "metadata": {},
        },
        verbose=True,
    )
    state: dict = {"filters": {}}
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.rag_chain = None
    runner._handle_set_filter(state, ["k:[/red]v"])
    # As --filters: a pair without a key or a value is refused.
    for pair in (":v", " :v", "k:"):
        runner._handle_set_filter(state, [pair])
    assert state["filters"] == {"k": "[/red]v"}

    async def fail(**kwargs: object) -> None:
        raise RuntimeError(f"boom on {_NAME}")

    monkeypatch.setattr(run_rag_chain, "create_rag_chain", fail)
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.config = Config()
    with pytest.raises(SystemExit):
        asyncio.run(runner._initialize_chain())

    out = capsys.readouterr().out
    assert "Note: [/red] note" in out
    assert "'k' = '[/red]v'" in out
    assert f"boom on {_NAME}" in out


def test_run_eval_prints_metric_names_and_warnings_literally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stats = dict.fromkeys(("mean", "median", "stdev", "min", "max"), 0.5)
    summary = EvaluationSummary.model_construct(
        total_queries=1,
        successful_evaluations=1,
        failed_evaluations=0,
        benchmarks=None,
        guardrail_blocked_counts={},
        average_response_time=None,
        metric_statistics={"[/red]metric": {**stats, "count": 1}},
        metric_outcomes={},
    )
    run_evaluation.EvaluationRunner._print_summary(summary, 1.0)
    run_evaluation.EvaluationRunner._print_comparison(
        {
            "baseline": "local",
            "strategies": {"local": {"metrics": {}, "routed_strategies": {}}},
            "deltas_vs_baseline": {},
            "warnings": ["[/red] fewer paired queries"],
        },
        1.0,
    )
    out = capsys.readouterr().out
    assert "[/red]metric" in out
    assert "Warning: [/red] fewer paired queries" in out


def test_run_ingestion_prints_integrity_errors_literally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = object.__new__(run_ingestion_pipeline.IngestionPipelineRunner)
    runner.args = argparse.Namespace(
        pipeline_id="p1", force_rebuild=False, source_directory="."
    )

    class _Pipeline:
        class state_manager:
            @staticmethod
            def pipeline_exists(pipeline_id: str) -> bool:
                return True

        @staticmethod
        def validate_resume_integrity(*args: object) -> tuple[bool, list[str]]:
            return False, ["[/red] stage inputs changed"]

    runner.pipeline = _Pipeline()  # type: ignore[assignment]
    runner._confirm = lambda prompt: False  # type: ignore[method-assign]
    assert runner._validate_pipeline_integrity() is False
    assert "[/red] stage inputs changed" in capsys.readouterr().out


def test_prepare_benchmark_prints_the_output_directory_literally(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "dev.jsonl"
    data.write_text("")
    target = tmp_path / "[/red]out"
    monkeypatch.setattr(musique, "check_output_directory", lambda path: None)
    monkeypatch.setattr(
        musique,
        "prepare",
        lambda *args, **kwargs: {"benchmark": "m", "questions": 1, "paragraphs": 2},
    )
    prepare_benchmark.main(
        ["musique", "--data-path", str(data), "--output-directory", str(target)]
    )
    assert "[/red]out" in capsys.readouterr().out
