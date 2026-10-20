# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CLI argument parsing + input validation (AWS-free).

Covers only the parser construction, flag defaults/choices, and the local
validation paths (``_validate_args``, data-path existence, output-format,
stage-name validation) of the five ``run-*`` entry points. The AWS-driven
``run()`` bodies (chain/pipeline/Bedrock) are deliberately NOT exercised here;
those are integration-level. Where a ``main()`` path is asserted, the chain /
pipeline / tuner construction is patched so nothing touches AWS.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
    run_visualization,
)
from unified_kg_rag.domain.models import (
    EvaluationSummary,
    EvaluatorType,
    SearchStrategy,
    SearchType,
)

pytestmark = pytest.mark.unit


# --- run_rag_chain: parser ----------------------------------------------


def _rag_parser() -> argparse.ArgumentParser:
    return run_rag_chain.CommandLineInterface._setup_arguments()


def test_rag_parser_defaults() -> None:
    args = _rag_parser().parse_args(["--query", "hello"])
    assert args.query == "hello"
    assert args.interactive is False
    assert args.mode == "rag"
    assert args.search_strategy == "auto"
    assert args.search_type == "hybrid"
    assert args.top_k == 10
    assert args.retrieval_multiplier == 1
    assert args.output_format == "text"
    assert args.use_memory is False
    assert args.disable_query_processing is False


def test_rag_parser_short_flags_and_overrides() -> None:
    args = _rag_parser().parse_args(
        ["-q", "q", "-v", "--top-k", "5", "--output-format", "json"]
    )
    assert args.query == "q"
    assert args.verbose is True
    assert args.top_k == 5
    assert args.output_format == "json"


@pytest.mark.parametrize("strategy", [s.value for s in SearchStrategy])
def test_rag_parser_accepts_every_search_strategy(strategy: str) -> None:
    args = _rag_parser().parse_args(["-q", "q", "--search-strategy", strategy])
    assert args.search_strategy == strategy


@pytest.mark.parametrize("stype", [s.value for s in SearchType])
def test_rag_parser_accepts_every_search_type(stype: str) -> None:
    args = _rag_parser().parse_args(["-q", "q", "--search-type", stype])
    assert args.search_type == stype


def test_rag_parser_rejects_unknown_search_strategy() -> None:
    with pytest.raises(SystemExit):
        _rag_parser().parse_args(["-q", "q", "--search-strategy", "bogus"])


def test_rag_parser_rejects_unknown_output_format() -> None:
    with pytest.raises(SystemExit):
        _rag_parser().parse_args(["-q", "q", "--output-format", "xml"])


def test_rag_parser_filters_nargs() -> None:
    args = _rag_parser().parse_args(["-q", "q", "--filters", "a:1", "b:2"])
    assert args.filters == ["a:1", "b:2"]


# --- run_rag_chain: RAGChainRunner validation ----------------------------


@pytest.mark.parametrize(
    ("argv", "code"),
    [(["--help"], 0), (["--top-k", "x"], 2), ([], 2)],
    ids=["help", "bad-arg", "no-query-or-interactive"],
)
def test_rag_main_parses_args_before_the_event_loop(
    mocker, monkeypatch: pytest.MonkeyPatch, argv: list[str], code: int
) -> None:
    # argparse's SystemExit raised inside asyncio.run (under nest_asyncio)
    # printed "Task exception was never retrieved ... SystemExit" tracebacks.
    monkeypatch.setattr(sys, "argv", ["run-rag", *argv])
    loop_run = mocker.patch.object(run_rag_chain.asyncio, "run")
    with pytest.raises(SystemExit) as exc:
        run_rag_chain.main()
    assert exc.value.code == code
    loop_run.assert_not_called()


async def test_rag_async_main_turns_system_exit_into_a_return_code(
    mocker,
) -> None:
    # A SystemExit escaping the task is what asyncio reports as "never
    # retrieved"; the coroutine returns the code instead.
    runner = mocker.MagicMock()
    runner.run = mocker.AsyncMock(side_effect=SystemExit(1))
    mocker.patch.object(run_rag_chain, "RAGChainRunner", return_value=runner)
    args = _rag_parser().parse_args(["-q", "hi"])
    assert await run_rag_chain.async_main(args) == 1


def test_rag_runner_accepts_query(config, mocker) -> None:
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    args = _rag_parser().parse_args(["-q", "hi"])
    runner = run_rag_chain.RAGChainRunner(args)  # no SystemExit
    assert runner.args.query == "hi"


def test_rag_runner_json_mode_routes_decorations_and_logs_to_stderr(
    config, mocker
) -> None:
    # stdout must carry only the JSON document so it can be piped to jq.
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    setup = mocker.patch.object(run_rag_chain, "setup_logging")
    mocker.patch.object(run_rag_chain.console, "stderr", False)
    args = _rag_parser().parse_args(["-q", "hi", "--output-format", "json"])

    run_rag_chain.RAGChainRunner(args)

    assert run_rag_chain.console.stderr is True
    assert setup.call_args.kwargs["stream"] is sys.stderr


def test_rag_runner_accepts_interactive(config, mocker) -> None:
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    args = _rag_parser().parse_args(["--interactive"])
    runner = run_rag_chain.RAGChainRunner(args)
    assert runner.args.interactive is True


@pytest.fixture
def store_config(config):
    """Default config with store endpoints, so run() passes the preflight."""
    config.aws.neptune.endpoint = "neptune.local"
    config.aws.opensearch.endpoint = "opensearch.local"
    return config


async def test_rag_runner_leaves_the_target_language_to_the_chain(
    store_config, mocker
) -> None:
    # An explicit target_language disables the chain's same-language skip, so
    # every CLI query paid a translation call even for a single-language corpus.
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=store_config)
    mocker.patch.object(run_rag_chain, "display_ascii_art")
    chain = mocker.AsyncMock()
    mocker.patch.object(run_rag_chain, "create_rag_chain", return_value=chain)
    runner = run_rag_chain.RAGChainRunner(_rag_parser().parse_args(["-q", "hi"]))
    run_query = mocker.patch.object(
        runner, "_run_query", return_value={"success": True}
    )
    mocker.patch.object(runner, "_print_result")

    await runner.run()

    (rag_input,) = run_query.call_args.args
    assert rag_input.target_language is None


@pytest.mark.parametrize(
    "flags",
    [["--use-memory"], ["--conversation-id", "s1"]],
    ids=["use-memory", "conversation-id"],
)
async def test_rag_runner_warns_that_a_single_query_has_no_memory(
    store_config, mocker, flags: list[str]
) -> None:
    # Memory lives in the process: a single --query never sees an earlier
    # run's turns, whatever --conversation-id says.
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=store_config)
    mocker.patch.object(run_rag_chain, "display_ascii_art")
    mocker.patch.object(
        run_rag_chain, "create_rag_chain", return_value=mocker.AsyncMock()
    )
    warning = mocker.patch.object(run_rag_chain.logger, "warning")
    runner = run_rag_chain.RAGChainRunner(
        _rag_parser().parse_args(["-q", "hi", *flags])
    )
    mocker.patch.object(runner, "_run_query", return_value={"success": True})
    mocker.patch.object(runner, "_print_result")

    await runner.run()

    assert any(
        "no effect on a single --query" in str(call.args[0])
        for call in warning.call_args_list
    )


def test_rag_cli_rejects_search_mode_with_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Interactive mode always generates; --mode search was silently ignored.
    monkeypatch.setattr(sys, "argv", ["run-rag", "--interactive", "--mode", "search"])
    with pytest.raises(SystemExit) as exc:
        run_rag_chain.CommandLineInterface().parse_args()
    assert exc.value.code == 2


def test_rag_cli_rejects_json_output_with_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Before, the chat printed nothing on stdout, JSON included.
    monkeypatch.setattr(
        sys, "argv", ["run-rag", "--interactive", "--output-format", "json"]
    )
    with pytest.raises(SystemExit) as exc:
        run_rag_chain.CommandLineInterface().parse_args()
    assert exc.value.code == 2


async def test_rag_runner_error_fallback_is_not_success(config, mocker) -> None:
    # Under ignore_errors the chain returns DEFAULT_ERROR_MESSAGE with
    # metadata.error instead of raising; the CLI must not call that success.
    from unified_kg_rag.application.retrieval.rag_chain import (
        DEFAULT_ERROR_MESSAGE,
        ProcessedQuery,
        RAGOutput,
    )
    from unified_kg_rag.domain.models import SearchQuery, SearchResult

    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    runner = run_rag_chain.RAGChainRunner(_rag_parser().parse_args(["-q", "hi"]))
    runner.rag_chain = mocker.AsyncMock()
    runner.rag_chain.ainvoke.return_value = RAGOutput(
        answer=DEFAULT_ERROR_MESSAGE,
        sources=[],
        search_results=SearchResult(
            query=SearchQuery(query="hi"),
            results=[],
            total_results=0,
            search_strategy="error",
            processing_time=0.0,
            metadata={"error": "Neptune down"},
        ),
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="hi", final_query="hi"),
        metadata={"error": True},
    )
    result = await runner._run_query(run_rag_chain.RAGInput(query="hi"))
    assert result["success"] is False
    assert result["error"] == "Neptune down"


async def test_rag_runner_exits_non_zero_on_error_fallback(
    store_config, mocker
) -> None:
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=store_config)
    mocker.patch.object(run_rag_chain, "display_ascii_art")
    mocker.patch.object(
        run_rag_chain, "create_rag_chain", return_value=mocker.AsyncMock()
    )
    runner = run_rag_chain.RAGChainRunner(_rag_parser().parse_args(["-q", "hi"]))
    runner.rag_chain = None
    mocker.patch.object(
        runner,
        "_run_query",
        return_value={"success": False, "error": "x", "metadata": {"error": True}},
    )
    print_result = mocker.patch.object(runner, "_print_result")
    with pytest.raises(SystemExit) as exc:
        await runner.run()
    assert exc.value.code == 1
    print_result.assert_called_once()  # exited on the result, not earlier


# --- run_rag_chain: _parse_filters --------------------------------------


def test_rag_parse_filters_key_value() -> None:
    out = run_rag_chain.RAGChainRunner._parse_filters(["entity_type:person", "x:y"])
    assert out == {"entity_type": "person", "x": "y"}


def test_rag_parse_filters_skips_malformed_and_handles_none() -> None:
    assert run_rag_chain.RAGChainRunner._parse_filters(None) == {}
    # "noColon" has no ':' -> skipped with a warning, not raised.
    assert run_rag_chain.RAGChainRunner._parse_filters(["noColon", "k:v"]) == {"k": "v"}


def test_rag_parse_filters_value_with_colon_splits_once() -> None:
    out = run_rag_chain.RAGChainRunner._parse_filters(["url:http://x:8080"])
    assert out == {"url": "http://x:8080"}


# --- run_evaluation: parser ----------------------------------------------


def _eval_parser() -> argparse.ArgumentParser:
    return run_evaluation.CommandLineInterface._setup_arguments()


def test_eval_parser_requires_data_path() -> None:
    with pytest.raises(SystemExit):
        _eval_parser().parse_args([])


def test_eval_parser_defaults() -> None:
    args = _eval_parser().parse_args(["--data-path", "data.json"])
    assert args.data_path == Path("data.json")
    assert args.search_strategy == ["auto"]
    assert args.repeats == 1 and args.pairwise_judge is False
    assert args.search_type == "hybrid"
    assert args.top_k == 10
    assert args.output_directory is None


@pytest.mark.parametrize("flag", ["--top-k", "--retrieval-multiplier"])
@pytest.mark.parametrize("parser", [_rag_parser, _eval_parser])
def test_query_width_flags_must_be_positive(parser, flag: str) -> None:
    # Both CLIs share the query flags; a zero width fails at parse time
    # instead of deep in the chain.
    required = ["--query", "q"] if parser is _rag_parser else ["--data-path", "d"]
    with pytest.raises(SystemExit):
        parser().parse_args([*required, flag, "0"])
    assert (
        getattr(parser().parse_args([*required, flag, "3"]), flag[2:].replace("-", "_"))
        == 3
    )


def test_eval_parser_rejects_bad_strategy() -> None:
    with pytest.raises(SystemExit):
        _eval_parser().parse_args(
            ["--data-path", "d.json", "--search-strategy", "nope"]
        )


def test_eval_runner_missing_file_is_an_invalid_dataset(
    config, mocker, tmp_path
) -> None:
    missing = tmp_path / "does-not-exist.json"
    args = _eval_parser().parse_args(["--data-path", str(missing)])
    with pytest.raises(run_evaluation.InvalidDatasetError, match="not found"):
        run_evaluation.EvaluationRunner(args, object(), config)


def test_run_eval_missing_dataset_exits_2(config, mocker, tmp_path, capsys) -> None:
    # A missing dataset is the same input error as an invalid one.
    load = mocker.patch.object(run_evaluation, "load_cli_config", return_value=config)
    mocker.patch.object(run_evaluation, "missing_endpoints_error", return_value=None)
    mocker.patch.object(run_evaluation, "GraphRAGChain", return_value=AsyncMock())
    missing = tmp_path / "does-not-exist.json"
    mocker.patch.object(sys, "argv", ["run-eval", "--data-path", str(missing)])
    with pytest.raises(SystemExit) as exc:
        run_evaluation.main()
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "invalid evaluation dataset" in out and "unexpected" not in out
    # The runner reuses main's config instead of loading it again.
    load.assert_called_once()


def test_eval_runner_existing_file_ok(config, mocker, tmp_path) -> None:
    data = tmp_path / "eval.json"
    data.write_text("[]", encoding="utf-8")
    args = _eval_parser().parse_args(["--data-path", str(data)])
    runner = run_evaluation.EvaluationRunner(args, object(), config)  # no exit
    assert runner.args.data_path == data


def test_eval_parser_max_failure_rate_default_and_range() -> None:
    assert _eval_parser().parse_args(["--data-path", "d"]).max_failure_rate == 1.0
    args = _eval_parser().parse_args(["--data-path", "d", "--max-failure-rate", "0.2"])
    assert args.max_failure_rate == 0.2
    with pytest.raises(SystemExit):
        _eval_parser().parse_args(["--data-path", "d", "--max-failure-rate", "2"])


@pytest.mark.parametrize(
    ("total", "failed", "budget", "expected"),
    [
        (10, 10, 1.0, True),  # all failed always fails the run
        (10, 9, 1.0, False),
        (10, 3, 0.2, True),
        (10, 2, 0.2, False),
        (10, 0, 0.0, False),
    ],
)
def test_eval_exceeds_failure_budget(total, failed, budget, expected) -> None:
    summary = EvaluationSummary(
        total_queries=total,
        successful_evaluations=total - failed,
        failed_evaluations=failed,
        evaluation_start_time=datetime(2026, 1, 1),
        evaluation_end_time=datetime(2026, 1, 1),
    )
    assert bool(run_evaluation.failure_budget_breaches(summary, budget)) is expected


@pytest.mark.parametrize(
    ("outcomes", "budget", "expected"),
    [
        # A metric that failed on every attempted query fails the run even
        # though every answer was generated.
        ({"langchain": {"correctness": {"scored": 0, "failed": 3}}}, 1.0, True),
        ({"langchain": {"correctness": {"scored": 2, "failed": 1}}}, 1.0, False),
        ({"langchain": {"correctness": {"scored": 2, "failed": 1}}}, 0.2, True),
        ({"ragas": {"faithfulness": {"scored": 9, "failed": 1}}}, 0.2, False),
        # Skipped (not applicable) is not a failure.
        (
            {"graph_aware": {"entity_coverage": {"scored": 0, "skipped": 3}}},
            0.0,
            False,
        ),
    ],
)
def test_eval_metric_failures_count_against_budget(outcomes, budget, expected) -> None:
    summary = EvaluationSummary(
        total_queries=3,
        successful_evaluations=3,
        failed_evaluations=0,
        metric_outcomes=outcomes,
        evaluation_start_time=datetime(2026, 1, 1),
        evaluation_end_time=datetime(2026, 1, 1),
    )
    assert bool(run_evaluation.failure_budget_breaches(summary, budget)) is expected


def test_eval_summary_names_the_benchmark(capsys) -> None:
    # A musique-full-answerable run is not comparable to a MuSiQue-Ans one:
    # the printed summary says which was scored.
    summary = EvaluationSummary(
        total_queries=2,
        successful_evaluations=2,
        failed_evaluations=0,
        benchmarks={"musique-full-answerable": 2},
        evaluation_start_time=datetime(2026, 1, 1),
        evaluation_end_time=datetime(2026, 1, 1),
    )
    run_evaluation.EvaluationRunner._print_summary(summary, 1.0)
    assert "Benchmark: musique-full-answerable (2)" in capsys.readouterr().out


def test_eval_comparison_names_the_benchmark(capsys) -> None:
    # A multi-strategy run prints the comparison table, not each summary:
    # before, it did not say which benchmark the strategies were scored on.
    stats = {"mean": 0.5}
    comparison = {
        "baseline": "local",
        "strategies": {
            name: {"metrics": {"answer_match/f1": stats}, "routed_strategies": {}}
            for name in ("local", "mix")
        },
        "deltas_vs_baseline": {"mix": {}},
        "run_manifest": {"dataset": {"benchmarks": {"musique-full-answerable": 2}}},
    }
    run_evaluation.EvaluationRunner._print_comparison(comparison, 1.0)
    assert "Benchmark: musique-full-answerable (2)" in capsys.readouterr().out


class _DictChain:
    def __init__(self, error: bool) -> None:
        self.error = error

    async def ainvoke(self, inputs, config=None):
        meta = {"search_strategy": "local", "processing_time": 0.1}
        if self.error:
            meta["error"] = True
        return {"answer": "Vendor", "sources": [], "metadata": meta}

    async def abatch(self, inputs, config=None):
        return [await self.ainvoke(i) for i in inputs]


@pytest.mark.parametrize(("error", "code"), [(False, 0), (True, 1)])
async def test_eval_runner_exit_code_and_manifest(
    config, mocker, tmp_path, error, code
) -> None:
    import json

    config.evaluation.enabled_evaluators = []
    mocker.patch.object(run_evaluation, "display_ascii_art")
    data = tmp_path / "eval.json"
    data.write_text(json.dumps([{"question": "Who ships?", "answer": "Vendor"}]))
    out = tmp_path / "out"
    args = _eval_parser().parse_args(
        ["--data-path", str(data), "--output-directory", str(out)]
    )
    runner = run_evaluation.EvaluationRunner(args, _DictChain(error), config)
    assert await runner.run() == code
    summary_file = next(out.glob("evaluation_summary_*.json"))
    manifest = json.loads(summary_file.read_text())["run_manifest"]
    assert manifest["cli_args"]["data_path"] == str(data)
    assert len(manifest["dataset"]["sha256"]) == 64


def test_eval_parser_takes_a_strategy_list() -> None:
    args = _eval_parser().parse_args(
        ["--data-path", "d", "--search-strategy", "local, mix,global"]
    )
    assert args.search_strategy == ["local", "mix", "global"]
    for bad in ("local,nope", "local,local", ","):
        with pytest.raises(SystemExit):
            _eval_parser().parse_args(["--data-path", "d", "--search-strategy", bad])
    with pytest.raises(SystemExit):
        _eval_parser().parse_args(["--data-path", "d", "--repeats", "0"])


class _StrategyChain:
    """Answers by strategy: local is right, mix wrong; auto routes to local."""

    def __init__(self) -> None:
        self.strategies: list[str] = []

    async def ainvoke(self, inputs, config=None):
        requested = inputs.get("search_strategy", "auto")
        self.strategies.append(requested)
        used = "local" if requested == "auto" else requested
        answer = "Vendor" if used == "local" else "Buyer"
        meta = {"search_strategy": used, "processing_time": 0.1}
        return {"answer": answer, "sources": [], "metadata": meta}


async def test_eval_runner_compares_strategies_pairwise(
    config, mocker, tmp_path
) -> None:
    import json

    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    mocker.patch.object(run_evaluation, "display_ascii_art")
    data = tmp_path / "eval.json"
    data.write_text(
        json.dumps(
            [
                # A per-item strategy does not override the compared ones.
                {
                    "question": f"Who ships {i}?",
                    "answer": "Vendor",
                    "metadata": {"search_strategy": "global"},
                }
                for i in range(4)
            ]
        )
    )
    out = tmp_path / "out"
    args = _eval_parser().parse_args(
        [
            "--data-path",
            str(data),
            "--output-directory",
            str(out),
            "--search-strategy",
            "local,mix,auto",
            "--repeats",
            "2",
        ]
    )
    chain = _StrategyChain()
    runner = run_evaluation.EvaluationRunner(args, chain, config)
    assert await runner.run() == 0
    assert chain.strategies == ["local"] * 8 + ["mix"] * 8 + ["auto"] * 8
    assert len(list(out.glob("evaluation_summary_*.json"))) == 6
    comparison = json.loads(next(out.glob("evaluation_comparison_*.json")).read_text())
    assert comparison["baseline"] == "local"
    em = "answer_match/exact_match"
    mix = comparison["deltas_vs_baseline"]["mix"][em]
    assert (mix["pairs"], mix["mean_delta"], mix["ci_low"], mix["ci_high"]) == (
        4,
        -1.0,
        -1.0,
        -1.0,
    )
    assert (mix["wins"], mix["ties"], mix["losses"]) == (0, 0, 4)
    assert comparison["deltas_vs_baseline"]["auto"][em]["mean_delta"] == 0.0
    auto = comparison["strategies"]["auto"]
    assert auto["routed_strategies"] == {"local": 8}
    assert auto["repeats"] == 2 and auto["metrics"][em]["repeat_stdev"] == 0.0
    assert comparison["run_manifest"]["cli_args"]["search_strategy"] == [
        "local",
        "mix",
        "auto",
    ]


async def test_eval_runner_runs_the_pairwise_judge(config, mocker, tmp_path) -> None:
    import json

    config.evaluation.enabled_evaluators = []
    mocker.patch.object(run_evaluation, "display_ascii_art")
    judge = mocker.patch.object(run_evaluation, "PairwiseJudge")
    judged = {
        "judged": 1,
        "failed": 0,
        "candidate_errors": 0,
        "baseline_errors": 0,
        "criteria": {},
    }
    judge.return_value.acompare = mocker.AsyncMock(return_value=judged)
    data = tmp_path / "eval.json"
    data.write_text(json.dumps([{"question": "Who ships?", "answer": "Vendor"}]))
    out = tmp_path / "out"
    args = _eval_parser().parse_args(
        [
            "--data-path",
            str(data),
            "--output-directory",
            str(out),
            "--search-strategy",
            "local,mix",
            "--pairwise-judge",
        ]
    )
    assert (
        await run_evaluation.EvaluationRunner(args, _StrategyChain(), config).run() == 0
    )
    baseline, candidate = judge.return_value.acompare.await_args.args
    assert [r.generated_answer for r in baseline] == ["Vendor"]
    assert [r.generated_answer for r in candidate] == ["Buyer"]
    comparison = json.loads(next(out.glob("evaluation_comparison_*.json")).read_text())
    assert comparison["pairwise_vs_baseline"] == {"mix": judged}


# --- run_ingestion_pipeline: parser --------------------------------------


def _ing_parser() -> argparse.ArgumentParser:
    return run_ingestion_pipeline.CommandLineInterface._setup_arguments()


def test_ingestion_parser_defaults() -> None:
    args = _ing_parser().parse_args([])
    assert args.s3_prefix == "pipeline-runs"
    assert args.metrics_sink == "none"
    assert args.force_rebuild is False
    assert args.s3_sync is False


def test_ingestion_parser_enabled_stages_splits_csv() -> None:
    args = _ing_parser().parse_args(
        ["--enabled-stages", "DOCUMENT_PARSING, TEXT_CHUNKING"]
    )
    assert args.enabled_stages == ["DOCUMENT_PARSING", "TEXT_CHUNKING"]


def test_ingestion_parser_rejects_bad_metrics_sink() -> None:
    with pytest.raises(SystemExit):
        _ing_parser().parse_args(["--metrics-sink", "datadog"])


def test_ingestion_runner_requires_source_directory(config, mocker) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    args = _ing_parser().parse_args([])  # no source dir, no metadata op
    args.source_directory = None  # ensure env var fallback didn't set it
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 2  # a usage error


def test_ingestion_runner_missing_source_dir_exits(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    missing = tmp_path / "nope"
    args = _ing_parser().parse_args(["--source-directory", str(missing)])
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 2  # a usage error


@pytest.mark.parametrize("from_config", [False, True])
def test_ingestion_runner_rejects_the_source_as_target(
    config, mocker, tmp_path, capsys, from_config: bool
) -> None:
    # A clear usage error before the pipeline is built, not a traceback from
    # the parsing stage's constructor.
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    argv = ["--source-directory", str(src)]
    if from_config:
        config.processing.document_parsing.target_directory = str(src / ".")
    else:
        argv += ["--output-directory", str(src)]
    args = _ing_parser().parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 2  # a usage error
    raw = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().err)
    out = " ".join(raw.split())  # undo rich styling and wrapping
    assert "must not be the source directory" in out
    assert "Traceback" not in out


def test_ingestion_runner_s3_sync_needs_bucket(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(["--source-directory", str(src), "--s3-sync"])
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 2  # a usage error


def test_ingestion_s3_sync_takes_the_configured_bucket(
    config, mocker, tmp_path
) -> None:
    # The CDK task injects aws.s3.bucket_name (UKR__AWS__S3__BUCKET_NAME) and
    # had to pass the same bucket again as --s3-bucket-name.
    config.aws.s3.bucket_name = "cache-bucket"
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(["--source-directory", str(src), "--s3-sync"])
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert runner._create_pipeline_config().s3_bucket_name == "cache-bucket"
    args.s3_bucket_name = "flag-bucket"  # the flag wins
    assert runner._create_pipeline_config().s3_bucket_name == "flag-bucket"


def test_ingestion_runner_resume_needs_pipeline_id(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--resume-from-stage", "TEXT_CHUNKING"]
    )
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 2  # a usage error


def test_ingestion_runner_metadata_op_needs_pipeline_id(config, mocker) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    args = _ing_parser().parse_args(["--verify-metadata"])
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 2  # a usage error


def test_ingestion_runner_valid_source_sets_cache_dir(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    cache = tmp_path / "mycache"
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--cache-directory", str(cache)]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert runner.args.source_directory == src
    assert runner.args.cache_directory == cache
    assert cache.is_dir()  # created by _validate_args


def test_ingestion_retire_source_scope_is_repeatable_and_merged_into_config(
    config, mocker, tmp_path
) -> None:
    config.indexing.retire_source_scopes = ["s3://bucket/old/"]
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    assert _ing_parser().parse_args([]).retire_source_scope == []
    args = _ing_parser().parse_args(
        [
            "--source-directory",
            str(src),
            "--retire-source-scope",
            "/corpora/old",
            "--retire-source-scope",
            "s3://bucket/old/",
        ]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert runner.config.indexing.retire_source_scopes == [
        "s3://bucket/old/",
        "/corpora/old",
    ]


def test_ingestion_retire_source_scope_is_kept_as_given(
    config, mocker, tmp_path, monkeypatch
) -> None:
    # The run tries each value's readings (incremental.retired_scope_readings),
    # so the CLI must not expand them: a bare value expanded into both
    # readings retired both.
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    real = tmp_path / "real"
    (real / "old").mkdir(parents=True)
    (tmp_path / "link").symlink_to(real)
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.chdir(tmp_path)
    args = _ing_parser().parse_args(
        [
            "--source-directory",
            str(src),
            "--retire-source-scope",
            str(tmp_path / "link" / "old"),
            "--retire-source-scope",
            "./real/old/../old",
            "--retire-source-scope",
            "contracts",
        ]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    # Kept as given: the run resolves the readings of each value (see
    # incremental.retired_scope_readings).
    assert runner.config.indexing.retire_source_scopes == [
        str(tmp_path / "link" / "old"),
        "./real/old/../old",
        "contracts",
    ]


def test_ingestion_create_pipeline_config_rejects_invalid_stage(
    config, mocker, tmp_path
) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--enabled-stages", "NOT_A_REAL_STAGE"]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    with pytest.raises(SystemExit) as exc:
        runner._create_pipeline_config()
    assert exc.value.code == 2  # a usage error


def test_ingestion_create_pipeline_config_enables_subset(
    config, mocker, tmp_path
) -> None:
    from unified_kg_rag.domain.models import PipelineStageType

    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    # Use a real stage name from the enum.
    stage = next(iter(PipelineStageType)).name
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--enabled-stages", stage]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    pc = runner._create_pipeline_config()
    enabled = {s for s, on in pc.stages_enabled.items() if on}
    assert PipelineStageType[stage] in enabled
    # Only the requested stage is on.
    assert len(enabled) == 1


def test_ingestion_create_pipeline_config_defaults_all_stages(
    config, mocker, tmp_path
) -> None:
    from unified_kg_rag.domain.models import PipelineStageType

    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(["--source-directory", str(src)])
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    pc = runner._create_pipeline_config()
    assert all(pc.stages_enabled.values())
    assert set(pc.stages_enabled.keys()) == set(PipelineStageType)


def test_ingestion_build_metrics_sink(config, mocker, tmp_path) -> None:
    from unified_kg_rag.shared import CloudWatchEMFSink, NullMetricsSink

    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()

    args_none = _ing_parser().parse_args(["--source-directory", str(src)])
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args_none)
    assert isinstance(runner._build_metrics_sink(), NullMetricsSink)

    args_cw = _ing_parser().parse_args(
        ["--source-directory", str(src), "--metrics-sink", "cloudwatch"]
    )
    runner_cw = run_ingestion_pipeline.IngestionPipelineRunner(args_cw)
    assert isinstance(runner_cw._build_metrics_sink(), CloudWatchEMFSink)


# --- run_prompt_tuning: parser + load_corpus_texts -----------------------


def test_prompt_tuning_parser_requires_source_dir() -> None:
    with pytest.raises(SystemExit):
        run_prompt_tuning._build_parser().parse_args([])


def test_prompt_tuning_parser_defaults() -> None:
    # Canonical flag is --source-directory (matches run-ingestion).
    args = run_prompt_tuning._build_parser().parse_args(["--source-directory", "docs"])
    assert args.source_directory == Path("docs")
    assert args.output_path == Path("tuned_prompts.yaml")
    assert args.max_docs == 20
    assert args.config_path is None


def test_prompt_tuning_parser_has_no_legacy_aliases() -> None:
    # One naming scheme across the CLIs: --source-dir and --output are gone.
    parser = run_prompt_tuning._build_parser()
    for legacy in (["--source-dir", "docs"], ["--output", "x.yaml"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--source-directory", "docs", *legacy])
        assert exc.value.code == 2


def test_load_corpus_texts_filters_and_limits(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "c.markdown").write_text("gamma", encoding="utf-8")
    (tmp_path / "broken.pdf").write_text("not a pdf", encoding="utf-8")
    (tmp_path / "skip.bin").write_text("ignored", encoding="utf-8")
    texts = run_prompt_tuning.load_corpus_texts(tmp_path, max_docs=10)
    assert set(texts) == {"alpha", "beta", "gamma"}


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def test_load_corpus_texts_parses_pdf_with_ingestion_loaders(tmp_path) -> None:
    (tmp_path / "terms.pdf").write_bytes(_minimal_pdf("Vendor ships to Buyer"))
    texts = run_prompt_tuning.load_corpus_texts(tmp_path, max_docs=10)
    assert len(texts) == 1 and "Vendor ships to Buyer" in texts[0]


def test_load_corpus_texts_respects_max_docs(tmp_path) -> None:
    for i in range(5):
        (tmp_path / f"{i}.txt").write_text(str(i), encoding="utf-8")
    texts = run_prompt_tuning.load_corpus_texts(tmp_path, max_docs=2)
    assert len(texts) == 2


def test_prompt_tuning_main_no_texts_returns_1(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_prompt_tuning, "load_cli_config", return_value=config)
    empty = tmp_path / "empty"
    empty.mkdir()
    mocker.patch("sys.argv", ["run-prompt-tuning", "--source-directory", str(empty)])
    # No text files -> early return 1, tuner/AWS never constructed.
    assert run_prompt_tuning.main() == 1


# --- run_visualization: parser + helpers ---------------------------------


def test_visualization_parser_requires_data_path() -> None:
    with pytest.raises(SystemExit):
        run_visualization._build_parser().parse_args([])


def test_visualization_parser_defaults() -> None:
    args = run_visualization._build_parser().parse_args(["--data-path", "g.json"])
    assert args.data_path == Path("g.json")
    assert args.output_directory == Path("visualization_outputs")
    assert isinstance(args.renderers, list) and args.renderers  # registry default


def test_visualization_parser_multiple_renderers() -> None:
    args = run_visualization._build_parser().parse_args(
        ["--data-path", "g.json", "--renderers", "interactive", "static"]
    )
    assert args.renderers == ["interactive", "static"]


def test_visualization_parser_rejects_unknown_renderer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Validated only after loading the data and computing centrality before.
    with pytest.raises(SystemExit) as excinfo:
        run_visualization._build_parser().parse_args(
            ["--data-path", "g.json", "--renderers", "interactive", "bogus"]
        )
    assert excinfo.value.code == 2
    assert "invalid choice: 'bogus'" in capsys.readouterr().err


def test_visualization_hierarchy_entries_from_dict() -> None:
    data = {"communities": {"hierarchy": [{"community_id": "c1"}]}}
    assert run_visualization._hierarchy_entries(data) == [{"community_id": "c1"}]


def test_visualization_hierarchy_entries_from_list() -> None:
    data = {"communities": [{"community_id": "c2"}]}
    assert run_visualization._hierarchy_entries(data) == [{"community_id": "c2"}]


def test_visualization_hierarchy_entries_empty() -> None:
    assert run_visualization._hierarchy_entries({}) == []


def test_visualization_to_communities_infers_size_from_nodes() -> None:
    entries = [{"community_id": "c1", "level": 0, "nodes": ["a", "b", "c"]}]
    comms = run_visualization._to_communities(entries)
    assert len(comms) == 1
    assert comms[0].id == "c1"
    assert comms[0].size == 3


def test_visualization_to_hierarchical_communities() -> None:
    entries = [
        {
            "community_id": "c1",
            "level": 1,
            "nodes": ["a", "b"],
            "parent": "root",
            "children": ["c2"],
        }
    ]
    hcs = run_visualization._to_hierarchical_communities(entries)
    assert len(hcs) == 1
    assert hcs[0].community_id == "c1"
    assert hcs[0].level == 1
    assert hcs[0].nodes == {"a", "b"}
    assert hcs[0].parent_id == "root"
    assert hcs[0].children_ids == ["c2"]


def test_visualization_load_render_context_roundtrip(tmp_path) -> None:
    import json

    data = {
        "nodes": [{"id": "n1", "attributes": {"label": "x"}}, {"id": "n2"}],
        "edges": [{"source": "n1", "target": "n2", "attributes": {"w": 1}}],
        "communities": {"hierarchy": []},
        "centrality": {"n1": {"node_id": "n1", "degree": 0.5}},
        "layout": {"n1": [0.0, 0.0]},
    }
    path = tmp_path / "viz.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    ctx = run_visualization.load_render_context(path)
    assert ctx.graph.number_of_nodes() == 2
    assert ctx.graph.number_of_edges() == 1
    assert "n1" in ctx.centrality
    assert ctx.layout == {"n1": [0.0, 0.0]}


# --- run_rag_chain: RAGChainRunner._run_query execution path (AWS-free) ------


async def test_rag_run_query_success_wraps_result(config, mocker) -> None:
    from unittest.mock import AsyncMock

    from unified_kg_rag.application.cli.run_rag_chain import RAGInput

    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    args = _rag_parser().parse_args(["-q", "hi"])
    runner = run_rag_chain.RAGChainRunner(args)
    # Stub the chain so no AWS is touched.
    runner.rag_chain = mocker.MagicMock()
    result = mocker.MagicMock()
    result.model_dump.return_value = {"answer": "A", "sources": []}
    runner.rag_chain.ainvoke = AsyncMock(return_value=result)

    out = await runner._run_query(RAGInput(query="hi"))
    assert out["success"] is True
    assert out["answer"] == "A"


async def test_rag_search_mode_runs_retrieval_only(config, mocker) -> None:
    from unittest.mock import AsyncMock

    from unified_kg_rag.application.cli.run_rag_chain import RAGInput

    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    args = _rag_parser().parse_args(["-q", "hi", "--mode", "search"])
    runner = run_rag_chain.RAGChainRunner(args)
    runner.rag_chain = mocker.MagicMock()
    result = mocker.MagicMock()
    result.model_dump.return_value = {"search_results": {"results": []}}
    runner.rag_chain.asearch = AsyncMock(return_value=result)
    runner.rag_chain.ainvoke = AsyncMock()

    out = await runner._run_query(RAGInput(query="hi"))
    assert out["success"] is True
    assert out["search_results"] == {"results": []}
    runner.rag_chain.ainvoke.assert_not_awaited()


async def test_rag_run_query_error_is_captured(config, mocker) -> None:
    from unittest.mock import AsyncMock

    from unified_kg_rag.application.cli.run_rag_chain import RAGInput

    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    args = _rag_parser().parse_args(["-q", "hi"])
    runner = run_rag_chain.RAGChainRunner(args)
    runner.rag_chain = mocker.MagicMock()
    runner.rag_chain.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    out = await runner._run_query(RAGInput(query="hi", conversation_id="c1"))
    assert out["success"] is False
    assert out["error"] == "boom"
    assert out["conversation_id"] == "c1"  # preserved for the caller


async def test_rag_run_query_raises_without_chain(config, mocker) -> None:
    from unified_kg_rag.application.cli.run_rag_chain import RAGInput

    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=config)
    args = _rag_parser().parse_args(["-q", "hi"])
    runner = run_rag_chain.RAGChainRunner(args)
    runner.rag_chain = None
    with pytest.raises(RuntimeError, match="not initialized"):
        await runner._run_query(RAGInput(query="hi"))


# --- run_ingestion_pipeline: metadata operations exit codes ---------------


def _metadata_runner(config, mocker, *flags: str):
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    mocker.patch.object(run_ingestion_pipeline.Confirm, "ask", return_value=True)
    mocker.patch.object(run_ingestion_pipeline.sys.stdin, "isatty", return_value=True)
    args = _ing_parser().parse_args([*flags, "--pipeline-id", "pid"])
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    runner.pipeline = mocker.MagicMock()
    return runner


def test_verify_metadata_unknown_pipeline_reports_not_found(config, mocker) -> None:
    from unified_kg_rag.shared import PipelineExecutionError

    runner = _metadata_runner(config, mocker, "--verify-metadata")
    runner.pipeline.state_manager.pipeline_exists.return_value = False
    with pytest.raises(PipelineExecutionError, match="not found"):
        runner._handle_metadata_operations()
    runner.pipeline.verify_pipeline_metadata.assert_not_called()


@pytest.mark.parametrize("name", ["GRAPH_EXTRACTION", "Graph_Extraction", " gleaning"])
def test_resume_from_stage_is_case_insensitive(name) -> None:
    args = _ing_parser().parse_args(["--resume-from-stage", name])
    assert args.resume_from_stage == name.strip().lower()


def test_resume_from_stage_rejects_an_unknown_stage(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        _ing_parser().parse_args(["--resume-from-stage", "grpah_extraction"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "grpah_extraction" in err and "graph_extraction" in err


def test_missing_config_path_is_a_usage_error(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        _ing_parser().parse_args(["--config-path", str(tmp_path / "nope.yaml")])
    assert exc.value.code == 2
    assert "configuration file not found" in capsys.readouterr().err


def test_force_rebuild_still_writes_the_stage_cache(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(["--source-directory", str(src), "--force-rebuild"])
    pipeline_config = run_ingestion_pipeline.IngestionPipelineRunner(
        args
    )._create_pipeline_config()
    assert pipeline_config.force_rebuild is True
    assert pipeline_config.cache_enabled is True


def test_verify_metadata_valid_is_handled(config, mocker) -> None:
    runner = _metadata_runner(config, mocker, "--verify-metadata")
    runner.pipeline.verify_pipeline_metadata.return_value = True
    assert runner._handle_metadata_operations() is True


def test_repair_metadata_is_gone(capsys) -> None:
    # It loaded the metadata it was meant to repair, so on a corrupt file it
    # failed exactly like --verify-metadata and never repaired anything.
    with pytest.raises(SystemExit) as exc:
        _ing_parser().parse_args(["--repair-metadata", "--pipeline-id", "pid"])
    assert exc.value.code == 2
    assert "--repair-metadata" in capsys.readouterr().err


def test_verify_metadata_corrupt_fails(config, mocker) -> None:
    from unified_kg_rag.shared import PipelineExecutionError

    runner = _metadata_runner(config, mocker, "--verify-metadata")
    runner.pipeline.verify_pipeline_metadata.return_value = False
    with pytest.raises(PipelineExecutionError, match="corrupted"):
        runner._handle_metadata_operations()


@pytest.mark.parametrize("fails", [False, True])
def test_eval_main_closes_the_chain(config, mocker, fails) -> None:
    """run-eval must release the chain's retriever sockets on every exit path,
    or the process ends with "Unclosed client session / connector" warnings."""
    chain = MagicMock()
    chain.aclose = AsyncMock()
    mocker.patch.object(run_evaluation, "GraphRAGChain", return_value=chain)
    mocker.patch.object(run_evaluation, "load_cli_config", return_value=config)
    mocker.patch.object(run_evaluation, "setup_logging")
    # The endpoint preflight runs before the chain is built; this test is
    # about closing the chain, so let the preflight pass.
    mocker.patch.object(run_evaluation, "missing_endpoints_error", return_value=None)
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=RuntimeError("boom") if fails else None)
    runner.run.return_value = 0
    mocker.patch.object(run_evaluation, "EvaluationRunner", return_value=runner)
    mocker.patch.object(sys, "argv", ["run-eval", "--data-path", "d.json"])

    with pytest.raises(SystemExit) if fails else contextlib.nullcontext():
        run_evaluation.main()

    chain.aclose.assert_awaited_once()


# --- help epilogs ---------------------------------------------------------


def _guide_section_slugs() -> set[str]:
    """GitHub-style section links of the user guide's headings."""
    guide = Path(__file__).resolve().parents[2] / "docs" / "user-guide.md"
    slugs = set()
    for line in guide.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip().lower()
            slug = "".join(c for c in title if c.isalnum() or c in " -_")
            slugs.add(slug.replace(" ", "-"))
    return slugs


@pytest.mark.parametrize(
    ("parser", "command"),
    [
        (_ing_parser, "run-ingestion"),
        (_rag_parser, "run-rag"),
        (_eval_parser, "run-eval"),
        (run_visualization._build_parser, "run-visualization"),
        (run_prompt_tuning._build_parser, "run-prompt-tuning"),
    ],
)
def test_help_epilog_shows_an_example_and_a_live_guide_section(parser, command) -> None:
    help_text = parser().format_help()
    assert f"example:\n  {command} " in help_text
    url = help_text.split("documentation:\n  ", 1)[1].split()[0]
    assert url.startswith(
        "https://github.com/awslabs/unified-kg-rag-on-aws/blob/main/docs/user-guide.md#"
    )
    assert url.split("#", 1)[1] in _guide_section_slugs()


@pytest.mark.parametrize("suffix", ["Bad", "a/b", "a,b", "x*", ""])
def test_an_invalid_suffix_is_a_usage_error(suffix: str) -> None:
    # Before, it was accepted and every query failed (exit 1).
    eval_parser = run_evaluation.CommandLineInterface._setup_arguments()
    for parser, args in (
        (_rag_parser(), ["--query", "q"]),
        (eval_parser, ["--data-path", "eval.json"]),
    ):
        assert parser.parse_args([*args, "--suffix", "ok"]).suffix == "ok"
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([*args, "--suffix", suffix])
        assert exc.value.code == 2


def test_a_valid_suffix_is_kept() -> None:
    assert (
        _rag_parser().parse_args(["--query", "q", "--suffix", "tenant-a"]).suffix
        == "tenant-a"
    )
