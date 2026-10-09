# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CLIs reject missing store endpoints before any paid model call."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_rag_chain,
)
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    strategy_roles,
)
from unified_kg_rag.domain.models import Config, RetrieverRole, SearchStrategy

pytestmark = pytest.mark.unit

_GRAPH, _DOC = RetrieverRole.GRAPH, RetrieverRole.DOCUMENT


def _config(neptune: str | None = None, opensearch: str | None = None) -> Config:
    config = Config()
    config.aws.neptune.endpoint = neptune
    config.aws.opensearch.endpoint = opensearch
    return config


@pytest.mark.parametrize(
    ("strategy", "roles"),
    [
        (SearchStrategy.SIMPLE, {_DOC}),
        (SearchStrategy.NAIVE, {_DOC}),
        (SearchStrategy.GLOBAL, {_DOC}),
        (SearchStrategy.LOCAL, {_DOC, _GRAPH}),
        (SearchStrategy.MIX, {_DOC, _GRAPH}),
        (SearchStrategy.AUTO, {_DOC, _GRAPH}),
    ],
)
def test_strategy_roles(strategy: SearchStrategy, roles: set[RetrieverRole]) -> None:
    assert strategy_roles(Config(), strategy) == roles


def test_auto_roles_follow_the_routable_strategies() -> None:
    config = Config()
    config.search.auto_routable_strategies = [SearchStrategy.GLOBAL]
    assert strategy_roles(config, SearchStrategy.AUTO) == {_DOC}


def test_missing_endpoints_error_names_config_key_and_env_var() -> None:
    error = missing_endpoints_error(_config(), {_DOC, _GRAPH}, "indexing")
    assert error is not None
    assert "aws.neptune.endpoint (env NEPTUNE_ENDPOINT)" in error
    assert "aws.opensearch.endpoint (env OPENSEARCH_ENDPOINT)" in error
    assert missing_endpoints_error(_config("n", "o"), {_DOC, _GRAPH}, "x") is None
    # Only the roles asked for are checked.
    assert missing_endpoints_error(_config(opensearch="o"), {_DOC}, "x") is None


# --- run-ingestion -------------------------------------------------------


def _ingestion_runner(mocker, tmp_path: Path, config: Config, *argv: str):
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    mocker.patch.object(run_ingestion_pipeline, "display_ascii_art")
    src = tmp_path / "src"
    src.mkdir()
    parser = run_ingestion_pipeline.CommandLineInterface._setup_arguments()
    args = parser.parse_args(
        ["--source-directory", str(src), "--cache-directory", str(tmp_path), *argv]
    )
    return run_ingestion_pipeline.IngestionPipelineRunner(args)


def test_ingestion_exits_before_the_pipeline_when_indexing_lacks_endpoints(
    mocker, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _ingestion_runner(mocker, tmp_path, _config(neptune="n"))
    pipeline = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    with pytest.raises(SystemExit) as exc:
        runner.run()
    assert exc.value.code == 1
    pipeline.assert_not_called()
    assert "aws.opensearch.endpoint" in capsys.readouterr().out


def test_ingestion_without_the_indexing_stage_needs_no_endpoints(
    mocker, tmp_path: Path
) -> None:
    runner = _ingestion_runner(
        mocker, tmp_path, _config(), "--enabled-stages", "DOCUMENT_PARSING"
    )
    pipeline = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    mocker.patch.object(runner, "_execute_pipeline", return_value=None)
    runner.run()
    pipeline.assert_called_once()


# --- run-rag -------------------------------------------------------------


async def test_rag_exits_before_the_chain_when_the_strategy_lacks_endpoints(
    mocker, capsys: pytest.CaptureFixture[str]
) -> None:
    mocker.patch.object(run_rag_chain, "get_config", return_value=_config())
    create = mocker.patch.object(run_rag_chain, "create_rag_chain")
    parser = run_rag_chain.CommandLineInterface._setup_arguments()
    args = parser.parse_args(["-q", "hi", "--search-strategy", "simple"])
    assert await run_rag_chain.async_main(args) == 1
    create.assert_not_called()
    out = capsys.readouterr().out
    assert "aws.opensearch.endpoint" in out
    assert "aws.neptune.endpoint" not in out  # simple does not use the graph


# --- run-eval ------------------------------------------------------------


def test_eval_exits_before_the_chain_when_the_strategy_lacks_endpoints(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "eval.json"
    data.write_text("[]")
    monkeypatch.setattr(sys, "argv", ["run-eval", "--eval-data-path", str(data)])
    mocker.patch.object(
        run_evaluation, "get_config", return_value=_config(opensearch="o")
    )
    chain = mocker.patch.object(run_evaluation, "GraphRAGChain")
    with pytest.raises(SystemExit) as exc:
        run_evaluation.main()
    assert exc.value.code == 1
    chain.assert_not_called()
    assert "aws.neptune.endpoint" in capsys.readouterr().out
