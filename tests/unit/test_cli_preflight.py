# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CLIs reject missing store endpoints and optional extras before any
paid model call."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from unified_kg_rag.adapters import provider_registry
from unified_kg_rag.adapters.provider_registry import (
    ProviderRole,
    provider_endpoint,
    provider_requirement,
    register_provider,
)
from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
)
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    missing_extras_error,
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


def test_auto_roles_skip_strategies_the_pipeline_cannot_serve() -> None:
    # Without community reports auto never routes to global/drift, so only the
    # roles of local/mix (both stores) count; with only global configured, auto
    # falls back to local.
    config = Config()
    config.graph.community_detection.enabled = False
    config.search.auto_routable_strategies = [SearchStrategy.GLOBAL]
    assert strategy_roles(config, SearchStrategy.AUTO) == {_DOC, _GRAPH}


def test_missing_endpoints_error_names_config_key_and_env_var() -> None:
    error = missing_endpoints_error(_config(), {_DOC, _GRAPH}, "indexing")
    assert error is not None
    assert "aws.neptune.endpoint (env UKR__AWS__NEPTUNE__ENDPOINT)" in error
    assert "aws.opensearch.endpoint (env UKR__AWS__OPENSEARCH__ENDPOINT)" in error
    assert missing_endpoints_error(_config("n", "o"), {_DOC, _GRAPH}, "x") is None
    # Only the roles asked for are checked.
    assert missing_endpoints_error(_config(opensearch="o"), {_DOC}, "x") is None


@pytest.mark.parametrize(
    ("role", "kind", "key"),
    [
        (ProviderRole.GRAPH_RETRIEVER, "neptune", "aws.neptune.endpoint"),
        (ProviderRole.GRAPH_RETRIEVER, "gremlin", "aws.neptune.endpoint"),
        (ProviderRole.VECTOR_RETRIEVER, "opensearch", "aws.opensearch.endpoint"),
    ],
)
def test_server_store_kinds_declare_their_endpoint(
    role: ProviderRole, kind: str, key: str
) -> None:
    import unified_kg_rag.adapters.providers  # noqa: F401  (registers the kinds)

    endpoint = provider_endpoint(role, kind)
    assert endpoint is not None and endpoint[0] == key
    assert provider_endpoint(role, f" {kind.upper()} ") == endpoint


@pytest.fixture
def embedded_kind() -> Iterator[str]:
    kind = "synthetic_embedded"
    for role in (ProviderRole.GRAPH_RETRIEVER, ProviderRole.VECTOR_RETRIEVER):
        register_provider(role, kind)(lambda providers: None)
    yield kind
    for role in ProviderRole:
        provider_registry._REGISTRY.pop((role, kind), None)


def test_kind_without_an_endpoint_is_not_checked(embedded_kind: str) -> None:
    config = _config()
    config.providers.graph.kind = embedded_kind
    assert missing_endpoints_error(config, {_GRAPH}, "x") is None
    # The other role still follows its own kind (opensearch, unset).
    error = missing_endpoints_error(config, {_DOC, _GRAPH}, "x")
    assert error is not None
    assert "aws.opensearch.endpoint" in error
    assert "aws.neptune.endpoint" not in error


def test_reregistering_a_kind_replaces_its_endpoint() -> None:
    kind = "synthetic_server"
    role = ProviderRole.GRAPH_RETRIEVER
    try:
        register_provider(role, kind, endpoint=("aws.neptune.endpoint", "X"))(
            lambda providers: None
        )
        assert provider_endpoint(role, kind) == ("aws.neptune.endpoint", "X")
        register_provider(role, kind)(lambda providers: None)
        assert provider_endpoint(role, kind) is None
    finally:
        provider_registry._REGISTRY.pop((role, kind), None)


# --- run-ingestion -------------------------------------------------------


def _ingestion_runner(mocker, tmp_path: Path, config: Config, *argv: str):
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
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
    assert exc.value.code == 2  # a configuration error
    pipeline.assert_not_called()
    assert "aws.opensearch.endpoint" in capsys.readouterr().err


def test_ingestion_checks_both_store_endpoints_through_the_retriever_kinds(
    mocker, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The indexer kinds declare no endpoint; run-ingestion still checks the
    # graph and vector endpoints, resolved via the shared providers sections.
    import unified_kg_rag.adapters.providers  # noqa: F401  (registers the kinds)

    assert provider_endpoint(ProviderRole.GRAPH_INDEXER, "neptune") is None
    runner = _ingestion_runner(mocker, tmp_path, _config())
    pipeline = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    with pytest.raises(SystemExit):
        runner.run()
    pipeline.assert_not_called()
    out = " ".join(capsys.readouterr().err.split())  # undo console wrapping
    assert "aws.neptune.endpoint (env UKR__AWS__NEPTUNE__ENDPOINT)" in out
    assert "aws.opensearch.endpoint (env UKR__AWS__OPENSEARCH__ENDPOINT)" in out


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
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=_config())
    create = mocker.patch.object(run_rag_chain, "create_rag_chain")
    parser = run_rag_chain.CommandLineInterface._setup_arguments()
    args = parser.parse_args(["-q", "hi", "--search-strategy", "simple"])
    assert await run_rag_chain.async_main(args) == 2  # a configuration error
    create.assert_not_called()
    out = capsys.readouterr().err
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
    monkeypatch.setattr(sys, "argv", ["run-eval", "--data-path", str(data)])
    mocker.patch.object(
        run_evaluation, "load_cli_config", return_value=_config(opensearch="o")
    )
    chain = mocker.patch.object(run_evaluation, "GraphRAGChain")
    with pytest.raises(SystemExit) as exc:
        run_evaluation.main()
    assert exc.value.code == 2  # a configuration error
    chain.assert_not_called()
    assert "aws.neptune.endpoint" in capsys.readouterr().err


# --- missing optional extras ---------------------------------------------


@pytest.fixture
def without_local_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes the module unimportable (and find_spec None).
    for module in ("langchain_ollama", "langchain_openai"):
        monkeypatch.setitem(sys.modules, module, None)


def _local_config() -> Config:
    return Config(backend="local")


def test_the_bedrock_kinds_need_no_extra(without_local_extra: None) -> None:
    assert missing_extras_error(Config()) is None


def test_a_missing_extra_is_named_with_the_settings_that_need_it(
    without_local_extra: None,
) -> None:
    error = missing_extras_error(_local_config())
    assert error is not None
    assert (
        "'langchain_ollama' (for providers.llm.kind 'ollama', "
        "providers.embedding.kind 'ollama')" in error
    )
    assert error.count("uv sync --extra local") == 1


def test_an_installed_extra_passes() -> None:
    pytest.importorskip("langchain_ollama")
    assert missing_extras_error(_local_config()) is None


def test_reregistering_a_kind_replaces_its_requirement() -> None:
    kind = "synthetic_model_server"
    role = ProviderRole.LLM
    config = Config()
    config.providers.llm.kind = kind
    try:
        register_provider(role, kind, requires=("synthetic_absent_module", "x"))(
            lambda providers: None
        )
        assert provider_requirement(role, kind) == ("synthetic_absent_module", "x")
        assert "'synthetic_absent_module'" in (missing_extras_error(config) or "")
        register_provider(role, kind)(lambda providers: None)
        assert provider_requirement(role, kind) is None
        assert missing_extras_error(config) is None
    finally:
        provider_registry._REGISTRY.pop((role, kind), None)


def test_ingestion_exits_before_the_pipeline_without_the_extra(
    mocker,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    without_local_extra: None,
) -> None:
    runner = _ingestion_runner(mocker, tmp_path, _local_config())
    pipeline = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    with pytest.raises(SystemExit) as exc:
        runner.run()
    assert exc.value.code == 2  # a configuration error
    pipeline.assert_not_called()
    assert "uv sync --extra local" in " ".join(capsys.readouterr().err.split())


async def test_rag_exits_before_the_chain_without_the_extra(
    mocker, capsys: pytest.CaptureFixture[str], without_local_extra: None
) -> None:
    mocker.patch.object(run_rag_chain, "load_cli_config", return_value=_local_config())
    create = mocker.patch.object(run_rag_chain, "create_rag_chain")
    parser = run_rag_chain.CommandLineInterface._setup_arguments()
    args = parser.parse_args(["-q", "hi"])
    assert await run_rag_chain.async_main(args) == 2
    create.assert_not_called()
    assert "langchain_ollama" in capsys.readouterr().err


def test_prompt_tuning_exits_before_tuning_without_the_extra(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    without_local_extra: None,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run-prompt-tuning", "--source-directory", str(tmp_path)]
    )
    mocker.patch.object(
        run_prompt_tuning, "load_cli_config", return_value=_local_config()
    )
    tuner = mocker.patch.object(run_prompt_tuning, "PromptTuner")
    with pytest.raises(SystemExit) as exc:
        run_prompt_tuning.main()
    assert exc.value.code == 2
    tuner.assert_not_called()
    assert "langchain_ollama" in capsys.readouterr().err


def test_eval_exits_before_the_chain_without_the_extra(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    without_local_extra: None,
) -> None:
    data = tmp_path / "eval.json"
    data.write_text("[]")
    monkeypatch.setattr(sys, "argv", ["run-eval", "--data-path", str(data)])
    mocker.patch.object(run_evaluation, "load_cli_config", return_value=_local_config())
    chain = mocker.patch.object(run_evaluation, "GraphRAGChain")
    with pytest.raises(SystemExit) as exc:
        run_evaluation.main()
    assert exc.value.code == 2
    chain.assert_not_called()
    assert "langchain_ollama" in capsys.readouterr().err
