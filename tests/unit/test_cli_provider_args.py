# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The provider flags of run-ingestion, run-rag, run-eval and run-prompt-tuning.

Parsing only, plus the runners' config loading: nothing is built.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
)
from unified_kg_rag.application.cli.provider_args import apply_provider_arguments
from unified_kg_rag.domain.models import Backend, Config

pytestmark = pytest.mark.unit

_PARSERS = {
    "run-ingestion": (
        run_ingestion_pipeline.CommandLineInterface._setup_arguments,
        ["--source-directory", "src"],
    ),
    "run-rag": (run_rag_chain.CommandLineInterface._setup_arguments, ["-q", "x"]),
    "run-eval": (
        run_evaluation.CommandLineInterface._setup_arguments,
        ["--eval-data-path", "eval.json"],
    ),
    "run-prompt-tuning": (run_prompt_tuning._build_parser, ["--source-directory", "s"]),
}


@pytest.mark.parametrize("cli", sorted(_PARSERS))
def test_every_cli_accepts_the_provider_flags(cli: str) -> None:
    build, required = _PARSERS[cli]
    parser: argparse.ArgumentParser = build()
    args = parser.parse_args(
        [
            *required,
            "--backend",
            "local",
            "--llm-provider",
            "openai_compatible",
            "--llm-model",
            "served-model",
            "--embedding-provider",
            "openai_compatible",
            "--embedding-model",
            "served-embed",
            "--embedding-dimension",
            "384",
            "--rerank-provider",
            "none",
            "--model-base-url",
            "http://model-server.test/v1",
        ]
    )
    config = apply_provider_arguments(Config(), args)
    assert config.backend is Backend.LOCAL
    assert config.providers.llm.kind == "openai_compatible"
    assert config.providers.llm.base_url == "http://model-server.test/v1"
    assert config.search.answer_generation_model_id == "served-model"
    assert config.indexing.opensearch.embedding_model_id == "served-embed"
    assert config.providers.embedding.dimension == 384
    assert config.providers.rerank.kind == "none"
    # The rest of the local preset still applies.
    assert config.providers.doc_status.kind == "sqlite"


@pytest.mark.parametrize("cli", sorted(_PARSERS))
def test_without_flags_the_config_is_unchanged(cli: str) -> None:
    build, required = _PARSERS[cli]
    args = build().parse_args(required)
    config = Config()
    assert apply_provider_arguments(config, args) is config


def test_backend_choices_are_validated() -> None:
    parser = run_rag_chain.CommandLineInterface._setup_arguments()
    with pytest.raises(SystemExit):
        parser.parse_args(["-q", "x", "--backend", "cloud"])


def test_rag_runner_applies_the_flags(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  log_to_file: false\n", encoding="utf-8")
    args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["-q", "x", "--config-path", str(path), "--backend", "local"]
    )
    runner = run_rag_chain.RAGChainRunner(args)
    assert runner.config.backend is Backend.LOCAL
    assert runner.config.logging.log_to_file is False
