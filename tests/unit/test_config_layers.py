# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config layer precedence: defaults < backend preset < file < env < CLI flags.

The CLI provider flags rebuild the loaded config from what it was built from
(``with_provider_overrides``); every environment override must survive that
rebuild, on the path all five CLIs share (``get_config`` then
``apply_provider_arguments``). AWS-free: nothing is built.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
    run_visualization,
)
from unified_kg_rag.application.cli.provider_args import apply_provider_arguments
from unified_kg_rag.domain.models import Backend, Config
from unified_kg_rag.shared.config import ENV_OVERRIDES, ConfigLoader

pytestmark = pytest.mark.unit

_IMAGE_CONFIG = Path(__file__).resolve().parents[2] / "docker" / "config.yaml"

# A valid value for every mapped variable, unlike both the default and
# docker/config.yaml, with the value the config holds once it is applied.
_SAMPLES: dict[str, tuple[str, Any]] = {
    "AWS_PROFILE": ("example-profile", "example-profile"),
    "AWS_REGION": ("eu-west-1", "eu-west-1"),
    "LOG_LEVEL": ("debug", "DEBUG"),
    "LOG_FORMAT": ("plain", "plain"),
    "LOG_TO_FILE": ("true", True),
    "LOG_FILE_PATH": ("logs/example.log", "logs/example.log"),
    "NEPTUNE_ENDPOINT": ("neptune.example.internal", "neptune.example.internal"),
    "OPENSEARCH_ENDPOINT": (
        "opensearch.example.internal",
        "opensearch.example.internal",
    ),
    "OPENSEARCH_USERNAME": ("example-user", "example-user"),
    "OPENSEARCH_PASSWORD": ("example-pass", "example-pass"),  # nosec B105
    "BEDROCK_REGION": ("eu-central-1", "eu-central-1"),
    "BEDROCK_GUARDRAIL_IDENTIFIER": ("gr-example", "gr-example"),
    "BEDROCK_GUARDRAIL_VERSION": ("3", "3"),
    "S3_BUCKET_NAME": ("example-bucket", "example-bucket"),
    "OPENSEARCH_NUMBER_OF_REPLICAS": ("2", 2),
    "OPENSEARCH_KOREAN_ANALYZER": ("cjk", "cjk"),
    "GRAPHRAG_DOC_STATUS_TABLE": ("example-doc-status", "example-doc-status"),
    "GRAPHRAG_SOURCE_SCOPE": (
        "s3://example-bucket/corpus/",
        "s3://example-bucket/corpus/",
    ),
    "GRAPHRAG_DOC_STATUS_CREATE_TABLE": ("false", False),
    "GRAPHRAG_OBSERVABILITY_ENABLED": ("true", True),
    "GRAPHRAG_OBSERVABILITY_BACKEND": ("otlp", "otlp"),
}

# The environment of a deployed task (CDK compute stack + entrypoint).
_TASK_ENV = {
    var: _SAMPLES[var][0]
    for var in (
        "AWS_REGION",
        "BEDROCK_REGION",
        "NEPTUNE_ENDPOINT",
        "OPENSEARCH_ENDPOINT",
        "OPENSEARCH_NUMBER_OF_REPLICAS",
        "OPENSEARCH_KOREAN_ANALYZER",
        "S3_BUCKET_NAME",
        "GRAPHRAG_DOC_STATUS_TABLE",
        "GRAPHRAG_DOC_STATUS_CREATE_TABLE",
        "LOG_FORMAT",
        "BEDROCK_GUARDRAIL_IDENTIFIER",
        "BEDROCK_GUARDRAIL_VERSION",
        "GRAPHRAG_OBSERVABILITY_ENABLED",
        "GRAPHRAG_OBSERVABILITY_BACKEND",
        "GRAPHRAG_SOURCE_SCOPE",
    )
}

_PARSERS = {
    "run-ingestion": (
        run_ingestion_pipeline.CommandLineInterface._setup_arguments,
        ["--source-directory", "src"],
    ),
    "run-rag": (run_rag_chain.CommandLineInterface._setup_arguments, ["-q", "x"]),
    "run-eval": (
        run_evaluation.CommandLineInterface._setup_arguments,
        ["--data-path", "eval.json"],
    ),
    "run-prompt-tuning": (run_prompt_tuning._build_parser, ["--source-directory", "s"]),
    "run-visualization": (run_visualization._build_parser, ["--data-path", "v.json"]),
}

# Every combination of the provider flags (None = flag omitted).
_FLAG_VALUES: dict[str, tuple[Any, ...]] = {
    "backend": (None, "aws", "local"),
    "rerank_provider": (None, "none"),
    "llm": (None, ("openai_compatible", "served-model")),
    "embedding": (None, ("openai_compatible", "served-embed", 384)),
    "model_base_url": (None, "http://model-server.test/v1"),
}


def _flag_namespaces() -> list[argparse.Namespace]:
    namespaces = []
    for values in itertools.product(*_FLAG_VALUES.values()):
        flags = dict(zip(_FLAG_VALUES, values, strict=True))
        llm, embedding = flags.pop("llm"), flags.pop("embedding")
        namespaces.append(
            argparse.Namespace(
                **flags,
                llm_provider=llm and llm[0],
                llm_model=llm and llm[1],
                embedding_provider=embedding and embedding[0],
                embedding_model=embedding and embedding[1],
                embedding_dimension=embedding and embedding[2],
            )
        )
    return namespaces


def _get(config: Config, path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key] if isinstance(value, dict) else getattr(value, key)
    return value.get_secret_value() if isinstance(value, SecretStr) else value


def _load(config_path: Path | None) -> Config:
    return ConfigLoader(config_path).load_config()


@pytest.fixture
def local_file(tmp_path: Path) -> Path:
    path = tmp_path / "local.yaml"
    path.write_text("backend: local\nlogging:\n  log_to_file: false\n", "utf-8")
    return path


def test_every_env_override_has_a_sample() -> None:
    assert set(_SAMPLES) == set(ENV_OVERRIDES)


@pytest.mark.parametrize("cli", sorted(_PARSERS))
def test_task_env_survives_rerank_provider_none_on_the_image_config(
    cli: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # run-rag / run-eval in the deployed task: docker/config.yaml + the stack's
    # environment + `--rerank-provider none`. The guardrail id in particular
    # must not be dropped (the query guardrail would silently turn off).
    for var, value in _TASK_ENV.items():
        monkeypatch.setenv(var, value)
    build, required = _PARSERS[cli]
    args = build().parse_args(
        [*required, "--config-path", str(_IMAGE_CONFIG), "--rerank-provider", "none"]
    )
    config = apply_provider_arguments(_load(Path(args.config_path)), args)
    assert config.providers.rerank.kind == "none"
    for var in _TASK_ENV:
        assert _get(config, ENV_OVERRIDES[var]) == _SAMPLES[var][1], var
    assert config.aws.bedrock.guardrail.identifier == "gr-example"
    assert config.aws.opensearch.use_iam is True  # the file's value is kept too


@pytest.mark.parametrize("cli", sorted(_PARSERS))
def test_env_survives_backend_aws_without_a_config_file(
    cli: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("NEPTUNE_ENDPOINT", "OPENSEARCH_ENDPOINT", "LOG_LEVEL"):
        monkeypatch.setenv(var, _SAMPLES[var][0])
    build, required = _PARSERS[cli]
    args = build().parse_args([*required, "--backend", "aws"])
    config = apply_provider_arguments(_load(None), args)
    assert config.aws.neptune.endpoint == "neptune.example.internal"
    assert config.aws.opensearch.endpoint == "opensearch.example.internal"
    assert config.logging.level == "DEBUG"


@pytest.mark.parametrize("env_var", sorted(ENV_OVERRIDES))
@pytest.mark.parametrize("config_file", ["none", "image", "local"])
def test_each_env_override_survives_every_flag_combination(
    env_var: str,
    config_file: str,
    local_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = {"none": None, "image": _IMAGE_CONFIG, "local": local_file}[config_file]
    raw, expected = _SAMPLES[env_var]
    monkeypatch.setenv(env_var, raw)
    loaded = _load(path)
    assert _get(loaded, ENV_OVERRIDES[env_var]) == expected
    for args in _flag_namespaces():
        config = apply_provider_arguments(loaded, args)
        assert _get(config, ENV_OVERRIDES[env_var]) == expected, args


def test_env_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("aws:\n  neptune:\n    endpoint: file.example\n", "utf-8")
    monkeypatch.setenv("NEPTUNE_ENDPOINT", "env.example")
    args = argparse.Namespace(rerank_provider="none")
    config = apply_provider_arguments(_load(path), args)
    assert config.aws.neptune.endpoint == "env.example"


def test_env_wins_over_the_backend_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    # The local preset points aws.neptune at localhost; the variable wins,
    # whether the preset comes from the file or from --backend local.
    monkeypatch.setenv("NEPTUNE_ENDPOINT", "env.example")
    config = apply_provider_arguments(_load(None), argparse.Namespace(backend="local"))
    assert config.backend is Backend.LOCAL
    assert config.aws.neptune.endpoint == "env.example"
    assert config.aws.neptune.port == 8182  # the rest of the preset applies


def test_backend_switch_strips_the_preset_but_keeps_the_env(
    local_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENSEARCH_ENDPOINT", "env.example")
    config = apply_provider_arguments(
        _load(local_file), argparse.Namespace(backend="aws")
    )
    assert config.providers.graph.kind == "neptune"
    assert config.aws.neptune.endpoint is None  # the preset's localhost is gone
    assert config.aws.opensearch.endpoint == "env.example"
    assert config.logging.log_to_file is False  # what the file set stays


def test_cli_flag_wins_over_env_for_the_same_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No shipped variable maps to a key a provider flag sets, so one is added:
    # the order must hold for any mapping.
    monkeypatch.setitem(
        ENV_OVERRIDES, "EXAMPLE_LLM_BASE_URL", ("providers", "llm", "base_url")
    )
    monkeypatch.setenv("EXAMPLE_LLM_BASE_URL", "http://env.test/v1")
    loaded = _load(None)
    assert loaded.providers.llm.base_url == "http://env.test/v1"

    flagged = apply_provider_arguments(
        loaded, argparse.Namespace(model_base_url="http://cli.test/v1")
    )
    assert flagged.providers.llm.base_url == "http://cli.test/v1"

    other_flag = apply_provider_arguments(
        loaded, argparse.Namespace(rerank_provider="none")
    )
    assert other_flag.providers.llm.base_url == "http://env.test/v1"


def test_env_value_is_validated_with_the_rest_of_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Applied before validation, a username from the environment keeps the
    # local preset's anonymous access off, as the same key in the file does.
    monkeypatch.setenv("OPENSEARCH_USERNAME", "example-user")
    config = apply_provider_arguments(_load(None), argparse.Namespace(backend="local"))
    assert config.aws.opensearch.username == "example-user"
    assert config.aws.opensearch.allow_anonymous is False
