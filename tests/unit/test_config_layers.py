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
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import SecretStr

from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
    run_visualization,
)
from unified_kg_rag.application.cli.provider_args import apply_provider_arguments
from unified_kg_rag.application.config import AWS_ENV_FALLBACKS, ConfigLoader, _env_path
from unified_kg_rag.domain.models import Backend, Config

pytestmark = pytest.mark.unit

_IMAGE_CONFIG = Path(__file__).resolve().parents[2] / "docker" / "config.yaml"

# A valid value for every mapped variable, unlike both the default and
# docker/config.yaml, with the value the config holds once it is applied.
_SAMPLES: dict[str, tuple[str, Any]] = {
    "AWS_PROFILE": ("example-profile", "example-profile"),
    "AWS_REGION": ("eu-west-1", "eu-west-1"),
    "UKR__LOGGING__LEVEL": ("debug", "DEBUG"),
    "UKR__LOGGING__LOG_FORMAT": ("plain", "plain"),
    "UKR__LOGGING__LOG_TO_FILE": ("true", True),
    "UKR__LOGGING__LOG_FILE_PATH": ("logs/example.log", "logs/example.log"),
    "UKR__AWS__NEPTUNE__ENDPOINT": (
        "neptune.example.internal",
        "neptune.example.internal",
    ),
    "UKR__AWS__OPENSEARCH__ENDPOINT": (
        "opensearch.example.internal",
        "opensearch.example.internal",
    ),
    "UKR__AWS__OPENSEARCH__USERNAME": ("example-user", "example-user"),
    "UKR__AWS__OPENSEARCH__PASSWORD": ("example-pass", "example-pass"),  # nosec B105
    "UKR__AWS__BEDROCK__REGION_NAME": ("eu-central-1", "eu-central-1"),
    "UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER": ("gr-example", "gr-example"),
    "UKR__AWS__BEDROCK__GUARDRAIL__VERSION": ("3", "3"),
    "UKR__AWS__S3__BUCKET_NAME": ("example-bucket", "example-bucket"),
    "UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS": ("2", 2),
    "UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO": ("cjk", "cjk"),
    "UKR__AWS__DYNAMODB__TABLE_NAME": ("example-doc-status", "example-doc-status"),
    "UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE": (
        "s3://example-bucket/corpus/",
        "s3://example-bucket/corpus/",
    ),
    "UKR__AWS__DYNAMODB__CREATE_TABLE_IF_MISSING": ("false", False),
    "UKR__OBSERVABILITY__ENABLED": ("true", True),
    "UKR__OBSERVABILITY__BACKEND": ("otlp", "otlp"),
}

# The environment of a deployed task (CDK compute stack + entrypoint).
_TASK_ENV = {
    var: _SAMPLES[var][0]
    for var in (
        "AWS_REGION",
        "UKR__AWS__BEDROCK__REGION_NAME",
        "UKR__AWS__NEPTUNE__ENDPOINT",
        "UKR__AWS__OPENSEARCH__ENDPOINT",
        "UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS",
        "UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO",
        "UKR__AWS__S3__BUCKET_NAME",
        "UKR__AWS__DYNAMODB__TABLE_NAME",
        "UKR__AWS__DYNAMODB__CREATE_TABLE_IF_MISSING",
        "UKR__LOGGING__LOG_FORMAT",
        "UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER",
        "UKR__AWS__BEDROCK__GUARDRAIL__VERSION",
        "UKR__OBSERVABILITY__ENABLED",
        "UKR__OBSERVABILITY__BACKEND",
        "UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE",
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


def _path(var: str) -> tuple[str, ...]:
    path = AWS_ENV_FALLBACKS.get(var) or _env_path(var)
    assert path is not None, var
    return path


def test_every_sample_names_a_config_key() -> None:
    assert all(_path(var) for var in _SAMPLES)


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
        assert _get(config, _path(var)) == _SAMPLES[var][1], var
    assert config.aws.bedrock.guardrail.identifier == "gr-example"
    assert config.aws.opensearch.use_iam is True  # the file's value is kept too


@pytest.mark.parametrize("cli", sorted(_PARSERS))
def test_env_survives_backend_aws_without_a_config_file(
    cli: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "UKR__AWS__NEPTUNE__ENDPOINT",
        "UKR__AWS__OPENSEARCH__ENDPOINT",
        "UKR__LOGGING__LEVEL",
    ):
        monkeypatch.setenv(var, _SAMPLES[var][0])
    build, required = _PARSERS[cli]
    args = build().parse_args([*required, "--backend", "aws"])
    config = apply_provider_arguments(_load(None), args)
    assert config.aws.neptune.endpoint == "neptune.example.internal"
    assert config.aws.opensearch.endpoint == "opensearch.example.internal"
    assert config.logging.level == "DEBUG"


@pytest.mark.parametrize("env_var", sorted(_SAMPLES))
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
    assert _get(loaded, _path(env_var)) == expected
    for args in _flag_namespaces():
        config = apply_provider_arguments(loaded, args)
        assert _get(config, _path(env_var)) == expected, args


def test_env_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("aws:\n  neptune:\n    endpoint: file.example\n", "utf-8")
    monkeypatch.setenv("UKR__AWS__NEPTUNE__ENDPOINT", "env.example")
    args = argparse.Namespace(rerank_provider="none")
    config = apply_provider_arguments(_load(path), args)
    assert config.aws.neptune.endpoint == "env.example"


def test_env_wins_over_the_backend_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    # The local preset points aws.neptune at localhost; the variable wins,
    # whether the preset comes from the file or from --backend local.
    monkeypatch.setenv("UKR__AWS__NEPTUNE__ENDPOINT", "env.example")
    config = apply_provider_arguments(_load(None), argparse.Namespace(backend="local"))
    assert config.backend is Backend.LOCAL
    assert config.aws.neptune.endpoint == "env.example"
    assert config.aws.neptune.port == 8182  # the rest of the preset applies


def test_backend_switch_strips_the_preset_but_keeps_the_env(
    local_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UKR__AWS__OPENSEARCH__ENDPOINT", "env.example")
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
    monkeypatch.setenv("UKR__PROVIDERS__LLM__BASE_URL", "http://env.test/v1")
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
    monkeypatch.setenv("UKR__AWS__OPENSEARCH__USERNAME", "example-user")
    config = apply_provider_arguments(_load(None), argparse.Namespace(backend="local"))
    assert config.aws.opensearch.username == "example-user"
    assert config.aws.opensearch.allow_anonymous is False


# --- provenance: what the user set survives a rebuild -------------------------


def test_backend_switch_keeps_a_file_value_equal_to_the_preset(
    tmp_path: Path,
) -> None:
    # The file pins 20 LLM calls in flight, which is also the local preset's
    # value: switching to aws must not take it for the preset's and return
    # it to the AWS default (40).
    path = tmp_path / "config.yaml"
    path.write_text(
        "backend: local\nprocessing:\n  max_concurrency: 20\n"
        "providers:\n  graph:\n    kind: sqlite\n",
        "utf-8",
    )
    config = apply_provider_arguments(_load(path), argparse.Namespace(backend="aws"))
    assert config.backend is Backend.AWS
    assert config.processing.max_concurrency == 20
    assert config.providers.graph.kind == "sqlite"
    # What only the preset filled in is gone.
    assert config.providers.vector.kind == "opensearch"
    assert config.aws.neptune.endpoint is None


def test_a_change_made_to_the_object_survives_the_provider_flags() -> None:
    # A library caller changes a loaded config and then applies the flags;
    # the rebuild used to keep only what the config was validated from.
    config = Config()
    config.search.reranking.top_k = 7
    config.processing.chunking.max_chunk_size = 3000
    rebuilt = apply_provider_arguments(
        config, argparse.Namespace(rerank_provider="none", backend="local")
    )
    assert rebuilt.providers.rerank.kind == "none"
    assert rebuilt.search.reranking.top_k == 7
    assert rebuilt.processing.chunking.max_chunk_size == 3000
    # The preset applies to everything the caller did not set.
    assert rebuilt.processing.max_concurrency == 20


# --- AWS_REGION / AWS_PROFILE --------------------------------------------------


def test_aws_region_fills_only_an_unset_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As in the AWS SDKs, an explicit setting wins over the environment: a
    # shell exporting AWS_REGION no longer moves a file's region.
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_PROFILE", "shell-profile")
    path = tmp_path / "config.yaml"
    path.write_text(
        "aws:\n  region_name: ap-northeast-2\n  profile_name: file-profile\n", "utf-8"
    )
    pinned = _load(path)
    assert pinned.aws.region_name == "ap-northeast-2"
    assert pinned.aws.profile_name == "file-profile"
    unset = _load(None)
    assert unset.aws.region_name == "eu-west-1"
    assert unset.aws.profile_name == "shell-profile"
    # The prefixed variable is an explicit override and wins over the file.
    monkeypatch.setenv("UKR__AWS__REGION_NAME", "us-east-2")
    assert _load(path).aws.region_name == "us-east-2"


# --- the UKR__ scheme ----------------------------------------------------------


def test_any_key_can_be_set_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UKR__SEARCH__RERANKING__TOP_K", "12")
    monkeypatch.setenv("ukr__models__roles__answer_generation", "served-model")
    monkeypatch.setenv("UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES", '["local", "global"]')
    config = _load(None)
    assert config.search.reranking.top_k == 12
    assert config.models.roles.answer_generation == "served-model"
    assert [s.value for s in config.search.auto_routable_strategies] == [
        "local",
        "global",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "UKR__SEARCH",  # a section, not a key
        "UKR__SEARCH__RERANKING__TOP_K__X",  # past a scalar
        "UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO__X",  # two past a mapping
    ],
)
def test_a_variable_naming_no_key_is_unknown(name: str) -> None:
    assert _env_path(name) is None


@pytest.mark.parametrize(
    "section",
    [
        {"indexing": {"additional_suffix": "{}"}},
        {"processing": {"document_parsing": {"index_value": "{}"}}},
    ],
)
@pytest.mark.parametrize("value", ["V2", "a.b", "x*", "a,b"])
def test_an_index_name_part_that_opensearch_rejects_fails_at_load(
    section: dict[str, Any], value: str
) -> None:
    # Before, an uppercase suffix loaded and failed every run at indexing,
    # after the extraction had been paid for.
    import json

    from pydantic import ValidationError

    data = json.loads(json.dumps(section).replace("{}", value))
    with pytest.raises(ValidationError, match="lowercase letters"):
        Config.model_validate(data)
    ok = json.loads(json.dumps(section).replace("{}", "tenant_b-2"))
    Config.model_validate(ok)


@pytest.mark.parametrize(
    ("section", "field"),
    [("indexing", "additional_suffix"), ("document_parsing", "index_value")],
)
def test_an_index_name_part_set_in_code_is_checked_too(
    section: str, field: str
) -> None:
    # Set after load, an uppercase suffix reached the index-creation probe,
    # which blamed indexing.opensearch.index_settings for the refused name.
    from pydantic import ValidationError

    config = Config()
    owner = (
        config.indexing if section == "indexing" else config.processing.document_parsing
    )
    with pytest.raises(ValidationError, match="lowercase letters"):
        setattr(owner, field, "V2")
    setattr(owner, field, "tenant_b-2")
    assert getattr(owner, field) == "tenant_b-2"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        # A string length_unit picked the character splitter (an `is` check)
        # while the stage-cache key read tokens.
        (lambda c: c.processing.chunking, "length_unit", "words"),
        (lambda c: c.processing.chunking, "max_chunk_size", 0),
        (lambda c: c.indexing.neptune, "vertex_ids", "Custom"),
        (lambda c: c.indexing.neptune, "max_hops", 0),
        (lambda c: c.retry, "max_attempts", 0),
        (lambda c: c.retry, "graph_store_max_attempts", 0),
        (lambda c: c.aws.bedrock.guardrail, "version", ""),
        (lambda c: c.aws.bedrock.guardrail.contextual_grounding, "enabled", "maybe"),
    ],
)
def test_the_new_sections_check_settings_set_in_code(
    section: Any, field: str, value: Any
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        setattr(section(Config()), field, value)


def test_a_chunk_length_unit_set_in_code_is_the_enum() -> None:
    from unified_kg_rag.domain.models.config import ChunkLengthUnit

    chunking = Config().processing.chunking
    chunking.length_unit = "tokens"  # type: ignore[assignment]
    assert chunking.length_unit is ChunkLengthUnit.TOKENS


def test_check_output_turned_off_under_grounding_in_code_is_refused() -> None:
    from pydantic import ValidationError

    guardrail = Config().aws.bedrock.guardrail
    guardrail.contextual_grounding.enabled = True
    with pytest.raises(ValidationError, match="requires check_output"):
        guardrail.check_output = False


def test_settings_that_do_nothing_are_warned_at_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "aws": {
                    "bedrock": {
                        "guardrail": {"contextual_grounding": {"enabled": True}}
                    }
                },
                "processing": {"chunking": {"length_unit": "tokens"}},
            }
        )
    )
    with caplog.at_level(logging.WARNING):
        ConfigLoader(path).load_config()
    messages = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("Config: ")
    ]
    assert len(messages) == 2, messages
    assert "no grounding check runs" in messages[0]
    assert "character default" in messages[1]
    # Sized in tokens, or with a guardrail: nothing to warn.
    path.write_text(
        yaml.safe_dump(
            {
                "aws": {
                    "bedrock": {
                        "guardrail": {
                            "identifier": "gr-1",
                            "contextual_grounding": {"enabled": True},
                        }
                    }
                },
                "processing": {
                    "chunking": {"length_unit": "tokens", "max_chunk_size": 2000}
                },
            }
        )
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ConfigLoader(path).load_config()
    assert not [r for r in caplog.records if r.getMessage().startswith("Config: ")]


def test_opensearch_indexing_settings_set_in_code_are_checked_too() -> None:
    # Set after load, a knn key or an invalid prefix reached index creation
    # unchecked (OpenSearchIndexingConfig did not validate assignments).
    from pydantic import ValidationError

    opensearch = Config().indexing.opensearch
    with pytest.raises(ValidationError, match="knn"):
        opensearch.index_settings = {"knn": True}
    with pytest.raises(ValidationError):
        opensearch.claims_index_prefix = "_claims"
    opensearch.index_settings = {"index": {"refresh_interval": "5s"}}
    assert opensearch.index_settings == {"index.refresh_interval": "5s"}


@pytest.mark.parametrize(
    "field",
    [
        "text_units_index_prefix",
        "entities_index_prefix",
        "community_reports_index_prefix",
        "relationships_index_prefix",
        "claims_index_prefix",
        "extraction_records_index_prefix",
    ],
)
@pytest.mark.parametrize("value", ["Graph", "a.b", "x*", "_hidden", "-dash"])
def test_an_index_prefix_that_opensearch_rejects_fails_at_load(
    field: str, value: str
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=field):
        Config.model_validate({"indexing": {"opensearch": {field: value}}})
    Config.model_validate({"indexing": {"opensearch": {field: "my-prefix_2"}}})


@pytest.mark.parametrize("field", ["entity_label_prefix", "community_label_prefix"])
def test_a_graph_label_prefix_is_checked_at_load(field: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="only letters, digits"):
        Config.model_validate({"indexing": {"neptune": {field: "Bad Label"}}})
    Config.model_validate({"indexing": {"neptune": {field: "Custom_Entity-2"}}})
