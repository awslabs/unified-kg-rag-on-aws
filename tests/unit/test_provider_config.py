# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``backend`` / ``providers`` config: defaults, the local preset, model ids.

Existing configs must keep their behaviour: no ``providers`` section means the
AWS providers, and embedding/rerank ids still validate against the Bedrock
catalog (as enum members) while the Bedrock providers are selected. AWS-free:
YAML parsing and Pydantic validation only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.fixtures.config_template import commented_key_paths, uncommented_template
from unified_kg_rag.domain.models import (
    Backend,
    Config,
    EmbeddingModelId,
    ModelRole,
    RerankModelId,
    model_id_value,
)
from unified_kg_rag.domain.models.provider_config import LOCAL_PRESET
from unified_kg_rag.shared.config import ConfigLoader

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]

_LOCAL_EMBEDDING = {"model_id": "synthetic-embed", "dimension": 384}


@pytest.mark.parametrize(
    "path",
    [
        "config-template.yaml",
        "config-template.yaml (uncommented)",
        "docker/config.yaml",
    ],
)
def test_shipped_yaml_without_providers_keeps_the_aws_providers(path: str) -> None:
    # The uncommented template also sets every commented-out key (the
    # providers section and connection keys) to the default it shows.
    if path.endswith("(uncommented)"):
        raw = uncommented_template()
    else:
        raw = yaml.safe_load((_ROOT / path).read_text(encoding="utf-8")) or {}
    config = Config(**raw)
    # docker/config.local.yaml selects `backend: local`; it is covered by the
    # local preset tests below.
    assert config.backend is Backend.AWS
    kinds = {
        role: getattr(config.providers, role).kind
        for role in ("llm", "embedding", "rerank", "graph", "vector", "doc_status")
    }
    assert kinds == {
        "llm": "bedrock",
        "embedding": "bedrock",
        "rerank": "bedrock",
        "graph": "neptune",
        "vector": "opensearch",
        "doc_status": "dynamodb",
    }
    assert isinstance(config.embedding_model_id, EmbeddingModelId)
    assert isinstance(config.rerank_model_id, RerankModelId)


def test_default_config_is_unchanged_by_the_new_sections() -> None:
    config = Config()
    assert config.embedding_model_id is EmbeddingModelId.TITAN_EMBED_V2
    assert config.rerank_model_id is RerankModelId.COHERE_RERANK_V3_5
    assert config.model_id(ModelRole.ANSWER_GENERATION) == config.tier_model_id(
        "default"
    )
    assert config.doc_status_enabled is False


@pytest.mark.parametrize(
    "value",
    [EmbeddingModelId.EMBED_V4, "cohere.embed-v4:0", " cohere.embed-v4:0 "],
)
def test_bedrock_embedding_ids_accept_enum_members_and_their_values(
    value: object,
) -> None:
    config = Config(models={"embedding": {"model_id": value}})
    assert config.embedding_model_id is EmbeddingModelId.EMBED_V4


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (
            {"models": {"embedding": {"model_id": "synthetic-embed"}}},
            "models.embedding.model_id",
        ),
        ({"models": {"rerank": "synthetic-rerank"}}, "models.rerank"),
    ],
)
def test_bedrock_providers_still_reject_ids_outside_the_catalog(
    section: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Config(**section)


def test_non_bedrock_embedding_takes_any_name() -> None:
    config = Config(
        providers={"embedding": {"kind": "ollama"}},
        models={"embedding": _LOCAL_EMBEDDING},
    )
    assert config.embedding_model_id == "synthetic-embed"
    assert model_id_value(config.embedding_model_id) == "synthetic-embed"
    assert config.embedding_dimension == 384


def test_non_bedrock_embedding_requires_a_model_and_dimension() -> None:
    with pytest.raises(ValidationError, match="models.embedding.model_id"):
        Config(
            providers={"embedding": {"kind": "ollama"}},
            models={"embedding": {"model_id": "x"}},
        )


def test_tiers_and_roles_resolve_through_the_models_section() -> None:
    config = Config(
        providers={"llm": {"kind": "openai_compatible"}},
        models={
            "default": "synthetic-large",
            "fast": "synthetic-small",
            "roles": {"answer_generation": "pinned-model", "query_rewrite": "default"},
        },
    )
    assert config.model_id(ModelRole.GRAPH_EXTRACTION) == "synthetic-large"
    assert config.model_id(ModelRole.CHUNKING) == "synthetic-small"
    assert config.model_id(ModelRole.ANSWER_GENERATION) == "pinned-model"
    # A tier name moves a role to the other tier.
    assert config.model_id(ModelRole.QUERY_REWRITE) == "synthetic-large"
    assert config.model_tier("synthetic-small") == "fast"
    assert config.model_tier("pinned-model") == "default"


def test_the_fast_tier_defaults_to_the_default_model() -> None:
    config = Config(providers={"llm": {"kind": "ollama"}}, models={"default": "m"})
    assert config.model_id(ModelRole.CHUNKING) == "m"
    assert config.model_id(ModelRole.GLOBAL_MAP) == "m"


def test_bedrock_tiers_default_to_the_shipped_models() -> None:
    config = Config(models={"default": "anthropic.claude-opus-5"})
    assert config.model_id(ModelRole.ANSWER_GENERATION) == "anthropic.claude-opus-5"
    # The fast tier keeps its own Bedrock default.
    assert config.model_id(ModelRole.CHUNKING) == "anthropic.claude-haiku-5-5"


def test_non_bedrock_llm_requires_a_default_model() -> None:
    with pytest.raises(ValidationError, match="models.default is required"):
        Config(providers={"llm": {"kind": "ollama"}})


@pytest.mark.parametrize(
    ("data", "replacement"),
    [
        (
            {"providers": {"llm": {"default_model_id": "m"}}},
            "providers.llm.default_model_id -> models.default",
        ),
        (
            {"aws": {"bedrock": {"fast_model_id": "m"}}},
            "aws.bedrock.fast_model_id -> models.fast",
        ),
        (
            {"search": {"answer_generation_model_id": "m"}},
            "search.answer_generation_model_id -> models.roles.answer_generation",
        ),
        (
            {"indexing": {"opensearch": {"embedding_dimension": 512}}},
            "indexing.opensearch.embedding_dimension -> models.embedding.dimension",
        ),
        (
            {"graph": {"visualization": {"embeddings": {}}}},
            "graph.visualization.embeddings -> models.embedding.model_id",
        ),
        (
            {"search": {"reranking": {"rerank_model_id": "m"}}},
            "search.reranking.rerank_model_id -> models.rerank",
        ),
    ],
)
def test_a_removed_model_key_names_its_replacement(
    data: dict, replacement: str
) -> None:
    # The settings used to be read only under their own provider, so a
    # Bedrock config silently ignored providers.llm.default_model_id.
    with pytest.raises(ValidationError, match=replacement):
        Config(**data)


def test_every_removed_model_key_is_refused() -> None:
    from unified_kg_rag.domain.models.config import MOVED_KEYS

    for old in MOVED_KEYS:
        data: dict = {}
        node = data
        *parents, leaf = old.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = "x"
        with pytest.raises(ValidationError, match="settings moved"):
            Config(**data)


def test_settings_of_an_inactive_provider_are_reported() -> None:
    bedrock = Config(providers={"llm": {"temperature": 0.7, "base_url": "http://x"}})
    assert bedrock.inactive_provider_settings() == [
        "providers.llm.base_url",
        "providers.llm.temperature",
    ]
    local = Config(backend="local", aws={"bedrock": {"default_effort": "max"}})
    assert local.inactive_provider_settings() == ["aws.bedrock.default_effort"]
    # The preset's own endpoint settings are the active provider's.
    assert Config(backend="local").inactive_provider_settings() == []
    assert Config().inactive_provider_settings() == []


def test_the_loader_warns_on_inactive_provider_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("providers:\n  llm:\n    temperature: 0.7\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        ConfigLoader(path).load_config()
    assert "providers.llm.temperature' is ignored" in caplog.text


def test_kinds_are_normalised() -> None:
    config = Config(
        providers={"graph": {"kind": " Gremlin "}}, aws={"neptune": {"use_iam": False}}
    )
    assert config.providers.graph.kind == "gremlin"


def test_gremlin_rejects_iam_auth() -> None:
    with pytest.raises(ValidationError, match="use_iam: false"):
        Config(providers={"graph": {"kind": "gremlin"}})


def test_local_preset_fills_every_role_and_the_store_endpoints() -> None:
    config = Config(backend="local")
    preset = LOCAL_PRESET["providers"]
    models = LOCAL_PRESET["models"]
    assert config.providers.llm.kind == preset["llm"]["kind"]
    assert config.embedding_dimension == models["embedding"]["dimension"]
    assert config.providers.rerank.kind == "cross_encoder"
    # The stores are the embedded ones; the endpoints below serve a config
    # that selects the gremlin and opensearch kinds (the containers).
    assert config.providers.graph.kind == "sqlite"
    assert config.providers.vector.kind == "sqlite"
    assert config.providers.doc_status.kind == "sqlite"
    assert config.model_id(ModelRole.ANSWER_GENERATION) == models["default"]
    assert config.embedding_model_id == models["embedding"]["model_id"]
    assert config.rerank_model_id == models["rerank"]
    assert (config.aws.neptune.use_iam, config.aws.neptune.use_ssl) == (False, False)
    assert config.aws.opensearch.allow_anonymous is True
    assert config.aws.opensearch.port == 9200


def test_local_preset_keeps_twenty_llm_calls_in_flight() -> None:
    # The AWS default rose to 40; a local model server queues what it cannot
    # run at once, and the wait counts against its request timeout.
    assert Config().processing.max_concurrency == 40
    assert Config(backend="local").processing.max_concurrency == 20
    assert (
        Config(
            backend="local", processing={"max_concurrency": 8}
        ).processing.max_concurrency
        == 8
    )


def test_local_preset_yields_to_values_the_config_sets() -> None:
    config = Config(
        backend="local",
        providers={
            "llm": {"base_url": "http://gpu:11434"},
            "rerank": {"kind": "none"},
        },
        models={"default": "my-model"},
        aws={"neptune": {"endpoint": "graph.internal.example", "port": 18182}},
    )
    assert config.providers.llm.kind == "ollama"
    assert config.tier_model_id("default") == "my-model"
    assert config.providers.llm.base_url == "http://gpu:11434"
    assert config.providers.rerank.kind == "none"
    assert config.rerank_model_id is None
    assert config.aws.neptune.endpoint == "graph.internal.example"
    assert config.aws.neptune.port == 18182
    assert config.aws.neptune.use_ssl is False


def test_local_preset_keeps_its_dimension_to_its_own_embedding_model() -> None:
    # The preset's 1024 is bge-m3's; another model took it silently and
    # built an index of the wrong dimension.
    with pytest.raises(ValidationError, match="models.embedding.dimension"):
        Config(backend="local", models={"embedding": {"model_id": "nomic-embed-text"}})
    config = Config(
        backend="local",
        models={"embedding": {"model_id": "nomic-embed-text", "dimension": 768}},
    )
    assert config.embedding_dimension == 768
    assert (
        Config(
            backend="local", models={"embedding": {"model_id": "bge-m3"}}
        ).embedding_dimension
        == 1024
    )


def test_local_preset_models_yield_to_another_provider() -> None:
    # A role moved back to Bedrock keeps the Bedrock defaults, not the
    # preset's Ollama model names.
    config = Config(
        backend="local",
        providers={
            "llm": {"kind": "bedrock"},
            "embedding": {"kind": "bedrock"},
            "rerank": {"kind": "bedrock"},
        },
    )
    assert config.model_id(ModelRole.ANSWER_GENERATION) == "anthropic.claude-sonnet-5-5"
    assert config.embedding_model_id is EmbeddingModelId.TITAN_EMBED_V2
    assert config.rerank_model_id is RerankModelId.COHERE_RERANK_V3_5


def test_local_preset_skips_anonymous_access_for_authenticated_opensearch() -> None:
    config = Config(backend="local", aws={"opensearch": {"username": "admin"}})
    assert config.aws.opensearch.allow_anonymous is False
    assert config.aws.opensearch.username == "admin"


def test_preset_does_not_mutate_its_template() -> None:
    before = yaml.safe_dump(LOCAL_PRESET)
    Config(backend="local", models={"default": "x"})
    assert yaml.safe_dump(LOCAL_PRESET) == before


def test_one_switch_turns_the_registry_on() -> None:
    # aws.dynamodb.enabled and providers.doc_status.enabled both did, the
    # second winning when set; the first is now a moved key.
    assert Config().doc_status_enabled is False
    assert Config(providers={"doc_status": {"enabled": True}}).doc_status_enabled
    with pytest.raises(ValueError, match="providers.doc_status.enabled"):
        Config(aws={"dynamodb": {"enabled": True}})


def test_local_backend_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "backend: local\n" "models:\n" "  default: synthetic-llm\n",
        encoding="utf-8",
    )
    config = ConfigLoader(path).load_config()
    assert config.backend is Backend.LOCAL
    assert config.model_id(ModelRole.ANSWER_GENERATION) == "synthetic-llm"
    # The preset's embedded stores live next to the config file, whatever
    # directory the run starts in.
    assert config.providers.graph.sqlite_path == str(
        tmp_path.resolve() / ".unified-kg-rag" / "graph.sqlite3"
    )
    assert config.providers.vector.sqlite_path == str(
        tmp_path.resolve() / ".unified-kg-rag" / "vector.sqlite3"
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "conf/.unified-kg-rag/doc-status.sqlite3"),
        ("../registry.sqlite3", "registry.sqlite3"),
        (":memory:", ":memory:"),
    ],
)
def test_a_relative_sqlite_path_follows_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured, expected
) -> None:
    # The registry must not depend on the directory a run starts in.
    (tmp_path / "conf").mkdir()
    path = tmp_path / "conf" / "config.yaml"
    line = f"    sqlite_path: '{configured}'\n" if configured else ""
    path.write_text(
        f"providers:\n  doc_status:\n    kind: sqlite\n{line}", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    loader = ConfigLoader(Path("conf/config.yaml"))
    sqlite_path = loader.load_config().providers.doc_status.sqlite_path

    assert sqlite_path == (
        expected if expected == ":memory:" else str(tmp_path.resolve() / expected)
    )


def test_api_key_is_masked() -> None:
    config = Config(
        providers={"llm": {"kind": "openai_compatible", "api_key": "synthetic-secret"}},
        models={"default": "m"},
    )
    assert "synthetic-secret" not in repr(config.providers.llm)


def test_shipped_local_config_selects_local_mode() -> None:
    raw = yaml.safe_load((_ROOT / "docker/config.local.yaml").read_text("utf-8"))
    config = Config(**raw)
    assert config.backend is Backend.LOCAL
    assert config.providers.llm.kind == "ollama"
    assert config.providers.doc_status.kind == "sqlite"
    assert config.doc_status_enabled is True
    # The compose flow keeps the containers for the stores.
    assert config.providers.graph.kind == "gremlin"
    assert config.providers.vector.kind == "opensearch"
    assert config.aws.neptune.use_iam is False


def test_shipped_local_config_sets_only_what_differs_from_the_preset() -> None:
    # Its commented-out keys show the preset's values: uncommenting them
    # changes nothing, and the file itself repeats none of the preset.
    text = (_ROOT / "docker/config.local.yaml").read_text("utf-8")
    raw = yaml.safe_load(text)
    uncommented = uncommented_template(text)
    assert Config(**uncommented) == Config(**raw)
    # Only the stores are set: the containers instead of the embedded stores,
    # and the registry (incremental indexing is off in the preset).
    assert set(raw) == {"backend", "providers", "logging"}
    assert set(raw["providers"]) == {"graph", "vector", "doc_status"}
    assert set(raw["providers"]["doc_status"]) == {"enabled", "sqlite_path"}
    shown = set(commented_key_paths(text))
    assert shown == {
        "models.default",
        "models.embedding.model_id",
        "models.embedding.dimension",
        "models.rerank",
        *(f"aws.neptune.{k}" for k in ("endpoint", "port", "use_ssl", "use_iam")),
        *(
            f"aws.opensearch.{k}"
            for k in ("endpoint", "port", "use_ssl", "verify_certs", "allow_anonymous")
        ),
    }


def test_local_preset_context_budget_fits_the_local_model_window() -> None:
    # The AWS default budget (30,000) is over what the default local model's
    # 32,768-token window accepts beside its output reservation, so every
    # local query was clamped with a WARNING.
    config = Config(backend="local")
    llm = config.providers.llm
    token_manager = config.search.token_manager
    ceiling = int(
        (llm.context_window_size - llm.max_output_tokens)
        * (1.0 - token_manager.context_window_headroom_ratio)
    )
    assert token_manager.max_context_tokens is not None
    assert token_manager.max_context_tokens <= ceiling
    # A value the file sets wins over the preset.
    explicit = Config(
        backend="local", search={"token_manager": {"max_context_tokens": 12000}}
    )
    assert explicit.search.token_manager.max_context_tokens == 12000
    assert Config().search.token_manager.max_context_tokens == 30_000


@pytest.mark.parametrize(
    "llm",
    [
        # Bedrock models are sized from the capability catalog.
        {"kind": "bedrock"},
        # An explicit window is not the one the preset budget was derived from.
        {"context_window_size": 131072},
        {"context_window_size": 32768},
    ],
)
def test_local_preset_budget_only_fits_the_default_local_window(
    llm: dict[str, object],
) -> None:
    config = Config(backend="local", providers={"llm": llm})
    assert config.search.token_manager.max_context_tokens == 30_000


@pytest.mark.parametrize(
    ("section", "default"),
    [("graph", "graph.sqlite3"), ("vector", "vector.sqlite3")],
)
def test_store_sqlite_path_follows_the_config_file_only_for_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str, default: str
) -> None:
    # Inert for the server kinds; resolved like the doc-status registry once
    # the section's kind is sqlite.
    (tmp_path / "conf").mkdir()
    path = tmp_path / "conf" / "config.yaml"
    monkeypatch.chdir(tmp_path)

    path.write_text("providers: {}\n", encoding="utf-8")
    config = ConfigLoader(Path("conf/config.yaml")).load_config()
    assert getattr(config.providers, section).sqlite_path == (
        f".unified-kg-rag/{default}"
    )

    path.write_text(f"providers:\n  {section}:\n    kind: sqlite\n", encoding="utf-8")
    config = ConfigLoader(Path("conf/config.yaml")).load_config()
    assert getattr(config.providers, section).sqlite_path == str(
        tmp_path.resolve() / "conf" / ".unified-kg-rag" / default
    )
