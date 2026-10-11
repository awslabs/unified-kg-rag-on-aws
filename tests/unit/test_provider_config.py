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

from unified_kg_rag.domain.models import (
    Backend,
    Config,
    EmbeddingModelId,
    RerankModelId,
    model_id_value,
)
from unified_kg_rag.domain.models.provider_config import LOCAL_PRESET
from unified_kg_rag.shared.config import ConfigLoader

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]

_LOCAL_EMBEDDING = {
    "kind": "ollama",
    "model_id": "synthetic-embed",
    "dimension": 384,
}


@pytest.mark.parametrize(
    "path",
    ["config-template.yaml", "docker/config.yaml", "docker/config.local.yaml"],
)
def test_shipped_yaml_without_providers_keeps_the_aws_providers(path: str) -> None:
    raw = yaml.safe_load((_ROOT / path).read_text(encoding="utf-8")) or {}
    config = Config(**raw)
    if "backend" in raw or "providers" in raw:
        pytest.skip(f"{path} selects providers explicitly")
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
    assert isinstance(config.indexing.opensearch.embedding_model_id, EmbeddingModelId)
    assert isinstance(config.search.reranking.rerank_model_id, RerankModelId)


def test_default_config_is_unchanged_by_the_new_sections() -> None:
    config = Config()
    assert config.indexing.opensearch.embedding_model_id is (
        EmbeddingModelId.TITAN_EMBED_V2
    )
    assert config.evaluation.embedding_model_id is EmbeddingModelId.TITAN_EMBED_V2
    assert config.search.reranking.rerank_model_id is RerankModelId.COHERE_RERANK_V3_5
    assert (
        config.search.answer_generation_model_id == config.aws.bedrock.default_model_id
    )
    assert config.doc_status_enabled is False


@pytest.mark.parametrize(
    "value",
    [EmbeddingModelId.EMBED_V4, "cohere.embed-v4:0", " cohere.embed-v4:0 "],
)
def test_bedrock_embedding_ids_accept_enum_members_and_their_values(
    value: object,
) -> None:
    config = Config(indexing={"opensearch": {"embedding_model_id": value}})
    assert config.indexing.opensearch.embedding_model_id is EmbeddingModelId.EMBED_V4


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (
            {"indexing": {"opensearch": {"embedding_model_id": "synthetic-embed"}}},
            "indexing.opensearch.embedding_model_id",
        ),
        (
            {"evaluation": {"embedding_model_id": "synthetic-embed"}},
            "evaluation.embedding_model_id",
        ),
        (
            {"search": {"reranking": {"rerank_model_id": "synthetic-rerank"}}},
            "search.reranking.rerank_model_id",
        ),
    ],
)
def test_bedrock_providers_still_reject_ids_outside_the_catalog(
    section: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Config(**section)


def test_non_bedrock_embedding_takes_any_name_and_defaults_every_field() -> None:
    config = Config(providers={"embedding": _LOCAL_EMBEDDING})
    assert config.indexing.opensearch.embedding_model_id == "synthetic-embed"
    assert config.evaluation.embedding_model_id == "synthetic-embed"
    assert model_id_value(config.indexing.opensearch.embedding_model_id) == (
        "synthetic-embed"
    )


def test_an_explicit_embedding_id_wins_over_the_provider_default() -> None:
    config = Config(
        providers={"embedding": _LOCAL_EMBEDDING},
        evaluation={"embedding_model_id": "other-embed"},
    )
    assert config.indexing.opensearch.embedding_model_id == "synthetic-embed"
    assert config.evaluation.embedding_model_id == "other-embed"


def test_non_bedrock_embedding_requires_a_model_and_dimension() -> None:
    with pytest.raises(ValidationError, match="providers.embedding.model_id"):
        Config(providers={"embedding": {"kind": "ollama", "model_id": "x"}})


def test_non_bedrock_llm_tiers_follow_the_provider_models() -> None:
    config = Config(
        providers={
            "llm": {
                "kind": "openai_compatible",
                "default_model_id": "synthetic-large",
                "fast_model_id": "synthetic-small",
            }
        },
        search={"answer_generation_model_id": "pinned-model"},
    )
    assert config.processing.graph_extraction.extraction_model_id == "synthetic-large"
    assert config.processing.chunking.chunking_model_id == "synthetic-small"
    assert config.search.answer_generation_model_id == "pinned-model"


def test_the_fast_tier_defaults_to_the_default_model() -> None:
    config = Config(providers={"llm": {"kind": "ollama", "default_model_id": "m"}})
    assert config.processing.chunking.chunking_model_id == "m"
    assert config.search.global_search.map_model_id == "m"


def test_non_bedrock_llm_requires_a_default_model() -> None:
    with pytest.raises(ValidationError, match="providers.llm.default_model_id"):
        Config(providers={"llm": {"kind": "ollama"}})


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
    assert config.providers.llm.kind == preset["llm"]["kind"]
    assert config.providers.embedding.dimension == preset["embedding"]["dimension"]
    assert config.providers.rerank.kind == "cross_encoder"
    assert config.providers.graph.kind == "gremlin"
    assert config.providers.doc_status.kind == "sqlite"
    assert config.search.answer_generation_model_id == preset["llm"]["default_model_id"]
    assert (
        config.indexing.opensearch.embedding_model_id == preset["embedding"]["model_id"]
    )
    assert config.search.reranking.rerank_model_id == preset["rerank"]["model_id"]
    assert (config.aws.neptune.use_iam, config.aws.neptune.use_ssl) == (False, False)
    assert config.aws.opensearch.allow_anonymous is True
    assert config.aws.opensearch.port == 9200


def test_local_preset_yields_to_values_the_config_sets() -> None:
    config = Config(
        backend="local",
        providers={
            "llm": {"default_model_id": "my-model", "base_url": "http://gpu:11434"},
            "rerank": {"kind": "none"},
        },
        aws={"neptune": {"endpoint": "graph.internal.example", "port": 18182}},
    )
    assert config.providers.llm.kind == "ollama"
    assert config.providers.llm.default_model_id == "my-model"
    assert config.providers.llm.base_url == "http://gpu:11434"
    assert config.providers.rerank.kind == "none"
    assert config.aws.neptune.endpoint == "graph.internal.example"
    assert config.aws.neptune.port == 18182
    assert config.aws.neptune.use_ssl is False


def test_local_preset_skips_anonymous_access_for_authenticated_opensearch() -> None:
    config = Config(backend="local", aws={"opensearch": {"username": "admin"}})
    assert config.aws.opensearch.allow_anonymous is False
    assert config.aws.opensearch.username == "admin"


def test_preset_does_not_mutate_its_template() -> None:
    before = yaml.safe_dump(LOCAL_PRESET)
    Config(backend="local", providers={"llm": {"default_model_id": "x"}})
    assert yaml.safe_dump(LOCAL_PRESET) == before


def test_doc_status_switch_follows_dynamodb_unless_set() -> None:
    config = Config()
    config.aws.dynamodb.enabled = True
    assert config.doc_status_enabled is True
    assert Config(providers={"doc_status": {"enabled": True}}).doc_status_enabled
    pinned_off = Config(
        aws={"dynamodb": {"enabled": True}},
        providers={"doc_status": {"enabled": False}},
    )
    assert pinned_off.doc_status_enabled is False


def test_local_backend_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "backend: local\n"
        "providers:\n"
        "  llm:\n"
        "    default_model_id: synthetic-llm\n",
        encoding="utf-8",
    )
    config = ConfigLoader(path).load_config()
    assert config.backend is Backend.LOCAL
    assert config.search.answer_generation_model_id == "synthetic-llm"


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
        providers={
            "llm": {
                "kind": "openai_compatible",
                "default_model_id": "m",
                "api_key": "synthetic-secret",
            }
        }
    )
    assert "synthetic-secret" not in repr(config.providers.llm)


def test_shipped_local_config_selects_local_mode() -> None:
    raw = yaml.safe_load((_ROOT / "docker/config.local.yaml").read_text("utf-8"))
    config = Config(**raw)
    assert config.backend is Backend.LOCAL
    assert config.providers.llm.kind == "ollama"
    assert config.providers.doc_status.kind == "sqlite"
    assert config.doc_status_enabled is True
    assert config.aws.neptune.use_iam is False
