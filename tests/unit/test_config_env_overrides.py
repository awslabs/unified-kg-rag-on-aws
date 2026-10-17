# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment overrides injected by the CDK compute stack (AWS-free)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unified_kg_rag.domain.models.config import ObservabilityBackend
from unified_kg_rag.shared.config import ConfigLoader
from unified_kg_rag.shared.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "UKR__AWS__DYNAMODB__TABLE_NAME",
        "UKR__AWS__DYNAMODB__CREATE_TABLE_IF_MISSING",
        "UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE",
        "UKR__OBSERVABILITY__ENABLED",
        "UKR__OBSERVABILITY__BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)


def test_source_scope_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The container entrypoint exports the S3 URI the corpus was synced from.
    monkeypatch.setenv(
        "UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE", "s3://example-bucket/corpus/"
    )
    cfg = ConfigLoader().load_config()
    assert cfg.processing.document_parsing.source_scope == "s3://example-bucket/corpus/"


def test_doc_status_table_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UKR__AWS__DYNAMODB__TABLE_NAME", "example-doc-status")
    cfg = ConfigLoader().load_config()
    assert cfg.aws.dynamodb.table_name == "example-doc-status"


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("true", True)])
def test_doc_status_create_table_from_env(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("UKR__AWS__DYNAMODB__CREATE_TABLE_IF_MISSING", raw)
    cfg = ConfigLoader().load_config()
    assert cfg.aws.dynamodb.create_table_if_missing is expected


def test_env_overrides_yaml_table_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The IaC-injected name must win over a stale name left in config.yaml, and
    # must not toggle incremental indexing on by itself.
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"aws": {"dynamodb": {"table_name": "stale-table"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("UKR__AWS__DYNAMODB__TABLE_NAME", "example-doc-status")
    monkeypatch.setenv("UKR__AWS__DYNAMODB__CREATE_TABLE_IF_MISSING", "false")
    cfg = ConfigLoader(config_file).load_config()
    assert cfg.aws.dynamodb.table_name == "example-doc-status"
    assert cfg.aws.dynamodb.create_table_if_missing is False
    assert cfg.providers.doc_status.enabled is False


def test_opensearch_replicas_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The compute stack sets one replica on a multi-node domain.
    monkeypatch.setenv(
        "UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS", "1"
    )
    settings = ConfigLoader().load_config().indexing.opensearch.index_settings
    assert settings["number_of_replicas"] == 1
    assert settings["number_of_shards"] == 1


def test_korean_analyzer_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The compute stack maps Korean to an analyzer Amazon OpenSearch Service
    # has without the optional Nori plugin; the other languages are kept.
    monkeypatch.setenv("UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO", "cjk")
    analyzers = ConfigLoader().load_config().indexing.opensearch.language_analyzers
    assert analyzers == {"en": "english", "ko": "cjk"}


def test_korean_analyzer_default_without_env() -> None:
    analyzers = ConfigLoader().load_config().indexing.opensearch.language_analyzers
    assert analyzers["ko"] == "nori"


def test_guardrail_version_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UKR__AWS__BEDROCK__GUARDRAIL__VERSION", "3")
    cfg = ConfigLoader().load_config()
    assert cfg.aws.bedrock.guardrail.version == "3"


def test_observability_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The compute stack turns tracing on for the ADOT collector sidecar; the
    # backend must be the enum member (its .value is logged), not a string.
    monkeypatch.setenv("UKR__OBSERVABILITY__ENABLED", "true")
    monkeypatch.setenv("UKR__OBSERVABILITY__BACKEND", "otlp")
    observability = ConfigLoader().load_config().observability
    assert observability.enabled is True
    assert observability.backend is ObservabilityBackend.OTLP
    assert observability.capture_content is False


def test_invalid_observability_backend_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UKR__OBSERVABILITY__BACKEND", "cloudwatch")
    with pytest.raises(ConfigurationError, match="UKR__OBSERVABILITY__BACKEND"):
        ConfigLoader().load_config()


def test_unset_env_keeps_defaults() -> None:
    cfg = ConfigLoader().load_config()
    assert cfg.aws.dynamodb.table_name == "unified-kg-rag-on-aws-doc-status-v2"
    assert cfg.aws.dynamodb.create_table_if_missing is True


def test_docker_image_config_takes_endpoints_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # docker/config.yaml is baked into the image and stays endpoint-free: every
    # deployment-specific value comes from the compute stack's env vars.
    image_config = Path(__file__).resolve().parents[2] / "docker" / "config.yaml"
    env = {
        "AWS_REGION": "us-east-1",
        "UKR__AWS__BEDROCK__REGION_NAME": "us-east-1",
        "UKR__AWS__NEPTUNE__ENDPOINT": "neptune.example.internal",
        "UKR__AWS__OPENSEARCH__ENDPOINT": "opensearch.example.internal",
        "UKR__AWS__S3__BUCKET_NAME": "example-cache-bucket",
        "UKR__AWS__DYNAMODB__TABLE_NAME": "example-doc-status",
        "UKR__AWS__DYNAMODB__CREATE_TABLE_IF_MISSING": "false",
    }
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    cfg = ConfigLoader(image_config).load_config()
    assert cfg.aws.region_name == "us-east-1"
    assert cfg.aws.bedrock.region_name == "us-east-1"
    assert cfg.aws.neptune.endpoint == "neptune.example.internal"
    assert cfg.aws.opensearch.endpoint == "opensearch.example.internal"
    assert cfg.aws.opensearch.use_iam is True
    assert cfg.aws.s3.bucket_name == "example-cache-bucket"
    assert cfg.aws.dynamodb.table_name == "example-doc-status"
    assert cfg.aws.dynamodb.create_table_if_missing is False
    assert cfg.graph.visualization.enabled is False


def test_bedrock_region_inherits_aws_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UKR__AWS__BEDROCK__REGION_NAME", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    path = tmp_path / "config.yaml"
    path.write_text("aws:\n  region_name: us-east-1\n", encoding="utf-8")
    cfg = ConfigLoader(path).load_config()
    # The file's region wins over AWS_REGION, and Bedrock follows it.
    assert cfg.aws.bedrock.region_name is None
    assert cfg.aws.bedrock_region == "us-east-1"
    assert ConfigLoader(None).load_config().aws.bedrock_region == "eu-central-1"


def test_explicit_bedrock_region_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UKR__AWS__BEDROCK__REGION_NAME", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    path = tmp_path / "config.yaml"
    path.write_text(
        "aws:\n  region_name: us-east-1\n  bedrock:\n    region_name: us-west-2\n",
        encoding="utf-8",
    )
    assert ConfigLoader(path).load_config().aws.bedrock_region == "us-west-2"
