# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment overrides injected by the CDK compute stack (AWS-free)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unified_kg_rag.shared.config import ConfigLoader

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GRAPHRAG_DOC_STATUS_TABLE", "GRAPHRAG_DOC_STATUS_CREATE_TABLE"):
        monkeypatch.delenv(var, raising=False)


def test_doc_status_table_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHRAG_DOC_STATUS_TABLE", "example-doc-status")
    cfg = ConfigLoader().load_config()
    assert cfg.aws.dynamodb.table_name == "example-doc-status"


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("true", True)])
def test_doc_status_create_table_from_env(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("GRAPHRAG_DOC_STATUS_CREATE_TABLE", raw)
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
    monkeypatch.setenv("GRAPHRAG_DOC_STATUS_TABLE", "example-doc-status")
    monkeypatch.setenv("GRAPHRAG_DOC_STATUS_CREATE_TABLE", "false")
    cfg = ConfigLoader(config_file).load_config()
    assert cfg.aws.dynamodb.table_name == "example-doc-status"
    assert cfg.aws.dynamodb.create_table_if_missing is False
    assert cfg.aws.dynamodb.enabled is False


def test_unset_env_keeps_defaults() -> None:
    cfg = ConfigLoader().load_config()
    assert cfg.aws.dynamodb.table_name == "unified-kg-rag-on-aws-doc-status"
    assert cfg.aws.dynamodb.create_table_if_missing is True
