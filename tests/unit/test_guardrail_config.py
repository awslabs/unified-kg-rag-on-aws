# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Bedrock Guardrails wiring (WAF security pillar, AWS-free)."""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.aws.bedrock import BedrockLanguageModelFactory
from aws_graphrag.domain.models import Config

pytestmark = pytest.mark.unit


def _factory_with(config: Config) -> BedrockLanguageModelFactory:
    # The base factory opens a boto client in __init__; bypass it entirely and
    # exercise only the pure guardrail-config logic.
    f = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
    f.config = config
    return f


def test_disabled_by_default() -> None:
    f = _factory_with(Config())
    config: dict = {}
    f._apply_guardrail(config, is_cross_region=True)
    f._apply_guardrail(config, is_cross_region=False)
    assert "guardrail_config" not in config
    assert "guardrails" not in config


def test_cross_region_uses_converse_shape() -> None:
    cfg = Config()
    cfg.aws.bedrock.guardrail.identifier = "gr-123"
    cfg.aws.bedrock.guardrail.version = "3"
    cfg.aws.bedrock.guardrail.trace = True
    f = _factory_with(cfg)
    config: dict = {}
    f._apply_guardrail(config, is_cross_region=True)
    assert config["guardrail_config"] == {
        "guardrailIdentifier": "gr-123",
        "guardrailVersion": "3",
        "trace": "enabled",
    }
    assert "guardrails" not in config


def test_non_cross_region_uses_invoke_shape() -> None:
    cfg = Config()
    cfg.aws.bedrock.guardrail.identifier = "gr-123"
    f = _factory_with(cfg)
    config: dict = {}
    f._apply_guardrail(config, is_cross_region=False)
    assert config["guardrails"]["guardrailIdentifier"] == "gr-123"
    # InvokeModel treats trace as a truthiness flag, so it must be a real bool
    # (a string like "disabled" would wrongly enable tracing).
    assert config["guardrails"]["trace"] is False
    assert "guardrail_config" not in config


def test_invoke_model_trace_enabled_is_bool_true() -> None:
    cfg = Config()
    cfg.aws.bedrock.guardrail.identifier = "gr-123"
    cfg.aws.bedrock.guardrail.trace = True
    f = _factory_with(cfg)
    config: dict = {}
    f._apply_guardrail(config, is_cross_region=False)
    assert config["guardrails"]["trace"] is True


def test_enabled_property() -> None:
    cfg = Config()
    assert cfg.aws.bedrock.guardrail.enabled is False
    cfg.aws.bedrock.guardrail.identifier = "gr-x"
    assert cfg.aws.bedrock.guardrail.enabled is True


def test_guardrail_identifier_from_env(monkeypatch) -> None:
    # IaC injects the deployed guardrail id via this env var (4-level nested
    # config path); verify the override lands and enables guardrails.
    monkeypatch.setenv("BEDROCK_GUARDRAIL_IDENTIFIER", "gr-from-env")
    from aws_graphrag.shared.config import ConfigLoader

    cfg = ConfigLoader().load_config()
    assert cfg.aws.bedrock.guardrail.identifier == "gr-from-env"
    assert cfg.aws.bedrock.guardrail.enabled is True
