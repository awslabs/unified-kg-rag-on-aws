# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for Bedrock Guardrails wiring (WAF security pillar, AWS-free)."""

from __future__ import annotations

import pytest

from aws_graphrag.aws.bedrock import BedrockLanguageModelFactory
from aws_graphrag.models import Config

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
    assert config["guardrails"]["trace"] == "disabled"
    assert "guardrail_config" not in config


def test_enabled_property() -> None:
    cfg = Config()
    assert cfg.aws.bedrock.guardrail.enabled is False
    cfg.aws.bedrock.guardrail.identifier = "gr-x"
    assert cfg.aws.bedrock.guardrail.enabled is True
