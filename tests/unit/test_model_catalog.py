# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model ids as plain strings, tier defaults, and capability resolution.

``models.roles`` entries accept any Bedrock model id and follow
``models.default`` / ``models.fast`` unless set. Capabilities
resolve from the curated table, then provider-family defaults, then a
conservative unknown default, with ``aws.bedrock.model_overrides`` on top.
AWS-free: factories use a stub session and patched model classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from tests.fixtures.fakes.boto import FakeBotoSession
from unified_kg_rag.adapters.aws import bedrock as bedrock_mod
from unified_kg_rag.adapters.aws import bedrock_models
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockLanguageModelFactory,
)
from unified_kg_rag.adapters.aws.bedrock_models import get_language_model_info
from unified_kg_rag.domain.models import Config, LanguageModelId, ModelRole
from unified_kg_rag.domain.models.config import (
    DEFAULT_MODEL_ID,
    FAST_MODEL_ID,
    ROLE_TIERS,
)
from unified_kg_rag.shared import LanguageModelError
from unified_kg_rag.shared import logging as shared_logging

pytestmark = pytest.mark.unit

TEMPLATE = Path(__file__).resolve().parents[2] / "config-template.yaml"


@pytest.fixture(autouse=True)
def _reset_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared_logging, "_warned_once", set())


# --- config ---------------------------------------------------------------


def test_every_role_has_a_tier_and_ships_its_default() -> None:
    config = Config()
    assert set(ROLE_TIERS) == set(ModelRole)
    tiers = list(ROLE_TIERS.values())
    assert tiers.count("default") == 8
    assert tiers.count("fast") == 11
    for role, tier in ROLE_TIERS.items():
        expected = DEFAULT_MODEL_ID if tier == "default" else FAST_MODEL_ID
        assert config.model_id(role) == expected, role


def test_tier_keys_move_every_unset_role() -> None:
    config = Config(
        models={"default": "openai.gpt-6-sol", "fast": "anthropic.claude-sonnet-4-6"}
    )
    for role, tier in ROLE_TIERS.items():
        expected = (
            "openai.gpt-6-sol" if tier == "default" else "anthropic.claude-sonnet-4-6"
        )
        assert config.model_id(role) == expected, role


def test_explicit_role_value_wins_over_its_tier() -> None:
    config = Config(
        models={
            "default": "openai.gpt-6-sol",
            "roles": {"answer_generation": "anthropic.claude-opus-5"},
        }
    )
    assert config.model_id(ModelRole.ANSWER_GENERATION) == "anthropic.claude-opus-5"
    assert config.model_id(ModelRole.ENTITY_EXTRACTION) == "openai.gpt-6-sol"


def test_resolution_writes_nothing_back() -> None:
    # A dump of what the config was built from stays what it was built from,
    # so a rebuild (the CLI provider flags) keeps following the tiers.
    config = Config(models={"default": "a.one"})
    assert config.models.roles.answer_generation is None
    assert config.model_dump(exclude_unset=True) == {"models": {"default": "a.one"}}


def test_enum_members_and_any_string_validate_as_plain_ids() -> None:
    config = Config(
        models={
            "roles": {
                "answer_generation": LanguageModelId.CLAUDE_V5_OPUS,
                "query_translation": "  amazon.nova-pro-v1:0 ",
            }
        }
    )
    answer = config.model_id(ModelRole.ANSWER_GENERATION)
    assert type(answer) is str and answer == "anthropic.claude-opus-5"
    assert config.model_id(ModelRole.QUERY_TRANSLATION) == "amazon.nova-pro-v1:0"
    with pytest.raises(ValidationError):
        Config(models={"roles": {"answer_generation": ""}})


def test_enum_member_renders_as_its_id() -> None:
    member = LanguageModelId.GPT_V6_SOL
    assert str(member) == f"{member}" == "openai.gpt-6-sol"
    record = logging.LogRecord("t", logging.INFO, "", 0, "%s", (member,), None)
    assert record.getMessage() == "openai.gpt-6-sol"


def test_template_documents_the_models_once() -> None:
    raw = yaml.safe_load(TEMPLATE.read_text())
    config = Config(models=raw["models"])
    assert config.tier_model_id("default") == DEFAULT_MODEL_ID
    assert config.tier_model_id("fast") == FAST_MODEL_ID
    # No role is pinned: every role follows its tier.
    assert raw["models"]["roles"] == {}
    text = TEMPLATE.read_text()
    assert "_model_id:" not in text


# --- capability resolution -----------------------------------------------


def test_curated_row_and_profile_prefixed_id_resolve_alike() -> None:
    curated = get_language_model_info(LanguageModelId.CLAUDE_V5_5_SONNET)
    assert get_language_model_info("us.anthropic.claude-sonnet-5-5") is curated
    assert get_language_model_info("global.anthropic.claude-sonnet-5-5") is curated


@pytest.mark.parametrize(
    ("model_id", "adaptive_only", "adaptive", "thinking", "sampling"),
    [
        ("anthropic.claude-sonnet-6", True, True, True, False),
        ("anthropic.claude-haiku-5-1", True, True, True, False),
        ("anthropic.claude-haiku-4-6", False, True, True, True),
        ("anthropic.claude-opus-4-5-20990101-v1:0", False, False, True, True),
        ("anthropic.claude-3-9-sonnet-20990101-v1:0", False, False, True, True),
        ("anthropic.claude-3-1-haiku-20990101-v1:0", False, False, False, True),
        ("anthropic.claude-instant-v1", False, False, False, True),
    ],
)
def test_uncurated_claude_ids_get_generation_defaults(
    model_id: str,
    adaptive_only: bool,
    adaptive: bool,
    thinking: bool,
    sampling: bool,
) -> None:
    info = get_language_model_info(model_id)
    assert info.provider == "anthropic"
    assert info.adaptive_thinking_only is adaptive_only
    assert info.uses_adaptive_thinking is adaptive
    assert info.supports_thinking is thinking
    assert info.supports_sampling_params is sampling


def test_date_suffix_is_not_read_as_a_minor_version() -> None:
    # 'claude-sonnet-4-20990101' is 4.0 (budget thinking), not 4.20 (adaptive).
    info = get_language_model_info("anthropic.claude-sonnet-4-20990101-v1:0")
    assert info.supports_thinking is True
    assert info.uses_adaptive_thinking is False


def test_uncurated_gpt_id_gets_openai_defaults() -> None:
    info = get_language_model_info("openai.gpt-7-sol")
    assert info.provider == "openai"
    assert info.always_reasons is True
    assert info.supports_sampling_params is False
    assert info.supports_prompt_caching is False


def test_unknown_provider_is_conservative_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=bedrock_models.logger.name):
        first = get_language_model_info("amazon.nova-pro-v1:0")
        get_language_model_info("amazon.nova-pro-v1:0")
    assert first.provider == "other"
    assert first.supports_thinking is False
    assert first.supports_sampling_params is False
    assert first.supports_prompt_caching is False
    warnings = [r for r in caplog.records if "amazon.nova-pro-v1:0" in r.message]
    assert len(warnings) == 1


def test_model_override_applies_and_silences_the_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    overrides = {"amazon.nova-pro-v1:0": {"context_window_size": 300000}}
    with caplog.at_level(logging.WARNING, logger=bedrock_models.logger.name):
        info = get_language_model_info("us.amazon.nova-pro-v1:0", overrides)
    assert info.context_window_size == 300000
    assert info.max_output_tokens == 4096
    assert not caplog.records


def test_model_override_corrects_a_curated_row() -> None:
    overrides = {LanguageModelId.CLAUDE_V4_5_HAIKU.value: {"max_output_tokens": 1000}}
    info = get_language_model_info(LanguageModelId.CLAUDE_V4_5_HAIKU, overrides)
    assert info.max_output_tokens == 1000
    assert get_language_model_info(LanguageModelId.CLAUDE_V4_5_HAIKU).max_output_tokens


def test_model_override_with_unknown_key_fails_fast() -> None:
    with pytest.raises(LanguageModelError, match="model_overrides"):
        get_language_model_info(
            "amazon.nova-pro-v1:0", {"amazon.nova-pro-v1:0": {"context_window": 1}}
        )


def test_claude_4_5_entries_require_an_inference_profile() -> None:
    for model_id in (
        LanguageModelId.CLAUDE_V4_5_HAIKU,
        LanguageModelId.CLAUDE_V4_5_SONNET,
        LanguageModelId.CLAUDE_V4_5_OPUS,
    ):
        assert get_language_model_info(model_id).requires_inference_profile, model_id


# --- factory ---------------------------------------------------------------


def _capture_model_classes(mocker: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _FakeConverse:
        def __new__(cls, **kwargs: Any) -> Any:
            captured.update(kwargs, model_class="converse")
            return "converse"

    class _FakeInvoke:
        def __new__(cls, **kwargs: Any) -> Any:
            captured.update(kwargs, model_class="invoke")
            return "invoke"

    mocker.patch.object(bedrock_mod, "ChatBedrockConverse", _FakeConverse)
    mocker.patch.object(bedrock_mod, "ChatBedrock", _FakeInvoke)
    return captured


def test_unknown_model_gets_a_plain_converse_request(mocker: Any) -> None:
    mocker.patch.object(
        BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value="amazon.nova-pro-v1:0",
    )
    captured = _capture_model_classes(mocker)
    factory = BedrockLanguageModelFactory(Config(), boto_session=FakeBotoSession())  # type: ignore[arg-type]
    factory.get_model("amazon.nova-pro-v1:0", enable_thinking=True)
    assert captured["model_class"] == "converse"
    assert captured["model_id"] == "amazon.nova-pro-v1:0"
    for key in ("temperature", "stop_sequences", "additional_model_request_fields"):
        assert key not in captured, key


def test_profile_id_is_invoked_as_is_through_converse(mocker: Any) -> None:
    captured = _capture_model_classes(mocker)
    factory = BedrockLanguageModelFactory(Config(), boto_session=FakeBotoSession())  # type: ignore[arg-type]
    factory.get_model("us.anthropic.claude-sonnet-5-5")
    assert captured["model_class"] == "converse"
    assert captured["model_id"] == "us.anthropic.claude-sonnet-5-5"


def test_cross_region_helper_keeps_a_profile_id() -> None:
    resolved = BedrockCrossRegionModelHelper.get_cross_region_model_id(
        FakeBotoSession(),  # type: ignore[arg-type]
        "eu.anthropic.claude-sonnet-5-5",
        "us-west-2",
    )
    assert resolved == "eu.anthropic.claude-sonnet-5-5"


def test_factory_applies_model_overrides(mocker: Any) -> None:
    config = Config(
        aws={
            "bedrock": {
                "model_overrides": {"amazon.nova-pro-v1:0": {"max_output_tokens": 5000}}
            }
        }
    )
    factory = BedrockLanguageModelFactory(config, boto_session=FakeBotoSession())  # type: ignore[arg-type]
    assert factory.get_model_info("amazon.nova-pro-v1:0").max_output_tokens == 5000
