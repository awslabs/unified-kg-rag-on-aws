# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Bedrock capability resolution (AWS-free).

Covers the model-info lookup tables, dimension validation, max-token clamping,
thinking/perf-optimization predicates, guardrail-config assembly, cross-region
model-id construction, and the assumed-role session helper. No boto client is
ever invoked against AWS: factories are built with a fake boto session whose
``client()`` returns a stub, and the assumed-role helper is exercised with the
no-op (``assumed_role_arn=None``) and patched-``assume_role`` paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.adapters.aws import bedrock as bedrock_mod
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    LanguageModelInfo,
    get_assumed_role_boto_session,
)
from unified_kg_rag.domain.models import (
    Config,
    EmbeddingModelId,
    LanguageModelId,
)
from unified_kg_rag.shared import EmbeddingModelError, LanguageModelError

pytestmark = pytest.mark.unit


# --- fake boto session ----------------------------------------------------


class _FakeSession:
    profile_name = "default"

    def __init__(self) -> None:
        self.clients_requested: list[str] = []

    def client(self, service_name: str, **kwargs: Any) -> Any:
        self.clients_requested.append(service_name)
        return object()  # opaque stub; capability logic never calls it

    def get_credentials(self) -> Any:
        return None  # no assume-role short-circuit path


def _lang_factory(config: Config | None = None) -> BedrockLanguageModelFactory:
    cfg = config or Config()
    return BedrockLanguageModelFactory(cfg, boto_session=_FakeSession())


def _embed_factory(config: Config | None = None) -> BedrockEmbeddingModelFactory:
    cfg = config or Config()
    return BedrockEmbeddingModelFactory(cfg, boto_session=_FakeSession())


# --- get_model_info / model-info tables ----------------------------------


def test_language_get_model_info_known_model() -> None:
    factory = _lang_factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V4_SONNET)
    assert info is not None
    assert info.context_window_size == 200000
    assert info.supports_thinking is True
    assert info.supports_1m_context_window is True
    assert info.supports_prompt_caching is True


def test_language_haiku_v3_no_thinking() -> None:
    factory = _lang_factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V3_HAIKU)
    assert info is not None
    assert info.supports_thinking is False
    assert info.supports_prompt_caching is True
    assert info.max_output_tokens == 4096


def test_embedding_get_model_info_dimensions() -> None:
    factory = _embed_factory()
    titan_v1 = factory.get_model_info(EmbeddingModelId.TITAN_EMBED_V1)
    titan_v2 = factory.get_model_info(EmbeddingModelId.TITAN_EMBED_V2)
    assert titan_v1 is not None and titan_v1.dimensions == 1536
    assert titan_v2 is not None and titan_v2.dimensions == [256, 512, 1024]


def test_every_language_model_enum_has_info() -> None:
    # Regression: CLAUDE_V3_SONNET / CLAUDE_V3_OPUS were selectable enum members
    # with no _LANGUAGE_MODEL_INFO entry, so get_model_info returned None and
    # get_model raised. Every advertised model id must resolve to capabilities.
    factory = _lang_factory()
    for model_id in LanguageModelId:
        assert factory.get_model_info(model_id) is not None, model_id


def test_get_model_info_resolves_for_every_known_embedding_model() -> None:
    factory = _embed_factory()
    for model_id in EmbeddingModelId:
        assert factory.get_model_info(model_id) is not None


# --- embedding dimension resolution --------------------------------------


def test_embedding_get_model_unsupported_dimension_raises(mocker) -> None:
    factory = _embed_factory()
    # Patch token counter + wrapper so we only exercise dimension validation.
    mocker.patch.object(bedrock_mod, "BedrockTokenCounter", return_value=object())
    with pytest.raises(EmbeddingModelError, match="Dimension 999 is not supported"):
        factory.get_model(EmbeddingModelId.TITAN_EMBED_V2, dimensions=999)


def test_embedding_get_model_single_value_dimension_mismatch_raises(mocker) -> None:
    factory = _embed_factory()
    mocker.patch.object(bedrock_mod, "BedrockTokenCounter", return_value=object())
    # Titan V1 supports a single int (1536); requesting 256 must fail.
    with pytest.raises(EmbeddingModelError, match="not supported"):
        factory.get_model(EmbeddingModelId.TITAN_EMBED_V1, dimensions=256)


# --- _validate_max_tokens -------------------------------------------------


def test_validate_max_tokens_clamps_to_model_max() -> None:
    info = LanguageModelInfo(context_window_size=200000, max_output_tokens=8192)
    assert BedrockLanguageModelFactory._validate_max_tokens(100000, info) == 8192


def test_validate_max_tokens_uses_default_when_none() -> None:
    info = LanguageModelInfo(context_window_size=200000, max_output_tokens=8192)
    assert BedrockLanguageModelFactory._validate_max_tokens(None, info) == 8192


def test_validate_max_tokens_keeps_in_range_value() -> None:
    info = LanguageModelInfo(context_window_size=200000, max_output_tokens=8192)
    assert BedrockLanguageModelFactory._validate_max_tokens(2000, info) == 2000


# --- thinking / performance predicates ------------------------------------


def test_should_enable_thinking() -> None:
    thinks = LanguageModelInfo(
        context_window_size=1, max_output_tokens=1, supports_thinking=True
    )
    no_think = LanguageModelInfo(context_window_size=1, max_output_tokens=1)
    assert BedrockLanguageModelFactory._should_enable_thinking(True, thinks) is True
    assert BedrockLanguageModelFactory._should_enable_thinking(False, thinks) is False
    assert BedrockLanguageModelFactory._should_enable_thinking(True, no_think) is False


def test_should_enable_thinking_for_adaptive_only_model() -> None:
    # Adaptive-thinking models think by default (Sonnet 5 / Fable 5 reject
    # {'type': 'disabled'} outright), and the adaptive block is what carries
    # the effort level — so it must be emitted even without an opt-in.
    adaptive = LanguageModelInfo(
        context_window_size=1,
        max_output_tokens=1,
        supports_thinking=True,
        adaptive_thinking_only=True,
    )
    assert BedrockLanguageModelFactory._should_enable_thinking(False, adaptive) is True


# --- Claude 5 request shaping ---------------------------------------------


def test_claude_v5_models_declare_adaptive_only_and_no_sampling() -> None:
    factory = _lang_factory()
    for model_id in (
        LanguageModelId.CLAUDE_V5_SONNET,
        LanguageModelId.CLAUDE_V5_OPUS,
    ):
        info = factory.get_model_info(model_id)
        assert info is not None, model_id
        assert info.adaptive_thinking_only is True, model_id
        assert info.supports_sampling_params is False, model_id
        assert info.native_1m_context_window is True, model_id
        assert info.context_window_size == 1000000, model_id
        assert info.max_output_tokens == 128000, model_id


def test_thinking_config_adaptive_puts_effort_outside_thinking() -> None:
    # Nesting 'effort' inside 'thinking' is a ValidationException on Bedrock;
    # it belongs in a sibling 'output_config' object.
    factory = _lang_factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V5_SONNET)
    assert info is not None
    out = factory._build_thinking_config(info)
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in out["thinking"]
    assert "effort" not in out["thinking"]


def test_thinking_config_legacy_model_keeps_budget_tokens() -> None:
    factory = _lang_factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V4_5_SONNET)
    assert info is not None
    out = factory._build_thinking_config(info, thinking_budget_tokens=4096)
    assert out == {"thinking": {"type": "enabled", "budget_tokens": 4096}}
    assert "output_config" not in out


def test_thinking_config_effort_from_config_and_override() -> None:
    config = Config()
    config.aws.bedrock.effort = "low"
    factory = _lang_factory(config)
    info = factory.get_model_info(LanguageModelId.CLAUDE_V5_OPUS)
    assert info is not None
    assert factory._build_thinking_config(info)["output_config"] == {"effort": "low"}
    # A per-call override wins over the config default.
    assert factory._build_thinking_config(info, effort="max")["output_config"] == {
        "effort": "max"
    }


def test_thinking_config_rejects_invalid_effort() -> None:
    factory = _lang_factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V5_SONNET)
    assert info is not None
    with pytest.raises(LanguageModelError, match="Invalid effort level"):
        factory._build_thinking_config(info, effort="ludicrous")


def test_base_config_omits_top_k_for_claude_v5() -> None:
    factory = _lang_factory()
    v5 = factory.get_model_info(LanguageModelId.CLAUDE_V5_SONNET)
    legacy = factory.get_model_info(LanguageModelId.CLAUDE_V4_5_SONNET)
    assert v5 is not None and legacy is not None
    v5_cfg = factory._build_base_config("m", False, v5)
    assert "top_k" not in v5_cfg["model_kwargs"]
    legacy_cfg = factory._build_base_config("m", False, legacy)
    assert legacy_cfg["model_kwargs"]["top_k"] == factory.DEFAULT_TOP_K


def test_model_config_omits_temperature_for_claude_v5() -> None:
    factory = _lang_factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V5_SONNET)
    assert info is not None
    cfg = factory._build_model_config(info, "apac.anthropic.claude-sonnet-5", True)
    assert "temperature" not in cfg
    assert "top_k" not in cfg
    # Adaptive thinking is emitted even without enable_thinking (always-on).
    fields = cfg["additional_model_request_fields"]
    assert fields["thinking"] == {"type": "adaptive"}
    assert fields["output_config"] == {"effort": "high"}


def _config_with_1m(enabled: bool) -> Config:
    config = Config()
    config.aws.bedrock.enable_1m_context = enabled
    return config


def test_model_config_skips_1m_beta_header_for_claude_v5() -> None:
    # 1M is the default window on Claude 5; the beta opt-in header older
    # models need must not be sent even when the opt-in is on.
    factory = _lang_factory(_config_with_1m(True))
    info = factory.get_model_info(LanguageModelId.CLAUDE_V5_SONNET)
    assert info is not None
    cfg = factory._build_model_config(info, "apac.anthropic.claude-sonnet-5", True)
    assert "anthropic_beta" not in cfg.get("additional_model_request_fields", {})


def test_model_config_1m_beta_header_follows_config_flag() -> None:
    # The 1M opt-in is a config flag: previously it was a kwarg no caller ever
    # passed, so the header could never be sent at all.
    legacy_id = "apac.anthropic.claude-sonnet-4-5-20250929-v1:0"
    on = _lang_factory(_config_with_1m(True))
    info = on.get_model_info(LanguageModelId.CLAUDE_V4_5_SONNET)
    assert info is not None
    cfg = on._build_model_config(info, legacy_id, True)
    assert cfg["additional_model_request_fields"]["anthropic_beta"] == [
        "context-1m-2025-08-07"
    ]

    off = _lang_factory(_config_with_1m(False))
    cfg_off = off._build_model_config(info, legacy_id, True)
    assert "anthropic_beta" not in cfg_off.get("additional_model_request_fields", {})


def test_effective_context_window_tracks_1m_opt_in() -> None:
    factory = _lang_factory()
    beta = factory.get_model_info(LanguageModelId.CLAUDE_V4_5_SONNET)
    native = factory.get_model_info(LanguageModelId.CLAUDE_V5_SONNET)
    plain = factory.get_model_info(LanguageModelId.CLAUDE_V4_5_HAIKU)
    assert beta is not None and native is not None and plain is not None
    # Beta-gated: baseline until the opt-in is on.
    assert beta.effective_context_window(False) == 200000
    assert beta.effective_context_window(True) == 1000000
    # Native 1M is already the declared window.
    assert native.effective_context_window(False) == 1000000
    # No 1M support: the flag must not inflate the window.
    assert plain.effective_context_window(True) == 200000


def test_model_config_keeps_temperature_for_legacy_model() -> None:
    factory = _lang_factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V4_5_SONNET)
    assert info is not None
    cfg = factory._build_model_config(
        info, "apac.anthropic.claude-sonnet-4-5-20250929-v1:0", True
    )
    assert cfg["temperature"] == factory.DEFAULT_TEMPERATURE


def test_build_cross_region_model_id_handles_suffixless_v5_id() -> None:
    # Claude 5 ids carry no date/revision suffix; prefixing must still work.
    out = BedrockCrossRegionModelHelper._build_cross_region_model_id(
        LanguageModelId.CLAUDE_V5_SONNET, "ap-northeast-2"
    )
    assert out == "apac.anthropic.claude-sonnet-5"


def test_get_model_raises_when_profile_only_model_has_no_profile(mocker) -> None:
    # Claude 5 is INFERENCE_PROFILE-only: the bare id is not invocable. When
    # resolution falls back to it, fail with the remedy rather than letting an
    # opaque Bedrock error surface on the first call.
    factory = _lang_factory()
    mocker.patch.object(
        bedrock_mod.BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value=LanguageModelId.CLAUDE_V5_SONNET.value,
    )
    with pytest.raises(LanguageModelError, match="cross-region inference profile"):
        factory.get_model(LanguageModelId.CLAUDE_V5_SONNET)


def test_get_model_allows_on_demand_model_without_profile(mocker) -> None:
    # Legacy ON_DEMAND models must still work when no profile resolves.
    factory = _lang_factory()
    mocker.patch.object(
        bedrock_mod.BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value=LanguageModelId.CLAUDE_V3_HAIKU.value,
    )
    sentinel = object()

    class _FakeChatBedrock:  # needs __name__ for the debug log
        def __new__(cls, **kwargs: Any) -> Any:
            return sentinel

    mocker.patch.object(bedrock_mod, "ChatBedrock", _FakeChatBedrock)
    assert factory.get_model(LanguageModelId.CLAUDE_V3_HAIKU) is sentinel


def test_should_enable_performance_optimization() -> None:
    perf = LanguageModelInfo(
        context_window_size=1,
        max_output_tokens=1,
        supports_performance_optimization=True,
    )
    f = BedrockLanguageModelFactory._should_enable_performance_optimization
    assert f(True, perf, is_cross_region=False) is True
    # Cross-region disables perf optimization.
    assert f(True, perf, is_cross_region=True) is False
    # Model without support.
    no_perf = LanguageModelInfo(context_window_size=1, max_output_tokens=1)
    assert f(True, no_perf, is_cross_region=False) is False


# --- _apply_guardrail -----------------------------------------------------


def test_apply_guardrail_noop_when_disabled() -> None:
    config = Config()
    # No identifier -> GuardrailConfig.enabled is False.
    factory = _lang_factory(config)
    cfg: dict[str, Any] = {}
    factory._apply_guardrail(cfg, is_cross_region=True)
    assert cfg == {}


def test_apply_guardrail_converse_shape_when_cross_region() -> None:
    config = Config()
    gr = config.aws.bedrock.guardrail
    gr.identifier = "gid-1"  # setting identifier flips .enabled to True
    gr.version = "DRAFT"
    gr.trace = True
    factory = _lang_factory(config)
    cfg: dict[str, Any] = {}
    factory._apply_guardrail(cfg, is_cross_region=True)
    assert cfg["guardrail_config"]["guardrailIdentifier"] == "gid-1"
    assert cfg["guardrail_config"]["guardrailVersion"] == "DRAFT"
    assert cfg["guardrail_config"]["trace"] == "enabled"


def test_apply_guardrail_invoke_shape_uses_bool_trace() -> None:
    config = Config()
    gr = config.aws.bedrock.guardrail
    gr.identifier = "gid-2"
    gr.version = "1"
    gr.trace = False
    factory = _lang_factory(config)
    cfg: dict[str, Any] = {}
    factory._apply_guardrail(cfg, is_cross_region=False)
    # InvokeModel shape: trace stays a bool, not the literal "disabled".
    assert cfg["guardrails"]["trace"] is False
    assert cfg["guardrails"]["guardrailIdentifier"] == "gid-2"


# --- cross-region model id construction -----------------------------------


def test_build_cross_region_model_id_global() -> None:
    out = BedrockCrossRegionModelHelper._build_cross_region_model_id(
        LanguageModelId.CLAUDE_V4_SONNET, "us-east-1", is_global=True
    )
    assert out == f"global.{LanguageModelId.CLAUDE_V4_SONNET.value}"


def test_build_cross_region_model_id_apac_prefix() -> None:
    out = BedrockCrossRegionModelHelper._build_cross_region_model_id(
        LanguageModelId.CLAUDE_V4_SONNET, "ap-northeast-2"
    )
    assert out == f"apac.{LanguageModelId.CLAUDE_V4_SONNET.value}"


def test_build_cross_region_model_id_us_prefix() -> None:
    out = BedrockCrossRegionModelHelper._build_cross_region_model_id(
        LanguageModelId.CLAUDE_V4_SONNET, "us-east-1"
    )
    assert out == f"us.{LanguageModelId.CLAUDE_V4_SONNET.value}"


def test_is_cross_region_model_available_true() -> None:
    BedrockCrossRegionModelHelper._profiles_by_region.clear()

    class _Client:
        def list_inference_profiles(self, **kwargs: Any) -> dict:
            return {"inferenceProfileSummaries": [{"inferenceProfileId": "us.model-x"}]}

    assert (
        BedrockCrossRegionModelHelper._is_cross_region_model_available(
            _Client(), "us.model-x", "us-east-1"
        )
        is True
    )


def test_is_cross_region_model_available_false() -> None:
    BedrockCrossRegionModelHelper._profiles_by_region.clear()

    class _Client:
        def list_inference_profiles(self, **kwargs: Any) -> dict:
            return {"inferenceProfileSummaries": []}

    assert (
        BedrockCrossRegionModelHelper._is_cross_region_model_available(
            _Client(), "us.model-x", "us-east-1"
        )
        is False
    )


def test_inference_profiles_fetched_once_per_region() -> None:
    # The list_inference_profiles call must be cached per region so resolving
    # many models does not re-issue it.
    BedrockCrossRegionModelHelper._profiles_by_region.clear()
    calls = {"n": 0}

    class _Client:
        def list_inference_profiles(self, **kwargs: Any) -> dict:
            calls["n"] += 1
            return {"inferenceProfileSummaries": [{"inferenceProfileId": "us.a"}]}

    c = _Client()
    for _ in range(5):
        BedrockCrossRegionModelHelper._is_cross_region_model_available(
            c, "us.a", "us-east-1"
        )
    assert calls["n"] == 1  # fetched once, then cached


def test_get_cross_region_model_id_falls_back_on_error(mocker) -> None:
    # If the bedrock client blows up, the helper logs and returns the plain id.
    session = _FakeSession()
    mocker.patch.object(
        bedrock_mod,
        "get_assumed_role_boto_session",
        side_effect=RuntimeError("no sts"),
    )
    out = BedrockCrossRegionModelHelper.get_cross_region_model_id(
        session, LanguageModelId.CLAUDE_V4_SONNET, "us-east-1"
    )
    assert out == LanguageModelId.CLAUDE_V4_SONNET.value


# --- get_assumed_role_boto_session ----------------------------------------


def test_assumed_role_session_returns_input_when_arn_none() -> None:
    session = _FakeSession()
    assert get_assumed_role_boto_session(session, assumed_role_arn=None) is session


def test_assumed_role_session_calls_assume_role(mocker) -> None:
    session = _FakeSession()
    new_session = object()
    spy = mocker.patch.object(bedrock_mod, "assume_role", return_value=new_session)
    out = get_assumed_role_boto_session(
        session, assumed_role_arn="arn:aws:iam::123:role/Target"
    )
    assert out is new_session
    spy.assert_called_once()
    # Default session name + 1h duration wired through.
    _, kwargs = spy.call_args
    assert kwargs["RoleSessionName"] == bedrock_mod.DEFAULT_ROLE_SESSION_NAME
    assert kwargs["DurationSeconds"] == 3600
