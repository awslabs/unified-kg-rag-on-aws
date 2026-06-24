# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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

from aws_graphrag.adapters.aws import bedrock as bedrock_mod
from aws_graphrag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    LanguageModelInfo,
    get_assumed_role_boto_session,
)
from aws_graphrag.domain.models import (
    Config,
    EmbeddingModelId,
    LanguageModelId,
)
from aws_graphrag.shared import EmbeddingModelError

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


def test_get_model_info_returns_none_for_unmapped_model() -> None:
    # CLAUDE_V3_SONNET is a valid enum member but has no _LANGUAGE_MODEL_INFO
    # entry, so capability lookup degrades to None (callers raise on this).
    factory = _lang_factory()
    assert factory.get_model_info(LanguageModelId.CLAUDE_V3_SONNET) is None


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
    class _Client:
        def list_inference_profiles(self, **kwargs: Any) -> dict:
            return {"inferenceProfileSummaries": [{"inferenceProfileId": "us.model-x"}]}

    assert (
        BedrockCrossRegionModelHelper._is_cross_region_model_available(
            _Client(), "us.model-x"
        )
        is True
    )


def test_is_cross_region_model_available_false() -> None:
    class _Client:
        def list_inference_profiles(self, **kwargs: Any) -> dict:
            return {"inferenceProfileSummaries": []}

    assert (
        BedrockCrossRegionModelHelper._is_cross_region_model_available(
            _Client(), "us.model-x"
        )
        is False
    )


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
