# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Bedrock Guardrails wiring (WAF security pillar, AWS-free)."""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from unified_kg_rag.adapters.aws.bedrock import (
    BedrockLanguageModelFactory,
    GuardrailInterventionHandler,
)
from unified_kg_rag.domain.models import Config, ModelPurpose

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
    from unified_kg_rag.shared.config import ConfigLoader

    cfg = ConfigLoader().load_config()
    assert cfg.aws.bedrock.guardrail.identifier == "gr-from-env"
    assert cfg.aws.bedrock.guardrail.enabled is True


# --- Guardrail scope (apply_to) and intervention visibility -----------------


def _guarded_config(apply_to: str = "query") -> Config:
    cfg = Config()
    cfg.aws.bedrock.guardrail.identifier = "gr-123"
    cfg.aws.bedrock.guardrail.apply_to = apply_to  # type: ignore[assignment]
    return cfg


def test_apply_to_defaults_to_query() -> None:
    assert Config().aws.bedrock.guardrail.apply_to == "query"


def test_apply_to_rejects_unknown_value() -> None:
    from pydantic import ValidationError

    from unified_kg_rag.domain.models.config import GuardrailConfig

    with pytest.raises(ValidationError):
        GuardrailConfig(identifier="gr-123", apply_to="ingestion")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("apply_to", "purpose", "expected"),
    [
        ("query", ModelPurpose.QUERY, True),
        ("query", ModelPurpose.INGESTION, False),
        ("query", ModelPurpose.EVALUATION, False),
        ("all", ModelPurpose.QUERY, True),
        ("all", ModelPurpose.INGESTION, True),
        ("all", ModelPurpose.EVALUATION, True),
    ],
)
@pytest.mark.parametrize("is_cross_region", [True, False])
def test_guardrail_scoped_by_purpose(
    apply_to: str, purpose: ModelPurpose, expected: bool, is_cross_region: bool
) -> None:
    f = _factory_with(_guarded_config(apply_to))
    config: dict = {"callbacks": None}
    f._apply_guardrail(config, is_cross_region=is_cross_region, purpose=purpose)
    key = "guardrail_config" if is_cross_region else "guardrails"
    assert (key in config) is expected
    handlers = config["callbacks"] or []
    attached = any(isinstance(h, GuardrailInterventionHandler) for h in handlers)
    assert attached is expected


def test_disabled_guardrail_never_applies_even_with_all() -> None:
    cfg = Config()
    cfg.aws.bedrock.guardrail.apply_to = "all"
    assert cfg.aws.bedrock.guardrail.applies_to(ModelPurpose.QUERY) is False


def test_get_model_purpose_kwarg_reaches_guardrail(mocker) -> None:
    """The ``model_purpose`` get_model kwarg (via _apply_model_features) scopes it."""
    f = _factory_with(_guarded_config("query"))
    spy = mocker.spy(f, "_apply_guardrail")
    info = mocker.MagicMock(supports_thinking=False)
    f._apply_model_features({}, info, True, model_purpose=ModelPurpose.INGESTION)
    f._apply_model_features({}, info, True)
    assert spy.call_args_list[0].args[2] is ModelPurpose.INGESTION
    # Unmarked callers default to QUERY (guarded, the pre-existing behaviour).
    assert spy.call_args_list[1].args[2] is ModelPurpose.QUERY


def test_handler_appended_to_existing_callbacks() -> None:
    from langchain_core.callbacks import BaseCallbackHandler

    existing = BaseCallbackHandler()
    f = _factory_with(_guarded_config())
    config: dict = {"callbacks": [existing]}
    f._apply_guardrail(config, is_cross_region=True)
    assert config["callbacks"][0] is existing
    assert isinstance(config["callbacks"][1], GuardrailInterventionHandler)


def test_handler_added_to_copy_of_caller_callback_manager() -> None:
    from langchain_core.callbacks import BaseCallbackHandler, CallbackManager

    existing = BaseCallbackHandler()
    caller_manager = CallbackManager(handlers=[existing])
    f = _factory_with(_guarded_config())
    config: dict = {"callbacks": caller_manager}
    f._apply_guardrail(config, is_cross_region=True)

    # The caller's manager is untouched (it may be shared by unguarded models).
    assert caller_manager.handlers == [existing]
    assert config["callbacks"] is not caller_manager
    assert isinstance(config["callbacks"], CallbackManager)
    assert config["callbacks"].handlers[0] is existing
    assert isinstance(config["callbacks"].handlers[1], GuardrailInterventionHandler)
    assert not any(
        isinstance(h, GuardrailInterventionHandler)
        for h in config["callbacks"].inheritable_handlers
    )


def test_shared_callback_manager_not_polluted_across_models() -> None:
    from langchain_core.callbacks import CallbackManager

    shared = CallbackManager(handlers=[])
    f = _factory_with(_guarded_config())
    first: dict = {"callbacks": shared}
    second: dict = {"callbacks": shared}
    f._apply_guardrail(first, is_cross_region=True)
    f._apply_guardrail(second, is_cross_region=False)
    assert shared.handlers == []
    assert len(first["callbacks"].handlers) == 1
    assert len(second["callbacks"].handlers) == 1


def _llm_result(metadata: dict) -> LLMResult:
    message = AIMessage(content="x", response_metadata=metadata)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


@pytest.fixture
def handler() -> Iterator[GuardrailInterventionHandler]:
    GuardrailInterventionHandler.reset_count()
    yield GuardrailInterventionHandler("gr-123", ModelPurpose.QUERY)
    GuardrailInterventionHandler.reset_count()


@pytest.mark.parametrize(
    "metadata",
    [
        {"stopReason": "guardrail_intervened"},
        {"stop_reason": "guardrail_intervened"},
        {"amazon-bedrock-guardrailAction": "INTERVENED"},
    ],
)
def test_handler_detects_intervention(handler, metadata, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        handler.on_llm_end(_llm_result(metadata), run_id=uuid4())
    assert GuardrailInterventionHandler.intervention_count() == 1
    assert "gr-123" in caplog.text
    assert "query" in caplog.text


@pytest.mark.parametrize(
    "metadata",
    [{"stopReason": "end_turn"}, {}, {"amazon-bedrock-guardrailAction": "NONE"}],
)
def test_handler_ignores_normal_responses(handler, metadata) -> None:
    handler.on_llm_end(_llm_result(metadata), run_id=uuid4())
    assert GuardrailInterventionHandler.intervention_count() == 0


def test_handler_counts_invoke_model_signal_once(handler) -> None:
    # InvokeModel + trace reports via on_llm_error, then still calls on_llm_end.
    run_id = uuid4()
    handler.on_llm_error(
        Exception("guardrail"), run_id=run_id, reason="GUARDRAIL_INTERVENED"
    )
    handler.on_llm_end(
        _llm_result({"amazon-bedrock-guardrailAction": "INTERVENED"}), run_id=run_id
    )
    assert GuardrailInterventionHandler.intervention_count() == 1


def test_handler_ignores_unrelated_errors(handler) -> None:
    handler.on_llm_error(Exception("throttled"), run_id=uuid4())
    assert GuardrailInterventionHandler.intervention_count() == 0


def test_handler_fires_through_a_real_chain(handler) -> None:
    """End-to-end through LangChain callbacks: an intervened response is counted."""
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )

    blocked = AIMessage(
        content="This request was blocked by content policy.",
        response_metadata={"stopReason": "guardrail_intervened"},
    )
    model = GenericFakeChatModel(messages=iter([blocked]), callbacks=[handler])
    model.invoke("hello")
    assert GuardrailInterventionHandler.intervention_count() == 1


# --- Call sites pass the right purpose ---------------------------------------


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_model(self, model_id, **kwargs):  # noqa: ANN001, ANN003
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        self.calls.append(kwargs)
        return FakeListChatModel(responses=["ok"])

    def get_model_info(self, model_id):  # noqa: ANN001
        return None


def test_setup_chain_forwards_purpose_and_defaults_to_query() -> None:
    from langchain_core.output_parsers import StrOutputParser

    from unified_kg_rag.adapters.aws.chain_factory import setup_chain
    from unified_kg_rag.domain.models import LanguageModelId
    from unified_kg_rag.domain.prompts import TextTranslationPrompt

    factory = _RecordingFactory()
    common = {
        "factory": factory,
        "model_id": LanguageModelId.CLAUDE_V4_5_HAIKU,
        "prompt_class": TextTranslationPrompt,
        "parser": StrOutputParser(),
    }
    setup_chain(**common)
    setup_chain(**common, model_purpose=ModelPurpose.INGESTION)
    assert [c["model_purpose"] for c in factory.calls] == [
        ModelPurpose.QUERY,
        ModelPurpose.INGESTION,
    ]


def test_output_fixing_llm_inherits_purpose() -> None:
    from unified_kg_rag.adapters.aws.chain_factory import (
        create_robust_xml_output_parser,
    )
    from unified_kg_rag.domain.models import LanguageModelId

    factory = _RecordingFactory()
    create_robust_xml_output_parser(
        factory=factory,
        enable_output_fixing=True,
        output_fixing_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        model_purpose=ModelPurpose.INGESTION,
    )
    assert factory.calls == [{"model_purpose": ModelPurpose.INGESTION}]


def test_graph_extractor_models_are_ingestion(mocker) -> None:
    factory = _RecordingFactory()
    mocker.patch(
        "unified_kg_rag.adapters.ingestion.graph_extractor."
        "BedrockLanguageModelFactory",
        return_value=factory,
    )
    from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor

    cfg = Config()
    cfg.fixing.enabled = True
    GraphExtractor(config=cfg, boto_session=mocker.MagicMock())
    assert len(factory.calls) == 2  # output fixer + extraction chain
    assert all(c["model_purpose"] is ModelPurpose.INGESTION for c in factory.calls)


_PKG = Path(__file__).resolve().parents[2] / "unified_kg_rag"
_CHAIN_BUILDERS = {"setup_chain", "create_robust_xml_output_parser"}
# Modules whose LLM chains process corpus text (ingestion) and so must never
# fall back to the query-path guardrail scope.
_INGESTION_MODULES = sorted(
    [
        *(_PKG / "adapters" / "ingestion").glob("*.py"),
        _PKG / "application" / "prompts" / "tuner.py",
    ]
)


def _chain_builder_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CHAIN_BUILDERS
    ]


def test_ingestion_modules_build_chains() -> None:
    # Sanity check so the parametrized guard below cannot pass vacuously.
    assert sum(len(_chain_builder_calls(p)) for p in _INGESTION_MODULES) >= 10


@pytest.mark.parametrize("path", _INGESTION_MODULES, ids=lambda p: p.name)
def test_ingestion_chain_builders_mark_ingestion_purpose(path: Path) -> None:
    """Every chain built in an ingestion module passes INGESTION explicitly.

    An unmarked call defaults to QUERY, which would silently re-attach a
    PII-anonymizing guardrail to extraction (the bug this guards against).
    """
    for call in _chain_builder_calls(path):
        purposes = [
            ast.unparse(kw.value) for kw in call.keywords if kw.arg == "model_purpose"
        ]
        assert purposes == ["ModelPurpose.INGESTION"], (
            f"{path.name}:{call.lineno} {ast.unparse(call.func)}() must pass "
            "model_purpose=ModelPurpose.INGESTION"
        )
