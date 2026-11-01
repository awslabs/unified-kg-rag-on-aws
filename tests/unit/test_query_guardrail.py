# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query-path guardrail: ApplyGuardrail on the raw query and the final answer.

The Bedrock adapter runs against a real ``bedrock-runtime`` client with a
botocore ``Stubber``, so every request is validated against the service
model and an unexpected (unstubbed) call fails the test. The chain tests use
a fake model factory and retrievers, as in ``test_rag_chain_streaming``.
"""

from __future__ import annotations

import inspect
import itertools
import logging
from collections.abc import Callable, Iterator
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import fake_model_info
from unified_kg_rag.adapters.guardrail.bedrock import (
    DEFAULT_BLOCKED_MESSAGE,
    GROUNDING_ANSWER_MAX_CHARS,
    GROUNDING_QUERY_MAX_CHARS,
    GROUNDING_SOURCE_MAX_CHARS,
    BedrockGuardrail,
    grounding_from_response,
    select_grounding_source,
)
from unified_kg_rag.adapters.retrieval.token_manager import EMPTY_CONTEXT_PLACEHOLDER
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
    SearchOutput,
    _guard_record,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import GuardrailConfig
from unified_kg_rag.ports import (
    ContentGuardPort,
    GuardAction,
    GuardSource,
    GuardVerdict,
)
from unified_kg_rag.ports.guardrail import GroundingCheck, GroundingContext
from unified_kg_rag.shared import ConfigurationError, GuardrailError

pytestmark = pytest.mark.unit

_GUARDRAIL_ID = "gr-test"
_QUERY = "What does Vendor supply to Buyer?"
_ANSWER = "Vendor supplies Buyer with parts under the agreement."
_REFUSAL = "Sorry, this request is not allowed."

_USAGE = dict.fromkeys(
    (
        "topicPolicyUnits",
        "contentPolicyUnits",
        "wordPolicyUnits",
        "sensitiveInformationPolicyUnits",
        "sensitiveInformationPolicyFreeUnits",
        "contextualGroundingPolicyUnits",
    ),
    1,
)
_NONE: dict[str, Any] = {"action": "NONE", "outputs": [], "assessments": [{}]}
_PROMPT_ATTACK: dict[str, Any] = {
    "action": "GUARDRAIL_INTERVENED",
    "outputs": [{"text": _REFUSAL}],
    "assessments": [
        {
            "contentPolicy": {
                "filters": [
                    {
                        "type": "PROMPT_ATTACK",
                        "confidence": "HIGH",
                        "action": "BLOCKED",
                    }
                ]
            }
        }
    ],
}


def _anonymized(masked: str, *pii: str) -> dict[str, Any]:
    return {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": masked}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": t, "match": "x", "action": "ANONYMIZED"} for t in pii
                    ],
                    "regexes": [],
                }
            }
        ],
    }


def _request(text: str, source: str = "INPUT") -> dict[str, Any]:
    return {
        "guardrailIdentifier": _GUARDRAIL_ID,
        "guardrailVersion": "DRAFT",
        "source": source,
        "content": [{"text": {"text": text}}],
    }


def _guarded_config(**guardrail: Any) -> Config:
    config = Config()
    config.aws.bedrock.guardrail.identifier = _GUARDRAIL_ID
    for key, value in guardrail.items():
        setattr(config.aws.bedrock.guardrail, key, value)
    return config


@pytest.fixture
def runtime() -> Iterator[tuple[Any, Stubber]]:
    client = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    ).client("bedrock-runtime")
    stubber = Stubber(client)
    add_response = stubber.add_response

    def add_with_usage(method: str, response: dict, expected: Any = None) -> None:
        add_response(method, {"usage": _USAGE, **response}, expected)

    stubber.add_response = add_with_usage  # type: ignore[method-assign]
    with stubber:
        yield client, stubber
        # Every stubbed response was consumed: no expected check was skipped.
        stubber.assert_no_pending_responses()


# --- adapter -----------------------------------------------------------------


def test_bedrock_guardrail_conforms_to_port(runtime) -> None:  # noqa: ANN001
    client, _ = runtime
    assert isinstance(
        BedrockGuardrail(_guarded_config(), client=client), ContentGuardPort
    )


async def test_screens_only_the_given_text(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    guard = BedrockGuardrail(_guarded_config(), client=client)

    verdict = await guard.acheck(_QUERY, GuardSource.INPUT)

    assert verdict.action is GuardAction.NONE
    assert verdict.text == _QUERY
    assert not verdict.intervened


async def test_trace_requests_full_output(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response(
        "apply_guardrail",
        _NONE,
        {**_request(_ANSWER, "OUTPUT"), "outputScope": "FULL"},
    )
    guard = BedrockGuardrail(_guarded_config(trace=True), client=client)
    await guard.acheck(_ANSWER, GuardSource.OUTPUT)


async def test_prompt_attack_is_blocked(runtime, caplog) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _PROMPT_ATTACK, _request(_QUERY))
    guard = BedrockGuardrail(_guarded_config(), client=client)

    with caplog.at_level(logging.WARNING):
        verdict = await guard.acheck(_QUERY, GuardSource.INPUT)

    assert verdict.action is GuardAction.BLOCKED
    assert verdict.text == _REFUSAL
    assert verdict.policies == ["contentPolicy:PROMPT_ATTACK"]
    assert _GUARDRAIL_ID in caplog.text and "blocked the input" in caplog.text


async def test_anonymized_only_is_masked(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    masked = "Write to {EMAIL} about the order."
    stubber.add_response(
        "apply_guardrail",
        _anonymized(masked, "EMAIL"),
        _request("Write to a@example.com about the order."),
    )
    guard = BedrockGuardrail(_guarded_config(), client=client)

    verdict = await guard.acheck(
        "Write to a@example.com about the order.", GuardSource.INPUT
    )

    assert verdict.action is GuardAction.MASKED
    assert verdict.text == masked
    assert verdict.policies == ["sensitiveInformationPolicy:EMAIL"]


async def test_any_blocked_entry_blocks(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    response = _anonymized("ignored", "NAME")
    response["assessments"].append(
        {
            "topicPolicy": {
                "topics": [{"name": "Advice", "type": "DENY", "action": "BLOCKED"}]
            }
        }
    )
    stubber.add_response("apply_guardrail", response, _request(_QUERY))
    guard = BedrockGuardrail(_guarded_config(), client=client)

    verdict = await guard.acheck(_QUERY, GuardSource.INPUT)

    assert verdict.action is GuardAction.BLOCKED
    assert verdict.policies == [
        "sensitiveInformationPolicy:NAME",
        "topicPolicy:Advice",
    ]


async def test_unexplained_intervention_fails_closed(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response(
        "apply_guardrail",
        {"action": "GUARDRAIL_INTERVENED", "outputs": [], "assessments": [{}]},
        _request(_QUERY),
    )
    guard = BedrockGuardrail(_guarded_config(), client=client)

    verdict = await guard.acheck(_QUERY, GuardSource.INPUT)

    assert verdict.action is GuardAction.BLOCKED
    assert verdict.text == DEFAULT_BLOCKED_MESSAGE


async def test_detect_mode_entries_do_not_intervene(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    detected = {
        "action": "NONE",
        "outputs": [],
        "assessments": [
            {
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "PROMPT_ATTACK",
                            "confidence": "HIGH",
                            "action": "NONE",
                            "detected": True,
                        }
                    ]
                }
            }
        ],
    }
    stubber.add_response("apply_guardrail", detected, _request(_QUERY))
    guard = BedrockGuardrail(_guarded_config(), client=client)

    assert (await guard.acheck(_QUERY, GuardSource.INPUT)).action is GuardAction.NONE


async def test_api_error_raises_guardrail_error(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_client_error("apply_guardrail", "AccessDeniedException", "denied")
    guard = BedrockGuardrail(_guarded_config(), client=client)

    with pytest.raises(GuardrailError, match="could not screen the input"):
        await guard.acheck(_QUERY, GuardSource.INPUT)


def test_adapter_requires_an_identifier() -> None:
    with pytest.raises(GuardrailError):
        BedrockGuardrail(Config(), client=object())


# --- chain -------------------------------------------------------------------


class _Factory:
    """LLMFactoryPort whose models answer ``_ANSWER``; records each prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.model_kwargs: list[dict[str, Any]] = []

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.model_kwargs.append(kwargs)
        prompts = self.prompts

        class _Model(GenericFakeChatModel):
            def _generate(self, messages, *args, **kw):  # type: ignore[no-untyped-def]
                prompts.append(" ".join(str(m.content) for m in messages))
                return super()._generate(messages, *args, **kw)

            async def _astream(self, messages, *args, **kw):  # type: ignore[no-untyped-def]
                prompts.append(" ".join(str(m.content) for m in messages))
                async for chunk in super()._astream(messages, *args, **kw):
                    yield chunk

        return _Model(messages=itertools.repeat(AIMessage(content=_ANSWER)))

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


class _Retriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query.query)
        return [
            RetrievalResult(
                content="Vendor and Buyer signed the supply agreement.",
                score=0.9,
                source="doc-1",
                retriever_type="document",
                metadata={"id": "doc-1", "text_unit_ids": []},
            )
        ]


class _Memory:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []

    async def load_history(self, key: Any) -> str:
        return ""

    async def add_turn(self, key: Any, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))


def _chain(
    config: Config,
    client: Any,
    *,
    ignore_errors: bool = False,
) -> tuple[GraphRAGChain, _Factory, _Retriever, _Memory]:
    config.processing.ignore_errors = ignore_errors
    factory, retriever, memory = _Factory(), _Retriever(), _Memory()
    chain = GraphRAGChain(
        config=config,
        model_factory=factory,
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
        guard=BedrockGuardrail(config, client=client),
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    chain.memory_manager = memory  # type: ignore[assignment]
    return chain, factory, retriever, memory


def _input(query: str = _QUERY, **overrides: Any) -> RAGInput:
    return RAGInput(
        query=query,
        search_strategy=SearchStrategy.SIMPLE,
        enable_query_processing=False,
        use_memory=True,
        conversation_id="conv-1",
        **overrides,
    )


async def test_blocked_query_never_reaches_retrieval_models_or_memory(
    runtime,  # noqa: ANN001
) -> None:
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _PROMPT_ATTACK, _request(_QUERY))
    chain, factory, retriever, memory = _chain(_guarded_config(), client)

    output = await chain.ainvoke(_input())

    assert isinstance(output, RAGOutput)
    assert output.answer == _REFUSAL
    assert output.sources == []
    assert output.metadata["blocked"] is True
    assert output.metadata["guardrail"] == [
        {
            "source": "input",
            "action": "blocked",
            "policies": ["contentPolicy:PROMPT_ATTACK"],
        }
    ]
    assert retriever.queries == [] and factory.prompts == [] and memory.turns == []


async def test_clean_query_screens_query_then_answer_only(
    runtime,
) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    stubber.add_response("apply_guardrail", _NONE, _request(_ANSWER, "OUTPUT"))
    chain, factory, retriever, memory = _chain(_guarded_config(), client)

    output = await chain.ainvoke(_input())

    assert isinstance(output, RAGOutput)
    assert output.answer == _ANSWER
    assert "guardrail" not in output.metadata
    assert retriever.queries and factory.prompts
    # The answer model call carries no attached guardrail: it would judge the
    # retrieved context and the developer prompt as user input.
    assert all("guardrail" not in str(kwargs) for kwargs in factory.model_kwargs)
    assert memory.turns == [(_QUERY, _ANSWER)]


async def test_masked_query_is_all_that_retrieval_models_and_memory_see(
    runtime,  # noqa: ANN001
) -> None:
    client, stubber = runtime
    raw = "What did a@example.com order from Vendor?"
    masked = "What did {EMAIL} order from Vendor?"
    stubber.add_response("apply_guardrail", _anonymized(masked, "EMAIL"), _request(raw))
    stubber.add_response("apply_guardrail", _NONE, _request(_ANSWER, "OUTPUT"))
    chain, factory, retriever, memory = _chain(_guarded_config(), client)
    rag_input = _input(raw)

    output = await chain.ainvoke(rag_input)

    assert isinstance(output, RAGOutput)
    assert rag_input.query == raw  # the caller's object is not rewritten
    assert retriever.queries and set(retriever.queries) == {masked}
    assert factory.prompts and all("a@example.com" not in p for p in factory.prompts)
    assert output.processed_query.original_query == masked
    assert memory.turns == [(masked, _ANSWER)]
    assert output.metadata["guardrail"][0]["action"] == "masked"


async def test_blocked_answer_is_replaced_before_memory(
    runtime,
) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    stubber.add_response("apply_guardrail", _PROMPT_ATTACK, _request(_ANSWER, "OUTPUT"))
    chain, _, _, memory = _chain(_guarded_config(), client)

    output = await chain.ainvoke(_input())

    assert isinstance(output, RAGOutput)
    assert output.answer == _REFUSAL
    assert output.metadata["guardrail"][0]["source"] == "output"
    assert "blocked" not in output.metadata  # the query itself was answered
    assert memory.turns == [(_QUERY, _REFUSAL)]
    # Nor does the refused answer hand out the evidence it was built from.
    assert output.sources == []
    assert output.search_results.results == []
    assert output.search_results.total_results == 0


async def test_masked_answer_keeps_its_sources(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    stubber.add_response(
        "apply_guardrail",
        _anonymized("masked answer", "EMAIL"),
        _request(_ANSWER, "OUTPUT"),
    )
    chain, *_ = _chain(_guarded_config(), client)

    output = await chain.ainvoke(_input())

    assert isinstance(output, RAGOutput)
    assert output.answer == "masked answer"
    assert output.sources and output.search_results.results


async def test_checks_follow_their_switches(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    # Only the answer is screened; an input check would be an unstubbed call.
    stubber.add_response("apply_guardrail", _NONE, _request(_ANSWER, "OUTPUT"))
    chain, _, _, _ = _chain(_guarded_config(check_input=False), client)
    await chain.ainvoke(_input())


async def test_guardrail_error_fails_closed(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_client_error("apply_guardrail", "ThrottlingException", "slow down")
    chain, factory, retriever, _ = _chain(_guarded_config(), client)

    with pytest.raises(GuardrailError):
        await chain.ainvoke(_input())
    assert retriever.queries == [] and factory.prompts == []


async def test_guardrail_error_with_ignore_errors_is_the_error_answer(
    runtime,  # noqa: ANN001
) -> None:
    client, stubber = runtime
    stubber.add_client_error("apply_guardrail", "ThrottlingException", "slow down")
    chain, _, retriever, _ = _chain(_guarded_config(), client, ignore_errors=True)

    output = await chain.ainvoke(_input())

    assert isinstance(output, RAGOutput)
    assert output.answer == DEFAULT_ERROR_MESSAGE
    assert output.metadata["error"] is True
    assert retriever.queries == []


async def test_search_mode_blocked_query(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _PROMPT_ATTACK, _request(_QUERY))
    chain, _, retriever, _ = _chain(_guarded_config(), client)

    output = await chain.asearch(_input())

    assert isinstance(output, SearchOutput)
    assert output.metadata["blocked"] is True
    assert output.search_results.results == []
    assert retriever.queries == []


async def test_astream_blocked_query_yields_the_message_once(
    runtime,  # noqa: ANN001
) -> None:
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _PROMPT_ATTACK, _request(_QUERY))
    chain, factory, retriever, memory = _chain(_guarded_config(), client)

    chunks = [c async for c in chain.astream(_input())]

    assert chunks == [_REFUSAL]
    assert retriever.queries == [] and factory.prompts == [] and memory.turns == []


async def test_astream_screened_answer_is_one_chunk(runtime) -> None:  # noqa: ANN001
    client, stubber = runtime
    masked = "Vendor supplies {NAME} with parts under the agreement."
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    stubber.add_response(
        "apply_guardrail", _anonymized(masked, "NAME"), _request(_ANSWER, "OUTPUT")
    )
    chain, _, _, memory = _chain(_guarded_config(), client)

    chunks = [c async for c in chain.astream(_input())]

    assert chunks == [masked]
    assert memory.turns == [(_QUERY, masked)]


async def test_astream_without_output_check_still_streams(
    runtime,
) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    chain, _, _, _ = _chain(_guarded_config(check_output=False), client)

    chunks = [c async for c in chain.astream(_input())]

    assert len(chunks) > 1 and "".join(chunks) == _ANSWER


def test_no_guard_without_identifier() -> None:
    chain = GraphRAGChain(config=Config(), model_factory=_Factory())
    assert chain.guard is None


def test_no_guard_when_both_checks_are_off() -> None:
    config = _guarded_config(check_input=False, check_output=False)
    chain = GraphRAGChain(config=config, model_factory=_Factory())
    assert chain.guard is None


async def test_screening_spans_name_the_guardrail_action_and_policies(
    runtime,  # noqa: ANN001
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from unified_kg_rag.application.observability import configure_observability
    from unified_kg_rag.domain.models import ObservabilityConfig
    from unified_kg_rag.shared import observability

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_observability(ObservabilityConfig(enabled=True), tracer_provider=provider)
    try:
        client, stubber = runtime
        stubber.add_response("apply_guardrail", _PROMPT_ATTACK, _request(_QUERY))
        chain, _, _, _ = _chain(_guarded_config(), client)
        await chain.ainvoke(_input())
    finally:
        observability.uninstall()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    screen, query = spans["kg_rag.guardrail"], spans["kg_rag.query"]
    assert screen.parent is not None and screen.parent.span_id == query.context.span_id
    attrs = dict(screen.attributes or {})
    assert attrs["aws.bedrock.guardrail.id"] == _GUARDRAIL_ID
    assert attrs["aws.bedrock.guardrail.source"] == "input"
    assert attrs["aws.bedrock.guardrail.action"] == "blocked"
    assert attrs["aws.bedrock.guardrail.intervened"] is True
    assert list(attrs["aws.bedrock.guardrail.policies"]) == [
        "contentPolicy:PROMPT_ATTACK"
    ]
    assert dict(query.attributes or {})["kg_rag.blocked"] is True


async def _query_span_input(run: Any) -> Any:
    """The ``input.value`` of the ``kg_rag.query`` span ``run`` produces,
    with content capture on."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from unified_kg_rag.application.observability import configure_observability
    from unified_kg_rag.domain.models import ObservabilityConfig
    from unified_kg_rag.shared import observability

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_observability(
        ObservabilityConfig(enabled=True, capture_content=True),
        tracer_provider=provider,
    )
    try:
        await run()
    finally:
        observability.uninstall()
    (query,) = [s for s in exporter.get_finished_spans() if s.name == "kg_rag.query"]
    return dict(query.attributes or {}).get("input.value")


_RAW = "What did a@example.com order from Vendor?"
_MASKED = "What did {EMAIL} order from Vendor?"


@pytest.mark.parametrize("stream", [False, True])
async def test_query_span_records_only_the_masked_query(
    runtime, stream: bool  # noqa: ANN001
) -> None:
    client, stubber = runtime
    stubber.add_response(
        "apply_guardrail", _anonymized(_MASKED, "EMAIL"), _request(_RAW)
    )
    stubber.add_response("apply_guardrail", _NONE, _request(_ANSWER, "OUTPUT"))
    chain, *_ = _chain(_guarded_config(), client)

    async def run() -> None:
        if stream:
            async for _ in chain.astream(_input(_RAW)):
                pass
        else:
            await chain.ainvoke(_input(_RAW))

    assert await _query_span_input(run) == _MASKED


@pytest.mark.parametrize("stream", [False, True])
async def test_query_span_records_no_text_for_a_blocked_query(
    runtime, stream: bool  # noqa: ANN001
) -> None:
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _PROMPT_ATTACK, _request(_QUERY))
    chain, *_ = _chain(_guarded_config(), client)

    async def run() -> None:
        if stream:
            async for _ in chain.astream(_input()):
                pass
        else:
            await chain.ainvoke(_input())

    assert await _query_span_input(run) is None


# --- contextual grounding ------------------------------------------------------

_CONTEXT = "Vendor and Buyer signed the supply agreement."


def _scored(grounding: float, relevance: float, threshold: float = 0.7) -> dict:
    """A FULL-scope reply scoring the answer; blocked below ``threshold``."""
    blocked = min(grounding, relevance) < threshold
    return {
        "action": "GUARDRAIL_INTERVENED" if blocked else "NONE",
        "outputs": [{"text": _REFUSAL}] if blocked else [],
        "assessments": [
            {
                "contextualGroundingPolicy": {
                    "filters": [
                        {
                            "type": kind,
                            "threshold": threshold,
                            "score": score,
                            "action": "BLOCKED" if score < threshold else "NONE",
                            "detected": score < threshold,
                        }
                        for kind, score in (
                            ("GROUNDING", grounding),
                            ("RELEVANCE", relevance),
                        )
                    ]
                }
            }
        ],
    }


def _grounded_request(
    answer: str = _ANSWER, source: str = _CONTEXT, query: str = _QUERY
) -> dict[str, Any]:
    return {
        **_request(answer, "OUTPUT"),
        "content": [
            {"text": {"text": answer, "qualifiers": ["guard_content"]}},
            {"text": {"text": source, "qualifiers": ["grounding_source"]}},
            {"text": {"text": query, "qualifiers": ["query"]}},
        ],
        "outputScope": "FULL",
    }


def _grounded_config() -> Config:
    config = _guarded_config()
    config.aws.bedrock.guardrail.contextual_grounding.enabled = True
    return config


def _context(
    source: str = _CONTEXT, query: str = _QUERY, ranked: list[str] | None = None
) -> GroundingContext:
    return GroundingContext(query=query, source=source, ranked_sources=ranked or [])


async def _ground(runtime: Any, response: dict, text: str = _ANSWER, **kw: Any) -> Any:
    client, stubber = runtime
    stubber.add_response("apply_guardrail", response, kw.pop("request", None))
    guard = BedrockGuardrail(_grounded_config(), client=client)
    return await guard.acheck(text, GuardSource.OUTPUT, grounding=_context(**kw))


async def test_grounding_sends_answer_reference_and_query(
    runtime,
) -> None:  # noqa: ANN001
    verdict = await _ground(runtime, _scored(0.93, 0.88), request=_grounded_request())

    assert verdict.action is GuardAction.NONE and verdict.text == _ANSWER
    assert verdict.grounding == GroundingCheck(
        checked=True, scores={"grounding": 0.93, "relevance": 0.88}
    )


async def test_ungrounded_answer_is_blocked(runtime, caplog) -> None:  # noqa: ANN001
    with caplog.at_level(logging.WARNING):
        verdict = await _ground(runtime, _scored(0.12, 0.91))

    assert verdict.action is GuardAction.BLOCKED and verdict.text == _REFUSAL
    assert verdict.policies == ["contextualGroundingPolicy:GROUNDING"]
    assert verdict.grounding is not None and verdict.grounding.failed == ["grounding"]
    assert verdict.grounding.blocked
    assert "on an answer blocked" in caplog.text and "grounding 0.12" in caplog.text


async def test_detect_only_failure_passes_but_is_logged(
    runtime, caplog
) -> None:  # noqa: ANN001
    response = _scored(0.2, 0.9)
    response["action"], response["outputs"] = "NONE", []
    for entry in response["assessments"][0]["contextualGroundingPolicy"]["filters"]:
        entry["action"] = "NONE"
    with caplog.at_level(logging.WARNING):
        verdict = await _ground(runtime, response)

    assert verdict.action is GuardAction.NONE
    assert verdict.grounding is not None and verdict.grounding.failed == ["grounding"]
    assert not verdict.grounding.blocked
    assert "failed but not blocked" in caplog.text


async def test_another_policys_block_is_not_a_grounding_block(
    runtime, caplog
) -> None:  # noqa: ANN001
    # Blocked for violence while the grounding check failed detect-only.
    response = _scored(0.2, 0.9)
    for entry in response["assessments"][0]["contextualGroundingPolicy"]["filters"]:
        entry["action"] = "NONE"
    response["assessments"][0]["contentPolicy"] = {
        "filters": [{"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}]
    }
    with caplog.at_level(logging.WARNING):
        verdict = await _ground(runtime, response)

    assert verdict.action is GuardAction.BLOCKED
    assert verdict.policies == ["contentPolicy:VIOLENCE"]
    assert verdict.grounding is not None and verdict.grounding.failed == ["grounding"]
    assert not verdict.grounding.blocked
    assert "failed but not blocked" in caplog.text


def test_a_blocking_entry_without_an_intervention_blocked_nothing() -> None:
    response = _scored(0.2, 0.9)
    response["action"], response["outputs"] = "NONE", []
    check = grounding_from_response(response)
    assert check.failed == ["grounding"] and not check.blocked


async def test_guardrail_without_grounding_policy_is_logged(
    runtime, caplog
) -> None:  # noqa: ANN001
    with caplog.at_level(logging.WARNING):
        verdict = await _ground(runtime, _NONE)

    assert verdict.grounding == GroundingCheck(
        checked=False, skipped_reason="not_assessed"
    )
    assert "no contextual grounding policy" in caplog.text


async def test_masked_grounded_answer_never_takes_the_reference(
    runtime,
) -> None:  # noqa: ANN001
    masked = "Write to {EMAIL} about the order."
    response = _anonymized(masked, "EMAIL")
    # A reply that mirrors every block sent: only the first is the answer.
    response["outputs"] += [{"text": _CONTEXT}, {"text": _QUERY}]
    verdict = await _ground(runtime, response)

    assert verdict.action is GuardAction.MASKED and verdict.text == masked


async def test_grounding_is_not_sent_with_a_query_check(
    runtime,
) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    guard = BedrockGuardrail(_grounded_config(), client=client)

    verdict = await guard.acheck(_QUERY, GuardSource.INPUT, grounding=_context())

    assert verdict.grounding is None


def test_long_reference_keeps_the_highest_ranked_sections() -> None:
    top, second, third = "A" * 60_000, "B" * 39_000, "C" * 5_000
    # The rendered context groups sections by type, not by rank.
    context = _context("\n".join((third, second, top)), ranked=[top, second, third])

    reference = select_grounding_source(context)

    assert len(reference) == GROUNDING_SOURCE_MAX_CHARS
    # Whole sections in rank order, then the head of the next one.
    assert reference == "\n".join((top, second, third[:998]))
    # At the limit the context goes verbatim; without sections, its head.
    exact = "x" * GROUNDING_SOURCE_MAX_CHARS
    assert select_grounding_source(_context(exact, ranked=["a"])) == exact
    assert select_grounding_source(_context(exact + "y")) == exact


async def test_reference_and_query_are_cut_to_bedrock_limits(
    runtime,
) -> None:  # noqa: ANN001
    top, rest = "A" * 70_000, "B" * 70_000
    await _ground(
        runtime,
        _scored(0.9, 0.9),
        request=_grounded_request(
            source=top + "\n" + rest[: GROUNDING_SOURCE_MAX_CHARS - 70_001],
            query="Q" * GROUNDING_QUERY_MAX_CHARS,
        ),
        source=rest + top,
        query="Q" * (GROUNDING_QUERY_MAX_CHARS + 50),
        ranked=[top, rest],
    )


async def test_oversize_answer_skips_grounding_and_says_so(
    runtime, caplog
) -> None:  # noqa: ANN001
    long_answer = "Vendor supplies parts. " * 250
    assert len(long_answer) > GROUNDING_ANSWER_MAX_CHARS
    with caplog.at_level(logging.WARNING):
        # Screened by the other policies only: no grounding blocks are sent.
        verdict = await _ground(
            runtime, _NONE, long_answer, request=_request(long_answer, "OUTPUT")
        )

    assert verdict.action is GuardAction.NONE
    assert verdict.grounding == GroundingCheck(
        checked=False, skipped_reason="answer_too_long"
    )
    assert "exceed the 5000-character limit" in caplog.text


def test_grounding_requires_the_output_check() -> None:
    with pytest.raises(ValueError, match="requires check_output"):
        GuardrailConfig(
            identifier=_GUARDRAIL_ID,
            check_output=False,
            contextual_grounding={"enabled": True},
        )


def _grounding_on_after_check_output_turned_off(config: Config) -> None:
    # The nested section validates alone: it cannot see check_output.
    config.aws.bedrock.guardrail.check_output = False
    config.aws.bedrock.guardrail.contextual_grounding.enabled = True


def _check_output_refused_but_applied(config: Config) -> None:
    config.aws.bedrock.guardrail.contextual_grounding.enabled = True
    with pytest.raises(ValueError, match="requires check_output"):
        config.aws.bedrock.guardrail.check_output = False


@pytest.mark.parametrize(
    "change",
    [_grounding_on_after_check_output_turned_off, _check_output_refused_but_applied],
)
def test_the_chain_refuses_grounding_without_the_output_check_set_in_code(
    change: Callable[[Config], None],
) -> None:
    # Either order of setting it in code ends with the grounding check on and
    # check_output off; the chain refuses to build rather than never running
    # the check.
    config = _guarded_config()
    change(config)
    assert config.aws.bedrock.guardrail.contextual_grounding.enabled
    assert not config.aws.bedrock.guardrail.check_output
    with pytest.raises(ConfigurationError, match="requires check_output"):
        GraphRAGChain(config=config, model_factory=_Factory())


def test_adapter_accepts_the_port_call_shape() -> None:
    def shape(func: Any) -> list[tuple[str, Any, bool]]:
        return [
            (p.name, p.kind, p.default is not inspect.Parameter.empty)
            for p in inspect.signature(func).parameters.values()
        ]

    assert shape(BedrockGuardrail.acheck) == shape(ContentGuardPort.acheck)


class _RecordingGuard:
    """A ContentGuardPort that passes everything and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, GuardSource, dict[str, Any]]] = []

    async def acheck(
        self, text: str, source: GuardSource, **kwargs: Any
    ) -> GuardVerdict:
        self.calls.append((text, source, kwargs))
        return GuardVerdict(action=GuardAction.NONE, text=text)


def _recording_chain(config: Config) -> tuple[GraphRAGChain, _RecordingGuard, _Factory]:
    guard, factory, retriever = _RecordingGuard(), _Factory(), _Retriever()
    chain = GraphRAGChain(
        config=config,
        model_factory=factory,
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
        guard=guard,
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    chain.memory_manager = _Memory()  # type: ignore[assignment]
    return chain, guard, factory


@pytest.mark.parametrize("grounded", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_answer_check_gets_what_the_answer_model_was_given(
    grounded: bool, stream: bool
) -> None:
    config = _grounded_config() if grounded else _guarded_config()
    chain, guard, factory = _recording_chain(config)

    if stream:
        assert [c async for c in chain.astream(_input())] == [_ANSWER]
    else:
        output = await chain.ainvoke(_input())
        assert isinstance(output, RAGOutput)
        assert "grounding" not in output.metadata  # the guard reported none

    text, source, kwargs = guard.calls[-1]
    assert (text, source) == (_ANSWER, GuardSource.OUTPUT)
    if not grounded:
        # Off: the plain two-argument call, so a guard written before the
        # grounding check keeps working.
        assert kwargs == {}
        return
    grounding = kwargs["grounding"]
    assert grounding.query == _QUERY
    # The reference is the context string the answer prompt carried.
    assert _CONTEXT in grounding.source and grounding.source in factory.prompts[-1]
    assert grounding.ranked_sources == ["[doc-1] " + _CONTEXT]


async def test_ungrounded_answer_takes_the_refusal_path(
    runtime,
) -> None:  # noqa: ANN001
    client, stubber = runtime
    stubber.add_response("apply_guardrail", _NONE, _request(_QUERY))
    stubber.add_response("apply_guardrail", _scored(0.1, 0.95))
    chain, _, _, memory = _chain(_grounded_config(), client)

    output = await chain.ainvoke(_input())

    assert isinstance(output, RAGOutput)
    assert output.answer == _REFUSAL
    assert output.sources == [] and output.search_results.results == []
    assert output.metadata["guardrail"] == [
        {
            "source": "output",
            "action": "blocked",
            "policies": ["contextualGroundingPolicy:GROUNDING"],
            "reason": "contextual_grounding",
        }
    ]
    assert output.metadata["grounding"]["scores"] == {
        "grounding": 0.1,
        "relevance": 0.95,
    }
    assert memory.turns == [(_QUERY, _REFUSAL)]


@pytest.mark.parametrize(
    ("policies", "grounding", "reason"),
    [
        # Any guard: the reason comes from its grounding check, not from
        # how it names its policies.
        (["acme:ungrounded"], GroundingCheck(checked=True, blocked=True), True),
        # Blocked by another policy while the check failed detect-only.
        (
            ["contentPolicy:VIOLENCE"],
            GroundingCheck(checked=True, failed=["grounding"]),
            False,
        ),
        (["contentPolicy:VIOLENCE"], None, False),
    ],
)
def test_the_block_reason_follows_the_grounding_check(
    policies: list[str], grounding: GroundingCheck | None, reason: bool
) -> None:
    verdict = GuardVerdict(
        action=GuardAction.BLOCKED,
        text=_REFUSAL,
        policies=policies,
        grounding=grounding,
    )
    record = _guard_record(GuardSource.OUTPUT, verdict)
    assert record.get("reason") == ("contextual_grounding" if reason else None)


def test_no_context_means_no_grounding_reference() -> None:
    chain, _, _ = _recording_chain(_grounded_config())
    # The answer step short-circuits to the fixed no-context answer, which is
    # not screened at all.
    for context in ("", "   ", EMPTY_CONTEXT_PLACEHOLDER):
        assert chain._grounding_context({"context": context}) is None
