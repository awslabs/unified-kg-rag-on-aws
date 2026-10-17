# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query-path guardrail: ApplyGuardrail on the raw query and the final answer.

The Bedrock adapter runs against a real ``bedrock-runtime`` client with a
botocore ``Stubber``, so every request is validated against the service
model and an unexpected (unstubbed) call fails the test. The chain tests use
a fake model factory and retrievers, as in ``test_rag_chain_streaming``.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
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
    BedrockGuardrail,
)
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
    SearchOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.ports import ContentGuardPort, GuardAction, GuardSource
from unified_kg_rag.shared import GuardrailError

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

    output = await chain.ainvoke(_input(raw))

    assert isinstance(output, RAGOutput)
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
