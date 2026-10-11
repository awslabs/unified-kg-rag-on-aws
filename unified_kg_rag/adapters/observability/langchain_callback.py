# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LangChain callback handler that traces each chat model call.

One span per model call, named and attributed per the OpenTelemetry GenAI
semantic conventions: ``gen_ai.operation.name``, ``gen_ai.provider.name``,
``gen_ai.request.model``, request parameters, ``gen_ai.usage.*`` token counts
(cache reads and writes when the provider reports them), finish reasons and
whether a guardrail intervened. Cost is not recorded: the package has no
model price table to derive it from.

Prompt and completion text (``gen_ai.input.messages`` /
``gen_ai.output.messages``) is recorded only when
``observability.capture_content`` is on, after ``redact_patterns``.

The handler goes through the ``shared.observability`` port, never
OpenTelemetry itself, so it costs one check per callback when tracing is off.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from unified_kg_rag.shared import observability
from unified_kg_rag.shared.observability import ContentPolicy, SpanHandle

# GenAI semantic-convention attribute names (see adapters.observability.otel
# for the pinned schema version).
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GUARDRAIL_INTERVENED = "aws.bedrock.guardrail.intervened"
MODEL_PURPOSE = "kg_rag.model.purpose"

PROVIDER_AWS_BEDROCK = "aws.bedrock"

_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _content_text(content: Any) -> str:
    """The text of a message's content (a string or a list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _input_messages(messages: Sequence[BaseMessage], policy: ContentPolicy) -> str:
    return json.dumps(
        [
            {
                "role": _ROLES.get(message.type, message.type),
                "parts": [
                    {
                        "type": "text",
                        "content": policy.redact(_content_text(message.content)),
                    }
                ],
            }
            for message in messages
        ],
        ensure_ascii=False,
    )


def _finish_reason(metadata: Mapping[str, Any]) -> str | None:
    reason = metadata.get("stopReason") or metadata.get("stop_reason")
    return str(reason) if reason else None


class GenAITracingCallbackHandler(BaseCallbackHandler):
    """Trace chat model calls as GenAI spans (see the module docstring).

    ``guardrail_detector`` tells from a response's metadata whether a
    guardrail intervened (the Bedrock factory passes its detector).
    """

    # Run in the caller's thread and context even on async calls, so the model
    # span is a child of the span that made the call.
    run_inline = True

    def __init__(
        self,
        model_id: str,
        *,
        provider_name: str = PROVIDER_AWS_BEDROCK,
        purpose: str | None = None,
        guardrail_detector: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.model_id = model_id
        self.provider_name = provider_name
        self.purpose = purpose
        self.guardrail_detector = guardrail_detector
        self._spans: dict[UUID, SpanHandle] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not observability.is_enabled():
            return
        params = kwargs.get("invocation_params") or {}
        attributes: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: "chat",
            GEN_AI_PROVIDER_NAME: self.provider_name,
            GEN_AI_REQUEST_MODEL: self.model_id,
            GEN_AI_REQUEST_MAX_TOKENS: params.get("max_tokens"),
            GEN_AI_REQUEST_TEMPERATURE: params.get("temperature"),
            MODEL_PURPOSE: self.purpose,
        }
        policy = observability.content_policy()
        if policy.capture_content and messages:
            attributes[GEN_AI_INPUT_MESSAGES] = _input_messages(messages[0], policy)
        span = observability.start_span(f"chat {self.model_id}", attributes)
        with self._lock:
            self._spans[run_id] = span

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            span = self._spans.pop(run_id, None)
        if span is None:
            return
        span.set_attributes(self._response_attributes(response))
        span.end()

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if kwargs.get("reason") == "GUARDRAIL_INTERVENED":
            # InvokeModel with guardrail trace reports the intervention here
            # and then still calls on_llm_end, which ends the span.
            with self._lock:
                span = self._spans.get(run_id)
            if span is not None:
                span.set_attribute(GUARDRAIL_INTERVENED, True)
            return
        with self._lock:
            span = self._spans.pop(run_id, None)
        if span is None:
            return
        span.record_exception(error)
        span.set_error(type(error).__name__)
        span.end()

    def _response_attributes(self, response: LLMResult) -> dict[str, Any]:
        input_tokens = output_tokens = cache_read = cache_creation = reasoning = 0
        has_usage = False
        finish_reasons: list[str] = []
        response_model: str | None = None
        intervened = False
        outputs: list[dict[str, Any]] = []
        policy = observability.content_policy()

        for generations in response.generations:
            for generation in generations:
                if not isinstance(generation, ChatGeneration):
                    continue
                message = generation.message
                metadata = message.response_metadata or {}
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    has_usage = True
                    input_tokens += int(usage.get("input_tokens") or 0)
                    output_tokens += int(usage.get("output_tokens") or 0)
                    input_details = usage.get("input_token_details") or {}
                    cache_read += int(input_details.get("cache_read") or 0)
                    cache_creation += int(input_details.get("cache_creation") or 0)
                    output_details = usage.get("output_token_details") or {}
                    reasoning += int(output_details.get("reasoning") or 0)
                reason = _finish_reason(metadata)
                if reason:
                    finish_reasons.append(reason)
                response_model = response_model or (
                    metadata.get("model_id") or metadata.get("model_name")
                )
                if self.guardrail_detector is not None and self.guardrail_detector(
                    metadata
                ):
                    intervened = True
                if policy.capture_content:
                    output: dict[str, Any] = {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "text",
                                "content": policy.redact(generation.text),
                            }
                        ],
                    }
                    if reason:
                        output["finish_reason"] = reason
                    outputs.append(output)

        if not has_usage:
            # ChatBedrock (InvokeModel) also reports the totals in llm_output.
            usage = (response.llm_output or {}).get("usage") or {}
            if usage:
                has_usage = True
                input_tokens = int(
                    usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                )
                output_tokens = int(
                    usage.get("completion_tokens") or usage.get("output_tokens") or 0
                )

        attributes: dict[str, Any] = {
            GEN_AI_RESPONSE_MODEL: response_model,
            GEN_AI_RESPONSE_FINISH_REASONS: finish_reasons or None,
            GUARDRAIL_INTERVENED: True if intervened else None,
        }
        if has_usage:
            attributes[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
            attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
            if cache_read:
                attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read
            if cache_creation:
                attributes[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = cache_creation
            if reasoning:
                attributes[GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] = reasoning
        if outputs:
            attributes[GEN_AI_OUTPUT_MESSAGES] = json.dumps(outputs, ensure_ascii=False)
        return attributes
