# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite finished spans before export: vendor attribute names and redaction.

:class:`TransformingSpanExporter` wraps the OTLP exporter of the backends this
package installs. It runs in the batch processor's export thread, never on a
request path, and applies, in order:

1. a vendor mapping that *adds* the attribute names a backend reads to the
   ones the package records (``gen_ai.*``, ``kg_rag.*``):

   * Langfuse: ``langfuse.observation.type`` (``generation`` for model calls,
     ``embedding``, ``retriever`` for retrieval and rerank, ``span``
     otherwise; explicit, because Langfuse would otherwise infer
     ``generation`` from any span with a model attribute, such as rerank) and
     ``langfuse.observation.input`` / ``output`` from captured content.
   * Phoenix (OpenInference): ``openinference.span.kind``,
     ``llm.model_name``, ``llm.provider`` (``aws`` for ``aws.*`` providers,
     else the provider name), ``llm.token_count.*`` and
     ``input.value`` / ``output.value`` from captured content.

   Both read ``session.id`` / ``user.id`` (see
   ``shared.observability.bind_identity``) and ``gen_ai.usage.*`` as recorded.

2. redaction: every ``redact_patterns`` match in a string attribute (string
   list items included), event attribute (an exception message can quote a
   query) and status description becomes ``[REDACTED]``, whatever code set
   it, including the vendor copies made in step 1. This is the second layer
   for captured content: the callback handler already redacts prompt and
   completion text at capture, as raw text, on every backend, because here it
   is JSON-encoded and a pattern that uses ``\\b``, ``\\s`` or ``^...$``
   can miss a match next to an encoded newline.

The mappings are plain functions over attribute dicts, testable without an
exporter.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status

from unified_kg_rag.domain.models.config import ObservabilityBackend
from unified_kg_rag.shared.observability import REDACTED

Attributes = Mapping[str, Any]
AttributeMapper = Callable[[str, Attributes], dict[str, Any]]

_OPERATION = "gen_ai.operation.name"
_REQUEST_MODEL = "gen_ai.request.model"
_RESPONSE_MODEL = "gen_ai.response.model"
_PROVIDER = "gen_ai.provider.name"
_INPUT_MESSAGES = "gen_ai.input.messages"
_OUTPUT_MESSAGES = "gen_ai.output.messages"
_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
_CACHE_WRITE = "gen_ai.usage.cache_creation.input_tokens"

_RETRIEVE_SPAN = "kg_rag.retrieve"
_RERANK_SPAN = "kg_rag.rerank"


def _kind(name: str, attributes: Attributes) -> str:
    """``llm`` / ``embedding`` / ``retriever`` / ``reranker`` / ``chain``."""
    operation = attributes.get(_OPERATION)
    if operation in ("chat", "text_completion", "generate_content"):
        return "llm"
    if operation == "embeddings":
        return "embedding"
    if name == _RETRIEVE_SPAN:
        return "retriever"
    if name == _RERANK_SPAN:
        return "reranker"
    return "chain"


_LANGFUSE_TYPES = {
    "llm": "generation",
    "embedding": "embedding",
    "retriever": "retriever",
    "reranker": "retriever",
    "chain": "span",
}


def langfuse_attributes(name: str, attributes: Attributes) -> dict[str, Any]:
    """Langfuse attributes to add to a span (see the module docstring)."""
    added: dict[str, Any] = {
        "langfuse.observation.type": _LANGFUSE_TYPES[_kind(name, attributes)]
    }
    if _INPUT_MESSAGES in attributes:
        added["langfuse.observation.input"] = attributes[_INPUT_MESSAGES]
    if _OUTPUT_MESSAGES in attributes:
        added["langfuse.observation.output"] = attributes[_OUTPUT_MESSAGES]
    return added


_OPENINFERENCE_KINDS = {
    "llm": "LLM",
    "embedding": "EMBEDDING",
    "retriever": "RETRIEVER",
    "reranker": "RERANKER",
    "chain": "CHAIN",
}


def openinference_attributes(name: str, attributes: Attributes) -> dict[str, Any]:
    """OpenInference (Phoenix) attributes to add to a span."""
    kind = _kind(name, attributes)
    added: dict[str, Any] = {"openinference.span.kind": _OPENINFERENCE_KINDS[kind]}
    model = attributes.get(_RESPONSE_MODEL) or attributes.get(_REQUEST_MODEL)
    if kind == "embedding" and model:
        added["embedding.model_name"] = model
    elif model:
        added["llm.model_name"] = model
    if kind == "llm":
        provider = attributes.get(_PROVIDER)
        if isinstance(provider, str):
            added["llm.provider"] = "aws" if provider.startswith("aws.") else provider
        prompt = attributes.get(_INPUT_TOKENS)
        completion = attributes.get(_OUTPUT_TOKENS)
        if isinstance(prompt, int):
            added["llm.token_count.prompt"] = prompt
        if isinstance(completion, int):
            added["llm.token_count.completion"] = completion
        if isinstance(prompt, int) and isinstance(completion, int):
            added["llm.token_count.total"] = prompt + completion
        if isinstance(cache_read := attributes.get(_CACHE_READ), int):
            added["llm.token_count.prompt_details.cache_read"] = cache_read
        if isinstance(cache_write := attributes.get(_CACHE_WRITE), int):
            added["llm.token_count.prompt_details.cache_write"] = cache_write
    if _INPUT_MESSAGES in attributes:
        added["input.value"] = attributes[_INPUT_MESSAGES]
        added["input.mime_type"] = "application/json"
    if _OUTPUT_MESSAGES in attributes:
        added["output.value"] = attributes[_OUTPUT_MESSAGES]
        added["output.mime_type"] = "application/json"
    return added


VENDOR_MAPPERS: dict[ObservabilityBackend, AttributeMapper] = {
    ObservabilityBackend.LANGFUSE: langfuse_attributes,
    ObservabilityBackend.PHOENIX: openinference_attributes,
}


class Redactor:
    """Replace every pattern match in string values with ``[REDACTED]``."""

    def __init__(self, patterns: Sequence[str | re.Pattern[str]]) -> None:
        self.patterns = tuple(re.compile(p) for p in patterns)

    def __bool__(self) -> bool:
        return bool(self.patterns)

    def text(self, value: str) -> str:
        for pattern in self.patterns:
            value = pattern.sub(REDACTED, value)
        return value

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list | tuple) and any(isinstance(v, str) for v in value):
            return [self.text(v) if isinstance(v, str) else v for v in value]
        return value

    def attributes(self, attributes: Attributes) -> dict[str, Any]:
        return {key: self.value(value) for key, value in attributes.items()}


class TransformingSpanExporter(SpanExporter):
    """Apply a vendor mapping and redaction to spans, then export them."""

    def __init__(
        self,
        exporter: SpanExporter,
        *,
        mapper: AttributeMapper | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.exporter = exporter
        self.mapper = mapper
        self.redactor = redactor if redactor else None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.exporter.export([self.transform(span) for span in spans])

    def transform(self, span: ReadableSpan) -> ReadableSpan:
        if self.mapper is None and self.redactor is None:
            return span
        attributes: dict[str, Any] = dict(span.attributes or {})
        if self.mapper is not None:
            attributes.update(self.mapper(span.name, attributes))
        events: Sequence[Event] = span.events
        status = span.status
        if self.redactor is not None:
            redactor = self.redactor
            attributes = redactor.attributes(attributes)
            events = [
                Event(
                    event.name,
                    redactor.attributes(event.attributes or {}),
                    event.timestamp,
                )
                for event in span.events
            ]
            if status.description:
                status = Status(status.status_code, redactor.text(status.description))
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=events,
            links=span.links,
            kind=span.kind,
            status=status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.exporter.force_flush(timeout_millis)
