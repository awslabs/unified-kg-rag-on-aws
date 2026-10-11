# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry GenAI semantic-convention names and the embedding-call span.

The attribute names every model adapter, the LangChain callback and the
export-time vendor mapping share, so a renamed convention changes in one
place. Dependency-free (only the ``shared.observability`` facade): importing it
loads neither OpenTelemetry nor LangChain.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from unified_kg_rag.shared import observability
from unified_kg_rag.shared.observability import SpanHandle

GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
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

PROVIDER_AWS_BEDROCK = "aws.bedrock"


def embedding_span(
    provider_name: str, model_id: str, *, texts: int, characters: int
) -> AbstractContextManager[SpanHandle]:
    """One GenAI ``embeddings`` client span (counts only, no text content)."""
    return observability.span(
        f"embeddings {model_id}",
        {
            GEN_AI_OPERATION_NAME: "embeddings",
            GEN_AI_PROVIDER_NAME: provider_name,
            GEN_AI_REQUEST_MODEL: model_id,
        },
        kind="client",
        texts=texts,
        characters=characters,
    )
