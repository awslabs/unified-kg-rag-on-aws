# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-output guards in ``setup_chain`` (AWS-free).

A response that stopped at its output-token limit is cut mid-answer; parsing
it would silently keep whatever came before the cut (e.g. entities without
the ``<relationships>`` section). The chain fails such a response before the
parser runs. The fake model reports the stop reason the way langchain-aws
does: ``response_metadata["stopReason"]`` (Converse) or ``["stop_reason"]``
(InvokeModel).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.domain.models import ModelPurpose
from unified_kg_rag.domain.prompts import GraphExtractionPrompt
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError
from unified_kg_rag.shared.utils.langchain import (
    BATCH_ITEM_FAILED,
    BatchProcessor,
    RobustXMLOutputParser,
)

pytestmark = pytest.mark.unit

_PARTIAL = (
    "<entities>\n<entity><name>Vendor</name><type>ORG</type></entity>\n"
    "</entities>\n<relationships>\n<relationship><source>Vendor"
)
_COMPLETE = (
    "<entities>\n<entity><name>Vendor</name><type>ORG</type></entity>\n"
    "</entities>\n<relationships>\n</relationships>"
)


class _StopReasonModel(BaseChatModel):
    text: str
    metadata: dict[str, Any]

    @property
    def _llm_type(self) -> str:
        return "stop-reason"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = AIMessage(content=self.text, response_metadata=dict(self.metadata))
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for word in self.text.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata=dict(self.metadata))
        )


class _Factory:
    def __init__(self, text: str, metadata: dict[str, Any]) -> None:
        self.text = text
        self.metadata = metadata

    def get_model(self, model_id: Any, **kwargs: Any) -> _StopReasonModel:
        return _StopReasonModel(text=self.text, metadata=self.metadata)

    def get_model_info(self, model_id: Any) -> None:
        return None


class _RecordingParser(RobustXMLOutputParser):
    calls: int = 0

    def parse(self, text: str) -> dict[str, Any]:
        type(self).calls += 1
        return super().parse(text)


_INPUT = {
    "input_text": "Vendor ships parts to Buyer.",
    "max_entities_per_chunk": 10,
    "max_relationships_per_chunk": 10,
    "entity_types": "ORG",
}


def _chain(text: str, metadata: dict[str, Any], parser: Any = None) -> Any:
    return setup_chain(
        factory=_Factory(text, metadata),
        model_id="fake-model",
        prompt_class=GraphExtractionPrompt,
        parser=parser or RobustXMLOutputParser(),
        model_purpose=ModelPurpose.INGESTION,
    )


@pytest.mark.parametrize(
    "metadata", [{"stopReason": "max_tokens"}, {"stop_reason": "max_tokens"}]
)
def test_truncated_response_fails_before_parsing(metadata, caplog) -> None:
    _RecordingParser.calls = 0
    chain = _chain(_PARTIAL, metadata, parser=_RecordingParser())
    with caplog.at_level(logging.WARNING), pytest.raises(LLMOutputTruncatedError):
        chain.invoke(_INPUT)
    assert _RecordingParser.calls == 0
    assert "GraphExtractionPrompt" in caplog.text
    assert "max_tokens" in caplog.text


async def test_truncated_response_fails_async() -> None:
    chain = _chain(_PARTIAL, {"stopReason": "max_tokens"})
    with pytest.raises(LLMOutputTruncatedError):
        await chain.ainvoke(_INPUT)


@pytest.mark.parametrize(
    "metadata", [{"stopReason": "end_turn"}, {"stop_reason": "end_turn"}, {}]
)
def test_complete_response_parses(metadata) -> None:
    out = _chain(_COMPLETE, metadata).invoke(_INPUT)
    assert out == {
        "entities": {"entity": {"name": "Vendor", "type": "ORG"}},
        "relationships": {},
    }


def test_truncated_item_is_a_batch_failure() -> None:
    chain = _chain(_PARTIAL, {"stopReason": "max_tokens"})
    results = BatchProcessor(max_attempts=1).execute_with_fallback(
        items_to_process=[_INPUT],
        prepare_inputs_func=lambda items: list(items),
        batch_func=chain.batch,
        sequential_func=chain.invoke,
        task_name="extraction",
        show_progress=False,
    )
    assert results == [BATCH_ITEM_FAILED]


def test_streaming_passes_chunks_through_and_warns(caplog) -> None:
    # Streamed output has already reached the caller when the stop reason
    # arrives, so it is logged instead of raised.
    chain = setup_chain(
        factory=_Factory("Vendor ships parts", {"stopReason": "max_tokens"}),
        model_id="fake-model",
        prompt_class=GraphExtractionPrompt,
        parser=StrOutputParser(),
        model_purpose=ModelPurpose.INGESTION,
    )
    with caplog.at_level(logging.WARNING):
        chunks = list(chain.stream(_INPUT))
    assert len(chunks) > 1
    assert "".join(chunks) == "Vendor ships parts "
    assert "max_tokens" in caplog.text
