# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The output guardrail screens the model-written query texts a caller sees:
query processing's rewrite and translation, the entities and LightRAG
keywords, and the search query built from them; a refused answer keeps none
of them."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.unit.test_structured_search import _Documents
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGOutput
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports.guardrail import GuardAction, GuardSource, GuardVerdict

pytestmark = pytest.mark.unit

_QUERY = "Who supplies the Buyer?"
_HISTORY = [
    {"role": "user", "content": "Tell me about Acme"},
    {"role": "assistant", "content": "Acme is a vendor."},
]


class _Guard:
    """Masks "Secret" in an output, refuses an output naming "Forbidden"
    (and an answer when ``block_answer``); passes every input."""

    def __init__(self, *, block_answer: bool = False) -> None:
        self.block_answer = block_answer
        self.outputs: list[str] = []

    async def acheck(self, text: str, source: GuardSource, **_: Any) -> GuardVerdict:
        if source is GuardSource.INPUT:
            return GuardVerdict(action=GuardAction.NONE, text=text)
        self.outputs.append(text)
        if "Forbidden" in text or (self.block_answer and text.startswith("Acme")):
            return GuardVerdict(action=GuardAction.BLOCKED, text="Refused.")
        if "Secret" in text:
            return GuardVerdict(
                action=GuardAction.MASKED, text=text.replace("Secret", "{NAME}")
            )
        return GuardVerdict(action=GuardAction.NONE, text=text)


def _respond(system: str, _human: str) -> str:
    head = system[:200].lower()
    if "keyword" in head:
        return json.dumps(
            {
                "high_level_keywords": ["Secret supply theme", "Forbidden theme"],
                "low_level_keywords": ["Secret Vendor", "Buyer"],
            }
        )
    if "rewrite" in head:
        return json.dumps({"question": "Who supplies the Secret Buyer?"})
    return "Acme supplies the Buyer [1]."


async def _ask(guard: _Guard, strategy: str) -> RAGOutput:
    config = Config()
    config.aws.bedrock.guardrail.identifier = "test-guardrail"
    config.providers.rerank.kind = "none"
    chain = GraphRAGChain(
        config=config,
        model_factory=ScriptedLLMFactory(_respond),
        retriever_builders={
            "document": lambda: _Documents(),
            "graph": lambda: _Documents(),
        },
        guard=guard,  # type: ignore[arg-type]
    )
    try:
        output = await chain.ainvoke(
            {
                "query": _QUERY,
                "search_strategy": strategy,
                "enable_query_processing": True,
                "history": _HISTORY,
            }
        )
    finally:
        await chain.aclose()
    assert isinstance(output, RAGOutput)
    return output


async def test_the_rewrite_and_keywords_are_screened_before_the_caller_sees_them() -> (
    None
):
    output = await _ask(_Guard(), "mix")

    processed, searched = output.processed_query, output.search_results.query
    assert processed.original_query == _QUERY
    assert processed.standalone_query == "Who supplies the {NAME} Buyer?"
    assert processed.final_query == "Who supplies the {NAME} Buyer?"
    assert processed.hl_keywords == ["{NAME} supply theme"]  # the refused one left out
    assert processed.ll_keywords == ["{NAME} Vendor", "Buyer"]
    assert searched.query == "Who supplies the {NAME} Buyer?"
    assert searched.hl_keywords == ["{NAME} supply theme"]
    assert searched.ll_keywords == ["{NAME} Vendor", "Buyer"]
    assert "Secret" not in output.model_dump_json()
    assert "Forbidden" not in output.model_dump_json()
    records = [r for r in output.metadata["guardrail"] if r["source"] == "output"]
    assert {r["action"] for r in records} == {"masked", "blocked"}


async def test_a_refused_answer_keeps_no_model_reading_of_the_question() -> None:
    output = await _ask(_Guard(block_answer=True), "mix")

    assert output.answer == "Refused."
    processed, searched = output.processed_query, output.search_results.query
    assert processed.final_query == _QUERY and processed.standalone_query is None
    assert processed.hl_keywords == processed.ll_keywords == processed.entities == []
    assert searched.query == _QUERY
    assert searched.hl_keywords == searched.ll_keywords == searched.entity_focus == []


async def test_clean_query_texts_cost_one_check() -> None:
    guard = _Guard()
    chain = GraphRAGChain(
        config=Config(), model_factory=ScriptedLLMFactory(_respond), guard=guard  # type: ignore[arg-type]
    )
    chain._screen_outputs = True
    output = RAGOutput.model_validate(
        {
            "answer": "x",
            "sources": [],
            "conversation_id": None,
            "processed_query": {
                "original_query": _QUERY,
                "final_query": "Who supplies Buyer?",
                "hl_keywords": ["supply"],
            },
            "search_results": {
                "query": {"query": "Who supplies Buyer?", "entity_focus": ["Buyer"]},
                "results": [],
                "total_results": 0,
                "search_strategy": "mix",
                "processing_time": 0.0,
            },
        }
    )
    before = output.model_dump()

    await chain._screen_query_texts(output, [])

    assert guard.outputs == ["Who supplies Buyer?\nsupply\nBuyer"]
    assert output.model_dump() == before
