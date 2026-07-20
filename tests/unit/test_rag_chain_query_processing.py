# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query-processing flow tests for GraphRAGChain (AWS-free).

Covers _process_query_step end-to-end — the node that composes translation +
entity extraction with the same-language skip, the translation-refusal fallback,
and the ignore_errors degrade — which was previously only tested via its
isolated building blocks. Also covers the top-level ainvoke error contract.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401 (registers strategies)
from unified_kg_rag.application.retrieval.rag_chain import (
    EntityExtractionPrompt,
    GraphRAGChain,
    TranslationPrompt,
)
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


class _FakeChain:
    """A stand-in for a prompt|llm|parser Runnable with an async invoke."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, _inputs):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _chain(config: Config) -> GraphRAGChain:
    return GraphRAGChain(config=config)


def _install_fakes(chain: GraphRAGChain, *, translation, entities) -> None:
    def fake_get_chain(prompt_class, parser, **kwargs):
        if prompt_class is TranslationPrompt:
            return _FakeChain(translation)
        if prompt_class is EntityExtractionPrompt:
            return _FakeChain(entities)
        return _FakeChain("")

    chain._get_chain_for_prompt = fake_get_chain  # type: ignore[assignment]


async def test_same_language_skips_translation(config: Config) -> None:
    # EN corpus, EN target (is_noop) and no explicit target => translator is
    # never called; final_query stays the original.
    config.processing.translation.source_language = (
        config.processing.translation.target_language
    )
    chain = _chain(config)
    # If the translator WERE called it would raise, proving it was skipped.
    _install_fakes(
        chain, translation=RuntimeError("translator must not run"), entities=["oda"]
    )
    result = await chain._process_query_step({"query": "Tell me about Oda"})
    assert result.translated_query is None
    assert result.final_query == "Tell me about Oda"
    assert result.entities == ["oda"]


async def test_translation_refusal_falls_back_to_original(config: Config) -> None:
    # Force a non-noop config so translation runs, but the LLM returns a refusal.
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    _install_fakes(
        chain,
        translation="I notice the text you provided is already in English.",
        entities=["nobunaga"],
    )
    result = await chain._process_query_step({"query": "織田信長について"})
    # The refusal is rejected; search uses the original query, not the meta-text.
    assert result.translated_query is None
    assert result.final_query == "織田信長について"


async def test_genuine_translation_is_used(config: Config) -> None:
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    _install_fakes(chain, translation="About Oda Nobunaga", entities=["nobunaga"])
    result = await chain._process_query_step({"query": "織田信長について"})
    assert result.translated_query == "About Oda Nobunaga"
    assert result.final_query == "About Oda Nobunaga"


async def test_ignore_errors_degrades_to_original_query(config: Config) -> None:
    config.processing.ignore_errors = True
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    _install_fakes(
        chain,
        translation=RuntimeError("bedrock down"),
        entities=RuntimeError("bedrock down"),
    )
    result = await chain._process_query_step({"query": "織田信長"})
    # Both sub-tasks failed but ignore_errors keeps the query flowing.
    assert result.final_query == "織田信長"
    assert result.entities == []
