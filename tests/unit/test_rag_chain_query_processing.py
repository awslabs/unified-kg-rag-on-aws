# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query-processing flow tests for GraphRAGChain (AWS-free).

Covers _process_query_step end-to-end — the node that composes translation +
entity extraction with the same-language skip, the translation-refusal fallback,
and the ignore_errors degrade — which was previously only tested via its
isolated building blocks. Also covers the top-level ainvoke error contract.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401 (registers strategies)
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.retrieval.base import degradation_scope
from unified_kg_rag.application.retrieval import query_processing
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
)
from unified_kg_rag.domain.models import Config, SearchStrategy
from unified_kg_rag.domain.prompts import (
    EntityExtractionPrompt,
    TranslationPrompt,
)

pytestmark = pytest.mark.unit


class _FakeChain:
    """A stand-in for a prompt|llm|parser Runnable with an async invoke."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, _inputs, config=None):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _chain(config: Config) -> GraphRAGChain:
    chain = GraphRAGChain(config=config)
    # The output fixer's model: a refusal is the sentinel (as it is told).
    chain.factory = ScriptedLLMFactory(
        lambda _system, _human: '{"translation": "[CANNOT_TRANSLATE]"}'
    )
    return chain


def _install_fakes(chain: GraphRAGChain, *, translation, entities) -> None:
    def fake_get_chain(prompt_class, parser, **kwargs):
        if prompt_class is TranslationPrompt:
            # The translation answers JSON (a refusal is prose: repaired).
            return _FakeChain(
                json.dumps({"translation": translation})
                if isinstance(translation, str) and not translation.startswith("I ")
                else translation
            )
        if prompt_class is EntityExtractionPrompt:
            # The model's text answer, split by split_entity_list.
            return _FakeChain(
                ", ".join(entities) if isinstance(entities, list) else entities
            )
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


@pytest.mark.parametrize("requested", [None, "en"])
async def test_disabled_translation_skips_the_query_translation(
    config: Config, requested: str | None
) -> None:
    # A ja corpus indexed untranslated (translation disabled) was searched
    # with the query translated into English, unless a caller asks for a
    # language.
    config.processing.translation.enabled = False
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    _install_fakes(chain, translation="About Oda Nobunaga", entities=["nobunaga"])
    inputs = {"query": "織田信長について"}
    if requested:
        inputs["target_language"] = requested
    result = await chain._process_query_step(inputs)
    expected = "About Oda Nobunaga" if requested else "織田信長について"
    assert result.final_query == expected


@pytest.mark.parametrize(
    ("enabled", "requested", "expected"),
    [
        # Disabled: the index keeps the source text and entity names, so the
        # extractions run in the source language, not the unused target.
        (False, None, "English"),
        # A requested language is the translation's only: the entity,
        # relationship and report indexes hold the index language alone.
        (False, "ja", "English"),
        # Enabled: the index is in the target language.
        (True, None, "Korean"),
        (True, "ja", "Korean"),
    ],
)
async def test_query_extractions_use_the_language_the_index_was_built_in(
    config: Config, enabled: bool, requested: str | None, expected: str
) -> None:
    config.processing.translation.enabled = enabled
    config.processing.translation.source_language = "en"
    config.processing.translation.target_language = "ko"
    chain = _chain(config)
    seen: dict[str, str] = {}

    async def entities(query, language, config=None):
        seen["entities"] = language
        return []

    async def keywords(query, language, config=None):
        seen["keywords"] = language
        return None

    async def translate(query, language, config=None):
        seen["translation"] = language
        return query

    chain._extract_query_entities = entities  # type: ignore[method-assign]
    chain._extract_dual_keywords = keywords  # type: ignore[method-assign]
    chain._translate_query = translate  # type: ignore[method-assign]
    inputs = {"query": "Where is Vendor based?"}
    if requested:
        inputs["target_language"] = requested
    tasks = chain._start_query_inputs(inputs, entities=True, keywords=True, config=None)
    await asyncio.gather(*tasks.values())

    assert seen["entities"] == seen["keywords"] == expected
    # Translation runs only when the stage ran or a language was requested,
    # into the requested language (the text units carry it).
    assert ("translation" in seen) is (enabled or requested is not None)
    if requested:
        assert seen["translation"] == "Japanese"
    # A strategy's own prompts (DRIFT follow-ups, iterative's controller)
    # search the indexes too: the index language.
    assert query_processing.query_language(chain, inputs) == expected


@pytest.mark.parametrize(
    ("enabled", "query", "requested", "translated_into"),
    [
        # Regression (rv23): a question in another language than the index
        # was searched as written (BM25 on Korean text over English units).
        (False, "공급사는 어디에 있나요?", None, "English"),
        (False, "供給者はどこにありますか？", None, "English"),
        # The opposite: after a ko->en ingest an English question paid a
        # translator call into English.
        (True, "Where is the vendor based?", None, None),
        (True, "공급사는 어디에 있나요?", None, "English"),
        # A requested language the query is already written in.
        (False, "공급사는 어디에 있나요?", "ko", None),
        (False, "Where is the vendor based?", "ko", "Korean"),
    ],
)
async def test_a_query_is_translated_exactly_when_its_language_differs(
    config: Config,
    enabled: bool,
    query: str,
    requested: str | None,
    translated_into: str | None,
) -> None:
    config.processing.translation.enabled = enabled
    config.processing.translation.source_language = "ko" if enabled else "en"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    seen: list[str] = []

    async def translate(query, language, config=None):
        seen.append(language)
        return f"<{language}> {query}"

    chain._translate_query = translate  # type: ignore[method-assign]
    inputs = {"query": query}
    if requested:
        inputs["target_language"] = requested
    tasks = chain._start_query_inputs(
        inputs, entities=False, keywords=False, config=None
    )
    await asyncio.gather(*tasks.values())
    assert seen == ([translated_into] if translated_into else [])


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
    # The prose goes to the fixer, which answers the sentinel; search uses
    # the original query, not the meta-text.
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


@pytest.mark.parametrize(
    ("translation", "entities", "degraded"),
    [
        ("About Oda", ["oda"], None),
        (RuntimeError("bedrock down"), ["oda"], "RuntimeError"),
        ("About Oda", TimeoutError("slow"), "TimeoutError"),
        # A translation refusal is a deliberate fallback, no failure.
        ("I notice the text is already in English.", ["oda"], None),
    ],
)
async def test_an_absorbed_query_processing_failure_degrades_the_answer(
    config: Config, translation, entities, degraded
) -> None:
    # Under ignore_errors a failed translation or extraction searched the
    # untranslated question or without entities, and left no marker: the
    # answer counted as undegraded in run-eval's failure budget.
    config.processing.ignore_errors = True
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    _install_fakes(chain, translation=translation, entities=entities)
    with degradation_scope() as ledger:
        await chain._process_query_step({"query": "織田信長について"})
    assert ledger.first_error_type == degraded


async def test_an_absorbed_keyword_failure_degrades_the_answer(
    config: Config,
) -> None:
    config.processing.ignore_errors = True
    chain = _chain(config)

    async def keywords_fail(*_args, **_kwargs):
        raise ValueError("no JSON")

    def fake_get_chain(prompt_class, parser, **kwargs):
        return _FakeChain(ValueError("no JSON"))

    chain._get_chain_for_prompt = fake_get_chain  # type: ignore[assignment]
    keywords = await chain._extract_dual_keywords("q", "English")
    assert keywords == ([], [])  # the contract callers rely on
    with degradation_scope() as ledger:
        result = await chain._process_query_step(
            {"query": "Who supplies Buyer A?", "search_strategy": "mix"}
        )
    assert result.hl_keywords == result.ll_keywords == []
    assert ledger.first_error_type is not None


async def test_a_failed_query_processing_step_degrades_the_answer(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.processing.ignore_errors = True
    chain = _chain(config)

    async def broken(*_args, **_kwargs):
        raise KeyError("state")

    monkeypatch.setattr(query_processing, "process_query", broken)
    with degradation_scope() as ledger:
        result = await chain._process_query_step({"query": "q"})
    assert result.final_query == "q" and ledger.first_error_type == "KeyError"
    assert [e.step for e in ledger.entries] == ["query_processing"]


@pytest.mark.parametrize(
    ("translation", "entities"),
    [
        (httpx.ConnectError("All connection attempts failed"), ["oda"]),
        ("About Oda", ConnectionRefusedError(111, "Connection refused")),
    ],
    ids=["translation", "entities"],
)
async def test_an_unreachable_model_server_is_never_absorbed(
    config: Config, translation, entities
) -> None:
    # Regression (rv7 features M1): ignore_errors searched without the
    # failed call, and every later call met the same server: the query
    # fails at once, naming it.
    config.processing.ignore_errors = True
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    _install_fakes(chain, translation=translation, entities=entities)
    with pytest.raises((httpx.ConnectError, ConnectionRefusedError)):
        await chain._process_query_step({"query": "織田信長について"})

    chain._get_chain_for_prompt = (  # type: ignore[assignment]
        lambda *a, **k: _FakeChain(httpx.ConnectError("refused"))
    )
    with pytest.raises(httpx.ConnectError):
        await chain._extract_dual_keywords("q", "English")


async def test_a_translation_the_fixer_cannot_repair_is_degraded(
    config: Config,
) -> None:
    # Not a failure of the query, under ignore_errors=False too: the query
    # is searched as written and the answer is marked degraded.
    config.processing.ignore_errors = False
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    chain.factory = ScriptedLLMFactory(lambda _system, _human: "still not JSON")
    _install_fakes(chain, translation="I cannot do that.", entities=["nobunaga"])
    with degradation_scope() as ledger:
        result = await chain._process_query_step({"query": "織田信長について"})
    assert result.final_query == "織田信長について"
    assert ledger.first_error_type == "OutputParserException"


async def test_unreadable_answers_are_recorded_in_the_query_ledger(
    config: Config,
) -> None:
    # The translation and the follow-up rewrite: one repair, then a
    # degraded step with the parse error, in the query's ledger.
    from unified_kg_rag.adapters.retrieval.base import degradation_scope

    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    chain.factory = ScriptedLLMFactory(lambda _system, _human: "still not JSON")
    _install_fakes(chain, translation="I cannot do that.", entities=["nobunaga"])
    with degradation_scope() as ledger:
        await chain._process_query_step({"query": "織田信長について"})
        assert await query_processing.read_rewrite(chain, "Sorry, no.") is None
    assert [(e.step, e.error_type) for e in ledger.entries] == [
        ("translation", "OutputParserException"),
        ("follow_up_rewrite", "OutputParserException"),
    ]


async def test_an_unreadable_keyword_answer_is_a_recorded_degradation(
    config: Config,
) -> None:
    # Regression (rv8 retrieval M3): under ignore_errors=False the whole
    # mix query failed; now the step is in the ledger and mix searches the
    # raw query (LightRAGSearchStrategy._apply_keyword_fallback).
    from unified_kg_rag.adapters.retrieval.base import degradation_scope

    config.processing.ignore_errors = False
    chain = _chain(config)
    chain.factory = ScriptedLLMFactory(lambda _system, _human: "still not JSON")
    chain._get_chain_for_prompt = (  # type: ignore[assignment]
        lambda *a, **k: _FakeChain("Sorry, no keywords today.")
    )
    with degradation_scope() as ledger:
        result = await chain._process_query_step(
            {"query": "Who supplies Buyer A?", "resolved_strategy": SearchStrategy.MIX}
        )
    assert result.hl_keywords == result.ll_keywords == []
    assert [(e.step, e.error_type) for e in ledger.entries] == [
        ("keyword_extraction", "OutputParserException")
    ]
    assert ledger.entries[0].reason == (
        "keyword answer unreadable; searched the raw query as its low-level keyword"
    )


async def test_the_keyword_degradation_says_when_no_fallback_ran(
    config: Config,
) -> None:
    # Regression (rv9 query M2): the reason (and the WARNING) said the raw
    # query was searched when it was too long for the fallback.
    from unified_kg_rag.adapters.retrieval.base import degradation_scope

    config.processing.ignore_errors = False
    chain = _chain(config)
    chain.factory = ScriptedLLMFactory(lambda _system, _human: "still not JSON")
    chain._get_chain_for_prompt = (  # type: ignore[assignment]
        lambda *a, **k: _FakeChain("Sorry, no keywords today.")
    )
    question = "Which suppliers of Buyer A delivered parts to the warehouse in 2024?"
    with degradation_scope() as ledger:
        await chain._process_query_step(
            {"query": question, "resolved_strategy": SearchStrategy.MIX}
        )
    assert [e.reason for e in ledger.entries] == [
        f"keyword answer unreadable; no raw-query fallback ({len(question)} "
        "characters, limit 50): no entities or relationships searched"
    ]


async def test_a_failed_keyword_call_still_fails_without_ignore_errors(
    config: Config,
) -> None:
    # Only an unreadable answer degrades; a failed call obeys ignore_errors.
    config.processing.ignore_errors = False
    chain = _chain(config)
    chain._get_chain_for_prompt = (  # type: ignore[assignment]
        lambda *a, **k: _FakeChain(RuntimeError("model overloaded"))
    )
    with pytest.raises(RuntimeError, match="overloaded"):
        await chain._extract_dual_keywords("q", "English")


async def test_an_unreachable_fixer_is_never_absorbed(config: Config) -> None:
    # The repair call meets the same unreachable server: the query fails,
    # under ignore_errors too, instead of degrading.
    def unreachable(_system: str, _human: str) -> str:
        raise httpx.ConnectError("All connection attempts failed")

    config.processing.ignore_errors = True
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    chain = _chain(config)
    chain.factory = ScriptedLLMFactory(unreachable)
    _install_fakes(chain, translation="I cannot do that.", entities=["nobunaga"])
    with pytest.raises(httpx.ConnectError):
        await chain._process_query_step({"query": "織田信長について"})
    with pytest.raises(httpx.ConnectError):
        await query_processing.read_rewrite(chain, "Sorry, no.")
