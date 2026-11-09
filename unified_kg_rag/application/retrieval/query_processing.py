# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query processing of ``GraphRAGChain``: the translation of a query and
the entity and keyword extractions its strategy reads, the extraction
cache, and the reading of the translation's and the follow-up rewrite's
JSON answers (a sentinel, not a refusal phrase, says the model could not).

The functions take the chain they process for: it holds the config, the
prompt chains (``_get_chain_for_prompt``) and the extraction cache, and it
keeps a delegating method for each public function here, which these call
through (a test or a subclass replaces a step on the chain).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from unified_kg_rag.adapters.llm.chain_factory import usage_model_id
from unified_kg_rag.adapters.llm.reachability import is_unreachable_error
from unified_kg_rag.adapters.retrieval.base import record_degradation
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    QueryInput,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    query_inputs as strategy_query_inputs,
)
from unified_kg_rag.domain.ingestion.cache_keys import index_language
from unified_kg_rag.domain.models import (
    Config,
    ModelPurpose,
    ModelRole,
    language_name,
)
from unified_kg_rag.domain.prompts import (
    BasePrompt,
    EntityExtractionPrompt,
    KeywordsExtractionPrompt,
    QueryRewritePrompt,
    TranslationPrompt,
)
from unified_kg_rag.domain.prompts.retrieval import (
    CANNOT_REWRITE_MARKER,
    CANNOT_TRANSLATE_MARKER,
)
from unified_kg_rag.domain.retrieval.degradation import DegradationStep
from unified_kg_rag.domain.retrieval.keyword_fallback import fallback_outcome
from unified_kg_rag.domain.retrieval.query_entities import split_entity_list
from unified_kg_rag.shared import LanguageModelError, get_logger
from unified_kg_rag.shared.usage import record_cache_hit
from unified_kg_rag.shared.utils import text_digest

if TYPE_CHECKING:
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        _ExtractionCache,
    )

logger = get_logger(__name__)

# The two keyword lists of the keyword extraction's answer.
_KEYWORD_LEVELS: tuple[str, str] = (
    "high_level_keywords",
    "low_level_keywords",
)


class _TranslationAnswer(BaseModel):
    """The query translation's answer (``TranslationPrompt``)."""

    translation: str


class _RewriteAnswer(BaseModel):
    """The follow-up rewrite's answer (``QueryRewritePrompt``)."""

    question: str


# What the output fixer is told besides the JSON shape: a refusal is the
# sentinel, never an answer dressed as one.
_TRANSLATION_FIX = (
    "If the completion is not a translation (a refusal, a comment about the "
    f'request), return {{"translation": "{CANNOT_TRANSLATE_MARKER}"}}.'
)
_REWRITE_FIX = (
    "If the completion is not a rewritten question (a refusal, a request "
    f'for clarification), return {{"question": "{CANNOT_REWRITE_MARKER}"}}.'
)


def _query_language(chain: GraphRAGChain, inputs: dict[str, Any]) -> tuple[str, bool]:
    """The language the query is processed into, and whether its
    translation is skipped.

    Translation is skipped when no explicit target was requested AND
    ingestion did not translate: the translation stage is disabled (the
    corpus is indexed as written) or would be a no-op (a same-language
    corpus, TranslationConfig.is_noop). This avoids paying an LLM
    call per query for e.g. an English-only or Japanese-only corpus, and
    — more importantly — avoids the failure mode where the translator
    LLM returns a meta-response ("I notice the text you...") for a query
    already in the target language, which would then be used verbatim as
    the search query.

    The language is the requested one, else the one the index was built
    in (``index_language``): the source language when the stage is
    disabled, the target language otherwise (equal to the source for a
    no-op). It is the translation's language only: the query's entities
    and keywords are extracted in the index language whatever was
    requested (``_extraction_language``).
    """
    requested_language = inputs.get("target_language")
    translation = chain.config.processing.translation
    skip_translation = requested_language is None and (
        not translation.enabled or translation.is_noop
    )
    language = requested_language or index_language(chain.config)
    return language_name(language), skip_translation


def _extraction_language(chain: GraphRAGChain) -> str:
    """The language (its name) the query's entities and keywords are
    extracted in: the index's (``index_language``), not a requested one.

    They search the entity, relationship, report and claim indexes, which
    hold the index language only (just the text units carry a
    ``translated_text_<language>`` per additional target), so an entity
    extracted in a requested language would match neither their BM25 nor
    their ``compact_name_key`` seeds. The requested language reaches the
    text units through the translated query and the answer through the
    query itself (the answer prompt writes in the query's language).
    """
    return language_name(index_language(chain.config))


def start_query_inputs(
    chain: GraphRAGChain,
    inputs: dict[str, Any],
    *,
    entities: bool,
    keywords: bool,
    translation: bool = True,
    config: RunnableConfig | None,
) -> dict[str, asyncio.Future[Any]]:
    """Start the query processing calls as tasks: the entity and keyword
    extractions asked for, and the translation (``translation``) unless
    the query needs none (``_query_language``).

    Their errors surface when they are awaited (``process_query``).
    """
    question = chain._question(inputs)
    target_language, skip_translation = _query_language(chain, inputs)
    extraction_language = _extraction_language(chain)
    coroutines: dict[str, Awaitable[Any]] = {}
    if entities:
        coroutines["entities"] = chain._extract_query_entities(
            question, extraction_language, config
        )
    # The keyword prompt writes its keywords in the index language itself,
    # so it reads the untranslated question and runs alongside the
    # translation instead of after it.
    if keywords:
        coroutines["keywords"] = chain._extract_dual_keywords(
            question, extraction_language, config
        )
    if translation and not skip_translation:
        coroutines["translation"] = chain._translate_query(
            question, target_language, config
        )
    return {
        name: asyncio.ensure_future(coroutine) for name, coroutine in coroutines.items()
    }


async def translate_query(
    chain: GraphRAGChain,
    query: str,
    target_language: str,
    config: RunnableConfig | None,
) -> str | None:
    """The query in ``target_language``, or None when the model answered
    that it cannot translate it (``CANNOT_TRANSLATE_MARKER``).

    Raises ``OutputParserException`` when the answer is no translation
    JSON even after one repair (``process_query`` marks it degraded).
    """
    translator = chain._get_chain_for_prompt(TranslationPrompt, StrOutputParser())
    raw = str(
        await translator.ainvoke(
            {"query": query, "target_language": target_language}, config
        )
    )
    if CANNOT_TRANSLATE_MARKER in raw:
        return None
    answer = await chain._json_parser(
        TranslationPrompt, schema=_TranslationAnswer, instructions=_TRANSLATION_FIX
    ).aparse(raw)
    text = str(answer.translation).strip()
    return None if not text or CANNOT_TRANSLATE_MARKER in text else text


async def process_query(
    chain: GraphRAGChain,
    inputs: dict[str, Any],
    question: str,
    config: RunnableConfig | None,
    prefetched: dict[str, asyncio.Future[Any]],
) -> dict[str, Any]:
    """The ``ProcessedQuery`` fields query processing gives ``question``:
    the translation and what the resolved strategy reads.

    A call already started for this query (``prefetched``, auto's
    speculative extractions) is awaited instead of started again; it is
    taken out of ``prefetched``.
    """
    # The strategy is resolved (AUTO routing included) before this step,
    # so only pay for the LLM extractions the resolved strategy reads.
    needs = {
        "entities": needs_query_entities(inputs, chain.config),
        "keywords": needs_dual_keywords(inputs, chain.config),
    }
    tasks: dict[str, Any] = {
        name: prefetched.pop(name)
        for name in ("entities", "keywords", "translation")
        if name in prefetched and needs.get(name, True)
    }
    tasks.update(
        chain._start_query_inputs(
            inputs,
            entities=needs["entities"] and "entities" not in tasks,
            keywords=needs["keywords"] and "keywords" not in tasks,
            translation="translation" not in tasks,
            config=config,
        )
    )

    results = (
        await asyncio.gather(*tasks.values(), return_exceptions=True) if tasks else []
    )
    results_map = dict(zip(tasks.keys(), results, strict=True))
    # A failure ignore_errors absorbed degrades the answer (searched
    # untranslated, or without entities or keywords): the query's ledger.
    translated_query = results_map.get("translation")
    if isinstance(translated_query, OutputParserException):
        # No translation JSON even repaired: search the query as written
        # rather than the model's prose, and say the answer is degraded.
        logger.warning(
            "Query translation answer is not a translation; searching the "
            "query as written"
        )
        record_degradation(
            DegradationStep.TRANSLATION,
            "translation answer unreadable, searched untranslated",
            translated_query,
        )
        translated_query = None
    elif isinstance(translated_query, Exception):
        if not chain.ignore_errors or is_unreachable_error(translated_query):
            raise translated_query
        logger.warning("Query translation failed: %s", translated_query)
        record_degradation(
            DegradationStep.TRANSLATION, "searched untranslated", translated_query
        )
        translated_query = None
    elif "translation" in results_map and translated_query is None:
        logger.info(
            "The translator could not translate the query; searching it as written"
        )

    entity_data = results_map.get("entities", [])
    if not isinstance(entity_data, list):
        if isinstance(entity_data, Exception):
            if chain.ignore_errors and not is_unreachable_error(entity_data):
                logger.warning("Entity extraction failed: %s", entity_data)
                record_degradation(
                    DegradationStep.ENTITY_EXTRACTION,
                    "searched without query entities",
                    entity_data,
                )
            else:
                raise entity_data
        entity_data = []

    hl_keywords: list[str] = []
    ll_keywords: list[str] = []
    if "keywords" in results_map:
        keyword_data = results_map["keywords"]
        if isinstance(keyword_data, BaseException):
            # extract_dual_keywords already degrades to ([], []) what does
            # not fail the query, so an exception here must propagate.
            raise keyword_data
        if isinstance(keyword_data, AbsorbedKeywordFailure):
            # Whether mix and hybrid searched the raw query then (their
            # fallback's one gate) or no entity at all.
            outcome = fallback_outcome(
                str(translated_query or question),
                chain.config.search.lightrag_search.raw_query_fallback_max_len,
            )
            record_degradation(
                DegradationStep.KEYWORD_EXTRACTION,
                f"{keyword_data.reason}; {outcome}",
                keyword_data.error_type,
            )
        hl_keywords, ll_keywords = keyword_data

    return {
        "translated_query": translated_query,
        "final_query": translated_query or question,
        "entities": entity_data,
        "hl_keywords": hl_keywords,
        "ll_keywords": ll_keywords,
    }


def needs_query_entities(state: dict[str, Any], config: Config) -> bool:
    """Whether the resolved strategy reads the query's entities (under
    ``config``, a wrapper reads what it wraps: ``iterative``'s base)."""
    # Without a resolved strategy (direct step invocation) stay conservative
    # and extract, matching the pre-gating behavior.
    strategy = state.get("resolved_strategy")
    return strategy is None or QueryInput.ENTITIES in strategy_query_inputs(
        strategy, config
    )


def needs_dual_keywords(state: dict[str, Any], config: Config) -> bool:
    """Whether the resolved strategy reads the dual-level keywords."""
    strategy = state.get("resolved_strategy")
    return strategy is not None and QueryInput.DUAL_KEYWORDS in (
        strategy_query_inputs(strategy, config)
    )


def query_language(chain: GraphRAGChain, inputs: dict[str, Any]) -> str:
    """The language (its name, ``"Korean"``) a strategy's own prompts write
    in (``SearchQuery.metadata["query_language"]``): the index's, as the
    query's entities and keywords (``_extraction_language``), whatever
    language was requested. Their output either searches the indexes
    (DRIFT's and iterative's follow-up queries, iterative's bridge names)
    or is context for the answer model (DRIFT's intermediate answers,
    global's map points), which answers in the query's language itself."""
    return _extraction_language(chain)


async def extract_query_entities(
    chain: GraphRAGChain,
    query: str,
    target_language: str,
    config: RunnableConfig | None,
) -> list[str]:
    """The entities the query names (entity extraction), from the
    extraction cache when this question was extracted before."""
    key = chain._extraction_key(EntityExtractionPrompt, query, target_language)

    async def extract() -> tuple[str, ...]:
        extractor = chain._get_chain_for_prompt(
            EntityExtractionPrompt,
            StrOutputParser(),
            output_ceiling=chain.config.search.output_ceilings.query_extraction,
        )
        raw = await extractor.ainvoke(
            {"query": query, "target_language": target_language}, config
        )
        if not isinstance(raw, str):
            return ()
        # Not split at the ASCII comma alone: a Chinese or Japanese list
        # separates its items with 、 or ，.
        entities = tuple(split_entity_list(raw))
        chain._cache_extraction(key, entities)
        return entities

    return list(await shared_extraction(chain, key, extract))


async def shared_extraction(
    chain: GraphRAGChain,
    key: tuple[str, ...],
    extract: Callable[[], Awaitable[Any]],
) -> Any:
    """The extraction of ``key``: cached, or the one in flight in this
    extraction scope (another query of the scope started it: wait for it,
    a cache hit), or ``extract()``'s. Interleaved run-eval arms of one
    repeat ask the same question together, so the first pays for the call
    and the others share its keywords (under ABBA the payer alternates);
    a failed or cancelled first call leaves each waiter to extract on its
    own."""
    if (cached := chain._cached_extraction(key)) is not None:
        return cached
    cache: _ExtractionCache | None = getattr(chain, "_extractions", None)
    if cache is None or not hasattr(cache, "claim"):
        return await extract()
    future, owner = cache.claim(key)
    if not owner and future is not None:
        value = await asyncio.shield(future)
        if value is not None:
            _record_hit(chain, key)
            return value
        return await extract()
    value = None
    try:
        value = await extract()
        return value
    finally:
        cache.release(key, future, value)


def extraction_key(
    chain: GraphRAGChain,
    prompt_class: type[BasePrompt],
    query: str,
    target_language: str,
) -> tuple[str, ...]:
    """The extraction cache key: the prompt, its model, a digest of its
    templates (a custom_prompts override is another key), the question
    and the target language."""
    resolved = prompt_class.resolve(custom_prompts=chain.config.custom_prompts)
    templates = hashlib.sha256(
        f"{resolved.system_prompt_template}\0{resolved.human_prompt_template}".encode()
    ).hexdigest()
    return (
        prompt_class.__name__,
        chain.config.model_id(ModelRole.ENTITY_EXTRACTION),
        templates,
        query,
        target_language,
    )


def cached_extraction(chain: GraphRAGChain, key: tuple[str, ...]) -> Any:
    cache: _ExtractionCache | None = getattr(chain, "_extractions", None)
    hit = cache.get(key) if cache is not None else None
    if hit is not None:
        _record_hit(chain, key)
    return hit


def _record_hit(chain: GraphRAGChain, key: tuple[str, ...]) -> None:
    logger.debug("Reused the cached %s for query %s", key[0], text_digest(key[3]))
    # A model call the query did not make (shared.usage), under the
    # id the call's usage is recorded by.
    record_cache_hit(ModelPurpose.QUERY.value, _usage_model_id(chain, key[1]))


def _usage_model_id(chain: GraphRAGChain, model_id: str) -> str:
    """The id the LLM factory records ``model_id``'s usage under: the
    Bedrock factory's resolved inference profile, else ``model_id``."""
    return usage_model_id(getattr(chain, "factory", None), model_id)


def cache_extraction(chain: GraphRAGChain, key: tuple[str, ...], value: Any) -> None:
    cache: _ExtractionCache | None = getattr(chain, "_extractions", None)
    if cache is not None:
        cache.put(key, value)


class AbsorbedKeywordFailure(tuple[list[str], list[str]]):
    """The ``([], [])`` of a keyword extraction that failed without failing
    the query (an answer unreadable after the repair, or an error
    ``ignore_errors`` absorbed), with the error's type name and what failed,
    so the answer is marked degraded (the query's ledger,
    ``keyword_extraction``, with how the query went on:
    ``keyword_fallback.fallback_outcome``)."""

    error_type: str
    reason: str

    def __new__(
        cls, error_type: str, reason: str = "keyword extraction failed"
    ) -> AbsorbedKeywordFailure:
        value = super().__new__(cls, ([], []))
        value.error_type = error_type
        value.reason = reason
        return value


class _UnreadableKeywords(Exception):
    """A keyword answer no reading or repair made usable (its cause)."""


async def extract_dual_keywords(
    chain: GraphRAGChain,
    query: str,
    target_language: str,
    config: RunnableConfig | None = None,
) -> tuple[list[str], list[str]]:
    """Extract LightRAG high/low-level keywords as two lists.

    Robust to the LLM wrapping the JSON in prose or code fences. An answer
    still unreadable after the one repair is a degraded step, never a
    failed query: empty lists (``AbsorbedKeywordFailure``), so ``mix`` and
    ``hybrid`` search the raw query as their keyword when it is short enough
    (``domain.retrieval.keyword_fallback``), as upstream LightRAG
    (``_parse_keywords_payload`` returns empty lists). A failed call
    returns them under ``ignore_errors`` only.
    """
    try:
        key = chain._extraction_key(KeywordsExtractionPrompt, query, target_language)

        async def extract() -> tuple[tuple[str, ...], tuple[str, ...]]:
            extractor = chain._get_chain_for_prompt(
                KeywordsExtractionPrompt,
                StrOutputParser(),
                output_ceiling=chain.config.search.output_ceilings.query_extraction,
            )
            raw = await extractor.ainvoke(
                {"query": query, "target_language": target_language}, config
            )
            # The one JSON reading (and one repair).
            reader = chain._json_parser(
                KeywordsExtractionPrompt, answer_keys=_KEYWORD_LEVELS
            )
            try:
                payload = await reader.aparse(str(raw))
                hl_keywords, ll_keywords = _keyword_lists(payload)
            except (ValueError, LanguageModelError) as e:
                if is_unreachable_error(e):
                    raise
                raise _UnreadableKeywords() from e
            value = (tuple(hl_keywords), tuple(ll_keywords))
            chain._cache_extraction(key, value)
            return value

        high, low = await shared_extraction(chain, key, extract)
        return list(high), list(low)
    except _UnreadableKeywords as e:
        cause = e.__cause__ or e
        # The error text can quote the model's answer: DEBUG only.
        logger.warning(
            "Query keyword extraction (model role %s, '%s') returned no usable "
            "keyword JSON (%s); searching without keywords",
            ModelRole.ENTITY_EXTRACTION.value,
            chain.config.model_id(ModelRole.ENTITY_EXTRACTION),
            type(cause).__name__,
        )
        logger.debug("Dual-keyword answer error: %s", cause)
        return AbsorbedKeywordFailure(type(cause).__name__, "keyword answer unreadable")
    except Exception as e:
        # A model server that cannot be reached fails every later call too.
        if not chain.ignore_errors or is_unreachable_error(e):
            raise
        # The error text can quote the model's answer: DEBUG only.
        logger.warning("Dual-keyword extraction failed: %s", type(e).__name__)
        logger.debug("Dual-keyword extraction error: %s", e)
        return AbsorbedKeywordFailure(type(e).__name__)


def _keyword_lists(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate the keyword payload: an object whose keyword fields are
    lists of strings or numbers (a missing level is empty).

    Numbers are kept as text (a year or quantity is a real keyword). Anything
    else raises: iterating a bare string would yield one-character keywords,
    null/boolean/nested items are not keywords, and a non-object payload
    parses to ``{}``, which must not pass for "no keywords".
    """
    if not any(level in payload for level in _KEYWORD_LEVELS):
        raise LanguageModelError(
            "Keyword extraction returned no high_level_keywords or "
            "low_level_keywords object"
        )
    lists: list[list[str]] = []
    for level in _KEYWORD_LEVELS:
        value = payload.get(level, [])
        if not isinstance(value, list) or not all(
            isinstance(k, str | int | float) and not isinstance(k, bool) for k in value
        ):
            raise LanguageModelError(
                f"Keyword extraction returned {level} that is not a list of "
                f"strings or numbers: {value!r}"
            )
        keywords = (str(k).strip() for k in value)
        lists.append([k for k in keywords if k])
    return lists[0], lists[1]


async def read_rewrite(chain: GraphRAGChain, raw: str) -> str | None:
    """The follow-up rewrite's standalone question, or None (keep the query
    as is) when the model answered that it cannot rewrite it
    (``CANNOT_REWRITE_MARKER``), the question is empty, or the answer is no
    rewrite JSON even after one repair.

    An empty reply or a refusal would replace the user's question for
    routing, retrieval and the answer; the sentinel says so in any
    language, where a list of apology openings knew four.
    """
    if CANNOT_REWRITE_MARKER in raw:
        logger.info("The model could not rewrite the follow-up; using it as is")
        return None
    try:
        answer = await chain._json_parser(
            QueryRewritePrompt, schema=_RewriteAnswer, instructions=_REWRITE_FIX
        ).aparse(raw)
    except OutputParserException as e:
        logger.warning("Follow-up rewrite answer is not a rewrite; using the query")
        record_degradation(
            DegradationStep.FOLLOW_UP_REWRITE, "rewrite answer unreadable", e
        )
        return None
    text = str(answer.question).strip()
    if not text or CANNOT_REWRITE_MARKER in text:
        return None
    return text
