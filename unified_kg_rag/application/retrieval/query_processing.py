# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query processing of ``GraphRAGChain``: the translation of a query and
the entity and keyword extractions its strategy reads, the extraction
cache, and the heuristics that tell a model's refusal from the text asked
for (a translation, a follow-up rewrite).

The functions take the chain they process for: it holds the config, the
prompt chains (``_get_chain_for_prompt``) and the extraction cache, and it
keeps a delegating method for each public function here, which these call
through (a test or a subclass replaces a step on the chain).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.llm.chain_factory import usage_model_id
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
    TranslationPrompt,
)
from unified_kg_rag.domain.retrieval.query_entities import split_entity_list
from unified_kg_rag.shared import LanguageModelError, get_logger
from unified_kg_rag.shared.usage import record_cache_hit
from unified_kg_rag.shared.utils import parse_llm_json, text_digest

if TYPE_CHECKING:
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        _ExtractionCache,
    )

logger = get_logger(__name__)

# Openings of a model refusal or meta-reply rather than the requested text
# (a rewritten question, a translation). Matched case-insensitively, in
# English, Korean, Japanese and Chinese (a model replies in the user's
# language); a refusal in another language is not recognised.
_REFUSAL_OPENINGS: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "sorry,",
    "i cannot",
    "i can't",
    "i can not",
    "i'm unable",
    "i am unable",
    "as an ai",
    "죄송",
    "유감스럽게도",
    "ai로서",
    "申し訳",
    "すみません",
    "恐れ入りますが",
    "aiとして",
    "抱歉",
    "对不起",
    "對不起",
    "很抱歉",
    "我无法",
    "我無法",
    "我不能",
    "作为ai",
    "作为一个ai",
)
# A translation request also draws commentary about the input (seen when
# the query is already in the target language).
_TRANSLATION_REFUSAL_MARKERS: tuple[str, ...] = _REFUSAL_OPENINGS + (
    "i appreciate",
    "i notice",
    "it appears",
    "the text you",
    "there is no text",
    "no text was provided",
)

# The two keyword lists of the keyword extraction's answer.
_KEYWORD_LEVELS: tuple[str, str] = (
    "high_level_keywords",
    "low_level_keywords",
)

# A request for clarification aimed back at the user: after a refusal
# opening it is a meta-reply even though it ends with a question mark.
_CLARIFICATION_REQUEST: re.Pattern[str] = re.compile(
    r"\b(?:could|can|would) you (?:please )?(?:clarify|specify|confirm)"
    r"|\bplease (?:clarify|specify)"
    r"|\b(?:what|which)\b[^?]*\b(?:do|did) you (?:mean|refer)"
    r"|\bare you referring\b"
    # Korean, Japanese, Chinese.
    r"|명확히\s*해\s*주|구체적으로\s*(?:말씀|알려)|말씀하시는|의미하시는"
    r"|具体的に(?:教え|お知らせ|ご指定)|指していますか|どの.*(?:ですか|でしょうか)"
    r"|请(?:澄清|说明|具体说明|明确)|請(?:澄清|說明)|您(?:是指|指的是)|你(?:是指|指的是)"
)
# A question's closing mark: ASCII or fullwidth (Chinese, Japanese).
_QUESTION_MARKS = ("?", "\uff1f")


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
) -> str:
    translator = chain._get_chain_for_prompt(TranslationPrompt, StrOutputParser())
    return str(
        await translator.ainvoke(
            {"query": query, "target_language": target_language}, config
        )
    )


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
    # The first failure ignore_errors absorbed: the answer is degraded
    # (searched untranslated, or without entities or keywords).
    degraded: str | None = None
    translated_query = results_map.get("translation")
    if not isinstance(translated_query, str):
        if isinstance(translated_query, Exception):
            if chain.ignore_errors:
                logger.warning("Query translation failed: %s", translated_query)
                degraded = type(translated_query).__name__
            else:
                raise translated_query
        translated_query = None
    elif chain._looks_like_translation_refusal(translated_query, question):
        # The translator returned prose about the request instead of a
        # translation (common when the query is already in the target
        # language). Fall back to the original query rather than searching
        # for the LLM's meta-response.
        logger.warning(
            "Query translation looks like non-translation LLM output; "
            "falling back to the original query."
        )
        translated_query = None

    entity_data = results_map.get("entities", [])
    if not isinstance(entity_data, list):
        if isinstance(entity_data, Exception):
            if chain.ignore_errors:
                logger.warning("Entity extraction failed: %s", entity_data)
                degraded = degraded or type(entity_data).__name__
            else:
                raise entity_data
        entity_data = []

    hl_keywords: list[str] = []
    ll_keywords: list[str] = []
    if "keywords" in results_map:
        keyword_data = results_map["keywords"]
        if isinstance(keyword_data, BaseException):
            # extract_dual_keywords already degrades to ([], []) under
            # ignore_errors, so an exception here must propagate.
            raise keyword_data
        if isinstance(keyword_data, AbsorbedKeywordFailure):
            degraded = degraded or keyword_data.error_type
        hl_keywords, ll_keywords = keyword_data

    return {
        "translated_query": translated_query,
        "final_query": translated_query or question,
        "entities": entity_data,
        "hl_keywords": hl_keywords,
        "ll_keywords": ll_keywords,
        "degraded_error": degraded,
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


def _refusal_text(text: str) -> str:
    """``text`` normalised for matching the refusal phrases: stripped,
    lower-cased, a typographic apostrophe (``I\u2019m``) made ASCII."""
    return text.strip().lower().replace("\u2019", "'")


def looks_like_translation_refusal(candidate: str, original: str) -> bool:
    """Heuristic: does the 'translation' look like LLM meta-output, not a
    translation? Only flags when the candidate differs from the original
    (an identical passthrough is fine) and opens with a known refusal/notice
    phrase, so genuine translations that merely contain such words mid-text
    are not caught.

    The openings are English, Korean, Japanese and Chinese
    (``_REFUSAL_OPENINGS``), the notices English: a meta-reply written
    otherwise passes as a translation. No language-agnostic test separates the two
    reliably (a script mismatch misses a reply in the target language,
    which is the common case, and lengths differ legitimately across
    scripts), so the miss is accepted; the reply then searches as the
    query."""
    text = _refusal_text(candidate)
    if not text or text == _refusal_text(original):
        return False
    return text.startswith(_TRANSLATION_REFUSAL_MARKERS)


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
    """The ``([], [])`` of a keyword extraction whose failure
    ``ignore_errors`` absorbed, with the error's type name, so the answer
    is marked degraded (``ProcessedQuery.degraded_error``)."""

    error_type: str

    def __new__(cls, error_type: str) -> AbsorbedKeywordFailure:
        value = super().__new__(cls, ([], []))
        value.error_type = error_type
        return value


async def extract_dual_keywords(
    chain: GraphRAGChain,
    query: str,
    target_language: str,
    config: RunnableConfig | None = None,
) -> tuple[list[str], list[str]]:
    """Extract LightRAG high/low-level keywords as two lists.

    Robust to the LLM wrapping the JSON in prose or code fences. Returns
    empty lists on failure when ``ignore_errors`` is set
    (``AbsorbedKeywordFailure``).
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
            # Strict: with ignore_errors=False a broken keyword extraction
            # surfaces instead of silently yielding empty keyword lists.
            try:
                payload = parse_llm_json(raw, strict=True, answer_keys=_KEYWORD_LEVELS)
                hl_keywords, ll_keywords = _keyword_lists(payload)
            except (ValueError, LanguageModelError) as e:
                raise _keyword_json_error(chain, e) from e
            value = (tuple(hl_keywords), tuple(ll_keywords))
            chain._cache_extraction(key, value)
            return value

        high, low = await shared_extraction(chain, key, extract)
        return list(high), list(low)
    except Exception as e:
        if not chain.ignore_errors:
            raise
        # The error text can quote the model's answer: DEBUG only.
        logger.warning("Dual-keyword extraction failed: %s", type(e).__name__)
        logger.debug("Dual-keyword extraction error: %s", e)
        return AbsorbedKeywordFailure(type(e).__name__)


def _keyword_json_error(chain: GraphRAGChain, error: Exception) -> LanguageModelError:
    """The error of a keyword extraction whose answer is no usable JSON:
    which model role answered and what to do, not only the parser's
    'Expecting value: line 1 column 1'."""
    model = chain.config.model_id(ModelRole.ENTITY_EXTRACTION)
    return LanguageModelError(
        "Query keyword extraction (model role "
        f"{ModelRole.ENTITY_EXTRACTION.value}, '{model}') returned no usable "
        f"keyword JSON: {error}. Set models.roles.{ModelRole.ENTITY_EXTRACTION.value} "
        "to a model that follows the JSON instructions, set "
        "processing.ignore_errors: true to search without keywords, or query "
        "with --disable-query-processing (RAGInput.enable_query_processing: "
        "false), which searches a short question as its own keyword."
    )


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


def usable_rewrite(rewritten: str) -> str | None:
    """The rewrite, or None (keep the query as is) when it is unusable.

    An empty reply or a refusal would replace the user's question for
    routing, retrieval and the answer, so the original query is kept.
    """
    text = rewritten.strip()
    prefix = "standalone question:"
    if text.lower().startswith(prefix):
        text = text[len(prefix) :].strip()
    if not text:
        return None
    # A refusal opens with an apology or an inability and asks nothing; a
    # rewritten question may start the same way ("I can't find …; where
    # is it?") but ends with a question mark, unless the question asks
    # the user to clarify ("I'm sorry, could you clarify which contract?").
    opening = _refusal_text(text)
    if opening.startswith(_REFUSAL_OPENINGS) and (
        not text.endswith(_QUESTION_MARKS) or _CLARIFICATION_REQUEST.search(opening)
    ):
        logger.warning("Follow-up rewrite was a refusal; using the query as is")
        return None
    return text
