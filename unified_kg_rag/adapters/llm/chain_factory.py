# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LangChain chain assembly for any ``LLMFactoryPort`` (adapters layer).

These build ``prompt | llm | parser`` chains from a language-model factory,
so they live in the adapters layer rather than the shared kernel — keeping
``shared/`` free of any adapter dependency. The backend-agnostic
``RobustXMLOutputParser`` stays in ``adapters.llm.langchain``.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import logging
import threading
import time
from collections.abc import AsyncIterator, Collection, Iterator
from typing import TYPE_CHECKING, Any, Literal

from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, PrivateAttr

from unified_kg_rag.adapters.llm.langchain import (
    READ_TEXT_SECTIONS_AS_EMPTY,
    LLMJsonOutputParser,
    RobustXMLOutputParser,
    TextOnlySectionError,
)
from unified_kg_rag.adapters.llm.reachability import is_unreachable_error
from unified_kg_rag.adapters.llm.retry import (
    TransientErrorPredicate,
    acall_with_transient_retry,
    call_with_transient_retry,
    next_transient_retry_delay,
)
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.domain.models import ModelPurpose
from unified_kg_rag.domain.prompts import BasePrompt, ResolvedPrompt
from unified_kg_rag.domain.prompts.delimiters import (
    delimiter_tags,
    neutralise_delimiters,
)
from unified_kg_rag.ports.model_factory import LLMFactoryPort
from unified_kg_rag.shared import (
    GraphRAGException,
    LLMOutputTruncatedError,
    ModelOutputCutOffError,
    get_logger,
)
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.utils.model_output import (
    leading_reasoning_end,
    may_open_reasoning,
    opens_reasoning,
    strip_leading_reasoning,
)

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import (
        CustomPromptConfig,
        TransientRetryConfig,
    )

logger = get_logger(__name__)


PromptCacheMarker = Literal["cache_control", "cache_point"]

# Converse-native cache checkpoint. langchain-aws passes it through to the
# request's system blocks; it drops an Anthropic cache_control key instead.
_CACHE_POINT_BLOCK: dict[str, Any] = {"cachePoint": {"type": "default"}}


def _prompt_cache_marker(
    llm: Any, model_info: Any, resolved: ResolvedPrompt
) -> PromptCacheMarker | None:
    """Which cache marker the system prompt gets, or None for no caching.

    Converse (``ChatBedrockConverse``, every inference profile) takes a native
    ``cachePoint`` block; the InvokeModel body (``ChatBedrock``) takes an
    Anthropic ``cache_control`` key. A system prompt shorter than the model's
    minimum checkpoint size would be accepted but not cached, so it gets none.
    """
    if not (model_info and model_info.supports_prompt_caching):
        return None
    minimum = getattr(model_info, "min_cache_tokens", 0)
    if estimate_token_count(resolved.system_prompt_template) < minimum:
        return None
    # Only Bedrock records support prompt caching, so this import only runs
    # for a Bedrock model: provider-neutral callers never load langchain-aws.
    from langchain_aws import ChatBedrockConverse

    if isinstance(llm, ChatBedrockConverse):
        return "cache_point"
    return "cache_control"


# The TTL a cache checkpoint gets when none is asked for: Bedrock's default,
# sent as no ``ttl`` key at all.
_DEFAULT_CACHE_TTL = "5m"


def _prompt_cache_ttl(
    factory: Any,
    model_info: Any,
    cache_marker: PromptCacheMarker | None,
    model_purpose: ModelPurpose,
) -> str | None:
    """The extended TTL for this chain's checkpoint, or None for the default.

    Only query chains opt in (``query_prompt_cache_ttl`` on the factory):
    ingestion reuses each prompt within seconds, so the 2x write price of a
    1-hour entry would buy nothing. The TTL rides on the Converse
    ``cachePoint`` only: langchain-aws rewrites an InvokeModel
    ``cache_control`` to ``{"type": "ephemeral"}``, dropping any ``ttl``.
    """
    if model_purpose is not ModelPurpose.QUERY or cache_marker != "cache_point":
        return None
    ttl = getattr(factory, "query_prompt_cache_ttl", None)
    if not ttl or ttl == _DEFAULT_CACHE_TTL:
        return None
    if not getattr(model_info, "supports_1h_cache_ttl", False):
        return None
    return str(ttl)


def _build_chat_prompt(
    resolved: ResolvedPrompt,
    cache_marker: PromptCacheMarker | None,
    cache_ttl: str | None = None,
) -> ChatPromptTemplate:
    """Assemble a LangChain ChatPromptTemplate from a backend-agnostic prompt.

    Lives in the adapter layer: turning the domain's ResolvedPrompt into
    LangChain message templates is a backend concern. ``cache_marker`` marks
    the end of the system prompt as a prompt-cache checkpoint, with
    ``cache_ttl`` as its lifetime when set (else the 5-minute default).
    """
    system_template: str | list[str | dict[str, Any]]
    # Templated content blocks (not a literal SystemMessage) so system-side
    # placeholders such as {entity_types} are substituted and escaped braces
    # render exactly as in the non-cache path; LangChain keeps extra block keys
    # and non-text blocks, so the cache markers survive formatting.
    if cache_marker == "cache_control":
        system_template = [
            {
                "type": "text",
                "text": resolved.system_prompt_template,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif cache_marker == "cache_point":
        cache_point = {**_CACHE_POINT_BLOCK["cachePoint"]}
        if cache_ttl is not None:
            cache_point["ttl"] = cache_ttl
        system_template = [
            {"type": "text", "text": resolved.system_prompt_template},
            {"cachePoint": cache_point},
        ]
    else:
        system_template = resolved.system_prompt_template
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(resolved.human_prompt_template),
    ]
    tags = delimiter_tags(
        resolved.system_prompt_template, resolved.human_prompt_template
    )
    if not tags:
        return ChatPromptTemplate.from_messages(messages)
    prompt = DelimitedChatPromptTemplate.from_messages(messages)
    return prompt.model_copy(update={"delimiter_tags": tags})


def _serialize_once(*runnables: Any) -> None:
    """Compute each runnable's cached serialized form now, in this thread.

    LangChain prompts and chat models compute it at their first call and keep
    it in the instance (``_serialized``, a ``functools.cached_property``,
    unlocked since Python 3.12). Computing it iterates the instance's
    ``__dict__``, so when a batch's worker threads make their first calls on
    one shared chain together, one thread's cache write fails another's call
    with "dictionary changed size during iteration".
    """
    for runnable in runnables:
        getattr(runnable, "_serialized", None)


class DelimitedChatPromptTemplate(ChatPromptTemplate):
    """Chat prompt that escapes its delimiter tags in every string input.

    Corpus text, retrieved context and reports are bound inside tag pairs the
    prompt calls data (``<context>{context}</context>``); a value containing
    ``</context>`` would close the block and turn the text after it into
    instructions. ``delimiter_tags`` are read from the resolved templates
    (``delimiter_tags()``), so a custom prompt's own delimiters are covered.
    """

    delimiter_tags: frozenset[str] = frozenset()

    def _neutralised(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (
                neutralise_delimiters(value, self.delimiter_tags)
                if isinstance(value, str)
                else value
            )
            for key, value in kwargs.items()
        }

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        return super().format_messages(**self._neutralised(kwargs))

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        return await super().aformat_messages(**self._neutralised(kwargs))


# How each provider's LangChain integration marks a response cut at its
# output-token limit: response-metadata key -> value. langchain-aws: Converse
# ``stopReason`` / InvokeModel ``stop_reason``; langchain-openai (any
# OpenAI-compatible server) ``finish_reason``; langchain-ollama ``done_reason``.
_TRUNCATION_MARKERS = {
    "stopReason": "max_tokens",
    "stop_reason": "max_tokens",
    "finish_reason": "length",
    "done_reason": "length",
}


# Set by a caller that consumes a stream as a whole (GraphRAGChain.ainvoke
# streaming its answer to a callback): a truncated stream then fails as an
# unstreamed response does, after its last chunk, instead of being logged.
STRICT_STREAM_TRUNCATION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "kg_rag_strict_stream_truncation", default=False
)


def _truncation_reason(message: Any) -> str | None:
    """``key=value`` of the metadata marking ``message`` as cut, else None."""
    metadata = getattr(message, "response_metadata", None) or {}
    for key, value in _TRUNCATION_MARKERS.items():
        if metadata.get(key) == value:
            return f"{key}={value}"
    return None


class TruncationGuard(Runnable[Any, Any]):
    """Fail a model response that stopped at its output-token limit.

    A response cut at ``max_tokens`` still parses: the XML parser recovers the
    sections before the cut (entities without relationships, half a report),
    so the item would count as a success with data missing. Sits between the
    model and the parser and raises :class:`LLMOutputTruncatedError` instead,
    which callers count as a failed item; the error is not transient, so a
    retry does not re-pay a generation that would hit the same limit.

    Streaming passes every chunk through: the stop reason arrives with the
    last chunk, after the output reached the caller, so it is logged only;
    under ``STRICT_STREAM_TRUNCATION`` it then raises, as unstreamed.
    """

    def __init__(self, prompt_name: str, model_id: str, purpose: ModelPurpose) -> None:
        self.prompt_name = prompt_name
        self.model_id = model_id
        self.purpose = purpose

    def _warn(self, reason: str) -> None:
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure - LLM token counts, not credentials
        logger.warning(
            "'%s' response (%s, model '%s') stopped at the output token limit " "(%s)",
            self.prompt_name,
            self.purpose.value,
            self.model_id,
            reason,
        )

    def _check(self, message: Any) -> Any:
        if reason := _truncation_reason(message):
            self._check_reason(reason)
        return message

    def _check_reason(self, reason: str) -> None:
        self._warn(reason)
        raise LLMOutputTruncatedError(
            f"'{self.prompt_name}' response from model '{self.model_id}' "
            f"was cut at the output token limit"
        )

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return self._check(input)

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return self._check(input)

    def transform(
        self,
        input: Iterator[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        reason = None
        for chunk in input:
            reason = reason or _truncation_reason(chunk)
            yield chunk
        if reason:
            self._streamed_cut(reason)

    async def atransform(
        self,
        input: AsyncIterator[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        reason = None
        async for chunk in input:
            reason = reason or _truncation_reason(chunk)
            yield chunk
        if reason:
            self._streamed_cut(reason)

    def _streamed_cut(self, reason: str) -> None:
        if STRICT_STREAM_TRUNCATION.get():
            self._check_reason(reason)
        self._warn(reason)


def _content_text(message: Any) -> str | None:
    """A model message's (or a plain string's) text content, else None (a
    list of content blocks: Bedrock's reasoning arrives as typed blocks)."""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else None


def _with_content(message: Any, text: str) -> Any:
    if isinstance(message, str):
        return text
    return message.model_copy(update={"content": text})


class ReasoningStripper(Runnable[Any, Any]):
    """Drop the reasoning block a model answer starts with, before any parser.

    A reasoning model whose server leaves its reasoning in the answer (an
    Ollama or vLLM server without a reasoning parser, a distilled R1 or
    Qwen3) writes ``<think>...</think>`` (or ``<thinking>``, ``<reasoning>``)
    before it: every answer then started with it, extraction descriptions
    stored and embedded it, the router read a strategy it named, and the
    abstention token was no longer the answer's start. One normaliser in
    ``setup_chain`` (and the output fixers' models) removes it for every
    parser (:func:`~unified_kg_rag.shared.utils.strip_leading_reasoning`).

    A stream holds its first chunks only until it can tell whether they
    open a block, and a block's chunks pass on emptied (their metadata, the
    stop reason with the last one, kept). A message whose answer is empty
    while its server returned reasoning separately
    (``additional_kwargs["reasoning_content"]``) is logged: the output limit
    went to reasoning.
    """

    def __init__(self, prompt_name: str = "") -> None:
        self.prompt_name = prompt_name

    def _strip(self, message: Any) -> Any:
        text = _content_text(message)
        if text is None:
            return message
        stripped = strip_leading_reasoning(text)
        if not stripped.strip() and getattr(message, "additional_kwargs", {}).get(
            "reasoning_content"
        ):
            logger.warning(
                "'%s' response holds reasoning and no answer: raise the output "
                "token limit or lower the reasoning effort",
                self.prompt_name,
            )
        if stripped == text:
            return message
        logger.debug("'%s' response: dropped its leading reasoning", self.prompt_name)
        return _with_content(message, stripped)

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return self._strip(input)

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return self._strip(input)

    def transform(
        self,
        input: Iterator[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        stream = _ReasoningStream(self.prompt_name)
        for chunk in input:
            yield from stream.feed(chunk)
        yield from stream.flush()

    async def atransform(
        self,
        input: AsyncIterator[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        stream = _ReasoningStream(self.prompt_name)
        async for chunk in input:
            for out in stream.feed(chunk):
                yield out
        for out in stream.flush():
            yield out


class _ReasoningStream:
    """``ReasoningStripper`` over a stream: held chunks until the start
    tells, then the block's chunks emptied, then the answer as it comes."""

    def __init__(self, prompt_name: str) -> None:
        self.prompt_name = prompt_name
        self.mode: Literal["start", "reasoning", "after", "answer"] = "start"
        self.held: list[Any] = []
        self.text = ""

    def feed(self, chunk: Any) -> list[Any]:
        text = _content_text(chunk)
        if self.mode == "answer" or (text is None and self.mode == "start"):
            self.mode = "answer"
            return [*self._release(), chunk]
        text = text or ""
        if self.mode == "after":
            if not text.strip():
                return [_with_content(chunk, "")]
            self.mode = "answer"
            return [_with_content(chunk, text.lstrip())]
        self.held.append(chunk)
        self.text += text
        if self.mode == "start":
            if not opens_reasoning(self.text):
                if may_open_reasoning(self.text):
                    return []
                self.mode = "answer"
                return self._release()
            self.mode = "reasoning"
        end = leading_reasoning_end(self.text)
        if end is None:
            return []
        rest = self.text[end:]
        emptied = [_with_content(c, "") for c in self.held[:-1]]
        last = _with_content(self.held[-1], rest)
        self.held, self.text = [], ""
        self.mode = "answer" if rest.strip() else "after"
        logger.debug("'%s' stream: dropped its leading reasoning", self.prompt_name)
        return [*emptied, last]

    def _release(self) -> list[Any]:
        held, self.held, self.text = self.held, [], ""
        return held

    def flush(self) -> list[Any]:
        if self.mode == "reasoning" and self.held:
            # Never closed: the output ran out while reasoning.
            held, self.held = self.held, []
            return [_with_content(c, "") for c in held]
        return self._release()


class TransientRetryRunnable(Runnable[Any, Any]):
    """Retry a chain on transient errors (predicate-based ``with_retry``).

    LangChain's ``Runnable.with_retry`` only matches exception *types*, but a
    transient Bedrock fault is a ``botocore.exceptions.ClientError`` whose
    error code decides retryability (424 ``ModelErrorException`` yes,
    ``ValidationException`` no), so the factory supplies the predicate
    (``is_transient_error``) and :mod:`~unified_kg_rag.adapters.llm.retry`
    applies the same policy as the embedding path. Non-transient errors and
    exhausted retries re-raise the original exception unchanged.

    ``batch``/``abatch`` use the base-class per-input ``invoke``/``ainvoke``, so
    each input is retried independently. Streaming retries only while no chunk
    has been emitted; once output reached the caller a failure propagates.
    """

    def __init__(
        self,
        bound: Runnable[Any, Any],
        *,
        operation: str,
        retry: TransientRetryConfig,
        is_transient: TransientErrorPredicate,
    ) -> None:
        self.bound = bound
        self.operation = operation
        self.retry = retry
        self.is_transient = is_transient

    @property
    def InputType(self) -> Any:  # noqa: N802 - LangChain API name
        return self.bound.InputType

    @property
    def OutputType(self) -> Any:  # noqa: N802 - LangChain API name
        return self.bound.OutputType

    def _delay(
        self, exc: BaseException, attempt: int, started_at: float
    ) -> float | None:
        return next_transient_retry_delay(
            exc,
            is_transient=self.is_transient,
            operation=self.operation,
            attempt=attempt,
            started_at=started_at,
            policy=self.retry,
        )

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return call_with_transient_retry(
            lambda: self.bound.invoke(input, config, **kwargs),
            is_transient=self.is_transient,
            operation=self.operation,
            policy=self.retry,
        )

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return await acall_with_transient_retry(
            lambda: self.bound.ainvoke(input, config, **kwargs),
            is_transient=self.is_transient,
            operation=self.operation,
            policy=self.retry,
        )

    # Streaming keeps its own loop: a retry is allowed only before the first
    # chunk, which the call-level helpers cannot observe.
    def stream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        started_at = time.monotonic()
        attempt = 1
        while True:
            emitted = False
            try:
                for chunk in self.bound.stream(input, config, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                delay = None if emitted else self._delay(exc, attempt, started_at)
                if delay is None:
                    raise
                time.sleep(delay)
                attempt += 1

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        started_at = time.monotonic()
        attempt = 1
        while True:
            emitted = False
            try:
                async for chunk in self.bound.astream(input, config, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                delay = None if emitted else self._delay(exc, attempt, started_at)
                if delay is None:
                    raise
                await asyncio.sleep(delay)
                attempt += 1


def _reachable_and(is_transient: TransientErrorPredicate) -> TransientErrorPredicate:
    """``is_transient`` but for a model server that could not be reached
    (``is_unreachable_error``): a query fails at once naming it
    (``ModelUnreachableError``), not after the retries' backoff, since every
    later call of the query meets the same server."""
    return lambda exc: is_transient(exc) and not is_unreachable_error(exc)


def with_transient_retry(
    runnable: Runnable[Any, Any],
    *,
    operation: str,
    retry: TransientRetryConfig | None,
    is_transient: TransientErrorPredicate | None = None,
) -> Runnable[Any, Any]:
    """Wrap ``runnable`` in :class:`TransientRetryRunnable` unless retry is off."""
    if retry is None or retry.max_attempts <= 1 or is_transient is None:
        return runnable
    return TransientRetryRunnable(
        runnable, operation=operation, retry=retry, is_transient=is_transient
    )


class XMLOutputFixingParser(OutputFixingParser[Any]):
    """``OutputFixingParser`` that never asks the fixer to repair a blank answer
    (the XML parsers' and the JSON parser's fixer). A list section the fixer
    still answers with text only (``TextOnlySectionError``) is read as empty
    and recorded instead of failing the item.

    An empty or whitespace-only completion (e.g. a thinking-only response)
    carries nothing to repair, so a fixer could only invent the structure;
    it fails with the parser's own ``OutputParserException`` instead, which
    the batch retry treats as retryable.
    """

    def parse(self, completion: str) -> Any:
        if not completion.strip():
            return self.parser.parse(completion)
        try:
            return super().parse(completion)
        except TextOnlySectionError as e:
            return self._read_text_as_empty(e)

    async def aparse(self, completion: str) -> Any:
        if not completion.strip():
            return await self.parser.aparse(completion)
        try:
            return await super().aparse(completion)
        except TextOnlySectionError as e:
            return self._read_text_as_empty(e)

    def _read_text_as_empty(self, error: TextOnlySectionError) -> Any:
        """The fixer's answer (``error.llm_output``) whose list section still
        holds text only, read with that section empty and recorded: the
        model and its repair both said, in prose, that it lists nothing."""
        token = READ_TEXT_SECTIONS_AS_EMPTY.set(True)
        try:
            return self.parser.parse(str(error.llm_output or ""))
        finally:
            READ_TEXT_SECTIONS_AS_EMPTY.reset(token)


def usage_model_id(factory: object, model_id: str) -> str:
    """The id ``factory`` records ``model_id``'s usage under (``shared.usage``):
    the Bedrock factory's resolved inference profile, else ``model_id``. A
    cache hit recorded by it counts under the same model as the calls."""
    resolve = getattr(factory, "resolved_model_id", None)
    return str(resolve(model_id)) if resolve is not None else model_id


def create_robust_xml_output_parser(
    factory: LLMFactoryPort,
    enable_output_fixing: bool,
    output_fixing_model_id: str,
    *,
    output_tags: list[str],
    model_purpose: ModelPurpose = ModelPurpose.QUERY,
    min_output_tokens: int = 0,
    parser_class: type[RobustXMLOutputParser] = RobustXMLOutputParser,
    container_tags: Collection[str] = (),
) -> BaseOutputParser[Any]:
    """Build the XML parser (a ``parser_class``), optionally wrapped in an
    LLM output fixer.

    ``output_tags`` are the top-level elements the prompt asks for; the fixer
    is told them, so it repairs toward the prompt's structure. ``model_purpose``
    is forwarded to the fixing LLM so it gets the same per-path policy (the
    response cache, the ingestion effort) as the chain it repairs, and ``min_output_tokens`` (the
    repaired prompt's output floor) so a long output can be re-emitted in full.
    ``container_tags`` adds to the parser class's sections that list
    elements (``RobustXMLOutputParser.container_tags``).
    """
    base_parser = parser_class(tags=output_tags)
    if container_tags:
        base_parser.container_tags = base_parser.container_tags | frozenset(
            container_tags
        )
    if not enable_output_fixing:
        return base_parser

    try:
        fixing_llm = factory.get_model(
            model_id=output_fixing_model_id,
            model_purpose=model_purpose,
            min_output_tokens=min_output_tokens,
        )
        logger.info(
            "Created OutputFixingParser with model: '%s'", output_fixing_model_id
        )
        return _with_fixer(base_parser, fixing_llm)
    except Exception as e:
        log_error(
            logger,
            logging.ERROR,
            "Failed to create OutputFixingParser with model %s: %s",
            output_fixing_model_id,
            error=e,
        )
        raise GraphRAGException(f"Failed to create OutputFixingParser: {e}") from e


def _with_fixer(
    parser: BaseOutputParser[Any], fixing_llm: Any
) -> BaseOutputParser[Any]:
    """``parser`` wrapped in one repair call to ``fixing_llm`` (its answer
    stripped of leading reasoning, as every chain's)."""
    fixer = XMLOutputFixingParser.from_llm(
        parser=parser, llm=fixing_llm | ReasoningStripper("OutputFixingParser")
    )
    _serialize_once(*getattr(fixer.retry_chain, "steps", ()))
    return fixer


class FixingJsonOutputParser(LLMJsonOutputParser):
    """``LLMJsonOutputParser`` that asks a fixing model once when nothing
    parses: the one failure policy of a JSON answer.

    Read the answer whatever the model family's habits
    (``parse_model_object``), else ask the fixer once (its answer stripped
    of reasoning too), else raise ``OutputParserException``, which the
    caller records as a degraded step (``metadata.degraded_error``) or a
    failed item. The fixing model is built on the first failure only, so a
    chain whose answers parse never builds it. A blank answer has nothing
    to repair and raises at once, as does one cut off inside its last
    object.
    """

    _factory: Any = PrivateAttr(default=None)
    _fixing_model_id: str | None = PrivateAttr(default=None)
    _model_purpose: ModelPurpose = PrivateAttr(default=ModelPurpose.QUERY)
    _fixer: BaseOutputParser[Any] | None = PrivateAttr(default=None)
    _fixer_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _fixing_parser(self) -> BaseOutputParser[Any]:
        with self._fixer_lock:
            if self._fixer is None:
                try:
                    llm = self._factory.get_model(
                        model_id=self._fixing_model_id,
                        model_purpose=self._model_purpose,
                    )
                    plain = LLMJsonOutputParser(
                        answer_keys=self.answer_keys,
                        schema_model=self.schema_model,
                        instructions=self.instructions,
                    )
                    self._fixer = _with_fixer(plain, llm)
                except Exception as e:
                    raise GraphRAGException(
                        f"Failed to create OutputFixingParser: {e}"
                    ) from e
            return self._fixer

    def _repairable(self, text: str, error: OutputParserException) -> bool:
        """Whether a fixer may repair ``text``: not blank, not cut off (a
        fixer could only invent the rest), and a fixing model is set."""
        return (
            bool(text.strip())
            and not isinstance(error.__cause__, ModelOutputCutOffError)
            and self._fixing_model_id is not None
        )

    def parse(self, text: str) -> Any:
        try:
            return super().parse(text)
        except OutputParserException as e:
            if not self._repairable(text, e):
                raise
            logger.info("JSON answer did not parse: asking the output fixer once")
            return self._fixing_parser().parse(text)

    async def aparse(self, text: str) -> Any:
        try:
            return super().parse(text)
        except OutputParserException as e:
            if not self._repairable(text, e):
                raise
            logger.info("JSON answer did not parse: asking the output fixer once")
            return await self._fixing_parser().aparse(text)


def create_json_output_parser(
    factory: LLMFactoryPort,
    output_fixing_model_id: str | None,
    *,
    answer_keys: tuple[str, ...] = (),
    schema: type[BaseModel] | None = None,
    instructions: str = "",
    model_purpose: ModelPurpose = ModelPurpose.QUERY,
) -> FixingJsonOutputParser:
    """The JSON answer parser every JSON prompt uses
    (:class:`FixingJsonOutputParser`), repairing through
    ``output_fixing_model_id`` (``models.roles.fixing``; None: no repair).

    Put it in the chain, or call its ``aparse`` on a chain's raw text.
    ``answer_keys`` and ``schema`` pick and check the answer
    (``parse_model_object``); ``instructions`` tells the fixer what the
    prompt allows besides the object (a sentinel), so it does not dress a
    refusal up as an answer.
    """
    parser = FixingJsonOutputParser(
        answer_keys=answer_keys, schema_model=schema, instructions=instructions
    )
    parser._factory = factory
    parser._fixing_model_id = output_fixing_model_id
    parser._model_purpose = model_purpose
    return parser


def setup_chain(
    factory: LLMFactoryPort,
    model_id: str,
    prompt_class: type[BasePrompt],
    parser: BaseOutputParser[Any],
    custom_prompts: CustomPromptConfig | None = None,
    model_purpose: ModelPurpose = ModelPurpose.QUERY,
    min_output_tokens: int | None = None,
    system_suffix: str = "",
    **kwargs: Any,
) -> Runnable[Any, Any]:
    """Build ``prompt | llm | parser`` with the policies of ``model_purpose``.

    ``QUERY`` (the default) gets, when the factory classifies transient
    errors and has a chain retry policy (``is_transient_error`` and
    ``transient_retry``, the Bedrock and Ollama factories), a transient-error
    retry around the whole chain (a model server that could not be reached is
    not retried: the query fails naming it). Ingestion and evaluation call sites pass
    their purpose explicitly: they get no chain-level retry, because ``BatchProcessor`` already retries them and a
    second layer would multiply the attempts. A factory without a
    ``transient_retry`` policy (the OpenAI-compatible one) leaves query
    retries to its client. ``min_output_tokens`` overrides the prompt's static
    floor with one derived from the config (``BasePrompt.output_floor``).
    ``system_suffix`` is appended to the resolved system template (built-in
    or custom), e.g. the answer prompts' ``search.abstention`` instructions;
    it is template text, so literal braces must be doubled.
    """
    try:
        llm = factory.get_model(
            model_id=model_id,
            model_purpose=model_purpose,
            min_output_tokens=(
                prompt_class.min_output_tokens
                if min_output_tokens is None
                else min_output_tokens
            ),
            **kwargs,
        )
        model_info = factory.get_model_info(model_id)
        resolved = prompt_class.resolve(custom_prompts=custom_prompts)
        if system_suffix:
            resolved = dataclasses.replace(
                resolved,
                system_prompt_template=resolved.system_prompt_template + system_suffix,
            )
        cache_marker = _prompt_cache_marker(llm, model_info, resolved)
        prompt = _build_chat_prompt(
            resolved,
            cache_marker,
            _prompt_cache_ttl(factory, model_info, cache_marker, model_purpose),
        )
        _serialize_once(prompt, llm)
        guard = TruncationGuard(prompt_class.__name__, model_id, model_purpose)
        # Every parser reads the answer without a leading reasoning block.
        normalizer = ReasoningStripper(prompt_class.__name__)
        # Named after the prompt, so a trace shows "AnswerGenerationPrompt"
        # instead of an anonymous "RunnableSequence".
        chain: Runnable[Any, Any] = (
            prompt | llm | guard | normalizer | parser
        ).with_config(run_name=prompt_class.__name__)
        logger.debug("Successfully created LLM chain with model: '%s'", model_id)
        is_transient = getattr(factory, "is_transient_error", None)
        if model_purpose is not ModelPurpose.QUERY or is_transient is None:
            return chain
        return with_transient_retry(
            chain,
            operation=prompt_class.__name__,
            retry=getattr(factory, "transient_retry", None),
            is_transient=_reachable_and(is_transient),
        )
    except Exception as e:
        log_error(
            logger,
            logging.ERROR,
            "Failed to setup LLM chain with model '%s': %s",
            model_id,
            error=e,
        )
        raise GraphRAGException(
            f"Failed to setup LLM chain with model '{model_id}': {e}"
        ) from e
