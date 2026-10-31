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
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Literal

from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.llm.langchain import RobustXMLOutputParser
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
    get_logger,
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
    last chunk, after the output reached the caller, so it is logged only.
    """

    def __init__(self, prompt_name: str, model_id: str, purpose: ModelPurpose) -> None:
        self.prompt_name = prompt_name
        self.model_id = model_id
        self.purpose = purpose

    def _warn(self, reason: str) -> None:
        logger.warning(
            "'%s' response (%s, model '%s') stopped at the output token limit " "(%s)",
            self.prompt_name,
            self.purpose.value,
            self.model_id,
            reason,
        )

    def _check(self, message: Any) -> Any:
        if reason := _truncation_reason(message):
            self._warn(reason)
            raise LLMOutputTruncatedError(
                f"'{self.prompt_name}' response from model '{self.model_id}' "
                f"was cut at the output token limit"
            )
        return message

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
            self._warn(reason)

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
            self._warn(reason)


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


class XMLOutputFixingParser(OutputFixingParser[dict[str, Any]]):
    """``OutputFixingParser`` that never asks the fixer to repair a blank answer.

    An empty or whitespace-only completion (e.g. a thinking-only response)
    carries nothing to repair, so a fixer could only invent the structure;
    it fails with the parser's own ``OutputParserException`` instead, which
    the batch retry treats as retryable.
    """

    def parse(self, completion: str) -> dict[str, Any]:
        if not completion.strip():
            return dict(self.parser.parse(completion))
        return super().parse(completion)

    async def aparse(self, completion: str) -> dict[str, Any]:
        if not completion.strip():
            return dict(await self.parser.aparse(completion))
        return await super().aparse(completion)


def create_robust_xml_output_parser(
    factory: LLMFactoryPort,
    enable_output_fixing: bool,
    output_fixing_model_id: str,
    *,
    output_tags: list[str],
    model_purpose: ModelPurpose = ModelPurpose.QUERY,
    min_output_tokens: int = 0,
    parser_class: type[RobustXMLOutputParser] = RobustXMLOutputParser,
) -> BaseOutputParser:
    """Build the XML parser (a ``parser_class``), optionally wrapped in an
    LLM output fixer.

    ``output_tags`` are the top-level elements the prompt asks for; the fixer
    is told them, so it repairs toward the prompt's structure. ``model_purpose``
    is forwarded to the fixing LLM so it gets the same per-path policy (the
    response cache, the ingestion effort) as the chain it repairs, and ``min_output_tokens`` (the
    repaired prompt's output floor) so a long output can be re-emitted in full.
    """
    base_parser = parser_class(tags=output_tags)
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
        fixer = XMLOutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)
        _serialize_once(*getattr(fixer.retry_chain, "steps", ()))
        return fixer
    except Exception as e:
        logger.error(
            "Failed to create OutputFixingParser with model %s: %s",
            output_fixing_model_id,
            e,
        )
        raise GraphRAGException(f"Failed to create OutputFixingParser: {e}") from e


def setup_chain(
    factory: LLMFactoryPort,
    model_id: str,
    prompt_class: type[BasePrompt],
    parser: BaseOutputParser,
    custom_prompts: CustomPromptConfig | None = None,
    model_purpose: ModelPurpose = ModelPurpose.QUERY,
    min_output_tokens: int | None = None,
    **kwargs: Any,
) -> Runnable:
    """Build ``prompt | llm | parser`` with the policies of ``model_purpose``.

    ``QUERY`` (the default) gets, when the factory classifies transient
    errors and has a chain retry policy (``is_transient_error`` and
    ``transient_retry``, the Bedrock and Ollama factories), a transient-error
    retry around the whole chain. Ingestion and evaluation call sites pass
    their purpose explicitly: they get no chain-level retry, because ``BatchProcessor`` already retries them and a
    second layer would multiply the attempts. A factory without a
    ``transient_retry`` policy (the OpenAI-compatible one) leaves query
    retries to its client. ``min_output_tokens`` overrides the prompt's static
    floor with one derived from the config (``BasePrompt.output_floor``).
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
        cache_marker = _prompt_cache_marker(llm, model_info, resolved)
        prompt = _build_chat_prompt(
            resolved,
            cache_marker,
            _prompt_cache_ttl(factory, model_info, cache_marker, model_purpose),
        )
        _serialize_once(prompt, llm)
        guard = TruncationGuard(prompt_class.__name__, model_id, model_purpose)
        # Named after the prompt, so a trace shows "AnswerGenerationPrompt"
        # instead of an anonymous "RunnableSequence".
        chain: Runnable = (prompt | llm | guard | parser).with_config(
            run_name=prompt_class.__name__
        )
        logger.debug("Successfully created LLM chain with model: '%s'", model_id)
        is_transient = getattr(factory, "is_transient_error", None)
        if model_purpose is not ModelPurpose.QUERY or is_transient is None:
            return chain
        return with_transient_retry(
            chain,
            operation=prompt_class.__name__,
            retry=getattr(factory, "transient_retry", None),
            is_transient=is_transient,
        )
    except Exception as e:
        logger.error("Failed to setup LLM chain with model '%s': %s", model_id, e)
        raise GraphRAGException(
            f"Failed to setup LLM chain with model '{model_id}': {e}"
        ) from e


# Chat-model classes whose ``with_structured_output(method="json_schema")``
# sends a native JSON-schema response format (constrained decoding):
# Converse ``outputConfig.textFormat``, Ollama ``format``, OpenAI
# ``response_format``. Matched by class name so the optional local
# integrations are never imported here.
_JSON_SCHEMA_CHAT_CLASSES = frozenset(
    {"ChatBedrockConverse", "ChatOllama", "ChatOpenAI"}
)


def structured_output_unsupported_reason(llm: Any, model_info: Any) -> str | None:
    """Why ``llm`` cannot take a native JSON-schema output format, or None.

    - ``ChatBedrock`` (InvokeModel) offers only tool-based structured output,
      which needs a forced tool choice that Claude 5.x rejects.
    - On Converse, Bedrock documents the format for Claude 4.5 and later; an
      Anthropic request that already carries ``output_config`` (the adaptive
      effort, ``additional_model_request_fields``) is left out: the
      Anthropic-native form of the format is ``output_config.format``, the
      object the effort rides in, and whether Bedrock merges the two is
      unverified.
    """
    names = {cls.__name__ for cls in type(llm).__mro__}
    if not names & _JSON_SCHEMA_CHAT_CLASSES:
        return f"{type(llm).__name__} has no JSON-schema output format"
    if "ChatBedrockConverse" not in names:
        return None
    if not getattr(model_info, "supports_structured_output", False):
        return "the model has no Bedrock structured-output support"
    fields = getattr(llm, "additional_model_request_fields", None) or {}
    if "output_config" in fields:
        return (
            "the request already carries output_config (reasoning effort); "
            "combining it with outputConfig.textFormat is unverified"
        )
    return None


def setup_structured_chain(
    factory: LLMFactoryPort,
    model_id: str,
    prompt_class: type[BasePrompt],
    schema: dict[str, Any],
    custom_prompts: CustomPromptConfig | None = None,
    model_purpose: ModelPurpose = ModelPurpose.QUERY,
) -> Runnable | None:
    """``prompt | llm.with_structured_output(schema, method="json_schema")``.

    Returns None when the model cannot take the format
    (:func:`structured_output_unsupported_reason`), so the caller keeps its
    text-parsing chain. The chain yields the parsed object (a dict). Query
    chains get the same transient-error retry as :func:`setup_chain`.
    """
    llm = factory.get_model(
        model_id=model_id,
        model_purpose=model_purpose,
        min_output_tokens=prompt_class.min_output_tokens,
    )
    model_info = factory.get_model_info(model_id)
    reason = structured_output_unsupported_reason(llm, model_info)
    if reason is not None:
        logger.info(
            "Structured output off for %s on '%s': %s",
            prompt_class.__name__,
            model_id,
            reason,
        )
        return None
    resolved = prompt_class.resolve(custom_prompts=custom_prompts)
    cache_marker = _prompt_cache_marker(llm, model_info, resolved)
    prompt = _build_chat_prompt(
        resolved,
        cache_marker,
        _prompt_cache_ttl(factory, model_info, cache_marker, model_purpose),
    )
    structured = llm.with_structured_output(schema, method="json_schema")
    _serialize_once(prompt, llm)
    chain: Runnable = (prompt | structured).with_config(
        run_name=f"{prompt_class.__name__}Structured"
    )
    is_transient = getattr(factory, "is_transient_error", None)
    if model_purpose is not ModelPurpose.QUERY or is_transient is None:
        return chain
    return with_transient_retry(
        chain,
        operation=prompt_class.__name__,
        retry=getattr(factory, "transient_retry", None),
        is_transient=is_transient,
    )
