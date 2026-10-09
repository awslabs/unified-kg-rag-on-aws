# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bedrock-coupled LangChain assembly helpers (adapters layer).

These build ``prompt | llm | parser`` chains from a concrete
``BedrockLanguageModelFactory``, so they live in the adapters layer rather than
the shared kernel — keeping ``shared/`` free of any adapter dependency. The
backend-agnostic ``RobustXMLOutputParser`` stays in ``shared.utils.langchain``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Literal

from langchain_aws import ChatBedrockConverse
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.bedrock_retry import (
    acall_with_transient_retry,
    call_with_transient_retry,
    next_transient_retry_delay,
)
from unified_kg_rag.adapters.aws.token_counter import estimate_token_count
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
from unified_kg_rag.shared.utils.langchain import RobustXMLOutputParser

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
    if isinstance(llm, ChatBedrockConverse):
        return "cache_point"
    return "cache_control"


def _build_chat_prompt(
    resolved: ResolvedPrompt, cache_marker: PromptCacheMarker | None
) -> ChatPromptTemplate:
    """Assemble a LangChain ChatPromptTemplate from a backend-agnostic prompt.

    Lives in the adapter layer: turning the domain's ResolvedPrompt into
    LangChain message templates is a backend concern. ``cache_marker`` marks
    the end of the system prompt as a prompt-cache checkpoint.
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
        system_template = [
            {"type": "text", "text": resolved.system_prompt_template},
            dict(_CACHE_POINT_BLOCK),
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


# Stop reason of a response cut at its output-token limit, under the key
# langchain-aws reports it: Converse ``stopReason``, InvokeModel ``stop_reason``.
_TRUNCATED_STOP_REASON = "max_tokens"


def _stop_reason(message: Any) -> str | None:
    metadata = getattr(message, "response_metadata", None) or {}
    reason = metadata.get("stopReason") or metadata.get("stop_reason")
    return str(reason) if reason else None


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

    def _warn(self) -> None:
        logger.warning(
            "'%s' response (%s, model '%s') stopped at the output token limit "
            "(stop reason %s)",
            self.prompt_name,
            self.purpose.value,
            self.model_id,
            _TRUNCATED_STOP_REASON,
        )

    def _check(self, message: Any) -> Any:
        if _stop_reason(message) == _TRUNCATED_STOP_REASON:
            self._warn()
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
        truncated = False
        for chunk in input:
            truncated = truncated or _stop_reason(chunk) == _TRUNCATED_STOP_REASON
            yield chunk
        if truncated:
            self._warn()

    async def atransform(
        self,
        input: AsyncIterator[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        truncated = False
        async for chunk in input:
            truncated = truncated or _stop_reason(chunk) == _TRUNCATED_STOP_REASON
            yield chunk
        if truncated:
            self._warn()


class TransientRetryRunnable(Runnable[Any, Any]):
    """Retry a chain on transient Bedrock errors (predicate-based ``with_retry``).

    LangChain's ``Runnable.with_retry`` only matches exception *types*, but a
    transient Bedrock fault is a ``botocore.exceptions.ClientError`` whose
    error code decides retryability (424 ``ModelErrorException`` yes,
    ``ValidationException`` no). This wrapper applies the same
    :mod:`~unified_kg_rag.adapters.aws.bedrock_retry` policy as the embedding
    path. Non-transient errors and exhausted retries re-raise the original
    exception unchanged.

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
    ) -> None:
        self.bound = bound
        self.operation = operation
        self.retry = retry

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
            operation=self.operation,
            policy=self.retry,
        )

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return await acall_with_transient_retry(
            lambda: self.bound.ainvoke(input, config, **kwargs),
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
) -> Runnable[Any, Any]:
    """Wrap ``runnable`` in :class:`TransientRetryRunnable` unless retry is off."""
    if retry is None or retry.max_attempts <= 1:
        return runnable
    return TransientRetryRunnable(runnable, operation=operation, retry=retry)


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
) -> BaseOutputParser:
    """Build the XML parser, optionally wrapped in an LLM output fixer.

    ``output_tags`` are the top-level elements the prompt asks for; the fixer
    is told them, so it repairs toward the prompt's structure. ``model_purpose``
    is forwarded to the fixing LLM so it gets the same per-path policy (e.g.
    guardrail scope) as the chain it repairs, and ``min_output_tokens`` (the
    repaired prompt's output floor) so a long output can be re-emitted in full.
    """
    base_parser = RobustXMLOutputParser(tags=output_tags)
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
        return XMLOutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)
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

    ``QUERY`` (the default, so an unmarked call site stays on the guarded,
    conservative side) gets the query guardrail scope (``guardrail.apply_to``,
    applied by the factory) and, on a Bedrock factory, a transient-error retry
    around the whole chain. Ingestion and evaluation call sites pass their
    purpose explicitly: they run unguarded under ``apply_to: query`` and get no
    chain-level retry, because ``BatchProcessor`` already retries them and a
    second layer would multiply the attempts. A non-Bedrock factory owns its
    own retry policy. ``min_output_tokens`` overrides the prompt's static
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
        prompt = _build_chat_prompt(
            resolved, _prompt_cache_marker(llm, model_info, resolved)
        )
        # Named after the prompt, so a trace shows "AnswerGenerationPrompt"
        # instead of an anonymous "RunnableSequence".
        guard = TruncationGuard(prompt_class.__name__, model_id, model_purpose)
        chain: Runnable = (prompt | llm | guard | parser).with_config(
            run_name=prompt_class.__name__
        )
        logger.debug("Successfully created LLM chain with model: '%s'", model_id)
        retry = (
            factory.config.aws.bedrock.transient_retry
            if model_purpose is ModelPurpose.QUERY
            and isinstance(factory, BedrockLanguageModelFactory)
            else None
        )
        return with_transient_retry(chain, operation=prompt_class.__name__, retry=retry)
    except Exception as e:
        logger.error("Failed to setup LLM chain with model '%s': %s", model_id, e)
        raise GraphRAGException(
            f"Failed to setup LLM chain with model '{model_id}': {e}"
        ) from e
