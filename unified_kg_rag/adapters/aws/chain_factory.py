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
from typing import TYPE_CHECKING, Any

from langchain_classic.output_parsers import OutputFixingParser
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
from unified_kg_rag.domain.models import ModelPurpose
from unified_kg_rag.domain.prompts import BasePrompt, ResolvedPrompt
from unified_kg_rag.ports.model_factory import LLMFactoryPort
from unified_kg_rag.shared import GraphRAGException, get_logger
from unified_kg_rag.shared.utils.langchain import RobustXMLOutputParser

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import (
        CustomPromptConfig,
        TransientRetryConfig,
    )

logger = get_logger(__name__)


def _build_chat_prompt(
    resolved: ResolvedPrompt, enable_prompt_cache: bool
) -> ChatPromptTemplate:
    """Assemble a LangChain ChatPromptTemplate from a backend-agnostic prompt.

    Lives in the adapter layer: turning the domain's ResolvedPrompt into
    LangChain message templates is a backend concern. When prompt caching is
    enabled, the system message carries an ephemeral cache_control marker.
    """
    system_template: str | list[str | dict[str, Any]]
    if enable_prompt_cache:
        # A templated content block (not a literal SystemMessage) so system-side
        # placeholders such as {entity_types} are substituted and escaped braces
        # render exactly as in the non-cache path; LangChain keeps extra block
        # keys, so the cache_control marker survives formatting.
        system_template = [
            {
                "type": "text",
                "text": resolved.system_prompt_template,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_template = resolved.system_prompt_template
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(resolved.human_prompt_template),
    ]
    return ChatPromptTemplate.from_messages(messages)


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


def create_robust_xml_output_parser(
    factory: LLMFactoryPort,
    enable_output_fixing: bool,
    output_fixing_model_id: str,
    model_purpose: ModelPurpose = ModelPurpose.QUERY,
) -> BaseOutputParser:
    """Build the XML parser, optionally wrapped in an LLM output fixer.

    ``model_purpose`` is forwarded to the fixing LLM so it gets the same
    per-path policy (e.g. guardrail scope) as the chain it repairs.
    """
    base_parser = RobustXMLOutputParser()
    if not enable_output_fixing:
        return base_parser

    try:
        fixing_llm = factory.get_model(
            model_id=output_fixing_model_id, model_purpose=model_purpose
        )
        logger.info(
            "Created OutputFixingParser with model: '%s'", output_fixing_model_id
        )
        return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)
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
    own retry policy.
    """
    try:
        llm = factory.get_model(
            model_id=model_id, model_purpose=model_purpose, **kwargs
        )
        model_info = factory.get_model_info(model_id)
        enable_prompt_cache = (
            model_info.supports_prompt_caching if model_info else False
        )
        resolved = prompt_class.resolve(custom_prompts=custom_prompts)
        prompt = _build_chat_prompt(resolved, enable_prompt_cache)
        chain: Runnable = prompt | llm | parser
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
