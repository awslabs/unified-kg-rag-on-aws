# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bedrock-coupled LangChain assembly helpers (adapters layer).

These build ``prompt | llm | parser`` chains from a concrete
``BedrockLanguageModelFactory``, so they live in the adapters layer rather than
the shared kernel — keeping ``shared/`` free of any adapter dependency. The
backend-agnostic ``RobustXMLOutputParser`` stays in ``shared.utils.langchain``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.output_parsers import OutputFixingParser
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable

from unified_kg_rag.domain.models import LanguageModelId
from unified_kg_rag.domain.prompts import BasePrompt, ResolvedPrompt
from unified_kg_rag.ports.model_factory import LLMFactoryPort
from unified_kg_rag.shared import GraphRAGException, get_logger
from unified_kg_rag.shared.utils.langchain import RobustXMLOutputParser

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import CustomPromptConfig

logger = get_logger(__name__)


def _build_chat_prompt(
    resolved: ResolvedPrompt, enable_prompt_cache: bool
) -> ChatPromptTemplate:
    """Assemble a LangChain ChatPromptTemplate from a backend-agnostic prompt.

    Lives in the adapter layer: turning the domain's ResolvedPrompt into
    LangChain message templates is a backend concern. When prompt caching is
    enabled, the system message carries an ephemeral cache_control marker.
    """
    messages: list[Any]
    if enable_prompt_cache:
        system_msg = SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": resolved.system_prompt_template,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )
        messages = [
            system_msg,
            HumanMessagePromptTemplate.from_template(resolved.human_prompt_template),
        ]
    else:
        messages = [
            SystemMessagePromptTemplate.from_template(resolved.system_prompt_template),
            HumanMessagePromptTemplate.from_template(resolved.human_prompt_template),
        ]
    return ChatPromptTemplate.from_messages(messages)


def create_robust_xml_output_parser(
    factory: LLMFactoryPort,
    enable_output_fixing: bool,
    output_fixing_model_id: LanguageModelId,
) -> BaseOutputParser:
    base_parser = RobustXMLOutputParser()
    if not enable_output_fixing:
        return base_parser

    try:
        fixing_llm = factory.get_model(model_id=output_fixing_model_id)
        logger.info(
            "Created OutputFixingParser with model: '%s'", output_fixing_model_id.value
        )
        return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)
    except Exception as e:
        logger.error(
            "Failed to create OutputFixingParser with model %s: %s",
            output_fixing_model_id.value,
            e,
        )
        raise GraphRAGException(f"Failed to create OutputFixingParser: {e}") from e


def setup_chain(
    factory: LLMFactoryPort,
    model_id: LanguageModelId,
    prompt_class: type[BasePrompt],
    parser: BaseOutputParser,
    custom_prompts: CustomPromptConfig | None = None,
    **kwargs: Any,
) -> Runnable:
    try:
        llm = factory.get_model(model_id=model_id, **kwargs)
        model_info = factory.get_model_info(model_id)
        enable_prompt_cache = (
            model_info.supports_prompt_caching if model_info else False
        )
        resolved = prompt_class.resolve(custom_prompts=custom_prompts)
        prompt = _build_chat_prompt(resolved, enable_prompt_cache)
        chain: Runnable = prompt | llm | parser
        logger.debug("Successfully created LLM chain with model: '%s'", model_id.value)
        return chain
    except Exception as e:
        logger.error("Failed to setup LLM chain with model '%s': %s", model_id.value, e)
        raise GraphRAGException(
            f"Failed to setup LLM chain with model '{model_id.value}': {e}"
        ) from e
