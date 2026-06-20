# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Bedrock-coupled LangChain assembly helpers (adapters layer).

These build ``prompt | llm | parser`` chains from a concrete
``BedrockLanguageModelFactory``, so they live in the adapters layer rather than
the shared kernel — keeping ``shared/`` free of any adapter dependency. The
backend-agnostic ``RobustXMLOutputParser`` stays in ``shared.utils.langchain``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable

from aws_graphrag.adapters.aws import BedrockLanguageModelFactory
from aws_graphrag.domain.models import LanguageModelId
from aws_graphrag.domain.prompts import BasePrompt
from aws_graphrag.shared import GraphRAGException, get_logger
from aws_graphrag.shared.utils.langchain import RobustXMLOutputParser

if TYPE_CHECKING:
    from aws_graphrag.domain.models.config import CustomPromptConfig

logger = get_logger(__name__)


def create_robust_xml_output_parser(
    factory: BedrockLanguageModelFactory,
    enable_output_fixing: bool,
    output_fixing_model_id: LanguageModelId,
) -> BaseOutputParser:
    base_parser = RobustXMLOutputParser()
    if not enable_output_fixing:
        return base_parser

    try:
        fixing_llm = factory.get_model(model_id=output_fixing_model_id)
        logger.info(
            f"Created OutputFixingParser with model: '{output_fixing_model_id.value}'"
        )
        return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)
    except Exception as e:
        logger.error(
            f"Failed to create OutputFixingParser with model {output_fixing_model_id.value}: {e}"
        )
        raise GraphRAGException(f"Failed to create OutputFixingParser: {e}") from e


def setup_chain(
    factory: BedrockLanguageModelFactory,
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
        prompt = prompt_class.get_prompt(
            enable_prompt_cache=enable_prompt_cache, custom_prompts=custom_prompts
        )
        chain = prompt | llm | parser
        logger.debug(f"Successfully created LLM chain with model: '{model_id.value}'")
        return chain
    except Exception as e:
        logger.error(f"Failed to setup LLM chain with model '{model_id.value}': {e}")
        raise GraphRAGException(
            f"Failed to setup LLM chain with model '{model_id.value}': {e}"
        ) from e
