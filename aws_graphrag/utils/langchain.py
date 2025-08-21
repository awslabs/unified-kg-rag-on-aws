import asyncio
import html
import math
import re
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tenacity
from langchain.output_parsers import OutputFixingParser, XMLOutputParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from lxml import etree
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from aws_graphrag.aws import BedrockLanguageModelFactory
from aws_graphrag.core import GraphRAGException, get_logger
from aws_graphrag.models import LanguageModelId
from aws_graphrag.prompts import BasePrompt

if TYPE_CHECKING:
    from ..models.config import CustomPromptConfig

logger = get_logger(__name__)


class BatchProcessor(BaseModel):
    max_concurrency: int = Field(
        default=5,
        ge=1,
        description="Maximum number of operations that can run concurrently during batch processing",
    )
    retry_multiplier: float = Field(
        default=30.0,
        ge=1.0,
        description="Base multiplier for exponential backoff retry delays in seconds",
    )
    retry_max_wait: int = Field(
        default=120,
        ge=0,
        description="Maximum allowed wait time between retry attempts in seconds",
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        description="Maximum number of retry attempts before giving up on a failed operation",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Size of each mini-batch when processing items in chunks for better memory management and performance",
    )

    def execute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., list[Any]],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []

        if run_config:
            self.max_concurrency = run_config.get(
                "max_concurrency", self.max_concurrency
            )
            self.batch_size = run_config.get("batch_size", self.batch_size)

        prepared_batch_func = self._create_batch_func(batch_func)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )

        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / self.batch_size)

        logger.info(
            f"Starting processing for '{task_name}': {num_items} items in {num_chunks} chunks "
            f"(batch size: {self.batch_size})"
        )

        for i in tqdm(
            range(0, num_items, self.batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        ):
            chunk_items = items_to_process[i : i + self.batch_size]
            chunk_num = (i // self.batch_size) + 1

            logger.debug(
                f"Processing chunk {chunk_num}/{num_chunks} ({len(chunk_items)} items)"
            )

            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    f"No valid inputs prepared for chunk {chunk_num}, skipping"
                )
                continue

            try:
                logger.debug(f"Attempting batch processing for chunk {chunk_num}")
                chunk_results = prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
                logger.debug(f"Chunk {chunk_num} processed successfully in batch mode")
            except Exception as e:
                logger.warning(
                    f"Batch processing failed for chunk {chunk_num}: {e}. "
                    f"Falling back to sequential processing"
                )
                chunk_results = self._process_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress=show_progress,
                )
                all_results.extend(chunk_results)

        logger.info(f"Completed '{task_name}': processed {len(all_results)} results")
        return all_results

    def _create_batch_func(self, batch_func: Callable[..., list[Any]]) -> Callable:
        def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return batch_func(
                inputs, config=RunnableConfig(max_concurrency=self.max_concurrency)
            )

        return _batch_func

    def _create_retry_decorator(self, operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=self.retry_multiplier, max=self.retry_max_wait
            ),
            stop=tenacity.stop_after_attempt(self.max_retries),
            before_sleep=self._create_retry_log_callback(operation_name),
            reraise=True,
        )

    @staticmethod
    def _create_retry_log_callback(operation_name: str) -> Callable:
        def log_retry(retry_state):
            wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
            logger.warning(
                f"Retrying '{operation_name}' (attempt {retry_state.attempt_number} failed). Waiting {wait_time:.1f}s"
            )

        return log_retry

    @staticmethod
    def _process_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info(f"Processing {len(inputs)} items sequentially for '{task_name}'")

        results = []
        progress_desc = f"Sequential Processing: '{task_name}'"
        successful_count = 0

        for single_input in tqdm(inputs, desc=progress_desc, disable=not show_progress):
            try:
                result = sequential_func(single_input)
                results.append(result)
                successful_count += 1
            except Exception as e:
                logger.error(
                    f"Sequential processing failed for single item in '{task_name}': {e}"
                )
                continue

        logger.info(
            f"Sequential processing completed for '{task_name}': {successful_count}/{len(inputs)} items processed successfully"
        )
        return results

    async def aexecute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., Any],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []

        if run_config:
            self.max_concurrency = run_config.get(
                "max_concurrency", self.max_concurrency
            )
            self.batch_size = run_config.get("batch_size", self.batch_size)

        prepared_batch_func = self._create_async_batch_func(batch_func)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )

        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / self.batch_size)

        logger.info(
            f"Starting async processing for '{task_name}': {num_items} items in {num_chunks} chunks "
            f"(batch size: {self.batch_size})"
        )

        chunk_iterator = async_tqdm(
            range(0, num_items, self.batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        )

        for i in chunk_iterator:
            chunk_items = items_to_process[i : i + self.batch_size]
            chunk_num = (i // self.batch_size) + 1
            logger.debug(
                f"Processing chunk {chunk_num}/{num_chunks} ({len(chunk_items)} items)"
            )

            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    f"No valid inputs prepared for chunk {chunk_num}, skipping"
                )
                continue

            try:
                chunk_results = await prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
            except Exception as e:
                logger.warning(
                    f"Async batch processing failed for chunk {chunk_num}: {e}. "
                    f"Falling back to concurrent sequential processing"
                )
                chunk_results = await self._aprocess_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress,
                )
                all_results.extend(chunk_results)

        logger.info(f"Completed '{task_name}': processed {len(all_results)} results")
        return all_results

    def _create_async_batch_func(self, batch_func: Callable[..., Any]) -> Callable:
        async def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return await batch_func(
                inputs, config=RunnableConfig(max_concurrency=self.max_concurrency)
            )

        return _batch_func

    async def _aprocess_sequentially_with_fallback(
        self,
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info(f"Processing {len(inputs)} items concurrently for '{task_name}'")
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_one(single_input):
            async with semaphore:
                try:
                    return await sequential_func(single_input)
                except Exception as e:
                    logger.error(
                        f"Concurrent sequential processing failed for item in '{task_name}': {e}"
                    )
                    return None

        tasks = [_process_one(single_input) for single_input in inputs]

        progress_desc = f"Concurrent Fallback: '{task_name}'"
        results = await async_tqdm.gather(
            *tasks, disable=not show_progress, desc=progress_desc
        )

        successful_results = [res for res in results if res is not None]
        logger.info(
            f"Concurrent sequential processing completed for '{task_name}': {len(successful_results)}/{len(inputs)} items processed successfully"
        )
        return successful_results


class RobustXMLOutputParser(XMLOutputParser):
    def parse(self, text: str) -> dict[str, Any]:
        original_sections = self._detect_xml_sections(text)

        try:
            result = super().parse(text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in parsed result")
        except Exception:
            pass

        try:
            cleaned_text = self._clean_xml_for_lxml(text)
            result = self._try_lxml_recover_parse(cleaned_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in lxml result")
        except Exception:
            pass

        try:
            sanitized_text = self._sanitize_xml_content(text)
            result = super().parse(sanitized_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in sanitized result")
        except Exception:
            pass

        try:
            aggressively_cleaned = self._aggressively_clean_xml(text)
            result = super().parse(aggressively_cleaned)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in aggressive result")
        except Exception:
            pass

        try:
            fallback_result = self._extract_xml_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception:
            pass

        try:
            fallback_result = self._extract_tags_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception:
            pass

        try:
            fallback_result = self._extract_list_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception:
            pass

        logger.error(f"All XML parsing attempts failed for content: '{text[:200]}...'")
        raise ValueError(
            f"Failed to parse XML after multiple attempts. Content preview: '{text[:200]}...'"
        )

    @staticmethod
    def _detect_xml_sections(text: str) -> set[str]:
        pattern = r"<([a-zA-Z0-9_]+)>.*?</\1>"
        matches = re.findall(pattern, text, re.DOTALL)
        return set(matches)

    @staticmethod
    def _sections_preserved(
        original_sections: set[str], parsed_result: dict[str, Any]
    ) -> bool:
        if not original_sections:
            return True

        parsed_sections = (
            set(parsed_result.keys()) if isinstance(parsed_result, dict) else set()
        )
        missing_sections = original_sections - parsed_sections

        if missing_sections:
            return False
        return True

    @staticmethod
    def _extract_xml_fallback(text: str) -> dict[str, Any] | None:
        result = {}

        try:
            section_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
            section_matches = re.findall(section_pattern, text, re.DOTALL)

            for section_name, section_content in section_matches:
                section_result = RobustXMLOutputParser._parse_xml_section(
                    section_content
                )
                if section_result is not None:
                    result[section_name] = section_result

            return result if result else None

        except Exception:
            return None

    @staticmethod
    def _parse_xml_section(
        content: str,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        content = content.strip()
        if not content:
            return None

        child_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
        child_matches = re.findall(child_pattern, content, re.DOTALL)

        if not child_matches:
            return {"#text": content}

        children_by_tag = defaultdict(list)
        for child_tag, child_content in child_matches:
            parsed_child = RobustXMLOutputParser._parse_xml_element(child_content)
            children_by_tag[child_tag].append(parsed_child)

        result = {}
        for tag, children in children_by_tag.items():
            result[tag] = children[0] if len(children) == 1 else children

        if len(children_by_tag) == 1:
            child_tag = list(children_by_tag.keys())[0]
            children = children_by_tag[child_tag]
            if len(children) > 1:
                return {child_tag: children}

        return result

    @staticmethod
    def _parse_xml_element(content: str) -> dict[str, Any] | str:
        content = content.strip()
        if not content:
            return ""

        nested_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
        nested_matches = re.findall(nested_pattern, content, re.DOTALL)

        if not nested_matches:
            return content

        result = {}
        for nested_tag, nested_content in nested_matches:
            parsed_nested = RobustXMLOutputParser._parse_xml_element(nested_content)

            if nested_tag in result:
                if not isinstance(result[nested_tag], list):
                    result[nested_tag] = [result[nested_tag]]
                result[nested_tag].append(parsed_nested)
            else:
                result[nested_tag] = parsed_nested

        text_content = content
        for nested_tag, nested_content in nested_matches:
            full_nested = f"<{nested_tag}>{nested_content}</{nested_tag}>"
            text_content = text_content.replace(full_nested, "").strip()

        if text_content and result:
            result["#text"] = text_content
        elif text_content and not result:
            return text_content

        return result

    @staticmethod
    def _clean_xml_for_lxml(text: str) -> bytes:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip().encode("utf-8")

    @staticmethod
    def _try_lxml_recover_parse(xml_bytes: bytes) -> dict[str, Any]:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        tree = etree.fromstring(xml_bytes, parser=parser)

        if tree is None:
            raise ValueError("lxml parser recovered a null tree")

        def _convert_etree_to_dict(element: etree._Element) -> dict[str, Any]:
            result = {}
            children = list(element)

            if children:
                child_dict = defaultdict(list)
                for child in children:
                    child_result = _convert_etree_to_dict(child)
                    for key, value in child_result.items():
                        child_dict[key].append(value)

                processed_children = {
                    key: val[0] if len(val) == 1 else val
                    for key, val in child_dict.items()
                }
                result[element.tag] = processed_children
            else:
                result[element.tag] = {}

            if element.attrib:
                if not isinstance(result[element.tag], dict):
                    result[element.tag] = {"#text": result[element.tag]}
                result[element.tag].update(
                    {f"@{k}": v for k, v in element.attrib.items()}
                )

            if element.text and element.text.strip():
                text = element.text.strip()
                if not result[element.tag]:
                    result[element.tag] = text
                elif isinstance(result[element.tag], dict):
                    if "#text" not in result[element.tag]:
                        result[element.tag]["#text"] = text

            if not result[element.tag]:
                result[element.tag] = ""

            return result

        return _convert_etree_to_dict(tree)

    @staticmethod
    def _sanitize_xml_content(xml_content: str) -> str:
        def escape_text_content(match: re.Match) -> str:
            tag_open = match.group(1)
            content = match.group(2)
            tag_close = match.group(3)
            escaped_content = html.escape(content, quote=False)
            return f"{tag_open}{escaped_content}{tag_close}"

        pattern = r"(<[a-zA-Z0-9_]+\s*[^>]*>)(.*?)(</[a-zA-Z0-9_]+>)"
        return re.sub(pattern, escape_text_content, xml_content, flags=re.DOTALL)

    @staticmethod
    def _aggressively_clean_xml(xml_content: str) -> str:
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml_content)
        cleaned = re.sub(r"&(?!(?:amp|lt|gt|quot|apos);)", "&amp;", cleaned)

        def selective_escape(match: re.Match) -> str:
            return html.escape(match.group(0), quote=False)

        cleaned = re.sub(r">([^<]*)<", selective_escape, cleaned)
        return cleaned.strip()

    @staticmethod
    def _extract_tags_fallback(text: str) -> dict[str, Any] | None:
        pattern = re.compile(r"<([a-zA-Z0-9_]+)\s*.*?>(.*?)</\1>", re.DOTALL)
        matches = pattern.findall(text)

        if not matches:
            return None

        content_map = defaultdict(list)
        for tag, content in matches:
            stripped_content = content.strip()
            if stripped_content:
                content_map[tag].append(stripped_content)

        if not content_map:
            return None

        result = {
            key: val[0] if len(val) == 1 else val for key, val in content_map.items()
        }

        return result

    @staticmethod
    def _extract_list_fallback(text: str) -> dict[str, Any] | None:
        item_patterns = [
            r"^\s*[•\-\*]\s*(.+?)(?=\n\s*[•\-\*]|\Z)",
            r"^\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)",
        ]

        for pattern in item_patterns:
            items = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if items:
                stripped_items = [item.strip() for item in items if item.strip()]
                if stripped_items:
                    return {"items": stripped_items}

        return None


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
) -> Runnable:
    try:
        llm = factory.get_model(model_id=model_id)
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


def setup_chain_with_output_fixing(
    factory: BedrockLanguageModelFactory,
    model_id: LanguageModelId,
    prompt_class: type[BasePrompt],
    parser: BaseOutputParser,
    custom_prompts=None,
) -> Runnable:
    try:
        llm = factory.get_model(model_id=model_id)
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
