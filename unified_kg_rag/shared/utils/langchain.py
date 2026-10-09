# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

import tenacity
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.runnables import RunnableConfig
from lxml import etree
from pydantic import BaseModel, Field
from tenacity import RetryCallState
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.common import text_digest
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class _BatchItemFailed:
    """Type of :data:`BATCH_ITEM_FAILED`; there is only one instance."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BATCH_ITEM_FAILED"


BATCH_ITEM_FAILED: Final = _BatchItemFailed()
"""Result placeholder for an item that failed every attempt.

``BatchProcessor`` keeps its results 1:1 with the inputs, so a failed item is
returned in place as this marker. Test it with ``result is BATCH_ITEM_FAILED``;
an empty result (``{}``, ``""``) is a real LLM output, not a failure.
"""


class ProgressLogger:
    """Log ``done/total``, rate and ETA at INFO every ~10% or 60s.

    tqdm is disabled without a TTY (e.g. Fargate/CloudWatch, where its
    carriage-return redraws become junk lines), so this is the progress signal
    an operator sees in the logs.
    """

    def __init__(
        self,
        task_name: str,
        total: int,
        step_fraction: float = 0.1,
        interval_seconds: float = 60.0,
    ) -> None:
        self.task_name = task_name
        self.total = total
        self.step = max(1, math.ceil(total * step_fraction))
        self.interval_seconds = interval_seconds
        self.done = 0
        self._next_mark = self.step
        self._start = self._last = time.monotonic()

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.monotonic()
        if self.done < self._next_mark and now - self._last < self.interval_seconds:
            return
        self._last = now
        self._next_mark = (self.done // self.step + 1) * self.step
        elapsed = now - self._start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        logger.info(
            "Progress '%s': %s/%s items (%.0f%%), %.2f items/s, ETA %.0fs",
            self.task_name,
            self.done,
            self.total,
            100.0 * self.done / self.total if self.total else 100.0,
            rate,
            eta,
        )


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
    max_attempts: int = Field(
        default=5,
        ge=1,
        description="Calls an item gets after its first call fails: at least "
        "one, and up to this many while its error is retryable",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Size of each mini-batch when processing items in chunks for better memory management and performance",
    )
    chunk_concurrency: int = Field(
        default=4,
        ge=1,
        description="How many mini-batch chunks to run concurrently. These stages "
        "are Bedrock-I/O-bound (CPU/memory near-idle), so overlapping chunks' "
        "network waits cuts wall-clock time. 1 = strictly serial (legacy).",
    )
    call_timeout_seconds: int = Field(
        default=300,
        ge=0,
        description="Wall-clock timeout for one item's LLM call. botocore's "
        "read_timeout only measures the gap between bytes, so a server that "
        "dribbles keep-alive data can hang a call indefinitely; this hard "
        "ceiling aborts such a call and retries that item alone. Keep it below "
        "the Bedrock client's read timeout so it is the limit that fires. "
        "0 disables.",
    )
    is_transient_error: Callable[[BaseException], bool] | None = Field(
        default=None,
        description="Classifies backend errors worth retrying (e.g. "
        "adapters.aws.bedrock_retry.is_transient_bedrock_error); other errors "
        "fail fast. Output-parse failures and call timeouts are always retried. "
        "None retries every error, for callers that span several backends.",
    )

    @staticmethod
    def _run_with_timeout(
        func: Callable[[], Any], timeout_seconds: int, label: str
    ) -> Any:
        """Run ``func`` under a wall-clock timeout.

        A hung Bedrock call (no completion despite an open socket) would otherwise
        block its worker indefinitely; this bounds it so the item can be retried.
        ``timeout_seconds <= 0`` runs ``func`` directly with no timeout.

        Python cannot kill a thread, so a timed-out call is abandoned, not
        cancelled: its in-flight Bedrock request keeps running (and billing)
        until botocore's own read timeout or completion, while the retry
        re-issues that one item. Keep ``call_timeout_seconds`` well above normal
        call latency so this path stays reserved for genuinely hung calls.
        """
        if timeout_seconds <= 0:
            return func()
        # NOT a `with` block: ThreadPoolExecutor.__exit__ calls
        # shutdown(wait=True), which JOINS the worker thread — so on timeout it
        # would block until the hung call actually returns (up to the full
        # BOTO_READ_TIMEOUT), defeating the whole point of the timeout. Instead
        # abandon the doomed thread with shutdown(wait=False, cancel_futures=True)
        # so control returns to the caller immediately for the item's retry.
        pool = ContextThreadPoolExecutor(max_workers=1)
        future = pool.submit(func)
        try:
            result = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            pool.shutdown(wait=False, cancel_futures=True)
            logger.warning(
                "'%s' exceeded the %ss call timeout; the abandoned call may still "
                "complete in the background while the fallback re-issues it",
                label,
                timeout_seconds,
            )
            raise TimeoutError(
                f"'{label}' exceeded the {timeout_seconds}s call timeout"
            ) from exc
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=False)
            return result

    def execute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        """Call ``sequential_func`` once per item, in chunks, and retry failures.

        Each item is its own call under its own ``call_timeout_seconds``, run
        ``max_concurrency`` at a time within a chunk, so a slow or failed item
        neither discards the finished results of its chunk nor makes them run
        again: only the failed items are retried (``max_attempts``). Results
        are 1:1 with the prepared inputs; an item that failed every attempt is
        :data:`BATCH_ITEM_FAILED`.
        """
        if not items_to_process:
            return []

        if run_config:
            self.max_concurrency = run_config.get(
                "max_concurrency", self.max_concurrency
            )
            self.batch_size = run_config.get("batch_size", self.batch_size)
            self.chunk_concurrency = run_config.get(
                "chunk_concurrency", self.chunk_concurrency
            )
            self.max_attempts = run_config.get("max_attempts", self.max_attempts)

        # Bound every item call by the wall-clock ceiling so a hung item
        # aborts and is retried instead of blocking.
        def timed_sequential_func(single_input: dict[str, Any]) -> Any:
            return self._run_with_timeout(
                lambda: sequential_func(single_input),
                self.call_timeout_seconds,
                f"{task_name} item",
            )

        retrying_sequential_func = self._create_retry_decorator(task_name)(
            timed_sequential_func
        )

        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / self.batch_size)

        logger.info(
            "Starting processing for '%s': %s items in %s chunks (batch size: %s, "
            "chunk concurrency: %s)",
            task_name,
            num_items,
            num_chunks,
            self.batch_size,
            self.chunk_concurrency,
        )

        chunk_specs = [
            (idx + 1, items_to_process[i : i + self.batch_size])
            for idx, i in enumerate(range(0, num_items, self.batch_size))
        ]

        def process_chunk(chunk_num: int, chunk_items: list[Any]) -> list[Any]:
            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %s, skipping", chunk_num
                )
                return []
            results = self._call_each(timed_sequential_func, chunk_inputs)
            # Keep the successful results and retry ONLY the failed positions.
            failed = self._failed_indices(results)
            if not failed:
                logger.debug("Chunk %s processed successfully", chunk_num)
                return results
            self._log_partial_failure(task_name, chunk_num, results, failed)
            retried = self._process_sequentially_with_fallback(
                [chunk_inputs[i] for i in failed],
                retrying_sequential_func,
                f"{task_name} (chunk {chunk_num} retry)",
                show_progress=show_progress,
            )
            return self._splice_retried(results, failed, retried)

        # Chunks are independent Bedrock-bound batches; run them concurrently so
        # chunk N+1's LLM calls overlap chunk N's network wait instead of
        # blocking on it (CPU/memory are near-idle during these stages). Results
        # are reassembled in chunk order. chunk_concurrency=1 restores the old
        # strictly-serial behaviour.
        chunk_results_by_idx: dict[int, list[Any]] = {}
        progress = ProgressLogger(task_name, num_items)
        if self.chunk_concurrency <= 1 or len(chunk_specs) <= 1:
            for chunk_num, chunk_items in tqdm(
                chunk_specs,
                desc=f"Processing: {task_name}",
                disable=None if show_progress else True,
            ):
                chunk_results_by_idx[chunk_num] = process_chunk(chunk_num, chunk_items)
                progress.update(len(chunk_items))
        else:
            workers = min(self.chunk_concurrency, len(chunk_specs))
            chunk_sizes = {num: len(items) for num, items in chunk_specs}
            with ContextThreadPoolExecutor(max_workers=workers) as executor:
                future_to_num = {
                    executor.submit(process_chunk, num, items): num
                    for num, items in chunk_specs
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_num),
                    total=len(future_to_num),
                    desc=f"Processing: {task_name}",
                    disable=None if show_progress else True,
                ):
                    num = future_to_num[future]
                    chunk_results_by_idx[num] = future.result()
                    progress.update(chunk_sizes[num])

        all_results: list[Any] = []
        for chunk_num, _ in chunk_specs:
            all_results.extend(chunk_results_by_idx.get(chunk_num, []))

        logger.info("Completed '%s': processed %s results", task_name, len(all_results))
        return all_results

    def _call_each(
        self, func: Callable[[dict[str, Any]], Any], inputs: list[dict[str, Any]]
    ) -> list[Any]:
        """``func`` per input, ``max_concurrency`` at a time; errors in place."""
        workers = min(self.max_concurrency, len(inputs))
        with ContextThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(func, single_input) for single_input in inputs]
            results: list[Any] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(e)
        return results

    def _batch_kwargs(self) -> dict[str, Any]:
        # batch_func follows the Runnable.batch/abatch signature; per-item
        # exceptions come back in place so only the failed items are retried.
        return {
            "config": RunnableConfig(max_concurrency=self.max_concurrency),
            "return_exceptions": True,
        }

    @staticmethod
    def _failed_indices(results: list[Any]) -> list[int]:
        return [i for i, res in enumerate(results) if isinstance(res, BaseException)]

    @staticmethod
    def _log_partial_failure(
        task_name: str, chunk_num: int, results: list[Any], failed: list[int]
    ) -> None:
        logger.warning(
            "Batch chunk %s of '%s': %s/%s items failed (first error: %s). "
            "Retrying only the failed items",
            chunk_num,
            task_name,
            len(failed),
            len(results),
            results[failed[0]],
        )

    @staticmethod
    def _splice_retried(
        results: list[Any], failed: list[int], retried: list[Any]
    ) -> list[Any]:
        merged = list(results)
        for idx, res in zip(failed, retried, strict=True):
            merged[idx] = res
        return merged

    def _is_retryable(self, exc: BaseException) -> bool:
        # A malformed LLM response or a hung call can succeed on a new attempt.
        if isinstance(exc, OutputParserException | TimeoutError):
            return True
        return self.is_transient_error is None or self.is_transient_error(exc)

    def _create_retry_decorator(self, operation_name: str) -> Callable:
        # Exponential backoff WITH jitter (wait_random_exponential) to spread
        # concurrent retries and avoid hammering a throttled Bedrock endpoint in
        # lock-step. Only retryable errors are retried, so a permanent failure
        # (access denied, validation, missing model) fails fast.
        return tenacity.retry(
            retry=tenacity.retry_if_exception(self._is_retryable),
            wait=tenacity.wait_random_exponential(
                multiplier=self.retry_multiplier, max=self.retry_max_wait
            ),
            stop=tenacity.stop_after_attempt(self.max_attempts),
            before_sleep=self._create_retry_log_callback(operation_name),
            reraise=True,
        )

    @staticmethod
    def _create_retry_log_callback(operation_name: str) -> Callable:
        def log_retry(retry_state: RetryCallState) -> None:
            wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
            logger.warning(
                "Retrying '%s' (attempt %s failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                wait_time,
            )

        return log_retry

    @staticmethod
    def _process_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %s items sequentially for '%s'", len(inputs), task_name)

        results = []
        progress_desc = f"Sequential Processing: '{task_name}'"
        successful_count = 0

        for single_input in tqdm(
            inputs, desc=progress_desc, disable=None if show_progress else True
        ):
            try:
                result = sequential_func(single_input)
                results.append(result)
                successful_count += 1
            except Exception as e:
                logger.error(
                    "Sequential processing failed for single item in '%s': %s",
                    task_name,
                    e,
                )
                # Keep results 1:1 with inputs so callers can zip them back.
                results.append(BATCH_ITEM_FAILED)

        logger.info(
            "Sequential processing completed for '%s': %s/%s items processed successfully",
            task_name,
            successful_count,
            len(inputs),
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
            self.max_attempts = run_config.get("max_attempts", self.max_attempts)

        prepared_batch_func = self._create_async_batch_func(batch_func)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )

        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / self.batch_size)

        logger.info(
            "Starting async processing for '%s': %s items in %s chunks (batch size: %s)",
            task_name,
            num_items,
            num_chunks,
            self.batch_size,
        )

        chunk_iterator = async_tqdm(
            range(0, num_items, self.batch_size),
            desc=f"Processing: {task_name}",
            disable=None if show_progress else True,
        )
        progress = ProgressLogger(task_name, num_items)

        for i in chunk_iterator:
            chunk_items = items_to_process[i : i + self.batch_size]
            chunk_num = (i // self.batch_size) + 1
            logger.debug(
                "Processing chunk %s/%s (%s items)",
                chunk_num,
                num_chunks,
                len(chunk_items),
            )

            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %s, skipping", chunk_num
                )
                continue

            try:
                chunk_results = await prepared_batch_func(chunk_inputs)
            except Exception as e:
                logger.warning(
                    "Async batch processing failed for chunk %s: %s. Falling back to concurrent sequential processing",
                    chunk_num,
                    e,
                )
                chunk_results = await self._aprocess_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress,
                )
                all_results.extend(chunk_results)
                progress.update(len(chunk_items))
                continue

            # Retry only the failed positions (see the sync path).
            failed = self._failed_indices(chunk_results)
            if failed:
                self._log_partial_failure(task_name, chunk_num, chunk_results, failed)
                retried = await self._aprocess_sequentially_with_fallback(
                    [chunk_inputs[idx] for idx in failed],
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num} retry)",
                    show_progress,
                )
                chunk_results = self._splice_retried(chunk_results, failed, retried)
            all_results.extend(chunk_results)
            progress.update(len(chunk_items))

        logger.info("Completed '%s': processed %s results", task_name, len(all_results))
        return all_results

    def _create_async_batch_func(self, batch_func: Callable[..., Any]) -> Callable:
        batch_kwargs = self._batch_kwargs()

        async def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            result = await batch_func(inputs, **batch_kwargs)
            return list(result)

        return _batch_func

    async def _aprocess_sequentially_with_fallback(
        self,
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %s items concurrently for '%s'", len(inputs), task_name)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_one(single_input: dict[str, Any]) -> Any:
            async with semaphore:
                try:
                    return await sequential_func(single_input)
                except Exception as e:
                    logger.error(
                        "Concurrent sequential processing failed for item in '%s': %s",
                        task_name,
                        e,
                    )
                    return BATCH_ITEM_FAILED

        tasks = [_process_one(single_input) for single_input in inputs]

        progress_desc = f"Concurrent Fallback: '{task_name}'"
        results: list[Any] = await async_tqdm.gather(
            *tasks, disable=None if show_progress else True, desc=progress_desc
        )

        # Failed items stay in place as BATCH_ITEM_FAILED, NOT dropped: callers
        # zip the results back against the inputs (e.g. EvaluationManager with
        # strict=True), so a dropped item would abort the whole run.
        successful = sum(1 for res in results if res is not BATCH_ITEM_FAILED)
        logger.info(
            "Concurrent sequential processing completed for '%s': %s/%s items processed successfully",
            task_name,
            successful,
            len(inputs),
        )
        return results


_FORMAT_INSTRUCTIONS = (
    "Return the completion as well-formed XML made of {sections}. Keep every "
    "element and all of its text, add nothing that is not in the completion, "
    "and open and close every tag. Return only the XML."
)
# Synthetic root the response is wrapped in before lxml recovery.
_RESPONSE_ROOT = b"llm_response"
# Only valid at the very start of a document, so not inside the wrapper.
_XML_DECLARATION = re.compile(r"<\?xml[^>]*\?>")
# `&` not starting one of the XML predefined or numeric character references.
_BARE_AMPERSAND = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);)")
# `<` not followed by what can start a tag, end tag, comment or declaration.
_TEXT_LESS_THAN = re.compile(r"<(?![A-Za-z_/!?])")
_TAG = re.compile(r"<(/?)([A-Za-z_][\w.\-:]*)([^<>]*)>")


class RobustXMLOutputParser(XMLOutputParser):
    """Parse LLM XML into nested dicts, recovering from malformed output.

    Every path yields the same shape: an element's children keyed by tag, a
    repeated tag as a list, a leaf as its text. ``XMLOutputParser.parse`` is
    deliberately not used: it returns lists of one-key dicts
    (``{"claims": [{"claim": [{"subject": ...}, ...]}]}``) that the
    extractors cannot read, so a well-formed single-root response would parse
    into a different shape than a recovered one.
    """

    def get_format_instructions(self) -> str:
        """Instructions for the output fixer, naming the expected sections.

        ``XMLOutputParser``'s own text describes ``tags`` as one nesting path
        and prints "None" without them, which misleads a repair of a
        multi-section answer such as ``<entities>`` + ``<relationships>``.
        """
        tags = ", ".join(f"<{tag}>" for tag in self.tags or [])
        return _FORMAT_INSTRUCTIONS.format(
            sections=f"the top-level elements {tags}" if tags else "XML elements"
        )

    def parse(self, text: str) -> dict[str, Any]:
        if not text.strip():
            # Nothing to recover (e.g. a thinking-only answer); the output
            # fixer skips it too, since it could only invent the structure.
            raise OutputParserException("The model returned no output", llm_output=text)
        original_sections = self._detect_xml_sections(text)

        try:
            result = self._parse_top_level_elements(self._clean_xml_for_lxml(text))
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in lxml result")
        except Exception as e:
            logger.debug(
                "LXML recovery parsing failed: %s: %s. Trying XML fallback...",
                type(e).__name__,
                e,
            )

        try:
            fallback_result = self._extract_xml_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                "XML fallback extraction failed: %s: %s. Trying tags fallback...",
                type(e).__name__,
                e,
            )

        try:
            fallback_result = self._extract_tags_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                "Tags fallback extraction failed: %s: %s. Trying list fallback...",
                type(e).__name__,
                e,
            )

        try:
            fallback_result = self._extract_list_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                "List fallback extraction failed: %s: %s. All methods exhausted.",
                type(e).__name__,
                e,
            )

        # Model output can echo corpus or query text: only its size reaches
        # ERROR and the exception message; the text itself is DEBUG-only (and
        # carried on ``llm_output`` for OutputFixingParser).
        logger.error(
            "All XML parsing attempts failed for content: %s", text_digest(text)
        )
        logger.debug("Unparseable XML content: '%s'", text[:200])
        # OutputParserException (a ValueError) is what OutputFixingParser
        # catches to run its repair LLM; a bare ValueError bypassed it.
        raise OutputParserException(
            "Failed to parse XML after multiple attempts "
            f"(content {text_digest(text)})",
            llm_output=text,
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
                # An empty section is an empty answer (e.g. a chunk with no
                # relationships), kept as {} like the lxml path does, so a
                # response of only empty sections still parses.
                result[section_name] = {} if section_result is None else section_result

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

        result: dict[str, Any] = {}
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
        """Strip control characters and the XML declaration, escape text markup.

        A bare ``&`` (not starting an XML entity) and a ``<`` that cannot start
        a tag are text in LLM output ("AT&T", "budget < 5M"); lxml recovery
        would drop them and the word after them.
        """
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = _XML_DECLARATION.sub("", text)
        text = _BARE_AMPERSAND.sub("&amp;", text)
        text = _TEXT_LESS_THAN.sub("&lt;", text)
        text = RobustXMLOutputParser._drop_unmatched_closing_tags(text)
        return text.strip().encode("utf-8")

    @staticmethod
    def _drop_unmatched_closing_tags(text: str) -> str:
        """Make the element nesting well formed before lxml recovery.

        Models sometimes emit an end tag whose name matches nothing open: a
        stray ``</entity_placeholder>`` inside a ``<relationship>``, or a
        misnamed close such as ``<strength>7</strong>`` or ``</entity>`` for
        ``</relationship>``. lxml recovery neither drops such a tag nor closes
        the elements it skips, so every later sibling ends up nested inside
        the open element and is lost. Here:

        - an end tag naming an element further up the stack closes the
          elements above it explicitly, as XML nesting implies;
        - an unmatched end tag repeating the element just closed is dropped
          (a duplicate close);
        - any other unmatched end tag closes the innermost open element (the
          one the model was closing), or is dropped when nothing is open.
        """
        stack: list[str] = []
        last_closed = ""
        out: list[str] = []
        pos = 0
        for match in _TAG.finditer(text):
            is_close, name, rest = match.group(1), match.group(2), match.group(3)
            tag = match.group(0)
            if is_close:
                if name in stack:
                    closes = []
                    while stack:
                        last_closed = stack.pop()
                        closes.append(f"</{last_closed}>")
                        if last_closed == name:
                            break
                    tag = "".join(closes)
                elif name == last_closed or not stack:
                    tag = ""
                else:
                    last_closed = stack.pop()
                    tag = f"</{last_closed}>"
            elif not rest.rstrip().endswith("/"):
                stack.append(name)
            out.append(text[pos : match.start()])
            out.append(tag)
            pos = match.end()
        out.append(text[pos:])
        return "".join(out)

    @classmethod
    def _parse_top_level_elements(cls, xml_bytes: bytes) -> dict[str, Any]:
        """Parse every top-level element of the response, keyed by tag.

        The response is wrapped in one synthetic root, so a multi-section
        answer (``<entities>`` then ``<relationships>``) and a run of repeated
        siblings (``<line_number>`` without its ``<chunk_boundaries>``) parse
        like a single-root one instead of keeping only the first element.
        Prose around the elements is ignored.
        """
        wrapped = b"<%s>%s</%s>" % (_RESPONSE_ROOT, xml_bytes, _RESPONSE_ROOT)
        elements = cls._try_lxml_recover_parse(wrapped)[_RESPONSE_ROOT.decode()]
        if isinstance(elements, dict):
            elements.pop("#text", None)
        if not elements or not isinstance(elements, dict):
            raise ValueError("No XML element in the response")
        return elements

    @staticmethod
    def _try_lxml_recover_parse(xml_bytes: bytes) -> dict[str, Any]:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        tree = etree.fromstring(xml_bytes, parser=parser)

        if tree is None:
            raise ValueError("lxml parser recovered a null tree")

        def _convert_etree_to_dict(element: etree._Element) -> dict[str, Any]:
            result: dict[str, Any] = {}
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
                if isinstance(result[element.tag], dict):
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
                result[element.tag] = {}

            return result

        return _convert_etree_to_dict(tree)

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


# NOTE: the Bedrock-coupled chain builders (`setup_chain`,
# `create_robust_xml_output_parser`) moved to
# `unified_kg_rag.adapters.aws.chain_factory` so this kernel module stays free of
# any adapter dependency (hexagonal dependency rule). Import them from there.
