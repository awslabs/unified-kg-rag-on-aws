# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import functools
import inspect
import math
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Final

import tenacity
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import XMLOutputParser
from lxml import etree
from pydantic import BaseModel, ConfigDict, Field
from tenacity import RetryCallState
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError
from unified_kg_rag.shared.utils.common import text_digest
from unified_kg_rag.shared.utils.concurrency import (
    ContextThreadPoolExecutor,
    cancel_on_exit,
)

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import ProcessingConfig

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


class _BatchAbandoned(Exception):
    """An item's next attempt, not made: its batch was abandoned."""


def _batch_outcome(items: list[Any], results: list[Any]) -> dict[str, Any]:
    """Span attributes for a batch: item and failure counts."""
    return {
        "kg_rag.items": len(items),
        "kg_rag.results": len(results),
        "kg_rag.failed": sum(1 for r in results if r is BATCH_ITEM_FAILED),
    }


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
    # A removed setting (batch_size, chunk_concurrency, max_retries) raises
    # instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    max_concurrency: int = Field(
        default=5,
        ge=1,
        description="Items in flight at once: the next item starts as soon as "
        "one finishes, so this is both the peak and the steady concurrency",
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
        description="Total calls per item, including the first (1 disables "
        "the retry). Only retryable errors are retried: an item whose error is "
        "not retryable is not called again",
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
        description="Classifies backend errors worth retrying (the ingestion "
        "stages pass the LLM factory's, adapters.llm.retry."
        "transient_error_predicate); other errors "
        "fail fast. Output-parse failures and call timeouts are always retried. "
        "None retries every error, for callers that span several backends.",
    )

    @classmethod
    def from_processing(
        cls, processing: "ProcessingConfig", **settings: Any
    ) -> "BatchProcessor":
        """A processor sized by ``config.processing`` (the ingestion stages)."""
        return cls(
            max_concurrency=processing.max_concurrency,
            max_attempts=processing.max_attempts,
            **settings,
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
            if future.done():
                # ``func`` raised a TimeoutError of its own (the builtin is
                # concurrent.futures.TimeoutError since Python 3.11): not ours.
                raise
            logger.warning(
                "'%s' exceeded the %ss call timeout; the abandoned call may still "
                "complete in the background while the item is retried",
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
        show_progress: bool = True,
        errors: dict[int, BaseException] | None = None,
    ) -> list[Any]:
        """Call ``sequential_func`` once per item and retry failures.

        One pool keeps up to ``max_concurrency`` items in flight, starting the
        next item as soon as a slot frees up, so a slow item holds only its
        own slot. Each item is its own call under its own
        ``call_timeout_seconds``; an item whose call fails with a retryable
        error is retried alone, in its slot (``max_attempts``). Results are
        1:1 with the prepared inputs; an item that failed every attempt is
        :data:`BATCH_ITEM_FAILED`, and its last error is put in ``errors``
        (when given) under its index (see :meth:`is_item_error`).
        """
        with observability.span(
            "kg_rag.batch", task=task_name, items=len(items_to_process)
        ) as span:
            results = self._execute_with_fallback(
                items_to_process,
                prepare_inputs_func,
                sequential_func,
                task_name,
                show_progress,
                errors,
            )
            span.set_attributes(_batch_outcome(items_to_process, results))
            return results

    def _execute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool,
        errors: dict[int, BaseException] | None = None,
    ) -> list[Any]:
        inputs = self._prepare(items_to_process, prepare_inputs_func, task_name)
        if not inputs:
            return []

        # Set when the calling thread abandons the batch (a stop, Ctrl-C):
        # the items still running make no further attempt, and a backoff
        # wait ends at once, so the pool shuts down after the calls in flight.
        abandoned = threading.Event()

        # Bound every item call by the wall-clock ceiling so a hung item
        # aborts and is retried instead of blocking.
        def timed_sequential_func(single_input: dict[str, Any]) -> Any:
            if abandoned.is_set():
                raise _BatchAbandoned
            return self._run_with_timeout(
                lambda: sequential_func(single_input),
                self.call_timeout_seconds,
                f"{task_name} item",
            )

        retrying_sequential_func = self._create_retry_decorator(task_name, abandoned)(
            timed_sequential_func
        )

        def process_one(idx: int, single_input: dict[str, Any]) -> Any:
            try:
                return retrying_sequential_func(single_input)
            except _BatchAbandoned:
                return BATCH_ITEM_FAILED
            except Exception as error:
                if errors is not None:
                    errors[idx] = error
                return self._failed(task_name, error)

        results: list[Any] = [BATCH_ITEM_FAILED] * len(inputs)
        progress = ProgressLogger(task_name, len(inputs))
        # The block cancels the queued items of a batch that is being
        # abandoned. On a stop (Ctrl-C, SIGTERM) it does not wait for the
        # running calls (a stop must not wait for a model call in flight); on
        # an error it waits for them (see ContextThreadPoolExecutor).
        with ContextThreadPoolExecutor(
            max_workers=min(self.max_concurrency, len(inputs))
        ) as executor:
            try:
                futures = {
                    executor.submit(process_one, idx, single_input): idx
                    for idx, single_input in enumerate(inputs)
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Processing: {task_name}",
                    disable=None if show_progress else True,
                ):
                    results[futures[future]] = future.result()
                    progress.update()
            except BaseException:
                # Nor start another attempt of the running ones.
                abandoned.set()
                raise
        return self._completed(task_name, results)

    def _prepare(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        task_name: str,
    ) -> list[dict[str, Any]]:
        if not items_to_process:
            return []
        inputs = prepare_inputs_func(items_to_process)
        if not inputs:
            logger.warning("No valid inputs prepared for '%s', skipping", task_name)
            return []
        logger.info(
            "Starting processing for '%s': %s items, up to %s at a time",
            task_name,
            len(inputs),
            self.max_concurrency,
        )
        return inputs

    def _failed(self, task_name: str, error: BaseException) -> Any:
        if self._is_retryable(error):
            logger.error("Item failed every attempt in '%s': %s", task_name, error)
        else:
            logger.error(
                "Item failed in '%s' with a non-retryable error: %s", task_name, error
            )
        return BATCH_ITEM_FAILED

    @staticmethod
    def _completed(task_name: str, results: list[Any]) -> list[Any]:
        logger.info(
            "Completed '%s': %s results, %s failed",
            task_name,
            len(results),
            sum(1 for result in results if result is BATCH_ITEM_FAILED),
        )
        return results

    def is_item_error(self, exc: BaseException) -> bool:
        """Whether ``exc``, an item's last error, is the item's own failure
        rather than the backend's: a response that does not parse or was cut
        off, or an error the backend's classifier (``is_transient_error``)
        does not call transient (a 4xx such as a validation error). Throttling,
        5xx, timeouts and lost connections are not, nor is any error without a
        classifier: they say nothing about the item.
        """
        if isinstance(exc, OutputParserException | LLMOutputTruncatedError):
            return True
        if not isinstance(exc, Exception) or isinstance(
            exc, TimeoutError | _BatchAbandoned
        ):
            return False
        return self.is_transient_error is not None and not self.is_transient_error(exc)

    def _is_retryable(self, exc: BaseException) -> bool:
        # A cancelled call (or Ctrl-C) is a stop, not a failure: retrying it
        # started the work again after the caller had given up.
        if not isinstance(exc, Exception) or isinstance(exc, _BatchAbandoned):
            return False
        # The same input hits the same output-token limit again.
        if isinstance(exc, LLMOutputTruncatedError):
            return False
        # A malformed LLM response or a hung call can succeed on a new attempt.
        if isinstance(exc, OutputParserException | TimeoutError):
            return True
        return self.is_transient_error is None or self.is_transient_error(exc)

    def _create_retry_decorator(
        self, operation_name: str, abandoned: threading.Event | None = None
    ) -> Callable:
        # Exponential backoff WITH jitter (wait_random_exponential) to spread
        # concurrent retries and avoid hammering a throttled Bedrock endpoint in
        # lock-step. Only retryable errors are retried, so a permanent failure
        # (access denied, validation, missing model) fails fast. Once
        # ``abandoned`` is set no retry is made and a backoff wait ends.
        stop: tenacity.stop.stop_base = tenacity.stop_after_attempt(self.max_attempts)
        options: dict[str, Any] = {}
        if abandoned is not None:
            stop = stop | tenacity.stop_when_event_set(abandoned)
            options["sleep"] = abandoned.wait
        policy: dict[str, Any] = {
            "retry": tenacity.retry_if_exception(self._is_retryable),
            "wait": tenacity.wait_random_exponential(
                multiplier=self.retry_multiplier, max=self.retry_max_wait
            ),
            "stop": stop,
            "before_sleep": self._create_retry_log_callback(operation_name),
            "reraise": True,
        }
        # Not tenacity.retry: its wrapper keeps one statistics dict for every
        # call of the wrapped function and clears it on each call (tenacity
        # 9.2), so a worker about to back off read another's cleared stats
        # (KeyError 'idle_for', not retried: the item failed). Each call runs
        # on a copy of the policy, with statistics of its own (in a thread or
        # a coroutine alike).
        retrying = tenacity.Retrying(**policy, **options)
        aretrying = tenacity.AsyncRetrying(**policy)

        def decorator(func: Callable) -> Callable:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def acall(*args: Any, **kwargs: Any) -> Any:
                    return await aretrying.copy()(func, *args, **kwargs)

                return acall

            @functools.wraps(func)
            def call(*args: Any, **kwargs: Any) -> Any:
                return retrying.copy()(func, *args, **kwargs)

            return call

        return decorator

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

    async def aexecute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        sequential_func: Callable[..., Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        """The async :meth:`execute_with_fallback`, on the event loop.

        Each item is one awaited ``sequential_func`` call (``Runnable.ainvoke``),
        up to ``max_concurrency`` in flight; an item whose call fails with a
        retryable error is retried alone (``max_attempts``).
        """
        with observability.span(
            "kg_rag.batch", task=task_name, items=len(items_to_process)
        ) as span:
            results = await self._aexecute_with_fallback(
                items_to_process,
                prepare_inputs_func,
                sequential_func,
                task_name,
                show_progress,
            )
            span.set_attributes(_batch_outcome(items_to_process, results))
            return results

    async def _aexecute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        sequential_func: Callable[..., Any],
        task_name: str,
        show_progress: bool,
    ) -> list[Any]:
        inputs = self._prepare(items_to_process, prepare_inputs_func, task_name)
        if not inputs:
            return []

        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)
        progress = ProgressLogger(task_name, len(inputs))

        async def process_one(single_input: dict[str, Any]) -> Any:
            async with semaphore:
                try:
                    result = await retrying_sequential_func(single_input)
                except Exception as error:
                    result = self._failed(task_name, error)
            progress.update()
            return result

        tasks = [asyncio.ensure_future(process_one(i)) for i in inputs]
        async with cancel_on_exit(tasks):
            results: list[Any] = await async_tqdm.gather(
                *tasks,
                desc=f"Processing: {task_name}",
                disable=None if show_progress else True,
            )
        return self._completed(task_name, results)


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


def _escape_markup(tag: str) -> str:
    """Return a tag as XML text (its ``<`` escaped)."""
    return tag.replace("<", "&lt;")


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
            result = self._parse_top_level_elements(
                self._clean_xml_for_lxml(text, self.tags or ())
            )
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

    def _detect_xml_sections(self, text: str) -> set[str]:
        """The sections of ``text`` a parse result must keep to be accepted.

        With expected top-level ``tags``, only those count, closed or not: a
        nested element (``<relationship>`` inside ``<relationships>``) is not
        a section, and an expected last section the model left unclosed is
        still one. Without ``tags``, every closed element counts.
        """
        if self.tags:
            return {
                tag for tag in self.tags if re.search(rf"<{re.escape(tag)}[\s/>]", text)
            }
        return set(re.findall(r"<([a-zA-Z0-9_]+)>.*?</\1>", text, re.DOTALL))

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
    def _clean_xml_for_lxml(text: str, known_tags: Iterable[str] = ()) -> bytes:
        """Strip control characters and the XML declaration, escape text markup.

        A bare ``&`` (not starting an XML entity) and a ``<`` that cannot start
        a tag are text in LLM output ("AT&T", "budget < 5M"); lxml recovery
        would drop them and the word after them. ``known_tags`` are element
        names the caller expects, which are never escaped as text.
        """
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = _XML_DECLARATION.sub("", text)
        text = _BARE_AMPERSAND.sub("&amp;", text)
        text = _TEXT_LESS_THAN.sub("&lt;", text)
        text = RobustXMLOutputParser._drop_unmatched_closing_tags(text, known_tags)
        return text.strip().encode("utf-8")

    @staticmethod
    def _structural_tag_names(text: str) -> set[str]:
        """Names of the opening tags that sit in element structure, not in text.

        A tag is structural when only whitespace separates it from the
        previous end tag or structural opening tag (or the start of the
        text): ``<entity>\\n<name>`` is markup, ``Optional<User>`` and
        ``uses <br> tag`` are not. The names found this way are the response's
        element vocabulary.
        """
        names: set[str] = set()
        prev_end = 0
        prev_structural = True
        for match in _TAG.finditer(text):
            is_close, name = match.group(1), match.group(2)
            gap_is_blank = not text[prev_end : match.start()].strip()
            if is_close:
                prev_structural = True
            else:
                prev_structural = gap_is_blank and prev_structural
                if prev_structural:
                    names.add(name)
            prev_end = match.end()
        return names

    @staticmethod
    def _drop_unmatched_closing_tags(text: str, known_tags: Iterable[str] = ()) -> str:
        """Make the element nesting well formed before lxml recovery.

        Models sometimes emit an end tag whose name matches nothing open: a
        stray ``</entity_placeholder>`` inside a ``<relationship>``, or a
        misnamed close such as ``<strength>7</strong>`` or ``</entity>`` for
        ``</relationship>``. lxml recovery neither drops such a tag nor closes
        the elements it skips, so every later sibling ends up nested inside
        the open element and is lost. Field text can also hold tag-like
        sequences (``Optional<User>``, ``<br>``) that are not markup. Here:

        - an end tag naming an open element closes the elements above it
          explicitly, as XML nesting implies;
        - an unmatched end tag repeating the element just closed (nothing
          opened since) is dropped as a duplicate close;
        - any other unmatched end tag closes the innermost open element only
          when that element is a leaf field (no child elements) and the
          field's own end tag does not follow it (the misnamed close
          ``<strength>7</strong>``); in a record (an element with children)
          or with nothing open it is dropped, so a stray tag between two
          fields does not end the record early;
        - an opening tag naming an open record (an element with children)
          closes that record first: records never nest in one with their own
          name, so ``</entity>`` written for ``</relationship>`` followed by
          the next ``<relationship>`` still yields two records;
        - inside a field that already has text, an opening tag that is
          neither in ``known_tags`` nor in the response's structural
          vocabulary (see ``_structural_tag_names``) is text, escaped along
          with its matching end tag, so ``Optional<User>``, ``<br>`` and
          ``<T>`` stay in the field verbatim.

        Any other ``<`` left in text is escaped too.
        """
        vocabulary = RobustXMLOutputParser._structural_tag_names(text) | set(known_tags)
        # Per open element: [name, has_child, has_text, escaped tag names].
        stack: list[list[Any]] = []
        last_closed = ""
        out: list[str] = []
        pos = 0
        matches = list(_TAG.finditer(text))

        def close_through(index: int) -> str:
            nonlocal last_closed
            closes = []
            while len(stack) > index:
                last_closed = stack.pop()[0]
                closes.append(f"</{last_closed}>")
            if stack:
                stack[-1][1] = True
            return "".join(closes)

        def next_tag_closes_innermost(i: int) -> bool:
            if i + 1 >= len(matches):
                return False
            following = matches[i + 1]
            return bool(following.group(1)) and following.group(2) == stack[-1][0]

        for i, match in enumerate(matches):
            is_close, name, rest = match.group(1), match.group(2), match.group(3)
            tag = match.group(0)
            segment = text[pos : match.start()]
            segment = re.sub(r"<(?![!?])", "&lt;", segment)
            if stack and segment.strip():
                stack[-1][2] = True
            top = stack[-1] if stack else None
            names = [frame[0] for frame in stack]
            if is_close:
                if top is not None and name in top[3]:
                    top[3].remove(name)
                    tag = _escape_markup(tag)
                elif name in names:
                    index = len(names) - 1 - names[::-1].index(name)
                    tag = close_through(index)
                elif top is None or name == last_closed or top[1]:
                    tag = ""
                elif next_tag_closes_innermost(i):
                    tag = ""
                else:
                    tag = close_through(len(stack) - 1)
            elif top is not None and top[2] and not top[1] and name not in vocabulary:
                if not rest.rstrip().endswith("/"):
                    top[3].append(name)
                tag = _escape_markup(tag)
            else:
                prefix = ""
                for index in range(len(stack) - 1, -1, -1):
                    if stack[index][0] == name and stack[index][1]:
                        prefix = close_through(index)
                        break
                if stack:
                    stack[-1][1] = True
                last_closed = ""
                if not rest.rstrip().endswith("/"):
                    stack.append([name, False, False, []])
                tag = prefix + tag
            out.append(segment)
            out.append(tag)
            pos = match.end()
        out.append(re.sub(r"<(?![!?])", "&lt;", text[pos:]))
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
# `unified_kg_rag.adapters.llm.chain_factory` so this kernel module stays free of
# any adapter dependency (hexagonal dependency rule). Import them from there.


def with_callback(callbacks: Any, handler: BaseCallbackHandler) -> Any:
    """``callbacks`` (None, a handler list or a manager) plus ``handler``.

    Never mutates the caller's value: a manager is caller-owned and may be
    shared by other models, which must not pick up this model's handler.
    """
    if callbacks is None:
        return [handler]
    if isinstance(callbacks, BaseCallbackManager):
        manager = callbacks.copy()
        manager.add_handler(handler, inherit=False)
        return manager
    return [*callbacks, handler]
