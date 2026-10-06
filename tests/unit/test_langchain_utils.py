# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for BatchProcessor and RobustXMLOutputParser.

These exercise the pure orchestration / parsing surface of
``unified_kg_rag.shared.utils.langchain`` with plain fake callables (no real LLM,
no boto3). The wall-clock timeout / chunk-ordering / concurrency cases already
live in ``test_batch_processor_timeout.py``; this module covers the
complementary branches: the batch-success happy path, run_config overrides,
empty input, the sequential ``{}`` filler on per-item failure, the retry
decorator, the async ``aexecute_with_fallback`` path, and the multi-stage
``RobustXMLOutputParser`` recovery ladder.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

import unified_kg_rag.shared.utils.langchain as langchain_module
from unified_kg_rag.shared.utils.langchain import (
    BatchProcessor,
    ProgressLogger,
    RobustXMLOutputParser,
)

pytestmark = pytest.mark.unit

# Retries still run, but without the production backoff (30s multiplier, 120s
# cap) that made the failure-path tests sleep for minutes.
_NO_BACKOFF: dict[str, Any] = {"retry_multiplier": 1.0, "retry_max_wait": 0}


class _TransientError(Exception):
    """Stands in for a backend error the injected classifier calls transient."""


# --------------------------------------------------------------------------- #
# BatchProcessor.execute_with_fallback
# --------------------------------------------------------------------------- #
class TestExecuteWithFallback:
    def test_empty_items_returns_empty(self) -> None:
        bp = BatchProcessor()
        called = []

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
            called.append(inputs)
            return []

        out = bp.execute_with_fallback(
            items_to_process=[],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=lambda item: item,
            task_name="empty",
            show_progress=False,
        )
        assert out == []
        assert called == []  # short-circuit before any batch call

    def test_batch_happy_path_single_chunk(self) -> None:
        # All items fit one chunk; batch succeeds -> sequential never invoked.
        bp = BatchProcessor(batch_size=10, chunk_concurrency=1, call_timeout_seconds=0)
        seq_calls = []

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
            return [{"echo": i["v"]} for i in inputs]

        def sequential(item):  # noqa: ANN001
            seq_calls.append(item)
            return {"echo": item["v"]}

        out = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": 1}, {"echo": 2}, {"echo": 3}]
        assert seq_calls == []

    def test_batch_passes_max_concurrency_config(self) -> None:
        # _create_batch_func injects a RunnableConfig(max_concurrency=...).
        bp = BatchProcessor(max_concurrency=7, batch_size=10, call_timeout_seconds=0)
        seen = {}

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001
            seen["config"] = config
            return [{"ok": 1} for _ in inputs]

        bp.execute_with_fallback(
            items_to_process=[1],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=lambda item: {},
            task_name="t",
            show_progress=False,
        )
        assert seen["config"]["max_concurrency"] == 7

    def test_run_config_overrides_fields(self) -> None:
        bp = BatchProcessor(max_concurrency=1, batch_size=99, chunk_concurrency=1)
        # batch_size override to 1 -> two chunks for two items.
        chunk_sizes = []

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
            chunk_sizes.append(len(inputs))
            return [{"echo": i["v"]} for i in inputs]

        out = bp.execute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=lambda item: {},
            task_name="t",
            run_config={"max_concurrency": 3, "batch_size": 1, "chunk_concurrency": 1},
            show_progress=False,
        )
        assert bp.batch_size == 1
        assert bp.max_concurrency == 3
        assert chunk_sizes == [1, 1]  # split into 2 single-item chunks
        assert out == [{"echo": 1}, {"echo": 2}]

    def test_empty_prepared_inputs_chunk_skipped(self) -> None:
        # prepare returns [] for a chunk -> that chunk yields [] (skipped),
        # contributing nothing to the assembled results.
        bp = BatchProcessor(batch_size=10, chunk_concurrency=1, call_timeout_seconds=0)

        out = bp.execute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=lambda items: [],
            batch_func=lambda inputs, **_: [{"x": 1}],  # noqa: ARG005
            sequential_func=lambda item: {},
            task_name="t",
            show_progress=False,
        )
        assert out == []

    def test_sequential_fallback_fills_empty_dict_on_item_failure(self) -> None:
        # Batch fails -> sequential path; one item raises and is back-filled with
        # {} so positional zip alignment downstream is preserved.
        bp = BatchProcessor(
            batch_size=10, chunk_concurrency=1, call_timeout_seconds=0, **_NO_BACKOFF
        )

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
            raise RuntimeError("batch boom")

        def sequential(item):  # noqa: ANN001
            if item["v"] == 2:
                raise ValueError("item 2 fails")
            return {"echo": item["v"]}

        out = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        # item 2 failed all retries -> back-filled with {}.
        assert out == [{"echo": 1}, {}, {"echo": 3}]

    def test_concurrent_chunks_use_distinct_threads(self) -> None:
        # With chunk_concurrency>1 and >1 chunk, process_chunk runs on a pool.
        bp = BatchProcessor(batch_size=1, chunk_concurrency=4, call_timeout_seconds=0)
        thread_ids: set[int] = set()
        lock = threading.Lock()
        barrier = threading.Barrier(3)

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
            barrier.wait(timeout=5)  # force genuine overlap across 3 chunks
            with lock:
                thread_ids.add(threading.get_ident())
            return [{"echo": inputs[0]["v"]}]

        out = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=lambda item: {},
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": 1}, {"echo": 2}, {"echo": 3}]
        assert len(thread_ids) >= 2  # ran on multiple worker threads


# --------------------------------------------------------------------------- #
# BatchProcessor retry decorator
# --------------------------------------------------------------------------- #
class TestRetryDecorator:
    def test_retries_then_succeeds(self) -> None:
        # multiplier tiny so backoff sleep is negligible; succeeds on 3rd call.
        bp = BatchProcessor(
            max_retries=5, retry_multiplier=1.0, retry_max_wait=0, batch_size=10
        )
        decorator = bp._create_retry_decorator("op")
        attempts = {"n": 0}

        @decorator
        def flaky():  # noqa: ANN202
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "done"

        assert flaky() == "done"
        assert attempts["n"] == 3

    def test_reraises_after_exhausting_attempts(self) -> None:
        bp = BatchProcessor(
            max_retries=2, retry_multiplier=1.0, retry_max_wait=0, batch_size=10
        )
        decorator = bp._create_retry_decorator("op")
        attempts = {"n": 0}

        @decorator
        def always_fail():  # noqa: ANN202
            attempts["n"] += 1
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            always_fail()
        assert attempts["n"] == 2  # stop_after_attempt(2)

    @pytest.mark.parametrize(
        ("error", "attempts_made"),
        [
            (_TransientError("throttled"), 3),
            (OutputParserException("malformed XML"), 3),
            (TimeoutError("hung call"), 3),
            # Permanent (e.g. access denied / validation): fail fast.
            (RuntimeError("access denied"), 1),
        ],
    )
    def test_classifier_limits_retries_to_retryable_errors(
        self, error: Exception, attempts_made: int
    ) -> None:
        bp = BatchProcessor(
            max_retries=3,
            is_transient_error=lambda e: isinstance(e, _TransientError),
            **_NO_BACKOFF,
        )
        attempts = {"n": 0}

        @bp._create_retry_decorator("op")
        def always_fail():  # noqa: ANN202
            attempts["n"] += 1
            raise error

        with pytest.raises(type(error)):
            always_fail()
        assert attempts["n"] == attempts_made

    def test_retry_log_callback_handles_none_next_action(self) -> None:
        # Defensive branch: next_action None -> wait_time 0, no crash.
        cb = BatchProcessor._create_retry_log_callback("op")

        class _State:
            next_action = None
            attempt_number = 1

        cb(_State())  # should not raise


# --------------------------------------------------------------------------- #
# BatchProcessor.aexecute_with_fallback (async path)
# --------------------------------------------------------------------------- #
class TestAExecuteWithFallback:
    async def test_async_empty_returns_empty(self) -> None:
        bp = BatchProcessor()
        out = await bp.aexecute_with_fallback(
            items_to_process=[],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=None,
            sequential_func=None,
            task_name="t",
            show_progress=False,
        )
        assert out == []

    async def test_async_batch_happy_path(self) -> None:
        bp = BatchProcessor(batch_size=10, max_concurrency=2)

        async def batch(
            inputs, config=None, return_exceptions=False
        ):  # noqa: ANN001, ARG001
            return [{"echo": i["v"]} for i in inputs]

        out = await bp.aexecute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=None,
            task_name="t",
            run_config={"max_concurrency": 4, "batch_size": 10},
            show_progress=False,
        )
        assert out == [{"echo": 1}, {"echo": 2}, {"echo": 3}]
        assert bp.max_concurrency == 4

    async def test_async_batch_failure_falls_back_concurrently(self) -> None:
        # Batch raises -> concurrent sequential fallback; a failing item is kept
        # as an empty-dict sentinel (NOT dropped) so the result list stays
        # positionally aligned with the inputs — callers zip it back with
        # strict=True and a dropped item would abort the whole run.
        bp = BatchProcessor(batch_size=10, max_concurrency=2, **_NO_BACKOFF)

        async def batch(
            inputs, config=None, return_exceptions=False
        ):  # noqa: ANN001, ARG001
            raise RuntimeError("async batch boom")

        async def sequential(item):  # noqa: ANN001
            if item["v"] == 2:
                raise ValueError("item 2 fails")
            return {"echo": item["v"]}

        out = await bp.aexecute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        # Position preserved: the failing item is {} so len(out) == len(inputs).
        assert out == [{"echo": 1}, {}, {"echo": 3}]
        assert len(out) == 3

    async def test_async_empty_prepared_chunk_skipped(self) -> None:
        bp = BatchProcessor(batch_size=10)

        async def batch(
            inputs, config=None, return_exceptions=False
        ):  # noqa: ANN001, ARG001
            return [{"x": 1}]

        out = await bp.aexecute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=lambda items: [],
            batch_func=batch,
            sequential_func=None,
            task_name="t",
            show_progress=False,
        )
        assert out == []


# --------------------------------------------------------------------------- #
# Partial batch failure: retry only the failed items
# --------------------------------------------------------------------------- #
class _CountingRunnable:
    """Real LangChain Runnable whose per-item calls are counted.

    ``fail_first`` maps an input value to how many of its calls raise before it
    succeeds (``None`` = always raises).
    """

    def __init__(self, fail_first: dict[int, int | None] | None = None) -> None:
        self.fail_first = fail_first or {}
        self.calls: Counter[int] = Counter()
        self._lock = threading.Lock()
        self.runnable = RunnableLambda(self._call, afunc=self._acall)

    def _call(self, item: dict[str, Any]) -> dict[str, Any]:
        v = item["v"]
        with self._lock:
            self.calls[v] += 1
            n = self.calls[v]
        budget = self.fail_first.get(v, 0)
        if budget is None or n <= budget:
            raise ValueError(f"synthetic failure for item {v}")
        return {"echo": v}

    async def _acall(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._call(item)


def _fast_bp(**kwargs: Any) -> BatchProcessor:
    return BatchProcessor(
        retry_multiplier=1.0,
        retry_max_wait=0,
        call_timeout_seconds=0,
        chunk_concurrency=1,
        **kwargs,
    )


class TestRetryFailedItemsOnly:
    def test_one_failure_reinvokes_only_that_item(self) -> None:
        fake = _CountingRunnable(fail_first={4: 1})  # item 4 fails once
        bp = _fast_bp(batch_size=10, max_retries=3)
        out = bp.execute_with_fallback(
            items_to_process=list(range(10)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.batch,
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": i} for i in range(10)]
        assert all(fake.calls[i] == 1 for i in range(10) if i != 4)
        assert fake.calls[4] == 2  # batch attempt + one retry

    def test_permanently_failing_item_gets_sentinel_in_place(self) -> None:
        fake = _CountingRunnable(fail_first={0: None, 7: None})
        bp = _fast_bp(batch_size=10, max_retries=2)
        out = bp.execute_with_fallback(
            items_to_process=list(range(10)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.batch,
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        expected: list[Any] = [{"echo": i} for i in range(10)]
        expected[0] = {}
        expected[7] = {}
        assert out == expected
        assert fake.calls[0] == 3  # batch attempt + max_retries
        assert fake.calls[7] == 3
        assert all(fake.calls[i] == 1 for i in range(1, 10) if i != 7)

    def test_run_config_sets_the_retry_count(self) -> None:
        # Ingestion stages pass config.processing as run_config; its
        # max_retries used to be ignored in favour of the processor default.
        fake = _CountingRunnable(fail_first={0: None})
        bp = _fast_bp(batch_size=10, max_retries=4)
        bp.execute_with_fallback(
            items_to_process=list(range(3)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.batch,
            sequential_func=fake.runnable.invoke,
            task_name="t",
            run_config={"max_retries": 1},
            show_progress=False,
        )
        assert fake.calls[0] == 2  # batch attempt + one retry

    def test_order_preserved_across_concurrent_chunks(self) -> None:
        fake = _CountingRunnable(fail_first={2: 1, 9: 1, 13: 1})
        bp = BatchProcessor(
            batch_size=4,
            chunk_concurrency=3,
            max_retries=3,
            retry_multiplier=1.0,
            retry_max_wait=0,
            call_timeout_seconds=0,
        )
        out = bp.execute_with_fallback(
            items_to_process=list(range(15)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.batch,
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": i} for i in range(15)]
        assert sum(fake.calls.values()) == 15 + 3

    def test_all_items_failing_behaves_as_before(self) -> None:
        fake = _CountingRunnable(fail_first=dict.fromkeys(range(4)))
        bp = _fast_bp(batch_size=10, max_retries=2)
        out = bp.execute_with_fallback(
            items_to_process=list(range(4)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.batch,
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [{}, {}, {}, {}]
        assert all(fake.calls[i] == 3 for i in range(4))

    def test_whole_batch_exception_still_reruns_every_item(self) -> None:
        # A batch_func that raises outright (no per-item results) keeps the
        # legacy full sequential fallback.
        seq_calls: list[int] = []

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
            raise RuntimeError("batch boom")

        def sequential(item):  # noqa: ANN001
            seq_calls.append(item["v"])
            return {"echo": item["v"]}

        out = _fast_bp(batch_size=10).execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": 1}, {"echo": 2}, {"echo": 3}]
        assert seq_calls == [1, 2, 3]

    def test_batch_func_always_gets_return_exceptions(self) -> None:
        seen: dict[str, Any] = {}

        def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001
            seen["config"] = config
            seen["return_exceptions"] = return_exceptions
            return [{"ok": 1} for _ in inputs]

        _fast_bp().execute_with_fallback(
            items_to_process=[1],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=lambda item: {},
            task_name="t",
            show_progress=False,
        )
        assert seen["config"] is not None
        assert seen["return_exceptions"] is True

    async def test_async_one_failure_reinvokes_only_that_item(self) -> None:
        fake = _CountingRunnable(fail_first={3: 1})
        bp = _fast_bp(batch_size=10, max_retries=3)
        out = await bp.aexecute_with_fallback(
            items_to_process=list(range(10)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.abatch,
            sequential_func=fake.runnable.ainvoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": i} for i in range(10)]
        assert all(fake.calls[i] == 1 for i in range(10) if i != 3)
        assert fake.calls[3] == 2

    async def test_async_all_failing_keeps_sentinels(self) -> None:
        fake = _CountingRunnable(fail_first=dict.fromkeys(range(3)))
        bp = _fast_bp(batch_size=10, max_retries=2)
        out = await bp.aexecute_with_fallback(
            items_to_process=list(range(3)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.abatch,
            sequential_func=fake.runnable.ainvoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [{}, {}, {}]
        assert all(fake.calls[i] == 3 for i in range(3))


# --------------------------------------------------------------------------- #
# RobustXMLOutputParser
# --------------------------------------------------------------------------- #
class TestRobustXMLOutputParser:
    def test_standard_parse_well_formed(self) -> None:
        parser = RobustXMLOutputParser()
        out = parser.parse("<root><name>Alice</name></root>")
        assert isinstance(out, dict)
        assert "root" in out

    def test_detect_xml_sections(self) -> None:
        sections = RobustXMLOutputParser._detect_xml_sections("<a>1</a> noise <b>2</b>")
        assert sections == {"a", "b"}

    def test_sections_preserved_true_when_no_sections(self) -> None:
        assert RobustXMLOutputParser._sections_preserved(set(), {}) is True

    def test_sections_preserved_false_when_missing(self) -> None:
        assert RobustXMLOutputParser._sections_preserved({"a", "b"}, {"a": 1}) is False

    def test_sections_preserved_non_dict_result(self) -> None:
        # parsed result not a dict -> treated as having no keys -> missing.
        assert RobustXMLOutputParser._sections_preserved({"a"}, ["x"]) is False

    def test_lxml_recovery_on_malformed(self) -> None:
        # Unclosed tag defeats the strict parser; lxml recover handles it and the
        # top-level <plan> section is preserved.
        parser = RobustXMLOutputParser()
        out = parser.parse("<plan><item>one</item><item>two</plan>")
        assert isinstance(out, dict)
        assert "plan" in out

    def test_sanitization_recovers_unescaped_ampersand(self) -> None:
        parser = RobustXMLOutputParser()
        # A bare & in text content; recovery ladder should yield a dict with the
        # section preserved.
        out = parser.parse("<note>Tom & Jerry</note>")
        assert isinstance(out, dict)
        assert "note" in out

    def test_extract_xml_fallback_nested(self) -> None:
        text = "<issues><issue>a</issue><issue>b</issue></issues>"
        out = RobustXMLOutputParser._extract_xml_fallback(text)
        assert out is not None
        assert "issues" in out

    def test_extract_xml_fallback_returns_none_on_no_match(self) -> None:
        assert RobustXMLOutputParser._extract_xml_fallback("plain text") is None

    def test_parse_xml_section_text_only(self) -> None:
        assert RobustXMLOutputParser._parse_xml_section("just text") == {
            "#text": "just text"
        }

    def test_parse_xml_section_empty_is_none(self) -> None:
        assert RobustXMLOutputParser._parse_xml_section("   ") is None

    def test_parse_xml_section_repeated_children_become_list(self) -> None:
        out = RobustXMLOutputParser._parse_xml_section("<x>1</x><x>2</x>")
        assert out == {"x": ["1", "2"]}

    def test_parse_xml_element_plain_text(self) -> None:
        assert RobustXMLOutputParser._parse_xml_element("hi") == "hi"
        assert RobustXMLOutputParser._parse_xml_element("  ") == ""

    def test_parse_xml_element_nested_with_trailing_text(self) -> None:
        out = RobustXMLOutputParser._parse_xml_element("<a>x</a>tail")
        assert out["a"] == "x"
        assert out["#text"] == "tail"

    def test_parse_xml_element_repeated_tags_to_list(self) -> None:
        out = RobustXMLOutputParser._parse_xml_element("<a>1</a><a>2</a>")
        assert out["a"] == ["1", "2"]

    def test_extract_tags_fallback(self) -> None:
        out = RobustXMLOutputParser._extract_tags_fallback(
            "<title>Hi</title><title>Yo</title><empty>  </empty>"
        )
        assert out == {"title": ["Hi", "Yo"]}

    def test_extract_tags_fallback_none_when_empty(self) -> None:
        assert RobustXMLOutputParser._extract_tags_fallback("no tags") is None

    def test_extract_list_fallback_bullets(self) -> None:
        out = RobustXMLOutputParser._extract_list_fallback("- one\n- two\n- three")
        assert out == {"items": ["one", "two", "three"]}

    def test_extract_list_fallback_numbered(self) -> None:
        out = RobustXMLOutputParser._extract_list_fallback("1. alpha\n2. beta")
        assert out == {"items": ["alpha", "beta"]}

    def test_extract_list_fallback_none(self) -> None:
        assert RobustXMLOutputParser._extract_list_fallback("nothing here") is None

    def test_clean_xml_strips_control_chars(self) -> None:
        out = RobustXMLOutputParser._clean_xml_for_lxml("a\x00b\x07c")
        assert out == b"abc"

    def test_aggressively_clean_escapes_bare_ampersand(self) -> None:
        out = RobustXMLOutputParser._aggressively_clean_xml("<a>x & y</a>")
        assert "&amp;" in out

    def test_sanitize_xml_content_escapes_inner(self) -> None:
        out = RobustXMLOutputParser._sanitize_xml_content("<a>1 < 2</a>")
        assert "&lt;" in out

    def test_try_lxml_recover_parse_nested(self) -> None:
        out = RobustXMLOutputParser._try_lxml_recover_parse(
            b"<root><a>1</a><b>2</b></root>"
        )
        assert out["root"]["a"] == "1"
        assert out["root"]["b"] == "2"

    def test_try_lxml_recover_parse_with_attributes(self) -> None:
        out = RobustXMLOutputParser._try_lxml_recover_parse(b'<root id="7">text</root>')
        assert out["root"]["@id"] == "7"

    def test_all_methods_exhausted_raises(self) -> None:
        # Plain prose with no recoverable tag/bullet/number structure: every
        # recovery method (strict parse, lxml recover, sanitize, aggressive
        # clean, xml/tags/list fallbacks) fails or returns None, so the ladder
        # exhausts and raises.
        parser = RobustXMLOutputParser()
        with pytest.raises(OutputParserException, match="Failed to parse XML"):
            parser.parse("this is just prose with no structure at all")

    def test_exhausted_parse_triggers_output_fixing(self) -> None:
        from langchain_classic.output_parsers import OutputFixingParser
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        fixer = OutputFixingParser.from_llm(
            parser=RobustXMLOutputParser(),
            llm=FakeListChatModel(responses=["<name>Vendor</name>"]),
        )
        assert fixer.parse("this is just prose with no structure at all") == {
            "name": "Vendor"
        }


# --------------------------------------------------------------------------- #
# ProgressLogger (INFO progress where tqdm is disabled, e.g. no TTY)
# --------------------------------------------------------------------------- #
def _progress_lines(log: Any) -> list[tuple]:
    return [
        c.args
        for c in log.info.call_args_list
        if c.args and c.args[0].startswith("Progress")
    ]


def test_progress_logger_logs_every_ten_percent(mocker) -> None:
    log = mocker.patch.object(langchain_module, "logger")
    progress = ProgressLogger("Extract", total=100)
    for _ in range(100):
        progress.update()
    done_counts = [args[2] for args in _progress_lines(log)]
    assert done_counts == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def test_progress_logger_logs_on_interval_between_marks(mocker) -> None:
    log = mocker.patch.object(langchain_module, "logger")
    clock = mocker.patch.object(langchain_module.time, "monotonic", return_value=0.0)
    progress = ProgressLogger("Extract", total=1000)
    progress.update()  # 0.1%, well below the first 10% mark
    assert _progress_lines(log) == []
    clock.return_value = 61.0
    progress.update()
    ((_, task, done, total, pct, rate, eta),) = _progress_lines(log)
    assert (task, done, total) == ("Extract", 2, 1000)
    assert rate == pytest.approx(2 / 61.0)
    assert eta == pytest.approx(998 / (2 / 61.0))


def test_execute_with_fallback_reports_progress(mocker) -> None:
    log = mocker.patch.object(langchain_module, "logger")
    processor = BatchProcessor(batch_size=2, chunk_concurrency=1)
    processor.execute_with_fallback(
        items_to_process=list(range(4)),
        prepare_inputs_func=lambda chunk: [{"x": x} for x in chunk],
        batch_func=lambda inputs, **_: [i["x"] for i in inputs],  # noqa: ARG005
        sequential_func=lambda i: i["x"],
        task_name="T",
        show_progress=False,
    )
    assert [args[2] for args in _progress_lines(log)][-1] == 4
