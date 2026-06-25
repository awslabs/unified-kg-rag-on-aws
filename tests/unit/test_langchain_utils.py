# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for BatchProcessor and RobustXMLOutputParser.

These exercise the pure orchestration / parsing surface of
``aws_graphrag.shared.utils.langchain`` with plain fake callables (no real LLM,
no boto3). The wall-clock timeout / chunk-ordering / concurrency cases already
live in ``test_batch_processor_timeout.py``; this module covers the
complementary branches: the batch-success happy path, run_config overrides,
empty input, the sequential ``{}`` filler on per-item failure, the retry
decorator, the async ``aexecute_with_fallback`` path, and the multi-stage
``RobustXMLOutputParser`` recovery ladder.
"""

from __future__ import annotations

import threading

import pytest

from aws_graphrag.shared.utils.langchain import (
    BatchProcessor,
    RobustXMLOutputParser,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# BatchProcessor.execute_with_fallback
# --------------------------------------------------------------------------- #
class TestExecuteWithFallback:
    def test_empty_items_returns_empty(self) -> None:
        bp = BatchProcessor()
        called = []

        def batch(inputs, config=None):  # noqa: ANN001, ARG001
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

        def batch(inputs, config=None):  # noqa: ANN001, ARG001
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

        def batch(inputs, config=None):  # noqa: ANN001
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

        def batch(inputs, config=None):  # noqa: ANN001, ARG001
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
            batch_func=lambda inputs, config=None: [{"x": 1}],  # noqa: ARG005
            sequential_func=lambda item: {},
            task_name="t",
            show_progress=False,
        )
        assert out == []

    def test_sequential_fallback_fills_empty_dict_on_item_failure(self) -> None:
        # Batch fails -> sequential path; one item raises and is back-filled with
        # {} so positional zip alignment downstream is preserved.
        bp = BatchProcessor(batch_size=10, chunk_concurrency=1, call_timeout_seconds=0)

        def batch(inputs, config=None):  # noqa: ANN001, ARG001
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

        def batch(inputs, config=None):  # noqa: ANN001, ARG001
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

        async def batch(inputs, config=None):  # noqa: ANN001, ARG001
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
        bp = BatchProcessor(batch_size=10, max_concurrency=2)

        async def batch(inputs, config=None):  # noqa: ANN001, ARG001
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

        async def batch(inputs, config=None):  # noqa: ANN001, ARG001
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
        with pytest.raises(ValueError, match="Failed to parse XML"):
            parser.parse("this is just prose with no structure at all")
