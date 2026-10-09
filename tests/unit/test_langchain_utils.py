# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for BatchProcessor and RobustXMLOutputParser.

These exercise the pure orchestration / parsing surface of
``unified_kg_rag.shared.utils.langchain`` with plain fake callables (no real LLM,
no boto3). The wall-clock timeout / chunk-ordering / concurrency cases already
live in ``test_batch_processor_timeout.py``; this module covers the
complementary branches: the per-item happy path, run_config overrides,
empty input, the ``BATCH_ITEM_FAILED`` filler on per-item failure, the retry
decorator, the async ``aexecute_with_fallback`` path, and the multi-stage
``RobustXMLOutputParser`` recovery ladder.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.runnables import RunnableLambda

import unified_kg_rag.shared.utils.langchain as langchain_module
from unified_kg_rag.shared.utils import ensure_list
from unified_kg_rag.shared.utils.langchain import (
    BATCH_ITEM_FAILED,
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

        out = bp.execute_with_fallback(
            items_to_process=[],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=called.append,
            task_name="empty",
            show_progress=False,
        )
        assert out == []
        assert called == []  # short-circuit before any call

    def test_happy_path_calls_each_item_once(self) -> None:
        bp = BatchProcessor(batch_size=10, chunk_concurrency=1, call_timeout_seconds=0)
        calls: list[int] = []
        lock = threading.Lock()

        def sequential(item):  # noqa: ANN001
            with lock:
                calls.append(item["v"])
            return {"echo": item["v"]}

        out = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": 1}, {"echo": 2}, {"echo": 3}]
        assert sorted(calls) == [1, 2, 3]

    def test_max_concurrency_bounds_items_in_flight(self) -> None:
        bp = BatchProcessor(max_concurrency=2, batch_size=6, call_timeout_seconds=0)
        lock = threading.Lock()
        in_flight = 0
        max_in_flight = 0
        two_running = threading.Barrier(2, timeout=10)

        def sequential(item):  # noqa: ANN001
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                two_running.wait()  # needs two items running at once
            finally:
                with lock:
                    in_flight -= 1
            return item["v"]

        out = bp.execute_with_fallback(
            items_to_process=list(range(6)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        assert out == list(range(6))
        assert max_in_flight == 2

    def test_run_config_overrides_fields(self) -> None:
        bp = BatchProcessor(max_concurrency=1, batch_size=99, chunk_concurrency=1)
        # batch_size override to 1 -> two chunks for two items.
        chunk_sizes = []

        def prepare(items):  # noqa: ANN001
            chunk_sizes.append(len(items))
            return [{"v": i} for i in items]

        out = bp.execute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=prepare,
            sequential_func=lambda item: {"echo": item["v"]},
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
            sequential_func=lambda item: {},
            task_name="t",
            show_progress=False,
        )
        assert out == []

    def test_item_failure_is_marked_in_place(self) -> None:
        # One item raises on every attempt and is back-filled with
        # BATCH_ITEM_FAILED so positional zip alignment downstream is preserved.
        bp = BatchProcessor(
            batch_size=10, chunk_concurrency=1, call_timeout_seconds=0, **_NO_BACKOFF
        )

        def sequential(item):  # noqa: ANN001
            if item["v"] == 2:
                raise ValueError("item 2 fails")
            return {"echo": item["v"]}

        out = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": 1}, BATCH_ITEM_FAILED, {"echo": 3}]

    def test_concurrent_chunks_use_distinct_threads(self) -> None:
        # With chunk_concurrency>1 and >1 chunk, process_chunk runs on a pool.
        bp = BatchProcessor(batch_size=1, chunk_concurrency=4, call_timeout_seconds=0)
        thread_ids: set[int] = set()
        lock = threading.Lock()
        barrier = threading.Barrier(3)

        def sequential(item):  # noqa: ANN001
            barrier.wait(timeout=5)  # force genuine overlap across 3 chunks
            with lock:
                thread_ids.add(threading.get_ident())
            return {"echo": item["v"]}

        out = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=sequential,
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
            max_attempts=5, retry_multiplier=1.0, retry_max_wait=0, batch_size=10
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
            max_attempts=2, retry_multiplier=1.0, retry_max_wait=0, batch_size=10
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
            max_attempts=3,
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

    async def test_empty_results_are_not_failures(self) -> None:
        # A real empty output ({} / None) stays distinguishable from a failure.
        bp = BatchProcessor(batch_size=10, max_concurrency=2, **_NO_BACKOFF)

        async def batch(
            inputs, config=None, return_exceptions=False
        ):  # noqa: ANN001, ARG001
            raise RuntimeError("async batch boom")

        async def sequential(item):  # noqa: ANN001
            return {} if item["v"] == 1 else None

        out = await bp.aexecute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        assert out == [{}, None]

    async def test_async_batch_failure_falls_back_concurrently(self) -> None:
        # Batch raises -> concurrent sequential fallback; a failing item is kept
        # as BATCH_ITEM_FAILED (NOT dropped) so the result list stays
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
        # Position preserved: the failing item is marked, not dropped.
        assert out == [{"echo": 1}, BATCH_ITEM_FAILED, {"echo": 3}]
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
        bp = _fast_bp(batch_size=10, max_attempts=3)
        out = bp.execute_with_fallback(
            items_to_process=list(range(10)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": i} for i in range(10)]
        assert all(fake.calls[i] == 1 for i in range(10) if i != 4)
        assert fake.calls[4] == 2  # first call + one retry

    def test_permanently_failing_item_gets_sentinel_in_place(self) -> None:
        fake = _CountingRunnable(fail_first={0: None, 7: None})
        bp = _fast_bp(batch_size=10, max_attempts=2)
        out = bp.execute_with_fallback(
            items_to_process=list(range(10)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        expected: list[Any] = [{"echo": i} for i in range(10)]
        expected[0] = BATCH_ITEM_FAILED
        expected[7] = BATCH_ITEM_FAILED
        assert out == expected
        assert fake.calls[0] == 3  # first call + max_attempts
        assert fake.calls[7] == 3
        assert all(fake.calls[i] == 1 for i in range(1, 10) if i != 7)

    def test_run_config_sets_the_retry_count(self) -> None:
        # Ingestion stages pass config.processing as run_config; its
        # retry setting used to be ignored in favour of the processor default.
        fake = _CountingRunnable(fail_first={0: None})
        bp = _fast_bp(batch_size=10, max_attempts=4)
        bp.execute_with_fallback(
            items_to_process=list(range(3)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=fake.runnable.invoke,
            task_name="t",
            run_config={"max_attempts": 1},
            show_progress=False,
        )
        assert fake.calls[0] == 2  # first call + one retry

    def test_order_preserved_across_concurrent_chunks(self) -> None:
        fake = _CountingRunnable(fail_first={2: 1, 9: 1, 13: 1})
        bp = BatchProcessor(
            batch_size=4,
            chunk_concurrency=3,
            max_attempts=3,
            retry_multiplier=1.0,
            retry_max_wait=0,
            call_timeout_seconds=0,
        )
        out = bp.execute_with_fallback(
            items_to_process=list(range(15)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [{"echo": i} for i in range(15)]
        assert sum(fake.calls.values()) == 15 + 3

    def test_all_items_failing_behaves_as_before(self) -> None:
        fake = _CountingRunnable(fail_first=dict.fromkeys(range(4)))
        bp = _fast_bp(batch_size=10, max_attempts=2)
        out = bp.execute_with_fallback(
            items_to_process=list(range(4)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=fake.runnable.invoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [BATCH_ITEM_FAILED] * 4
        assert all(fake.calls[i] == 3 for i in range(4))

    async def test_async_batch_func_always_gets_return_exceptions(self) -> None:
        seen: dict[str, Any] = {}

        async def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001
            seen["config"] = config
            seen["return_exceptions"] = return_exceptions
            return [{"ok": 1} for _ in inputs]

        async def sequential(item):  # noqa: ANN001
            return {}

        await _fast_bp().aexecute_with_fallback(
            items_to_process=[1],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=sequential,
            task_name="t",
            show_progress=False,
        )
        assert seen["config"] is not None
        assert seen["return_exceptions"] is True

    async def test_async_one_failure_reinvokes_only_that_item(self) -> None:
        fake = _CountingRunnable(fail_first={3: 1})
        bp = _fast_bp(batch_size=10, max_attempts=3)
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
        bp = _fast_bp(batch_size=10, max_attempts=2)
        out = await bp.aexecute_with_fallback(
            items_to_process=list(range(3)),
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=fake.runnable.abatch,
            sequential_func=fake.runnable.ainvoke,
            task_name="t",
            show_progress=False,
        )
        assert out == [BATCH_ITEM_FAILED] * 3
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
        # An unclosed tag is recovered by lxml and the top-level <plan>
        # section is preserved.
        parser = RobustXMLOutputParser()
        out = parser.parse("<plan><item>one</item><item>two</plan>")
        assert isinstance(out, dict)
        assert "plan" in out

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("<name>A&B Corp</name>", {"name": "A&B Corp"}),
            ("<note>Tom & Jerry</note>", {"note": "Tom & Jerry"}),
            (
                "<plan><item>R&D budget < 5M</item></plan>",
                {"plan": {"item": "R&D budget < 5M"}},
            ),
            ("<note>x&lt;y &amp; z &#38; w</note>", {"note": "x<y & z & w"}),
        ],
    )
    def test_bare_ampersand_and_less_than_keep_text(self, text, expected) -> None:
        # A bare & or a < that does not start a tag is text, not markup:
        # recovery must not drop it (or the word after it).
        assert RobustXMLOutputParser().parse(text) == expected

    def test_bare_ampersand_and_less_than_keep_text_across_sections(self) -> None:
        text = (
            "<entities>\n<entity><name>A&B Corp</name><description>R&D spend < 5M"
            "</description></entity>\n</entities>\n<relationships>\n"
            "<relationship><source>A&B Corp</source><target>Smith & Sons</target>"
            "<description>x < y</description></relationship>\n</relationships>"
        )
        assert RobustXMLOutputParser().parse(text) == {
            "entities": {
                "entity": {"name": "A&B Corp", "description": "R&D spend < 5M"}
            },
            "relationships": {
                "relationship": {
                    "source": "A&B Corp",
                    "target": "Smith & Sons",
                    "description": "x < y",
                }
            },
        }

    def test_repeated_top_level_tag_keeps_every_sibling(self) -> None:
        # A response without a root element (the model continued an open
        # <chunk_boundaries>) is a run of sibling elements; every one is kept,
        # not only the first.
        text = (
            "<line_number>4</line_number>\n<line_number>7</line_number>\n"
            "<line_number>10</line_number>\n</chunk_boundaries>"
        )
        assert RobustXMLOutputParser().parse(text) == {"line_number": ["4", "7", "10"]}

    def test_sections_decode_character_references(self) -> None:
        # Multi-section responses go through the same lxml pass as
        # single-root ones, so their text is decoded the same way.
        text = "<entities>x &amp; y</entities><relationships>a &lt; b</relationships>"
        assert RobustXMLOutputParser().parse(text) == {
            "entities": "x & y",
            "relationships": "a < b",
        }

    def test_xml_declaration_and_surrounding_prose_are_ignored(self) -> None:
        text = (
            'Here is the output:\n<?xml version="1.0" encoding="UTF-8"?>\n'
            "<chunk_boundaries>\n<line_number>4</line_number>\n"
            "<line_number>7</line_number>\n</chunk_boundaries>\nDone."
        )
        assert RobustXMLOutputParser().parse(text) == {
            "chunk_boundaries": {"line_number": ["4", "7"]}
        }

    def test_single_root_parses_to_children_by_tag(self) -> None:
        # A well-formed single-root response (claims, refinement plan) parses
        # into children keyed by tag, the shape the extractors read, not
        # XMLOutputParser's lists of one-key dicts.
        text = (
            "<claims>\n<claim><subject>Vendor</subject><object>Buyer</object>"
            "</claim>\n<claim><subject>Buyer</subject><object>Vendor</object>"
            "</claim>\n</claims>"
        )
        assert RobustXMLOutputParser().parse(text) == {
            "claims": {
                "claim": [
                    {"subject": "Vendor", "object": "Buyer"},
                    {"subject": "Buyer", "object": "Vendor"},
                ]
            }
        }

    def test_single_root_shape_does_not_depend_on_defusedxml(self, mocker) -> None:
        # XMLOutputParser.parse raises ImportError without defusedxml and
        # returns a different shape with it; the parser must not call it.
        strict = mocker.patch.object(XMLOutputParser, "parse")
        out = RobustXMLOutputParser().parse(
            "<refinement_plan><quality_scores><completeness_score>0.8"
            "</completeness_score></quality_scores></refinement_plan>"
        )
        strict.assert_not_called()
        assert out == {
            "refinement_plan": {"quality_scores": {"completeness_score": "0.8"}}
        }

    def test_multiple_sections_parse_by_section(self) -> None:
        # Extraction output has two top-level sections (no single root).
        text = (
            "<entities>\n<entity><name>Vendor</name><type>ORG</type></entity>\n"
            "</entities>\n<relationships>\n<relationship><source>Vendor</source>"
            "<target>Buyer</target></relationship>\n</relationships>"
        )
        assert RobustXMLOutputParser().parse(text) == {
            "entities": {"entity": {"name": "Vendor", "type": "ORG"}},
            "relationships": {"relationship": {"source": "Vendor", "target": "Buyer"}},
        }

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

    @pytest.mark.parametrize(
        "text",
        [
            "<entities></entities><relationships></relationships>",
            "<entities>\n</entities>\n<relationships>\n</relationships>\n",
        ],
    )
    def test_only_empty_sections_parse_as_empty_result(self, text) -> None:
        # A chunk with nothing to extract is a valid answer, not a parse
        # failure (which would be retried and sent to the fixing LLM).
        out = RobustXMLOutputParser().parse(text)
        assert out == {"entities": {}, "relationships": {}}

    def test_extract_xml_fallback_keeps_empty_sections(self) -> None:
        out = RobustXMLOutputParser._extract_xml_fallback(
            "<entities><entity>a</entity></entities><relationships></relationships>"
        )
        assert out == {"entities": {"entity": "a"}, "relationships": {}}

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
        # recovery method (lxml recover, xml/tags/list fallbacks) fails or
        # returns None, so the ladder exhausts and raises.
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
        sequential_func=lambda i: i["x"],
        task_name="T",
        show_progress=False,
    )
    assert [args[2] for args in _progress_lines(log)][-1] == 4


def test_stray_closing_tag_keeps_later_relationships() -> None:
    # Models occasionally emit an end tag that closes nothing (seen in real
    # extraction output inside the first <relationship>). It must not cut the
    # section short: every relationship after it is kept.
    raw = (
        "<entities>\n"
        "<entity><name>Vendor A</name><type>ORGANIZATION</type></entity>\n"
        "<entity><name>Buyer B</name><type>ORGANIZATION</type></entity>\n"
        "</entities>\n<relationships>\n"
        "<relationship><source>Vendor A</source><target>Buyer B</target>"
        "<type>SUPPLIES</type><source_text>Vendor A supplies Buyer B</source_text>\n"
        "</entity_placeholder>\n</relationship>\n"
        "<relationship><source>Buyer B</source><target>Vendor A</target>"
        "<type>PAYS</type></relationship>\n"
        "<relationship><source>Vendor A</source><target>Vendor A</target>"
        "<type>SELF</type></relationship>\n"
        "</relationships>"
    )
    parsed = RobustXMLOutputParser(tags=["entities", "relationships"]).parse(raw)
    rels = parsed["relationships"]["relationship"]
    assert isinstance(rels, list)
    assert [r["type"] for r in rels] == ["SUPPLIES", "PAYS", "SELF"]
    assert len(parsed["entities"]["entity"]) == 2


def test_matched_closing_tags_are_untouched() -> None:
    text = "<a><b>x</b><c/></a>"
    assert RobustXMLOutputParser._drop_unmatched_closing_tags(text) == text
    assert (
        RobustXMLOutputParser._drop_unmatched_closing_tags("<a>x</z></a>") == "<a>x</a>"
    )


def test_wrong_record_end_tag_closes_the_open_record() -> None:
    raw = (
        "<relationships>"
        "<relationship><source>A</source><target>B</target></entity>"
        "<relationship><source>B</source><target>C</target></relationship>"
        "</relationships>"
    )
    parsed = RobustXMLOutputParser(tags=["relationships"]).parse(raw)
    rels = parsed["relationships"]["relationship"]
    assert [(r["source"], r["target"]) for r in rels] == [("A", "B"), ("B", "C")]


def _relationship_pairs(raw: str) -> list[tuple[Any, Any]]:
    parsed = RobustXMLOutputParser(tags=["relationships"]).parse(raw)
    rels = ensure_list(parsed["relationships"], inner_key="relationship")
    return [(r.get("source"), r.get("target")) for r in rels]


def test_stray_end_tag_between_fields_keeps_the_record_open() -> None:
    # A stray end tag inside a record (between two fields) is dropped; it
    # must not end the record and push its later fields out of it.
    raw = (
        "<relationships><relationship><source>A</source></entity_placeholder>"
        "<target>B</target><type>OWNS</type></relationship>"
        "<relationship><source>B</source><target>A</target><type>OWNED_BY</type>"
        "</relationship></relationships>"
    )
    assert _relationship_pairs(raw) == [("A", "B"), ("B", "A")]


def test_misnamed_field_end_tag_closes_the_field() -> None:
    # `<strength>7</strong>`, `<target>B</source>` (naming the field just
    # closed) and `<source>B</source_text>` close the open field.
    raw = (
        "<relationships><relationship><source>A</source><target>B</source>"
        "<strength>7</strong><type>OWNS</type></relationship>"
        "<relationship><source>B</source_text><target>C</target></relationship>"
        "</relationships>"
    )
    parsed = RobustXMLOutputParser(tags=["relationships"]).parse(raw)
    first, second = parsed["relationships"]["relationship"]
    assert first == {"source": "A", "target": "B", "strength": "7", "type": "OWNS"}
    assert second == {"source": "B", "target": "C"}


def test_stray_and_misnamed_end_tags_combined() -> None:
    # A stray tag between fields, a misnamed field close and `</entity>` for
    # `</relationship>` in one response.
    raw = (
        "<relationships>\n"
        "<relationship><source>A</source>\n</entity_placeholder>\n"
        "<target>B</target><strength>7</strong></entity>\n"
        "<relationship><source>B</source><target>C</target></relationship>\n"
        "<relationship><source>C</source><target>A</target>\n"
        "</entity_placeholder>\n</relationship>\n"
        "</relationships>"
    )
    assert _relationship_pairs(raw) == [("A", "B"), ("B", "C"), ("C", "A")]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "findById returns Optional<User> when found",
            "findById returns Optional<User> when found",
        ),
        ("uses <br> tag and <T> generic", "uses <br> tag and <T> generic"),
        ("a line break <br/> here", "a line break <br/> here"),
        ("wraps it in <b>bold</b> text", "wraps it in <b>bold</b> text"),
        ("if a < b then a <b", "if a < b then a <b"),
        ("AT&T & Smith &amp; Sons", "AT&T & Smith & Sons"),
        # An unmatched end tag right before the field's own close is dropped.
        ("closes with </div> only", "closes with  only"),
    ],
)
def test_tag_like_text_in_a_field_stays_text(value: str, expected: str) -> None:
    raw = (
        "<relationships><relationship><source>A</source><target>B</target>"
        f"<source_text>{value}</source_text></relationship></relationships>"
    )
    parsed = RobustXMLOutputParser(tags=["relationships"]).parse(raw)
    rel = parsed["relationships"]["relationship"]
    assert rel == {"source": "A", "target": "B", "source_text": expected}


def test_tag_like_text_in_descriptions_across_sections() -> None:
    raw = (
        "<entities>\n<entity><name>Repository</name>"
        "<description>findById returns Optional<User> or <T></description>"
        "</entity>\n<entity><name>User</name><description>A record</description>"
        "</entity>\n</entities>\n<relationships>\n<relationship>"
        "<source>Repository</source><target>User</target>"
        "<description>uses <br> tag & a < b</description></relationship>\n"
        "</relationships>"
    )
    parsed = RobustXMLOutputParser(tags=["entities", "relationships"]).parse(raw)
    first, second = parsed["entities"]["entity"]
    assert first["description"] == "findById returns Optional<User> or <T>"
    assert second == {"name": "User", "description": "A record"}
    rel = parsed["relationships"]["relationship"]
    assert rel["description"] == "uses <br> tag & a < b"


def test_record_tag_inside_an_open_field_closes_the_record() -> None:
    # A field left open before the next record: the record tag is markup
    # (it is in the response's vocabulary) and closes the open record.
    raw = (
        "<relationships><relationship><source>A</source><target>B"
        "<relationship><source>B</source><target>C</target></relationship>"
        "</relationships>"
    )
    assert _relationship_pairs(raw) == [("A", "B"), ("B", "C")]
