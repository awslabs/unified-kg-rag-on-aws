# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-query audit record (``unified_kg_rag.audit``): one per query,
however it ends, from the chain every surface runs, with no raw text."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from tests.unit.test_observability import (
    _PASSAGE,
    _SIMPLE_QUERY,
    _DownFactory,
    _FakeModelFactory,
    _RecordingSink,
    _simple_chain,
)
from unified_kg_rag.adapters.observability.usage_callback import (
    UsageCallbackHandler,
)
from unified_kg_rag.application.retrieval import query_audit
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.ingestion.cache_keys import answer_settings_fingerprint
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.audit import AuditOutcome, QueryAuditRecord
from unified_kg_rag.domain.models.config import ModelPrice
from unified_kg_rag.shared.exceptions import (
    InvalidArgumentError,
    QueryTimeoutError,
)
from unified_kg_rag.shared.logging import LoggingSetup

pytestmark = pytest.mark.unit

_ANSWER = "Vendor ships 120 units."
_MODEL = "vendor.model-v1"


def _counted(factory: Any) -> Any:
    """``factory`` whose models count their calls (``shared.usage``)."""

    class Counted(type(factory)):  # type: ignore[misc]
        def get_model(self, model_id: Any, **kwargs: Any) -> Any:
            model = super().get_model(model_id, **kwargs)
            model.callbacks = [UsageCallbackHandler(_MODEL, role="query")]
            return model

    return Counted()


class _Captured(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def fields(self) -> list[dict[str, Any]]:
        """Each record's fields: the event dict under ``setup_logging``,
        else the ``extra`` keys (library mode)."""
        names = set(QueryAuditRecord.model_fields)
        found = []
        for record in self.records:
            data = record.msg if isinstance(record.msg, dict) else vars(record)
            found.append({k: v for k, v in data.items() if k in names})
        return found


@pytest.fixture
def audit() -> Iterator[_Captured]:
    logger = logging.getLogger(query_audit.AUDIT_LOGGER_NAME)
    saved = logger.level
    handler = _Captured()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(saved)


def _one(audit: _Captured) -> QueryAuditRecord:
    (fields,) = audit.fields()
    return QueryAuditRecord.model_validate(fields)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def test_an_answer_writes_one_record_naming_its_build_settings_and_evidence(
    audit: _Captured,
) -> None:
    # G-1: no surface recorded which build and settings answered a query.
    chain = _simple_chain(_counted(_FakeModelFactory()))
    chain.config.models.prices = {
        _MODEL: ModelPrice(input_per_million_tokens=1.0, output_per_million_tokens=2.0)
    }

    async def build_record(suffix: str | None) -> dict[str, Any]:
        return {
            "pipeline_id": "run-7",
            "mode": "incremental",
            "built_at": "2026-11-20T01:02:03Z",
            "documents": 2,
            "extraction_fingerprint": "f" * 16,
            "package_version": "0.3.0",
            "full_build": {"pipeline_id": "run-1", "mode": "full"},
        }

    chain.aread_build_record = build_record  # type: ignore[method-assign]
    query = _SIMPLE_QUERY.model_copy(
        update={"user_id": "user-42", "conversation_id": "c-1"}
    )
    try:
        output = await chain.ainvoke(query)
    finally:
        await chain.aclose()
    record = _one(audit)
    assert record.record_version == 1
    assert record.query_id == output.metadata["query_id"]
    assert (record.mode, record.namespace) == ("rag", "default")
    assert (record.user_id, record.conversation_id) == ("user-42", "c-1")
    assert (record.requested_strategy, record.strategy) == ("simple", "simple")
    assert record.outcome is AuditOutcome.ANSWERED and record.error_class is None
    assert record.config_fingerprint == answer_settings_fingerprint(chain.config)
    assert record.build is not None
    assert (record.build.pipeline_id, record.build.mode) == ("run-7", "incremental")
    assert record.build.full_build_pipeline_id == "run-1"
    assert record.question is not None and record.answer is not None
    assert record.question.sha256 == _sha(query.query)
    assert record.question.length == len(query.query)
    assert record.answer.sha256 == _sha(output.answer)
    assert record.evidence is not None
    assert record.evidence.sources == len(output.sources)
    assert record.evidence.source_ids == ["doc-0", "doc-1"]
    assert record.usage is not None
    assert record.usage.llm_calls == output.metadata["usage"]["llm_calls"]
    assert record.usage.llm_calls == 1
    assert record.models == [_MODEL]
    assert record.cost_usd is not None and record.cost_usd > 0
    assert record.unpriced_models == []


async def test_the_record_holds_no_text_of_the_question_answer_or_sources(
    audit: _Captured,
) -> None:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, list(LoggingSetup._handlers))
    stream = io.StringIO()
    config = Config()
    config.logging.log_to_file = False
    try:
        # The rendered JSON line, as the CLIs write it.
        LoggingSetup.setup_logging(config, stream=stream)
        chain = _simple_chain()
        try:
            output = await chain.ainvoke(_SIMPLE_QUERY)
        finally:
            await chain.aclose()
    finally:
        for handler in LoggingSetup._handlers:
            root.removeHandler(handler)
            handler.close()
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        LoggingSetup._handlers = saved[2]
    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line]
    (line,) = [r for r in lines if r.get("event") == query_audit.AUDIT_EVENT]
    assert line["logger"] == query_audit.AUDIT_LOGGER_NAME
    assert line["level"] == "info"
    QueryAuditRecord.model_validate(
        {k: v for k, v in line.items() if k in QueryAuditRecord.model_fields}
    )
    rendered = json.dumps(line, ensure_ascii=False)
    for text in (_SIMPLE_QUERY.query, output.answer, _PASSAGE, "Vendor"):
        assert text not in rendered


async def test_a_failed_query_writes_one_failed_record(audit: _Captured) -> None:
    chain = _simple_chain(model_factory=_counted(_DownFactory()))
    try:
        with pytest.raises(RuntimeError, match="model down"):
            await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    record = _one(audit)
    assert record.outcome is AuditOutcome.FAILED
    assert record.error_class == "failure"
    assert record.strategy is None and record.answer is None
    assert record.evidence is None
    # Its usage up to there: the failed call.
    assert record.usage is not None and record.usage.failed_calls >= 1
    assert "model down" not in json.dumps(audit.fields(), default=str)


async def test_a_timed_out_query_writes_one_timeout_record(
    audit: _Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _simple_chain()
    chain.config.search.query_timeout_seconds = 0.05
    real_screen = GraphRAGChain._screen_input

    async def slow_screen(self: GraphRAGChain, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(1)
        return await real_screen(self, *args, **kwargs)

    monkeypatch.setattr(GraphRAGChain, "_screen_input", slow_screen)
    try:
        with pytest.raises(QueryTimeoutError):
            await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    record = _one(audit)
    assert (record.outcome, record.error_class) == (AuditOutcome.FAILED, "timeout")


async def test_a_stream_its_reader_stopped_writes_one_stopped_record(
    audit: _Captured,
) -> None:
    chain = _simple_chain()
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        stream = chain.astream(_SIMPLE_QUERY)
        async for _ in stream:
            break
        await stream.aclose()  # type: ignore[attr-defined]
    finally:
        await chain.aclose()
    record = _one(audit)
    assert record.outcome is AuditOutcome.STOPPED
    assert record.error_class is None and record.answer is None
    # As its metrics: one each.
    assert len(sink.emitted) == 1


async def test_a_refused_input_and_a_search_each_write_one_record(
    audit: _Captured,
) -> None:
    chain = _simple_chain()
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        with pytest.raises(InvalidArgumentError):
            await chain.ainvoke({"query": "q", "search_strategy": "no-such"})
        searched = await chain.asearch(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    refused, search = (QueryAuditRecord.model_validate(f) for f in audit.fields())
    assert refused.outcome is AuditOutcome.REFUSED
    assert refused.error_class == "invalid_argument"
    assert refused.requested_strategy == "no-such" and refused.build is None
    assert (search.mode, search.outcome) == ("search", AuditOutcome.ANSWERED)
    assert search.query_id == searched.metadata["query_id"]
    assert search.evidence is not None
    assert search.evidence.source_ids == [
        r.source for r in searched.search_results.results
    ]
    assert search.answer is None
    # A search publishes no answer metrics; its audit record stands.
    assert len(sink.emitted) == 1


async def test_a_query_a_host_gave_up_on_writes_one_record(audit: _Captured) -> None:
    # The MCP server's backstop: the chain was cancelled, the host records.
    chain = _simple_chain()
    try:
        chain.record_unrun_query(
            {"search_strategy": "simple", "suffix": None},
            QueryTimeoutError("past the deadline", timeout_seconds=1),
            started=0.0,
        )
    finally:
        await chain.aclose()
    record = _one(audit)
    assert (record.outcome, record.error_class) == (AuditOutcome.FAILED, "timeout")
    assert record.namespace == "default" and record.question is None


async def test_a_cancelled_ainvoke_writes_none(audit: _Captured) -> None:
    # As its metrics: the host that cancels it records it.
    chain = _simple_chain()
    started = asyncio.Event()
    real_screen = GraphRAGChain._screen_input

    async def held(self: GraphRAGChain, *args: Any, **kwargs: Any) -> Any:
        started.set()
        await asyncio.sleep(10)
        return await real_screen(self, *args, **kwargs)

    chain._screen_input = held.__get__(chain)  # type: ignore[method-assign]
    try:
        task = asyncio.create_task(chain.ainvoke(_SIMPLE_QUERY))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await chain.aclose()
    assert audit.fields() == []


async def test_no_record_is_built_while_the_audit_logger_is_off(
    audit: _Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging.getLogger(query_audit.AUDIT_LOGGER_NAME).setLevel(logging.WARNING)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("built while off")

    monkeypatch.setattr(query_audit, "query_record", refuse)
    chain = _simple_chain()
    try:
        await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    assert audit.fields() == []


async def test_a_record_that_fails_does_not_fail_the_query(
    audit: _Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("broken record")

    monkeypatch.setattr(query_audit, "query_record", broken)
    chain = _simple_chain()
    try:
        output = await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    assert output.answer == _ANSWER
    assert audit.fields() == []


def test_cited_ids_are_bounded_and_flagged() -> None:
    from types import SimpleNamespace

    from unified_kg_rag.domain.models import Citation
    from unified_kg_rag.domain.models.citation import CitedDocument
    from unified_kg_rag.shared.observability import MAX_RECORDED_IDS

    count = MAX_RECORDED_IDS + 5
    output = SimpleNamespace(
        answer="a",
        sources=[{"source": f"s-{i}"} for i in range(count)],
        citations=[
            Citation(
                marker=i + 1,
                cited=i % 2 == 0,
                section_type="text",
                source_id=f"s-{i}",
                text_unit_ids=[f"u-{i}"],
                documents=[CitedDocument(document_id="d-1")],
            )
            for i in range(count)
        ],
    )
    evidence = query_audit._evidence(output)
    assert evidence is not None
    assert evidence.sources == count
    assert len(evidence.source_ids) == MAX_RECORDED_IDS and evidence.truncated
    assert evidence.cited_source_ids[:2] == ["s-0", "s-2"]
    assert evidence.cited_document_ids == ["d-1"]


def test_the_outcome_follows_the_console_s_words() -> None:
    from unified_kg_rag.shared.exceptions import ErrorClass

    outcome = query_audit.outcome_of
    assert outcome(None, raised=False) is AuditOutcome.ANSWERED
    assert outcome(None, raised=False, blocked=True) is AuditOutcome.BLOCKED
    assert outcome(None, raised=False, abstained=True) is AuditOutcome.ABSTAINED
    assert outcome(ErrorClass.FAILURE, raised=False) is AuditOutcome.ERROR
    assert outcome(ErrorClass.TIMEOUT, raised=True) is AuditOutcome.FAILED
    assert outcome(ErrorClass.NOT_INDEXED, raised=True) is AuditOutcome.REFUSED
    assert outcome(ErrorClass.FAILURE, raised=True, stopped=True) is (
        AuditOutcome.STOPPED
    )


def test_the_fingerprint_is_the_eval_manifest_s_hash_of_the_answer_settings() -> None:
    from unified_kg_rag.domain.ingestion.cache_keys import answer_settings

    config = Config()
    # The encoding run-eval manifests used for retrieval_config_sha256.
    encoded = json.dumps(
        answer_settings(config), sort_keys=True, ensure_ascii=False, default=str
    )
    assert answer_settings_fingerprint(config) == _sha(encoded)
    changed = Config()
    changed.search.query_timeout_seconds = 7
    assert answer_settings_fingerprint(changed) != answer_settings_fingerprint(config)
    quiet = Config()
    quiet.logging.level = "WARNING"
    assert answer_settings_fingerprint(quiet) == answer_settings_fingerprint(config)


def test_a_channel_logger_is_the_package_s() -> None:
    from unified_kg_rag.shared.logging import get_channel_logger

    with pytest.raises(ValueError, match="a channel is under"):
        get_channel_logger("audit")
