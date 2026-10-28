# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ingestion LLM response cache (AWS-free)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.llm.response_cache import (
    CachingLLMFactory,
    LLMResponseCache,
)
from unified_kg_rag.domain.models import ModelPurpose
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

pytestmark = pytest.mark.unit


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, system: str, human: str) -> str:
        self.calls += 1
        return f"answer to {human}"


def _model(cache: LLMResponseCache, respond: Any, purpose=ModelPurpose.INGESTION):  # type: ignore[no-untyped-def]
    factory = CachingLLMFactory(ScriptedLLMFactory(respond), cache)
    return factory.get_model("model-a", model_purpose=purpose)


def _ask(model: Any, text: str = "q") -> str:
    return str(model.invoke([HumanMessage(content=text)]).content)


def test_a_later_run_answers_from_the_responses_of_an_earlier_one(
    tmp_path: Path,
) -> None:
    respond = _Counter()
    first = LLMResponseCache()
    first.open(tmp_path)
    assert _ask(_model(first, respond)) == "answer to q"
    first.close()

    second = LLMResponseCache()
    second.open(tmp_path)
    assert _ask(_model(second, respond)) == "answer to q"
    assert respond.calls == 1
    assert (second.hits, second.misses) == (1, 0)


def test_an_entry_is_served_once_per_run(tmp_path: Path) -> None:
    # A retry within the run (after its response failed to parse, say) asks
    # the model again, and the new response replaces the old one.
    respond = _Counter()
    first = LLMResponseCache()
    first.open(tmp_path)
    _ask(_model(first, respond))
    first.close()

    second = LLMResponseCache()
    second.open(tmp_path)
    model = _model(second, respond)
    _ask(model)
    _ask(model)
    assert respond.calls == 2


def test_responses_of_the_same_run_are_not_served(tmp_path: Path) -> None:
    respond = _Counter()
    cache = LLMResponseCache()
    cache.open(tmp_path)
    model = _model(cache, respond)
    _ask(model)
    _ask(model)
    assert respond.calls == 2


def test_the_key_covers_the_prompt_and_the_model(tmp_path: Path) -> None:
    cache = LLMResponseCache()
    cache.open(tmp_path)
    cache.update("p", "model-a", [ChatGeneration(message=AIMessage(content="a"))])
    cache.close()
    cache.open(tmp_path)
    assert cache.lookup("p", "model-b") is None
    assert cache.lookup("q", "model-a") is None


def test_object_addresses_do_not_change_the_key(tmp_path: Path) -> None:
    # A model that is not LangChain-serializable lists its fields in the key,
    # an object's repr included, whose address differs per process.
    cache = LLMResponseCache()
    cache.open(tmp_path)
    reply = [ChatGeneration(message=AIMessage(content="a"))]
    cache.update("p", "[('fn', <function f at 0x7f00aa>)]", reply)
    cache.close()
    cache.open(tmp_path)
    hit = cache.lookup("p", "[('fn', <function f at 0x7fbb11>)]")
    assert hit is not None and hit[0].text == "a"


def test_a_truncated_response_is_not_stored(tmp_path: Path) -> None:
    # Its failure is not retried, so replaying it would fail the item forever.
    cache = LLMResponseCache()
    cache.open(tmp_path)
    cut = AIMessage(
        content="<entities>", response_metadata={"stopReason": "max_tokens"}
    )
    cache.update("p", "m", [ChatGeneration(message=cut)])
    cache.close()
    cache.open(tmp_path)
    assert cache.lookup("p", "m") is None


def test_a_force_rebuild_stores_but_serves_nothing(tmp_path: Path) -> None:
    respond = _Counter()
    first = LLMResponseCache()
    first.open(tmp_path)
    _ask(_model(first, respond))
    first.close()

    rebuild = LLMResponseCache()
    rebuild.open(tmp_path, lookups=False)
    _ask(_model(rebuild, respond))
    rebuild.close()
    assert respond.calls == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_a_full_segment_is_written_and_handed_over(tmp_path: Path) -> None:
    written: list[Path] = []
    cache = LLMResponseCache(flush_every=2)
    cache.open(tmp_path, on_segment=written.append)
    model = _model(cache, _Counter())
    _ask(model, "a")
    assert written == []
    _ask(model, "b")
    assert len(written) == 1 and written[0].is_file()
    _ask(model, "c")
    cache.close()  # the partial segment
    assert len(written) == 2


def test_a_failed_hand_off_never_fails_the_model_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    handed: list[Path] = []
    down = [True]

    def upload(path: Path) -> None:
        if down[0]:
            raise ConnectionError("s3 unreachable")
        handed.append(path)

    cache = LLMResponseCache(flush_every=1)
    cache.open(tmp_path, on_segment=upload)
    model = _model(cache, _Counter())
    assert _ask(model, "a") == "answer to a"
    assert _ask(model, "b") == "answer to b"
    assert "s3 unreachable" in caplog.text and handed == []
    # Back up: the next write hands off the earlier segments too, in order.
    down[0] = False
    _ask(model, "c")
    assert handed == sorted(tmp_path.glob("*.json"))
    assert len(handed) == 3


def test_an_unreadable_segment_is_skipped(tmp_path: Path) -> None:
    respond = _Counter()
    first = LLMResponseCache()
    first.open(tmp_path)
    _ask(_model(first, respond))
    first.close()
    (tmp_path / "00000000000000000000-bad.json").write_text("{not json")

    second = LLMResponseCache()
    second.open(tmp_path)
    _ask(_model(second, respond))
    assert respond.calls == 1


def test_an_unopened_cache_neither_serves_nor_stores(tmp_path: Path) -> None:
    respond = _Counter()
    cache = LLMResponseCache()
    _ask(_model(cache, respond))
    _ask(_model(cache, respond))
    assert respond.calls == 2


@pytest.mark.parametrize("purpose", [ModelPurpose.QUERY, ModelPurpose.EVALUATION])
def test_only_ingestion_models_are_cached(tmp_path: Path, purpose) -> None:  # type: ignore[no-untyped-def]
    cache = LLMResponseCache()
    model = _model(cache, _Counter(), purpose)
    assert model.cache is None


def test_the_caching_factory_keeps_the_wrapped_factory_s_policies() -> None:
    inner = ScriptedLLMFactory(_Counter())
    inner.transient_retry = "policy"  # type: ignore[attr-defined]
    factory = CachingLLMFactory(inner, LLMResponseCache())
    assert factory.transient_retry == "policy"
    assert factory.get_model_info("model-a") == inner.get_model_info("model-a")


def _batch(
    model: Any, texts: list[str], parse: Any, max_attempts: int = 1
) -> list[Any]:
    """Ask ``model`` each of ``texts`` as one item of a batch, its answer
    checked by ``parse`` (raising fails the item)."""
    return BatchProcessor(
        max_attempts=max_attempts, retry_multiplier=1.0, retry_max_wait=0
    ).execute_with_fallback(
        texts,
        lambda items: [{"text": text} for text in items],
        lambda request: parse(_ask(model, request["text"])),
        "test",
        show_progress=False,
    )


def _rejecting(bad: str) -> Any:
    def parse(answer: str) -> str:
        if bad in answer:
            raise OutputParserException(f"cannot parse {answer}")
        return answer

    return parse


@pytest.mark.parametrize("max_attempts", [1, 3])
def test_the_responses_of_a_failed_item_are_not_served_again(
    tmp_path: Path, max_attempts: int
) -> None:
    # A re-run must ask the model again for the item a bad response failed
    # (not replay it and record the same failure), and only for that item.
    respond = _Counter()
    first = LLMResponseCache()
    first.open(tmp_path)
    results = _batch(
        _model(first, respond), ["good", "bad"], _rejecting("bad"), max_attempts
    )
    assert results[0] == "answer to good" and results[1] is BATCH_ITEM_FAILED
    assert respond.calls == 1 + max_attempts
    first.close()

    second = LLMResponseCache()
    second.open(tmp_path)
    asked = respond.calls
    results = _batch(_model(second, respond), ["good", "bad"], lambda answer: answer)
    assert results == ["answer to good", "answer to bad"]
    assert respond.calls == asked + 1
    assert (second.hits, second.misses) == (1, 1)
    second.close()

    # The answer that parsed is kept for the next run.
    third = LLMResponseCache()
    third.open(tmp_path)
    _batch(_model(third, respond), ["good", "bad"], lambda answer: answer)
    assert respond.calls == asked + 1


def test_a_served_response_its_item_rejects_is_dropped(tmp_path: Path) -> None:
    # Stored by an earlier run (a parser since tightened, a response from
    # before this rule): once its item fails on it, no later run gets it.
    respond = _Counter()
    first = LLMResponseCache()
    first.open(tmp_path)
    _ask(_model(first, respond), "bad")
    first.close()

    second = LLMResponseCache()
    second.open(tmp_path)
    results = _batch(_model(second, respond), ["bad"], _rejecting("bad"))
    assert results == [BATCH_ITEM_FAILED] and respond.calls == 1
    second.close()

    third = LLMResponseCache()
    third.open(tmp_path)
    _ask(_model(third, respond), "bad")
    assert respond.calls == 2


def _late_first_attempt(second: str) -> tuple[Any, threading.Event]:
    """A model whose first call hangs past the call timeout and answers
    "late" once released; later calls answer ``second``."""
    release, answered = threading.Event(), threading.Event()
    calls: list[str] = []

    def respond(system: str, human: str) -> str:
        calls.append(human)
        if len(calls) == 1:
            release.wait(10)
            answered.set()
            return "late"
        return second

    respond.release = release  # type: ignore[attr-defined]
    return respond, answered


@pytest.mark.parametrize("second", ["good", "bad"])
def test_a_timed_out_attempt_s_late_answer_is_not_stored(
    tmp_path: Path, second: str
) -> None:
    # The abandoned call answered after its item's retry: stored last, a
    # re-run was served it (over the retry's good answer), or kept it after
    # the item's failure had already dropped its responses.
    respond, answered = _late_first_attempt(second)
    cache = LLMResponseCache()
    cache.open(tmp_path)
    model = _model(cache, respond)
    results = BatchProcessor(
        max_attempts=2, retry_multiplier=1.0, retry_max_wait=0, call_timeout_seconds=1
    ).execute_with_fallback(
        ["q"],
        lambda items: [{"text": text} for text in items],
        lambda request: _rejecting("bad")(_ask(model, request["text"])),
        "test",
        show_progress=False,
    )
    assert results == (["good"] if second == "good" else [BATCH_ITEM_FAILED])
    respond.release.set()  # type: ignore[attr-defined]
    assert answered.wait(10)
    time.sleep(0.2)  # the abandoned call's cache update
    cache.close()

    rerun = LLMResponseCache()
    rerun.open(tmp_path)
    expected = "good" if second == "good" else "asked again"
    assert _ask(_model(rerun, lambda system, human: "asked again")) == expected


def test_a_dropped_response_still_pending_writes_nothing(tmp_path: Path) -> None:
    # A response stored and dropped before any flush is never written: no
    # entry, no removal entry (nothing on disk to remove).
    respond = _Counter()
    cache = LLMResponseCache()
    cache.open(tmp_path)
    _batch(_model(cache, respond), ["good", "bad"], _rejecting("bad"))
    cache.close()
    (segment,) = tmp_path.glob("*.json")
    entries = json.loads(segment.read_text(encoding="utf-8"))["entries"]
    assert [entry["generations"] is None for entry in entries] == [False]


def test_a_dropped_response_already_written_is_removed(tmp_path: Path) -> None:
    # Flushed (a full segment) before its item failed: a removal entry.
    respond = _Counter()
    cache = LLMResponseCache(flush_every=1)
    cache.open(tmp_path)
    _batch(_model(cache, respond), ["bad"], _rejecting("bad"))
    cache.close()
    rerun = LLMResponseCache()
    rerun.open(tmp_path)
    _ask(_model(rerun, respond), "bad")
    assert respond.calls == 2
