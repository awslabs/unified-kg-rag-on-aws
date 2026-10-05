# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BedrockTokenCounter (Bedrock-only counting + degradation)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

import pytest
from botocore.exceptions import ClientError

from unified_kg_rag.adapters.aws.token_counter import (
    BedrockTokenCounter,
    clear_count_tokens_unsupported_cache,
    estimate_token_count,
    is_count_tokens_known_unsupported,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_unsupported_cache() -> Iterator[None]:
    # The negative cache is process-wide; isolate every test from it.
    clear_count_tokens_unsupported_cache()
    yield
    clear_count_tokens_unsupported_cache()


class TestEstimateTokenCount:
    def test_empty_is_zero(self) -> None:
        assert estimate_token_count("") == 0

    def test_english_uses_word_count(self) -> None:
        # 5 short words; char/4 (~5) and word count (5) are comparable.
        assert estimate_token_count("the quick brown fox jumps") >= 5

    def test_spaceless_cjk_not_undercounted(self) -> None:
        # Whitespace split would yield 1 and a flat chars/4 would yield len/4
        # (~4x too low). Dense CJK/Hangul chars must be counted at ~1 token each
        # so an over-limit chunk is truncated before the embedding call rather
        # than slipping past and failing with "Too many input tokens".
        text = "한국어문장입니다이것은긴문장이다"  # 16 chars, 0 spaces
        assert estimate_token_count(text) == len(text)
        assert estimate_token_count(text) > len(text) // 4
        assert estimate_token_count(text) > len(text.split())

    def test_latin_uses_char_over_four(self) -> None:
        # Non-dense scripts stay at ~4 chars/token (word count floors it).
        text = "the quick brown fox jumps over the lazy dog"
        assert estimate_token_count(text) <= len(text) // 4 + len(text.split())

    def test_mixed_script_adds_dense_and_sparse(self) -> None:
        # 4 dense chars (~4 tokens) + 8 latin chars (~2 tokens) ~= 6.
        assert estimate_token_count("한국어문 abcdefgh") >= 5

    def test_never_zero_for_nonempty(self) -> None:
        assert estimate_token_count("가") == 1


def test_count_tokens_degrades_to_script_aware_estimate(mocker) -> None:
    # When the Bedrock API raises, count_tokens must fall back to the
    # script-aware estimate (not whitespace split) for a space-less string.
    counter = BedrockTokenCounter("model", object())
    mocker.patch.object(counter, "_cached_count", side_effect=RuntimeError("api down"))
    text = "한국어문장입니다이것은긴문장이다"
    assert counter.count_tokens(text) == estimate_token_count(text)


class FakeBedrockClient:
    """Returns a fixed token count proportional to character length.

    Mirrors the real CountTokens wire shape: the payload nests under ``input``
    and the response key is ``inputTokens``. An earlier fake accepted a
    top-level ``converse=`` and returned ``totalTokens``, which matched the
    (wrong) call the adapter made — so the tests passed while every real call
    failed and silently degraded to the estimate.
    """

    def __init__(self, chars_per_token: int = 4) -> None:
        self.chars_per_token = chars_per_token
        self.calls = 0

    def count_tokens(self, **kwargs) -> dict:
        self.calls += 1
        if "converse" in kwargs:
            raise AssertionError(
                "'converse' must be nested under 'input', not passed top-level"
            )
        text = kwargs["input"]["converse"]["messages"][0]["content"][0]["text"]
        return {"inputTokens": max(1, len(text) // self.chars_per_token)}


class FailingClient:
    def count_tokens(self, **kwargs) -> dict:
        raise RuntimeError("model does not support count_tokens")


def test_uses_bedrock_api() -> None:
    counter = BedrockTokenCounter("model", FakeBedrockClient(chars_per_token=4))
    assert counter.count_tokens("a" * 40) == 10


def test_empty_text_is_zero_without_api_call() -> None:
    client = FakeBedrockClient()
    counter = BedrockTokenCounter("model", client)
    assert counter.count_tokens("") == 0
    assert client.calls == 0


def test_lru_cache_avoids_repeat_calls() -> None:
    client = FakeBedrockClient()
    counter = BedrockTokenCounter("model", client)
    counter.count_tokens("hello world")
    counter.count_tokens("hello world")
    assert client.calls == 1


def test_degrades_to_word_count_on_api_failure() -> None:
    counter = BedrockTokenCounter("model", FailingClient())
    # No exception propagates; degrades to whitespace word count.
    assert counter.count_tokens("one two three four") == 4


def test_request_uses_documented_count_tokens_wire_shape() -> None:
    # Regression: the request must nest 'converse' under 'input' and read
    # 'inputTokens' back. Getting either wrong is invisible at runtime — the
    # failure is swallowed by the estimate fallback — so pin the shape here.
    captured: dict[str, object] = {}

    class _Recorder:
        def count_tokens(self, **kwargs: object) -> dict:
            captured.update(kwargs)
            return {"inputTokens": 7}

    counter = BedrockTokenCounter("model-x", _Recorder())
    assert counter.count_tokens("hello") == 7
    assert captured["modelId"] == "model-x"
    assert "converse" not in captured
    payload = captured["input"]
    assert isinstance(payload, dict)
    assert payload["converse"]["messages"][0]["content"][0]["text"] == "hello"


def test_truncate_converges_under_limit() -> None:
    counter = BedrockTokenCounter("model", FakeBedrockClient(chars_per_token=4))
    text = "word " * 200  # ~1000 chars -> ~250 tokens
    truncated, count = counter.truncate_to_token_limit(text, max_tokens=50)
    assert count <= 50
    assert len(truncated) < len(text)


def test_truncate_noop_when_within_limit() -> None:
    counter = BedrockTokenCounter("model", FakeBedrockClient())
    text = "short text"
    truncated, count = counter.truncate_to_token_limit(text, max_tokens=1000)
    assert truncated == text


# --------------------------------------------------------------------------- #
# Negative cache for models that reject CountTokens
# --------------------------------------------------------------------------- #
_UNSUPPORTED_MODEL_MESSAGE = "The provided model doesn't support counting tokens."


def _client_error(code: str, message: str | None = None) -> ClientError:
    if message is None:
        # A ValidationException is only permanent when it names the model as
        # unsupported; other codes are permanent regardless of the message.
        message = (
            _UNSUPPORTED_MODEL_MESSAGE if code == "ValidationException" else "synthetic"
        )
    return ClientError({"Error": {"Code": code, "Message": message}}, "CountTokens")


class ErrorClient:
    """Raises a configured error on every call and counts calls."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def count_tokens(self, **kwargs: object) -> dict:
        self.calls += 1
        raise self.error


@pytest.mark.parametrize(
    "code",
    [
        "ValidationException",
        "AccessDeniedException",
        "UnknownOperationException",
        "ResourceNotFoundException",
    ],
)
def test_unsupported_error_makes_exactly_one_api_call(code: str) -> None:
    client = ErrorClient(_client_error(code))
    counter = BedrockTokenCounter("unsupported-model", client)
    texts = ["same text"] * 5 + [f"different text {i}" for i in range(20)]
    results = [counter.count_tokens(t) for t in texts]
    assert client.calls == 1
    assert results == [estimate_token_count(t) for t in texts]


def test_negative_cache_is_shared_across_instances_of_same_model() -> None:
    first = ErrorClient(_client_error("ValidationException"))
    BedrockTokenCounter("shared-model", first).count_tokens("alpha")
    second = ErrorClient(_client_error("ValidationException"))
    BedrockTokenCounter("shared-model", second).count_tokens("beta")
    assert first.calls == 1
    assert second.calls == 0


def test_negative_cache_is_per_model_id() -> None:
    BedrockTokenCounter(
        "bad-model", ErrorClient(_client_error("ValidationException"))
    ).count_tokens("alpha")
    good = FakeBedrockClient(chars_per_token=4)
    assert BedrockTokenCounter("good-model", good).count_tokens("a" * 40) == 10
    assert good.calls == 1


@pytest.mark.parametrize(
    "error",
    [
        _client_error("ValidationException", "Malformed input request: text is blank"),
        _client_error("ThrottlingException"),
        _client_error("ServiceUnavailableException"),
        _client_error("InternalServerException"),
        _client_error("ModelNotReadyException"),
        TimeoutError("read timed out"),
        RuntimeError("connection reset"),
    ],
)
def test_transient_error_does_not_disable_api(error: Exception) -> None:
    client = ErrorClient(error)
    counter = BedrockTokenCounter("flaky-model", client)
    counter.count_tokens("one two")
    counter.count_tokens("one two")
    counter.count_tokens("three four")
    assert client.calls == 3
    assert not is_count_tokens_known_unsupported("flaky-model")


@pytest.mark.parametrize(
    "message",
    [
        "The provided model doesn't support counting tokens.",
        "This model does not support the CountTokens operation",
        "Operation not supported for this model",
        "UNSUPPORTED model",
    ],
)
def test_validation_exception_unsupported_message_marks_model(message: str) -> None:
    client = ErrorClient(_client_error("ValidationException", message))
    counter = BedrockTokenCounter("vx-model", client)
    counter.count_tokens("alpha")
    counter.count_tokens("beta")
    assert client.calls == 1
    assert is_count_tokens_known_unsupported("vx-model")


def test_input_validation_exception_fails_only_that_call() -> None:
    class _RejectsOneInput:
        def __init__(self) -> None:
            self.calls = 0

        def count_tokens(self, **kwargs: object) -> dict:
            self.calls += 1
            text = kwargs["input"]["converse"]["messages"][0]["content"][0]["text"]  # type: ignore[index]
            if text == "bad input":
                raise _client_error(
                    "ValidationException", "Input is too long for model"
                )
            return {"inputTokens": 7}

    client = _RejectsOneInput()
    counter = BedrockTokenCounter("input-model", client)
    assert counter.count_tokens("bad input") == estimate_token_count("bad input")
    assert not is_count_tokens_known_unsupported("input-model")
    assert counter.count_tokens("good input") == 7
    assert client.calls == 2


@pytest.mark.parametrize("text", ["\n\n", "   ", "\t \r\n"])
def test_whitespace_only_text_skips_api_and_keeps_it_enabled(text: str) -> None:
    client = FakeBedrockClient()
    counter = BedrockTokenCounter("ws-model", client)
    assert counter.count_tokens(text) == estimate_token_count(text)
    assert client.calls == 0
    assert not is_count_tokens_known_unsupported("ws-model")
    counter.count_tokens("real text")
    assert client.calls == 1


def test_api_recovers_after_transient_error() -> None:
    class _FlakyOnce:
        def __init__(self) -> None:
            self.calls = 0

        def count_tokens(self, **kwargs: object) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise _client_error("ThrottlingException")
            return {"inputTokens": 42}

    client = _FlakyOnce()
    counter = BedrockTokenCounter("model", client)
    assert counter.count_tokens("x y z") == estimate_token_count("x y z")
    assert counter.count_tokens("x y z") == 42
    assert counter.count_tokens("x y z") == 42
    assert client.calls == 2


def test_supported_model_still_caches_per_text() -> None:
    client = FakeBedrockClient()
    counter = BedrockTokenCounter("model", client)
    for _ in range(3):
        counter.count_tokens("alpha")
        counter.count_tokens("beta")
    assert client.calls == 2


def test_client_missing_operation_is_treated_as_unsupported() -> None:
    # An outdated botocore client has no count_tokens attribute.
    counter = BedrockTokenCounter("old-sdk-model", object())
    counter.count_tokens("alpha")
    assert is_count_tokens_known_unsupported("old-sdk-model")


def test_api_supported_false_never_calls_client() -> None:
    client = FakeBedrockClient()
    counter = BedrockTokenCounter("embed-model", client, api_supported=False)
    text = "one two three"
    assert counter.count_tokens(text) == estimate_token_count(text)
    counter.truncate_to_token_limit("word " * 200, max_tokens=10)
    assert client.calls == 0


def test_none_client_falls_back_to_estimate() -> None:
    counter = BedrockTokenCounter("rerank-model", None)
    assert counter.count_tokens("one two") == estimate_token_count("one two")


def test_downgrade_logged_once_at_info(caplog: pytest.LogCaptureFixture) -> None:
    client = ErrorClient(_client_error("ValidationException"))
    counter = BedrockTokenCounter("log-model", client)
    with caplog.at_level(logging.INFO):
        for i in range(5):
            counter.count_tokens(f"text {i}")
    records = [r for r in caplog.records if "unsupported for model" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO


def test_concurrent_callers_stop_calling_once_marked() -> None:
    client = ErrorClient(_client_error("ValidationException"))
    counter = BedrockTokenCounter("race-model", client)
    threads = [
        threading.Thread(target=counter.count_tokens, args=(f"t{i}",))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert is_count_tokens_known_unsupported("race-model")
    # Racing threads may each issue a first call, but never more than one each.
    assert 1 <= client.calls <= 16
    before = client.calls
    counter.count_tokens("after")
    assert client.calls == before
