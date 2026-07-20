# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BedrockTokenCounter (Bedrock-only counting + degradation)."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.aws.token_counter import (
    BedrockTokenCounter,
    estimate_token_count,
)

pytestmark = pytest.mark.unit


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
    """Returns a fixed token count proportional to character length."""

    def __init__(self, chars_per_token: int = 4) -> None:
        self.chars_per_token = chars_per_token
        self.calls = 0

    def count_tokens(self, **kwargs) -> dict:
        self.calls += 1
        text = kwargs["converse"]["messages"][0]["content"][0]["text"]
        return {"totalTokens": max(1, len(text) // self.chars_per_token)}


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
