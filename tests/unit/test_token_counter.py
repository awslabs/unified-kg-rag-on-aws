# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for BedrockTokenCounter (Bedrock-only counting + degradation)."""
from __future__ import annotations

import pytest

from aws_graphrag.aws.token_counter import BedrockTokenCounter

pytestmark = pytest.mark.unit


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
