# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model-free ``TokenCounterPort``: one token per whitespace-separated word."""

from __future__ import annotations


class WordTokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> tuple[str, int]:
        words = text.split()[:max_tokens]
        return " ".join(words), len(words)
