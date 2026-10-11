# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query flags shared by run-rag and run-eval (one RAGInput per query)."""

from __future__ import annotations

import argparse

from unified_kg_rag.domain.models import SearchStrategy, SearchType


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def add_query_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the index suffix, thinking, strategy, search type and width flags."""
    parser.add_argument(
        "--suffix",
        help="Suffix for multi-tenant or versioned indices",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking mode for language model reasoning and step-by-step "
            "problem solving. No-op for models that always reason (Claude 4.7+ / "
            "5.x, OpenAI GPT); tune their depth with aws.bedrock.default_effort "
            "/ fast_effort in the config"
        ),
    )
    parser.add_argument(
        "--search-strategy",
        default="auto",
        choices=[ss.value for ss in SearchStrategy],
        help="Choose the search strategy",
    )
    parser.add_argument(
        "--search-type",
        default="hybrid",
        choices=[st.value for st in SearchType],
        help="Specify the search method",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=10,
        help="Set the maximum number of search results",
    )
    parser.add_argument(
        "--retrieval-multiplier",
        type=_positive_int,
        default=1,
        help="Set the retrieval multiplier for increasing search depth",
    )
