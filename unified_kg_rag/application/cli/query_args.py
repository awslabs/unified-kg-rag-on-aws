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


def _strategy_list(value: str) -> list[str]:
    """A comma-separated list of distinct search strategies (``local,mix``)."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    valid = [s.value for s in SearchStrategy]
    unknown = [name for name in names if name not in valid]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid strategy {', '.join(unknown) or repr(value)} "
            f"(choose from {', '.join(valid)})"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"strategy listed twice in '{value}'")
    return names


def add_query_arguments(
    parser: argparse.ArgumentParser, *, multiple_strategies: bool = False
) -> None:
    """Add the index suffix, thinking, strategy, search type and width flags.

    With ``multiple_strategies`` (run-eval), ``--search-strategy`` takes a
    comma-separated list and parses to a list of names.
    """
    parser.add_argument(
        "--suffix",
        help="Index suffix to query (default: 'default'); must match the "
        "corpus's processing.document_parsing.index_value",
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
    if multiple_strategies:
        parser.add_argument(
            "--search-strategy",
            default=["auto"],
            type=_strategy_list,
            help=(
                "Search strategy, or a comma-separated list (e.g. local,mix,global) "
                "to answer the same queries with each and compare them pairwise "
                "against the first"
            ),
        )
    else:
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
        help="Search method. The lexical side searches the first 150 words of "
        "the question (a Hangul, Han or Kana character counts as a word) and "
        "tolerates typos only in questions of up to 10 words; the vector side "
        "embeds the whole question",
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
        help="Multiplier that increases the retrieval depth",
    )
