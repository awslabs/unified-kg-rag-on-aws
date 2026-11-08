# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query flags shared by run-rag and run-eval (one RAGInput per query)."""

from __future__ import annotations

import argparse

from unified_kg_rag.application.cli.conventions import add_removed_flag
from unified_kg_rag.domain.models import (
    Config,
    SearchStrategy,
    SearchType,
    validate_safe_name,
)
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


def positive_int(value: str) -> int:
    """An argparse ``type``: an integer of at least 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def suffix_arg(value: str) -> str:
    """An argparse ``type``: an index suffix, validated like ``index_value``."""
    try:
        return validate_safe_name(value, "--suffix")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def query_suffix(args: argparse.Namespace, config: Config) -> str:
    """The index suffix to query: ``--suffix``, else the config's
    (``resolve_namespace``)."""
    given: str | None = getattr(args, "suffix", None)
    return resolve_namespace(config, given)


def namespace_name(suffix: str | None, config: Config) -> str:
    """The index namespace a run writes or queries, as the CLIs print it:
    the suffix (``default`` when unset) and ``indexing.additional_suffix``,
    what every index name and graph label ends in."""
    name = resolve_namespace(config, suffix)
    if additional := config.indexing.additional_suffix:
        name = f"{name}-{additional}"
    return name


def _strategy_list(value: str) -> list[str]:
    """A comma-separated list of distinct search strategies (``local,mix``)."""
    names = [name.strip() for name in value.split(",")]
    valid = [s.value for s in SearchStrategy]
    unknown = [name for name in names if name and name not in valid]
    if "" in names or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid strategy {', '.join(unknown) or repr(value)} "
            f"(choose from {', '.join(valid)})"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"strategy listed twice in '{value}'")
    return names


class _StrategyListOnce(argparse.Action):
    """Store a ``_strategy_list``, given once: a second ``--search-strategy``
    replaced the first without a word, so a comparison ran one strategy."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        # Still the default list object until the flag is first given.
        if getattr(namespace, self.dest) is not self.default:
            parser.error(
                f"{option_string} given more than once; list the strategies in "
                "one, comma-separated (local,mix)"
            )
        setattr(namespace, self.dest, values)


def add_query_arguments(
    parser: argparse.ArgumentParser, *, multiple_strategies: bool = False
) -> None:
    """Add the index suffix, strategy, search type and width flags.

    With ``multiple_strategies`` (run-eval), ``--search-strategy`` takes a
    comma-separated list and parses to a list of names.
    """
    parser.add_argument(
        "--suffix",
        type=suffix_arg,
        help="Index suffix to query (default: the config's "
        "processing.document_parsing.index_value, as ingestion writes, else "
        "'default'); must match the index_value the corpus was ingested with",
    )
    add_removed_flag(
        parser,
        "--enable-thinking",
        "the default models (Claude 5.x) always reason; tune their depth with "
        "aws.bedrock.default_effort / fast_effort in the config",
    )
    if multiple_strategies:
        parser.add_argument(
            "--search-strategy",
            default=["auto"],
            type=_strategy_list,
            action=_StrategyListOnce,
            help=(
                "Search strategy, or a comma-separated list (e.g. local,mix,global) "
                "to answer the same queries with each and compare them pairwise "
                "against the first. Give the flag once: a second one is a usage "
                "error"
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
        help="Search method: hybrid (default) fuses a lexical (BM25) and a "
        "vector (embedding kNN) search of each index, lexical and vector run "
        "one of them alone. The lexical side searches the first 150 words of "
        "the question (a Hangul, Han or Kana character counts as a word) and, "
        "on OpenSearch, tolerates typos only in questions of up to 10 words "
        "with no Hangul, Han or Kana; the vector side embeds the whole question",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=10,
        help="Set the maximum number of search results (times "
        "--retrieval-multiplier; default 10)",
    )
    parser.add_argument(
        "--retrieval-multiplier",
        type=positive_int,
        default=1,
        help="Widen the search N times: each index and graph expansion fetches "
        "up to top-k x N candidates and the fused list keeps up to top-k x N "
        "for the reranker and the context budget, at the cost of larger "
        "searches (default 1)",
    )
