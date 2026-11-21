# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query flags shared by run-rag and run-eval (one RAGInput per query)."""

from __future__ import annotations

import argparse
import sys
from typing import Any, get_args

from unified_kg_rag.application.cli.conventions import add_removed_flag
from unified_kg_rag.domain.models import (
    Config,
    SearchStrategy,
    SearchType,
    validate_safe_name,
)
from unified_kg_rag.domain.models.config import EffortLevel
from unified_kg_rag.domain.models.namespace import (
    DEPRECATED_NAMESPACE_NAMES,
    deprecated_namespace_warning,
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


def namespace_arg(value: str) -> str:
    """An argparse ``type``: a namespace, validated as the config's is."""
    try:
        return validate_safe_name(value, "--namespace")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


class _NamespaceFlag(argparse.Action):
    """``--namespace``, or its deprecated alias ``--suffix`` (a warning on
    stderr): given twice, both must name the same namespace."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        given = str(values)
        previous: str | None = getattr(namespace, self.dest)
        if option_string in DEPRECATED_NAMESPACE_NAMES:
            print(
                f"{parser.prog}: warning: {deprecated_namespace_warning(option_string)}",
                file=sys.stderr,
            )
        if previous is not None and previous != given:
            parser.error(
                f"'{previous}' and '{given}' name different namespaces; give "
                "--namespace once"
            )
        setattr(namespace, self.dest, given)


def add_namespace_argument(parser: Any, help: str) -> None:
    """Add ``--namespace`` (``help``) and its deprecated alias ``--suffix``
    (left out of ``--help``) to ``parser`` (or an argument group), both
    into ``args.namespace``."""
    parser.add_argument(
        "--namespace", type=namespace_arg, action=_NamespaceFlag, help=help
    )
    for old, new in DEPRECATED_NAMESPACE_NAMES.items():
        if new == "--namespace":
            parser.add_argument(
                old,
                type=namespace_arg,
                action=_NamespaceFlag,
                dest="namespace",
                help=argparse.SUPPRESS,
            )


def query_namespace(args: argparse.Namespace, config: Config) -> str:
    """The namespace to query: ``--namespace``, else the config's
    (``resolve_namespace``)."""
    given: str | None = getattr(args, "namespace", None)
    return resolve_namespace(config, given)


def namespace_name(namespace: str | None, config: Config) -> str:
    """The index namespace a run writes or queries, as the CLIs print it:
    the namespace (``default`` when unset) and ``indexing.additional_suffix``,
    what every index name and graph label ends in."""
    name = resolve_namespace(config, namespace)
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
    """Add the namespace, strategy, search type and width flags.

    With ``multiple_strategies`` (run-eval), ``--search-strategy`` takes a
    comma-separated list and parses to a list of names.
    """
    add_namespace_argument(
        parser,
        "Namespace to query (default: the config's "
        "processing.document_parsing.namespace, as ingestion writes, else "
        "'default'); must match the namespace the corpus was ingested into. "
        "--suffix is its deprecated alias",
    )
    efforts = get_args(EffortLevel)
    parser.add_argument(
        "--effort",
        choices=efforts,
        default=None,
        help="Reasoning effort of the answer (RAGInput.effort): the answer "
        "model reasons at that level, a deeper one at more output tokens and "
        "latency; only the answer call takes it (the query's other calls keep "
        "their tier's). A model that does not reason, or does not take the "
        "level, refuses it before any call (exit 2). Default: the effort of "
        "the answer model's tier (aws.bedrock.default_effort, or fast_effort "
        "when models.roles.answer_generation is the fast tier's model). "
        + (
            # run-eval
            "A dataset item's metadata.effort wins over it; every effort the "
            "run asks for is checked before any run starts (with each --arm's "
            "and --closed-book's answer model), so one a model refuses stops "
            "the run before any answer"
            if multiple_strategies
            else "A search (--mode search) makes no answer and ignores it"
        ),
    )
    add_removed_flag(
        parser,
        "--enable-thinking",
        f"set the answer's reasoning effort with --effort {{{','.join(efforts)}}} "
        "(RAGInput.effort); the default models (Claude 5.x) always reason, at "
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
        default=None,
        choices=[st.value for st in SearchType],
        help="Search method (default: the config's search.search_type, "
        "hybrid unless set): hybrid fuses a lexical (BM25) and a "
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
        help="Set the most results a retrieval-only search returns (times "
        "--retrieval-multiplier; default 10), cut by rank after fusion and "
        "rerank. An answer's context may hold more: each strategy reads "
        "wider pools of its own",
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
