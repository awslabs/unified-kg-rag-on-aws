# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark``: a public benchmark as a corpus and a run-eval dataset.

Writes ``<output-directory>/corpus/`` (ingest it with ``run-ingestion``) and
``<output-directory>/eval_data.json`` (score it with ``run-eval``). The
benchmark data is read from a local copy (``--data-path``) or downloaded at
run time; it is not part of this package. ``--unanswerable N`` adds N
unanswerable questions (``metadata.answerable: false``) from the full split,
filtered so the pooled corpus does not answer them. ``prepare-benchmark
assertions`` instead generates questions and assertions from your own corpus
(``--source-directory``; ``application.cli.assertion_benchmark``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unified_kg_rag.application.cli.assertion_benchmark import (
    add_assertion_arguments,
    run_assertion_benchmark,
)
from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    ArgumentParser,
    add_output_format_argument,
    exits_on_sigterm,
    json_output,
    print_json,
    usage_error,
)
from unified_kg_rag.application.cli.display import console, markup_safe
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.query_args import positive_int
from unified_kg_rag.evaluation.benchmarks import musique
from unified_kg_rag.shared import EvaluationException


def _hops(value: str) -> dict[int, int | None]:
    try:
        return musique.parse_hops(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        allow_abbrev=False,
        description=(
            "Prepare a public benchmark (MuSiQue) as a corpus directory and a "
            "run-eval dataset, or generate questions and assertions from your "
            "corpus (assertions)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "guides/benchmarks.md#run-it",
            "prepare-benchmark musique --output-directory bench/musique --limit 100",
        ),
    )
    parser.add_argument(
        "benchmark",
        choices=["musique", "assertions"],
        help=(
            "Benchmark to prepare: musique, or assertions (questions and "
            "assertions generated from --source-directory)"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for corpus/ and eval_data.json (corpus/ must be new or empty)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help=(
            "Local official JSONL (e.g. musique_ans_v1.0_dev.jsonl from the "
            "official download; a musique_full_v1.0_*.jsonl for "
            "--unanswerable); default: download the answerable dev split, or "
            "the full one with --unanswerable"
        ),
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Number of answerable questions to sample (default: all 2,417 of dev)",
    )
    parser.add_argument(
        "--unanswerable",
        type=positive_int,
        metavar="N",
        help=(
            "Also write N unanswerable questions of the full split "
            "(metadata.answerable: false), leaving out any whose answerable "
            "twin is sampled or whose removed supporting paragraph's exact "
            "text is in the pooled corpus; the counts are reported "
            "(default: none)"
        ),
    )
    parser.add_argument(
        "--hops",
        type=_hops,
        metavar="HOPS",
        help=(
            "Hop counts to sample from: a list (3,4) filters both answerable "
            "and unanswerable questions; quotas (2=50,3=50,4=100) draw exactly "
            "that many answerable questions per hop count, instead of "
            "--limit (default: every hop count, sampled as one pool)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default 0)")
    add_assertion_arguments(parser)
    add_output_format_argument(parser)
    return parser


@exits_on_sigterm
def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    as_json = json_output(args)
    if args.benchmark == "assertions":
        _assertions(args, as_json)
        return
    if args.data_path is not None and not args.data_path.is_file():
        usage_error(f"benchmark data not found: '{args.data_path}'")
    if args.hops and None not in args.hops.values() and args.limit is not None:
        usage_error("--hops quotas set the sample size; drop --limit")
    try:
        # Before the download: a used directory is a usage error.
        musique.check_output_directory(args.output_directory)
    except EvaluationException as e:
        usage_error(str(e))
    try:
        full = args.unanswerable is not None
        source = args.data_path or musique.download_dev(
            args.output_directory
            / "source"
            / (musique.FULL_DEV_FILE if full else "musique_ans_v1.0_dev.jsonl"),
            full=full,
        )
        counts = musique.prepare(
            source,
            args.output_directory,
            limit=args.limit,
            seed=args.seed,
            unanswerable=args.unanswerable,
            hops=args.hops,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark preparation interrupted by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except musique.InvalidBenchmarkDataError as e:
        usage_error(str(e))
    except (EvaluationException, OSError) as e:
        console.print(f"[red]Error: {markup_safe(e)}[/red]")
        sys.exit(EXIT_FAILURE)
    console.print(
        f"[green]MuSiQue ({musique.LICENSE}), labelled '{counts['benchmark']}': "
        f"{counts['questions']} questions, "
        f"{counts['paragraphs']} paragraphs written to "
        f"'{markup_safe(args.output_directory)}'.[/green]\n"
        f"Cite: {musique.CITATION}"
    )
    if (selection := counts.get("unanswerable_filter")) is not None:
        console.print(
            f"Unanswerable: {selection['kept']} kept of {selection['examined']} "
            f"examined ({selection['excluded_twin']} with their answerable twin "
            f"sampled, {selection['excluded_contaminated']} answerable from "
            f"the pooled corpus by exact text, {selection['excluded_no_twin']} "
            "without a twin); written as 'musique-full-unanswerable' with "
            "metadata.answerable false."
        )
    if as_json:
        print_json({"output_directory": str(args.output_directory), **counts})


def _assertions(args: argparse.Namespace, as_json: bool) -> None:
    try:
        report = run_assertion_benchmark(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark generation interrupted by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except (EvaluationException, OSError) as e:
        console.print(f"[red]Error: {markup_safe(e)}[/red]")
        sys.exit(EXIT_FAILURE)
    if as_json:
        print_json({"output_directory": str(args.output_directory), **report})


if __name__ == "__main__":
    main()
