# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark``: a benchmark as a corpus and a run-eval dataset.

Writes ``<output-directory>/corpus/`` (ingest it with ``run-ingestion``) and
``<output-directory>/eval_data.json`` (score it with ``run-eval``). The
benchmark data is read from a local copy (``--data-path``) or downloaded at
run time; it is not part of this package. ``--unanswerable N`` adds N
unanswerable questions (``metadata.answerable: false``) from the full split,
filtered so the pooled corpus does not answer them. ``prepare-benchmark
assertions`` instead generates questions and assertions from your own corpus
(``--source-directory``; ``application.cli.assertion_benchmark``), and
``prepare-benchmark structural`` a synthetic procurement world with a seed
graph and graph-structure questions for the structured strategy
(``evaluation.benchmarks.structural``; generated, never downloaded).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from unified_kg_rag import __version__
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
    setup_cli_logging,
    usage_error,
)
from unified_kg_rag.application.cli.display import console, markup_safe
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.query_args import positive_int
from unified_kg_rag.domain.models import Config
from unified_kg_rag.evaluation.benchmarks import musique, structural
from unified_kg_rag.shared import EvaluationException, ModelUnreachableError


def _hops(value: str) -> dict[int, int | None]:
    try:
        return musique.parse_hops(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _parser() -> argparse.ArgumentParser:
    """One subcommand per benchmark, each with its own flags: a flag of
    another one (``structural --limit``) is refused, not ignored."""
    parser = ArgumentParser(
        allow_abbrev=False,
        description=(
            "Prepare a public benchmark (MuSiQue) as a corpus directory and a "
            "run-eval dataset, generate questions and assertions from your "
            "corpus (assertions), or generate a synthetic structural "
            "benchmark for the structured strategy (structural)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "guides/benchmarks.md#run-it",
            "prepare-benchmark musique --output-directory bench/musique --limit 100",
        ),
    )
    # The other flags are each subcommand's, given after its name.
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the version and exit",
    )
    benchmarks = parser.add_subparsers(
        dest="benchmark",
        required=True,
        metavar="benchmark",
        parser_class=ArgumentParser,
        help=(
            "Benchmark to prepare: musique, assertions (questions and "
            "assertions generated from --source-directory), or structural (a "
            "generated procurement world: corpus/, seed/structural.jsonl, "
            "structural-config.yaml and graph-structure questions with gold "
            "answers computed over the generated graph)"
        ),
    )
    _musique_parser(benchmarks)
    assertions = _benchmark_parser(
        benchmarks,
        "assertions",
        "Generate questions and assertions from your corpus",
        seed="Sampling seed of the text units (default 0)",
    )
    add_assertion_arguments(assertions)
    structural = _benchmark_parser(
        benchmarks,
        "structural",
        "Generate a synthetic procurement world, its seed graph and "
        "graph-structure questions for the structured strategy",
        seed="The generated world's seed (default 0)",
    )
    structural.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help=(
            "The world's size (small: 8 vendors, 6 buyers, up to 4 questions "
            "a template; medium: 40 vendors, 30 buyers, up to 25 a template; "
            "large: 400 vendors, 300 buyers, so a seed-filtered count scans "
            "past search.structured.max_scan_rows; default small)"
        ),
    )
    return parser


def _benchmark_parser(
    benchmarks: Any, name: str, description: str, *, seed: str
) -> argparse.ArgumentParser:
    """The subcommand ``name`` with the flags every benchmark has."""
    parser: argparse.ArgumentParser = benchmarks.add_parser(
        name, help=description, description=description, allow_abbrev=False
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for corpus/ and eval_data.json (corpus/ must be new or empty)",
    )
    parser.add_argument("--seed", type=int, default=0, help=seed)
    add_output_format_argument(parser)
    return parser


def _musique_parser(benchmarks: Any) -> None:
    parser = _benchmark_parser(
        benchmarks,
        "musique",
        "Prepare MuSiQue (downloaded at run time, or a local copy) as a "
        "corpus directory and a run-eval dataset",
        seed="Sampling seed (default 0)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help=(
            "Local official JSONL of the answerable questions (e.g. "
            "musique_ans_v1.0_dev.jsonl from the official download; a "
            "musique_full_v1.0_*.jsonl supplies its answerable half, labelled "
            "musique-full-answerable, and with --unanswerable and no "
            "--unanswerable-data-path the unanswerable questions too); "
            "default: download the answerable dev split"
        ),
    )
    parser.add_argument(
        "--unanswerable-data-path",
        type=Path,
        help=(
            "Local musique_full_v1.0_*.jsonl to draw the --unanswerable "
            "questions from (the answerable ones still come from --data-path "
            "or the default download); default: the --data-path file when it "
            "is a full file, else download the full dev split"
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


# The subcommands, in the order _parser adds them.
BENCHMARKS = ("musique", "assertions", "structural")
# The flags of the command itself, given before a benchmark name.
_OWN_FLAGS = frozenset({"-h", "--help", "--version"})


def _flag_before_benchmark(argv: list[str]) -> str | None:
    """The first flag of a benchmark given before its name
    (``--output-format json musique``), which argparse reads as the name."""
    for arg in argv:
        if arg in BENCHMARKS:
            return None
        if arg.startswith("-"):
            return None if arg in _OWN_FLAGS else arg
    return None


@exits_on_sigterm
def main(argv: list[str] | None = None) -> None:
    given = sys.argv[1:] if argv is None else argv
    if flag := _flag_before_benchmark(given):
        usage_error(
            f"{flag}: each benchmark takes its flags after its name "
            f"(prepare-benchmark {{{','.join(BENCHMARKS)}}} {flag} ...)"
        )
    args = _parser().parse_args(argv)
    as_json = json_output(args)
    if args.benchmark != "assertions":  # which loads its configuration
        setup_cli_logging(Config(), args)
    if args.benchmark == "assertions":
        _assertions(args, as_json)
        return
    if args.benchmark == "structural":
        _structural(args, as_json)
        return
    for path in (args.data_path, args.unanswerable_data_path):
        if path is not None and not path.is_file():
            usage_error(f"benchmark data not found: '{path}'")
    if args.unanswerable_data_path is not None and args.unanswerable is None:
        usage_error("--unanswerable-data-path needs --unanswerable")
    if args.hops and None not in args.hops.values() and args.limit is not None:
        usage_error("--hops quotas set the sample size; drop --limit")
    try:
        # Before the download: a used directory is a usage error.
        musique.check_output_directory(args.output_directory)
    except EvaluationException as e:
        usage_error(str(e))
    try:
        source = args.data_path or musique.download_dev(
            args.output_directory / "source" / "musique_ans_v1.0_dev.jsonl"
        )
        unanswerable_source = args.unanswerable_data_path
        if (
            args.unanswerable is not None
            and unanswerable_source is None
            and not musique.has_unanswerable(source)
        ):
            # The answerable questions stay MuSiQue-Ans: the same sample the
            # seed draws without --unanswerable.
            unanswerable_source = musique.download_dev(
                args.output_directory / "source" / musique.FULL_DEV_FILE, full=True
            )
        counts = musique.prepare(
            source,
            args.output_directory,
            limit=args.limit,
            seed=args.seed,
            unanswerable=args.unanswerable,
            hops=args.hops,
            unanswerable_source=unanswerable_source,
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


def _structural(args: argparse.Namespace, as_json: bool) -> None:
    try:
        counts = structural.prepare(
            args.output_directory, seed=args.seed, size=args.size
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark preparation interrupted by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except EvaluationException as e:
        usage_error(str(e))
    except OSError as e:
        console.print(f"[red]Error: {markup_safe(e)}[/red]")
        sys.exit(EXIT_FAILURE)
    out = markup_safe(args.output_directory)
    console.print(
        f"[green]Structural benchmark (seed {counts['seed']}, {counts['size']}): "
        f"{counts['questions']} questions, {counts['documents']} documents and a "
        f"seed graph of {counts['nodes']} nodes and {counts['edges']} edges "
        f"written to '{out}'.[/green]\n"
        f"Ingest: run-ingestion --config-path {out}/structural-config.yaml "
        f"--source-directory {out}/corpus\n"
        f"Score: run-eval --config-path {out}/structural-config.yaml "
        f"--data-path {out}/eval_data.json "
        "--search-strategy structured,local,mix --repeats 3"
    )
    if as_json:
        print_json({"output_directory": str(args.output_directory), **counts})


def _assertions(args: argparse.Namespace, as_json: bool) -> None:
    try:
        report = run_assertion_benchmark(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark generation interrupted by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except (EvaluationException, ModelUnreachableError, OSError) as e:
        console.print(f"[red]Error: {markup_safe(e)}[/red]")
        sys.exit(EXIT_FAILURE)
    if as_json:
        print_json({"output_directory": str(args.output_directory), **report})


if __name__ == "__main__":
    main()
