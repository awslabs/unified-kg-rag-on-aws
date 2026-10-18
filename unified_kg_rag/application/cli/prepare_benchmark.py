# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark``: a public benchmark as a corpus and a run-eval dataset.

Writes ``<output-directory>/corpus/`` (ingest it with ``run-ingestion``) and
``<output-directory>/eval_data.json`` (score it with ``run-eval``). The
benchmark data is read from a local copy (``--data-path``) or downloaded at
run time; it is not part of this package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    add_output_format_argument,
    exits_on_sigterm,
    json_output,
    print_json,
    usage_error,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.query_args import _positive_int
from unified_kg_rag.evaluation.benchmarks import musique
from unified_kg_rag.shared import EvaluationException
from unified_kg_rag.shared.utils.display import console


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Prepare a public benchmark (MuSiQue answerable) as a corpus "
            "directory and a run-eval dataset"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "benchmarks-musique",
            "prepare-benchmark musique --output-directory bench/musique --limit 100",
        ),
    )
    parser.add_argument("benchmark", choices=["musique"], help="Benchmark to prepare")
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
            "official download); default: download the answerable dev split"
        ),
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Number of questions to sample (default: all 2,417 of dev)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default 0)")
    add_output_format_argument(parser)
    return parser


@exits_on_sigterm
def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    as_json = json_output(args)
    if args.data_path is not None and not args.data_path.is_file():
        usage_error(f"benchmark data not found: '{args.data_path}'")
    try:
        # Before the download: a used directory is a usage error.
        musique.check_output_directory(args.output_directory)
    except EvaluationException as e:
        usage_error(str(e))
    try:
        source = args.data_path or musique.download_dev(
            args.output_directory / "source" / "musique_ans_v1.0_dev.jsonl"
        )
        counts = musique.prepare(
            source, args.output_directory, limit=args.limit, seed=args.seed
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark preparation interrupted by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except musique.InvalidBenchmarkDataError as e:
        usage_error(str(e))
    except (EvaluationException, OSError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(EXIT_FAILURE)
    console.print(
        f"[green]MuSiQue ({musique.LICENSE}): {counts['questions']} questions, "
        f"{counts['paragraphs']} paragraphs written to "
        f"'{args.output_directory}'.[/green]\n"
        f"Cite: {musique.CITATION}"
    )
    if as_json:
        print_json({"output_directory": str(args.output_directory), **counts})


if __name__ == "__main__":
    main()
