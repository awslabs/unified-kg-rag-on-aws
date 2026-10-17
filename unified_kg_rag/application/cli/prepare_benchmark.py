# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark``: a public benchmark as a corpus and a run-eval dataset.

Writes ``<output-dir>/corpus/`` (ingest it with ``run-ingestion``) and
``<output-dir>/eval_data.json`` (score it with ``run-eval``). The benchmark
data is read from a local copy (``--source``) or downloaded at run time; it
is not part of this package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.query_args import _positive_int
from unified_kg_rag.evaluation.benchmarks import musique
from unified_kg_rag.shared import EvaluationException
from unified_kg_rag.shared.utils.display import console


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a public benchmark (MuSiQue answerable) as a corpus "
            "directory and a run-eval dataset"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "benchmarks-musique",
            "prepare-benchmark musique --output-dir bench/musique --limit 100",
        ),
    )
    parser.add_argument("benchmark", choices=["musique"], help="Benchmark to prepare")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for corpus/ and eval_data.json (corpus/ must be new or empty)",
    )
    parser.add_argument(
        "--source",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        source = args.source or musique.download_dev(
            args.output_dir / "source" / "musique_ans_v1.0_dev.jsonl"
        )
        counts = musique.prepare(
            source, args.output_dir, limit=args.limit, seed=args.seed
        )
    except (EvaluationException, OSError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    console.print(
        f"[green]MuSiQue ({musique.LICENSE}): {counts['questions']} questions, "
        f"{counts['paragraphs']} paragraphs written to '{args.output_dir}'.[/green]\n"
        f"Cite: {musique.CITATION}"
    )


if __name__ == "__main__":
    main()
