# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark assertions``: the flags and the run.

Generates a question and assertion set from a corpus directory
(``evaluation.benchmarks.autoq``) with the configured evaluation and
embedding models, writes it to ``--output-directory`` and prints what the
models spent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.cli.config_args import (
    add_config_arguments,
    exit_on_config_error,
    load_cli_config,
)
from unified_kg_rag.application.cli.conventions import stoppable, usage_error
from unified_kg_rag.application.cli.display import console, markup_safe
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.application.cli.query_args import positive_int
from unified_kg_rag.domain.models import LanguageCode, LLMUsage
from unified_kg_rag.evaluation.benchmarks import autoq
from unified_kg_rag.evaluation.comparison import answer_cost_usd
from unified_kg_rag.shared import EvaluationException, setup_logging
from unified_kg_rag.shared.usage import merge_usage
from unified_kg_rag.shared.utils import event_loop


def _classes(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in autoq.QUESTION_CLASSES]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"invalid class {', '.join(unknown) or repr(value)} (choose from "
            f"{', '.join(autoq.QUESTION_CLASSES)})"
        )
    return list(dict.fromkeys(names))


def add_assertion_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags of ``prepare-benchmark assertions`` (one argument group)."""
    group = parser.add_argument_group(
        "assertions",
        "Generate questions and assertions from your corpus "
        "(prepare-benchmark assertions; see docs/guides/evaluate.md)",
    )
    group.add_argument(
        "--source-directory",
        type=Path,
        help=(
            "assertions: directory of .txt/.md files to generate from (the "
            "corpus you ingest); required for assertions"
        ),
    )
    group.add_argument(
        "--questions-per-class",
        type=positive_int,
        default=10,
        help="assertions: questions to keep per class (default 10)",
    )
    group.add_argument(
        "--classes",
        type=_classes,
        default=list(autoq.QUESTION_CLASSES),
        help=(
            "assertions: comma-separated question classes (default all: "
            f"{','.join(autoq.QUESTION_CLASSES)}); only the data_* classes get "
            "assertions"
        ),
    )
    group.add_argument(
        "--clusters",
        type=positive_int,
        default=20,
        help="assertions: k-means clusters of the text units (default 20)",
    )
    group.add_argument(
        "--units-per-cluster",
        type=positive_int,
        default=8,
        help="assertions: text units in each cluster's sample (default 8)",
    )
    group.add_argument(
        "--unit-tokens",
        type=positive_int,
        default=600,
        help="assertions: estimated tokens per text unit (default 600)",
    )
    group.add_argument(
        "--max-units",
        type=positive_int,
        default=3000,
        help="assertions: text units sampled (with --seed) and embedded (default 3000)",
    )
    group.add_argument(
        "--max-assertions",
        type=positive_int,
        default=10,
        help="assertions: assertions kept per question (default 10)",
    )
    group.add_argument(
        "--language",
        choices=[code.value for code in LanguageCode],
        help=(
            "assertions: language of the questions and assertions (default: "
            "processing.translation.source_language)"
        ),
    )
    add_config_arguments(group)
    add_provider_arguments(parser)


def _usage_lines(report: dict[str, Any], prices: dict[str, Any]) -> list[str]:
    usage = report.get("usage") or {}
    lines = []
    for step, snapshot in usage.items():
        lines.append(
            f"  {step}: {snapshot.get('llm_calls', 0)} calls, "
            f"{snapshot.get('input_tokens', 0)} input / "
            f"{snapshot.get('output_tokens', 0)} output tokens"
        )
    total = LLMUsage.model_validate(merge_usage(list(usage.values())) if usage else {})
    cost = ""
    if prices:
        usd, unpriced = answer_cost_usd(total, prices)
        cost = (
            f", about ${usd:.2f}"
            if usd is not None
            else f" (no price for {', '.join(sorted(unpriced))})"
        )
    lines.append(
        f"  total: {total.llm_calls} calls, {total.input_tokens} input / "
        f"{total.output_tokens} output tokens{cost}"
    )
    return lines


def run_assertion_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Generate, write and summarize; returns the generation report.

    Raises:
        EvaluationException: The corpus holds no text, or generation failed.
    """
    corpus = args.source_directory
    if corpus is None or not corpus.is_dir():
        usage_error(
            "prepare-benchmark assertions needs --source-directory, a directory "
            f"of .txt/.md files (got {corpus})"
        )
    try:
        autoq.check_output_directory(args.output_directory)
    except EvaluationException as e:
        usage_error(str(e))
    with exit_on_config_error():
        config = apply_provider_arguments(load_cli_config(args), args)
    setup_logging(config)
    settings = autoq.GenerationSettings(
        questions_per_class=args.questions_per_class,
        classes=args.classes,
        clusters=args.clusters,
        units_per_cluster=args.units_per_cluster,
        unit_tokens=args.unit_tokens,
        max_units=args.max_units,
        max_assertions=args.max_assertions,
        seed=args.seed,
        language=args.language or config.processing.translation.source_language.value,
    )
    generator = autoq.AssertionBenchmarkGenerator(
        config, settings, providers=build_providers(config)
    )
    generated = event_loop.run(
        stoppable(generator.agenerate(corpus)), io_workers=config.processing.io_workers
    )
    autoq.write_outputs(args.output_directory, generated)
    report = generated["report"]
    counts = ", ".join(f"{k} {v}" for k, v in report["questions"].items()) or "none"
    console.print(
        f"[green]Generated {len(generated['rows'])} questions ({markup_safe(counts)}) "
        f"with {report['assertions']} assertions from "
        f"{report['corpus']['sampled_units']} text units in "
        f"'{markup_safe(args.output_directory)}'.[/green]\n"
        f"{autoq.LICENSE_NOTE}\nModel use:"
    )
    for line in _usage_lines(report, config.evaluation.prices):
        console.print(markup_safe(line))
    if report["dropped"]:
        console.print(
            f"[yellow]Dropped: {markup_safe(report['dropped'])} (see "
            "generation_report.json)[/yellow]"
        )
    return dict(report)
