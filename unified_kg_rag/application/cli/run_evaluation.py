# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.table import Table

from unified_kg_rag import __version__
from unified_kg_rag.adapters.aws.s3_transfer import (
    download_s3_object,
    parse_s3_uri,
    s3_client_for,
    upload_directory_to_s3,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.cli.arm_args import (
    arm_config,
    arm_spec,
    overlay_sha256,
)
from unified_kg_rag.application.cli.config_args import (
    add_config_arguments,
    exit_on_config_error,
    load_cli_config,
)
from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    ArgumentParser,
    add_output_format_argument,
    exits_on_sigterm,
    failure_fields,
    json_output,
    print_json,
    setup_cli_logging,
    stoppable,
    usage_error,
)
from unified_kg_rag.application.cli.display import (
    PRODUCT_NAME,
    console,
    display_ascii_art,
    markup_safe,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.lazy import lazy_names
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    missing_extras_error,
    strategy_roles,
)
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.application.cli.query_args import (
    add_query_arguments,
    positive_int,
    query_suffix,
)
from unified_kg_rag.application.config import load_cli_dotenv
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    EvaluatorType,
    LLMUsage,
    ModelRole,
    SearchStrategy,
    model_id_value,
)
from unified_kg_rag.domain.models.config import REMOVED_EVALUATORS
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    DataProcessingError,
    EvaluationException,
    IndexNotFoundError,
    MethodologyNotIndexedError,
    get_logger,
    observability,
)
from unified_kg_rag.shared.usage import usage_scope
from unified_kg_rag.shared.utils import event_loop
from unified_kg_rag.shared.utils.s3_uri import is_s3_uri

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        extraction_cache_scope,
    )
    from unified_kg_rag.evaluation.answer_match_evaluator import answer_scoring
    from unified_kg_rag.evaluation.arms import (
        arm_label,
        arm_pairs,
        is_arm_label,
        label_strategy,
    )
    from unified_kg_rag.evaluation.closed_book import (
        CLOSED_BOOK_ARM,
        closed_book_arms,
        closed_book_chain,
    )
    from unified_kg_rag.evaluation.comparison import (
        adjust_for_multiplicity,
        baseline_comparison,
        paired_comparison,
        per_query_rows,
    )
    from unified_kg_rag.evaluation.evaluation_manager import (
        AnswerArm,
        EvaluationManager,
    )
    from unified_kg_rag.evaluation.gates import Gate, gate_breaches
    from unified_kg_rag.evaluation.pairwise import PairwiseJudge
    from unified_kg_rag.evaluation.report import (
        baseline_metric_keys,
        compared_pairs,
        comparison_csv,
        comparison_markdown,
        delta_cell,
    )
    from unified_kg_rag.evaluation.saved_results import (
        SavedRun,
        check_answer_scoring,
        citation_warnings,
        dataset_warnings,
        group_runs,
        interleaving_warnings,
        load_saved_runs,
        pooling_errors,
        pooling_warnings,
        rerank_unmetered,
        strategy_order,
        summary_warnings,
    )


def _chain_module() -> ModuleType:
    from unified_kg_rag.application.retrieval import rag_chain

    return rag_chain


def _manager_module() -> ModuleType:
    from unified_kg_rag.evaluation import evaluation_manager

    return evaluation_manager


def _answer_match_module() -> ModuleType:
    from unified_kg_rag.evaluation import answer_match_evaluator

    return answer_match_evaluator


def _closed_book_module() -> ModuleType:
    from unified_kg_rag.evaluation import closed_book

    return closed_book


def _comparison_module() -> ModuleType:
    from unified_kg_rag.evaluation import comparison

    return comparison


def _gates_module() -> ModuleType:
    from unified_kg_rag.evaluation import gates

    return gates


def _pairwise_module() -> ModuleType:
    from unified_kg_rag.evaluation import pairwise

    return pairwise


def _report_module() -> ModuleType:
    from unified_kg_rag.evaluation import report

    return report


def _arms_module() -> ModuleType:
    from unified_kg_rag.evaluation import arms

    return arms


def _saved_results_module() -> ModuleType:
    from unified_kg_rag.evaluation import saved_results

    return saved_results


# Imported on first use: the chain and the evaluation manager load LangChain
# and every search strategy (seconds), which --help and a refused command
# line never need. Tests patch them as attributes of this module.
__getattr__, _load_evaluation = lazy_names(
    globals(),
    {
        "GraphRAGChain": _chain_module,
        "extraction_cache_scope": _chain_module,
        "EvaluationManager": _manager_module,
        "AnswerArm": _manager_module,
        "answer_scoring": _answer_match_module,
        "CLOSED_BOOK_ARM": _closed_book_module,
        "closed_book_arms": _closed_book_module,
        "closed_book_chain": _closed_book_module,
        "adjust_for_multiplicity": _comparison_module,
        "baseline_comparison": _comparison_module,
        "paired_comparison": _comparison_module,
        "per_query_rows": _comparison_module,
        "gate_breaches": _gates_module,
        "PairwiseJudge": _pairwise_module,
        "baseline_metric_keys": _report_module,
        "compared_pairs": _report_module,
        "comparison_csv": _report_module,
        "comparison_markdown": _report_module,
        "delta_cell": _report_module,
        "SavedRun": _saved_results_module,
        "strategy_order": _saved_results_module,
        "arm_label": _arms_module,
        "arm_pairs": _arms_module,
        "is_arm_label": _arms_module,
        "label_strategy": _arms_module,
        "check_answer_scoring": _saved_results_module,
        "citation_warnings": _saved_results_module,
        "dataset_warnings": _saved_results_module,
        "group_runs": _saved_results_module,
        "interleaving_warnings": _saved_results_module,
        "load_saved_runs": _saved_results_module,
        "pooling_errors": _saved_results_module,
        "pooling_warnings": _saved_results_module,
        "rerank_unmetered": _saved_results_module,
        "summary_warnings": _saved_results_module,
    },
)

# One run: the reports and results of one strategy and repeat
# (``comparison.Run``).
Run = tuple[list[EvaluationReport], list[EvaluationResult]]

logger = get_logger(__name__)

_STRATEGY_VALUES = frozenset(s.value for s in SearchStrategy)


class InvalidDatasetError(DataProcessingError):
    """The evaluation dataset cannot be read or fails validation (exit 2)."""


def _failure_rate(value: str) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return rate


def _data_path(value: str) -> str | Path:
    """An ``s3://`` URI stays a string (``Path`` would collapse its ``//``)."""
    return value if is_s3_uri(value) else Path(value)


def _gate(value: str) -> Gate:
    from unified_kg_rag.evaluation.gates import parse_gate

    try:
        return parse_gate(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _primary_metric(value: str) -> str:
    evaluator, _, metric = value.strip().partition("/")
    if not evaluator or not metric or "/" in metric:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not <evaluator>/<metric> (e.g. answer_match/token_f1)"
        )
    return value.strip()


def _model_id(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value.strip()


def with_judge_model(config: Config, model_id: str | None) -> Config:
    """``config`` with ``models.roles.evaluation`` set to ``model_id``."""
    if not model_id:
        return config
    models = config.models
    roles = models.roles.model_copy(update={"evaluation": model_id})
    return config.model_copy(
        update={"models": models.model_copy(update={"roles": roles})}
    )


def _evaluator_list(value: str) -> list[EvaluatorType]:
    """A comma-separated list of distinct evaluators (``answer_match,ragas``)."""
    names = [name.strip() for name in value.split(",")]
    valid = [t.value for t in EvaluatorType if t.value not in REMOVED_EVALUATORS]
    removed = [name for name in names if name in REMOVED_EVALUATORS]
    if removed:
        raise argparse.ArgumentTypeError(
            f"the {removed[0]} evaluator is removed: {REMOVED_EVALUATORS[removed[0]]}"
        )
    unknown = [name for name in names if name and name not in valid]
    if "" in names or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid evaluator {', '.join(unknown) or repr(value)} "
            f"(choose from {', '.join(valid)})"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"evaluator listed twice in '{value}'")
    return [EvaluatorType(name) for name in names]


def with_evaluators(config: Config, evaluators: list[EvaluatorType] | None) -> Config:
    """``config`` with ``evaluation.enabled_evaluators`` set to ``evaluators``."""
    if evaluators is None:
        return config
    evaluation = config.evaluation.model_copy(
        update={"enabled_evaluators": list(evaluators)}
    )
    return config.model_copy(update={"evaluation": evaluation})


def with_pairwise_mode(config: Config, mode: str | None) -> Config:
    """``config`` with ``evaluation.pairwise_mode`` set to ``mode``."""
    if not mode:
        return config
    evaluation = config.evaluation.model_copy(update={"pairwise_mode": mode})
    return config.model_copy(update={"evaluation": evaluation})


# The evaluators that call the LLM judge (models.roles.evaluation).
_JUDGE_EVALUATORS = frozenset({EvaluatorType.RAGAS, EvaluatorType.ASSERTION})


def failure_budget_breaches(
    summary: EvaluationSummary, max_failure_rate: float
) -> list[str]:
    """Why the run must exit non-zero; empty when it is within budget.

    The budget applies to answer generation (failed / total queries) and to
    every metric (failed / attempted, where attempted = scored + failed;
    skipped metrics were not applicable and do not count). A share of 1.0,
    or an empty run, always breaches it.
    """
    breaches = []
    if _over(summary.failed_evaluations, summary.total_queries, max_failure_rate):
        breaches.append(
            f"{summary.failed_evaluations}/{summary.total_queries} queries failed"
        )
    for evaluator_name, metrics in summary.metric_outcomes.items():
        for metric_name, counts in metrics.items():
            failed = counts.get("failed", 0)
            attempted = counts.get("scored", 0) + failed
            if failed and _over(failed, attempted, max_failure_rate):
                breaches.append(
                    f"{evaluator_name}/{metric_name}: {failed}/{attempted} failed"
                )
    return breaches


def pairwise_budget_breaches(
    pairwise: dict[str, dict[str, Any]], max_failure_rate: float
) -> list[str]:
    """The failure budget applied to the pairwise judge: per strategy, the
    pairs it failed to judge over the pairs it was asked to judge."""
    breaches = []
    for strategy, judged in pairwise.items():
        failed = judged["failed"]
        attempted = judged["judged"] + failed
        if failed and _over(failed, attempted, max_failure_rate):
            breaches.append(f"pairwise judge {strategy}: {failed}/{attempted} failed")
    return breaches


def pairwise_warnings(
    pairwise: dict[str, dict[str, Any]], manifest: dict[str, Any] | None
) -> list[str]:
    """Warnings on the pairwise judge's reliability: it is the answer model
    (self-preference), or its verdicts flip with the answer order."""
    warnings = []
    if ((manifest or {}).get("models") or {}).get("judge_is_answer_model"):
        warnings.append(
            "The pairwise judge is the answer model (models.roles.evaluation "
            "names it, or another provider than Bedrock has no other default "
            "judge); LLM judges favour their own answers, so set it to "
            "another model to compare strategies."
        )
    for strategy, judged in pairwise.items():
        consistency = judged.get("position_consistency")
        if consistency is not None and consistency < 0.5:
            warnings.append(
                f"Pairwise judge {strategy}: the two answer orders agreed on "
                f"{consistency:.0%} of the verdicts, so its ties and win rate "
                "mostly reflect position bias."
            )
    for warning in warnings:
        logger.warning("%s", warning)
    return warnings


def _over(failed: int, total: int, max_failure_rate: float) -> bool:
    return total == 0 or failed >= total or failed / total > max_failure_rate


def _benchmark_names(labels: dict[str, int]) -> str:
    """Which benchmark was scored (musique-full-answerable is not comparable
    to MuSiQue-Ans), with its query count."""
    return ", ".join(f"{name} ({count})" for name, count in labels.items())


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = ArgumentParser(
            allow_abbrev=False,
            description=f"{PRODUCT_NAME}: evaluate retrieval and answers on a "
            "dataset, and compare strategies",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=doc_epilog(
                "guides/evaluate.md#examples",
                "run-eval --data-path eval_data.json --config-path config.yaml",
            ),
        )

        # One input: a dataset to answer, or saved answers to compare.
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--data-path",
            type=_data_path,
            help=(
                "Path to the unified evaluation data file (JSON format), containing "
                "both questions and ground truths. Accepts a local path or an "
                "s3://bucket/key URI (downloaded before the run). This or "
                "--from-results is required."
            ),
        )
        source.add_argument(
            "--from-results",
            type=Path,
            nargs="+",
            metavar="PATH",
            help=(
                "Rebuild the comparison from saved runs instead of answering: "
                "result directories (read with their run subdirectories) or "
                "evaluation_results_*.json files (with their reports and "
                "summary next to them). A strategy's runs from several run-eval "
                "invocations are refused without --pool-runs. No answer is "
                "generated and no store is queried; --pairwise-judge re-judges "
                "the saved answers (with --judge-model, another model). The "
                "strategies compare in --search-strategy order when given, else "
                "in the order the oldest run listed them. Local paths only."
            ),
        )
        parser.add_argument(
            "--rescore",
            nargs="?",
            const="",
            metavar="DATASET",
            help=(
                "With --from-results, score the saved answers again with this "
                "run's evaluators (--evaluators, evaluation.*: answer_match's "
                "span and normalization, the citation metrics, the LLM judges, "
                "which are called again) instead of comparing the scores they "
                "hold, and save the re-scored runs in this run's directory. "
                "The gold answers are those saved with each answer, or "
                "DATASET's (a run-eval dataset whose items pair by query id: "
                "their answer, aliases, metadata and expectations replace the "
                "saved ones). No answer is generated and no store is queried"
            ),
        )
        parser.add_argument(
            "--baseline-results",
            type=Path,
            nargs="+",
            metavar="PATH",
            help=(
                "Compare against an earlier run: its saved results (directories "
                "or evaluation_results_*.json files) become the baselines, as "
                "'baseline:<strategy>', and each strategy of this run, or of "
                "--from-results, is compared with the baseline of the same "
                "strategy (with the earlier run's first strategy, and a "
                "warning, when it has none). Refused when the runs scored "
                "exact_match and token_f1 with different answer_scoring "
                "settings"
            ),
        )
        parser.add_argument(
            "--pool-runs",
            action="store_true",
            help=(
                "Pool the saved runs of one strategy as its repeats although "
                "they come from several run-eval invocations, or ran with "
                "different settings (a warning names them). Without it "
                "--from-results and --baseline-results refuse such runs "
                "(exit 2): list the result files, or the run subdirectory, of "
                "one invocation"
            ),
        )
        parser.add_argument(
            "--fail-if",
            type=_gate,
            action="append",
            metavar="MEASURE:STAT<OP>VALUE",
            help=(
                "Regression gate, repeatable: exit 1 when a compared strategy's "
                "paired delta against the baseline meets the condition, or has "
                "no delta of MEASURE. MEASURE is an <evaluator>/<metric> of "
                "deltas_vs_baseline (abstention/recall and abstention/precision "
                "with unanswerable items) or a cost measure (usage/llm_calls, "
                "usage/input_tokens, usage/output_tokens, usage/total_tokens, "
                "usage/cost_usd, latency/response_time_seconds); STAT is "
                "mean_delta, ci_low or ci_high, or adjusted for the many "
                "deltas read (--primary-metric): adjusted_ci_low, "
                "adjusted_ci_high, p_value or adjusted_p_value; OP is <, <=, "
                "> or >=. Example: answer_match/token_f1:adjusted_ci_high<-0.02. "
                "Needs a comparison (two strategies or --baseline-results)"
            ),
        )
        parser.add_argument(
            "--judge-model",
            type=_model_id,
            metavar="MODEL_ID",
            help=(
                "Model of the LLM judges for this run (models.roles.evaluation): "
                "the pairwise judge and the ragas/assertion evaluators"
            ),
        )
        parser.add_argument(
            "--output-directory",
            type=str,
            help=(
                "Directory to save evaluation results (default: "
                "evaluation.outputs_directory), in a subdirectory of its own "
                "per run-eval invocation, run_<timestamp>_<id> "
                "(evaluation.run_subdirectories). Accepts a local path or an "
                "s3://bucket/prefix/ URI (results are uploaded after the run; "
                "a failed upload exits non-zero)."
            ),
        )
        add_query_arguments(parser, multiple_strategies=True)
        parser.add_argument(
            "--evaluators",
            type=_evaluator_list,
            metavar="NAME[,NAME...]",
            help=(
                "Evaluators that score this run, comma-separated (default: "
                "evaluation.enabled_evaluators, answer_match,retrieval,"
                "graph_aware). The LLM judges are opt-in: ragas (the eval "
                "extra; several judge calls per answer, one per context for "
                "context_precision) and assertion (evaluation.assertion_trials "
                "calls per answer)"
            ),
        )
        parser.add_argument(
            "--repeats",
            type=positive_int,
            default=1,
            help=(
                "Answer and score the dataset this many times per strategy; the "
                "comparison averages each query over the repeats and reports "
                "the run-to-run spread (default 1)"
            ),
        )
        parser.add_argument(
            "--primary-metric",
            type=_primary_metric,
            metavar="EVALUATOR/METRIC",
            help=(
                "The comparison's primary metric, declared in advance "
                "(default: evaluation.primary_metric, else "
                "answer_match/answer_contains when scored): adjusted only "
                "over the compared strategies, while every other metric's "
                "delta is Holm-adjusted over all of them (adjusted_p_value, "
                "adjusted_ci_low/high, which --fail-if can read)"
            ),
        )
        parser.add_argument(
            "--run-order",
            choices=["interleaved", "sequential"],
            help=(
                "How the runs of a comparison are answered (default: "
                "evaluation.run_order, 'interleaved'): every query by every "
                "run before the next query, the runs' order alternating "
                "(ABBA), so latency deltas do not measure time; or "
                "'sequential', one run after another"
            ),
        )
        parser.add_argument(
            "--repeat-extractions",
            choices=["resample", "reuse"],
            help=(
                "With --repeats, whether each repeat extracts the query's "
                "entities and keywords again (default: "
                "evaluation.repeat_extractions, 'resample': repeat_stdev "
                "includes the extraction's variance, at one set of fast-model "
                "extraction calls per query and repeat) or reuses the first "
                "repeat's ('reuse')"
            ),
        )
        parser.add_argument(
            "--pairwise-judge",
            action="store_true",
            help=(
                "With two or more strategies, also judge each strategy's answers "
                "against the first's with an LLM (comprehensiveness, diversity, "
                "empowerment, directness; both orders) using "
                "models.roles.evaluation"
            ),
        )
        parser.add_argument(
            "--pairwise-mode",
            choices=["standard", "differential"],
            help=(
                "How --pairwise-judge reads the two answers (default: "
                "evaluation.pairwise_mode, 'standard'). 'differential' first "
                "splits them into shared and unique statements with "
                "models.roles.pairwise_overlap (fast tier, one call per pair) "
                "and judges the unique statements only, so shared content and "
                "the length it adds earn neither answer credit. Both modes "
                "report win rates per answer-length bucket"
            ),
        )
        parser.add_argument(
            "--pairwise-repeats",
            choices=["first", "all"],
            help=(
                "Which repeats --pairwise-judge judges (default: "
                "evaluation.pairwise_repeats, 'first'): the first repeat's "
                "answers, or 'all', repeat r against repeat r (the judge's "
                "calls times --repeats; each query's repeats averaged first)"
            ),
        )
        parser.add_argument(
            "--closed-book",
            action="store_true",
            help=(
                "Also answer every question with the answer model alone, no "
                "retrieval (the 'closed_book' arm, compared with the first "
                "strategy like another strategy and judged by --pairwise-judge): "
                "its scores bound what the model knows without the corpus. One "
                "answer call per question and repeat; its retrieval and context "
                "metrics score an empty source list, and --fail-if skips it"
            ),
        )
        parser.add_argument(
            "--arm",
            type=arm_spec,
            action="append",
            metavar="NAME=OVERLAY[,suffix=SUFFIX]",
            help=(
                "A configuration arm, repeatable: answer the dataset again with "
                "the config under OVERLAY (a YAML mapping merged over it, as "
                "a config file is validated; its evaluation section is the "
                "run's), as a chain of its own, with each --search-strategy. "
                "Its runs are labelled <strategy>@NAME and compared with the "
                "same strategy's base run, interleaved with it (--run-order) "
                "and Holm-adjusted with the other deltas; the run manifest "
                "records the overlay's path and sha256. suffix= queries "
                "another index namespace with the arm. An empty overlay is an "
                "A/A arm (the noise floor)"
            ),
        )
        parser.add_argument(
            "--max-failure-rate",
            type=_failure_rate,
            default=1.0,
            help=(
                "Exit non-zero when the fraction of queries whose answer generation "
                "failed, or the fraction of attempted values of any metric that "
                "failed, exceeds this value (0.0-1.0). Skipped metrics do not count. "
                "Default 1.0 = fail only when all queries, or all attempts of a "
                "metric, failed."
            ),
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="(deprecated: use --log-level DEBUG, which it sets) Log at "
            "DEBUG. --verbose means more of a result's detail in run-rag; the "
            "log level of every CLI is --log-level",
        )
        add_config_arguments(parser)
        add_output_format_argument(parser)
        add_provider_arguments(parser)
        return parser

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


# The widest a metric column is kept at; a longer name folds.
_METRIC_COLUMN_MAX = 32


def _widest(names: Iterable[str]) -> int:
    """The width a table's metric column is kept at: its longest name, up
    to ``_METRIC_COLUMN_MAX`` (longer ones fold onto the next line), so a
    metric name is never cut to fit the terminal."""
    return min(max((len(name) for name in names), default=6), _METRIC_COLUMN_MAX)


class EvaluationRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        rag_chain: Runnable[Any, Any] | None,
        config: Config,
        providers: Providers | None = None,
        *,
        chain_factory: Callable[[Config], Runnable[Any, Any]] | None = None,
    ) -> None:
        """``config`` is the run's configuration, provider flags applied (the
        one ``rag_chain`` was built from). ``rag_chain`` is ``None`` with
        ``--from-results`` (nothing is answered); ``providers`` builds the
        pairwise judge then (default: from ``config``). ``chain_factory``
        builds the chain of each ``--arm`` from its configuration (default:
        ``GraphRAGChain``); the runner closes them."""
        _load_evaluation()
        self.args = args
        self.chain_factory = chain_factory
        self._arm_chains: list[Any] = []
        self.config = with_pairwise_mode(
            with_judge_model(
                with_evaluators(config, getattr(args, "evaluators", None)),
                getattr(args, "judge_model", None),
            ),
            getattr(args, "pairwise_mode", None),
        )
        self.rag_chain = rag_chain
        self.providers = providers
        # This invocation, in every run manifest it writes: the runs a
        # comparison reads back tell the invocations apart by it.
        self.run_id = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        self.json_output = json_output(args)
        self.evaluation_manager: EvaluationManager | None = None
        self._validate_args()

    def _output_directory(self) -> str:
        return str(
            self.args.output_directory or self.config.evaluation.outputs_directory
        )

    def _run_target(self) -> str:
        """Where this invocation writes its files: a subdirectory of the
        outputs directory named by ``run_id`` (``evaluation.run_subdirectories``),
        else the outputs directory itself."""
        target = self._output_directory()
        if not self.config.evaluation.run_subdirectories:
            return target
        if is_s3_uri(target):
            return f"{target.rstrip('/')}/{self.run_id}/"
        return str(Path(target) / self.run_id)

    @property
    def _from_results(self) -> list[Path] | None:
        return getattr(self.args, "from_results", None)

    @property
    def _arms(self) -> list[dict[str, Any]]:
        """The ``--arm`` configuration arms."""
        return list(getattr(self.args, "arm", None) or [])

    @property
    def _rescoring(self) -> bool:
        """Whether --rescore scores the saved answers of --from-results again."""
        return (
            self._from_results is not None
            and getattr(self.args, "rescore", None) is not None
        )

    @property
    def _baseline_results(self) -> list[Path] | None:
        return getattr(self.args, "baseline_results", None)

    @property
    def _gates(self) -> list[Gate]:
        return list(getattr(self.args, "fail_if", None) or [])

    def _strategies_given(self) -> bool:
        """Whether --search-strategy was given (not its default list)."""
        default = CommandLineInterface().parser.get_default("search_strategy")
        return list(self.args.search_strategy) != list(default)

    def _judge_model_warnings(self) -> list[str]:
        """A warning when --judge-model judges nothing: no --pairwise-judge,
        and no LLM-judge evaluator scores this run (none re-scores saved
        runs)."""
        if not getattr(self.args, "judge_model", None) or getattr(
            self.args, "pairwise_judge", False
        ):
            return []
        if (self._from_results is None or self._rescoring) and _JUDGE_EVALUATORS & set(
            self.config.evaluation.enabled_evaluators
        ):
            return []
        return [
            "--judge-model is given without --pairwise-judge and no LLM-judge "
            "evaluator (ragas, assertion) scores this run, so it "
            "judges nothing."
        ]

    def _output_is_baseline(self) -> bool:
        """Whether this run saves its results where --baseline-results reads
        the baseline from (the next run reading it pools them in). In a run
        subdirectory of its own it does not."""
        target = self._output_directory()
        if (
            self._from_results is not None
            or is_s3_uri(target)
            or self.config.evaluation.run_subdirectories
        ):
            return False
        out = Path(target).resolve()
        return any(
            (path if path.is_dir() else path.parent).resolve() == out
            for path in self._baseline_results or []
        )

    def _uses_llm_judge(self) -> bool:
        """Whether this run calls an LLM judge: --pairwise-judge, or a
        ragas/assertion evaluator scoring answers it generates."""
        if getattr(self.args, "pairwise_judge", False):
            return True
        return (self._from_results is None or self._rescoring) and bool(
            _JUDGE_EVALUATORS & set(self.config.evaluation.enabled_evaluators)
        )

    def _judge_choice(self) -> tuple[str, str]:
        """The LLM judge and why, as ``Config.judge_model_choice`` picks it,
        from the models of every answer it judges: those of the saved runs
        it reads (``--from-results``, ``--baseline-results``) too. A default
        judge that wrote them falls back as for this run's answer model and
        is pinned, so every judge this run builds uses it."""
        saved = self._saved_answer_models()
        if not saved:
            return self.config.judge_model_choice()
        answers = list(saved)
        if self._from_results is None:  # this run answers too
            answers.append(self.config.model_id(ModelRole.ANSWER_GENERATION))
        judge, reason = self.config.judge_model_choice(answers)
        if self.config.models.roles.evaluation is None:
            self.config = with_judge_model(self.config, judge)
        return judge, reason

    def _saved_answer_models(self) -> list[str]:
        """The answer models the saved runs this run reads recorded."""
        paths = [*(self._from_results or []), *(self._baseline_results or [])]
        if not paths:
            return []
        try:
            saved = load_saved_runs(paths)
        except EvaluationException:
            return []  # reported where the runs are read
        return sorted(
            {
                str(model)
                for run in saved
                if (
                    model := ((run.manifest or {}).get("models") or {}).get(
                        "answer_generation"
                    )
                )
            }
        )

    def _validate_args(self) -> None:
        if (
            getattr(self.args, "rescore", None) is not None
            and self._from_results is None
        ):
            usage_error("--rescore scores saved answers again: give --from-results")
        for warning in self._judge_model_warnings():
            logger.warning("%s", warning)
        if self._uses_llm_judge():
            judge, reason = self._judge_choice()
            logger.info("LLM judge: %s (%s)", judge, reason)
        names = [arm["name"] for arm in self._arms]
        if len(set(names)) != len(names):
            usage_error(f"--arm names must differ (got {', '.join(names)})")
        if self._from_results is not None:
            if self._arms:
                usage_error(
                    "--arm answers the questions, --from-results answers none "
                    "(the saved runs' arms are compared when they hold some)"
                )
            if getattr(self.args, "closed_book", False):
                usage_error(
                    "--closed-book answers the questions, --from-results answers "
                    "none (the saved runs' closed_book arm is compared when "
                    "they hold one)"
                )
            if self.args.repeats != 1:
                usage_error(
                    "--repeats does not apply to --from-results (the saved runs "
                    "are the repeats)"
                )
            for flag in ("run_order", "repeat_extractions"):
                if getattr(self.args, flag, None) is not None:
                    usage_error(
                        f"--{flag.replace('_', '-')} does not apply to "
                        "--from-results (no answer is generated)"
                    )
            if getattr(self.args, "evaluators", None) is not None and not (
                self._rescoring
            ):
                usage_error(
                    "--evaluators does not apply to --from-results (the saved "
                    "runs are compared on the scores they hold; --rescore "
                    "scores their answers again)"
                )
            dataset = getattr(self.args, "rescore", None)
            if dataset and not Path(dataset).is_file():
                raise InvalidDatasetError(f"--rescore dataset '{dataset}' not found")
            try:
                if is_s3_uri(self._output_directory()):
                    parse_s3_uri(self._output_directory(), require_key=False)
            except ConfigurationError as e:
                usage_error(str(e))
            return
        if (
            self._gates
            and len(self.args.search_strategy) < 2
            and not self._baseline_results
            and not self._arms
        ):
            usage_error(
                "--fail-if needs a comparison: list two or more strategies in "
                "--search-strategy, or give --baseline-results"
            )
        eval_data_path = self.args.data_path
        # Malformed S3 URIs fail here, before any model or store is touched.
        try:
            if is_s3_uri(eval_data_path):
                parse_s3_uri(eval_data_path, require_key=True)
            if is_s3_uri(self._output_directory()):
                parse_s3_uri(self._output_directory(), require_key=False)
        except ConfigurationError as e:
            usage_error(str(e))
        if is_s3_uri(eval_data_path):
            return
        if eval_data_path and not Path(eval_data_path).exists():
            # The same input error as a dataset that fails validation (exit 2).
            raise InvalidDatasetError(
                f"evaluation dataset '{eval_data_path}' not found"
            )

    @staticmethod
    def _print_summary(summary: EvaluationSummary, total_time: float) -> None:
        summary_text = (
            f"Total Queries: [bold]{summary.total_queries}[/bold]\n"
            f"Successful: [green]{summary.successful_evaluations}[/green]\n"
            f"Failed: [red]{summary.failed_evaluations}[/red]\n"
        )
        if labels := summary.benchmarks:
            summary_text += (
                f"Benchmark: [bold]{markup_safe(_benchmark_names(labels))}[/bold]\n"
            )
        if blocked := summary.guardrail_blocked_counts:
            # Scored on the blocked message: a low score may be the guardrail.
            per_text = markup_safe(
                ", ".join(f"{text} {count}" for text, count in blocked.items())
            )
            summary_text += (
                f"Guardrail blocked: [yellow]{sum(blocked.values())}[/yellow]"
                f" ({per_text})\n"
            )

        if structured := summary.structured_statistics:
            reasons = markup_safe(
                ", ".join(
                    f"{reason} {count}"
                    for reason, count in structured["fallback_reasons"].items()
                )
                or "none"
            )
            status = structured["status"]
            summary_text += (
                f"Structured: answered {status.get('answered', 0)} of "
                f"{structured['queries']}, valid plans "
                f"{structured['valid_plan_rate']:.0%} (fallback reasons: "
                f"{reasons})\n"
            )

        if iterative := summary.iterative_statistics:
            rounds = markup_safe(
                ", ".join(f"{n} {count}" for n, count in iterative["rounds"].items())
            )
            stops = markup_safe(
                ", ".join(
                    f"{reason} {count}"
                    for reason, count in iterative["stop_reasons"].items()
                )
            )
            summary_text += (
                f"Iterative: {iterative['queries']} answers, rounds {rounds} "
                f"(mean {iterative['mean_rounds']:.2f}), stopped by {stops}; "
                f"{iterative['max_rounds_rate']:.0%} at max_rounds\n"
            )

        if summary.total_queries > 0:
            success_rate = (
                summary.successful_evaluations / summary.total_queries
            ) * 100
            summary_text += f"Success Rate: [bold]{success_rate:.1f}%[/bold]\n"

        if summary.average_response_time:
            summary_text += f"Avg. Response Time: [cyan]{summary.average_response_time:.3f}s[/cyan]\n"

        summary_text += f"Total Evaluation Time: [cyan]{total_time:.3f}s[/cyan]"

        console.print(
            Panel(
                summary_text,
                title="[bold blue]Evaluation Summary[/bold blue]",
                border_style="blue",
            )
        )

        if summary.metric_statistics:
            table = Table(
                title="[bold]Metric Statistics[/bold]",
                show_header=True,
                header_style="bold magenta",
            )
            # Never cut ("answer_match/answer_contai…"): a long name folds.
            table.add_column(
                "Metric",
                min_width=_widest(summary.metric_statistics),
                overflow="fold",
            )
            table.add_column("Mean", style="green")
            table.add_column("Median", style="green")
            table.add_column("StdDev", style="cyan")
            table.add_column("Min", style="yellow")
            table.add_column("Max", style="yellow")
            table.add_column("Count", style="dim")

            for metric, stats in summary.metric_statistics.items():
                table.add_row(
                    markup_safe(metric),
                    f"{stats['mean']:.3f}",
                    f"{stats['median']:.3f}",
                    f"{stats['stdev']:.3f}",
                    f"{stats['min']:.3f}",
                    f"{stats['max']:.3f}",
                    str(int(stats["count"])),
                )
            console.print(table)

        # Failed/skipped metric values are excluded from the means above; say so
        # explicitly so a mean over a subset is not read as a mean over all.
        for evaluator_name, metrics in summary.metric_outcomes.items():
            for metric_name, counts in metrics.items():
                if counts.get("failed") or counts.get("skipped"):
                    console.print(
                        f"[yellow]{markup_safe(evaluator_name)}/"
                        f"{markup_safe(metric_name)}: "
                        f"{counts.get('scored', 0)} scored, "
                        f"{counts.get('failed', 0)} failed, "
                        f"{counts.get('skipped', 0)} skipped "
                        "(failed/skipped excluded from statistics)[/yellow]"
                    )

    @staticmethod
    def _save_comparison(comparison: dict[str, Any], outputs_dir: Path) -> Path:
        """Write the comparison next to the runs' result files."""
        outputs_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        path = outputs_dir / f"evaluation_comparison_{stamp}.json"
        path.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _print_comparison(comparison: dict[str, Any], total_time: float) -> None:
        """Mean per strategy and the paired delta (95% CI) against each
        strategy's baseline."""
        baseline = comparison["baseline"]
        strategies = comparison["strategies"]
        pairs = compared_pairs(comparison)
        title = (
            f"vs '{baseline}'"
            if len(set(pairs.values())) == 1
            else "vs each strategy's baseline"
        )
        table = Table(
            title=f"[bold]Paired comparison {title} ({total_time:.0f}s)[/bold]",
            show_header=True,
            header_style="bold magenta",
        )
        keys = baseline_metric_keys(comparison)
        # Never cut ("answer_match/answer_contai…"): a long name or value
        # folds onto the next line instead.
        table.add_column("Metric", min_width=_widest(keys), overflow="fold")
        closed = set(comparison.get("closed_book_arms") or ())
        for strategy in strategies:
            marked = f"{strategy} (no retrieval)" if strategy in closed else strategy
            table.add_column(markup_safe(marked), style="green", overflow="fold")
        for strategy, label in pairs.items():
            table.add_column(
                f"{strategy} - {label} [CI]", style="cyan", overflow="fold"
            )
        primary = (comparison.get("multiplicity") or {}).get("primary_metric")
        for key in keys:
            row = [markup_safe(f"{key} (primary)" if key == primary else key)]
            for strategy in strategies:
                stats = strategies[strategy]["metrics"].get(key)
                row.append(f"{stats['mean']:.3f}" if stats else "-")
            for strategy in pairs:
                row.append(
                    delta_cell(comparison["deltas_vs_baseline"][strategy].get(key))
                )
            table.add_row(*row)
        console.print(table)
        dataset = (comparison.get("run_manifest") or {}).get("dataset") or {}
        if labels := dataset.get("benchmarks"):
            console.print(
                f"Benchmark: [bold]{markup_safe(_benchmark_names(labels))}[/bold]"
            )
        for strategy, stats in strategies.items():
            routed = stats["routed_strategies"]
            if routed and list(routed) != [strategy]:
                console.print(f"[dim]{strategy} routed to: {routed}[/dim]")
        for warning in comparison.get("warnings") or []:
            console.print(f"[yellow]Warning: {markup_safe(warning)}[/yellow]")
        for strategy, judged in (comparison.get("pairwise_vs_baseline") or {}).items():
            rates = ", ".join(
                f"{criterion} {stats['candidate_win_rate']:.2f}"
                for criterion, stats in judged["criteria"].items()
                if stats["candidate_win_rate"] is not None
            )
            errors = judged["candidate_errors"] + judged["baseline_errors"]
            consistency = judged.get("position_consistency")
            agreed = (
                f", orders agreed {consistency:.0%}" if consistency is not None else ""
            )
            console.print(
                f"[cyan]Pairwise judge ({judged.get('mode', 'standard')} mode), "
                f"{strategy} win rate vs "
                f"{pairs.get(strategy, baseline)} "
                f"({judged['judged']} judged, {errors} decided by a failed "
                f"answer, {judged['failed']} judge failures{agreed}): "
                f"{rates or 'none'}[/cyan]"
            )

    async def run(self) -> int:
        """Evaluate; 0 on success, 1 on a breached failure budget or failed upload.

        An ``s3://`` dataset is downloaded to a temporary directory first; an
        ``s3://`` outputs directory is written locally and then uploaded. A
        run stopped part-way (SIGTERM, Ctrl-C, a crash) still uploads the
        results of the runs that finished, or keeps them locally and logs
        where.
        """
        work_dir = Path(tempfile.mkdtemp(prefix="graphrag-eval-"))
        keep_work_dir = False
        try:
            exit_code, keep_work_dir = await self._run(work_dir)
            return exit_code
        except BaseException:
            keep_work_dir = not self._upload_finished_runs(work_dir)
            raise
        finally:
            if not keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            await self._close_arm_chains()

    async def _close_arm_chains(self) -> None:
        """Release the arms' store connections (the base chain is the
        caller's)."""
        chains, self._arm_chains = self._arm_chains, []
        for chain in chains:
            close = getattr(chain, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception as e:  # noqa: BLE001 - closing is best-effort
                    logger.warning("Could not close an --arm chain: %s", e)

    def _local_outputs(self, work_dir: Path) -> Path:
        """Where the result files are written (``work_dir`` for an S3 target)."""
        target = self._run_target()
        return work_dir / "outputs" if is_s3_uri(target) else Path(target)

    def _upload_finished_runs(self, work_dir: Path) -> bool:
        """Upload what an interrupted run already wrote to its S3 target.

        Returns whether ``work_dir`` may go: nothing to upload, or uploaded.
        """
        target = self._run_target()
        outputs = self._local_outputs(work_dir)
        if not is_s3_uri(target) or not any(outputs.glob("*")):
            return True
        # Before the upload: one the stop watchdog cuts short says no more.
        logger.warning(
            "Evaluation stopped; uploading the finished runs' results in '%s' "
            "to '%s' (they stay there unless the upload finishes).",
            outputs,
            target,
        )
        try:
            upload_directory_to_s3(
                s3_client_for(self.config), outputs, target, self.config
            )
        except Exception as e:  # noqa: BLE001 - already failing; keep the files
            logger.error(
                "Evaluation stopped; could not upload the finished runs' "
                "results to '%s' (%s). They are kept in '%s'.",
                target,
                e,
                outputs,
            )
            return False
        logger.warning(
            "Evaluation stopped; the finished runs' results were uploaded to '%s'.",
            target,
        )
        return True

    async def _run(self, work_dir: Path) -> tuple[int, bool]:
        """Run the evaluation; returns ``(exit code, keep work_dir)``."""
        outputs_target = self._run_target()
        s3_client = (
            s3_client_for(self.config)
            if is_s3_uri(self.args.data_path or "") or is_s3_uri(outputs_target)
            else None
        )
        display_ascii_art(version=__version__)
        local_outputs = self._local_outputs(work_dir)
        baseline = self._load_baseline()
        start_time = time.time()
        if self._from_results is not None:
            measured = await self._saved_runs_input(local_outputs)
        else:
            measured = await self._answer_and_score(
                s3_client, work_dir, local_outputs, baseline
            )

        comparison: dict[str, Any] | None = None
        gate_failures: list[str] = []
        if measured.comparing:
            comparison, gate_failures = await self._compare(
                measured, baseline, local_outputs, time.time() - start_time
            )

        upload_error: AWSServiceError | None = None
        if s3_client is not None and is_s3_uri(outputs_target):
            try:
                upload_directory_to_s3(
                    s3_client, local_outputs, outputs_target, self.config
                )
            except AWSServiceError as e:
                upload_error = e

        console.rule("[bold]Evaluation Complete[/bold]", style="blue")
        console.print(
            "\n[bold green]Results saved to "
            f"'{markup_safe(local_outputs if upload_error else outputs_target)}'"
            "[/bold green]"
        )

        exit_code = 0
        if upload_error:
            # Keep the local copy so a failed upload loses no results.
            console.print(
                f"[red]Error: {markup_safe(upload_error)}. Local results kept in "
                f"'{markup_safe(local_outputs)}'.[/red]"
            )
            exit_code = EXIT_FAILURE

        breaches = measured.breaches
        if breaches:
            console.print(
                f"[red]Failure budget exceeded ({markup_safe('; '.join(breaches))}; "
                f"--max-failure-rate {self.args.max_failure_rate}, which a run "
                "whose every query, or every attempt of a metric, failed "
                "breaches at any value).[/red]"
            )
            exit_code = EXIT_FAILURE
        if gate_failures:
            console.print(
                "[red]Regression gate failed: "
                f"{markup_safe('; '.join(gate_failures))}.[/red]"
            )
            exit_code = EXIT_FAILURE
        if self.json_output:
            print_json(
                {
                    "output_directory": str(
                        local_outputs if upload_error else outputs_target
                    ),
                    "upload_error": str(upload_error) if upload_error else None,
                    "runs": measured.summaries,
                    "comparison": comparison,
                    "failure_budget_breaches": breaches,
                    "gate_failures": gate_failures,
                    **self._failure(exit_code, breaches, gate_failures, upload_error),
                }
            )
        return exit_code, upload_error is not None

    @staticmethod
    def _failure(
        exit_code: int,
        breaches: list[str],
        gate_failures: list[str],
        upload_error: Exception | None,
    ) -> dict[str, Any]:
        """The failure keys of a result that exits 1 (a breached budget, a
        failed gate, a failed upload), as every CLI's failure document has
        them; ``error_type`` names the first."""
        if not exit_code:
            return {}
        errors = [*breaches, *gate_failures]
        if upload_error is not None:
            errors.append(str(upload_error))
        if breaches:
            kind = "FailureBudgetExceeded"
        elif gate_failures:
            kind = "RegressionGateFailed"
        else:
            kind = type(upload_error).__name__
        return failure_fields(exit_code, "; ".join(errors) or None, kind)

    def _load_baseline(self) -> dict[str, list[SavedRun]] | None:
        """The ``--baseline-results`` runs by strategy (in the order the
        earlier run listed them), labelled ``baseline:<strategy>``; checked
        against this run's answer scoring before any answer is generated."""
        if not self._baseline_results:
            return None
        try:
            grouped = group_runs(load_saved_runs(self._baseline_results))
        except EvaluationException as e:
            raise InvalidDatasetError(f"--baseline-results: {e}") from e
        self._check_pooling(grouped, "--baseline-results")
        return {f"baseline:{strategy}": runs for strategy, runs in grouped.items()}

    def _check_pooling(self, grouped: dict[str, list[SavedRun]], flag: str) -> None:
        """Refuse (exit 2) the saved runs of a strategy that are not one
        run's repeats, unless ``--pool-runs``: then a warning per strategy
        names what its pooled runs mix (logged here; the settings ones
        reach the comparison's warnings too)."""
        errors = pooling_errors(grouped)
        if not errors:
            return
        if getattr(self.args, "pool_runs", False):
            for error in errors:
                logger.warning("%s: pooled with --pool-runs: %s", flag, error)
            return
        raise InvalidDatasetError(
            f"{flag}: {'; '.join(errors)}. They are not one run's repeats: "
            "list the result files, or the run subdirectory, of one run-eval "
            "invocation, or give --pool-runs to pool them anyway"
        )

    def _check_baseline_scoring(
        self,
        baseline: dict[str, list[SavedRun]] | None,
        queries: list[EvaluationQuery],
        ground_truths: list[EvaluationGroundTruth],
    ) -> None:
        """Refuse a ``--baseline-results`` run scored unlike this one once
        the dataset is read (whether it holds Han, Hangul or Kana text
        decides if token F1's tokenizations differ), before any answer is
        generated."""
        if not baseline:
            return
        try:
            check_answer_scoring(
                [
                    *(
                        ("baseline", run.manifest)
                        for runs in baseline.values()
                        for run in runs
                    ),
                    ("this run", {"answer_scoring": self._answer_scoring()}),
                ],
                dense_script=EvaluationManager.holds_dense_script(
                    queries, ground_truths
                ),
            )
        except EvaluationException as e:
            raise InvalidDatasetError(f"--baseline-results: {e}") from e

    async def _check_methodologies(
        self,
        queries: list[EvaluationQuery],
        strategies: list[str] | None,
        chain: Any = None,
        where: str = "",
    ) -> None:
        """Refuse, before any answer, a strategy a queried namespace does
        not serve (``MethodologyNotIndexedError``: every one of its queries
        would fail; exit 2, as ``run-rag``): each of ``strategies`` (a
        comparison's), else each query's own, on each query's suffix."""
        check = getattr(chain or self.rag_chain, "acheck_methodology", None)
        if not inspect.iscoroutinefunction(check):
            return
        pairs = dict.fromkeys(
            (strategy, q.metadata.get("suffix"))
            for q in queries
            for strategy in (
                strategies
                if strategies is not None
                else [q.metadata.get("search_strategy")]
            )
            if strategy and strategy in _STRATEGY_VALUES
        )
        for strategy, suffix in pairs:
            try:
                await check(SearchStrategy(strategy), suffix)
            except MethodologyNotIndexedError as e:
                raise MethodologyNotIndexedError(
                    f"--search-strategy {strategy}{where}: {e}"
                ) from e

    def _answer_scoring(self) -> dict[str, Any] | None:
        """The answer_scoring this run's manifests will record."""
        if EvaluatorType.ANSWER_MATCH not in self.config.evaluation.enabled_evaluators:
            return None
        return answer_scoring(self.config)

    async def _saved_runs_input(self, local_outputs: Path) -> _Measured:
        """The comparison input rebuilt from ``--from-results`` (with
        ``--rescore``, from its answers scored again)."""
        try:
            saved = load_saved_runs(self._from_results or [])
            order = None
            if self._strategies_given():
                # The closed-book arm is no --search-strategy: compared last.
                labels = list(dict.fromkeys(run.strategy for run in saved))
                order = [
                    *self.args.search_strategy,
                    # The --arm runs of the listed strategies.
                    *(
                        label
                        for label in strategy_order(saved)
                        if is_arm_label(label)
                        and label_strategy(label) in self.args.search_strategy
                    ),
                    *closed_book_arms(labels),
                ]
            grouped = group_runs(saved, order)
        except EvaluationException as e:
            raise InvalidDatasetError(f"--from-results: {e}") from e
        self._check_pooling(grouped, "--from-results")
        warnings = [
            *summary_warnings(saved, "--from-results"),
            *pooling_warnings(grouped),
        ]
        if self._rescoring:
            grouped = await self._rescore(grouped, local_outputs)
            saved = [run for runs in grouped.values() for run in runs]
        console.rule(
            f"[bold]Rebuilding the comparison from {len(saved)} saved run(s)[/bold]",
            style="blue",
        )
        summaries: list[dict[str, Any]] = []
        for strategy, runs in grouped.items():
            for repeat, run in enumerate(runs, start=1):
                summaries.append(
                    {
                        "strategy": strategy,
                        "repeat": repeat,
                        "path": str(run.path),
                        **(
                            run.summary.model_dump(mode="json", include=_SUMMARY_FIELDS)
                            if run.summary
                            else {}
                        ),
                    }
                )
        categories: dict[str, str] = {}
        for runs in grouped.values():
            for run in runs:
                for result in run.results:
                    if result.category:
                        categories.setdefault(result.query_id, result.category)
        first = next(iter(grouped.values()))[0]
        return _Measured(
            runs={s: [run.run for run in runs] for s, runs in grouped.items()},
            manifests=[
                (s, run.manifest) for s, runs in grouped.items() for run in runs
            ],
            manifest=first.manifest,
            categories=categories,
            summaries=summaries,
            breaches=[],
            comparing=True,
            warnings=warnings,
        )

    def _rescore_dataset(
        self,
    ) -> tuple[Path | None, dict[str, tuple[EvaluationQuery, EvaluationGroundTruth]]]:
        """``--rescore DATASET``'s path and its items by query id (none
        without a DATASET: the saved gold answers score)."""
        dataset = getattr(self.args, "rescore", None)
        if not dataset:
            return None, {}
        try:
            queries, ground_truths = EvaluationManager.load_data(dataset)
        except (json.JSONDecodeError, EvaluationException) as e:
            raise InvalidDatasetError(f"--rescore: {e}") from e
        golds = {gt.query_id: gt for gt in ground_truths}
        return Path(dataset), {
            q.query_id: (
                q,
                golds.get(q.query_id)
                or EvaluationGroundTruth(query_id=q.query_id, ground_truth=""),
            )
            for q in queries
        }

    async def _rescore(
        self, grouped: dict[str, list[SavedRun]], local_outputs: Path
    ) -> dict[str, list[SavedRun]]:
        """Score every saved run's answers again with this run's evaluators,
        save them as runs of this invocation's directory, and return them in
        place of the saved ones."""
        dataset_path, dataset = self._rescore_dataset()
        console.rule("[bold]Re-scoring the saved answers[/bold]", style="blue")
        managers: dict[bool, EvaluationManager] = {}
        rescored: dict[str, list[SavedRun]] = {}
        for label, runs in grouped.items():
            rescored[label] = []
            for run in runs:
                # answer_match and graph_aware read the answers without
                # their citation markers when the run's answers carried them.
                cited = bool(
                    ((run.manifest or {}).get("citations") or {}).get(
                        "enabled", self.config.search.citations.enabled
                    )
                )
                if cited not in managers:
                    config = self.config.model_copy(deep=True)
                    config.search.citations.enabled = cited
                    managers[cited] = EvaluationManager(
                        config=config,
                        rag_chain=_no_answers(),
                        providers=(
                            self.evaluation_manager.providers
                            if self.evaluation_manager is not None
                            else self.providers
                        ),
                    )
                    self.evaluation_manager = self.evaluation_manager or managers[cited]
                rescored[label].append(
                    await self._rescore_run(
                        managers[cited],
                        label,
                        run,
                        (dataset_path, dataset),
                        local_outputs,
                    )
                )
        return rescored

    async def _rescore_run(
        self,
        manager: EvaluationManager,
        label: str,
        run: SavedRun,
        dataset: tuple[
            Path | None, dict[str, tuple[EvaluationQuery, EvaluationGroundTruth]]
        ],
        local_outputs: Path,
    ) -> SavedRun:
        """One saved run, scored again and saved."""
        dataset_path, items = dataset
        results = [result.model_copy(deep=True) for result in run.results]
        queries: list[EvaluationQuery] = []
        ground_truths: list[EvaluationGroundTruth] = []
        for result in results:
            metadata = dict(result.metadata)
            category = result.category
            if items:
                if result.query_id not in items:
                    raise InvalidDatasetError(
                        f"--rescore: the dataset has no query '{result.query_id}' "
                        f"of {run.path.name}"
                    )
                query, gold = items[result.query_id]
                if query.question != result.question:
                    raise InvalidDatasetError(
                        f"--rescore: query '{result.query_id}' asks another "
                        f"question in the dataset than in {run.path.name}"
                    )
                for key in _SAVED_EXPECTATIONS:
                    metadata.pop(key, None)
                # The dataset's metadata over the saved one; the run's own
                # strategy and query flags stay.
                metadata.update(
                    {
                        k: v
                        for k, v in query.metadata.items()
                        if k not in _RUN_METADATA or k not in metadata
                    }
                )
                category = query.category or category
            else:
                gold = EvaluationGroundTruth(
                    query_id=result.query_id,
                    ground_truth=result.ground_truth,
                    reference_sources=metadata.get("reference_sources") or [],
                    expected_entities=metadata.get("expected_entities") or [],
                    expected_relationships=(
                        metadata.get("expected_relationships") or []
                    ),
                    assertions=metadata.get("assertions") or [],
                )
            result.metadata = metadata
            result.category = category
            queries.append(
                EvaluationQuery(
                    query_id=result.query_id,
                    question=result.question,
                    category=category,
                    metadata=metadata,
                )
            )
            ground_truths.append(gold)
        old = run.summary
        answer_usage = ((old.usage if old else None) or {}).get("answer_generation")
        reports, summary = await manager.score_answers(
            queries,
            results,
            ground_truths,
            start_time=(
                old.evaluation_start_time
                if old and old.evaluation_start_time
                else datetime.now()
            ),
            answer_usage=(
                answer_usage
                if isinstance(answer_usage, LLMUsage)
                else LLMUsage.model_validate(answer_usage or {})
            ),
            dataset_path=dataset_path,
            cli_args=vars(self.args),
        )
        summary.run_manifest = rescored_manifest(
            run.manifest, summary.run_manifest or {}, run.path, dataset_path
        )
        summary.configuration["outputs_directory"] = self._run_target()
        path = manager.save_results(
            results=results,
            reports=reports,
            summary=summary,
            outputs_dir=local_outputs,
            label=label,
        )
        return SavedRun(
            path=path,
            strategy=label,
            stamp=run.stamp,
            results=results,
            reports=reports,
            summary=summary,
        )

    async def _answer_and_score(
        self,
        s3_client: Any,
        work_dir: Path,
        local_outputs: Path,
        baseline: dict[str, list[SavedRun]] | None,
    ) -> _Measured:
        """Answer and score the dataset once per strategy and repeat."""
        assert self.rag_chain is not None
        eval_data_uri = self.args.data_path
        outputs_target = self._run_target()
        eval_data_path: str | Path = eval_data_uri
        if s3_client is not None and is_s3_uri(eval_data_uri):
            eval_data_path = download_s3_object(
                s3_client,
                eval_data_uri,
                work_dir / "dataset" / "eval_data.json",
            )

        console.rule("[bold]Initializing Evaluation[/bold]", style="blue")

        self.evaluation_manager = EvaluationManager(
            config=self.config, rag_chain=self.rag_chain
        )

        strategies: list[str] = list(self.args.search_strategy)
        closed_book = bool(getattr(self.args, "closed_book", False))
        comparing = (
            len(strategies) > 1
            or self.args.repeats > 1
            or bool(baseline)
            or closed_book
            or bool(self._arms)
        )
        try:
            queries, ground_truths = self.evaluation_manager.load_data(
                eval_data_path=eval_data_path,
                base_metadata={
                    "suffix": query_suffix(self.args, self.config),
                    # A comparison sets each run's strategy itself (below).
                    "search_strategy": None if comparing else strategies[0],
                    "search_type": self.args.search_type,
                    "top_k": self.args.top_k,
                    "retrieval_multiplier": self.args.retrieval_multiplier,
                },
                # The keys the chain would refuse per query, refused now.
                check_filter_keys=getattr(self.rag_chain, "check_filter_keys", None),
                run_strategies=strategies if comparing else None,
            )
        except FileNotFoundError as e:
            raise InvalidDatasetError(
                f"evaluation dataset '{eval_data_uri}' not found"
            ) from e
        except json.JSONDecodeError as e:
            raise InvalidDatasetError(
                f"evaluation dataset '{eval_data_uri}' is not valid JSON: {e}"
            ) from e
        except EvaluationException as e:
            raise InvalidDatasetError(str(e)) from e

        if not ground_truths:
            logger.warning(
                "No ground truth data found in file; some evaluators may not work."
            )
            ground_truths = [
                EvaluationGroundTruth(query_id=q.query_id, ground_truth="")
                for q in queries
            ]
        self._check_baseline_scoring(baseline, queries, ground_truths)
        await self._check_methodologies(queries, strategies if comparing else None)

        recorded = _Measured(
            runs={},
            manifests=[],
            manifest=None,
            categories={q.query_id: q.category for q in queries if q.category},
            summaries=[],
            comparing=comparing,
        )
        managers = dict.fromkeys(strategies, self.evaluation_manager)
        # Per run label, its strategy and its arm (None: the base config).
        runs_of: dict[str, tuple[str, dict[str, Any] | None]] = {
            strategy: (strategy, None) for strategy in strategies
        }
        for arm in self._arms:
            manager = await self._arm_manager(arm, queries, strategies)
            for strategy in strategies:
                label = arm_label(strategy, arm["name"])
                managers[label] = manager
                runs_of[label] = (strategy, arm)
        if closed_book:
            # No retrieval: the answer model alone, scored by the same
            # evaluators (sharing the strategies' providers).
            providers = self.evaluation_manager.providers
            managers[CLOSED_BOOK_ARM] = EvaluationManager(
                config=self.config,
                rag_chain=closed_book_chain(self.config, providers),
                providers=providers,
            )

        def arm_queries(label: str) -> list[EvaluationQuery]:
            # The run's strategy overrides a per-item one: every run of a
            # comparison must answer the same queries the same way.
            if not comparing:
                return queries
            strategy, config_arm = runs_of.get(label, (label, None))
            arm: dict[str, Any] = (
                {"search_strategy": strategy, "retrieval": "none"}
                if strategy == CLOSED_BOOK_ARM
                else {"search_strategy": strategy}
            )
            if config_arm is not None and config_arm.get("suffix"):
                arm["suffix"] = config_arm["suffix"]
            return [
                q.model_copy(update={"metadata": {**q.metadata, **arm}})
                for q in queries
            ]

        planned = [
            (strategy, repeat)
            for strategy in managers
            for repeat in range(1, self.args.repeats + 1)
        ]
        context = _RunContext(
            ground_truths=ground_truths,
            dataset_path=eval_data_path,
            dataset_uri=eval_data_uri,
            outputs_target=outputs_target,
            local_outputs=local_outputs,
            run_order=self._run_order() if len(planned) > 1 else None,
            repeat_extractions=(
                self._repeat_extractions() if self.args.repeats > 1 else None
            ),
            arms={
                label: {
                    "name": arm["name"],
                    "overlay": arm["overlay"],
                    "overlay_sha256": overlay_sha256(arm),
                    "suffix": arm.get("suffix"),
                }
                for label, (_, arm) in runs_of.items()
                if arm is not None
            },
            compared_runs=list(managers),
        )
        if context.run_order == "interleaved":
            await self._answer_interleaved(
                planned, managers, arm_queries, context, recorded
            )
            return recorded
        for strategy, repeat in planned:
            manager = managers[strategy]
            run_queries = arm_queries(strategy)
            label = (
                f" ({strategy}"
                f"{', no retrieval' if strategy == CLOSED_BOOK_ARM else ''}, "
                f"repeat {repeat}/{self.args.repeats})"
                if comparing
                else ""
            )
            console.rule(
                f"[bold]Running Evaluation on {len(queries)} Queries{label}[/bold]",
                style="blue",
            )
            if repeat == 1 and queries and strategy != CLOSED_BOOK_ARM:
                # Untimed: the first strategy must not pay for warming
                # the connections and models the others then reuse.
                await manager.warm_up(run_queries[0])
            run_start = time.time()
            # On the run's root spans (kg_rag.batch, kg_rag.evaluate), so
            # the runs of a comparison tell apart.
            with (
                observability.bind_attributes(
                    **{"eval.strategy": strategy, "eval.repeat": repeat}
                ),
                self._extraction_scope(repeat),
            ):
                results, reports, summary = await manager.evaluate_dataset(
                    queries=run_queries,
                    ground_truths=ground_truths,
                    show_progress=True,
                    dataset_path=eval_data_path,
                    cli_args=vars(self.args),
                )
            self._record_run(
                strategy,
                repeat,
                manager,
                (reports, results),
                summary,
                context,
                recorded,
            )
            if not comparing:
                self._print_summary(summary, time.time() - run_start)
        return recorded

    async def _arm_manager(
        self,
        arm: dict[str, Any],
        queries: list[EvaluationQuery],
        strategies: list[str],
    ) -> EvaluationManager:
        """The evaluation manager of an ``--arm``: its configuration (the
        overlay over the run's), its own chain and the run's evaluators.
        Its chain shares the base chain's query extraction cache (keyed by
        prompt, model and question), so an arm whose overlay leaves the
        extraction alone retrieves on the same keywords.

        Raises:
            ConfigurationError: The overlay is invalid, or the arm's
                configuration lacks an extra or an endpoint its strategies
                need (exit 2, before any answer).
        """
        config, warnings = arm_config(
            self.config,
            arm,
            allow_unknown_keys=bool(getattr(self.args, "allow_unknown_keys", False)),
        )
        for warning in warnings:
            logger.warning("%s", warning)
        error = missing_extras_error(config, seeds=False) or missing_endpoints_error(
            config,
            {
                role
                for name in strategies
                for role in strategy_roles(config, SearchStrategy(name))
            },
            f"--arm {arm['name']}",
        )
        if error:
            raise ConfigurationError(error)
        chain = (self.chain_factory or GraphRAGChain)(config)
        self._arm_chains.append(chain)
        shared = getattr(self.rag_chain, "_extractions", None)
        if shared is not None and hasattr(chain, "_extractions"):
            chain._extractions = shared
        suffix = {"suffix": arm["suffix"]} if arm.get("suffix") else {}
        await self._check_methodologies(
            [
                q.model_copy(update={"metadata": {**q.metadata, **suffix}})
                for q in queries
            ],
            strategies,
            chain,
            f" (--arm {arm['name']})",
        )
        return EvaluationManager(config=config, rag_chain=chain)

    def _run_order(self) -> str:
        """``--run-order``, else ``evaluation.run_order``."""
        return str(
            getattr(self.args, "run_order", None) or self.config.evaluation.run_order
        )

    def _repeat_extractions(self) -> str:
        """``--repeat-extractions``, else ``evaluation.repeat_extractions``."""
        return str(
            getattr(self.args, "repeat_extractions", None)
            or self.config.evaluation.repeat_extractions
        )

    def _extraction_scope_name(self, repeat: int) -> str | None:
        """The query extraction cache scope of ``repeat``: its own with
        several repeats resampling their extractions, else ``None``."""
        if self.args.repeats > 1 and self._repeat_extractions() == "resample":
            return f"{self.run_id}/repeat-{repeat}"
        return None

    def _extraction_scope(self, repeat: int) -> AbstractContextManager[None]:
        scope = self._extraction_scope_name(repeat)
        return nullcontext() if scope is None else extraction_cache_scope(scope)

    async def _answer_interleaved(
        self,
        planned: list[tuple[str, int]],
        managers: dict[str, EvaluationManager],
        arm_queries: Callable[[str], list[EvaluationQuery]],
        context: _RunContext,
        recorded: _Measured,
    ) -> None:
        """Answer every run of the comparison interleaved, query by query
        (``EvaluationManager.answer_interleaved``), then score each run."""
        arms = {
            (strategy, repeat): AnswerArm(
                manager=managers[strategy],
                queries=arm_queries(strategy),
                attributes={"eval.strategy": strategy, "eval.repeat": repeat},
                extraction_scope=self._extraction_scope_name(repeat),
            )
            for strategy, repeat in planned
        }
        for strategy, manager in managers.items():
            run_queries = arms[(strategy, 1)].queries
            if run_queries and strategy != CLOSED_BOOK_ARM:
                # Untimed, as before a sequential run.
                await manager.warm_up(run_queries[0])
        n_queries = len(arms[planned[0]].queries)
        console.rule(
            f"[bold]Running Evaluation on {n_queries} Queries x {len(planned)} "
            "runs, interleaved[/bold]",
            style="blue",
        )
        start = datetime.now()
        answered = await EvaluationManager.answer_interleaved(
            list(arms.values()), show_progress=True
        )
        for (strategy, repeat), (results, usage) in zip(planned, answered, strict=True):
            manager = managers[strategy]
            with observability.bind_attributes(
                **{"eval.strategy": strategy, "eval.repeat": repeat}
            ):
                reports, summary = await manager.score_answers(
                    arms[(strategy, repeat)].queries,
                    results,
                    context.ground_truths,
                    start_time=start,
                    answer_usage=usage,
                    dataset_path=context.dataset_path,
                    cli_args=vars(self.args),
                )
            self._record_run(
                strategy,
                repeat,
                manager,
                (reports, results),
                summary,
                context,
                recorded,
            )

    def _record_run(
        self,
        strategy: str,
        repeat: int,
        manager: EvaluationManager,
        run: Run,
        summary: EvaluationSummary,
        context: _RunContext,
        recorded: _Measured,
    ) -> None:
        """Save one run's files and add it to the comparison's input."""
        reports, results = run
        if summary.run_manifest is not None:
            if is_s3_uri(context.dataset_uri):
                # Record where the dataset came from, not the temporary copy.
                summary.run_manifest.get("dataset", {})["path"] = context.dataset_uri
            # Which invocation answered it, and how its runs were ordered:
            # the comparison warns on latency deltas across runs not
            # answered interleaved together.
            summary.run_manifest["run_id"] = self.run_id
            # Which repeat it is: the interleaved runs share a start time,
            # so a comparison rebuilt from the files pairs repeat r with
            # repeat r by this, not by the files' random stamps.
            summary.run_manifest["repeat"] = repeat
            if context.run_order is not None:
                summary.run_manifest["run_order"] = context.run_order
            if context.repeat_extractions is not None:
                summary.run_manifest["repeat_extractions"] = context.repeat_extractions
            if strategy in context.arms:
                # The configuration arm it ran under (--arm).
                summary.run_manifest["arm"] = context.arms[strategy]
            if context.arms:
                # The runs in comparison order: the files' stamps do not
                # order the arms of an interleaved run.
                summary.run_manifest["compared_runs"] = context.compared_runs
        # The directory this run wrote to (--output-directory, or the config
        # default, and its run subdirectory).
        summary.configuration["outputs_directory"] = context.outputs_target
        manager.save_results(
            results=results,
            reports=reports,
            summary=summary,
            outputs_dir=context.local_outputs,
            label=strategy if strategy in context.arms else None,
        )
        recorded.manifest = recorded.manifest or summary.run_manifest
        recorded.manifests.append((strategy, summary.run_manifest))
        recorded.runs.setdefault(strategy, []).append(run)
        recorded.summaries.append(
            {
                "strategy": strategy,
                "repeat": repeat,
                **summary.model_dump(mode="json", include=_SUMMARY_FIELDS),
            }
        )
        recorded.breaches.extend(
            f"{strategy}#{repeat}: {b}" if recorded.comparing else b
            for b in failure_budget_breaches(summary, self.args.max_failure_rate)
        )

    def _baseline_pairs(
        self, measured: _Measured, baseline: dict[str, list[SavedRun]]
    ) -> tuple[dict[str, str], list[str]]:
        """Each strategy's baseline label: ``baseline:<s>``, else the
        baseline's first strategy (warned about)."""
        first = next(iter(baseline))
        pairs: dict[str, str] = {}
        unmatched: list[str] = []
        for strategy in measured.runs:
            label = f"baseline:{strategy}"
            if label not in baseline:
                label = first
                unmatched.append(strategy)
            pairs[strategy] = label
        warnings = []
        if unmatched:
            warnings.append(
                f"--baseline-results has no run of {', '.join(unmatched)} "
                f"(it holds {', '.join(s.split(':', 1)[1] for s in baseline)}): "
                f"compared with {first}, the earlier run's first strategy, so "
                "those deltas compare two strategies, not two runs of one."
            )
        return pairs, warnings

    async def _compare(
        self,
        measured: _Measured,
        baseline: dict[str, list[SavedRun]] | None,
        local_outputs: Path,
        elapsed: float,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build, judge, gate, save and print the paired comparison; returns
        it and the gate failures."""
        runs = dict(measured.runs)
        manifests = list(measured.manifests)
        warnings = list(measured.warnings)
        pairs: dict[str, str] = {}
        if baseline is not None:
            pairs, unmatched = self._baseline_pairs(measured, baseline)
            used = {label: baseline[label] for label in dict.fromkeys(pairs.values())}
            runs = {
                **{label: [run.run for run in saved] for label, saved in used.items()},
                **runs,
            }
            manifests = [
                (label, run.manifest) for label, saved in used.items() for run in saved
            ] + manifests
            warnings += [
                *unmatched,
                *summary_warnings(
                    [run for saved in used.values() for run in saved],
                    "--baseline-results",
                ),
                *pooling_warnings(used),
            ]
            if self._output_is_baseline():
                warnings.append(
                    "--output-directory is a --baseline-results directory: this "
                    "run's results are saved next to the baseline's, and a later "
                    "run reading it as its baseline pools them in as repeats."
                )
        try:
            warnings += [
                *check_answer_scoring(manifests),
                *dataset_warnings(manifests),
                *interleaving_warnings(manifests),
                *citation_warnings(manifests),
            ]
        except EvaluationException as e:
            raise InvalidDatasetError(str(e)) from e
        for warning in warnings:
            logger.warning("%s", warning)
        if self._gates and len(runs) < 2:
            raise InvalidDatasetError(
                f"--fail-if needs two strategies to compare; the runs hold "
                f"{', '.join(runs) or 'none'}"
            )
        prices = self.config.evaluation.prices
        # Runs saved before their usage counted rerank requests.
        unmetered = rerank_unmetered(manifests)
        names = list(runs)
        if pairs:
            comparison = baseline_comparison(
                {label: runs[label] for label in dict.fromkeys(pairs.values())},
                measured.runs,
                pairs,
                categories=measured.categories,
                prices=prices,
                unmetered_rerank=unmetered,
                resamples=self.config.evaluation.resamples,
            )
            rows = [
                row
                for strategy, label in pairs.items()
                for row in per_query_rows(
                    {label: runs[label], strategy: runs[strategy]},
                    measured.categories,
                    prices,
                )
            ]
        elif arms_paired := arm_pairs(names):
            # Each --arm run against the same strategy's base run, the
            # base strategies against the first.
            pairs = arms_paired
            comparison = baseline_comparison(
                {label: runs[label] for label in dict.fromkeys(pairs.values())},
                {label: runs[label] for label in pairs},
                pairs,
                categories=measured.categories,
                prices=prices,
                unmetered_rerank=unmetered,
                resamples=self.config.evaluation.resamples,
            )
            rows = [
                row
                for label, base in pairs.items()
                for row in per_query_rows(
                    {base: runs[base], label: runs[label]},
                    measured.categories,
                    prices,
                )
            ]
        else:
            comparison = paired_comparison(
                runs,
                categories=measured.categories,
                prices=prices,
                unmetered_rerank=unmetered,
                resamples=self.config.evaluation.resamples,
            )
            rows = per_query_rows(runs, measured.categories, prices)
            pairs = dict.fromkeys(names[1:], names[0])
        # The --judge-model warning was logged when the arguments were read.
        comparison["warnings"] = [
            *self._judge_model_warnings(),
            *warnings,
            *comparison["warnings"],
        ]
        if self.args.pairwise_judge and len(names) > 1:
            judge = PairwiseJudge(
                self.config,
                providers=(
                    self.evaluation_manager.providers
                    if self.evaluation_manager is not None
                    else self.providers
                ),
            )
            every_repeat = (
                getattr(self.args, "pairwise_repeats", None)
                or self.config.evaluation.pairwise_repeats
            ) == "all"
            with usage_scope() as judging:
                comparison["pairwise_vs_baseline"] = {
                    strategy: (
                        await judge.acompare_repeats(
                            [results for _, results in runs[label]],
                            [results for _, results in runs[strategy]],
                        )
                        if every_repeat
                        else await judge.acompare(
                            runs[label][0][1], runs[strategy][0][1]
                        )
                    )
                    for strategy, label in pairs.items()
                }
            comparison["pairwise_judge_model"] = self.config.model_id(
                ModelRole.EVALUATION
            )
            comparison["pairwise_usage"] = judging.snapshot()
            comparison["warnings"].extend(
                pairwise_warnings(
                    comparison["pairwise_vs_baseline"], self._judge_manifest(measured)
                )
            )
            measured.breaches.extend(
                pairwise_budget_breaches(
                    comparison["pairwise_vs_baseline"],
                    self.args.max_failure_rate,
                )
            )
        arms = closed_book_arms(names)
        if arms:
            # No retrieval: a bound on prior knowledge, not a strategy to gate.
            comparison["closed_book_arms"] = arms
        adjust_for_multiplicity(
            comparison,
            primary_metric=(
                getattr(self.args, "primary_metric", None)
                or self.config.evaluation.primary_metric
            ),
            exclude=arms,
        )
        comparison["run_manifest"] = measured.manifest
        if self._from_results is not None or baseline is not None:
            comparison["sources"] = {
                "from_results": [str(p) for p in self._from_results or []] or None,
                "baseline_results": [str(p) for p in self._baseline_results or []]
                or None,
            }
        gate_failures = gate_breaches(comparison, self._gates)
        if self._gates:
            comparison["gates"] = {
                "conditions": [str(gate) for gate in self._gates],
                "breaches": gate_failures,
            }
        path = self._save_comparison(comparison, local_outputs)
        path.with_suffix(".csv").write_text(comparison_csv(rows), encoding="utf-8")
        path.with_suffix(".md").write_text(
            comparison_markdown(comparison), encoding="utf-8"
        )
        self._print_comparison(comparison, elapsed)
        return comparison, gate_failures

    def _judge_manifest(self, measured: _Measured) -> dict[str, Any] | None:
        """The manifest the judge warnings read: a re-judge's judge is this
        run's evaluation model, compared with the saved runs' answer model."""
        if self._from_results is None and self._baseline_results is None:
            return measured.manifest
        answer = ((measured.manifest or {}).get("models") or {}).get(
            "answer_generation"
        )
        judge = model_id_value(self.config.model_id(ModelRole.EVALUATION))
        return {
            "models": {
                "judge_is_answer_model": (
                    answer is not None and model_id_value(answer) == judge
                )
            }
        }


def _no_answers() -> Runnable[Any, Any]:
    """The chain of a manager that only scores (``--rescore``)."""
    from langchain_core.runnables import RunnableLambda

    def refuse(inputs: Any) -> Any:
        raise EvaluationException("re-scoring saved answers generates none")

    return RunnableLambda(refuse)


# The expectations the scoring wrote into a saved result's metadata, which a
# --rescore dataset replaces.
_SAVED_EXPECTATIONS = (
    "answer_aliases",
    "expected_entities",
    "expected_relationships",
    "reference_sources",
    "assertions",
)
# The query settings a run answered with (its strategy and query flags):
# a --rescore dataset does not overwrite them.
_RUN_METADATA = frozenset(
    {"search_strategy", "suffix", "search_type", "top_k", "retrieval_multiplier"}
)
# The manifest entries a re-score replaces: how the answers were scored.
# What answered them (models, index, configuration, run order) stays.
_SCORING_MANIFEST_KEYS = (
    "answer_scoring",
    "enabled_evaluators",
    "dropped_evaluators",
    "assertion_scoring",
)
_SCORING_MODEL_KEYS = (
    "evaluation_judge",
    "evaluation_embedding",
    "answer_span",
    "judge_effort",
)


def rescored_manifest(
    saved: dict[str, Any] | None,
    scored: dict[str, Any],
    source: Path,
    dataset_path: Path | None,
) -> dict[str, Any]:
    """The run manifest of a re-scored run: the saved one (what answered),
    with the scoring entries of ``scored`` (the manifest the re-score
    built) and a ``rescored`` record of when, from which file and by which
    code; the dataset's entry too when ``--rescore`` read one."""
    saved = dict(saved or {})
    merged = {**saved, **{key: scored.get(key) for key in _SCORING_MANIFEST_KEYS}}
    models = dict(saved.get("models") or {})
    new_models = scored.get("models") or {}
    models.update({key: new_models.get(key) for key in _SCORING_MODEL_KEYS})
    judge = models.get("evaluation_judge")
    answer = models.get("answer_generation")
    models["judge_is_answer_model"] = (
        model_id_value(judge) == model_id_value(answer)
        if judge is not None and answer is not None
        else None
    )
    merged["models"] = models
    citations = dict(saved.get("citations") or {})
    citations["scoring"] = (scored.get("citations") or {}).get("scoring")
    merged["citations"] = citations
    if dataset_path is not None:
        merged["dataset"] = scored.get("dataset")
    merged["rescored"] = {
        "created_at": scored.get("created_at"),
        "from": str(source),
        "package_version": scored.get("package_version"),
        "git_sha": scored.get("git_sha"),
        "dataset": str(dataset_path) if dataset_path is not None else None,
    }
    return merged


# The run summary fields the JSON result lists per run.
_SUMMARY_FIELDS = {
    "total_queries",
    "successful_evaluations",
    "failed_evaluations",
    "benchmarks",
    "average_response_time",
    "metric_statistics",
    "metric_outcomes",
    "abstention_statistics",
    "answer_span_statistics",
    "guardrail_blocked_counts",
    "structured_statistics",
    "iterative_statistics",
    "usage",
}


class _RunContext(BaseModel):
    """What every run of one run-eval invocation is recorded with."""

    ground_truths: list[EvaluationGroundTruth]
    dataset_path: str | Path
    dataset_uri: str | Path
    outputs_target: str
    local_outputs: Path
    # How several runs were ordered (None: one run), and whether repeats
    # resampled their extractions (None: one repeat).
    run_order: str | None = None
    repeat_extractions: str | None = None
    # Per --arm run label, what its manifest records of the arm; the run
    # labels in comparison order.
    arms: dict[str, dict[str, Any]] = Field(default_factory=dict)
    compared_runs: list[str] = Field(default_factory=list)


class _Measured(BaseModel):
    """What a run measured (or read back): the comparison's input."""

    runs: dict[str, list[Run]]
    # (strategy, run manifest) per run, for the comparability checks.
    manifests: list[tuple[str, dict[str, Any] | None]]
    manifest: dict[str, Any] | None
    categories: dict[str, str]
    summaries: list[dict[str, Any]]
    breaches: list[str] = Field(default_factory=list)
    comparing: bool = False
    # Read back from saved runs: their comparability warnings.
    warnings: list[str] = Field(default_factory=list)


@exits_on_sigterm
def main() -> None:
    load_cli_dotenv()
    try:
        cli = CommandLineInterface()
        args = cli.parse_args()
        _load_evaluation()

        with exit_on_config_error():
            config = apply_provider_arguments(
                load_cli_config(args),
                args,
            )
        json_output(args)
        setup_cli_logging(config, args)
        if args.verbose:
            logging.getLogger("unified_kg_rag").setLevel(logging.DEBUG)
            logger.warning(
                "run-eval --verbose is deprecated and will be removed: use "
                "--log-level DEBUG (every CLI's log level flag)."
            )

        if args.from_results is not None:
            # Rebuilt from saved answers: no chain, no store, no answer call.
            exit_code = event_loop.run(
                stoppable(EvaluationRunner(args, None, config).run()),
                io_workers=config.processing.io_workers,
            )
            if exit_code:
                sys.exit(exit_code)
            return

        error = missing_extras_error(config, seeds=False) or missing_endpoints_error(
            config,
            {
                role
                for name in args.search_strategy
                for role in strategy_roles(config, SearchStrategy(name))
            },
            f"search strategy '{','.join(args.search_strategy)}'",
        )
        if error:
            usage_error(error)

        rag_chain = GraphRAGChain(config)

        async def _run_and_close() -> int:
            # Release the retrievers' Neptune/OpenSearch sockets on every exit
            # path, on the loop they were opened on.
            try:
                return await EvaluationRunner(args, rag_chain, config).run()
            finally:
                await rag_chain.aclose()

        exit_code = event_loop.run(
            stoppable(_run_and_close()), io_workers=config.processing.io_workers
        )
        if exit_code:
            sys.exit(exit_code)

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)

    except InvalidDatasetError as e:
        # An input error, like a usage error: no traceback.
        console.print(
            f"\n[red]Error: invalid evaluation dataset: {markup_safe(e)}[/red]"
        )
        logger.error("Invalid evaluation dataset: %s", e)
        sys.exit(EXIT_USAGE)

    except (ConfigurationError, IndexNotFoundError) as e:
        # A setting or a namespace no query of the run can be answered
        # with (the warm-up raises it): a usage error, before the dataset.
        console.print(f"\n[red]Error: {markup_safe(e)}[/red]")
        logger.error("Evaluation configuration error: %s", e)
        sys.exit(EXIT_USAGE)

    except AWSServiceError as e:
        # e.g. an s3:// dataset that cannot be downloaded.
        console.print(f"\n[red]Error: {markup_safe(e)}[/red]")
        logger.error("Evaluation failed: %s", e)
        sys.exit(EXIT_FAILURE)

    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {markup_safe(e)}[/red]")
        logger.exception("Unexpected error during evaluation")
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()
