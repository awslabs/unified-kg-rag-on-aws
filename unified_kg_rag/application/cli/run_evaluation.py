# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.table import Table

from unified_kg_rag import __version__
from unified_kg_rag.adapters.aws.s3_transfer import (
    download_s3_object,
    is_s3_uri,
    parse_s3_uri,
    s3_client_for,
    upload_directory_to_s3,
)
from unified_kg_rag.adapters.providers import Providers
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
    json_output,
    print_json,
    stoppable,
    usage_error,
)
from unified_kg_rag.application.cli.display import (
    console,
    display_ascii_art,
    markup_safe,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
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
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationSummary,
    EvaluatorType,
    ModelRole,
    SearchStrategy,
    model_id_value,
)
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.evaluation.comparison import (
    Run,
    baseline_comparison,
    paired_comparison,
    per_query_rows,
)
from unified_kg_rag.evaluation.gates import Gate, gate_breaches, parse_gate
from unified_kg_rag.evaluation.pairwise import PairwiseJudge
from unified_kg_rag.evaluation.report import (
    baseline_metric_keys,
    compared_pairs,
    comparison_csv,
    comparison_markdown,
)
from unified_kg_rag.evaluation.saved_results import (
    SavedRun,
    check_answer_scoring,
    dataset_warnings,
    group_runs,
    load_saved_runs,
    pooling_warnings,
    summary_warnings,
)
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    DataProcessingError,
    EvaluationException,
    get_logger,
    observability,
    setup_logging,
)
from unified_kg_rag.shared.usage import usage_scope
from unified_kg_rag.shared.utils import event_loop

logger = get_logger(__name__)


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
    try:
        return parse_gate(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


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
            description="GraphRAG Evaluation System - Assess retrieval and generation performance",
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
                "result directories or evaluation_results_*.json files (with "
                "their reports and summary next to them). No answer is "
                "generated and no store is queried; --pairwise-judge re-judges "
                "the saved answers (with --judge-model, another model). The "
                "strategies compare in --search-strategy order when given, else "
                "in the order the oldest run listed them. Local paths only."
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
            "--fail-if",
            type=_gate,
            action="append",
            metavar="MEASURE:STAT<OP>VALUE",
            help=(
                "Regression gate, repeatable: exit 1 when a compared strategy's "
                "paired delta against the baseline meets the condition, or has "
                "no delta of MEASURE. MEASURE is an <evaluator>/<metric> of "
                "deltas_vs_baseline or a cost measure (usage/llm_calls, "
                "usage/input_tokens, usage/output_tokens, usage/total_tokens, "
                "usage/cost_usd, latency/response_time_seconds); STAT is "
                "mean_delta, ci_low or ci_high; OP is <, <=, > or >=. Example: "
                "answer_match/token_f1:ci_high<-0.02. Needs a comparison (two "
                "strategies or --baseline-results)"
            ),
        )
        parser.add_argument(
            "--judge-model",
            type=_model_id,
            metavar="MODEL_ID",
            help=(
                "Model of the LLM judges for this run (models.roles.evaluation): "
                "the pairwise judge and the langchain/ragas evaluators"
            ),
        )
        parser.add_argument(
            "--output-directory",
            type=str,
            help=(
                "Directory to save evaluation results (default: "
                "evaluation.outputs_directory). Accepts a local path or an "
                "s3://bucket/prefix/ URI (results are uploaded after the run; "
                "a failed upload exits non-zero)."
            ),
        )
        add_query_arguments(parser, multiple_strategies=True)
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
            help="Enable detailed logging output for debugging.",
        )
        add_config_arguments(parser, "Path to custom configuration file")
        add_output_format_argument(parser)
        add_provider_arguments(parser)
        return parser

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


class EvaluationRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        rag_chain: Runnable | None,
        config: Config,
        providers: Providers | None = None,
    ) -> None:
        """``config`` is the run's configuration, provider flags applied (the
        one ``rag_chain`` was built from). ``rag_chain`` is ``None`` with
        ``--from-results`` (nothing is answered); ``providers`` builds the
        pairwise judge then (default: from ``config``)."""
        self.args = args
        self.config = with_judge_model(config, getattr(args, "judge_model", None))
        self.rag_chain = rag_chain
        self.providers = providers
        self.json_output = json_output(args)
        self.evaluation_manager: EvaluationManager | None = None
        self._validate_args()

    def _output_directory(self) -> str:
        return str(
            self.args.output_directory or self.config.evaluation.outputs_directory
        )

    @property
    def _from_results(self) -> list[Path] | None:
        return getattr(self.args, "from_results", None)

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
        judges = {EvaluatorType.LANGCHAIN, EvaluatorType.RAGAS}
        if self._from_results is None and judges & set(
            self.config.evaluation.enabled_evaluators
        ):
            return []
        return [
            "--judge-model is given without --pairwise-judge and no LLM-judge "
            "evaluator (langchain, ragas) scores this run, so it judges "
            "nothing."
        ]

    def _output_is_baseline(self) -> bool:
        """Whether this run saves its results where --baseline-results reads
        the baseline from (the next run reading it pools them in)."""
        target = self._output_directory()
        if self._from_results is not None or is_s3_uri(target):
            return False
        out = Path(target).resolve()
        return any(
            (path if path.is_dir() else path.parent).resolve() == out
            for path in self._baseline_results or []
        )

    def _uses_llm_judge(self) -> bool:
        """Whether this run calls an LLM judge: --pairwise-judge, or a
        langchain/ragas evaluator scoring answers it generates."""
        if getattr(self.args, "pairwise_judge", False):
            return True
        judges = {EvaluatorType.LANGCHAIN, EvaluatorType.RAGAS}
        return self._from_results is None and bool(
            judges & set(self.config.evaluation.enabled_evaluators)
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
        for warning in self._judge_model_warnings():
            logger.warning("%s", warning)
        if self._uses_llm_judge():
            judge, reason = self._judge_choice()
            logger.info("LLM judge: %s (%s)", judge, reason)
        if self._from_results is not None:
            if self.args.repeats != 1:
                usage_error(
                    "--repeats does not apply to --from-results (the saved runs "
                    "are the repeats)"
                )
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
            table.add_column("Metric")
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
        table.add_column("Metric")
        for strategy in strategies:
            table.add_column(strategy, style="green")
        for strategy, label in pairs.items():
            table.add_column(f"{strategy} - {label} [CI]", style="cyan")
        for key in baseline_metric_keys(comparison):
            row = [markup_safe(key)]
            for strategy in strategies:
                stats = strategies[strategy]["metrics"].get(key)
                row.append(f"{stats['mean']:.3f}" if stats else "-")
            for strategy in pairs:
                delta = comparison["deltas_vs_baseline"][strategy].get(key)
                row.append(
                    f"{delta['mean_delta']:+.3f} "
                    f"[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}]"
                    if delta
                    else "-"
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
                f"[cyan]Pairwise judge, {strategy} win rate vs "
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

    def _local_outputs(self, work_dir: Path) -> Path:
        """Where the result files are written (``work_dir`` for an S3 target)."""
        target = self._output_directory()
        return work_dir / "outputs" if is_s3_uri(target) else Path(target)

    def _upload_finished_runs(self, work_dir: Path) -> bool:
        """Upload what an interrupted run already wrote to its S3 target.

        Returns whether ``work_dir`` may go: nothing to upload, or uploaded.
        """
        target = self._output_directory()
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
        outputs_target = self._output_directory()
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
            measured = self._saved_runs_input()
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
                f"[red]Failure budget exceeded ({markup_safe('; '.join(breaches))}; allowed "
                f"failure rate: {self.args.max_failure_rate}).[/red]"
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
                }
            )
        return exit_code, upload_error is not None

    def _load_baseline(self) -> dict[str, list[SavedRun]] | None:
        """The ``--baseline-results`` runs by strategy (in the order the
        earlier run listed them), labelled ``baseline:<strategy>``; checked
        against this run's answer scoring before any answer is generated."""
        if not self._baseline_results:
            return None
        try:
            saved = load_saved_runs(self._baseline_results)
            grouped = group_runs(saved)
            if self._from_results is None:
                # Refused now rather than after paying for the answers.
                check_answer_scoring(
                    [
                        *(("baseline", run.manifest) for run in saved),
                        ("this run", {"answer_scoring": self._answer_scoring()}),
                    ]
                )
        except EvaluationException as e:
            raise InvalidDatasetError(f"--baseline-results: {e}") from e
        return {f"baseline:{strategy}": runs for strategy, runs in grouped.items()}

    def _answer_scoring(self) -> dict[str, Any] | None:
        """The answer_scoring this run's manifests will record."""
        evaluation = self.config.evaluation
        if EvaluatorType.ANSWER_MATCH not in evaluation.enabled_evaluators:
            return None
        return {
            "answer_span": evaluation.answer_span,
            "answer_normalization": evaluation.answer_normalization,
        }

    def _saved_runs_input(self) -> "_Measured":
        """The comparison input rebuilt from ``--from-results``."""
        try:
            saved = load_saved_runs(self._from_results or [])
            grouped = group_runs(
                saved,
                list(self.args.search_strategy) if self._strategies_given() else None,
            )
        except EvaluationException as e:
            raise InvalidDatasetError(f"--from-results: {e}") from e
        warnings = [
            *summary_warnings(saved, "--from-results"),
            *pooling_warnings(grouped),
        ]
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

    async def _answer_and_score(
        self,
        s3_client: Any,
        work_dir: Path,
        local_outputs: Path,
        baseline: dict[str, list[SavedRun]] | None,
    ) -> "_Measured":
        """Answer and score the dataset once per strategy and repeat."""
        assert self.rag_chain is not None
        eval_data_uri = self.args.data_path
        outputs_target = self._output_directory()
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
        comparing = len(strategies) > 1 or self.args.repeats > 1 or bool(baseline)
        try:
            queries, ground_truths = self.evaluation_manager.load_data(
                eval_data_path=eval_data_path,
                base_metadata={
                    "suffix": query_suffix(self.args, self.config),
                    "enable_thinking": self.args.enable_thinking,
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

        runs: dict[str, list[Run]] = {}
        summaries: list[dict[str, Any]] = []
        breaches: list[str] = []
        manifest: dict[str, Any] | None = None
        manifests: list[tuple[str, dict[str, Any] | None]] = []
        for strategy in strategies:
            for repeat in range(1, self.args.repeats + 1):
                # The run's strategy overrides a per-item one: every run of a
                # comparison must answer the same queries the same way.
                run_queries = (
                    [
                        q.model_copy(
                            update={
                                "metadata": {**q.metadata, "search_strategy": strategy}
                            }
                        )
                        for q in queries
                    ]
                    if comparing
                    else queries
                )
                label = (
                    f" ({strategy}, repeat {repeat}/{self.args.repeats})"
                    if comparing
                    else ""
                )
                console.rule(
                    f"[bold]Running Evaluation on {len(queries)} Queries{label}[/bold]",
                    style="blue",
                )
                if repeat == 1 and self.config.evaluation.warm_up_query and queries:
                    # Untimed: the first strategy must not pay for warming
                    # the connections and models the others then reuse.
                    await self.evaluation_manager.warm_up(run_queries[0])
                run_start = time.time()
                # On the run's root spans (kg_rag.batch, kg_rag.evaluate), so
                # the runs of a comparison tell apart.
                with observability.bind_attributes(
                    **{"eval.strategy": strategy, "eval.repeat": repeat}
                ):
                    results, reports, summary = (
                        await self.evaluation_manager.evaluate_dataset(
                            queries=run_queries,
                            ground_truths=ground_truths,
                            show_progress=True,
                            dataset_path=eval_data_path,
                            cli_args=vars(self.args),
                        )
                    )
                if is_s3_uri(eval_data_uri) and summary.run_manifest:
                    # Record where the dataset came from, not the temporary copy.
                    summary.run_manifest.get("dataset", {})["path"] = eval_data_uri
                # The directory --output-directory chose, not the config default.
                summary.configuration["outputs_directory"] = outputs_target
                self.evaluation_manager.save_results(
                    results=results,
                    reports=reports,
                    summary=summary,
                    outputs_dir=local_outputs,
                )
                manifest = manifest or summary.run_manifest
                manifests.append((strategy, summary.run_manifest))
                runs.setdefault(strategy, []).append((reports, results))
                summaries.append(
                    {
                        "strategy": strategy,
                        "repeat": repeat,
                        **summary.model_dump(mode="json", include=_SUMMARY_FIELDS),
                    }
                )
                if not comparing:
                    self._print_summary(summary, time.time() - run_start)
                breaches.extend(
                    f"{strategy}#{repeat}: {b}" if comparing else b
                    for b in failure_budget_breaches(
                        summary, self.args.max_failure_rate
                    )
                )
        return _Measured(
            runs=runs,
            manifests=manifests,
            manifest=manifest,
            categories={q.query_id: q.category for q in queries if q.category},
            summaries=summaries,
            breaches=breaches,
            comparing=comparing,
        )

    def _baseline_pairs(
        self, measured: "_Measured", baseline: dict[str, list[SavedRun]]
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
        measured: "_Measured",
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
        names = list(runs)
        if pairs:
            comparison = baseline_comparison(
                {label: runs[label] for label in dict.fromkeys(pairs.values())},
                measured.runs,
                pairs,
                categories=measured.categories,
                prices=prices,
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
        else:
            comparison = paired_comparison(
                runs, categories=measured.categories, prices=prices
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
            with usage_scope() as judging:
                comparison["pairwise_vs_baseline"] = {
                    strategy: await judge.acompare(
                        runs[label][0][1], runs[strategy][0][1]
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

    def _judge_manifest(self, measured: "_Measured") -> dict[str, Any] | None:
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
    "usage",
}


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

        with exit_on_config_error():
            config = apply_provider_arguments(
                load_cli_config(args),
                args,
            )
        setup_logging(config, stream=sys.stderr if json_output(args) else None)
        if args.verbose:
            logging.getLogger("unified_kg_rag").setLevel(logging.DEBUG)

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

    except ConfigurationError as e:
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
