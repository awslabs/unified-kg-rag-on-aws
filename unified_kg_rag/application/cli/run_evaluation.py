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

import nest_asyncio
from langchain_core.runnables import Runnable
from rich.markup import escape
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
from unified_kg_rag.application.cli.config_args import (
    add_config_arguments,
    exit_on_config_error,
    load_cli_config,
)
from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    add_output_format_argument,
    exits_on_sigterm,
    json_output,
    print_json,
    usage_error,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    strategy_roles,
)
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.application.cli.query_args import (
    add_query_arguments,
    positive_int,
)
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationSummary,
    SearchStrategy,
)
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.evaluation.comparison import Run, paired_comparison
from unified_kg_rag.evaluation.pairwise import PairwiseJudge
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    DataProcessingError,
    EvaluationException,
    get_logger,
    setup_logging,
)
from unified_kg_rag.shared.config import load_cli_dotenv
from unified_kg_rag.shared.utils import event_loop
from unified_kg_rag.shared.utils.display import console, display_ascii_art

nest_asyncio.apply()
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


def _over(failed: int, total: int, max_failure_rate: float) -> bool:
    return total == 0 or failed >= total or failed / total > max_failure_rate


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            allow_abbrev=False,
            description="GraphRAG Evaluation System - Assess retrieval and generation performance",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=doc_epilog(
                "6-evaluation-run-eval",
                "run-eval --data-path eval_data.json --config-path config.yaml",
            ),
        )

        parser.add_argument(
            "--data-path",
            type=_data_path,
            required=True,
            help=(
                "Path to the unified evaluation data file (JSON format), containing "
                "both questions and ground truths. Accepts a local path or an "
                "s3://bucket/key URI (downloaded before the run)."
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
        self, args: argparse.Namespace, rag_chain: Runnable, config: Config
    ) -> None:
        """``config`` is the run's configuration, provider flags applied (the
        one ``rag_chain`` was built from)."""
        self.args = args
        self.config = config
        self.rag_chain = rag_chain
        self.json_output = json_output(args)
        self.evaluation_manager: EvaluationManager | None = None
        self._validate_args()

    def _output_directory(self) -> str:
        return str(
            self.args.output_directory or self.config.evaluation.outputs_directory
        )

    def _validate_args(self) -> None:
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
            # Which benchmark was scored: musique-full-answerable is not
            # comparable to MuSiQue-Ans.
            named = ", ".join(f"{name} ({count})" for name, count in labels.items())
            summary_text += f"Benchmark: [bold]{named}[/bold]\n"
        if blocked := summary.guardrail_blocked_counts:
            # Scored on the blocked message: a low score may be the guardrail.
            per_text = ", ".join(f"{text} {count}" for text, count in blocked.items())
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
                    metric,
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
                        f"[yellow]{evaluator_name}/{metric_name}: "
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
        """Mean per strategy and the paired delta (95% CI) against the baseline."""
        baseline = comparison["baseline"]
        strategies = comparison["strategies"]
        table = Table(
            title=(
                f"[bold]Paired comparison vs '{baseline}' "
                f"({total_time:.0f}s)[/bold]"
            ),
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric")
        for strategy in strategies:
            table.add_column(strategy, style="green")
        for strategy in list(strategies)[1:]:
            table.add_column(f"{strategy} - {baseline} [CI]", style="cyan")
        for key in strategies[baseline]["metrics"]:
            row = [key]
            for strategy in strategies:
                stats = strategies[strategy]["metrics"].get(key)
                row.append(f"{stats['mean']:.3f}" if stats else "-")
            for strategy in list(strategies)[1:]:
                delta = comparison["deltas_vs_baseline"][strategy].get(key)
                row.append(
                    f"{delta['mean_delta']:+.3f} "
                    f"[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}]"
                    if delta
                    else "-"
                )
            table.add_row(*row)
        console.print(table)
        for strategy, stats in strategies.items():
            routed = stats["routed_strategies"]
            if routed and list(routed) != [strategy]:
                console.print(f"[dim]{strategy} routed to: {routed}[/dim]")
        for warning in comparison.get("warnings") or []:
            console.print(f"[yellow]Warning: {escape(warning)}[/yellow]")
        for strategy, judged in (comparison.get("pairwise_vs_baseline") or {}).items():
            rates = ", ".join(
                f"{criterion} {stats['candidate_win_rate']:.2f}"
                for criterion, stats in judged["criteria"].items()
                if stats["candidate_win_rate"] is not None
            )
            errors = judged["candidate_errors"] + judged["baseline_errors"]
            console.print(
                f"[cyan]Pairwise judge, {strategy} win rate vs {baseline} "
                f"({judged['judged']} judged, {errors} decided by a failed "
                f"answer, {judged['failed']} judge failures): {rates}[/cyan]"
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
        eval_data_uri = self.args.data_path
        outputs_target = self._output_directory()
        s3_client = (
            s3_client_for(self.config)
            if is_s3_uri(eval_data_uri) or is_s3_uri(outputs_target)
            else None
        )
        eval_data_path: str | Path = eval_data_uri
        if s3_client is not None and is_s3_uri(eval_data_uri):
            eval_data_path = download_s3_object(
                s3_client, eval_data_uri, work_dir / "dataset" / "eval_data.json"
            )

        display_ascii_art(version=__version__)
        console.rule("[bold]Initializing Evaluation[/bold]", style="blue")

        self.evaluation_manager = EvaluationManager(
            config=self.config, rag_chain=self.rag_chain
        )

        strategies: list[str] = list(self.args.search_strategy)
        comparing = len(strategies) > 1 or self.args.repeats > 1
        try:
            queries, ground_truths = self.evaluation_manager.load_data(
                eval_data_path=eval_data_path,
                base_metadata={
                    "suffix": self.args.suffix,
                    "enable_thinking": self.args.enable_thinking,
                    # A comparison sets each run's strategy itself (below).
                    "search_strategy": None if comparing else strategies[0],
                    "search_type": self.args.search_type,
                    "top_k": self.args.top_k,
                    "retrieval_multiplier": self.args.retrieval_multiplier,
                },
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

        local_outputs = self._local_outputs(work_dir)
        runs: dict[str, list[Run]] = {}
        summaries: list[dict[str, Any]] = []
        comparison: dict[str, Any] | None = None
        breaches: list[str] = []
        manifest: dict[str, Any] | None = None
        start_time = time.time()
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
                run_start = time.time()
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
                runs.setdefault(strategy, []).append((reports, results))
                summaries.append(
                    {
                        "strategy": strategy,
                        "repeat": repeat,
                        **summary.model_dump(
                            mode="json",
                            include={
                                "total_queries",
                                "successful_evaluations",
                                "failed_evaluations",
                                "benchmarks",
                                "average_response_time",
                                "metric_statistics",
                                "metric_outcomes",
                                "guardrail_blocked_counts",
                            },
                        ),
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

        if comparing:
            comparison = paired_comparison(runs)
            if self.args.pairwise_judge and len(strategies) > 1:
                judge = PairwiseJudge(
                    self.config, providers=self.evaluation_manager.providers
                )
                baseline_results = runs[strategies[0]][0][1]
                comparison["pairwise_vs_baseline"] = {
                    strategy: await judge.acompare(
                        baseline_results, runs[strategy][0][1]
                    )
                    for strategy in strategies[1:]
                }
                breaches.extend(
                    pairwise_budget_breaches(
                        comparison["pairwise_vs_baseline"],
                        self.args.max_failure_rate,
                    )
                )
            comparison["run_manifest"] = manifest
            self._save_comparison(comparison, local_outputs)
            self._print_comparison(comparison, time.time() - start_time)

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
            f"'{local_outputs if upload_error else outputs_target}'[/bold green]"
        )

        exit_code = 0
        if upload_error:
            # Keep the local copy so a failed upload loses no results.
            console.print(
                f"[red]Error: {upload_error}. Local results kept in "
                f"'{local_outputs}'.[/red]"
            )
            exit_code = EXIT_FAILURE

        if breaches:
            console.print(
                f"[red]Failure budget exceeded ({'; '.join(breaches)}; allowed "
                f"failure rate: {self.args.max_failure_rate}).[/red]"
            )
            exit_code = EXIT_FAILURE
        if self.json_output:
            print_json(
                {
                    "output_directory": str(
                        local_outputs if upload_error else outputs_target
                    ),
                    "upload_error": str(upload_error) if upload_error else None,
                    "runs": summaries,
                    "comparison": comparison,
                    "failure_budget_breaches": breaches,
                }
            )
        return exit_code, upload_error is not None


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

        error = missing_endpoints_error(
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
            _run_and_close(), io_workers=config.processing.io_workers
        )
        if exit_code:
            sys.exit(exit_code)

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)

    except InvalidDatasetError as e:
        # An input error, like a usage error: no traceback.
        console.print(
            f"\n[red]Error: invalid evaluation dataset: {escape(str(e))}[/red]"
        )
        logger.error("Invalid evaluation dataset: %s", e)
        sys.exit(EXIT_USAGE)

    except ConfigurationError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error("Evaluation configuration error: %s", e)
        sys.exit(EXIT_USAGE)

    except AWSServiceError as e:
        # e.g. an s3:// dataset that cannot be downloaded.
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error("Evaluation failed: %s", e)
        sys.exit(EXIT_FAILURE)

    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {e}[/red]")
        logger.exception("Unexpected error during evaluation")
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()
