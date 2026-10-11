# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import shutil
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import nest_asyncio
from langchain_core.runnables import Runnable
from rich.panel import Panel
from rich.table import Table

from unified_kg_rag.adapters.aws.s3_transfer import (
    download_s3_object,
    is_s3_uri,
    parse_s3_uri,
    s3_client_for,
    upload_directory_to_s3,
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
from unified_kg_rag.application.cli.query_args import add_query_arguments
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    EvaluationGroundTruth,
    EvaluationSummary,
    SearchStrategy,
)
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    get_config,
    get_logger,
    setup_logging,
)
from unified_kg_rag.shared.utils import event_loop
from unified_kg_rag.shared.utils.display import console, display_ascii_art

nest_asyncio.apply()
logger = get_logger(__name__)

try:
    __version__ = version("unified-kg-rag-on-aws")
except (FileNotFoundError, ImportError, ValueError):
    __version__ = "unknown"


def _failure_rate(value: str) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return rate


def _eval_data_path(value: str) -> str | Path:
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

    def _over(failed: int, total: int) -> bool:
        return total == 0 or failed >= total or failed / total > max_failure_rate

    breaches = []
    if _over(summary.failed_evaluations, summary.total_queries):
        breaches.append(
            f"{summary.failed_evaluations}/{summary.total_queries} queries failed"
        )
    for evaluator_name, metrics in summary.metric_outcomes.items():
        for metric_name, counts in metrics.items():
            failed = counts.get("failed", 0)
            attempted = counts.get("scored", 0) + failed
            if failed and _over(failed, attempted):
                breaches.append(
                    f"{evaluator_name}/{metric_name}: {failed}/{attempted} failed"
                )
    return breaches


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="GraphRAG Evaluation System - Assess retrieval and generation performance",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=doc_epilog(
                "6-evaluation-run-eval",
                "run-eval --eval-data-path eval_data.json --config-path config.yaml",
            ),
        )

        parser.add_argument(
            "--eval-data-path",
            type=_eval_data_path,
            required=True,
            help=(
                "Path to the unified evaluation data file (JSON format), containing "
                "both questions and ground truths. Accepts a local path or an "
                "s3://bucket/key URI (downloaded before the run)."
            ),
        )
        parser.add_argument(
            "--outputs-directory",
            type=str,
            help=(
                "Directory to save evaluation results (default: "
                "evaluation.outputs_directory). Accepts a local path or an "
                "s3://bucket/prefix/ URI (results are uploaded after the run; "
                "a failed upload exits non-zero)."
            ),
        )
        add_query_arguments(parser)
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
        parser.add_argument(
            "--config-path",
            type=str,
            help="Path to custom configuration file",
        )
        add_provider_arguments(parser)
        return parser

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


class EvaluationRunner:
    def __init__(self, args: argparse.Namespace, rag_chain: Runnable) -> None:
        self.args = args
        self.config = apply_provider_arguments(
            get_config(Path(args.config_path) if args.config_path else None), args
        )
        self.rag_chain = rag_chain
        self.evaluation_manager: EvaluationManager | None = None
        self._validate_args()

    def _outputs_directory(self) -> str:
        return str(
            self.args.outputs_directory or self.config.evaluation.outputs_directory
        )

    def _validate_args(self) -> None:
        eval_data_path = self.args.eval_data_path
        # Malformed S3 URIs fail here, before any model or store is touched.
        try:
            if is_s3_uri(eval_data_path):
                parse_s3_uri(eval_data_path, require_key=True)
            if is_s3_uri(self._outputs_directory()):
                parse_s3_uri(self._outputs_directory(), require_key=False)
        except ConfigurationError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        if is_s3_uri(eval_data_path):
            return
        if eval_data_path and not Path(eval_data_path).exists():
            console.print(
                f"[red]Error: Evaluation data file not found: '{eval_data_path}'[/red]"
            )
            sys.exit(1)

    @staticmethod
    def _print_summary(
        summary: EvaluationSummary, total_time: float, outputs_directory: str
    ) -> None:
        summary_text = (
            f"Total Queries: [bold]{summary.total_queries}[/bold]\n"
            f"Successful: [green]{summary.successful_evaluations}[/green]\n"
            f"Failed: [red]{summary.failed_evaluations}[/red]\n"
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

        console.print(
            f"\n[bold green]Results saved to '{outputs_directory}'[/bold green]"
        )

    async def run(self) -> int:
        """Evaluate; 0 on success, 1 on a breached failure budget or failed upload.

        An ``s3://`` dataset is downloaded to a temporary directory first; an
        ``s3://`` outputs directory is written locally and then uploaded.
        """
        work_dir = Path(tempfile.mkdtemp(prefix="graphrag-eval-"))
        keep_work_dir = False
        try:
            exit_code, keep_work_dir = await self._run(work_dir)
            return exit_code
        finally:
            if not keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    async def _run(self, work_dir: Path) -> tuple[int, bool]:
        """Run the evaluation; returns ``(exit code, keep work_dir)``."""
        eval_data_uri = self.args.eval_data_path
        outputs_target = self._outputs_directory()
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

        queries, ground_truths = self.evaluation_manager.load_data(
            eval_data_path=eval_data_path,
            base_metadata={
                "suffix": self.args.suffix,
                "enable_thinking": self.args.enable_thinking,
                "search_strategy": self.args.search_strategy,
                "search_type": self.args.search_type,
                "top_k": self.args.top_k,
                "retrieval_multiplier": self.args.retrieval_multiplier,
            },
        )

        if not ground_truths:
            logger.warning(
                "No ground truth data found in file; some evaluators may not work."
            )
            ground_truths = [
                EvaluationGroundTruth(query_id=q.query_id, ground_truth="")
                for q in queries
            ]

        console.rule(
            f"[bold]Running Evaluation on {len(queries)} Queries[/bold]", style="blue"
        )

        start_time = time.time()
        results, reports, summary = await self.evaluation_manager.evaluate_dataset(
            queries=queries,
            ground_truths=ground_truths,
            show_progress=True,
            dataset_path=eval_data_path,
            cli_args=vars(self.args),
        )
        total_time = time.time() - start_time
        if is_s3_uri(eval_data_uri) and summary.run_manifest:
            # Record where the dataset came from, not the temporary copy.
            summary.run_manifest.get("dataset", {})["path"] = eval_data_uri

        local_outputs = (
            work_dir / "outputs" if is_s3_uri(outputs_target) else Path(outputs_target)
        )
        self.evaluation_manager.save_results(
            results=results,
            reports=reports,
            summary=summary,
            outputs_dir=local_outputs,
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
        self._print_summary(
            summary,
            total_time,
            str(local_outputs) if upload_error else outputs_target,
        )

        exit_code = 0
        if upload_error:
            # Keep the local copy so a failed upload loses no results.
            console.print(
                f"[red]Error: {upload_error}. Local results kept in "
                f"'{local_outputs}'.[/red]"
            )
            exit_code = 1

        breaches = failure_budget_breaches(summary, self.args.max_failure_rate)
        if breaches:
            console.print(
                f"[red]Failure budget exceeded ({'; '.join(breaches)}; allowed "
                f"failure rate: {self.args.max_failure_rate}).[/red]"
            )
            exit_code = 1
        return exit_code, upload_error is not None


def main() -> None:
    try:
        cli = CommandLineInterface()
        args = cli.parse_args()

        config = apply_provider_arguments(
            get_config(Path(args.config_path) if args.config_path else None), args
        )
        setup_logging(config)
        if args.verbose:
            logging.getLogger("unified_kg_rag").setLevel(logging.DEBUG)

        error = missing_endpoints_error(
            config,
            strategy_roles(config, SearchStrategy(args.search_strategy)),
            f"search strategy '{args.search_strategy}'",
        )
        if error:
            console.print(f"[red]Error: {error}[/red]")
            sys.exit(1)

        rag_chain = GraphRAGChain(config)

        async def _run_and_close() -> int:
            # Release the retrievers' Neptune/OpenSearch sockets on every exit
            # path, on the loop they were opened on.
            try:
                return await EvaluationRunner(args, rag_chain).run()
            finally:
                await rag_chain.aclose()

        exit_code = event_loop.run(
            _run_and_close(), io_workers=config.processing.io_workers
        )
        if exit_code:
            sys.exit(exit_code)

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user.[/yellow]")
        sys.exit(130)

    except (AWSServiceError, ConfigurationError) as e:
        # e.g. an s3:// dataset that cannot be downloaded.
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error("Evaluation failed: %s", e)
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {e}[/red]")
        logger.exception("Unexpected error during evaluation")
        sys.exit(1)


if __name__ == "__main__":
    main()
