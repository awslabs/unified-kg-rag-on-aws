# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import argparse
import asyncio
import logging
import sys
import time
from importlib.metadata import version
from pathlib import Path

import nest_asyncio
from langchain.schema.runnable import Runnable
from rich.panel import Panel
from rich.table import Table

from aws_graphrag.core import get_config, get_logger
from aws_graphrag.domain.models import (
    EvaluationGroundTruth,
    EvaluationSummary,
    SearchStrategy,
    SearchType,
)
from aws_graphrag.evaluation import EvaluationManager
from aws_graphrag.retrieval import GraphRAGChain
from aws_graphrag.utils import console, display_ascii_art

nest_asyncio.apply()
logger = get_logger(__name__)

try:
    __version__ = version("aws-graphrag")
except (FileNotFoundError, ImportError, ValueError):
    __version__ = "unknown"


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="GraphRAG Evaluation System - Assess retrieval and generation performance",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        parser.add_argument(
            "--eval-data-path",
            type=Path,
            required=True,
            help="Path to the unified evaluation data file (JSON format), containing both questions and ground truths.",
        )
        parser.add_argument(
            "--outputs-directory",
            type=str,
            help="Directory to save evaluation results",
        )
        parser.add_argument(
            "--suffix",
            help="Suffix for multi-tenant or versioned indices",
        )
        parser.add_argument(
            "--enable-thinking",
            action="store_true",
            help="Enable thinking mode for language model reasoning and step-by-step problem solving",
        )
        parser.add_argument(
            "--search-strategy",
            default="auto",
            choices=[ss.value for ss in SearchStrategy],
            help="Choose the search strategy",
        )
        parser.add_argument(
            "--search-type",
            default="hybrid",
            choices=[st.value for st in SearchType],
            help="Specify the search method",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=10,
            help="Set the maximum number of search results",
        )
        parser.add_argument(
            "--retrieval-multiplier",
            type=int,
            default=1,
            help="Set the retrieval multiplier for increasing search depth",
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
        return parser

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


class EvaluationRunner:
    def __init__(self, args: argparse.Namespace, rag_chain: Runnable) -> None:
        self.args = args
        self.config = get_config(Path(args.config_path) if args.config_path else None)
        self.rag_chain = rag_chain
        self.evaluation_manager: EvaluationManager | None = None
        self._validate_args()

    def _validate_args(self) -> None:
        eval_data_path = self.args.eval_data_path
        if eval_data_path and not Path(eval_data_path).exists():
            console.print(
                f"[red]Error: Evaluation data file not found: '{eval_data_path}'[/red]"
            )
            sys.exit(1)

    @staticmethod
    def _print_summary(
        summary: EvaluationSummary, total_time: float, outputs_directory: Path
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

        console.print(
            f"\n[bold green]Results saved to '{outputs_directory}'[/bold green]"
        )

    async def run(self) -> None:
        display_ascii_art(version=__version__)
        console.rule("[bold]Initializing Evaluation[/bold]", style="blue")

        self.evaluation_manager = EvaluationManager(
            config=self.config, rag_chain=self.rag_chain
        )

        queries, ground_truths = self.evaluation_manager.load_data(
            eval_data_path=self.args.eval_data_path,
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
            queries=queries, ground_truths=ground_truths, show_progress=True
        )
        total_time = time.time() - start_time

        outputs_directory = (
            self.args.outputs_directory or self.config.evaluation.outputs_directory
        )
        self.evaluation_manager.save_results(
            results=results,
            reports=reports,
            summary=summary,
            outputs_dir=outputs_directory,
        )

        console.rule("[bold]Evaluation Complete[/bold]", style="blue")
        self._print_summary(summary, total_time, Path(outputs_directory))


def main() -> None:
    try:
        cli = CommandLineInterface()
        args = cli.parse_args()

        if args.verbose:
            logging.getLogger("aws_graphrag").setLevel(logging.DEBUG)

        rag_chain = GraphRAGChain(
            get_config(Path(args.config_path) if args.config_path else None)
        )
        runner = EvaluationRunner(args, rag_chain)
        asyncio.run(runner.run())

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user.[/yellow]")
        sys.exit(130)

    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {e}[/red]")
        logger.exception("Unexpected error during evaluation")
        sys.exit(1)


if __name__ == "__main__":
    main()
