# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm

from unified_kg_rag.application.cli.config_args import (
    config_path_arg,
    exit_on_config_error,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.preflight import missing_endpoints_error
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    RetrieverRole,
    validate_path_segment,
)
from unified_kg_rag.shared import (
    CloudWatchEMFSink,
    ConfigurationError,
    MetricsSink,
    NullMetricsSink,
    PipelineExecutionError,
    get_config,
    get_logger,
    setup_logging,
)
from unified_kg_rag.shared.config import load_cli_dotenv
from unified_kg_rag.shared.utils.display import (
    console,
    display_ascii_art,
    display_pipeline_results,
)

logger = get_logger(__name__)


def _pipeline_id_arg(value: str) -> str | None:
    if not value:  # an empty GRAPHRAG_PIPELINE_ID means "unset"
        return None
    try:
        return validate_path_segment(value, "pipeline_id")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _stage_name_arg(value: str) -> str:
    """A pipeline stage name, case-insensitive; unknown names are a usage error."""
    try:
        normalized = DataIngestionPipeline.normalize_stage_name(value)
    except ConfigurationError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    assert normalized is not None
    return normalized


try:
    __version__ = version("unified-kg-rag-on-aws")
except (FileNotFoundError, ImportError, ValueError):
    __version__ = "unknown"


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="GraphRAG Data Ingestion Pipeline - Process documents and build knowledge graphs",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=doc_epilog(
                "3-ingestion-run-ingestion",
                "run-ingestion --source-directory ./source --config-path config.yaml",
            ),
        )

        parser.add_argument(
            "--source-directory",
            type=str,
            default=os.getenv("GRAPHRAG_SOURCE_DIRECTORY"),
            help="Path to directory containing source documents to be processed "
            "(falls back to the GRAPHRAG_SOURCE_DIRECTORY env var)",
        )
        parser.add_argument(
            "--target-directory",
            type=str,
            help="Export each parsed document as <stem>.json to this directory "
            "for inspection (no export when omitted; must not be the source "
            "directory)",
        )
        parser.add_argument(
            "--cache-directory",
            type=str,
            help="Path to directory for storing pipeline cache and intermediate results",
        )
        parser.add_argument(
            "--force-rebuild",
            action="store_true",
            help="Recompute every stage, ignoring existing cached stage outputs "
            "(the stage cache is rewritten with the new outputs)",
        )
        parser.add_argument(
            "--s3-sync",
            action="store_true",
            help="Enable automatic synchronization of cache data with Amazon S3",
        )
        parser.add_argument(
            "--s3-bucket-name",
            type=str,
            help="Amazon S3 bucket name for cache synchronization (required when using --s3-sync)",
        )
        parser.add_argument(
            "--s3-prefix",
            type=str,
            default="pipeline-runs",
            help="S3 object key prefix for organizing cache files (default: pipeline-runs)",
        )
        parser.add_argument(
            "--pipeline-id",
            # argparse applies ``type`` to a string default too, so the env
            # var (set by the Step Functions task) is validated as well.
            type=_pipeline_id_arg,
            default=os.getenv("GRAPHRAG_PIPELINE_ID"),
            help="Unique identifier of an existing pipeline run to resume or inspect "
            "(falls back to the GRAPHRAG_PIPELINE_ID env var)",
        )
        parser.add_argument(
            "--resume-from-stage",
            type=_stage_name_arg,
            help="Specific pipeline stage to resume execution from, by name "
            "(case-insensitive; requires --pipeline-id)",
        )
        parser.add_argument(
            "--verify-metadata",
            action="store_true",
            help="Verify the integrity and consistency of pipeline metadata files",
        )
        parser.add_argument(
            "--yes",
            "-y",
            action="store_true",
            help="Answer yes to confirmation prompts (continuing after a failed "
            "integrity check). Without it, a prompt that cannot be asked "
            "because stdin is not a terminal fails the run",
        )
        parser.add_argument(
            "--continue-on-error",
            action="store_true",
            help="Continue pipeline execution even when individual stages encounter errors",
        )
        parser.add_argument(
            "--enabled-stages",
            type=lambda s: [item.strip() for item in s.split(",")],
            help="Comma-separated list of pipeline stages to enable (e.g., 'DOCUMENT_PARSING,TEXT_CHUNKING'). If not specified, all stages are enabled.",
        )
        parser.add_argument(
            "--metrics-sink",
            type=str,
            choices=["none", "cloudwatch"],
            default="none",
            help="Where to forward pipeline metrics: 'none' (default) or "
            "'cloudwatch' (CloudWatch EMF to stdout, auto-extracted by CloudWatch Logs)",
        )
        parser.add_argument(
            "--retire-source-scope",
            action="append",
            default=[],
            metavar="SCOPE",
            help="Incremental runs: treat every doc-status record of this "
            "former source scope (in this run's index namespace) as deleted, "
            "removing its exclusive graph and vector content. Use it once "
            "after moving a corpus, with the old absolute source directory or "
            "former document_parsing.source_scope value. Repeatable; added to "
            "indexing.retire_source_scopes. Never this run's own scope.",
        )
        parser.add_argument(
            "--config-path",
            type=config_path_arg,
            help="Path to custom configuration file",
        )
        add_provider_arguments(parser)
        return parser

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


class IngestionPipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        with exit_on_config_error():
            self.config = apply_provider_arguments(
                get_config(Path(args.config_path) if args.config_path else None),
                args,
            )
        retired = getattr(args, "retire_source_scope", None) or []
        if retired:
            indexing = self.config.indexing
            indexing.retire_source_scopes = list(
                dict.fromkeys(
                    [*indexing.retire_source_scopes, *(s.strip() for s in retired)]
                )
            )
        setup_logging(self.config)
        self.pipeline: DataIngestionPipeline | None = None
        self._validate_args()

    def _validate_args(self) -> None:
        if not self.args.verify_metadata and not self.args.source_directory:
            console.print(
                "[red]Error: --source-directory is required for pipeline execution.[/red]"
            )
            sys.exit(1)

        if self.args.source_directory:
            self.args.source_directory = Path(self.args.source_directory)
            if not self.args.source_directory.is_dir():
                console.print(
                    f"[red]Error: Source directory not found: {self.args.source_directory}[/red]"
                )
                sys.exit(1)
            target = (
                self.args.target_directory
                or self.config.processing.document_parsing.target_directory
            )
            if target and (
                Path(target).resolve() == self.args.source_directory.resolve()
            ):
                # Parsed .json written into the corpus is re-ingested as new
                # documents on the next run.
                console.print(
                    "[red]Error: --target-directory (document_parsing."
                    "target_directory) must not be the source directory "
                    f"'{self.args.source_directory}': the parsed .json files "
                    "would be ingested as new documents on the next run.[/red]"
                )
                sys.exit(1)

        if self.args.s3_sync and not self.args.s3_bucket_name:
            console.print(
                "[red]Error: --s3-bucket-name must be specified for S3 sync.[/red]"
            )
            sys.exit(1)

        if self.args.resume_from_stage and not self.args.pipeline_id:
            console.print(
                "[red]Error: --pipeline-id is required for --resume-from-stage.[/red]"
            )
            sys.exit(1)

        if self.args.verify_metadata and not self.args.pipeline_id:
            console.print(
                "[red]Error: --pipeline-id is required for metadata operations.[/red]"
            )
            sys.exit(1)

        self.args.cache_directory = (
            Path(self.args.cache_directory)
            if self.args.cache_directory
            else Path("cache")
        )
        self.args.cache_directory.mkdir(parents=True, exist_ok=True)

    def _display_run_info(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_lines = [
            "[bold blue]Graph RAG Document Processing Pipeline[/bold blue]",
            f"[dim]Timestamp:[/dim] '{timestamp}'",
            f"[dim]Cache Directory:[/dim] '{self.args.cache_directory.absolute()}'",
        ]

        if self.args.source_directory:
            info_lines.insert(
                1,
                f"[dim]Source Directory:[/dim] '{self.args.source_directory.absolute()}'",
            )

        info_lines.extend(
            [
                f"[dim]Force Rebuild:[/dim] {self.args.force_rebuild}",
                f"[dim]S3 Sync:[/dim] {self.args.s3_sync}",
                f"[dim]Continue on Error:[/dim] {self.args.continue_on_error}",
            ]
        )

        if self.args.pipeline_id:
            info_lines.append(f"[dim]Pipeline ID:[/dim] '{self.args.pipeline_id}'")
            if self.args.resume_from_stage:
                info_lines.append(
                    f"[dim]Resume From Stage:[/dim] '{self.args.resume_from_stage}'"
                )
            else:
                info_lines.append(
                    "[dim]Resume Mode:[/dim] 'Auto-detect Failed/Incomplete Stage'"
                )
        else:
            info_lines.append("[dim]Mode:[/dim] 'New Pipeline Run'")

        info_content = "\n".join(info_lines)
        console.print(Panel.fit(info_content, border_style="blue"))

    def _build_metrics_sink(self) -> MetricsSink:
        if self.args.metrics_sink == "cloudwatch":
            logger.info("Forwarding pipeline metrics to CloudWatch (EMF)")
            return CloudWatchEMFSink()
        return NullMetricsSink()

    def _create_pipeline_config(self) -> PipelineConfig:
        all_stages = {stage.name: stage for stage in PipelineStageType}

        if self.args.enabled_stages:
            enabled_stages_set = {
                stage_name.upper() for stage_name in self.args.enabled_stages
            }
            stages = {
                stage_enum: stage_name in enabled_stages_set
                for stage_name, stage_enum in all_stages.items()
            }
            invalid_stages = enabled_stages_set - set(all_stages.keys())
            if invalid_stages:
                console.print(
                    f"[red]Error: Invalid stage names provided: {', '.join(invalid_stages)}[/red]"
                )
                console.print(
                    f"[yellow]Available stages are: {', '.join(all_stages.keys())}[/yellow]"
                )
                sys.exit(1)
        else:
            stages = dict.fromkeys(PipelineStageType, True)

        return PipelineConfig(
            stages_enabled=stages,
            continue_on_error=self.args.continue_on_error,
            # force_rebuild ignores cached outputs on read (FORCE_REFRESH) but
            # still saves the recomputed ones, so the next run resumes from them.
            cache_enabled=True,
            local_directory=self.args.cache_directory,
            s3_sync_enabled=self.args.s3_sync,
            s3_bucket_name=self.args.s3_bucket_name,
            s3_prefix=self.args.s3_prefix,
            force_rebuild=self.args.force_rebuild,
            pipeline_id=self.args.pipeline_id,
            resume_from_stage=self.args.resume_from_stage,
        )

    def _confirm(self, question: str) -> bool:
        """Ask ``question``; ``--yes`` answers it, and without a terminal to
        ask on the run fails rather than blocking or reading EOF."""
        if getattr(self.args, "yes", False):
            console.print(f"{question} [dim](--yes)[/dim]")
            return True
        if not sys.stdin.isatty():
            console.print(
                f"[red]Error: {question} needs confirmation, but stdin is not a "
                "terminal. Re-run with --yes to proceed.[/red]"
            )
            raise PipelineExecutionError(
                f"Confirmation required ({question}); pass --yes"
            )
        return Confirm.ask(question)

    def _handle_metadata_operations(self) -> bool:
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")

        if not self.args.verify_metadata:
            return False

        console.print(
            f"\n[yellow]Verifying metadata for pipeline: '{self.args.pipeline_id}'[/yellow]"
        )
        if not self.pipeline.state_manager.pipeline_exists(self.args.pipeline_id):
            console.print(
                f"[red]✗ No metadata found for pipeline "
                f"'{self.args.pipeline_id}' under "
                f"'{self.args.cache_directory}'[/red]"
            )
            raise PipelineExecutionError(
                f"Pipeline '{self.args.pipeline_id}' not found"
            )
        if not self.pipeline.verify_pipeline_metadata(self.args.pipeline_id):
            console.print("[red]✗ Metadata file is corrupted or invalid[/red]")
            # Raise so main() exits non-zero: a scripted check must see the
            # corruption, not a success exit code.
            raise PipelineExecutionError(
                f"Metadata for pipeline '{self.args.pipeline_id}' is "
                "corrupted or invalid"
            )
        console.print("[green]✓ Metadata file is valid[/green]")
        return True

    def _execute_pipeline(self) -> PipelineContext:
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")

        console.print("\n[bold green]Starting pipeline execution...[/bold green]")
        try:
            return self.pipeline.run(
                self.args.source_directory,
                pipeline_id=self.args.pipeline_id,
                resume_from_stage=self.args.resume_from_stage,
            )
        except PipelineExecutionError:
            # main() prints the error itself, after these suggestions.
            if self.args.pipeline_id:
                console.print("\n[yellow]Recovery suggestions:[/yellow]")
                console.print("1. Check logs for specific error details.")
                console.print(
                    "2. Resume from a different stage using --resume-from-stage."
                )
                console.print("3. Use --verify-metadata to check for corruption.")
                console.print("4. Use --force-rebuild to start fresh if needed.")
            raise

    def _validate_pipeline_integrity(self) -> bool:
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")

        # A force rebuild reads no cached output, so there is nothing to check.
        if (
            self.args.pipeline_id
            and not self.args.force_rebuild
            and self.pipeline.state_manager.pipeline_exists(self.args.pipeline_id)
        ):
            # Checked under the cache keys the run will read (they include the
            # corpus fingerprint), not keys built before the corpus is known.
            is_valid, errors = self.pipeline.validate_resume_integrity(
                self.args.source_directory, self.args.pipeline_id
            )
            if not is_valid:
                console.print(
                    f"\n[red]Pipeline integrity check failed for '{self.args.pipeline_id}':[/red]"
                )
                for error in errors:
                    console.print(f"  • {error}")
                console.print(
                    "[yellow]The inputs changed since these stages ran, or the "
                    "cache is incomplete; a resume without --resume-from-stage recomputes them.[/yellow]"
                )
                if not self._confirm("Continue despite integrity issues?"):
                    console.print("[yellow]Pipeline execution cancelled.[/yellow]")
                    return False
        return True

    def _require_indexing_endpoints(self, pipeline_config: PipelineConfig) -> None:
        # Fail before the paid LLM stages rather than at the indexing stage.
        if self.args.verify_metadata:
            return
        if not pipeline_config.stages_enabled.get(PipelineStageType.INDEXING):
            return
        error = missing_endpoints_error(
            self.config,
            {RetrieverRole.GRAPH, RetrieverRole.DOCUMENT},
            "the indexing stage",
        )
        if error:
            console.print(f"[red]Error: {error}[/red]")
            sys.exit(1)

    def run(self) -> None:
        display_ascii_art(__version__)
        self._display_run_info()

        console.print("\n[bold]Initializing pipeline...[/bold]")
        pipeline_config = self._create_pipeline_config()
        self._require_indexing_endpoints(pipeline_config)
        self.pipeline = DataIngestionPipeline(
            config=self.config,
            pipeline_config=pipeline_config,
            source_directory=self.args.source_directory,
            target_directory=self.args.target_directory,
            metrics_sink=self._build_metrics_sink(),
        )

        try:
            if self._handle_metadata_operations():
                return

            if not self._validate_pipeline_integrity():
                return

            context = self._execute_pipeline()

            if context:
                display_pipeline_results(
                    context, executed_stages=self.pipeline.executed_stages
                )
                # A soft stage failure stops the run but does NOT raise — the
                # pipeline returns a context with status=FAILED. Without this
                # check the CLI would print success and exit 0, so Step Functions
                # (RUN_JOB, keyed on container exit code) marks the phase
                # SUCCEEDED and the failure is invisible. Raise so main() exits
                # non-zero and the failure surfaces.
                if context.status == PipelineStageStatus.FAILED:
                    failed = [
                        r.stage_name
                        for r in context.stage_results
                        if r.status == PipelineStageStatus.FAILED
                    ]
                    raise PipelineExecutionError(
                        f"Pipeline failed at stage(s): {', '.join(failed) or 'unknown'}"
                    )
                success_msg = "✓ Pipeline completed successfully!"
                console.print(
                    Panel.fit(
                        f"[bold green]{success_msg}[/bold green]", border_style="green"
                    )
                )
        finally:
            # Release the indexers' Neptune/OpenSearch connections so a finished
            # ingest run does not leak websockets/HTTP pools until interpreter
            # exit. Best-effort: pipeline.close() never raises.
            self.pipeline.close()


def main() -> None:
    load_cli_dotenv()
    try:
        cli = CommandLineInterface()
        args = cli.parse_args()
        runner = IngestionPipelineRunner(args)
        runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(130)
    except PipelineExecutionError as e:
        # A failed stage, a failed metadata check or a refused prompt: say
        # which before exiting non-zero.
        console.print(f"\n[red]Error: {escape(str(e))}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {e}[/red]")
        logger.exception("Unexpected error during pipeline execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
