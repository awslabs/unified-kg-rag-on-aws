# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import boto3
from dotenv import load_dotenv
from rich.panel import Panel
from rich.prompt import Confirm

from aws_graphrag.domain.models import (
    PipelineConfig,
    PipelineContext,
    PipelineStageType,
)
from aws_graphrag.ingestion import DataIngestionPipeline
from aws_graphrag.shared import (
    CloudWatchEMFSink,
    MetricsSink,
    NullMetricsSink,
    PipelineExecutionError,
    get_config,
    get_logger,
)
from aws_graphrag.shared.utils import (
    console,
    display_ascii_art,
    display_pipeline_results,
)

load_dotenv()
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
            description="GraphRAG Data Ingestion Pipeline - Process documents and build knowledge graphs",
            formatter_class=argparse.RawDescriptionHelpFormatter,
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
            help="Path to directory for storing parsed documents (defaults to source directory)",
        )
        parser.add_argument(
            "--cache-directory",
            type=str,
            help="Path to directory for storing pipeline cache and intermediate results",
        )
        parser.add_argument(
            "--force-rebuild",
            action="store_true",
            help="Force complete pipeline rebuild, ignoring all existing cache data",
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
            type=str,
            default=os.getenv("GRAPHRAG_PIPELINE_ID"),
            help="Unique identifier of an existing pipeline run to resume or inspect "
            "(falls back to the GRAPHRAG_PIPELINE_ID env var)",
        )
        parser.add_argument(
            "--resume-from-stage",
            type=str,
            help="Specific pipeline stage to resume execution from (requires --pipeline-id)",
        )
        parser.add_argument(
            "--verify-metadata",
            action="store_true",
            help="Verify the integrity and consistency of pipeline metadata files",
        )
        parser.add_argument(
            "--repair-metadata",
            action="store_true",
            help="Attempt automatic repair of corrupted or inconsistent metadata files",
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
            "--config-path",
            type=str,
            help="Path to custom configuration file",
        )
        return parser

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


class IngestionPipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = get_config(Path(args.config_path) if args.config_path else None)
        self.pipeline: DataIngestionPipeline | None = None
        self._validate_args()

    def _validate_args(self) -> None:
        metadata_only = self.args.verify_metadata or self.args.repair_metadata

        if not metadata_only and not self.args.source_directory:
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

        if metadata_only and not self.args.pipeline_id:
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
            batch_size=self.config.processing.batch_size,
            max_retries=self.config.processing.max_retries,
            continue_on_error=self.args.continue_on_error,
            cache_enabled=not self.args.force_rebuild,
            local_directory=self.args.cache_directory,
            s3_sync_enabled=self.args.s3_sync,
            s3_bucket_name=self.args.s3_bucket_name,
            s3_prefix=self.args.s3_prefix,
            force_rebuild=self.args.force_rebuild,
            pipeline_id=self.args.pipeline_id,
            resume_from_stage=self.args.resume_from_stage,
        )

    def _handle_metadata_operations(self) -> bool:
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")

        if not (self.args.verify_metadata or self.args.repair_metadata):
            return False

        if self.args.verify_metadata:
            console.print(
                f"\n[yellow]Verifying metadata for pipeline: '{self.args.pipeline_id}'[/yellow]"
            )
            if self.pipeline.verify_pipeline_metadata(self.args.pipeline_id):
                console.print("[green]✓ Metadata file is valid[/green]")
            else:
                console.print("[red]✗ Metadata file is corrupted or invalid[/red]")
                if not self.args.repair_metadata:
                    return True

        if self.args.repair_metadata:
            console.print(
                f"\n[yellow]Attempting to repair metadata for pipeline: {self.args.pipeline_id}[/yellow]"
            )
            if not Confirm.ask("This may overwrite existing metadata. Continue?"):
                console.print("[yellow]Repair operation cancelled.[/yellow]")
                return True

            try:
                if self.pipeline.repair_pipeline_metadata(self.args.pipeline_id):
                    console.print(
                        "[green]✓ Metadata file repaired successfully[/green]"
                    )
                else:
                    console.print("[red]✗ Failed to repair metadata file[/red]")
            except Exception as e:
                console.print(f"[red]Error during repair: {e}[/red]")
            return True

        return self.args.verify_metadata and not self.args.repair_metadata

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
        except PipelineExecutionError as e:
            console.print(f"\n[red]Pipeline execution failed: {e}[/red]")
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

        if self.args.pipeline_id and self.pipeline.state_manager.pipeline_exists(
            self.args.pipeline_id
        ):
            is_valid, errors = self.pipeline.resume_manager.validate_pipeline_integrity(
                self.args.pipeline_id
            )
            if not is_valid:
                console.print(
                    f"\n[red]Pipeline integrity check failed for '{self.args.pipeline_id}':[/red]"
                )
                for error in errors:
                    console.print(f"  • {error}")
                if not Confirm.ask("Continue despite integrity issues?"):
                    console.print("[yellow]Pipeline execution cancelled.[/yellow]")
                    return False
        return True

    def run(self) -> None:
        display_ascii_art(__version__)
        self._display_run_info()

        console.print("\n[bold]Initializing pipeline...[/bold]")
        boto_session = boto3.Session(profile_name=self.config.aws.profile_name)
        pipeline_config = self._create_pipeline_config()
        self.pipeline = DataIngestionPipeline(
            config=self.config,
            pipeline_config=pipeline_config,
            source_directory=self.args.source_directory,
            target_directory=self.args.target_directory,
            boto_session=boto_session,
            metrics_sink=self._build_metrics_sink(),
        )

        try:
            if self._handle_metadata_operations():
                return

            if not self._validate_pipeline_integrity():
                return

            context = self._execute_pipeline()

            if context:
                display_pipeline_results(context)
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
    try:
        cli = CommandLineInterface()
        args = cli.parse_args()
        runner = IngestionPipelineRunner(args)
        runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(130)
    except PipelineExecutionError:
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {e}[/red]")
        logger.exception("Unexpected error during pipeline execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
