# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import signal
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.panel import Panel
from rich.prompt import Confirm

from unified_kg_rag import __version__
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
    print_status,
    usage_error,
)
from unified_kg_rag.application.cli.display import (
    console,
    display_ascii_art,
    display_pipeline_results,
    markup_safe,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.preflight import missing_endpoints_error
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.application.config import load_cli_dotenv
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
    PipelineInterrupted,
    StoreConnectionError,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def _pipeline_id_arg(value: str) -> str | None:
    if not value:  # an empty UKR_PIPELINE_ID means "unset"
        return None
    try:
        return validate_path_segment(value, "pipeline_id")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _source_scope_arg(value: str) -> str:
    """A source scope to retire, stripped; a blank one is a usage error."""
    if not value.strip():
        raise argparse.ArgumentTypeError("a source scope must not be blank")
    return value.strip()


def _stage_name_arg(value: str) -> str:
    """A pipeline stage name, case-insensitive; unknown names are a usage error."""
    try:
        normalized = DataIngestionPipeline.normalize_stage_name(value)
    except ConfigurationError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    assert normalized is not None
    return normalized


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = ArgumentParser(
            allow_abbrev=False,
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
            default=os.getenv("UKR_SOURCE_DIRECTORY"),
            help="Directory of the source documents to process; required unless "
            "--verify-metadata (falls back to the UKR_SOURCE_DIRECTORY env var)",
        )
        parser.add_argument(
            "--output-directory",
            type=str,
            help="Export each parsed document as <stem>.json to this directory "
            "for inspection (processing.document_parsing.target_directory; no "
            "export when omitted; must not be the source directory)",
        )
        parser.add_argument(
            "--cache-directory",
            type=str,
            help="Directory of the pipeline cache and intermediate results "
            "(default: cache)",
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
            help="Synchronize the cache with Amazon S3 (needs --s3-bucket-name "
            "or aws.s3.bucket_name)",
        )
        parser.add_argument(
            "--s3-bucket-name",
            type=str,
            help="Amazon S3 bucket for cache synchronization with --s3-sync "
            "(default: aws.s3.bucket_name)",
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
            default=os.getenv("UKR_PIPELINE_ID"),
            help="Identifier of the pipeline run to resume or inspect (falls "
            "back to the UKR_PIPELINE_ID env var). It names the run's cache "
            "directory and S3 prefix, so it takes letters, digits, '.', '_' and "
            "'-' only: up to 128 characters, starting with a letter or digit, "
            "with no '..'",
        )
        parser.add_argument(
            "--resume-from-stage",
            type=_stage_name_arg,
            help="Pipeline stage to resume from, by name (case-insensitive; "
            "requires --pipeline-id); an unknown name is a usage error",
        )
        parser.add_argument(
            "--verify-metadata",
            action="store_true",
            help="Verify the integrity of the run's pipeline metadata (needs "
            "--pipeline-id); exits non-zero when it is corrupt or the run is "
            "not found",
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
            help="Keep going when a stage fails. When an artifact stage "
            "(translation through community detection) crashes, every document "
            "of the run is recorded FAILED, so the next incremental run "
            "processes them again",
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
            "'cloudwatch' (CloudWatch EMF to stdout, auto-extracted by "
            "CloudWatch Logs, with an Environment dimension when "
            "UKR_METRICS_ENVIRONMENT is set; pipeline_id is a log property, not "
            "a dimension). A stage's metrics are emitted when it finishes and "
            "only by the run that executed it, so a resumed run does not "
            "repeat restored stages' counts",
        )
        parser.add_argument(
            "--retire-source-scope",
            action="append",
            type=_source_scope_arg,
            default=[],
            metavar="SCOPE",
            help="Incremental runs: treat every doc-status record of this "
            "former source scope (in this run's index namespace) as deleted, "
            "removing its exclusive graph and vector content. Use it once "
            "after moving a corpus, with the old absolute source directory or "
            "former document_parsing.source_scope value (a value that is "
            "not a URI is also tried as a path resolved like "
            "--source-directory; a bare value whose name and path readings "
            "both have records stops the run: use ./ or an absolute path). "
            "Repeatable; added to "
            "indexing.retire_source_scopes. Never this run's own scope.",
        )
        add_config_arguments(parser, "Path to custom configuration file")
        add_output_format_argument(parser)
        add_provider_arguments(parser)
        return parser

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        if args.output_format == "json" and args.metrics_sink == "cloudwatch":
            # CloudWatch EMF records are printed to stdout too.
            self.parser.error(
                "--output-format json cannot be combined with --metrics-sink "
                "cloudwatch, which writes its records to stdout"
            )
        return args


class IngestionPipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        with exit_on_config_error():
            self.config = apply_provider_arguments(
                load_cli_config(args),
                args,
            )
        retired = getattr(args, "retire_source_scope", None) or []
        if retired:
            indexing = self.config.indexing
            try:
                indexing.retire_source_scopes = [
                    *indexing.retire_source_scopes,
                    *retired,
                ]
            except ValidationError as e:
                usage_error(f"--retire-source-scope: {e}")
        self.json_output = json_output(args)
        setup_logging(self.config, stream=sys.stderr if self.json_output else None)
        self.pipeline: DataIngestionPipeline | None = None
        # Whether the run reported its result (a stop after it came while
        # the run closed its connections: nothing is left to resume).
        self.reported = False
        self._validate_args()

    def _validate_args(self) -> None:
        if not self.args.verify_metadata and not self.args.source_directory:
            usage_error("--source-directory is required for pipeline execution")

        if self.args.source_directory:
            self.args.source_directory = Path(self.args.source_directory)
            if not self.args.source_directory.is_dir():
                usage_error(
                    f"source directory not found: '{self.args.source_directory}'"
                )
            target = (
                self.args.output_directory
                or self.config.processing.document_parsing.target_directory
            )
            if target and (
                Path(target).resolve() == self.args.source_directory.resolve()
            ):
                # Parsed .json written into the corpus is re-ingested as new
                # documents on the next run.
                usage_error(
                    "--output-directory (document_parsing.target_directory) "
                    "must not be the source directory "
                    f"'{self.args.source_directory}': the parsed .json files "
                    "would be ingested as new documents on the next run"
                )

        if self.args.s3_sync and not self._s3_bucket_name():
            usage_error("--s3-sync needs --s3-bucket-name or aws.s3.bucket_name")

        if self.args.resume_from_stage and not self.args.pipeline_id:
            usage_error("--pipeline-id is required for --resume-from-stage")

        if self.args.verify_metadata and not self.args.pipeline_id:
            usage_error("--pipeline-id is required for --verify-metadata")

        self.args.cache_directory = (
            Path(self.args.cache_directory)
            if self.args.cache_directory
            else Path("cache")
        )
        self.args.cache_directory.mkdir(parents=True, exist_ok=True)

    def _s3_bucket_name(self) -> str | None:
        """The cache bucket: the flag, else the configured one (the CDK task
        injects it as UKR__AWS__S3__BUCKET_NAME)."""
        return self.args.s3_bucket_name or self.config.aws.s3.bucket_name

    def _display_run_info(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_lines = [
            "[bold blue]Graph RAG Document Processing Pipeline[/bold blue]",
            f"[dim]Timestamp:[/dim] '{timestamp}'",
            f"[dim]Cache Directory:[/dim] "
            f"'{markup_safe(self.args.cache_directory.absolute())}'",
        ]

        if self.args.source_directory:
            info_lines.insert(
                1,
                f"[dim]Source Directory:[/dim] "
                f"'{markup_safe(self.args.source_directory.absolute())}'",
            )

        info_lines.extend(
            [
                f"[dim]Force Rebuild:[/dim] {self.args.force_rebuild}",
                f"[dim]S3 Sync:[/dim] {self.args.s3_sync}",
                f"[dim]Continue on Error:[/dim] {self.args.continue_on_error}",
            ]
        )

        if self.args.pipeline_id:
            info_lines.append(
                f"[dim]Pipeline ID:[/dim] '{markup_safe(self.args.pipeline_id)}'"
            )
            if self.args.resume_from_stage:
                info_lines.append(
                    f"[dim]Resume From Stage:[/dim] "
                    f"'{markup_safe(self.args.resume_from_stage)}'"
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
            # Set by the CDK stack to its env_name, which its alarms filter on.
            environment = os.getenv("UKR_METRICS_ENVIRONMENT") or None
            logger.info(
                "Forwarding pipeline metrics to CloudWatch (EMF), environment '%s'",
                environment or "-",
            )
            return CloudWatchEMFSink(environment=environment)
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
                usage_error(
                    f"--enabled-stages: invalid stage names "
                    f"{', '.join(sorted(invalid_stages))} (available: "
                    f"{', '.join(all_stages.keys())})"
                )
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
            s3_bucket_name=self._s3_bucket_name(),
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
            f"\n[yellow]Verifying metadata for pipeline: "
            f"'{markup_safe(self.args.pipeline_id)}'[/yellow]"
        )
        if not self.pipeline.state_manager.pipeline_exists(self.args.pipeline_id):
            console.print(
                f"[red]✗ No metadata found for pipeline "
                f"'{markup_safe(self.args.pipeline_id)}' under "
                f"'{markup_safe(self.args.cache_directory)}'[/red]"
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
        if self.json_output:
            print_json({"pipeline_id": self.args.pipeline_id, "metadata_valid": True})
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
                    f"\n[red]Pipeline integrity check failed for "
                    f"'{markup_safe(self.args.pipeline_id)}':[/red]"
                )
                for error in errors:
                    console.print(f"  • {markup_safe(error)}")
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
            usage_error(error)

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
            target_directory=self.args.output_directory,
            metrics_sink=self._build_metrics_sink(),
        )

        previous_handlers = self._stop_on_signals(self.pipeline)
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
                if self.json_output:
                    print_json(self._json_result(context))
                self.reported = True
                # A soft stage failure stops the run but does NOT raise — the
                # pipeline returns a context with status=FAILED. Without this
                # check the CLI would print success and exit 0, so Step Functions
                # (RUN_JOB, keyed on container exit code) marks the task
                # SUCCEEDED and the failure is invisible. Raise so main() exits
                # non-zero and the failure surfaces.
                error = self._stage_failure(context)
                if error is not None:
                    raise PipelineExecutionError(error)
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
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    @staticmethod
    def _stage_failure(context: PipelineContext) -> str | None:
        """The error of a run a stage failed, or None."""
        if context.status != PipelineStageStatus.FAILED:
            return None
        failed = [
            r.stage_name
            for r in context.stage_results
            if r.status == PipelineStageStatus.FAILED
        ]
        return f"Pipeline failed at stage(s): {', '.join(failed) or 'unknown'}"

    def _json_result(self, context: PipelineContext) -> dict[str, Any]:
        """The run's ``--output-format json`` result; a run a stage failed
        also has its ``error`` and ``exit_code``, as a run that fails before
        printing does."""
        assert self.pipeline is not None
        error = self._stage_failure(context)
        return {
            "pipeline_id": context.pipeline_id,
            "status": context.status.value,
            **({"error": error, "exit_code": EXIT_FAILURE} if error else {}),
            "duration_seconds": context.duration_seconds,
            "failed_documents": len(context.recorded_failed_doc_ids),
            "documents_retrying": len(context.retrying_doc_ids),
            "documents_without_claims": len(context.claims_missing_doc_ids),
            "documents_excluded": (
                len(context.incremental_delta.excluded)
                if context.incremental_delta is not None
                else 0
            ),
            "stages": [
                {
                    **result.model_dump(
                        mode="json",
                        include={
                            "stage_name",
                            "status",
                            "duration_seconds",
                            "input_count",
                            "output_count",
                            "error_message",
                        },
                    ),
                    "executed": result.stage_name in self.pipeline.executed_stages,
                }
                for result in context.stage_results
            ],
        }

    @staticmethod
    def _stop_on_signals(pipeline: DataIngestionPipeline) -> dict[int, Any]:
        """Turn SIGTERM and Ctrl-C (SIGINT) into a checkpointed stop
        (``request_stop``, which defers one during a checkpoint or metadata
        write); returns the handlers it replaced (none off the main thread,
        where Python cannot install one). A SIGINT the parent ignores stays
        ignored. Raised as ``KeyboardInterrupt``, a Ctrl-C during the
        metadata save failed it (exit 1).

        ECS sends SIGTERM on StopTask and when a Step Functions execution times
        out, then kills the task once the container's stop timeout has passed.
        In the container this process is PID 1 (the entrypoint execs it), and
        the kernel does not deliver a signal to PID 1 that it has no handler
        for, so without one the task ran on until it was killed.
        """
        if threading.current_thread() is not threading.main_thread():
            return {}
        return {
            signum: signal.signal(
                signum, lambda signum, _: pipeline.request_stop(signum)
            )
            for signum in (signal.SIGTERM, signal.SIGINT)
            if signum == signal.SIGTERM
            or signal.getsignal(signum) is not signal.SIG_IGN
        }


def _next_run_writes_from(runner: IngestionPipelineRunner | None) -> str | None:
    """When the next run on the namespace may write first (UTC, ISO 8601),
    waiting for writes possibly still in flight (this run's, or those of the
    run whose expired lease it took over); ``None`` when it need not wait."""
    pipeline = runner.pipeline if runner else None
    until = pipeline.next_run_writes_from if pipeline is not None else 0.0
    if until <= time.time():
        return None
    return datetime.fromtimestamp(until, tz=UTC).isoformat(timespec="seconds")


def _next_run_wait(runner: IngestionPipelineRunner | None) -> str:
    """The stop message's note on the next run's wait (empty when none)."""
    when = _next_run_writes_from(runner)
    if when is None:
        return ""
    return (
        f" Writes may still be in flight: the next run on its namespace "
        f"waits until {when} before its first write."
    )


def _print_interrupted(runner: IngestionPipelineRunner | None, exit_code: int) -> None:
    """The JSON status of a stopped run, with the pipeline id to resume with
    and, when the next run must wait, from when it may write."""
    pipeline_id = runner.pipeline.pipeline_id if runner and runner.pipeline else None
    if runner and runner.json_output and pipeline_id and not runner.reported:
        status: dict[str, Any] = {
            "status": "interrupted",
            "exit_code": exit_code,
            "pipeline_id": pipeline_id,
        }
        if when := _next_run_writes_from(runner):
            status["next_run_writes_from"] = when
        print_json(status)


def _print_failed(
    runner: IngestionPipelineRunner | None, exit_code: int, error: Exception
) -> None:
    """The JSON status of a run that failed before it printed its result,
    with its error and, once resolved, its pipeline id."""
    pipeline_id = runner.pipeline.pipeline_id if runner and runner.pipeline else None
    print_status(
        exit_code, error, **({"pipeline_id": pipeline_id} if pipeline_id else {})
    )


@exits_on_sigterm
def main() -> None:
    load_cli_dotenv()
    runner: IngestionPipelineRunner | None = None
    try:
        cli = CommandLineInterface()
        args = cli.parse_args()
        runner = IngestionPipelineRunner(args)
        runner.run()
    except PipelineInterrupted as e:
        if runner and runner.reported:
            console.print(
                f"\n[yellow]Pipeline stopped by signal {e.signum} after it "
                "finished, while it closed its connections.[/yellow]"
            )
        else:
            console.print(
                f"\n[yellow]Pipeline stopped by signal {e.signum}; its finished "
                "stages are checkpointed, rerun with the same --pipeline-id to "
                f"resume.{_next_run_wait(runner)}[/yellow]"
            )
        _print_interrupted(runner, 128 + e.signum)
        sys.exit(128 + e.signum)
    except KeyboardInterrupt:
        console.print(
            f"\n[yellow]Pipeline interrupted by user.{_next_run_wait(runner)}[/yellow]"
        )
        _print_interrupted(runner, EXIT_INTERRUPTED)
        sys.exit(EXIT_INTERRUPTED)
    except ConfigurationError as e:
        console.print(f"\n[red]Error: {markup_safe(e)}[/red]")
        logger.error("Ingestion configuration error: %s", e)
        _print_failed(runner, EXIT_USAGE, e)
        sys.exit(EXIT_USAGE)
    except (PipelineExecutionError, StoreConnectionError) as e:
        # A failed stage, a failed metadata check, a refused prompt or an
        # unreachable store: say which before exiting non-zero.
        console.print(f"\n[red]Error: {markup_safe(e)}[/red]")
        _print_failed(runner, EXIT_FAILURE, e)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {markup_safe(e)}[/red]")
        logger.exception("Unexpected error during pipeline execution")
        _print_failed(runner, EXIT_FAILURE, e)
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()
