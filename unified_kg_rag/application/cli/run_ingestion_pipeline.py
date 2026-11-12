# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

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
    interrupted_fields,
    json_output,
    print_json,
    print_status,
    setup_cli_logging,
    usage_error,
)
from unified_kg_rag.application.cli.display import (
    PRODUCT_NAME,
    console,
    display_ascii_art,
    display_pipeline_results,
    markup_safe,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.lazy import lazy_names
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    missing_extras_error,
)
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
)
from unified_kg_rag.application.cli.query_args import namespace_name, suffix_arg
from unified_kg_rag.application.config import load_cli_dotenv, resolve_relative_paths
from unified_kg_rag.domain.ingestion.methodologies import writes_graph_store
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    RetrieverRole,
    validate_path_segment,
)
from unified_kg_rag.domain.models.config import merge_over
from unified_kg_rag.shared import (
    CloudWatchEMFSink,
    ConfigurationError,
    ConfirmationDeclinedError,
    MetricsSink,
    ModelUnreachableError,
    NullMetricsSink,
    PipelineExecutionError,
    PipelineInterrupted,
    StoreConnectionError,
    get_logger,
)
from unified_kg_rag.shared.utils.validation import format_validation_errors

if TYPE_CHECKING:
    from unified_kg_rag.application.ingestion.pipeline import (
        DataIngestionPipeline,
        run_token_usage,
        unneeded_stage_error,
    )

logger = get_logger(__name__)


def _pipeline_module() -> ModuleType:
    from unified_kg_rag.application.ingestion import pipeline

    return pipeline


# Imported on first use: --help and a refused command line load no pipeline.
__getattr__, _load_pipeline = lazy_names(
    globals(),
    dict.fromkeys(
        ("DataIngestionPipeline", "run_token_usage", "unneeded_stage_error"),
        _pipeline_module,
    ),
)


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


# The env var answering yes to a reset's question (--confirm-reset), for a
# task whose command is fixed (Step Functions) or overridden by env only.
CONFIRM_RESET_ENV = "UKR_CONFIRM_RESET"
# And to a mass deletion's (--confirm-deletions, indexing.max_deletion_share).
CONFIRM_DELETIONS_ENV = "UKR_CONFIRM_DELETIONS"
_TRUE = frozenset({"1", "true", "yes"})
_FALSE = frozenset({"", "0", "false", "no"})


def _stage_name_arg(value: str) -> str:
    """A pipeline stage name, case-insensitive; unknown names are a usage error."""
    _load_pipeline()
    try:
        normalized = DataIngestionPipeline.normalize_stage_name(value)
    except ConfigurationError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    assert normalized is not None
    return normalized


def _methodology_list(value: str) -> list[str]:
    """A comma-separated methodology list; the configuration validates the
    names (an unknown one is a usage error naming the valid ones)."""
    names = [name.strip() for name in value.split(",")]
    if not all(names):
        raise argparse.ArgumentTypeError(f"empty methodology name in '{value}'")
    return names


def apply_namespace_arguments(config: Config, args: argparse.Namespace) -> Config:
    """``config`` with ``--suffix``, ``--methodologies`` and ``--reset``
    applied (validated as the config file's keys are); ``config`` itself
    when none is given.

    Raises:
        ConfigurationError: A value the configuration refuses (an unknown
            methodology, a selection a switch leaves nothing of).
    """
    overrides: dict[str, Any] = {}
    if suffix := getattr(args, "suffix", None):
        overrides["processing"] = {"document_parsing": {"index_value": suffix}}
    indexing: dict[str, Any] = {}
    if methodologies := getattr(args, "methodologies", None):
        indexing["methodologies"] = methodologies
    if getattr(args, "reset", False):
        indexing["reset"] = True
    if indexing:
        overrides["indexing"] = indexing
    if not overrides:
        return config
    try:
        updated = Config.model_validate(merge_over(config.as_input(), overrides))
    except ValidationError as e:
        raise ConfigurationError(format_validation_errors("", e)) from e
    if config_path := getattr(args, "config_path", None):
        resolve_relative_paths(updated, config_path)
    return updated


def _build_record(context: PipelineContext) -> str | None:
    """What became of the suffix's build record in the run's indexing stage
    (its ``build_record`` metric), ``None`` when the stage did not run."""
    for result in reversed(context.stage_results):
        if result.stage_name == PipelineStageType.INDEXING.value:
            state = (result.metrics or {}).get("build_record")
            return state if isinstance(state, str) else None
    return None


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = ArgumentParser(
            allow_abbrev=False,
            description=f"{PRODUCT_NAME}: parse documents and build the "
            "knowledge graph and indexes the selected methodologies read",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=doc_epilog(
                "guides/ingest.md#examples",
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
            help="Recompute every stage, ignoring the cached stage outputs, the "
            "stored extraction records and the LLM response cache: every model "
            "call and embedding is paid again (the caches are rewritten with "
            "the new outputs)",
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
            "integrity check, or a reset that asks first, as --confirm-reset "
            "does, or a mass deletion, as --confirm-deletions does). Without "
            "it, a prompt that cannot be asked because stdin is not a "
            "terminal fails the run",
        )
        parser.add_argument(
            "--confirm-reset",
            action="store_true",
            help="Answer yes to the question of an indexing.reset run that "
            "drops another source scope's seed graph documents, or, from a "
            "corpus of seed graphs alone, another scope's documents, and of a "
            "run with incremental indexing off that rebuilds a non-empty "
            "index from the seed graphs alone; nothing else (falls back to "
            f"the {CONFIRM_RESET_ENV} env var: 1, true or yes). "
            "Without it (or --yes), the question is asked on the terminal, and "
            "the run stops (exit 1) when stdin is not one",
        )
        parser.add_argument(
            "--confirm-deletions",
            action="store_true",
            help="Answer yes to the question of an incremental run that would "
            "delete more than indexing.max_deletion_share of its source "
            "scope's documents (their files are gone from the corpus); "
            "nothing else (falls back to the "
            f"{CONFIRM_DELETIONS_ENV} env var: 1, true or yes). Without it "
            "(or --yes), the question is asked on the terminal, and the run "
            "stops (exit 1) before it writes when stdin is not one",
        )
        parser.add_argument(
            "--confirm-integrity",
            action="store_true",
            help="Answer yes to the question of a run that resumes a "
            "--pipeline-id whose inputs changed since its stages ran (or "
            "whose cache is incomplete): continue, recomputing those stages; "
            "nothing else. Without it (or --yes), the question is asked on "
            "the terminal, and the run stops (exit 1) when stdin is not one",
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
        namespace = parser.add_argument_group(
            "index namespace",
            "Override what the run writes; each flag applies to this run only.",
        )
        namespace.add_argument(
            "--suffix",
            type=suffix_arg,
            help="Index suffix (namespace) the run writes: "
            "processing.document_parsing.index_value (default: the config's, "
            "else 'default'). Query it with run-rag/run-eval --suffix",
        )
        namespace.add_argument(
            "--methodologies",
            type=_methodology_list,
            metavar="LIST",
            help="Comma-separated methodologies to build (graphrag, lightrag, "
            "hipporag, raptor): indexing.methodologies",
        )
        namespace.add_argument(
            "--reset",
            action="store_true",
            help="Rebuild the namespace from the whole corpus instead of "
            "applying a delta (indexing.reset: true for this run only, so a "
            "config file need not be edited back)",
        )
        add_config_arguments(parser)
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
        for flag, env in (
            ("confirm_reset", CONFIRM_RESET_ENV),
            ("confirm_deletions", CONFIRM_DELETIONS_ENV),
        ):
            if getattr(args, flag):
                continue
            value = os.getenv(env, "").strip().lower()
            if value not in _TRUE | _FALSE:
                self.parser.error(
                    f"{env} must be 1, true or yes (or 0, false, "
                    f"no or empty), not {os.environ[env]!r}"
                )
            setattr(args, flag, value in _TRUE)
        return args


class IngestionPipelineRunner:
    def __init__(self, args: argparse.Namespace):
        _load_pipeline()
        self.args = args
        with exit_on_config_error():
            self.config = apply_namespace_arguments(load_cli_config(args), args)
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
        setup_cli_logging(self.config, args)
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
            "[bold blue]Ingestion pipeline[/bold blue]",
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

        parsing = self.config.processing.document_parsing
        namespace = namespace_name(parsing.index_value, self.config)
        info_lines.extend(
            [
                f"[dim]Index Namespace:[/dim] '{markup_safe(namespace)}'",
                "[dim]Methodologies:[/dim] "
                + ", ".join(m.value for m in self.config.effective_methodologies),
                f"[dim]Reset:[/dim] {self.config.indexing.reset}",
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
            # Before any model call: a listed stage no selected methodology
            # needs would pay for what no query reads.
            if error := unneeded_stage_error(
                self.config, [all_stages[name] for name in sorted(enabled_stages_set)]
            ):
                usage_error(error, error_type=ConfigurationError.__name__)
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
        return self._ask(question, "--yes", getattr(self.args, "yes", False))

    def _confirm_integrity(self, question: str) -> bool:
        """Ask the integrity ``question`` of a resumed run whose inputs
        changed: ``--yes`` answers it; without a terminal the run stops,
        naming ``--confirm-integrity`` (the flag that answers it alone)."""
        return self._ask(
            question, "--confirm-integrity", getattr(self.args, "yes", False)
        )

    def _confirm_reset(self, question: str) -> bool:
        """Ask a reset's ``question`` (the pipeline's ``confirm``):
        ``--confirm-reset`` (or its env var) answers it, and so does
        ``--yes``; without a terminal to ask on the run stops."""
        if getattr(self.args, "confirm_reset", False):
            console.print(f"{question} [dim](--confirm-reset)[/dim]")
            return True
        return self._ask(
            question,
            f"--confirm-reset (or {CONFIRM_RESET_ENV}=1)",
            getattr(self.args, "yes", False),
        )

    def _confirm_deletions(self, question: str) -> bool:
        """Ask a mass deletion's ``question`` (the pipeline's
        ``confirm_deletions``): ``--confirm-deletions`` (or its env var)
        answers it, and so does ``--yes``; without a terminal to ask on the
        run stops."""
        if getattr(self.args, "confirm_deletions", False):
            console.print(f"{question} [dim](--confirm-deletions)[/dim]")
            return True
        return self._ask(
            question,
            f"--confirm-deletions (or {CONFIRM_DELETIONS_ENV}=1)",
            getattr(self.args, "yes", False),
        )

    @staticmethod
    def _ask(question: str, flag: str, answered: bool) -> bool:
        """Ask ``question`` on the terminal unless ``answered`` (by --yes);
        without a terminal the run stops, naming ``flag``."""
        if answered:
            console.print(f"{question} [dim](--yes)[/dim]")
            return True
        if not sys.stdin.isatty():
            console.print(
                f"[red]Error: {question} needs confirmation, but stdin is not a "
                f"terminal. Re-run with {flag} to proceed.[/red]"
            )
            raise ConfirmationDeclinedError(
                f"Confirmation required ({question}); pass {flag}"
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
        except ConfirmationDeclinedError:
            raise
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
                if getattr(self.args, "confirm_integrity", False):
                    console.print(
                        "Continue despite integrity issues? "
                        "[dim](--confirm-integrity)[/dim]"
                    )
                elif not self._confirm_integrity("Continue despite integrity issues?"):
                    console.print("[yellow]Pipeline execution cancelled.[/yellow]")
                    return False
        return True

    def _require_indexing_endpoints(self, pipeline_config: PipelineConfig) -> None:
        # Fail before the paid LLM stages rather than at the indexing stage.
        if self.args.verify_metadata:
            return
        if not pipeline_config.stages_enabled.get(PipelineStageType.INDEXING):
            return
        # A run that builds no graph-store artifact (raptor alone) writes
        # no graph store.
        roles = {RetrieverRole.DOCUMENT}
        if writes_graph_store(self.config):
            roles.add(RetrieverRole.GRAPH)
        error = missing_endpoints_error(self.config, roles, "the indexing stage")
        if error:
            usage_error(error)

    def run(self) -> None:
        _load_pipeline()
        display_ascii_art(__version__)
        self._display_run_info()

        console.print("\n[bold]Initializing pipeline...[/bold]")
        pipeline_config = self._create_pipeline_config()
        # The stages build their models, and the pipeline loads the seeds,
        # when the pipeline is built.
        if error := missing_extras_error(self.config, seeds=True):
            usage_error(error)
        self._require_indexing_endpoints(pipeline_config)
        self.pipeline = DataIngestionPipeline(
            config=self.config,
            pipeline_config=pipeline_config,
            source_directory=self.args.source_directory,
            target_directory=self.args.output_directory,
            metrics_sink=self._build_metrics_sink(),
            confirm=self._confirm_reset,
            confirm_deletions=self._confirm_deletions,
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
                    console.print(
                        f"\n[yellow]Next step: {markup_safe(self._next_step(context))}"
                        "[/yellow]"
                    )
                    raise PipelineExecutionError(error)
                success_msg = "✓ Pipeline completed successfully!"
                console.print(
                    Panel.fit(
                        f"[bold green]{success_msg}[/bold green]", border_style="green"
                    )
                )
                if context.visualization_data_path:
                    console.print(
                        "Graph for run-visualization: "
                        f"{markup_safe(context.visualization_data_path)}"
                    )
                if _build_record(context) in ("failed", "missing"):
                    console.print(
                        "[yellow]The namespace holds no build record: queries "
                        "take it to hold what their config builds until a run "
                        "writes one (the next run tries again).[/yellow]"
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
        """The error of a run a stage failed, with the first failed stage's
        own error (the cause, which was only in the logs), or None."""
        if context.status != PipelineStageStatus.FAILED:
            return None
        failed = [
            r for r in context.stage_results if r.status == PipelineStageStatus.FAILED
        ]
        error = (
            f"Pipeline failed at stage(s): "
            f"{', '.join(r.stage_name for r in failed) or 'unknown'}"
        )
        first = next((r for r in failed if r.error_message), None)
        if first is not None:
            error += f". First error ({first.stage_name}): {first.error_message}"
        return error

    @staticmethod
    def _next_step(context: PipelineContext) -> str:
        """What to do after a run a stage failed: the fix the error names
        (an embedding dimension the index refuses), then the resume."""
        error = " ".join(
            r.error_message or ""
            for r in context.stage_results
            if r.status == PipelineStageStatus.FAILED
        )
        step = ""
        if "dimension" in error.lower():
            step = (
                "The embedding dimension does not match the index: set "
                "models.embedding.dimension (--embedding-dimension) to what "
                "the embedding model returns, or ingest into a new namespace "
                "(--suffix) or with --reset. "
            )
        return (
            f"{step}Fix the cause, then rerun with --pipeline-id "
            f"{context.pipeline_id} to resume from the failed stage (the "
            "finished stages are cached; --force-rebuild recomputes them)."
        )

    def _json_result(self, context: PipelineContext) -> dict[str, Any]:
        """The run's ``--output-format json`` result; a run a stage failed
        also has its ``error`` and ``exit_code``, as a run that fails before
        printing does."""
        assert self.pipeline is not None
        _load_pipeline()
        error = self._stage_failure(context)
        return {
            "pipeline_id": context.pipeline_id,
            "status": context.status.value,
            # The keys of every failure document (status stays the run's).
            **(
                {
                    "error": error,
                    "exit_code": EXIT_FAILURE,
                    "error_type": PipelineExecutionError.__name__,
                }
                if error
                else {}
            ),
            "duration_seconds": context.duration_seconds,
            # The namespace's graph for run-visualization (None: not written).
            "visualization_data": context.visualization_data_path,
            # The suffix's build record after the run (written, kept, failed,
            # missing, unsupported; None: the indexing stage did not run).
            "build_record": _build_record(context),
            "failed_documents": len(context.recorded_failed_doc_ids),
            # Text units a stage failed on: counted in a full build too,
            # which records no document FAILED.
            "units_failed": len(context.failed_unit_ids()),
            "documents_retrying": len(context.retrying_doc_ids),
            "documents_without_claims": len(context.claims_missing_doc_ids),
            "documents_excluded": (
                len(context.incremental_delta.excluded)
                if context.incremental_delta is not None
                else 0
            ),
            # The model calls and tokens of the stages this run executed.
            "token_usage": run_token_usage(context, self.pipeline.executed_stages),
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
            **interrupted_fields(exit_code),
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
    except ConfirmationDeclinedError as e:
        # A question answered no (or not answerable): the run stopped before
        # it wrote anything.
        console.print(
            f"\n[yellow]Pipeline stopped before it wrote anything: "
            f"{markup_safe(e)}[/yellow]"
        )
        _print_failed(runner, EXIT_FAILURE, e)
        sys.exit(EXIT_FAILURE)
    except (PipelineExecutionError, StoreConnectionError, ModelUnreachableError) as e:
        # A failed stage, a failed metadata check, a refused prompt or an
        # unreachable store or model server: say which before exiting non-zero.
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
