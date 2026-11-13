# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe an index namespace, and export its graph (``run-describe``).

Prints what the namespace of ``--suffix`` holds: its methodologies and the
search strategies they serve, each index with its item count and embedding
model, the build record of the last ingestion run and the doc-status
registry's records (``application.storage.inspection.describe_namespace``).
With ``--output-path`` it also writes the namespace's graph, streamed from
the vector store, as JSON Lines or GraphML (``export_graph``). It calls no
model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    failed,
    json_output,
    print_json,
    refuse_overwrites,
    setup_cli_logging,
    usage_error,
)
from unified_kg_rag.application.cli.display import console, markup_safe
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.preflight import missing_extras_error
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
)
from unified_kg_rag.application.cli.query_args import positive_int
from unified_kg_rag.application.config import load_cli_dotenv
from unified_kg_rag.application.storage.inspection import (
    EXPORT_FORMATS,
    NamespaceDescription,
    describe_namespace,
    export_graph,
)
from unified_kg_rag.domain.models import DocStatus
from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.exceptions import (
    ErrorClass,
    IndexNotFoundError,
    StoreConnectionError,
)

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        allow_abbrev=False,
        description=(
            "Describe an index namespace (methodologies, servable strategies, "
            "indices, build record, registry) and optionally export its graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "guides/inspect.md",
            "run-describe --suffix tenant_a\n"
            "  run-describe --suffix tenant_a --output-path graph.graphml "
            "--export-format graphml",
        ),
    )
    parser.add_argument(
        "--suffix",
        help="Index suffix to describe (default: the config's "
        "processing.document_parsing.index_value, else 'default')",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Also export the namespace's graph to this file (entities, "
        "relationships, communities, claims, with the text units and "
        "documents they cite). An existing file is refused unless --force, "
        "and so are the --config-path file and the schema file the config "
        "loads, always",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Let --output-path replace an existing file (never the "
        "--config-path file or the schema file the config loads)",
    )
    parser.add_argument(
        "--export-format",
        choices=EXPORT_FORMATS,
        default="jsonl",
        help="Export format: jsonl (one record per line, every stored field "
        "but the vectors) or graphml (entities as nodes, relationships as "
        "edges, communities as nodes linked to their parent)",
    )
    parser.add_argument(
        "--max-items",
        type=positive_int,
        help="Export at most this many items of each kind (default: all)",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Export each text unit's text too (jsonl only)",
    )
    parser.add_argument(
        "--documents",
        type=_statuses,
        metavar="STATUSES",
        default=(),
        help="Also list the doc-status registry's documents of these "
        "comma-separated statuses (failed, quarantined, pending, processed): "
        "file, status, error_info, failure, retry and attempt counts, the "
        "pipeline id that wrote the record and when (needs "
        "providers.doc_status.enabled)",
    )
    add_config_arguments(parser)
    add_output_format_argument(parser)
    add_provider_arguments(parser)
    return parser


def _statuses(value: str) -> tuple[DocStatus, ...]:
    """argparse ``type`` of ``--documents``: registry statuses."""
    try:
        return tuple(
            dict.fromkeys(
                DocStatus(part.strip().lower())
                for part in value.split(",")
                if part.strip()
            )
        )
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid statuses {value!r} (choose from "
            f"{', '.join(s.value for s in DocStatus)})"
        ) from None


def _print(description: NamespaceDescription) -> None:
    console.print(
        f"[bold]Index namespace[/bold] {description.namespace} "
        f"(suffix {description.suffix})"
    )
    build = description.build or {}
    if build:
        console.print(
            f"Last build: {build.get('mode')} run {build.get('pipeline_id')} at "
            f"{build.get('built_at')}, {build.get('documents')} documents"
        )
    elif description.exists:
        console.print(
            "[yellow]Last build: no build record (a store rejected it, or a "
            "build from before records): queries take the namespace to hold "
            "what their config builds until a run writes one[/yellow]"
        )
    else:
        console.print(
            "[yellow]No index: no build finished in this namespace; the "
            "doc-status registry's records below show how far one got[/yellow]"
        )
    origin = (
        "" if description.artifacts_from == "build_record" else " (from the config)"
    )
    console.print(
        f"Methodologies{origin}: {', '.join(description.methodologies) or 'none'}"
    )
    console.print(f"Strategies: {', '.join(description.strategies) or 'none'}")
    console.print(f"auto routes to: {', '.join(description.auto_routes) or 'none'}")
    for index in description.indices:
        if index.index is None:
            console.print(f"  {index.kind}: no index")
            continue
        model = (
            f", {index.embedding_model_id} ({index.embedding_dimension})"
            if index.embedding_model_id
            else ""
        )
        console.print(f"  {index.kind}: {index.documents} items{model}")
    if description.registry is not None:
        counts = ", ".join(
            f"{n} {status}" for status, n in description.registry.items()
        )
        console.print(f"Doc-status registry: {counts or 'no record'}")
    elif description.registry_unavailable is not None:
        console.print(
            "[yellow]Doc-status registry: unavailable "
            f"({markup_safe(description.registry_unavailable)})[/yellow]"
        )
    for document in description.documents or ():
        console.print(
            f"  {markup_safe(document.file_path or document.doc_id)}: "
            f"{document.status.value}, failures {document.failure_count}, "
            f"retries {document.retry_count}, attempts {document.attempt_count}, "
            f"written by {markup_safe(document.written_by or '-')} at "
            f"{document.updated_at or '-'}"
            + (f": {markup_safe(document.error_info)}" if document.error_info else "")
        )


@exits_on_sigterm
def main() -> int:
    load_cli_dotenv()
    args = _build_parser().parse_args()
    as_json = json_output(args)
    try:
        with exit_on_config_error():
            config = load_cli_config(args)
        setup_cli_logging(config, args)
        refuse_overwrites(
            [("--output-path", args.output_path)],
            force=args.force,
            config_path=args.config_path,
            config=config,
        )
        if error := missing_extras_error(config, seeds=False):
            usage_error(error)
        if args.documents and not config.doc_status_enabled:
            usage_error(
                "--documents lists the doc-status registry's documents: set "
                "providers.doc_status.enabled"
            )
        description = describe_namespace(config, args.suffix, documents=args.documents)
        # A namespace without an index is still described when the registry
        # holds records of it: a failed first build, whose documents
        # --documents lists. With neither, or to export, it is refused.
        if not description.exists and (
            not description.registry or args.output_path is not None
        ):
            usage_error(
                f"the index namespace '{description.namespace}' holds no index: "
                "check --suffix (or the config's index_value)",
                # As run-rag names a query on it.
                error_type=IndexNotFoundError.__name__,
                error_class=ErrorClass.INDEX_NOT_FOUND,
            )
        exported = None
        if args.output_path is not None:
            exported = export_graph(
                config,
                args.output_path,
                suffix=description.suffix,
                format=args.export_format,
                max_items=args.max_items,
                include_text=args.include_text,
                overwrite=True,  # refuse_overwrites checked it
            )
        if as_json:
            print_json(
                {
                    **description.model_dump(mode="json"),
                    "export": exported.model_dump(mode="json") if exported else None,
                }
            )
        else:
            _print(description)
            if exported is not None:
                console.print(
                    f"Exported {exported.counts} to {exported.path}"
                    + (
                        f" (cut at --max-items: {', '.join(exported.truncated)})"
                        if exported.truncated
                        else ""
                    )
                )
        return 0
    except KeyboardInterrupt:
        logger.warning("Describe interrupted by user.")
        raise SystemExit(EXIT_INTERRUPTED) from None
    except ConfigurationError as e:
        logger.error("Describe input error: %s", e)
        return failed(EXIT_USAGE, e)
    except StoreConnectionError as e:
        logger.error("Describe failed: a store could not be reached: %s", e)
        return failed(EXIT_FAILURE, e)
    except Exception as e:
        logger.error("Describe failed: %s", e)
        logger.debug("Traceback:", exc_info=True)
        return failed(EXIT_FAILURE, e)


if __name__ == "__main__":
    sys.exit(main())
