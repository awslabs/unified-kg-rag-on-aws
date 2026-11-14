# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, NoReturn

from rich.panel import Panel

from unified_kg_rag import __version__
from unified_kg_rag.application.cli.config_args import (
    add_config_arguments,
    exit_on_config_error,
    load_cli_config,
)
from unified_kg_rag.application.cli.conventions import (
    EXIT_CODES,
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    RUN_FAILED,
    ArgumentParser,
    add_output_format_argument,
    add_removed_flag,
    exits_on_sigterm,
    failed,
    failure_fields,
    interruptible,
    json_output,
    print_json,
    print_status,
    setup_cli_logging,
    shown_text,
    stoppable,
    usage_error,
)
from unified_kg_rag.application.cli.display import (
    PRODUCT_NAME,
    console,
    display_ascii_art,
    markup_safe,
    terminal_safe,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.lazy import lazy_names
from unified_kg_rag.application.cli.metrics_args import (
    add_metrics_sink_argument,
    build_metrics_sink,
)
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    missing_extras_error,
    strategy_roles,
)
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
)
from unified_kg_rag.application.cli.query_args import (
    add_query_arguments,
    namespace_name,
    query_suffix,
)
from unified_kg_rag.application.config import load_cli_dotenv
from unified_kg_rag.domain.models import Citation, SearchStrategy
from unified_kg_rag.domain.retrieval.store_names import namespace_suffix
from unified_kg_rag.shared import (
    ConfigurationError,
    InvalidFilterError,
    MethodologyNotIndexedError,
    get_logger,
)
from unified_kg_rag.shared.exceptions import (
    INTERNAL_ERROR_MESSAGE,
    QUERY_FAILURE_IDS,
    ErrorClass,
    IndexNotFoundError,
    InvalidArgumentError,
    classify,
    error_class_of,
    shown_error_text,
)
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.utils import event_loop

if TYPE_CHECKING:
    from unified_kg_rag.application.bootstrap import build_providers
    from unified_kg_rag.application.presentation.answer_view import (
        AnswerView,
        QueryOutcome,
        SourceView,
    )
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        RAGInput,
        RAGOutput,
        SearchOutput,
    )

logger = get_logger(__name__)


def _chain_module() -> ModuleType:
    from unified_kg_rag.application.retrieval import rag_chain

    return rag_chain


def _view_module() -> ModuleType:
    from unified_kg_rag.application.presentation import answer_view

    return answer_view


def _bootstrap_module() -> ModuleType:
    from unified_kg_rag.application import bootstrap

    return bootstrap


# Imported on first use: --help and a refused command line load no chain.
__getattr__, _load_chain = lazy_names(
    globals(),
    {
        "GraphRAGChain": _chain_module,
        "RAGInput": _chain_module,
        "RAGOutput": _chain_module,
        "SearchOutput": _chain_module,
        "AnswerView": _view_module,
        "QueryOutcome": _view_module,
        "SourceView": _view_module,
        "build_providers": _bootstrap_module,
    },
)


def _rerank_line(rerank: dict[str, Any]) -> str:
    """``metadata.rerank`` as one line: the overall status and reason, then
    each stage when a query reranked more than once."""
    line = str(rerank.get("status", "unknown"))
    if reason := rerank.get("reason"):
        line += f" ({reason})"
    stages = rerank.get("stages") or []
    if len(stages) > 1:
        line += (
            " ["
            + "; ".join(
                f"{stage.get('scope')}: {stage.get('status')}" for stage in stages
            )
            + "]"
        )
    return line


def _citation_line(citation: Citation) -> str:
    """One cited section as ``run-rag`` prints it: its marker, file and
    pages (a graph or model-written section: its first document's; else
    its seed graphs), and what the section is (a passage's title, its file
    name, is not named twice)."""
    first = citation.documents[0] if citation.documents else None
    where = (
        citation.file_path
        or citation.file_name
        or (first.file_path or first.file_name if first else None)
    )
    page, page_end = citation.page, citation.page_end
    if where and page:
        where += (
            f", p. {page}" if page_end in (None, page) else f", pp. {page}-{page_end}"
        )
    provenance = citation.provenance or {}
    if not where and provenance.get("seed_graphs"):
        where = "seed graph " + ", ".join(provenance["seed_graphs"])
    others = len(citation.documents) - 1
    if where and others > 0:
        where += f" (+{others} more)"
    title = citation.title
    named = title and title not in (citation.file_name, citation.file_path)
    detail = (
        f"{citation.section_type}: {title}"
        if named
        else (citation.section_type if citation.section_type != "text" else None)
    )
    parts = [p for p in (where, detail) if p] or [str(citation.source_id)]
    return f"[{citation.marker}] " + " - ".join(parts)


def print_citations(view: AnswerView) -> None:
    """The context sections the answer cites (``AnswerView.citations``)."""
    cited = [c for c in view.citations if c.cited]
    if not cited:
        return
    console.print(
        Panel(
            "\n".join(f"  {markup_safe(_citation_line(c))}" for c in cited),
            title=f"[cyan]Citations ({len(cited)})[/cyan]",
            border_style="cyan",
        )
    )


def _outcome_lines(outcome: QueryOutcome) -> list[str]:
    """What the answer (or search) was given without: its degraded steps
    (a partial answer) and the citation reads that failed."""
    lines = []
    if outcome.partial:
        steps = "; ".join(f"{d.step} ({d.reason})" for d in outcome.degraded)
        lines.append(f"Partial answer: given without {steps}")
    if outcome.citations_unresolved:
        steps = ", ".join(dict.fromkeys(d.step for d in outcome.citations_unresolved))
        lines.append(f"Some citations name no document: {steps} failed")
    return [terminal_safe(line) for line in lines]


def _filter_pair(item: str) -> str:
    """A ``--filters`` pair: ``key:value`` with a key and a value. Anything
    else is a usage error rather than a filter left out or matching nothing
    (which widened or emptied the search)."""
    key, colon, value = item.partition(":")
    if not colon or not key.strip() or not value.strip():
        raise argparse.ArgumentTypeError(
            f"invalid filter '{item}': expected 'key:value'"
        )
    return item


def _refuse_filter(flag: str, error: InvalidFilterError) -> NoReturn:
    """Exit 2 on a filter ``flag`` gave, of the class every surface names a
    malformed filter by (``invalid_filter``), naming the flag."""
    usage_error(
        f"{flag}: {error}",
        error_type=type(error).__name__,
        error_class=ErrorClass.INVALID_FILTER,
    )


def _filters_by_flag(
    args: argparse.Namespace, filters: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """``filters`` split by the flag each key came from (``--filters`` or
    ``--filters-json``; parse_args refused a key given in both)."""
    pairs = {pair.split(":", 1)[0] for pair in args.filters or []}
    split = [
        ("--filters", {k: v for k, v in filters.items() if k in pairs}),
        ("--filters-json", {k: v for k, v in filters.items() if k not in pairs}),
    ]
    return [(flag, given) for flag, given in split if given]


def _filters_json(value: str) -> dict[str, Any]:
    """argparse ``type`` of ``--filters-json``: a non-empty JSON object
    (its values checked as a query checks them, in ``parse_args``)."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"invalid JSON: {e.msg}") from None
    if not isinstance(parsed, dict) or not parsed:
        raise argparse.ArgumentTypeError(
            "expected a non-empty JSON object of filters, e.g. "
            '\'{"start_date": {"gte": "2022-07-01"}}\''
        )
    return parsed


# The hits --mode search prints without --verbose, and the files the
# sources line of an answer names.
_HITS_SHOWN = 20
_FILES_SHOWN = 5
_SNIPPET_CHARS = 160
# Plurals of the result types a sources line counts.
_PLURALS = {"entity": "entities", "community": "community reports"}
# Execution metrics whose key does not read as a label once title-cased.
_METRIC_LABELS = {
    "hl_keyword_count": "High-Level Keywords",
    "ll_keyword_count": "Low-Level Keywords",
}


def _hit_label(view: SourceView) -> str:
    """What a result is, for a person: the file of a text unit (and its
    chunk when the file has several), else its title (the entity or
    community name, a relationship's ends), else the result's id."""
    if file := view.file_name:
        attributes = view.metadata.get("attributes") or {}
        total = attributes.get("total_chunks")
        if isinstance(total, int) and total > 1:
            return f"{file} (chunk {attributes.get('chunk_id')}/{total})"
        return file
    return view.title or view.id or "N/A"


def _snippet(text: str) -> str:
    """``text`` on one line, cut to ``_SNIPPET_CHARS`` characters."""
    line = " ".join(str(text).split())
    if len(line) > _SNIPPET_CHARS:
        return line[: _SNIPPET_CHARS - 3] + "..."
    return line


def _print_hits(views: list[SourceView], *, limit: int | None) -> None:
    """Print search results, sources or written sections, numbered: type,
    label, score and a snippet of the text (printed literally)."""
    shown = views if limit is None else views[:limit]
    for i, view in enumerate(shown, 1):
        score = view.score
        scored = f"  (score {score:.3f})" if isinstance(score, int | float) else ""
        console.print(
            f"{i:>3}. [{view.kind}] {terminal_safe(_hit_label(view))}{scored}",
            markup=False,
            highlight=False,
        )
        if snippet := _snippet(terminal_safe(view.text)):
            console.print(f"     {snippet}", markup=False, highlight=False, style="dim")
    if len(views) > len(shown):
        console.print(
            f"     ... and {len(views) - len(shown)} more (--verbose lists them "
            "all; --output-format json prints every field)",
            markup=False,
            highlight=False,
        )


def _sources_line(sources: list[SourceView]) -> str:
    """One line for an answer's sources: the files its text came from and
    how many graph elements of each type it read."""
    files = list(dict.fromkeys(v.file_name for v in sources if v.file_name))
    counts: dict[str, int] = {}
    for view in sources:
        if view.file_name is None:
            counts[view.kind] = counts.get(view.kind, 0) + 1
    parts = []
    if files:
        named = ", ".join(files[:_FILES_SHOWN])
        if len(files) > _FILES_SHOWN:
            named += f" and {len(files) - _FILES_SHOWN} more"
        parts.append(f"files: {named}")
    if counts:
        parts.append(
            ", ".join(
                f"{n} {kind if n == 1 else _PLURALS.get(kind, kind + 's')}"
                for kind, n in counts.items()
            )
        )
    return terminal_safe(
        f"Sources ({len(sources)}): {'; '.join(parts)} (--verbose lists each)"
    )


def _answer_view(result: dict[str, Any]) -> AnswerView:
    """The view of an answer's JSON document (its ``RAGOutput`` fields)."""
    _load_chain()
    output = RAGOutput.model_validate(
        {key: result[key] for key in RAGOutput.model_fields if key in result}
    )
    return AnswerView.from_output(output)


def _search_views(result: dict[str, Any]) -> tuple[list[SourceView], QueryOutcome]:
    """The hits of a search's JSON document and its outcome."""
    _load_chain()
    output = SearchOutput.model_validate(
        {key: result[key] for key in SearchOutput.model_fields if key in result}
    )
    views = [
        SourceView.of_result(i, r)
        for i, r in enumerate(output.search_results.results, 1)
    ]
    return views, QueryOutcome.of_output(output)


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _typed_filters(
        filter_args: list[str] | None, filters_json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """``key:value`` pairs and the ``--filters-json`` object as one
        mapping, a boolean or numeric field's value converted to its type
        (``in_schema:false`` is ``False``), as the Python API would pass it.

        Raises:
            InvalidFilterError: A value its field's type cannot take, or one
                a query refuses (a null, a nested list, an empty range).
        """
        filters: dict[str, Any] = dict(item.split(":", 1) for item in filter_args or [])
        filters.update(filters_json or {})
        _load_chain()
        GraphRAGChain.check_filter_values(filters)
        return GraphRAGChain.typed_filter_values(filters) or {}

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = ArgumentParser(
            allow_abbrev=False,
            description=f"{PRODUCT_NAME}: answer a question (or, with --mode "
            "search, retrieve) from the knowledge graph and indexes run-ingestion "
            "built, with the GraphRAG, LightRAG, HippoRAG or RAPTOR strategies",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=doc_epilog(
                "guides/query.md#examples",
                'run-rag --query "What are the main themes?" --search-strategy global '
                "--config-path config.yaml",
            ),
        )

        parser.add_argument(
            "--query",
            "-q",
            help="Answer one query; give this or --interactive (exactly one)",
        )
        parser.add_argument(
            "--interactive",
            "-i",
            action="store_true",
            help="Launch interactive chat mode for continuous conversation "
            "(always 'rag' mode, memory on; not with --output-format json)",
        )
        parser.add_argument(
            "--mode",
            default="rag",
            choices=["rag", "search"],
            help="Set the execution mode: 'rag' for full generation, 'search' for "
            "retrieval only (a single --query only; --interactive always generates)",
        )
        parser.add_argument(
            "--conversation-id",
            help="Name the conversation. Memory is kept in this process only: it "
            "carries over between --interactive turns, not between separate "
            "run-rag invocations",
        )
        add_removed_flag(
            parser,
            "--use-memory",
            "an --interactive session always keeps its conversation memory, "
            "and a single --query starts with no history",
        )
        add_query_arguments(parser)
        parser.add_argument(
            "--disable-query-processing",
            action="store_true",
            help="Skip query translation and the query-side entity and keyword "
            "extraction (no LLM call before retrieval). mix and hybrid then "
            "have no high- or low-level keywords: a question shorter than "
            "search.lightrag_search.raw_query_fallback_max_len characters is "
            "searched as its own low-level keyword, a longer one retrieves no "
            "entities or relationships",
        )
        parser.add_argument(
            "--filters",
            nargs="*",
            action="extend",
            type=_filter_pair,
            help="Attribute filters as space-separated 'key:value' pairs (a "
            "repeated --filters adds its pairs); a boolean or numeric field "
            "takes its type's value (in_schema:false, rank:2)",
        )
        parser.add_argument(
            "--filters-json",
            type=_filters_json,
            metavar="JSON",
            help="Filters as one JSON object, for the values a pair cannot "
            "give: a list (any of its values) or a range with gt/gte/lt/lte "
            '(\'{"start_date": {"gte": "2022-07-01"}, "category": ["a", "b"]}\'), '
            "as the Python API and the MCP server take them; with --filters, "
            "each key in one of the two",
        )
        add_output_format_argument(parser)
        add_metrics_sink_argument(
            parser,
            "Where to send query metrics: 'none' (default) or 'cloudwatch' "
            "(CloudWatch EMF to stdout, to stderr with --output-format json: "
            "query_latency_ms, the outcome counts query_failed, query_degraded "
            "and query_abstained, and query_throttled_requests of each --mode "
            "rag query under unified_kg_rag/query, with an Environment "
            "dimension when UKR_METRICS_ENVIRONMENT is set and the Strategy "
            "that ran)",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Show more of each result: the query processing, every source "
            "(or search hit) with its score and text, and the metrics; with "
            "--output-format json, also an answer's search_results.results "
            "(the logs: --log-level)",
        )
        add_config_arguments(parser)
        add_provider_arguments(parser)
        return parser

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        if not args.interactive and args.query is None:
            self.parser.error("either --query or --interactive is required")
        if args.query is not None:
            # An empty question is refused as the chain refuses it
            # (RAGInput.checked: invalid_argument), before any model call.
            _load_chain()
            try:
                RAGInput.checked(query=args.query)
            except InvalidArgumentError as e:
                self.parser.error(f"--query: {str(e).removeprefix('query: ')}")
        if args.interactive and args.mode != "rag":
            self.parser.error(
                f"--mode {args.mode} is not supported with --interactive, which "
                "always generates answers; use --query for retrieval-only output"
            )
        keys = [pair.split(":", 1)[0] for pair in args.filters or []]
        keys += list(args.filters_json or {})
        if repeated := sorted({key for key in keys if keys.count(key) > 1}):
            self.parser.error(
                f"--filters gives {', '.join(repeated)} more than once; give "
                "each key once (in --filters or --filters-json)"
            )
        for flag, pairs, given in (
            ("--filters", args.filters, None),
            ("--filters-json", None, args.filters_json),
        ):
            try:
                # A value its field's type cannot take (in_schema:maybe) is a
                # usage error, not a filter matching nothing.
                CommandLineInterface._typed_filters(pairs, given)
            except InvalidFilterError as e:
                _refuse_filter(flag, e)
        if args.interactive and args.output_format == "json":
            self.parser.error(
                "--output-format json is not supported with --interactive, a "
                "chat on the terminal; use --query for JSON output"
            )
        return args


# The interactive commands that take an argument (see _handle_help).
_ARGUMENT_COMMANDS = frozenset({"set-filter"})


class RAGChainRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        _load_chain()
        self.args = args
        with exit_on_config_error():
            self.config = load_cli_config(args)
        # JSON mode keeps stdout machine-parseable (pipeable to jq): the banner,
        # status lines and logs all go to stderr; only the JSON goes to stdout.
        self.json_output = json_output(args)
        setup_cli_logging(self.config, args)
        self.metrics_sink = build_metrics_sink(
            args.metrics_sink, query=True, stdout_taken=self.json_output
        )
        self.rag_chain: GraphRAGChain | None = None
        self.interactive_commands: dict[str, Callable[..., Any]] = {
            "help": self._handle_help,
            "new": self._handle_new_conversation,
            "set-filter": self._handle_set_filter,
            "clear-filters": self._handle_clear_filters,
            "show-config": self._handle_show_config,
        }

    def _get_suffix_display(self) -> str:
        """The namespace line: what the store names end in, the index suffix
        queried plus ``indexing.additional_suffix`` (as run-ingestion names
        the namespace it writes)."""
        chosen = query_suffix(self.args, self.config)
        name = namespace_name(chosen, self.config)
        color = "gray" if chosen == namespace_suffix(None) else "orange3"
        return f"[dim]Index Namespace:[/dim] [{color}]{markup_safe(name)}[/{color}]"

    async def run(self) -> None:
        # Only run-ingestion loads the seed graphs.
        extras = missing_extras_error(self.config, seeds=False)
        error = extras or missing_endpoints_error(
            self.config,
            strategy_roles(self.config, SearchStrategy(self.args.search_strategy)),
            f"search strategy '{self.args.search_strategy}'",
        )
        if error:
            usage_error(error)
        loop = asyncio.get_running_loop()
        executor = event_loop.configure_event_loop(
            loop, self.config.processing.io_workers
        )
        display_ascii_art(version=__version__)
        self._display_run_info()
        await self._initialize_chain()

        try:
            if self.args.interactive:
                await self._interactive_mode()
            else:
                if self.args.conversation_id:
                    logger.warning(
                        "--conversation-id has no effect on a single --query: conversation memory is kept in the "
                        "process only, so this query starts with no history "
                        "and its turn is not kept for a later run-rag. Use "
                        "--interactive for multi-turn conversations."
                    )
                rag_input = self._rag_input(
                    self.args.query,
                    conversation_id=self.args.conversation_id,
                    use_memory=False,
                    filters=self._parse_filters(
                        self.args.filters, self.args.filters_json
                    ),
                )

                result = await self._run_query(rag_input)
                if self.json_output:
                    print_json(self._json_document(result))
                else:
                    self._print_result(result, self.args.verbose)

                if not result.get("success", False):
                    sys.exit(EXIT_FAILURE)
        finally:
            # Release Neptune/OpenSearch sockets when the query/session ends so
            # a finished process does not leak the retrievers' connection pools.
            if self.rag_chain is not None:
                await self.rag_chain.aclose()
            # A query that timed out leaves its model call running on a worker
            # thread until the call's read timeout; the process exits without
            # waiting for it.
            event_loop.abandon_default_executor(loop, executor)

    async def _initialize_chain(self) -> None:
        console.print("[bold]Initializing RAG chain...[/bold]")
        _load_chain()
        try:
            # The chain publishes each answer's latency and outcome through
            # the sink of its providers (--metrics-sink).
            self.rag_chain = GraphRAGChain(
                config=self.config,
                providers=build_providers(self.config, metrics_sink=self.metrics_sink),
            )
            console.print("[green]Query chain initialized.[/green]")
        except Exception as e:  # noqa: BLE001 - logged and reported by class
            # Classified as a query's error is (_run_query): a setting the
            # chain refused (observability on without the otel extra, a bad
            # guardrail setting) is a usage error, exit 2.
            error_class = classify(e)
            exit_code = EXIT_CODES[error_class]
            if exit_code == EXIT_USAGE:
                usage_error(e)
            # The one rule of every surface: an unexpected failure's type and
            # where it happened logged (log_error), its text at DEBUG only,
            # masked on stdout and in the JSON document (shown_text); any
            # other class with its text.
            log_error(
                logger, logging.ERROR, "Failed to initialize RAG chain: %s", error=e
            )
            console.print(
                f"[red]Error during initialization: {markup_safe(shown_text(e))}[/red]"
            )
            print_status(exit_code, e)
            sys.exit(exit_code)

    async def _interactive_mode(self) -> None:
        console.print(
            "[bold blue]\nInteractive mode[/bold blue]\nType 'quit', 'exit' to end, "
            "or 'help' for commands."
        )
        console.print("---")

        filters = self._parse_filters(self.args.filters, self.args.filters_json)
        # A strategy the namespace cannot serve failed every question too.
        if self.rag_chain is not None:
            try:
                await self.rag_chain.acheck_methodology(
                    SearchStrategy(self.args.search_strategy),
                    query_suffix(self.args, self.config),
                )
            except MethodologyNotIndexedError as e:
                usage_error(f"--search-strategy: {e}")
            except IndexNotFoundError as e:
                usage_error(e)
        # An undeclared key failed every question of the session.
        if filters and self.rag_chain is not None:
            for flag, given in _filters_by_flag(self.args, filters):
                try:
                    self.rag_chain.check_filter_keys(
                        given, SearchStrategy(self.args.search_strategy)
                    )
                except InvalidFilterError as e:
                    _refuse_filter(flag, e)
        state = {
            "conversation_id": self.args.conversation_id or str(uuid.uuid4()),
            "filters": filters or {},
        }

        while True:
            try:
                # A stop while the prompt waits ends the session at once.
                with interruptible():
                    question = console.input("\n[bold cyan]Question:[/bold cyan] ")
                raw_input = question.strip()
                if not raw_input:
                    continue
                if raw_input.lower() in ["quit", "exit"]:
                    console.print("[yellow]Ending conversation.[/yellow]")
                    break

                # The rest of the line stays one argument, whitespace kept
                # (a filter value may hold several spaces).
                parts = raw_input.split(maxsplit=1)
                command, cmd_args = parts[0].lower(), parts[1:]

                # A command is the whole line, but for one that takes an
                # argument: "help me find ..." is a question.
                handler = self.interactive_commands.get(command)
                if handler and (not cmd_args or command in _ARGUMENT_COMMANDS):
                    if command == "new":
                        state["conversation_id"] = handler()
                    else:
                        handler(state=state, args=cmd_args)
                    continue
                rag_input = self._rag_input(
                    raw_input,
                    conversation_id=(
                        str(state["conversation_id"])
                        if state["conversation_id"] is not None
                        else None
                    ),
                    use_memory=True,
                    filters=(
                        state["filters"] if isinstance(state["filters"], dict) else None
                    ),
                )
                result = await self._run_query(rag_input)
                self._print_result(result, verbose=self.args.verbose)
                state["conversation_id"] = result.get(
                    "conversation_id", state["conversation_id"]
                )

            except (KeyboardInterrupt, EOFError):
                # Ctrl-C or Ctrl-D / piped-EOF. Without the EOFError case the
                # now-closed stdin re-raises EOF on every subsequent
                # console.input, spinning the generic handler below forever.
                console.print("\n[yellow]Conversation interrupted.[/yellow]")
                break
            except Exception as e:  # noqa: BLE001 - logged, the session goes on
                console.print(
                    f"[red]Unexpected error: {markup_safe(shown_text(e))}[/red]"
                )
                log_error(
                    logger,
                    logging.ERROR,
                    "Unexpected error in interactive mode: %s",
                    error=e,
                )

    def _rag_input(
        self,
        query: str,
        *,
        conversation_id: str | None,
        use_memory: bool,
        filters: dict[str, Any] | None,
    ) -> RAGInput:
        """A query with the command line's search settings."""
        _load_chain()
        return RAGInput.checked(
            query=query,
            suffix=query_suffix(self.args, self.config),
            search_strategy=self.args.search_strategy,
            search_type=self.args.search_type,
            conversation_id=conversation_id,
            use_memory=use_memory,
            filters=filters,
            top_k=self.args.top_k,
            retrieval_multiplier=self.args.retrieval_multiplier,
            enable_query_processing=not self.args.disable_query_processing,
        )

    async def _run_query(self, rag_input: RAGInput) -> dict[str, Any]:
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized")
        try:
            console.print(
                f"[bold]Executing RAG query in '{self.args.mode}' mode...[/bold]",
                "[dim]This may take some time depending on query complexity[/dim]",
            )
            # The chain publishes an answer's latency and outcome through
            # the metrics sink its providers hold (--metrics-sink).
            result = (
                await self.rag_chain.asearch(rag_input)
                if self.args.mode == "search"
                else await self.rag_chain.ainvoke(rag_input)
            )
            dumped = result.model_dump()
            outcome, fields = self._view_fields(result, dumped)
            output = {**dumped, **fields}
            # With processing.ignore_errors the chain does not raise: it returns
            # its error fallback flagged by metadata.error. That is a failure.
            if outcome.error_class is not None:
                error_type = outcome.error_type or RUN_FAILED
                error_class = error_class_of(outcome.error_class)
                # The absorbed error's text as a raised one's is shown
                # (shown_text): an unexpected failure's masked.
                error = shown_error_text(
                    error_class,
                    outcome.error_message or "RAG chain returned an error",
                    audience="operator",
                )
                if error == INTERNAL_ERROR_MESSAGE:
                    error = f"{error_type}: {error}"  # as operator_error_text
                # The ids of the query it absorbed, as a raised one names them.
                ids = {
                    name: value
                    for name in QUERY_FAILURE_IDS
                    if isinstance(value := result.metadata.get(name), str)
                }
                if EXIT_CODES[error_class] == EXIT_USAGE and not self.args.interactive:
                    # As raised without ignore_errors (below).
                    usage_error(
                        error, error_type=error_type, error_class=error_class, **ids
                    )
                # Printed by _print_result (text output) or in the document.
                return {
                    "success": False,
                    **output,
                    **failure_fields(
                        EXIT_FAILURE, error, error_type, error_class=error_class
                    ),
                    **ids,
                }
            console.print("\n[bold green]Query executed successfully![/bold green]")
            return {"success": True, **output}

        except Exception as e:  # noqa: BLE001 - logged and reported by class
            error_class = classify(e)
            if EXIT_CODES[error_class] == EXIT_USAGE and not self.args.interactive:
                # A --filters key no index declares, a --search-strategy
                # whose methodology the namespace does not hold, a setting
                # the index cannot serve (an embedding dimension it does not
                # hold) or a --suffix nothing was ingested under: the command
                # line or the configuration is wrong (exit 2).
                usage_error(e)
            # The one rule of every surface: an unexpected failure's type and
            # where it happened logged (log_error), its text at DEBUG only,
            # masked in the document _print_result prints (shown_text); any
            # other class with its text.
            log_error(
                logger, logging.ERROR, "Error during query execution: %s", error=e
            )
            return {
                "success": False,
                **failure_fields(EXIT_FAILURE, e),
                "conversation_id": rag_input.conversation_id,
            }

    @staticmethod
    def _view_fields(
        result: RAGOutput | SearchOutput, dumped: dict[str, Any]
    ) -> tuple[QueryOutcome, dict[str, Any]]:
        """The outcome of ``result`` and what the JSON document adds to the
        output, from the one view of an answer every surface reads
        (``AnswerView``): what it was given without (``degraded``,
        ``degraded_error``, ``citations_unresolved``), the class of an
        absorbed error (``error_class``) and, for an answer, the
        model-written sections it may cite (``written``)."""
        _load_chain()
        if not isinstance(result, RAGOutput):
            outcome = QueryOutcome.of(
                dumped.get("metadata"),
                (dumped.get("search_results") or {}).get("metadata"),
            )
            return outcome, outcome.outcome_fields()
        view = AnswerView.from_output(result)
        return view, {
            **view.outcome_fields(),
            "written": [w.model_dump(mode="json") for w in view.written],
        }

    def _json_document(self, result: dict[str, Any]) -> dict[str, Any]:
        """The ``--output-format json`` document of a query: an answer's
        ``search_results`` without its ``results``, which repeat the
        ``sources`` (``--verbose`` keeps them: every retrieved result, also
        those the context budget left out)."""
        search_results = result.get("search_results")
        if (
            self.args.verbose
            or "sources" not in result
            or not isinstance(search_results, dict)
        ):
            return result
        kept = {k: v for k, v in search_results.items() if k != "results"}
        return {**result, "search_results": kept}

    def _settings_lines(self) -> list[str]:
        """The search settings, as the run banner and show-config print them."""
        return [
            f"[dim]Search Strategy:[/dim] {self.args.search_strategy}",
            f"[dim]Search Type:[/dim] {self.args.search_type}",
            f"[dim]Top K Results:[/dim] {self.args.top_k}",
            f"[dim]Retrieval Multiplier:[/dim] {self.args.retrieval_multiplier}",
            f"[dim]Conversation Memory:[/dim] {self.args.interactive}",
            f"[dim]Query Processing:[/dim] {not self.args.disable_query_processing}",
            f"[dim]Verbose Output:[/dim] {self.args.verbose}",
            self._get_suffix_display(),
        ]

    def _display_run_info(self) -> None:
        config_lines = [
            "[bold blue]RAG Configuration[/bold blue]",
            "[bold]Current Configuration[/bold]",
            f"[dim]Execution Mode:[/dim] {self.args.mode}",
            *self._settings_lines(),
        ]

        if self.args.filters:
            config_lines.append(
                f"[dim]Filters:[/dim] {markup_safe(', '.join(self.args.filters))}"
            )

        if self.args.conversation_id:
            config_lines.append(
                f"[dim]Conversation ID:[/dim] {markup_safe(self.args.conversation_id)}"
            )

        if self.args.query:
            config_lines.append(f"[dim]Query:[/dim] {markup_safe(self.args.query)}")

        console.print(Panel.fit("\n".join(config_lines), border_style="blue"))

    @staticmethod
    def _parse_filters(
        filter_args: list[str] | None, filters_json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """``--filters`` and ``--filters-json`` as one mapping (each pair
        checked by ``_filter_pair``), boolean and numeric fields typed
        (``CommandLineInterface.parse_args`` refused a value that is not
        one)."""
        return CommandLineInterface._typed_filters(filter_args, filters_json)

    @staticmethod
    def _print_result(result: dict[str, Any], verbose: bool = False) -> None:
        """Print a query's JSON document (``_run_query``) for a person,
        from its ``AnswerView`` (an answer) or its hits' ``SourceView``."""
        if not result.get("success", False):
            error = markup_safe(result.get("error", "Unknown error"))
            console.print(f"\n[red]Error: {error}[/red]")
            return

        view: AnswerView | None = None
        outcome: QueryOutcome
        if "answer" in result:
            view = outcome = _answer_view(result)
            console.print("\n\n" + "=" * 50)
            console.print("[bold blue]Final Answer:[/bold blue]")
            # Model output: never interpret it as markup.
            console.print(terminal_safe(view.answer), markup=False, highlight=False)
            console.print("=" * 50)
            print_citations(view)
            if view.sources and not verbose:
                console.print(
                    _sources_line(view.sources), markup=False, highlight=False
                )
        else:
            # --mode search: the hits are the result.
            hits, outcome = _search_views(result)
            console.print("\n\n" + "=" * 50)
            console.print(f"[bold blue]Search Results ({len(hits)})[/bold blue]")
            console.print("=" * 50)
            _print_hits(hits, limit=None if verbose else _HITS_SHOWN)
        for line in _outcome_lines(outcome):
            console.print(line, markup=False, highlight=False, style="yellow")

        if verbose:
            console.print("\n[bold]Details:[/bold]")
            RAGChainRunner._print_query_processing(result.get("processed_query"))
            if view is not None and view.sources:
                console.print(
                    f"\n[bold cyan]Reference Sources ({len(view.sources)})[/bold cyan]"
                )
                _print_hits(view.sources, limit=None)
            if view is not None and view.written:
                # Model-written sections the answer may cite, no source.
                console.print(
                    "\n[bold cyan]Written Sections "
                    f"({len(view.written)})[/bold cyan]"
                )
                _print_hits(view.written, limit=None)
            RAGChainRunner._print_metrics(result)

    @staticmethod
    def _print_query_processing(pq: dict[str, Any] | None) -> None:
        if not pq:
            return
        pq_panel_content = ""
        if original_query := pq.get("original_query"):
            pq_panel_content += f"  - Original Query: {markup_safe(original_query)}\n"
        if translated_query := pq.get("translated_query"):
            pq_panel_content += (
                f"  - Translated Query: {markup_safe(translated_query)}\n"
            )
        if entities := pq.get("entities"):
            pq_panel_content += f"  - Extracted Entities: {len(entities)} found\n"
            if len(entities) <= 5:
                joined = markup_safe(", ".join(map(str, entities)))
                pq_panel_content += f"    {joined}\n"
        if pq_panel_content:
            console.print(
                Panel(
                    pq_panel_content.rstrip(),
                    title="[cyan]Query Processing Info[/cyan]",
                    border_style="cyan",
                )
            )

    @staticmethod
    def _print_metrics(result: dict[str, Any]) -> None:
        search_result = result.get("search_results", {})
        metadata = result.get("metadata", {})

        metrics_content = ""
        all_metadata = {**search_result, **metadata}

        for key, value in all_metadata.items():
            if key in ["results", "query"]:
                continue
            if key == "rerank" and isinstance(value, dict):
                metrics_content += f"  - Rerank: {markup_safe(_rerank_line(value))}\n"
            elif key == "search_strategy":
                metrics_content += f"  - Search Strategy: {value}\n"
            elif key == "processing_time":
                metrics_content += f"  - Processing Time: {value:.2f}s\n"
            elif isinstance(value, int | float | str | bool) or (
                isinstance(value, list | dict) and len(str(value)) < 100
            ):
                label = _METRIC_LABELS.get(key) or key.replace("_", " ").title()
                metrics_content += f"  - {label}: {markup_safe(value)}\n"

        if metrics_content:
            console.print(
                Panel(
                    metrics_content.rstrip(),
                    title="[cyan]Execution Metrics[/cyan]",
                    border_style="cyan",
                )
            )

    @staticmethod
    def _handle_help(**kwargs: Any) -> None:
        console.print("[bold]Available commands:[/bold]")
        console.print("  quit/exit      - End the conversation")
        console.print("  new            - Start a new conversation")
        console.print("  help           - Show this help message")
        console.print(
            "  set-filter <k:v> - Add/update filter (e.g., set-filter type:person)"
        )
        console.print("  clear-filters  - Remove all filters")
        console.print("  show-config    - Display current configuration")

    @staticmethod
    def _handle_new_conversation(**kwargs: Any) -> str:
        new_id = str(uuid.uuid4())
        console.print(
            f"[green]Starting new conversation. (ID: '{new_id[:8]}...')[/green]"
        )
        return new_id

    def _handle_set_filter(self, state: dict[str, Any], args: list[str]) -> None:
        try:
            pair = _filter_pair(" ".join(args))
        except argparse.ArgumentTypeError:
            console.print("[red]Invalid filter format. Use 'key:value'[/red]")
            return
        try:
            ((key, value),) = CommandLineInterface._typed_filters([pair]).items()
        except InvalidFilterError as e:
            console.print(f"[red]Filter not set: {markup_safe(e)}[/red]")
            return
        # Rejected now, not by every later question until clear-filters.
        # Checked with the filters already set: under auto, together they
        # may leave no strategy that applies them all.
        if self.rag_chain is not None:
            try:
                self.rag_chain.check_filter_keys(
                    {**state["filters"], key: value},
                    SearchStrategy(self.args.search_strategy),
                )
            except InvalidFilterError as e:
                console.print(f"[red]Filter not set: {markup_safe(e)}[/red]")
                return
        state["filters"][key] = value
        console.print(
            f"[green]Filter added: '{markup_safe(key)}' = '{markup_safe(value)}'"
            "[/green]"
        )

    @staticmethod
    def _handle_clear_filters(state: dict[str, Any], **kwargs: Any) -> None:
        state["filters"].clear()
        console.print("[green]All filters cleared.[/green]")

    def _handle_show_config(self, state: dict[str, Any], **kwargs: Any) -> None:
        config_lines = [
            "[bold]Current Configuration:[/bold]",
            "[dim]Execution Mode:[/dim] rag (Interactive mode)",
            *self._settings_lines(),
        ]

        if state["filters"]:
            filters_str = ", ".join([f"{k}:{v}" for k, v in state["filters"].items()])
            config_lines.append(f"[dim]Filters:[/dim] {markup_safe(filters_str)}")
        else:
            config_lines.append("[dim]Filters:[/dim] None")

        console.print(Panel.fit("\n".join(config_lines), border_style="blue"))


async def async_main(args: argparse.Namespace) -> int:
    """Run the CLI and return its exit code.

    Never lets ``SystemExit`` escape the task that ``asyncio.run`` drives:
    the exit code is returned, so the loop ends normally.
    """
    try:
        runner = RAGChainRunner(args)
        await runner.run()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_FAILURE
    except KeyboardInterrupt:
        console.print("\n[yellow]Execution interrupted by user.[/yellow]")
        return EXIT_INTERRUPTED
    except ConfigurationError as e:
        console.print(f"\n[red]Error: {markup_safe(e)}[/red]")
        log_error(logger, logging.ERROR, "Configuration error: %s", error=e)
        return failed(EXIT_USAGE, e)
    except Exception as e:  # noqa: BLE001 - logged, exit code
        console.print(
            f"\n[red]An unexpected error occurred: {markup_safe(shown_text(e))}[/red]"
        )
        log_error(
            logger, logging.ERROR, "Unexpected error during execution: %s", error=e
        )
        return failed(EXIT_FAILURE, e)


@exits_on_sigterm
def main() -> None:
    load_cli_dotenv()
    # Parse (and let argparse exit on --help or bad arguments) before the event
    # loop starts.
    args = CommandLineInterface().parse_args()
    try:
        code = asyncio.run(stoppable(async_main(args)))
    except KeyboardInterrupt:
        # A Ctrl-C while the loop runs cancels the task (exits_on_sigterm);
        # one raises KeyboardInterrupt here only outside it, defensively kept.
        console.print("\n[yellow]Execution interrupted by user.[/yellow]")
        code = EXIT_INTERRUPTED
    sys.exit(code)


if __name__ == "__main__":
    main()
