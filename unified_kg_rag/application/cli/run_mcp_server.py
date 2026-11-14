# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-mcp-server``: serve one index namespace to agents as read-only MCP
tools (docs/guides/mcp.md).

stdio by default: stdout carries only the protocol's messages, so the logs
and everything else go to stderr, and the command has no
``--output-format``. ``--transport streamable-http`` serves stateless
streamable HTTP on ``127.0.0.1:8000`` (``/mcp``); another address needs
``--allow-remote``, as the server authenticates no one. Exit codes: 0 once
the client closes stdio (or the HTTP server stops), 2 for a usage or
configuration error (the ``mcp`` extra missing among them), 1 for a
failure, 143 on SIGTERM and 130 on Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from unified_kg_rag.application.bootstrap import build_providers
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
    add_log_level_and_version_arguments,
    exits_on_sigterm,
    failed,
    is_loopback,
    stoppable,
    usage_error,
)
from unified_kg_rag.application.cli.display import console
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.mcp_server import (
    build_mcp_app,
    build_mcp_server,
    loopback_hosts,
    mcp_extra_error,
)
from unified_kg_rag.application.cli.mcp_tools import KnowledgeGraphTools
from unified_kg_rag.application.cli.metrics_args import (
    add_metrics_sink_argument,
    build_metrics_sink,
)
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    missing_extras_error,
)
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
)
from unified_kg_rag.application.config import load_cli_dotenv
from unified_kg_rag.application.retrieval import graph_lookup
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    RetrieverRole,
    validate_safe_name,
)
from unified_kg_rag.shared import (
    ConfigurationError,
    IndexNotFoundError,
    get_logger,
    setup_logging,
)
from unified_kg_rag.shared.exceptions import error_text
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.utils import event_loop

logger = get_logger(__name__)

TRANSPORTS = ("stdio", "streamable-http")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _suffix(value: str) -> str:
    try:
        return validate_safe_name(value, "--suffix")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"must be between 1 and 65535, got {port}")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        prog="run-mcp-server",
        allow_abbrev=False,
        description=(
            "Serve one index namespace to agents as read-only MCP (Model "
            "Context Protocol) tools: kg_search, kg_answer, kg_get_entity, "
            "kg_neighbors and kg_describe_schema. Needs the mcp extra"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "guides/mcp.md",
            "run-mcp-server --config-path config.yaml --suffix default",
        ),
    )
    parser.add_argument(
        "--suffix",
        type=_suffix,
        help="Index namespace (suffix) to serve (default: the config's "
        "processing.document_parsing.index_value, else 'default'); a tool "
        "call may name another only if mcp.allowed_namespaces lists it",
    )
    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default="stdio",
        help="stdio (default): the agent starts this command and talks over "
        "its stdin and stdout. streamable-http: a stateless HTTP endpoint at "
        "http://HOST:PORT/mcp",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Address the HTTP transport binds (default {DEFAULT_HOST}, this "
        "machine only); a non-loopback address needs --allow-remote",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"Port of the HTTP transport (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Let the HTTP transport bind a non-loopback address. The server "
        "authenticates no caller: put it behind a proxy or gateway that "
        "authenticates every request, and keep the port unreachable otherwise",
    )
    add_metrics_sink_argument(
        parser,
        "Where to send query metrics: 'none' (default) or 'cloudwatch' "
        "(CloudWatch EMF to stderr, with the logs, as stdout is the protocol's: "
        "query_latency_ms, the outcome counts query_failed, query_degraded "
        "and query_abstained, and query_throttled_requests of each kg_answer "
        "call under unified_kg_rag/query, "
        "with an Environment dimension when UKR_METRICS_ENVIRONMENT is set and "
        "the Strategy that ran), which the CDK stack's query alarms read",
    )
    add_log_level_and_version_arguments(parser)
    add_config_arguments(parser)
    add_provider_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.transport == "stdio" and (
        args.host != DEFAULT_HOST or args.port != DEFAULT_PORT or args.allow_remote
    ):
        parser.error(
            "--host, --port and --allow-remote apply to --transport streamable-http"
        )
    if not is_loopback(args.host) and not args.allow_remote:
        parser.error(
            f"--host {args.host} is not a loopback address: the server "
            "authenticates no caller, so binding it needs --allow-remote (and "
            "an authenticating proxy in front)"
        )
    return args


async def serve_http(tools: KnowledgeGraphTools, host: str, port: int) -> None:
    """Serve ``tools`` over stateless streamable HTTP on ``host:port``.

    Stateless: the tools keep no session state (an AgentCore Runtime MCP
    server is stateless too). On a loopback address the ``Host`` and
    ``Origin`` headers must name it or another loopback name (DNS
    rebinding: a web page cannot reach the server through a name it
    controls), whatever loopback address it is; on another address,
    reached through a proxy, they are not checked. uvicorn's logs,
    access lines included, go to the logging setup (stderr).
    """
    import uvicorn

    app = build_mcp_app(
        tools, allowed_hosts=loopback_hosts(host) if is_loopback(host) else None
    )
    config = uvicorn.Config(app, host=host, port=port, log_config=None)
    await uvicorn.Server(config).serve()


async def warn_of_missing_namespaces(
    chain: GraphRAGChain, namespaces: list[str]
) -> None:
    """WARN of each served namespace nothing was ingested under (a mistyped
    ``--suffix``): the server starts, as one may be ingested while it runs,
    but every tool call on it fails until then."""
    for namespace in namespaces:
        try:
            await graph_lookup.require_namespace(chain, namespace)
        except IndexNotFoundError as e:
            log_error(
                logger,
                logging.WARNING,
                "Namespace '%s' holds no index, so every tool call on it fails "
                "until it is ingested: %s",
                namespace,
                error=e,
            )
        except Exception as e:  # noqa: BLE001 - a call meets it again
            # A store down with its text, an unexpected failure by its type
            # and where it happened (log_error).
            log_error(
                logger,
                logging.WARNING,
                "Could not check namespace '%s': %s",
                namespace,
                error=e,
            )


async def serve(args: argparse.Namespace, config: Config) -> int:
    """Build the chain and the server, and serve until the transport ends."""
    loop = asyncio.get_running_loop()
    executor = event_loop.configure_event_loop(loop, config.processing.io_workers)
    # The query metrics go to stderr: stdout is the stdio protocol's.
    sink = build_metrics_sink(args.metrics_sink, query=True, stdout_taken=True)
    chain = GraphRAGChain(
        config=config,
        providers=build_providers(config, metrics_sink=sink),
        allowed_strategies=config.mcp.strategies,
    )
    try:
        tools = KnowledgeGraphTools(chain, namespace=args.suffix)
        await warn_of_missing_namespaces(chain, tools.namespaces)
        logger.info(
            "Serving namespace '%s' read-only over %s (tools: %s)",
            tools.namespace,
            args.transport,
            ", ".join(config.mcp.tools),
        )
        if args.transport == "stdio":
            await build_mcp_server(tools).run_stdio_async()
        else:
            await serve_http(tools, args.host, args.port)
        return 0
    finally:
        await chain.aclose()
        event_loop.abandon_default_executor(loop, executor)


async def async_main(args: argparse.Namespace) -> int:
    try:
        with exit_on_config_error():
            config = load_cli_config(args)
        # stdout is the protocol's: everything else goes to stderr.
        console.stderr = True
        level = {"level": args.log_level} if args.log_level else None
        setup_logging(config, level, stream=sys.stderr)
        error = (
            mcp_extra_error()
            or missing_extras_error(config, seeds=False)
            or missing_endpoints_error(
                config, {RetrieverRole.GRAPH, RetrieverRole.DOCUMENT}, "the MCP server"
            )
        )
        if error:
            usage_error(error)
        return await serve(args, config)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_FAILURE
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except ConfigurationError as e:
        log_error(logger, logging.ERROR, "Configuration error: %s", error=e)
        print(f"run-mcp-server: error: {error_text(e)}", file=sys.stderr)
        return failed(EXIT_USAGE, e)
    except Exception as e:  # noqa: BLE001 - logged, exit code
        # The one rule of every surface (log_error): an unexpected failure's
        # type and where it happened (text and traceback at DEBUG).
        log_error(logger, logging.ERROR, "The MCP server failed: %s", error=e)
        return failed(EXIT_FAILURE, e)


@exits_on_sigterm
def main(argv: list[str] | None = None) -> Any:
    load_cli_dotenv()
    # stdout is the protocol's from the first byte: the console too.
    console.stderr = True
    args = parse_args(argv)
    try:
        code = asyncio.run(stoppable(async_main(args)))
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    sys.exit(code)


if __name__ == "__main__":
    main()
