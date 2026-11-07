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
import ipaddress
import sys
from typing import Any

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
    exits_on_sigterm,
    failed,
    stoppable,
    usage_error,
)
from unified_kg_rag.application.cli.display import console
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.mcp_server import build_mcp_server, mcp_extra_error
from unified_kg_rag.application.cli.mcp_tools import KnowledgeGraphTools
from unified_kg_rag.application.cli.preflight import (
    missing_endpoints_error,
    missing_extras_error,
)
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.application.config import load_cli_dotenv
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    RetrieverRole,
    SearchStrategy,
    validate_safe_name,
)
from unified_kg_rag.shared import ConfigurationError, get_logger, setup_logging
from unified_kg_rag.shared.utils import event_loop

logger = get_logger(__name__)

TRANSPORTS = ("stdio", "streamable-http")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
_LOOPBACK_NAMES = frozenset({"localhost"})


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


def is_loopback(host: str) -> bool:
    """Whether ``host`` names a loopback address (``127.0.0.0/8``, ``::1``,
    ``localhost``): reachable from this machine only."""
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


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


def narrowed_config(config: Config) -> Config:
    """``config`` with ``auto`` routing only among ``mcp.strategies``, when
    the list allows ``auto`` (a strategy outside it could otherwise be
    picked by the router).

    Raises:
        ConfigurationError: No routable strategy is in the list.
    """
    allowed = config.mcp.strategies
    if not allowed or SearchStrategy.AUTO not in allowed:
        return config
    routable = config.search.auto_routable_strategies
    kept = [s for s in routable if s in allowed]
    if kept == routable:
        return config
    if not kept:
        raise ConfigurationError(
            "mcp.strategies allows auto but none of "
            "search.auto_routable_strategies "
            f"({', '.join(s.value for s in routable)}); list one of them"
        )
    narrowed = config.model_copy(deep=True)
    narrowed.search.auto_routable_strategies = kept
    logger.info(
        "auto routes among %s (mcp.strategies)", ", ".join(s.value for s in kept)
    )
    return narrowed


async def serve(args: argparse.Namespace, config: Config) -> int:
    """Build the chain and the server, and serve until the transport ends."""
    loop = asyncio.get_running_loop()
    executor = event_loop.configure_event_loop(loop, config.processing.io_workers)
    chain = GraphRAGChain(config=config)
    try:
        tools = KnowledgeGraphTools(chain, namespace=args.suffix)
        server = build_mcp_server(tools)
        logger.info(
            "Serving namespace '%s' read-only over %s (tools: %s)",
            tools.namespace,
            args.transport,
            ", ".join(config.mcp.tools),
        )
        if args.transport == "stdio":
            await server.run_stdio_async()
        else:
            # Stateless: no session state to keep or route to, as the tools
            # keep none (an AgentCore Runtime MCP server is stateless too).
            await server.run_streamable_http_async(
                host=args.host,
                port=args.port,
                stateless_http=True,
                json_response=True,
            )
        return 0
    finally:
        await chain.aclose()
        event_loop.abandon_default_executor(loop, executor)


async def async_main(args: argparse.Namespace) -> int:
    try:
        with exit_on_config_error():
            config = narrowed_config(
                apply_provider_arguments(load_cli_config(args), args)
            )
        # stdout is the protocol's: everything else goes to stderr.
        console.stderr = True
        setup_logging(config, stream=sys.stderr)
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
        logger.error("Configuration error: %s", e)
        print(f"run-mcp-server: error: {e}", file=sys.stderr)
        return failed(EXIT_USAGE, e)
    except Exception as e:
        logger.exception("The MCP server failed")
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
