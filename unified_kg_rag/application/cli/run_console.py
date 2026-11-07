# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-console``: the local web console (docs/guides/console.md).

Serves the console's API and its built web client on 127.0.0.1, behind a
per-launch access token the printed address carries. Another
address needs ``--allow-remote``, and then an authenticating reverse proxy
in front: the console has no login of its own. Needs the ``console`` extra.
"""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import os
import secrets
import socket
import sys
from pathlib import Path
from typing import Any

from unified_kg_rag import __version__
from unified_kg_rag.application.cli.config_args import add_config_arguments
from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    ArgumentParser,
    add_output_format_argument,
    exits_on_sigterm,
    json_output,
    print_json,
    usage_error,
)
from unified_kg_rag.application.cli.display import console, markup_safe
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.config import get_config, load_cli_dotenv
from unified_kg_rag.shared import ConfigurationError, get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_PORT = 8501
# The packages of the console extra (pyproject [project.optional-dependencies]).
EXTRA_PACKAGES = ("fastapi", "uvicorn", "multipart")
# The built client: the environment variable, else the source checkout's.
WEB_DIRECTORY_ENV = "UKR_CONSOLE_WEB_DIRECTORY"
# The access token, when a launcher chooses it (the demo); else a new random
# one each start.
TOKEN_ENV = "UKR_CONSOLE_TOKEN"
MIN_TOKEN_LENGTH = 16


def access_token() -> str:
    """This launch's access token: ``UKR_CONSOLE_TOKEN`` (16 characters or
    more), else a random one.

    Raises:
        SystemExit: The variable is set but too short (a usage error).
    """
    chosen = os.environ.get(TOKEN_ENV)
    if chosen is None:
        return secrets.token_urlsafe(24)
    if len(chosen) < MIN_TOKEN_LENGTH:
        usage_error(f"{TOKEN_ENV} must be at least {MIN_TOKEN_LENGTH} characters")
    return chosen


SAMPLE_QUESTIONS_FILE = "sample-questions.json"


def _port(value: str) -> int:
    port = int(value)
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    return port


def _build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        prog="run-console",
        allow_abbrev=False,
        description="Local web console: ask, ingest, evaluate, explore the "
        "graph, see usage and cost, edit the configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "guides/console.md#start-the-console",
            "run-console --config-path config.yaml --source-directory .",
        ),
    )
    parser.add_argument(
        "--source-directory",
        type=str,
        default=None,
        help="Workspace the console reads corpora and datasets from, and "
        "writes uploads to (its uploads/ folder); nothing outside it, the "
        "cache directory and the evaluation outputs is read or written "
        "(default: the working directory)",
    )
    parser.add_argument(
        "--cache-directory",
        type=str,
        default=None,
        help="Directory of the ingestion jobs' pipeline caches and of the "
        "console's own records (console/: questions asked, jobs, their logs) "
        "(default: cache)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind (default 127.0.0.1). Any other needs "
        "--allow-remote; the console has no login, so put an authenticating "
        "reverse proxy in front of it",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"Port to bind (default {DEFAULT_PORT}; 0 picks a free one)",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow a --host other than a loopback address. Anyone who can "
        "reach the port can then query, ingest and edit the configuration",
    )
    parser.add_argument(
        "--trace-url",
        metavar="TEMPLATE",
        help="Link each answer to its trace: a URL with {trace_id}, e.g. "
        "http://localhost:16686/trace/{trace_id} for Jaeger (default with "
        "observability.backend langfuse: LANGFUSE_HOST/trace/{trace_id})",
    )
    add_config_arguments(parser)
    add_output_format_argument(parser)
    return parser


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def missing_extra() -> list[str]:
    """The console extra's packages that are not installed."""
    return [name for name in EXTRA_PACKAGES if importlib.util.find_spec(name) is None]


def web_directory() -> Path:
    """Where the built client is: ``UKR_CONSOLE_WEB_DIRECTORY``, else
    ``console/web/dist`` of the source checkout this package runs from."""
    if configured := os.environ.get(WEB_DIRECTORY_ENV):
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3] / "console" / "web" / "dist"


def _sample_questions(workspace: Path) -> list[str]:
    path = workspace / SAMPLE_QUESTIONS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(q) for q in data[:8]] if isinstance(data, list) else []


def _bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host.strip("[]"), port))
    sock.set_inheritable(False)
    return sock


@exits_on_sigterm
def main(argv: list[str] | None = None) -> int:
    load_cli_dotenv()
    args = _build_parser().parse_args(argv)
    as_json = json_output(args)
    if missing := missing_extra():
        usage_error(
            "run-console needs the console extra (missing: "
            f"{', '.join(missing)}): uv sync --extra console"
        )
    if not _is_loopback(args.host) and not args.allow_remote:
        usage_error(
            f"--host {args.host} is not a loopback address: pass --allow-remote "
            "as well, and put an authenticating reverse proxy in front (the "
            "console has no login)"
        )
    workspace = Path(args.source_directory or ".").resolve()
    if not workspace.is_dir():
        usage_error(f"--source-directory: '{workspace}' is not a directory")
    cache = Path(args.cache_directory or "cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config_path).resolve() if args.config_path else None
    try:
        config = get_config(config_path, allow_unknown_keys=args.allow_unknown_keys)
        setup_logging(config, stream=sys.stderr)
    except (ConfigurationError, FileNotFoundError) as e:
        # The console still starts: its Settings page shows the error.
        console.print(
            f"[yellow]The configuration does not load: {markup_safe(e)}[/yellow]"
        )

    from unified_kg_rag.application.cli.console.server import (
        ConsoleOptions,
        create_app,
    )

    token = access_token()
    web = web_directory()
    options = ConsoleOptions(
        workspace=workspace,
        cache_directory=cache,
        config_path=config_path,
        allow_unknown_keys=args.allow_unknown_keys,
        loopback=_is_loopback(args.host),
        web_directory=web,
        trace_url_template=args.trace_url,
        sample_questions=_sample_questions(workspace),
        token=token,
    )
    try:
        sock = _bind(args.host, args.port)
    except OSError as e:
        console.print(
            f"[red]Cannot listen on {markup_safe(args.host)}:{args.port}: {markup_safe(e)}[/red]"
        )
        return EXIT_FAILURE
    port = sock.getsockname()[1]
    host = f"[{args.host.strip('[]')}]" if ":" in args.host else args.host
    # The token rides in the address once; the page then keeps it in an
    # HttpOnly cookie.
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '[::]') else host}:{port}/?token={token}"  # nosec B104 - display only
    built = (web / "index.html").is_file()
    status: dict[str, Any] = {
        "status": "serving",
        "url": url,
        "web_built": built,
        "version": __version__,
    }
    if as_json:
        print_json(status)
        sys.stdout.flush()
    else:
        console.print(
            f"[bold]Console[/bold] on [cyan]{url}[/cyan]  (Ctrl-C to stop)",
            soft_wrap=True,
        )
        if not built:
            console.print(
                f"[yellow]The web client is not built ({markup_safe(web)}): cd console/web && npm ci && npm run build[/yellow]"
            )
        if not options.loopback:
            console.print(
                "[yellow]Listening beyond loopback: anyone who reaches this port can use the console.[/yellow]"
            )

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(options),
            log_level="warning",
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=60,
        )
    )
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    sys.exit(main())
