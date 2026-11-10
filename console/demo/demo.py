# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the console demo and start the console on 127.0.0.1.

Writes a small synthetic corpus (PDFs, Markdown, HTML, CSV, JSON and text)
and a matching ``run-eval`` dataset (``corpus.py``) into a workspace, a
config for the chosen models (once: a restart keeps it, with what Settings
saved; ``--reset-config`` writes it again), ingests the corpus into the
embedded SQLite stores (``run-ingestion``) and starts ``run-console`` on the
loopback interface. Needs the ``console`` extra, and the ``local`` extra for
the fake and Ollama models (``uv run --extra console --extra local``, below),
both checked before anything is ingested.

Models (``--models``):

- ``fake``: a deterministic model server on a loopback port
  (``fake_models.py``): no model, no network, no account. For screenshots,
  recordings and CI; the answers only quote the corpus.
- ``ollama``: a running Ollama with the local preset's models
  (``qwen2.5:7b-instruct``, ``bge-m3``; ``--model-base-url`` to change).
- ``bedrock``: Amazon Bedrock with your AWS credentials (an SSO session, an
  instance role) in ``--region``. The stores stay local; every model call is
  billed to the account.

Usage::

    uv run --no-dev --extra console --extra local python console/demo/demo.py \
        --models fake
    uv run --no-dev --extra console python console/demo/demo.py \
        --models bedrock --region us-west-2

The console binds 127.0.0.1 only. On a remote host, forward the port
(``ssh -L 8501:127.0.0.1:8501 <host>``, or VS Code's port forwarding)
instead of binding another address.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus  # noqa: E402
import fake_models  # noqa: E402

from unified_kg_rag.application.cli.preflight import (  # noqa: E402
    missing_extras_error,
)
from unified_kg_rag.application.cli.run_console import (  # noqa: E402
    MIN_TOKEN_LENGTH,
    TOKEN_ENV,
    missing_extra,
    web_directory,
)
from unified_kg_rag.domain.models import Config  # noqa: E402

# Illustrative prices of the fake models, so the dashboard and the
# comparison show a cost column in the fake mode. Real runs take the current
# Amazon Bedrock prices of their region (evaluation.prices).
FAKE_PRICES = {
    "demo-model": {"input_per_million_tokens": 3.0, "output_per_million_tokens": 15.0},
    "demo-model-fast": {
        "input_per_million_tokens": 0.8,
        "output_per_million_tokens": 4.0,
    },
}
FAKE_DIMENSIONS = 64


def write_corpus(workspace: Path) -> tuple[Path, Path]:
    """The corpus directory and the dataset file, (re)written."""
    source = workspace / "corpus"
    source.mkdir(parents=True, exist_ok=True)
    for name in corpus.RETIRED_FILES:
        (source / name).unlink(missing_ok=True)
    for name, data in corpus.documents().items():
        (source / name).write_bytes(data)
    datasets = workspace / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    dataset = datasets / "demo-eval.json"
    dataset.write_text(
        json.dumps(corpus.DATASET, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (workspace / "sample-questions.json").write_text(
        json.dumps(corpus.SAMPLE_QUESTIONS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return source, dataset


def missing_local_extra(models: str) -> str | None:
    """What the providers of ``models``' config need that is not installed
    (the run's own preflight message), or None."""
    config = Config.model_validate(
        demo_config(models, model_base_url=None, region=None)
    )
    return missing_extras_error(config, seeds=False)


def demo_config(
    models: str, *, model_base_url: str | None, region: str | None
) -> dict[str, Any]:
    """The demo's config: the embedded stores in the workspace, incremental
    indexing on, the LLM-free evaluators, and the chosen models."""
    config: dict[str, Any] = {
        "providers": {
            "graph": {"kind": "sqlite", "sqlite_path": "stores/graph.sqlite3"},
            "vector": {"kind": "sqlite", "sqlite_path": "stores/vector.sqlite3"},
            "doc_status": {
                "kind": "sqlite",
                "enabled": True,
                "sqlite_path": "stores/doc-status.sqlite3",
            },
        },
        "processing": {
            # The demo's documents are a page or two: chunks of about a
            # page, cut at paragraphs without overlap, so a passage cited
            # from a PDF names the one page it is on.
            "chunking": {
                "chunker_type": "simple",
                "min_chunk_size": 100,
                "fallback_chunk_size": 600,
                "max_chunk_size": 2000,
                "chunk_overlap": 0,
            },
            "translation": {"enabled": False},
            "gleaning": {"enabled": False},
            "claim_extraction": {"enabled": False},
        },
        "evaluation": {
            "outputs_directory": "outputs/evaluation",
            "enabled_evaluators": ["answer_match", "retrieval", "graph_aware"],
        },
    }
    if models == "fake":
        url = model_base_url or "http://127.0.0.1:11500"
        config["backend"] = "local"
        config["providers"].update(
            {
                "llm": {"kind": "ollama", "base_url": url},
                "embedding": {"kind": "ollama", "base_url": url},
                "rerank": {"kind": "none"},
            }
        )
        config["models"] = {
            "default": "demo-model",
            "fast": "demo-model-fast",
            "embedding": {"model_id": "demo-embed", "dimension": FAKE_DIMENSIONS},
        }
        config["evaluation"]["prices"] = FAKE_PRICES
        config["retry"] = {"max_attempts": 1}
    elif models == "ollama":
        config["backend"] = "local"
        if model_base_url:
            config["providers"]["llm"] = {"kind": "ollama", "base_url": model_base_url}
            config["providers"]["embedding"] = {
                "kind": "ollama",
                "base_url": model_base_url,
            }
        # The cross-encoder needs the cross-encoder extra and a download.
        config["providers"]["rerank"] = {"kind": "none"}
    else:
        config["aws"] = {
            "region_name": region or os.environ.get("AWS_REGION") or "us-west-2"
        }
    return config


def kept_model_port(config_path: Path) -> int | None:
    """The loopback port the fake models answer on in an existing demo
    config (``providers.llm.base_url``), so a restart serves them where the
    kept config looks; ``None`` when it names none."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm = (
        ((data.get("providers") or {}).get("llm") or {})
        if isinstance(data, dict)
        else {}
    )
    url = llm.get("base_url") if isinstance(llm, dict) else None
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    return parts.port if parts.hostname in ("127.0.0.1", "localhost") else None


def check_config(config_path: Path) -> None:
    """Exit naming the problem if the kept config does not load as a run
    loads it (``--reset-config`` writes the demo's again)."""
    from unified_kg_rag.application.config import get_config
    from unified_kg_rag.shared.exceptions import ConfigurationError

    try:
        get_config(config_path)
    except ConfigurationError as e:
        raise SystemExit(
            f"{config_path} does not load: {e}\nFix it, or pass --reset-config "
            "to write the demo's config again."
        ) from e


def write_config(
    config_path: Path, models: str, *, model_base_url: str | None, region: str | None
) -> None:
    config_path.write_text(
        "# Written by console/demo/demo.py; edit it in the console's Settings.\n"
        + yaml.safe_dump(
            demo_config(models, model_base_url=model_base_url, region=region),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _script(name: str) -> str:
    """A console script of this environment (next to its interpreter)."""
    found = Path(sys.executable).with_name(name)
    if found.exists():
        return str(found)
    which = shutil.which(name)
    if which is None:
        raise SystemExit(f"{name} not found: run `uv sync --extra console` first")
    return which


def ingest(workspace: Path, config_path: Path, source: Path) -> None:
    """Ingest the corpus with run-ingestion (incremental: a re-run with the
    same corpus finds nothing new)."""
    command = [
        _script("run-ingestion"),
        "--config-path",
        str(config_path),
        "--source-directory",
        str(source),
        "--cache-directory",
        str(workspace / "cache"),
        "--pipeline-id",
        "demo",
        "--yes",
    ]
    print("Ingesting the demo corpus ...", flush=True)
    subprocess.run(command, check=True, cwd=workspace)


def missing_web_build() -> bool:
    """Whether the console's web client (``npm run build``) is missing."""
    return not (web_directory() / "index.html").is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and start the console demo.")
    parser.add_argument(
        "--models", choices=["fake", "ollama", "bedrock"], default="fake"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(".console-demo"),
        help="Directory of the demo's corpus, stores and results (default .console-demo)",
    )
    parser.add_argument(
        "--port", type=int, default=8501, help="Console port (default 8501)"
    )
    parser.add_argument(
        "--model-port",
        type=int,
        default=0,
        help="Fake model server port (default: a free one)",
    )
    parser.add_argument("--model-base-url", help="Ollama URL (--models ollama)")
    parser.add_argument(
        "--region",
        help="Bedrock region (--models bedrock; default AWS_REGION or us-west-2)",
    )
    parser.add_argument("--no-ingest", action="store_true", help="Skip the ingestion")
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help=(
            "Write the demo's config again over <workspace>/config.yaml (by "
            "default an existing one, with what Settings saved, is kept)"
        ),
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Prepare and ingest, then exit"
    )
    parser.add_argument(
        "--chunk-delay",
        type=float,
        default=0.03,
        help="Fake models: seconds between answer chunks",
    )
    args = parser.parse_args(argv)
    # Before the ingestion: without the extra, run-console would fail after
    # it, and the address printed would serve nothing.
    if not args.prepare_only and (missing := missing_extra()):
        raise SystemExit(
            f"The console demo needs the console extra (missing: "
            f"{', '.join(missing)}): uv run --no-dev --extra console python "
            "console/demo/demo.py ..."
        )

    # Nor without the built web client: the console would serve no page.
    if not args.prepare_only and missing_web_build():
        raise SystemExit(
            f"The console's web client is not built ({web_directory()} has no "
            "index.html): cd console/web && npm ci && npm run build"
        )

    # Nor with a token run-console refuses: the address printed would never
    # open.
    chosen = os.environ.get(TOKEN_ENV)
    if not args.prepare_only and chosen is not None and len(chosen) < MIN_TOKEN_LENGTH:
        raise SystemExit(
            f"{TOKEN_ENV} must be at least {MIN_TOKEN_LENGTH} characters (unset "
            "it for a random one)"
        )

    # The fake and Ollama models run in local mode, whose providers come
    # with the local extra: checked before the corpus is written, as the
    # ingestion would fail on it.
    if args.models in ("fake", "ollama") and (
        missing_local := missing_local_extra(args.models)
    ):
        raise SystemExit(
            f"The {args.models} demo needs the local extra ({missing_local}): "
            "uv run --no-dev --extra console --extra local python "
            "console/demo/demo.py ..."
        )

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source, dataset = write_corpus(workspace)
    config_path = workspace / "config.yaml"
    # The config is the user's once written: what Settings saved survives a
    # restart. --reset-config writes the demo's again.
    keep = config_path.exists() and not args.reset_config
    if keep:
        check_config(config_path)
        print(
            f"Keeping {config_path} (--reset-config writes the demo's again)",
            flush=True,
        )
    server = None
    base_url = args.model_base_url
    if args.models == "fake":
        port = args.model_port
        if keep and not port:
            # Where the kept config looks for the fake models.
            port = kept_model_port(config_path) or 0
        try:
            server, _ = fake_models.serve(
                port=port,
                dimensions=FAKE_DIMENSIONS,
                chunk_delay=args.chunk_delay,
            )
        except OSError as e:
            raise SystemExit(
                f"The fake models cannot listen on port {port}: {e}. Pass "
                "--model-port with a free port and --reset-config."
            ) from e
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        print(f"Fake models on {base_url}", flush=True)
        if keep and kept_model_port(config_path) != server.server_address[1]:
            print(
                f"Note: {config_path} sends its model calls elsewhere; pass "
                "--reset-config to point it at these fake models.",
                flush=True,
            )
    if not keep:
        write_config(
            config_path, args.models, model_base_url=base_url, region=args.region
        )
    try:
        if not args.no_ingest:
            ingest(workspace, config_path, source)
        if args.prepare_only:
            return 0
        # The console's access token, chosen here so the address can be
        # printed below as well.
        token = chosen or secrets.token_urlsafe(24)
        console = subprocess.Popen(
            [
                _script("run-console"),
                "--config-path",
                str(config_path),
                "--source-directory",
                str(workspace),
                "--cache-directory",
                str(workspace / "cache"),
                "--port",
                str(args.port),
            ],
            cwd=workspace,
            env={**os.environ, TOKEN_ENV: token},
        )
        # A SIGTERM (a container stop, kill) stops the console too, as
        # Ctrl-C does, instead of leaving it running without the demo.
        previous = signal.signal(
            signal.SIGTERM, lambda *_: console.send_signal(signal.SIGTERM)
        )
        print(f"Dataset: {dataset}", flush=True)
        print(f"Open http://127.0.0.1:{args.port}/?token={token}", flush=True)
        try:
            return console.wait()
        except KeyboardInterrupt:
            console.send_signal(signal.SIGINT)
            return console.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
