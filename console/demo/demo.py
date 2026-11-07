# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the console demo and start the console on 127.0.0.1.

Writes a small synthetic corpus and a matching ``run-eval`` dataset
(``corpus.py``) into a workspace, a config for the chosen models, ingests
the corpus into the embedded SQLite stores (``run-ingestion``) and starts
``run-console`` on the loopback interface. Needs the ``console`` extra:
``uv sync --extra console``.

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

    uv run python console/demo/demo.py --models fake
    uv run python console/demo/demo.py --models bedrock --region us-west-2

The console binds 127.0.0.1 only. On a remote host, forward the port
(``ssh -L 8501:127.0.0.1:8501 <host>``, or VS Code's port forwarding)
instead of binding another address.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus  # noqa: E402
import fake_models  # noqa: E402

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
    for name, text in corpus.DOCUMENTS.items():
        (source / name).write_text(text, encoding="utf-8")
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
            "chunking": {"chunker_type": "simple"},
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
        # The cross-encoder needs the [local] extra and a download.
        config["providers"]["rerank"] = {"kind": "none"}
    else:
        config["aws"] = {
            "region_name": region or os.environ.get("AWS_REGION") or "us-west-2"
        }
    return config


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
        "--prepare-only", action="store_true", help="Prepare and ingest, then exit"
    )
    parser.add_argument(
        "--chunk-delay",
        type=float,
        default=0.03,
        help="Fake models: seconds between answer chunks",
    )
    args = parser.parse_args(argv)

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source, dataset = write_corpus(workspace)
    server = None
    base_url = args.model_base_url
    if args.models == "fake":
        server, _ = fake_models.serve(
            port=args.model_port,
            dimensions=FAKE_DIMENSIONS,
            chunk_delay=args.chunk_delay,
        )
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        print(f"Fake models on {base_url}", flush=True)
    config_path = workspace / "config.yaml"
    config_path.write_text(
        "# Written by console/demo/demo.py; edit it in the console's Settings.\n"
        + yaml.safe_dump(
            demo_config(args.models, model_base_url=base_url, region=args.region),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    try:
        if not args.no_ingest:
            ingest(workspace, config_path, source)
        if args.prepare_only:
            return 0
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
        )
        print(f"Dataset: {dataset}", flush=True)
        try:
            return console.wait()
        except KeyboardInterrupt:
            console.send_signal(signal.SIGINT)
            return console.wait()
    finally:
        if server is not None:
            server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
