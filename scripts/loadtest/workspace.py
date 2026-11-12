# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a load-test workspace: a corpus, the demo's eval dataset and a
``backend: local`` config whose models are a loopback fake model server
(``chaos_models.py``).

The corpus is the console demo's (``console/demo/corpus.py``: PDFs,
Markdown, HTML, CSV, JSON and text in English, Korean and Japanese), or with
``--docs N`` a synthetic one of N text files built from the same made-up
entities, for bulk-ingestion and chaos runs. Nothing here names a real
company or person.

Usage::

    uv run python scripts/loadtest/workspace.py WS --model-port 11613
    uv run python scripts/loadtest/workspace.py WS --model-port 11613 \
        --docs 1500 --chars 4000 --lang mixed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "console" / "demo"

# Every methodology, so all twelve strategies have their artifacts.
ALL_METHODOLOGIES = ["graphrag", "lightrag", "hipporag", "raptor"]

# The demo corpus' entities (console/demo/fake_models.py ENTITIES), so the
# fake extractor finds them in the synthetic documents too.
_EN = [
    "Acme Robotics",
    "Globex Sensors",
    "Umbra Freight",
    "Kestrel Depot",
    "Orion Labs",
    "Alice Chen",
    "Ravi Patel",
    "lidar unit",
    "supply agreement",
]
_KO = ["애크미 로보틱스", "글로벡스 센서스", "엄브라 프레이트", "케스트럴 창고"]
_VERBS_EN = ["supplies", "inspects", "delivers to", "contracts with", "audits"]
_VERBS_KO = ["에 공급한다", "을 검사한다", "까지 운송한다", "와 계약했다"]


def _demo_modules() -> tuple[Any, Any]:
    sys.path.insert(0, str(DEMO))
    import corpus  # type: ignore[import-not-found]
    import demo  # type: ignore[import-not-found]

    return corpus, demo


def _sentence(rng: random.Random, lang: str, i: int) -> str:
    if lang == "ko" or (lang == "mixed" and rng.random() < 0.3):
        a, b = rng.sample(_KO, 2)
        return f"{a}는 {b}{rng.choice(_VERBS_KO)}. 계약 번호 {i}-{rng.randint(1, 999)}."
    a, b = rng.sample(_EN, 2)
    return (
        f"In batch {i}, {a} {rng.choice(_VERBS_EN)} {b} under order "
        f"{rng.randint(1000, 9999)}, and the shipment of {rng.randint(1, 500)} "
        f"units was recorded on day {rng.randint(1, 28)}."
    )


def synthetic_documents(
    docs: int, chars: int, lang: str = "en", seed: int = 1
) -> dict[str, str]:
    """``docs`` documents of about ``chars`` characters each, the same for
    the same arguments (a paragraph break every six sentences)."""
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for d in range(docs):
        parts: list[str] = []
        length = 0
        k = 0
        while length < chars:
            sentence = _sentence(rng, lang, d * 1000 + k)
            k += 1
            parts.append(sentence)
            length += len(sentence) + 1
            if k % 6 == 0:
                parts.append("\n\n")
        out[f"doc-{d:05d}.txt"] = " ".join(parts).replace(" \n\n ", "\n\n")
    return out


def write_workspace(
    directory: Path,
    model_port: int,
    *,
    docs: int = 0,
    chars: int = 2000,
    lang: str = "en",
    seed: int = 1,
    stores: str = "sqlite",
    all_methodologies: bool = True,
    lease_seconds: float | None = None,
) -> Path:
    """Write the workspace and return its config path."""
    corpus, demo = _demo_modules()
    source = directory / "corpus"
    source.mkdir(parents=True, exist_ok=True)
    if docs:
        for name, text in synthetic_documents(docs, chars, lang, seed).items():
            (source / name).write_text(text, encoding="utf-8")
    else:
        for name, data in corpus.documents().items():
            (source / name).write_bytes(data)
    (directory / "eval.json").write_text(
        json.dumps(corpus.DATASET, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    config = demo.demo_config(
        "fake", model_base_url=f"http://127.0.0.1:{model_port}", region=None
    )
    # Retries on, as a real deployment has them (the chaos runs inject the
    # faults retries absorb: 429, 503, dropped connections), with a short
    # backoff so a run under faults takes seconds, not minutes.
    config["retry"] = {
        "max_attempts": 3,
        "base_delay_seconds": 0.2,
        "max_delay_seconds": 2.0,
    }
    if all_methodologies:
        config.setdefault("indexing", {})["methodologies"] = ALL_METHODOLOGIES
    if lease_seconds is not None:
        config.setdefault("indexing", {})["lease_seconds"] = lease_seconds
    if stores == "containers":
        # docker/compose.local.yaml's OpenSearch and Gremlin Server.
        config["providers"]["graph"] = {"kind": "gremlin"}
        config["providers"]["vector"] = {"kind": "opensearch"}
    path = directory / "config.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model-port", type=int, required=True)
    parser.add_argument(
        "--docs", type=int, default=0, help="synthetic documents (0: the demo corpus)"
    )
    parser.add_argument("--chars", type=int, default=2000)
    parser.add_argument("--lang", choices=["en", "ko", "mixed"], default="en")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stores", choices=["sqlite", "containers"], default="sqlite")
    parser.add_argument(
        "--lightrag-graphrag-only",
        action="store_true",
        help="index the default methodologies only (no ppr, tree)",
    )
    parser.add_argument("--lease-seconds", type=float)
    args = parser.parse_args(argv)
    print(
        write_workspace(
            args.directory,
            args.model_port,
            docs=args.docs,
            chars=args.chars,
            lang=args.lang,
            seed=args.seed,
            stores=args.stores,
            all_methodologies=not args.lightrag_graphrag_only,
            lease_seconds=args.lease_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
