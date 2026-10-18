# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuSiQue (answerable) as a corpus plus an evaluation dataset.

MuSiQue (Trivedi et al., TACL 2022) composes 2-4 single-hop questions into
one multi-hop question; each item ships 20 paragraphs, of which the
supporting ones answer it and the rest are distractors. The data is
distributed under CC BY 4.0 (https://github.com/StonyBrookNLP/musique) and
is never committed here: ``prepare`` reads the official JSONL — a local copy
from the official download, or the answerable dev split fetched at run time
from a pinned revision of a public mirror, checked against its SHA-256.

``prepare`` writes one text file per distinct paragraph (title, blank line,
text) under ``corpus/`` and ``eval_data.json`` in the ``run-eval`` format:
``answer`` and ``answer_aliases`` for ``answer_match``, the supporting
paragraphs' file names as ``reference_sources`` for ``retrieval``, and the
hop count (``2hop``, ``3hop``, ``4hop``) as ``category``. Every selected question's
distractors are part of the corpus, as in the benchmark's own setting.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from unified_kg_rag.evaluation.retrieval_evaluator import reference_key
from unified_kg_rag.shared import EvaluationException, get_logger

logger = get_logger(__name__)

LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
CITATION = (
    "Trivedi, Balasubramanian, Khot, Sabharwal. MuSiQue: Multihop Questions "
    "via Single-hop Question Composition. TACL 2022."
)
# The answerable dev split, byte-identical to the official
# musique_ans_v1.0_dev.jsonl, at a pinned revision of a public mirror.
DEV_URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/"
    "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321/musique_ans_v1.0_dev.jsonl"
)
DEV_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"

_HOPS = re.compile(r"(\d+)hop")
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def download_dev(destination: Path, *, timeout: float = 120.0) -> Path:
    """Fetch the answerable dev split to ``destination`` and verify it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MuSiQue dev (%s) from %s", LICENSE, DEV_URL)
    with urllib.request.urlopen(DEV_URL, timeout=timeout) as response:  # noqa: S310
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != DEV_SHA256:
        raise EvaluationException(
            f"MuSiQue dev download has SHA-256 {digest}, expected {DEV_SHA256}; "
            "pass the official file with --data-path instead."
        )
    destination.write_bytes(data)
    return destination


def _items(source: Path) -> Iterator[dict[str, Any]]:
    with source.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise EvaluationException(
                    f"{source}:{number} is not a JSON line: {e}"
                ) from e


def _file_name(title: str, taken: set[str]) -> str:
    # No leading dot: the parser skips hidden files.
    stem = " ".join(_UNSAFE.sub(" ", title).split())[:150].lstrip(". ") or "paragraph"
    name, n = f"{stem}.txt", 2
    # Distinct under the retrieval evaluator's matching (case, NFKC, "_" as
    # " "), or two paragraphs would count as one file.
    while reference_key(name) in taken:
        name, n = f"{stem} ({n}).txt", n + 1
    taken.add(reference_key(name))
    return name


def check_output_directory(output_dir: Path) -> None:
    """Raise unless ``output_dir``'s ``corpus/`` is new or empty: files of
    another sample would be ingested with this one."""
    corpus = output_dir / "corpus"
    if corpus.exists() and any(corpus.iterdir()):
        raise EvaluationException(
            f"'{corpus}' is not empty; choose a new --output-directory"
        )


def prepare(
    source: Path, output_dir: Path, *, limit: int | None = None, seed: int = 0
) -> dict[str, int]:
    """Write ``corpus/`` and ``eval_data.json`` for (a sample of) ``source``.

    ``limit`` questions are drawn with ``seed`` (all when ``None``);
    unanswerable items (in the full, non-``ans`` files) are skipped.
    """
    check_output_directory(output_dir)
    corpus = output_dir / "corpus"
    items = [i for i in _items(source) if i.get("answerable", True)]
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    corpus.mkdir(parents=True, exist_ok=True)
    files: dict[tuple[str, str], str] = {}
    taken: set[str] = set()
    dataset: list[dict[str, Any]] = []
    for item in items:
        supporting: list[str] = []
        for paragraph in item["paragraphs"]:
            title = str(paragraph.get("title") or "")
            text = str(paragraph.get("paragraph_text") or "")
            key = (title, text)
            if key not in files:
                files[key] = _file_name(title, taken)
                (corpus / files[key]).write_text(f"{title}\n\n{text}\n", "utf-8")
            if paragraph.get("is_supporting"):
                supporting.append(files[key])
        item_id = str(item["id"])
        dataset.append(
            {
                "query_id": item_id,
                "question": item["question"],
                "answer": item["answer"],
                "answer_aliases": list(item.get("answer_aliases") or []),
                "reference_sources": supporting,
                "category": (
                    f"{hops.group(1)}hop"
                    if (hops := _HOPS.match(item_id))
                    else item_id.split("__", 1)[0]
                ),
                "metadata": {"benchmark": "musique"},
            }
        )
    (output_dir / "eval_data.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), "utf-8"
    )
    return {"questions": len(dataset), "paragraphs": len(files)}
