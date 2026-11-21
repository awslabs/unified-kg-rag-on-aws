# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public benchmarks prepared as a corpus plus a ``run-eval`` dataset.

The data is downloaded (or read from a local copy) at run time and never
committed to this repository.

Every prepared row carries ``metadata.split`` (:func:`split_of`): ``tune`` or
``test``, from a hash of its question id, so settings chosen on the ``tune``
half are reported on the ``test`` half (``run-eval --split``) without a
second sample. ``prepared.json`` (:data:`PREPARED_FILE`) next to
``eval_data.json`` records how the dataset was prepared, and the dataset it
describes (its file name and :func:`content_sha256`); ``run-eval`` copies it
into its run manifest (``dataset.prepared``) for that dataset alone, not
for another file in the same directory or one edited since.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from unified_kg_rag.shared.utils.files import atomic_write_text

# The question classes of the assertions benchmark (``autoq``), here so the
# prepare-benchmark parser lists them without importing its generator.
QUESTION_CLASSES = (
    "data_local",
    "data_global",
    "data_linked",
    "activity_local",
    "activity_global",
)

Split = Literal["tune", "test"]
# The halves of a prepared dataset, by the parity of the question id's hash.
SPLITS: tuple[Split, Split] = ("tune", "test")
# Written next to eval_data.json: the parameters the dataset was prepared with.
PREPARED_FILE = "prepared.json"
# The dataset every benchmark writes next to it.
DATASET_FILE = "eval_data.json"


def content_sha256(data: Any) -> str:
    """The SHA-256 of a dataset's parsed JSON (sorted keys, UTF-8): files
    differing only in formatting have the same one (a run manifest's
    ``dataset.content_sha256``)."""
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_of(question_id: str) -> Split:
    """The half a question belongs to: ``tune`` when the SHA-256 of its id
    (UTF-8), read as a number, is even (mod 2 = 0), else ``test``. Deterministic, independent of the
    sample, seed and size, so a question keeps its half in every
    preparation and the two halves never share a question."""
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return SPLITS[digest[-1] % 2]


def file_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes (a benchmark source, for ``prepared.json``)."""
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prepared(output_dir: Path, record: dict[str, Any], dataset: Any) -> None:
    """Write ``record`` (the benchmark, its parameters and sources) as
    ``output_dir/prepared.json``, with the dataset it describes
    (``dataset``: the rows written there as ``eval_data.json``, by file name
    and :func:`content_sha256`)."""
    record = {
        **record,
        "dataset": {"file": DATASET_FILE, "content_sha256": content_sha256(dataset)},
    }
    atomic_write_text(
        output_dir / PREPARED_FILE,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_prepared(eval_data_path: Path, data: Any) -> dict[str, Any] | None:
    """The ``prepared.json`` next to ``eval_data_path`` when it describes
    that dataset (``data``, its parsed JSON: the file name and content
    digest it records), else ``None``: none there, not a JSON object, or
    another dataset's (a file beside a prepared one, one edited since)."""
    path = eval_data_path.parent / PREPARED_FILE
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    described = record.get("dataset")
    if described != {
        "file": eval_data_path.name,
        "content_sha256": content_sha256(data),
    }:
        return None
    return record
