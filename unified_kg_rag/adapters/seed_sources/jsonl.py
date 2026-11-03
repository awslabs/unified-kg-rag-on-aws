# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kind: jsonl``: nodes and edges, one JSON object per line.

``path`` is one file (any name), a directory whose ``*.jsonl`` files
(recursively, names starting with ``.`` skipped) are read in sorted order,
an S3 object, or an S3 prefix read as a directory (``s3://bucket/seed/``).
Each non-blank line is an object whose ``"kind"`` says what it states::

    {"kind": "node", "id": "v1", "name": "Acme Supplies", "type": "VENDOR",
     "iri": "https://example.org/procure#acme", "properties": {"country": "DE"}}
    {"kind": "node", "id": "b1", "name": "Globex Buyer",
     "type_iri": "https://example.org/procure#Buyer"}
    {"kind": "edge", "source": "v1", "target": "b1", "type": "SUPPLIES",
     "description": "Acme supplies Globex with valves.", "weight": 2}

A node takes ``id`` (required), ``name`` (default: the id), ``type``,
``type_iri``, ``description``, ``iri``, ``confidence`` and ``properties``;
an edge ``source`` and ``target`` (node ids), ``type``, ``predicate_iri``,
``description``, ``weight`` and ``properties``. Property values are
strings, numbers or booleans (``null`` is dropped). An unknown key, a
duplicate node id, a line that is not a JSON object (or holds ``NaN``) and
a line over 1 MiB fail the import, naming the file and line.

Options: those of every kind (:class:`.registry.SeedSourceOptions`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unified_kg_rag.adapters.source_files.locations import SourceFile, split_s3
from unified_kg_rag.domain.models.seed_graph import SeedGraph

from .property_graph import (
    FileReader,
    GraphBuilder,
    check_path,
    check_total,
    is_hidden,
    local_file,
)
from .registry import SeedSourceContext, SeedSourceOptions, register_seed_source

SUFFIX = ".jsonl"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON number")


@register_seed_source("jsonl")
def load_jsonl_seed(context: SeedSourceContext) -> SeedGraph:
    """The seed graph the JSONL file(s) at ``source.path`` state.

    Raises:
        SeedGraphError: An option is unknown or invalid; ``secret_arn`` is
            set; the path is a URL other than ``file://`` or ``s3://``, is missing (or
            access is denied) or a directory without ``.jsonl`` files; the
            files hold more than ``max_bytes``; a line is too long, not
            UTF-8, not a JSON object, has an unknown ``kind`` or key, an
            invalid value or a non-scalar property, or repeats a node id;
            there are more nodes or edges than the caps.
    """
    context.options(SeedSourceOptions)
    path = check_path(context)
    reader = FileReader(context)
    files = (
        _locate_s3(context, reader, path)
        if isinstance(path, str)
        else _locate(context, path)
    )
    check_total(files, context)
    builder = GraphBuilder(context)
    for file in files:
        _read(builder, file, reader.text(file))
    return builder.graph(reader.report())


def _locate(context: SeedSourceContext, path: Path) -> list[SourceFile]:
    if not path.is_dir():
        return [local_file(path, context)]
    # Sorted by relative path as text, as an S3 listing is.
    found = sorted(
        (candidate.relative_to(path).as_posix(), candidate)
        for candidate in path.rglob(f"*{SUFFIX}")
        if candidate.is_file()
    )
    files = [
        SourceFile(location=str(candidate), size=candidate.stat().st_size)
        for relative, candidate in found
        if not is_hidden(relative)
    ]
    if not files:
        raise context.terms(f"directory '{path}' holds no {SUFFIX} file")
    return files


def _locate_s3(
    context: SeedSourceContext, reader: FileReader, location: str
) -> list[SourceFile]:
    _, key = split_s3(location, terms=context.terms)
    if key and not key.endswith("/"):
        return [reader.s3.head(location, prefix_hint=True)]
    files = [
        file
        for relative, file in reader.s3.list_prefix(location)
        if relative.endswith(SUFFIX) and not is_hidden(relative)
    ]
    if not files:
        raise context.terms(f"S3 prefix '{context.location}' holds no {SUFFIX} object")
    return files


def _read(builder: GraphBuilder, file: SourceFile, text: str) -> None:
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        where = f"'{file.name}' line {number}"
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except ValueError as e:
            raise builder.fail(where, f"not valid JSON ({e})") from e
        except RecursionError as e:
            # Arrays or objects nested deeper than the decoder recurses.
            raise builder.fail(where, "not valid JSON (nested too deeply)") from e
        if not isinstance(record, dict):
            raise builder.fail(where, "not a JSON object")
        kind = record.pop("kind", None)
        if kind == "node":
            builder.node(where, record)
        elif kind == "edge":
            builder.edge(where, record)
        else:
            raise builder.fail(
                where, f'"kind" is {json.dumps(kind)}; expected "node" or "edge"'
            )
