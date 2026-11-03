# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kind: csv``: a nodes file and an edges file.

``path`` is a directory, or an S3 prefix (``s3://bucket/seed/``), holding
``nodes.csv`` and ``edges.csv`` (``options.nodes`` and ``options.edges``
name other files in it; ``edges: null`` reads nodes only). The first row
names the columns::

    id,name,type,type_iri,description,iri,confidence,country
    v1,Acme Supplies,VENDOR,,Industrial valve vendor.,https://example.org/procure#acme,,DE

    source,target,type,predicate_iri,description,weight
    v1,b1,SUPPLIES,,Acme supplies Globex with valves.,2

Nodes need ``id``, edges ``source`` and ``target``; an empty cell is an
unset value. Any other column is a property, a string
(``options.property_columns`` keeps only the columns it lists). UTF-8, with
an optional byte-order mark; a row over 1 MiB, a field over Python's csv
field limit (131072 characters), a row with more cells than columns, a
repeated column and a duplicate node id fail the import, naming the file
and line.

Options: ``nodes``, ``edges``, ``property_columns`` and those of every kind
(:class:`.registry.SeedSourceOptions`).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field, field_validator

from unified_kg_rag.adapters.source_files.locations import SourceFile
from unified_kg_rag.domain.models.seed_graph import SeedGraph

from .property_graph import (
    FileReader,
    GraphBuilder,
    check_path,
    check_total,
    local_file,
)
from .registry import SeedSourceContext, SeedSourceOptions, register_seed_source

NODE_COLUMNS = ("id", "name", "type", "type_iri", "description", "iri", "confidence")
EDGE_COLUMNS = ("source", "target", "type", "predicate_iri", "description", "weight")
_REQUIRED = {"nodes": ("id",), "edges": ("source", "target")}


class CsvSeedOptions(SeedSourceOptions):
    """The options of kind ``csv``."""

    nodes: str = Field(
        default="nodes.csv", description="The nodes file, in the seed's directory"
    )
    edges: str | None = Field(
        default="edges.csv",
        description="The edges file, in the seed's directory; null: none",
    )
    property_columns: list[str] | None = Field(
        default=None,
        description="The only extra columns kept as properties (default: all)",
    )

    @field_validator("nodes", "edges")
    @classmethod
    def _file_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parts = PurePosixPath(value).parts
        if not value or value.startswith("/") or ".." in parts or "\\" in value:
            raise ValueError(
                f"'{value}' is not a file name relative to the seed's directory"
            )
        return value


@register_seed_source("csv")
def load_csv_seed(context: SeedSourceContext) -> SeedGraph:
    """The seed graph the CSV files in ``source.path`` state.

    Raises:
        SeedGraphError: An option is unknown or invalid; ``secret_arn`` is
            set; the path is a URL other than ``file://`` or ``s3://``, not a directory (or
            prefix), or a file is missing (or access is denied); the files
            hold more than ``max_bytes``; a row is too long or malformed, a
            column repeats or a required one is missing, a
            ``property_columns`` column is in neither file; a value is
            invalid or a node id repeats; there are more nodes or edges than
            the caps.
    """
    options = context.options(CsvSeedOptions)
    path = check_path(context)
    reader = FileReader(context)
    names = {"nodes": options.nodes, "edges": options.edges}
    files = {
        part: _locate(context, reader, path, name)
        for part, name in names.items()
        if name is not None
    }
    check_total(list(files.values()), context)
    builder = GraphBuilder(context)
    headers: set[str] = set()
    for part, file in files.items():
        headers |= _read(builder, part, file, reader.text(file), options)
    if options.property_columns is not None:
        missing = sorted(set(options.property_columns) - headers)
        if missing:
            raise context.terms(
                f"options.property_columns names {', '.join(missing)}, a "
                "column of neither file"
            )
    return builder.graph(reader.report())


def _locate(
    context: SeedSourceContext, reader: FileReader, path: str | Path, name: str
) -> SourceFile:
    if isinstance(path, Path):
        if not path.is_dir():
            raise context.terms(
                f"'{path}' is not a directory; kind 'csv' reads a directory "
                f"holding the nodes and edges files"
            )
        return local_file(path / name, context)
    if not path.endswith("/"):
        raise context.terms(
            f"'{context.location}' names an S3 object; kind 'csv' reads a "
            "prefix holding the nodes and edges files (end the path with '/')"
        )
    return reader.s3.head(path + name)


def _read(
    builder: GraphBuilder,
    part: str,
    file: SourceFile,
    text: str,
    options: CsvSeedOptions,
) -> set[str]:
    """Add the rows of the ``part`` (``nodes`` or ``edges``) file; its
    column names."""
    # strict: a stray quote fails instead of being read into a cell.
    rows = csv.reader(io.StringIO(text, newline=""), strict=True)
    known = NODE_COLUMNS if part == "nodes" else EDGE_COLUMNS
    try:
        header = next(rows, None)
        if header is None:
            raise builder.fail(
                f"'{file.name}'", "is empty; the first row names the columns"
            )
        where = f"'{file.name}' line 1"
        if any(not column.strip() for column in header):
            raise builder.fail(where, "a column has no name")
        repeated = sorted({c for c in header if header.count(c) > 1})
        if repeated:
            raise builder.fail(where, f"columns {', '.join(repeated)} repeat")
        missing = [c for c in _REQUIRED[part] if c not in header]
        if missing:
            raise builder.fail(where, f"no {', '.join(missing)} column")
        kept = (
            None if options.property_columns is None else set(options.property_columns)
        )
        for row in rows:
            where = f"'{file.name}' line {rows.line_num}"
            if not any(cell.strip() for cell in row):
                continue
            if len(row) > len(header):
                raise builder.fail(
                    where, f"{len(row)} cells, more than the {len(header)} columns"
                )
            data: dict[str, Any] = {}
            properties: dict[str, str] = {}
            for column, cell in zip(header, row, strict=False):
                if cell == "":
                    continue
                if column in known:
                    data[column] = cell
                elif kept is None or column in kept:
                    properties[column] = cell
            if properties:
                data["properties"] = properties
            (builder.node if part == "nodes" else builder.edge)(where, data)
        return set(header)
    except csv.Error as e:
        raise builder.fail(
            f"'{file.name}' line {rows.line_num}", f"not valid CSV ({e})"
        ) from e
