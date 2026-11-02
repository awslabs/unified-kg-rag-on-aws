# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the property-graph seed sources (``jsonl``, ``csv``) share: finding
and reading their files within the seed's caps, and building the
``SeedGraph`` from their records.

Files are local paths or S3 objects (:mod:`.source_files.locations`), sized
before any byte is read: the files together may hold at most the seed's
``max_bytes``, checked again while reading (a file may have grown). A line
over :data:`MAX_LINE_BYTES` fails, as does text that is not UTF-8 (a leading
byte-order mark is dropped). Records become nodes and edges through
:class:`GraphBuilder`, which names the file and line of what it refuses: a
duplicate node id, a property value that is not a string, number, boolean or
null, a node or edge count over the seed's caps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from unified_kg_rag.adapters.source_files.locations import (
    SourceFile,
    SourceReader,
    is_s3,
    split_s3,
)
from unified_kg_rag.domain.models.seed_graph import (
    SeedEdge,
    SeedGraph,
    SeedLoadReport,
    SeedNode,
)
from unified_kg_rag.shared.exceptions import ConfigurationError

from .registry import SeedSourceContext

# The most bytes one line (a JSONL record, a CSV row) may hold.
MAX_LINE_BYTES = 1024 * 1024
_BOM = b"\xef\xbb\xbf"


def check_path(context: SeedSourceContext) -> str | Path:
    """The local path ``source.path`` names, or the ``s3://`` URL as is.

    Raises:
        SeedGraphError: ``secret_arn`` is set, or the path is a URL other
            than ``s3://`` (or names no valid bucket).
    """
    source, terms = context.source, context.terms
    if source.secret_arn:
        raise terms(f"kind '{source.kind}' takes no secret_arn; leave it unset")
    if is_s3(source.path):
        split_s3(source.path, terms=terms)
        return source.path
    if "://" in source.path:
        raise terms(
            f"'{context.location}' is a URL; kind '{source.kind}' reads a "
            "local file or directory or an s3:// object or prefix. Copy the "
            "files next to the config (or into the image, or S3), or "
            "register a seed source for the location's scheme"
        )
    return Path(source.path)


def is_hidden(relative: str) -> bool:
    """True when a part of ``relative`` (a path below a directory) starts
    with ``.``: skipped, as the corpus skips it."""
    return any(part.startswith(".") for part in relative.split("/"))


def local_file(path: Path, context: SeedSourceContext) -> SourceFile:
    """The local file ``path``, sized.

    Raises:
        SeedGraphError: ``path`` is not a file.
    """
    if not path.is_file():
        raise context.terms(f"{context.terms.noun} '{path}' not found (no such file)")
    return SourceFile(location=str(path), size=path.stat().st_size)


def check_total(files: list[SourceFile], context: SeedSourceContext) -> None:
    """Refuse ``files`` when they hold more than the seed's ``max_bytes``,
    before any is read.

    Raises:
        SeedGraphError: They do.
    """
    total = sum(file.size for file in files)
    if total > context.source.max_bytes:
        raise context.terms(
            f"the files of '{context.location}' hold {total} bytes, more than "
            f"max_bytes ({context.source.max_bytes}); split the seed or raise "
            "max_bytes"
        )


class FileReader:
    """Reads a seed's files, local or S3, within its ``max_bytes`` in
    total, and decodes them."""

    def __init__(self, context: SeedSourceContext) -> None:
        self.context = context
        self.s3 = SourceReader(context, terms=context.terms)
        self.files = 0
        self.bytes_read = 0

    def read(self, file: SourceFile) -> bytes:
        """The bytes of ``file``.

        Raises:
            SeedGraphError: It cannot be read, or the files read so far hold
                more than ``max_bytes`` (one grew since it was sized).
        """
        terms, cap = self.context.terms, self.context.source.max_bytes
        left = cap - self.bytes_read
        if is_s3(file.location):
            data = self.s3.read(file, max_bytes=max(left, 0))
        else:
            try:
                data = Path(file.location).read_bytes()
            except OSError as e:
                raise terms(f"cannot read '{file.name}': {e.strerror or e}") from e
        self.files += 1
        self.bytes_read += len(data)
        if self.bytes_read > cap:
            raise terms(
                f"the files of '{self.context.location}' hold more than "
                f"max_bytes ({cap}) bytes (one grew since it was sized)"
            )
        return data

    def text(self, file: SourceFile) -> str:
        """The text of ``file``: UTF-8, a leading byte-order mark dropped.

        Raises:
            SeedGraphError: What :meth:`read` raises; a line holds more than
                :data:`MAX_LINE_BYTES`; the bytes are not UTF-8.
        """
        data = self.read(file)
        start = 0
        line = 1
        while start < len(data):
            end = data.find(b"\n", start)
            end = len(data) if end < 0 else end
            if end - start > MAX_LINE_BYTES:
                raise self.context.terms(
                    f"'{file.name}' line {line} holds more than "
                    f"{MAX_LINE_BYTES} bytes"
                )
            start, line = end + 1, line + 1
        if data.startswith(_BOM):
            data = data[len(_BOM) :]
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise self.context.terms(
                f"'{file.name}' is not valid UTF-8 (byte offset {e.start})"
            ) from e

    def report(self) -> SeedLoadReport:
        """The files and bytes read so far."""
        return SeedLoadReport(files=self.files, bytes_read=self.bytes_read)


def _scalar(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (str, int, bool))


class GraphBuilder:
    """Collects a seed's nodes and edges from records, refusing a
    duplicate node id, a non-scalar property and a count over the caps
    as soon as it sees one, naming where."""

    def __init__(self, context: SeedSourceContext) -> None:
        self.context = context
        self.nodes: list[SeedNode] = []
        self.edges: list[SeedEdge] = []
        # Node id -> where it was first stated.
        self._seen: dict[str, str] = {}

    def fail(self, where: str, message: str) -> ConfigurationError:
        return self.context.terms(f"{where}: {message}")

    def _properties(self, where: str, properties: Any) -> None:
        if properties is None:
            return
        if not isinstance(properties, dict):
            raise self.fail(where, "properties is not an object")
        for key, value in properties.items():
            if not _scalar(value):
                raise self.fail(
                    where,
                    f"property {key!r} is not a string, number, boolean or null",
                )

    def _validated(
        self, where: str, model: type[SeedNode] | type[SeedEdge], data: dict[str, Any]
    ) -> Any:
        try:
            return model.model_validate(data)
        except ValidationError as e:
            raise self.fail(
                where,
                "; ".join(
                    ".".join(str(part) for part in detail["loc"])
                    + (": " if detail["loc"] else "")
                    + detail["msg"]
                    for detail in e.errors(include_url=False, include_input=False)
                ),
            ) from e

    def node(self, where: str, data: dict[str, Any]) -> None:
        """Add the node ``data`` states (read at ``where``, ``file line n``).

        Raises:
            SeedGraphError: It is invalid, repeats an id, or is one node
                more than ``max_nodes``.
        """
        self._properties(where, data.get("properties"))
        node: SeedNode = self._validated(where, SeedNode, data)
        first = self._seen.get(node.id)
        if first is not None:
            raise self.fail(where, f"duplicate node id {node.id!r} (first at {first})")
        self._seen[node.id] = where
        self.nodes.append(node)
        cap = self.context.source.max_nodes
        if len(self.nodes) > cap:
            raise self.fail(where, f"more nodes than max_nodes ({cap})")

    def edge(self, where: str, data: dict[str, Any]) -> None:
        """Add the edge ``data`` states.

        Raises:
            SeedGraphError: It is invalid, or is one edge more than
                ``max_edges``.
        """
        self._properties(where, data.get("properties"))
        self.edges.append(self._validated(where, SeedEdge, data))
        cap = self.context.source.max_edges
        if len(self.edges) > cap:
            raise self.fail(where, f"more edges than max_edges ({cap})")

    def graph(self, report: SeedLoadReport) -> SeedGraph:
        """The nodes and edges added, with ``report``."""
        return SeedGraph(
            nodes=tuple(self.nodes), edges=tuple(self.edges), report=report
        )
