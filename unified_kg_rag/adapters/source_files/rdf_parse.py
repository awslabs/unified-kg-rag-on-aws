# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RDF files parsed into one rdflib graph (the ``ontology`` extra).

A source locates its files and checks their bytes (:mod:`.rdf_files`), then
hands each file's bytes to :func:`parse`, which adds what they state to
:class:`Statements`: the triples of every file together (named graphs
merged into one), and the prefixes the files bind. The bytes are parsed as
given, never re-read from a location; errors are the caller's
:class:`.locations.SourceTerms`.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from rdflib import Dataset, Graph
from rdflib.namespace import NamespaceManager

from .locations import SourceTerms

logger = logging.getLogger(__name__)

# Formats parsed into a dataset: their statements may sit in named graphs.
_QUAD_FORMATS = ("nquads", "trig", "json-ld")
_TRIG_PREFIX = re.compile(
    r"(?:@prefix|\bPREFIX)\s+([A-Za-z][\w.-]*)?:\s*<([^>\s]*)>", re.IGNORECASE
)
_WHITESPACE = re.compile(r"\s")
# A JSON-LD term whose IRI ends so is a prefix (JSON-LD 1.1, 4.1.5).
_PREFIX_END = (":", "/", "?", "#", "[", "]", "@")


class Statements:
    """What the files state together: their triples and prefix bindings."""

    def __init__(self) -> None:
        self.graph = Graph(bind_namespaces="none")
        self._bindings: dict[str, set[str]] = {}

    def bind(self, prefixes: Iterable[tuple[str, str]]) -> None:
        for prefix, namespace in prefixes:
            # No IRI has whitespace: the parsers let one through.
            if namespace and not _WHITESPACE.search(namespace):
                self._bindings.setdefault(prefix, set()).add(namespace)

    def prefixes(self) -> dict[str, str]:
        """The prefixes the files bind alike (one bound two ways is left
        out: the options must bind it)."""
        ambiguous = sorted(p for p, ns in self._bindings.items() if len(ns) > 1)
        if ambiguous:
            logger.debug(
                "RDF prefixes bound to several namespaces, left out: %s",
                ", ".join(ambiguous),
            )
        return {
            prefix: next(iter(namespaces))
            for prefix, namespaces in sorted(self._bindings.items())
            if len(namespaces) == 1
        }


def parse(
    statements: Statements,
    name: str,
    base: str,
    data: bytes,
    format_: str,
    *,
    terms: SourceTerms,
) -> None:
    """Add what ``data``, the bytes of the file ``name`` in ``format_``,
    states.

    ``data`` is parsed as given, never re-read from its location: ``base``
    (the file's ``file://`` URI or ``s3://`` URL) is only the base of its
    relative IRIs.

    Raises:
        terms.error: ``data`` does not parse.
    """
    public_id = base
    try:
        if format_ in _QUAD_FORMATS:
            dataset = Dataset()
            dataset.namespace_manager = NamespaceManager(dataset, "none")
            dataset.parse(data=data, format=format_, publicID=public_id)
            for subject, predicate, obj, _ in dataset.quads():
                statements.graph.add((subject, predicate, obj))
        else:
            graph = Graph(bind_namespaces="none")
            graph.parse(data=data, format=format_, publicID=public_id)
            statements.graph += graph
            statements.bind((p, str(ns)) for p, ns in graph.namespaces())
    # The parsers raise many types (BadSyntax, SAXParseException, ValueError,
    # ...): each is a file that does not parse.
    except Exception as e:  # noqa: BLE001
        raise terms(
            f"{terms.noun} file '{name}' does not parse as {format_}: "
            f"{_first_line(e)}"
        ) from e
    # The dataset parsers bind the library's own prefixes too: read the
    # file's.
    if format_ == "trig":
        text = data.decode("utf-8", errors="replace")
        statements.bind((p or "", ns) for p, ns in _TRIG_PREFIX.findall(text))
    elif format_ == "json-ld":
        statements.bind(_json_ld_prefixes(json.loads(data)).items())


_LINE_OF = re.compile(r"^at line (\d+) of ")


def _first_line(error: BaseException) -> str:
    """What ``error`` says is wrong, without the ontology's text: the
    Turtle and TriG parsers' "Bad syntax" line (up to the quoted context)
    with its line number, the XML parser's message with its position, else
    the first line up to its first ``:`` (the N-Triples parser quotes the
    line after it)."""
    from xml.sax import SAXParseException

    if isinstance(error, SAXParseException):
        return (
            f"line {error.getLineNumber()}, column {error.getColumnNumber()}: "
            f"{error.getMessage()}"
        )
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    if not lines:
        return type(error).__name__
    for line in lines:
        if line.startswith("Bad syntax"):
            line_of = _LINE_OF.match(lines[0])
            where = f"line {line_of.group(1)}: " if line_of else ""
            return where + line.partition(" at ^ in:")[0]
    return lines[0].partition(":")[0][:200]


def _json_ld_prefixes(document: Any) -> dict[str, str]:
    """The prefixes the top-level contexts of ``document`` define."""
    bindings: dict[str, str] = {}
    for top in document if isinstance(document, list) else [document]:
        if not isinstance(top, dict):
            continue
        contexts = top.get("@context")
        for context in contexts if isinstance(contexts, list) else [contexts]:
            if not isinstance(context, dict):
                continue
            for term, definition in context.items():
                iri = (
                    definition.get("@id")
                    if isinstance(definition, dict)
                    else definition
                )
                if (
                    isinstance(iri, str)
                    and not term.startswith("@")
                    and ":" not in term
                    and "://" in iri
                    and iri.endswith(_PREFIX_END)
                ):
                    bindings[term] = iri
    return bindings
