# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenSearch query DSL subset of the embedded search engine, over one index.

Exactly the clauses ``OpenSearchRetriever`` and ``OpenSearchIndexer`` build:
``match_all``, ``match_none``, ``ids``, ``term``, ``terms``, ``range``,
``multi_match`` (``best_fields``; ``fuzziness``/``max_expansions`` are
accepted and ignored), ``bool`` (``must``/``should``/``filter``) and ``knn``
(``vector``/``k``/``filter``). The top-level ``hybrid`` query is the engine's
(it fuses across indices). A body's ``sort`` takes ``{field: {"order":
...}}`` entries (``sort_hits``). Any other clause or parameter raises
``StoreRequestRejectedError``, as OpenSearch answers a query it cannot parse
with a 400, so a new query shape fails a test instead of silently matching
something else.

Scoring follows OpenSearch: ``multi_match`` takes the best field's BM25
(``bm25``); ``bool`` sums its ``must`` and ``should`` scores (``filter`` adds
nothing; ``should`` is optional next to ``must`` or ``filter``, else one must
match); ``match_all``, ``ids`` and a ``term``/``terms``/``range`` clause
outside a filter score 1; ``knn`` is an exact search (filter first, then the
top ``k``) scored by the field's ``space_type``.

Field values: a ``keyword`` field or an array matches when any element does;
a ``text`` field matches a ``term`` against its analyzed tokens; the
``name.keyword`` sub-field matches ``name`` exactly; ``attributes.<path>`` is
resolved as a dotted path; an unmapped string is a keyword.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unified_kg_rag.adapters.local.search_engine.analyzers import get_analyzer
from unified_kg_rag.adapters.local.search_engine.bm25 import FieldIndex
from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

Scores = dict[str, float]

NUMERIC_TYPES = frozenset(
    {"integer", "long", "short", "byte", "double", "float", "half_float"}
)
SPACE_TYPES = frozenset({"cosinesimil", "l2", "innerproduct"})


def rejected(reason: str) -> StoreRequestRejectedError:
    """The error OpenSearch's 400 for an unparsable query maps to."""
    reason = f"parsing_exception: {reason}"
    return StoreRequestRejectedError(
        f"Embedded search engine rejected the request: {reason}", reason=reason
    )


@dataclass(frozen=True)
class FieldSpec:
    type: str
    analyzer: str | None = None
    dimension: int | None = None
    space_type: str | None = None
    # A multi-field's parent (``name`` of ``name.keyword``).
    source_path: str | None = None


def field_specs(mappings: dict[str, Any]) -> dict[str, FieldSpec]:
    """Mapped field path -> its spec, sub-fields and object properties included."""
    specs: dict[str, FieldSpec] = {}

    def walk(properties: dict[str, Any], prefix: str) -> None:
        for name, body in (properties or {}).items():
            if not isinstance(body, dict):
                continue
            path = f"{prefix}{name}"
            if "properties" in body:
                walk(body["properties"], f"{path}.")
            kind = body.get("type", "object")
            method = body.get("method") or {}
            specs[path] = FieldSpec(
                type=kind,
                analyzer=body.get("analyzer") if kind == "text" else None,
                dimension=body.get("dimension"),
                space_type=(
                    method.get("space_type", "l2") if kind == "knn_vector" else None
                ),
            )
            for sub_name, sub in (body.get("fields") or {}).items():
                specs[f"{path}.{sub_name}"] = FieldSpec(
                    type=sub.get("type", "keyword"),
                    analyzer=sub.get("analyzer") if sub.get("type") == "text" else None,
                    source_path=path,
                )

    walk(mappings.get("properties") or {}, "")
    return specs


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for element in value for item in _flatten(element)]
    return [] if value is None else [value]


def resolve_path(source: dict[str, Any], path: str) -> list[Any]:
    """The values at ``path`` in ``source`` (a top-level key, else dotted)."""
    if path in source:
        return _flatten(source[path])
    values: list[Any] = [source]
    for part in path.split("."):
        values = [
            element[part]
            for value in values
            for element in _flatten(value)
            if isinstance(element, dict) and part in element
        ]
    return [item for value in values for item in _flatten(value)]


def _keyword(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


@dataclass
class VectorField:
    ids: list[str]
    rows: dict[str, int]
    matrix: np.ndarray  # (n, dimension) float32
    norms: np.ndarray  # (n,) float32
    dimension: int
    space_type: str


@dataclass
class LoadedIndex:
    """One index held in memory: its documents, vectors and lazy text indexes."""

    name: str
    generation: int
    mappings: dict[str, Any]
    settings: dict[str, Any]
    sources: dict[str, dict[str, Any]]
    vectors: dict[str, VectorField]
    specs: dict[str, FieldSpec] = field(init=False)
    _text: dict[str, FieldIndex] = field(init=False, default_factory=dict)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.specs = field_specs(self.mappings)

    def text_index(self, path: str) -> FieldIndex:
        """The BM25 index of text field ``path``, built on first use."""
        with self._lock:
            built = self._text.get(path)
            if built is None:
                analyze = get_analyzer(self.specs[path].analyzer)
                built = FieldIndex()
                for doc_id, source in self.sources.items():
                    texts = [str(v) for v in resolve_path(source, path)]
                    built.add(doc_id, [t for text in texts for t in analyze(text)])
                self._text[path] = built
            return built


Evaluator = Callable[[LoadedIndex, Any], Scores]
_CLAUSES: dict[str, Evaluator] = {}


def _clause(name: str) -> Callable[[Evaluator], Evaluator]:
    def decorator(func: Evaluator) -> Evaluator:
        _CLAUSES[name] = func
        return func

    return decorator


def evaluate(index: LoadedIndex, query: Any) -> Scores:
    """The documents of ``index`` that ``query`` matches, with their scores."""
    if not isinstance(query, dict) or len(query) != 1:
        raise rejected(f"a query must be an object with one clause, got {query!r}")
    ((name, body),) = query.items()
    handler = _CLAUSES.get(name)
    if handler is None:
        raise rejected(f"unsupported query clause [{name}]")
    return handler(index, body)


def _require_keys(clause: str, body: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise rejected(f"[{clause}] takes an object, got {body!r}")
    unknown = set(body) - allowed
    if unknown:
        raise rejected(f"[{clause}] does not support {sorted(unknown)}")
    return body


def _single_field(clause: str, body: Any) -> tuple[str, Any]:
    if not isinstance(body, dict) or len(body) != 1:
        raise rejected(f"[{clause}] takes exactly one field, got {body!r}")
    ((path, value),) = body.items()
    return path, value


@_clause("match_all")
def _match_all(index: LoadedIndex, body: Any) -> Scores:
    _require_keys("match_all", body, set())
    return dict.fromkeys(index.sources, 1.0)


@_clause("match_none")
def _match_none(index: LoadedIndex, body: Any) -> Scores:
    _require_keys("match_none", body, set())
    return {}


@_clause("ids")
def _ids(index: LoadedIndex, body: Any) -> Scores:
    values = _require_keys("ids", body, {"values"}).get("values")
    if not isinstance(values, list):
        raise rejected("[ids] takes a list of values")
    return {str(v): 1.0 for v in values if str(v) in index.sources}


def _matching(
    index: LoadedIndex, path: str, accept: Callable[[list[Any], FieldSpec | None], bool]
) -> Scores:
    spec = index.specs.get(path)
    source_path = spec.source_path if spec and spec.source_path else path
    return {
        doc_id: 1.0
        for doc_id, source in index.sources.items()
        if accept(resolve_path(source, source_path), spec)
    }


def _term_matcher(
    index: LoadedIndex, path: str, values: list[Any]
) -> Callable[[list[Any], FieldSpec | None], bool]:
    spec = index.specs.get(path)
    if spec is not None and spec.type in NUMERIC_TYPES:
        numbers = {_number(v) for v in values}
        if None in numbers:
            raise rejected(f"a value of [{path}] is not a number: {values!r}")
        return lambda doc_values, _spec: any(_number(v) in numbers for v in doc_values)
    if spec is not None and spec.type == "knn_vector":
        raise rejected(f"[{path}] is a knn_vector field and cannot be filtered")
    keywords = {_keyword(v) for v in values}
    return lambda doc_values, _spec: any(_keyword(v) in keywords for v in doc_values)


def _term_docs(index: LoadedIndex, path: str, values: list[Any]) -> Scores:
    if path == "_id":
        return {doc_id: 1.0 for v in values if (doc_id := _keyword(v)) in index.sources}
    spec = index.specs.get(path)
    if spec is not None and spec.type == "text":
        # A term is not analyzed: it matches a token of the field as indexed.
        text = index.text_index(path)
        return {doc_id: 1.0 for v in values for doc_id in text.docs_with(_keyword(v))}
    return _matching(index, path, _term_matcher(index, path, values))


@_clause("term")
def _term(index: LoadedIndex, body: Any) -> Scores:
    path, value = _single_field("term", body)
    if isinstance(value, dict):
        value = _require_keys("term", value, {"value"}).get("value")
    if isinstance(value, list | dict) or value is None:
        raise rejected(f"[term] takes one value for [{path}], got {value!r}")
    return _term_docs(index, path, [value])


@_clause("terms")
def _terms(index: LoadedIndex, body: Any) -> Scores:
    path, values = _single_field("terms", body)
    if not isinstance(values, list):
        raise rejected(f"[terms] takes a list for [{path}], got {values!r}")
    return _term_docs(index, path, values)


_RANGE_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}


@_clause("range")
def _range(index: LoadedIndex, body: Any) -> Scores:
    path, bounds = _single_field("range", body)
    bounds = _require_keys("range", bounds, set(_RANGE_OPERATORS))
    spec = index.specs.get(path)
    numeric = spec is not None and spec.type in NUMERIC_TYPES
    if numeric and any(_number(v) is None for v in bounds.values()):
        raise rejected(f"a bound of [{path}] is not a number: {bounds!r}")

    numeric_bounds = {op: _number(bound) for op, bound in bounds.items()}

    def accept(doc_values: list[Any], _spec: FieldSpec | None) -> bool:
        for value in doc_values:
            is_number = isinstance(value, int | float) and not isinstance(value, bool)
            number = _number(value) if numeric or is_number else None
            if number is not None:
                if all(b is not None for b in numeric_bounds.values()) and all(
                    _RANGE_OPERATORS[op](number, b) for op, b in numeric_bounds.items()
                ):
                    return True
            elif all(
                _RANGE_OPERATORS[op](_keyword(value), _keyword(b))
                for op, b in bounds.items()
            ):
                return True
        return False

    return _matching(index, path, accept)


def _clauses(clause: str, body: Any) -> list[Any]:
    if body is None:
        return []
    if isinstance(body, dict):
        return [body]
    if isinstance(body, list):
        return body
    raise rejected(f"[bool.{clause}] takes an object or a list, got {body!r}")


@_clause("bool")
def _bool(index: LoadedIndex, body: Any) -> Scores:
    body = _require_keys("bool", body, {"must", "should", "filter"})
    must = [evaluate(index, q) for q in _clauses("must", body.get("must"))]
    should = [evaluate(index, q) for q in _clauses("should", body.get("should"))]
    filters = [evaluate(index, q) for q in _clauses("filter", body.get("filter"))]
    required = [*must, *filters]
    candidates: set[str]
    if required:
        smallest: Scores = min(required, key=len)
        candidates = set(smallest)
        for matched in required:
            candidates.intersection_update(matched)
    elif should:
        candidates = {doc_id for scores in should for doc_id in scores}
    else:
        # An empty bool matches every document.
        return dict.fromkeys(index.sources, 1.0)
    if not must and not should:
        return dict.fromkeys(candidates, 0.0)
    return {
        doc_id: sum(s.get(doc_id, 0.0) for s in must)
        + sum(s.get(doc_id, 0.0) for s in should)
        for doc_id in candidates
    }


@_clause("multi_match")
def _multi_match(index: LoadedIndex, body: Any) -> Scores:
    body = _require_keys(
        "multi_match", body, {"query", "fields", "fuzziness", "max_expansions"}
    )
    text, fields = body.get("query"), body.get("fields")
    if not isinstance(text, str) or not isinstance(fields, list) or not fields:
        raise rejected("[multi_match] takes a query string and a list of fields")
    best: Scores = {}
    for path in fields:
        if not isinstance(path, str) or "^" in path or "*" in path:
            raise rejected(f"[multi_match] field [{path}] is not supported")
        spec = index.specs.get(path)
        if spec is None or spec.type != "text":
            continue  # OpenSearch skips an unmapped field.
        tokens = get_analyzer(spec.analyzer)(text)
        for doc_id, score in index.text_index(path).score(tokens).items():
            if score > best.get(doc_id, -1.0):
                best[doc_id] = score
    return best


def vector_scores(
    vectors: VectorField, query: np.ndarray, rows: np.ndarray | None = None
) -> np.ndarray:
    """OpenSearch's score of ``query`` against ``rows`` of the field (every
    row when ``None``, scored in place rather than on a copy)."""
    matrix = vectors.matrix if rows is None else vectors.matrix[rows]
    if vectors.space_type == "cosinesimil":
        norms = vectors.norms if rows is None else vectors.norms[rows]
        cosine = matrix @ query / (norms * float(np.linalg.norm(query)))
        return np.asarray((1.0 + cosine) / 2.0)
    if vectors.space_type == "l2":
        return np.asarray(1.0 / (1.0 + np.sum((matrix - query) ** 2, axis=1)))
    product = matrix @ query
    return np.asarray(
        np.where(product >= 0, product + 1.0, 1.0 / (1.0 - np.minimum(product, 0.0)))
    )


def _top_k(ids: list[str], scores: np.ndarray, k: int) -> Scores:
    """The ``k`` best ``ids`` by score, ties by id: only the rows scoring at
    least the k-th best score are sorted."""
    if len(ids) > k and not np.isnan(scores).any():
        kth = np.partition(scores, len(scores) - k)[len(scores) - k]
        candidates = np.flatnonzero(scores >= kth).tolist()
    else:
        candidates = range(len(ids))
    ranked = sorted(
        ((ids[i], float(scores[i])) for i in candidates), key=lambda p: (-p[1], p[0])
    )
    return dict(ranked[:k])


@_clause("knn")
def _knn(index: LoadedIndex, body: Any) -> Scores:
    path, params = _single_field("knn", body)
    params = _require_keys("knn", params, {"vector", "k", "filter"})
    spec = index.specs.get(path)
    if spec is None or spec.type != "knn_vector":
        raise rejected(f"Field '{path}' is not knn_vector type.")
    vector, k = params.get("vector"), params.get("k")
    if not isinstance(vector, list) or not isinstance(k, int) or k < 1:
        raise rejected("[knn] takes a vector and a positive k")
    if len(vector) != spec.dimension:
        raise rejected(
            f"Query vector has invalid dimension: {len(vector)}. "
            f"Dimension should be: {spec.dimension}"
        )
    query = np.asarray(vector, dtype=np.float32)
    if spec.space_type == "cosinesimil" and not np.any(query):
        raise rejected("zero vector is not supported when space type is [cosinesimil]")
    allowed = evaluate(index, params["filter"]) if "filter" in params else None
    stored = index.vectors.get(path)
    if stored is None:
        return {}
    if allowed is None:
        return _top_k(stored.ids, vector_scores(stored, query), k)
    ids = [i for i in stored.ids if i in allowed]
    if not ids:
        return {}
    rows = np.fromiter((stored.rows[i] for i in ids), dtype=np.int64, count=len(ids))
    return _top_k(ids, vector_scores(stored, query, rows), k)


Hit = tuple[LoadedIndex, str, float]


def _sort_fields(sort: Any) -> list[tuple[str, bool]]:
    """``[{field: {"order": "asc" | "desc"}}, ...]`` -> (field, descending)."""
    if not isinstance(sort, list):
        raise rejected(f"[sort] takes a list, got {sort!r}")
    fields = []
    for entry in sort:
        path, options = _single_field("sort", entry)
        order = _require_keys("sort", options, {"order"}).get("order", "asc")
        if order not in ("asc", "desc"):
            raise rejected(f"[sort] order must be asc or desc, got {order!r}")
        fields.append((path, order == "desc"))
    return fields


def sort_hits(hits: list[Hit], sort: Any) -> list[Hit]:
    """``hits`` in the order of a search body's ``sort``, as OpenSearch sorts.

    A numeric field sorts by number, any other by keyword; a multi-valued
    field by its smallest value ascending and its largest descending; a
    document without the field comes last either way. Ties keep the score
    order.
    """
    for path, descending in reversed(_sort_fields(sort)):

        def value(hit: Hit, path: str = path, descending: bool = descending) -> Any:
            index, doc_id, _score = hit
            spec = index.specs.get(path)
            raw = resolve_path(index.sources[doc_id], path)
            if spec is not None and spec.type in NUMERIC_TYPES:
                values: list[Any] = [n for v in raw if (n := _number(v)) is not None]
            else:
                values = [_keyword(v) for v in raw]
            if not values:
                return None
            return max(values) if descending else min(values)

        keyed = [(value(hit), hit) for hit in hits]
        present = [pair for pair in keyed if pair[0] is not None]
        present.sort(key=lambda pair: pair[0], reverse=descending)
        hits = [hit for _, hit in present] + [hit for key, hit in keyed if key is None]
    return hits
