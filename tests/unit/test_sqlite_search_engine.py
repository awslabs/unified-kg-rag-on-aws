# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SQLiteSearchEngine``: the query DSL subset, administration and errors.

The engine stands in for OpenSearch behind the unchanged ``OpenSearchIndexer``
and ``OpenSearchRetriever``, so besides each DSL clause this runs every request
shape ``OpenSearchRetriever._build_search_request`` builds against it, and the
indexer's alias swap, ``_meta`` and settings checks.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from opensearchpy.exceptions import NotFoundError

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.aws.opensearch import is_index_not_found
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.local.search_engine import engine as engine_module
from unified_kg_rag.adapters.local.search_engine.query import (
    LoadedIndex,
    VectorField,
    evaluate,
    vector_scores,
)
from unified_kg_rag.adapters.retrievers import OpenSearchRetriever
from unified_kg_rag.adapters.storage import OpenSearchIndexer
from unified_kg_rag.domain.models import (
    Claim,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    SearchQuery,
    SearchType,
    TextUnit,
)
from unified_kg_rag.domain.retrieval.hybrid_normalization import combine_min_max
from unified_kg_rag.shared import AWSServiceError, ConfigurationError
from unified_kg_rag.shared.exceptions import (
    StoreConfigurationError,
    StoreRequestRejectedError,
    StoreTimeoutError,
)

pytestmark = pytest.mark.unit

INDEX = "things-1"
ALIAS = "things"
PIPELINE = "hybrid-pipeline"


def _knn(space_type: str) -> dict[str, Any]:
    return {
        "type": "knn_vector",
        "dimension": 2,
        "method": {"name": "hnsw", "space_type": space_type, "engine": "lucene"},
    }


MAPPING: dict[str, Any] = {
    "settings": {"number_of_shards": 1, "index.knn": True},
    "mappings": {
        "_meta": {"embedding_model": "m", "embedding_dimension": 2},
        "dynamic_templates": [
            {
                "strings_as_keywords": {
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }
            }
        ],
        "properties": {
            "id": {"type": "keyword"},
            "name": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "description": {"type": "text", "analyzer": "standard"},
            "type": {"type": "keyword"},
            "rank": {"type": "double"},
            "text_unit_ids": {"type": "keyword"},
            "attributes": {"type": "object", "dynamic": True},
            "vec": _knn("cosinesimil"),
            "vec_l2": _knn("l2"),
            "vec_ip": _knn("innerproduct"),
        },
    },
}

DOCS: list[dict[str, Any]] = [
    {
        "id": "a",
        "name": "Vendor",
        "description": "Vendor supplies widgets",
        "type": "ORG",
        "rank": 3,
        "text_unit_ids": ["t1", "t2"],
        "attributes": {"region": {"code": "EU"}},
        "attr_category": "contract",
        "vec": [1.0, 0.0],
        "vec_l2": [0.0, 0.0],
        "vec_ip": [1.0, 0.0],
    },
    {
        "id": "b",
        "name": "Buyer",
        "description": "Buyer orders widgets from Vendor",
        "type": "ORG",
        "rank": 2,
        "text_unit_ids": ["t1"],
        "attributes": {"region": {"code": "US"}},
        "vec": [0.0, 1.0],
        "vec_l2": [3.0, 4.0],
        "vec_ip": [-1.0, 0.0],
    },
    {
        "id": "c",
        "name": "Depot",
        "description": "Depot stores widgets",
        "type": "LOCATION",
        "rank": 1,
        "text_unit_ids": ["t2"],
        "vec": [1.0, 1.0],
        "vec_l2": [1.0, 0.0],
        "vec_ip": [0.5, 0.5],
    },
]


@pytest.fixture
def engine(tmp_path: Path) -> SQLiteSearchEngine:
    engine = SQLiteSearchEngine(tmp_path / "vector.sqlite3")
    engine.create_index(INDEX, MAPPING)
    engine.update_alias(ALIAS, INDEX)
    assert engine.bulk_index(INDEX, DOCS) == {"errors": False, "items": []}
    engine.put_search_pipeline(
        PIPELINE,
        {
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.25, 0.75]},
                        },
                    }
                }
            ]
        },
    )
    yield engine
    engine.close()


def _search(engine: SQLiteSearchEngine, query: dict[str, Any], **extra: Any) -> dict:
    body = {"size": extra.pop("size", 10), "query": query}
    body.update(extra.pop("body", {}))
    return engine.search(index=ALIAS, body=body, **extra)


def _hits(engine: SQLiteSearchEngine, query: dict[str, Any], **extra: Any) -> dict:
    response = _search(engine, query, **extra)
    return {hit["_id"]: hit["_score"] for hit in response["hits"]["hits"]}


def _ids(engine: SQLiteSearchEngine, query: dict[str, Any], **extra: Any) -> list:
    return [hit["_id"] for hit in _search(engine, query, **extra)["hits"]["hits"]]


# --------------------------------------------------------------------------- #
# DSL clauses
# --------------------------------------------------------------------------- #


def test_match_all_scores_one_and_match_none_matches_nothing(engine) -> None:
    assert _hits(engine, {"match_all": {}}) == {"a": 1.0, "b": 1.0, "c": 1.0}
    assert _hits(engine, {"match_none": {}}) == {}


def test_ids_query_scores_one_and_skips_missing_ids(engine) -> None:
    assert _hits(engine, {"ids": {"values": ["c", "a", "zz"]}}) == {"a": 1.0, "c": 1.0}


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        ({"term": {"type": "ORG"}}, {"a", "b"}),
        ({"term": {"type": {"value": "LOCATION"}}}, {"c"}),
        # An array matches when any element does.
        ({"term": {"text_unit_ids": "t2"}}, {"a", "c"}),
        ({"terms": {"text_unit_ids": ["t1", "zz"]}}, {"a", "b"}),
        ({"terms": {"id": ["a", "c"]}}, {"a", "c"}),
        # A text field matches a term against its analyzed (lowercased) tokens;
        # the term itself is not analyzed.
        ({"term": {"name": "vendor"}}, {"a"}),
        ({"term": {"name": "Vendor"}}, set()),
        # The keyword sub-field matches the original value exactly.
        ({"term": {"name.keyword": "Vendor"}}, {"a"}),
        ({"term": {"name.keyword": "vendor"}}, set()),
        # attributes.<path> is a dotted path into the dynamic object.
        ({"term": {"attributes.region.code": "US"}}, {"b"}),
        # An unmapped string is a keyword.
        ({"term": {"attr_category": "contract"}}, {"a"}),
        ({"term": {"rank": 2}}, {"b"}),
        ({"terms": {"rank": ["1", 3]}}, {"a", "c"}),
        ({"range": {"rank": {"gte": 2}}}, {"a", "b"}),
        ({"range": {"rank": {"gt": 1, "lt": 3}}}, {"b"}),
        ({"range": {"rank": {"lte": 1}}}, {"c"}),
        ({"range": {"type": {"gte": "M"}}}, {"a", "b"}),
    ],
)
def test_filters(engine, clause: dict, expected: set) -> None:
    hits = _hits(engine, {"bool": {"filter": [clause]}})
    assert set(hits) == expected
    # A filter-only bool scores 0.
    assert all(score == 0.0 for score in hits.values())


def test_bool_must_with_filter_scores_the_must_clause(engine) -> None:
    lexical = {"multi_match": {"query": "widgets vendor", "fields": ["description"]}}
    plain = _hits(engine, lexical)
    filtered = _hits(
        engine, {"bool": {"must": [lexical], "filter": [{"term": {"type": "ORG"}}]}}
    )
    assert set(filtered) == {"a", "b"}
    assert filtered == {k: plain[k] for k in ("a", "b")}


def test_bool_should_is_optional_next_to_must_and_adds_its_score(engine) -> None:
    must = {"multi_match": {"query": "widgets", "fields": ["description"]}}
    should = {"multi_match": {"query": "depot", "fields": ["description"]}}
    base = _hits(engine, must)
    both = _hits(engine, {"bool": {"must": [must], "should": [should]}})
    assert set(both) == {"a", "b", "c"}
    assert both["a"] == pytest.approx(base["a"])
    assert both["c"] > base["c"]
    # Next to a filter alone, should is optional too (and is the score).
    with_filter = _hits(
        engine, {"bool": {"filter": [{"term": {"type": "ORG"}}], "should": [should]}}
    )
    assert with_filter == {"a": 0.0, "b": 0.0}


def test_bool_with_only_should_needs_one_match(engine) -> None:
    should = [
        {"term": {"type": "LOCATION"}},
        {"term": {"attr_category": "contract"}},
    ]
    assert _hits(engine, {"bool": {"should": should}}) == {"a": 1.0, "c": 1.0}
    assert set(_hits(engine, {"bool": {}})) == {"a", "b", "c"}


def test_multi_match_takes_the_best_field(engine) -> None:
    by_name = _hits(engine, {"multi_match": {"query": "vendor", "fields": ["name"]}})
    by_description = _hits(
        engine, {"multi_match": {"query": "vendor", "fields": ["description"]}}
    )
    both = _hits(
        engine,
        {
            "multi_match": {
                "query": "vendor",
                "fields": ["name", "description"],
                "fuzziness": "AUTO",
                "max_expansions": 10,
            }
        },
    )
    assert set(both) == {"a", "b"}
    assert both["a"] == pytest.approx(max(by_name["a"], by_description["a"]))
    assert both["b"] == pytest.approx(by_description["b"])
    # An unmapped or non-text field is skipped, as by OpenSearch.
    assert (
        _hits(engine, {"multi_match": {"query": "ORG", "fields": ["type", "nope"]}})
        == {}
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("fire system", "fire"),
        ("interest bill amount", "bill"),
        ("computer name detail", "computer"),
    ],
)
def test_english_multi_match_keeps_content_words(
    tmp_path: Path, query: str, expected: str
) -> None:
    # Only Lucene's 33 English stop words are dropped, as by OpenSearch; a
    # wider list (scikit-learn's 318) drops these whole queries.
    engine = SQLiteSearchEngine(tmp_path / "english.sqlite3")
    try:
        engine.create_index(
            "en",
            {
                "mappings": {
                    "properties": {"text": {"type": "text", "analyzer": "english"}}
                }
            },
        )
        engine.bulk_index(
            "en",
            [
                {"id": "fire", "text": "The fire system failed"},
                {"id": "bill", "text": "Interest on the bill amount is due"},
                {"id": "computer", "text": "Each computer has a name and a detail"},
            ],
        )
        response = engine.search(
            index="en",
            body={"query": {"multi_match": {"query": query, "fields": ["text"]}}},
        )
        assert [hit["_id"] for hit in response["hits"]["hits"]] == [expected]
    finally:
        engine.close()


def test_knn_cosine_scores_and_top_k(engine) -> None:
    hits = _hits(engine, {"knn": {"vec": {"vector": [1.0, 0.0], "k": 2}}})
    # (1 + cos) / 2: a = 1, c = (1 + 1/sqrt 2) / 2, b = 0.5 (not in the top 2)
    assert hits == {
        "a": pytest.approx(1.0),
        "c": pytest.approx((1 + 1 / math.sqrt(2)) / 2),
    }


def test_knn_l2_and_innerproduct_scores(engine) -> None:
    l2 = _hits(engine, {"knn": {"vec_l2": {"vector": [0.0, 0.0], "k": 3}}})
    assert l2 == {
        "a": pytest.approx(1.0),
        "c": pytest.approx(1 / 2),
        "b": pytest.approx(1 / 26),
    }
    ip = _hits(engine, {"knn": {"vec_ip": {"vector": [1.0, 0.0], "k": 3}}})
    # ip >= 0: ip + 1; ip < 0: 1 / (1 - ip)
    assert ip == {
        "a": pytest.approx(2.0),
        "c": pytest.approx(1.5),
        "b": pytest.approx(0.5),
    }


def test_knn_filters_first_then_takes_top_k(engine) -> None:
    hits = _hits(
        engine,
        {
            "knn": {
                "vec": {
                    "vector": [1.0, 0.0],
                    "k": 1,
                    "filter": {"bool": {"filter": [{"term": {"type": "ORG"}}]}},
                }
            }
        },
    )
    assert list(hits) == ["a"]
    hits = _hits(
        engine,
        {
            "knn": {
                "vec": {
                    "vector": [1.0, 0.0],
                    "k": 1,
                    "filter": {"bool": {"filter": [{"term": {"id": "b"}}]}},
                }
            }
        },
    )
    assert hits == {"b": pytest.approx(0.5)}


class _NoCopyMatrix(np.ndarray):
    """A stored matrix that fails when a query copies rows out of it."""

    def __getitem__(self, key: Any) -> Any:
        raise AssertionError("the kNN query copied the stored matrix")


def _vector_index(space_type: str, matrix: np.ndarray) -> LoadedIndex:
    ids = [f"d{i:03d}" for i in range(len(matrix))]
    spec = {
        "type": "knn_vector",
        "dimension": matrix.shape[1],
        "method": {"name": "hnsw", "space_type": space_type, "engine": "lucene"},
    }
    return LoadedIndex(
        name="vectors",
        generation=1,
        mappings={
            "properties": {"vec": spec, "group": {"type": "keyword"}},
        },
        settings={},
        sources={doc_id: {"group": str(i % 3)} for i, doc_id in enumerate(ids)},
        vectors={
            "vec": VectorField(
                ids=ids,
                rows={doc_id: i for i, doc_id in enumerate(ids)},
                matrix=matrix,
                norms=np.linalg.norm(matrix, axis=1).astype(np.float32),
                dimension=matrix.shape[1],
                space_type=space_type,
            )
        },
    )


def _knn_by_full_sort(
    index: LoadedIndex, query: list[float], k: int, group: str | None
) -> dict[str, float]:
    """The kNN answer as a sort of every candidate's score."""
    stored = index.vectors["vec"]
    ids = [i for i in stored.ids if group is None or index.sources[i]["group"] == group]
    rows = np.array([stored.rows[i] for i in ids], dtype=np.int64)
    scores = vector_scores(stored, np.asarray(query, dtype=np.float32), rows)
    ranked = sorted(zip(ids, scores.tolist(), strict=True), key=lambda p: (-p[1], p[0]))
    return dict(ranked[:k])


@pytest.mark.parametrize("space_type", ["cosinesimil", "l2", "innerproduct"])
def test_knn_top_k_matches_a_full_sort_with_ties(space_type: str) -> None:
    # Few distinct vectors, so many documents tie on the k-th score: the
    # ties are broken by id, as a sort of every score breaks them.
    rng = np.random.default_rng(7)
    distinct = rng.integers(-2, 3, size=(6, 4)).astype(np.float32)
    distinct[np.all(distinct == 0, axis=1)] = 1.0
    matrix = distinct[rng.integers(0, len(distinct), size=120)]
    index = _vector_index(space_type, matrix)
    for query in ([1.0, 0.0, -1.0, 2.0], distinct[0].tolist()):
        for k in (1, 5, 17, 40, 119, 120, 500):
            for group in (None, "1"):
                body: dict[str, Any] = {"vector": query, "k": k}
                if group is not None:
                    body["filter"] = {"term": {"group": group}}
                assert evaluate(index, {"knn": {"vec": body}}) == _knn_by_full_sort(
                    index, query, k, group
                ), (k, group)


def test_unfiltered_knn_scores_the_stored_matrix_in_place() -> None:
    # Copying the matrix for every query made a query over 50k vectors of
    # 1024 dimensions spend most of its time in the copy.
    matrix = np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0]].view(_NoCopyMatrix)
    index = _vector_index("cosinesimil", matrix)
    hits = evaluate(index, {"knn": {"vec": {"vector": [1.0, 0.0, 0.0, 0.0], "k": 2}}})
    assert hits == {"d000": 1.0, "d004": 1.0}


def test_bool_should_over_knn_clauses_sums_them(engine) -> None:
    one = _hits(engine, {"knn": {"vec": {"vector": [1.0, 0.0], "k": 3}}})
    two = _hits(engine, {"knn": {"vec_ip": {"vector": [1.0, 0.0], "k": 3}}})
    summed = _hits(
        engine,
        {
            "bool": {
                "should": [
                    {"knn": {"vec": {"vector": [1.0, 0.0], "k": 3}}},
                    {"knn": {"vec_ip": {"vector": [1.0, 0.0], "k": 3}}},
                ]
            }
        },
    )
    assert summed == {k: pytest.approx(one[k] + two[k]) for k in one}


def test_hybrid_fuses_with_the_stored_pipeline_weights(engine) -> None:
    lexical = {"multi_match": {"query": "vendor widgets", "fields": ["description"]}}
    vector = {"knn": {"vec": {"vector": [0.0, 1.0], "k": 2}}}
    lexical_hits = sorted(_hits(engine, lexical).items(), key=lambda p: (-p[1], p[0]))
    vector_hits = sorted(_hits(engine, vector).items(), key=lambda p: (-p[1], p[0]))
    keyed = [
        [(f"{INDEX}\x00{doc_id}", score) for doc_id, score in hits]
        for hits in (lexical_hits, vector_hits)
    ]
    expected = [
        (key.split("\x00")[1], score)
        for key, score in combine_min_max(keyed, [0.25, 0.75])
    ]
    response = _search(
        engine, {"hybrid": {"queries": [lexical, vector]}}, search_pipeline=PIPELINE
    )
    got = [(hit["_id"], hit["_score"]) for hit in response["hits"]["hits"]]
    assert got == [(i, pytest.approx(s)) for i, s in expected]
    assert len(got) == 3


def test_hybrid_normalizes_each_sub_query_over_its_top_size_hits(engine) -> None:
    lexical = {"match_all": {}}
    vector = {"knn": {"vec": {"vector": [1.0, 0.0], "k": 3}}}
    response = _search(
        engine,
        {"hybrid": {"queries": [lexical, vector]}},
        search_pipeline=PIPELINE,
        size=1,
    )
    # Size 1: the vector sub-query keeps only "a" (max == min -> 1.0), and the
    # all-1.0 lexical scores normalize to 1.0 too.
    assert [(h["_id"], h["_score"]) for h in response["hits"]["hits"]] == [
        ("a", pytest.approx(1.0))
    ]


def test_lexical_and_vector_queries_return_raw_scores(engine) -> None:
    lexical = _hits(engine, {"multi_match": {"query": "depot", "fields": ["name"]}})
    # BM25, not normalized: N = 3, n = 1, tf = dl = avgdl = 1.
    assert lexical == {"c": pytest.approx(math.log(1 + 2.5 / 1.5) * 2.2 / 2.2)}
    vector = _hits(engine, {"knn": {"vec_l2": {"vector": [0.0, 1.0], "k": 1}}})
    assert vector == {"a": pytest.approx(0.5)}  # 1 / (1 + 1)


def test_ties_break_by_id_and_size_truncates(engine) -> None:
    response = _search(engine, {"match_all": {}}, size=2)
    assert [h["_id"] for h in response["hits"]["hits"]] == ["a", "b"]
    assert response["hits"]["total"] == {"value": 3, "relation": "eq"}
    assert _search(engine, {"match_all": {}}, size=0)["hits"]["hits"] == []


def test_source_excludes_and_vectors(engine) -> None:
    (hit,) = _search(
        engine,
        {"ids": {"values": ["a"]}},
        body={"_source": {"excludes": ["*_l2", "vec_ip"]}},
    )["hits"]["hits"]
    assert hit["_index"] == INDEX
    assert hit["_source"]["vec"] == [1.0, 0.0]
    assert "vec_l2" not in hit["_source"] and "vec_ip" not in hit["_source"]
    assert hit["_source"]["attributes"] == {"region": {"code": "EU"}}
    (hit,) = _search(engine, {"term": {"id": "a"}}, body={"_source": False})["hits"][
        "hits"
    ]
    assert "_source" not in hit


def test_id_lookups_read_sql_without_loading_the_index(engine, monkeypatch) -> None:
    monkeypatch.setattr(
        engine, "_load", lambda index: pytest.fail("loaded the index for an id lookup")
    )
    assert _hits(engine, {"ids": {"values": ["b", "a"]}}) == {"a": 1.0, "b": 1.0}
    assert _hits(engine, {"bool": {"filter": [{"terms": {"id": ["c", "x"]}}]}}) == {
        "c": 0.0
    }
    (hit,) = _search(
        engine,
        {"bool": {"filter": {"term": {"id": "a"}}}},
        body={"_source": {"excludes": ["*_l2", "*_ip"]}},
    )["hits"]["hits"]
    assert hit["_source"]["vec"] == [1.0, 0.0] and "vec_l2" not in hit["_source"]


# --------------------------------------------------------------------------- #
# Fail fast on anything outside the subset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        {"match": {"name": "vendor"}},
        {"bool": {"must_not": [{"term": {"type": "ORG"}}]}},
        {"bool": {"must": [], "minimum_should_match": 1}},
        {"multi_match": {"query": "x", "fields": ["name"], "type": "phrase"}},
        {"multi_match": {"query": "x", "fields": ["name^2"]}},
        {"knn": {"vec": {"vector": [1.0, 0.0], "k": 1, "min_score": 0.5}}},
        {"knn": {"name": {"vector": [1.0, 0.0], "k": 1}}},
        {"knn": {"vec": {"vector": [0.0, 0.0], "k": 1}}},
        {"term": {"rank": "high"}},
        {"range": {"rank": {"gte": 1, "format": "x"}}},
        {"bool": {"filter": [{"hybrid": {"queries": []}}]}},
        {"term": {"type": "ORG"}, "match_all": {}},
    ],
)
def test_unsupported_queries_are_rejected(engine, query: dict) -> None:
    with pytest.raises(StoreRequestRejectedError):
        _search(engine, query)


def test_invalid_query_vector_dimension_is_the_opensearch_rejection(engine) -> None:
    with pytest.raises(StoreRequestRejectedError) as excinfo:
        _search(engine, {"knn": {"vec": {"vector": [1.0, 0.0, 0.0], "k": 1}}})
    assert "query vector has invalid dimension" in (excinfo.value.reason or "").lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index": ALIAS, "body": {"query": {"match_all": {}}, "sort": ["id"]}},
        {"index": ALIAS, "body": {"query": {"match_all": {}}}, "routing": "x"},
        {"index": ALIAS, "body": {"size": 10001}},
        {"index": ALIAS, "body": {"_source": {"includes": ["id"]}}},
        # hybrid needs a pipeline, an existing one, with one weight per query
        {
            "index": ALIAS,
            "body": {"query": {"hybrid": {"queries": [{"match_all": {}}]}}},
        },
        {
            "index": ALIAS,
            "body": {"query": {"hybrid": {"queries": [{"match_all": {}}]}}},
            "search_pipeline": PIPELINE,
        },
        {"index": ALIAS, "body": {}, "search_pipeline": "missing"},
    ],
)
def test_unsupported_requests_are_rejected(engine, kwargs: dict) -> None:
    with pytest.raises(StoreRequestRejectedError):
        engine.search(**kwargs)


@pytest.mark.parametrize(
    "body",
    [
        {"phase_results_processors": []},
        {
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "l2"},
                        "combination": {"technique": "arithmetic_mean"},
                    }
                }
            ]
        },
        {"request_processors": [{"filter_query": {}}]},
        {
            "request_processors": [{"filter_query": {}}],
            "phase_results_processors": [
                {"normalization-processor": {"normalization": {"technique": "min_max"}}}
            ],
        },
    ],
)
def test_unsupported_pipelines_are_rejected(engine, body: dict) -> None:
    with pytest.raises(StoreRequestRejectedError):
        engine.put_search_pipeline("p", body)


def test_unsupported_space_type_is_rejected_at_create(engine) -> None:
    body = {"mappings": {"properties": {"v": _knn("l1")}}}
    with pytest.raises(StoreRequestRejectedError):
        engine.create_index("other", body)


# --------------------------------------------------------------------------- #
# Writes and administration
# --------------------------------------------------------------------------- #


def test_bulk_index_reports_rejected_documents_in_the_opensearch_shape(engine) -> None:
    response = engine.bulk_index(
        INDEX,
        [
            {"id": "d", "name": "Dock", "vec": [1.0, 0.0, 0.0]},
            {"id": "e", "rank": "high"},
            {"id": "f", "name": "Fine", "vec": [0.5, 0.5]},
        ],
    )
    assert response["errors"] is True
    assert [next(iter(item.values()))["_id"] for item in response["items"]] == [
        "d",
        "e",
    ]
    assert all(item["index"]["status"] == 400 for item in response["items"])
    assert set(_hits(engine, {"match_all": {}})) == {"a", "b", "c", "f"}


def test_bulk_index_replaces_the_whole_document(engine) -> None:
    engine.bulk_index(INDEX, [{"id": "a", "name": "Vendor two"}])
    (hit,) = _search(engine, {"ids": {"values": ["a"]}})["hits"]["hits"]
    assert hit["_source"] == {"id": "a", "name": "Vendor two"}
    assert "a" not in _hits(engine, {"knn": {"vec": {"vector": [1.0, 0.0], "k": 3}}})


def test_bulk_update_merges_fields_and_keeps_vectors(engine) -> None:
    response = engine.bulk_update(
        INDEX,
        [
            {"id": "a", "text_unit_ids": ["t9"], "attributes": {"region": {"zone": 1}}},
            {"id": "missing", "text_unit_ids": ["t9"]},
        ],
    )
    assert response == {"errors": False, "items": []}
    (hit,) = _search(engine, {"ids": {"values": ["a"]}})["hits"]["hits"]
    source = hit["_source"]
    assert source["text_unit_ids"] == ["t9"]
    assert source["description"] == "Vendor supplies widgets"
    assert source["attributes"] == {"region": {"code": "EU", "zone": 1}}
    assert source["vec"] == [1.0, 0.0]
    assert _ids(engine, {"ids": {"values": ["missing"]}}) == []


def test_list_vectors_skip_the_json_round_trip(engine, monkeypatch) -> None:
    # A 1024-float list through json.dumps/json.loads cost most of the
    # indexing time; the other values are still made JSON types.
    dumped: list[Any] = []
    dumps = engine_module._dumps

    def spy(value: Any) -> str:
        dumped.append(value)
        return dumps(value)

    monkeypatch.setattr(engine_module, "_dumps", spy)
    document = {
        "id": "n",
        "name": "Numpy",
        "rank": np.int64(4),
        "vec": [np.float32(0.6), np.float32(0.8)],
        "vec_l2": np.array([2.0, 0.0], dtype=np.float32),
    }
    assert engine.bulk_index(INDEX, [document]) == {"errors": False, "items": []}
    assert engine.bulk_update(INDEX, [{"id": "n", "vec": [0.0, 1.0]}]) == {
        "errors": False,
        "items": [],
    }
    assert not any(isinstance(v, dict) and "vec" in v for v in dumped)
    (hit,) = _search(engine, {"ids": {"values": ["n"]}})["hits"]["hits"]
    assert hit["_source"]["rank"] == 4
    assert hit["_source"]["vec"] == [0.0, 1.0]
    assert hit["_source"]["vec_l2"] == [2.0, 0.0]
    assert _hits(engine, {"knn": {"vec_l2": {"vector": [2.0, 0.0], "k": 1}}}) == {
        "n": pytest.approx(1.0)
    }


def test_bulk_delete_treats_missing_ids_as_deleted(engine) -> None:
    assert engine.bulk_delete(INDEX, ["a", "zz"]) == {"errors": False, "items": []}
    assert set(_hits(engine, {"match_all": {}})) == {"b", "c"}
    assert "a" not in _hits(engine, {"knn": {"vec": {"vector": [1.0, 0.0], "k": 3}}})


def test_create_index_exist_ok_and_alias_names(engine) -> None:
    engine.create_index(INDEX, {"mappings": {}})  # kept
    assert engine.get_index_mappings(INDEX)["properties"]["id"] == {"type": "keyword"}
    with pytest.raises(AWSServiceError, match="already exists"):
        engine.create_index(INDEX, {"mappings": {}}, exist_ok=False)
    with pytest.raises(StoreRequestRejectedError):
        engine.create_index(ALIAS, {"mappings": {}})


def test_alias_administration(engine) -> None:
    engine.create_index("things-2", MAPPING)
    engine.update_alias(ALIAS, "things-2", remove_pattern="things-*")
    assert engine.get_indices_by_alias(ALIAS) == ["things-2"]
    assert engine.get_index_name_by_alias(ALIAS) == "things-2"
    assert engine.get_aliases_by_index("things-*") == {
        "things-1": [],
        "things-2": [ALIAS],
    }
    assert engine.get_index_name_by_alias("nope") is None
    assert engine.get_aliases_by_index("nope-*") == {}
    with pytest.raises(NotFoundError):
        engine.update_alias(ALIAS, "missing-index")
    with pytest.raises(StoreRequestRejectedError):
        engine.update_alias(INDEX, "things-2")
    engine.delete_alias(["things-2"], [ALIAS])
    assert engine.get_indices_by_alias(ALIAS) == []
    with pytest.raises(NotFoundError):
        engine.delete_alias("things-2", ALIAS)


def test_delete_indices_cascades_and_ignores_missing(engine) -> None:
    engine.delete_indices(["things-*", "missing"])
    assert engine.get_aliases_by_index("*") == {}
    assert engine.get_indices_by_alias(ALIAS) == []
    with pytest.raises(NotFoundError):
        engine.search(index=INDEX, body={})
    # A new index of the same name starts empty (no stale cached copy).
    engine.create_index(INDEX, MAPPING)
    assert engine.search(index=INDEX, body={})["hits"]["hits"] == []


def test_meta_round_trip(engine) -> None:
    engine.put_index_meta(INDEX, {"embedding_dimension": 2, "x": True})
    mappings = engine.get_index_mappings(INDEX)
    assert mappings["_meta"] == {"embedding_dimension": 2, "x": True}
    assert "vec" in mappings["properties"]
    assert asyncio.run(engine.aget_mappings(ALIAS)) == {INDEX: mappings}
    assert engine.get_index_mappings(ALIAS) == {}  # as OpenSearchClient's lookup


def test_async_surface(engine) -> None:
    async def run() -> tuple:
        return (
            await engine.aindex_exists(ALIAS),
            await engine.aindex_exists(INDEX),
            await engine.aindex_exists("missing"),
            await engine.asearch(
                index=ALIAS, body={"query": {"ids": {"values": ["a"]}}}
            ),
        )

    alias, index, missing, response = asyncio.run(run())
    assert (alias, index, missing) == (True, True, False)
    assert response["hits"]["hits"][0]["_id"] == "a"
    asyncio.run(engine.aclose())


# --------------------------------------------------------------------------- #
# Cache, persistence and errors
# --------------------------------------------------------------------------- #


def test_generation_reload_across_two_engines(engine, tmp_path: Path) -> None:
    reader = SQLiteSearchEngine(tmp_path / "vector.sqlite3")
    try:
        assert set(_hits(reader, {"match_all": {}})) == {"a", "b", "c"}
        loaded = engine_module._INDEX_CACHE[(reader._cache_path, INDEX)]
        engine.bulk_index(INDEX, [{"id": "d", "name": "Dock"}])
        assert set(_hits(reader, {"match_all": {}})) == {"a", "b", "c", "d"}
        reloaded = engine_module._INDEX_CACHE[(reader._cache_path, INDEX)]
        assert reloaded.generation > loaded.generation
        # Unchanged: the cached copy is reused.
        _hits(reader, {"match_all": {}})
        assert engine_module._INDEX_CACHE[(reader._cache_path, INDEX)] is reloaded
    finally:
        reader.close()


class _CommitBefore:
    """A connection that lets ``write`` commit just before the first statement
    starting with ``prefix``: between two reads of one operation."""

    def __init__(self, real: sqlite3.Connection, prefix: str, write: Any) -> None:
        self.real, self.prefix, self.write, self.fired = real, prefix, write, False

    def execute(self, sql: str, *args: Any) -> Any:
        if sql.startswith(self.prefix) and not self.fired:
            self.fired = True
            self.write()
        return self.real.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)


def test_index_load_reads_one_snapshot(engine, tmp_path: Path) -> None:
    reader = SQLiteSearchEngine(tmp_path / "vector.sqlite3")
    try:
        engine_module._INDEX_CACHE.clear()
        reader._connection = _CommitBefore(  # type: ignore[assignment]
            reader.connection,
            "SELECT id, vec FROM vectors",
            lambda: engine.bulk_index(INDEX, [{"id": "d", "vec": [1.0, 0.0]}]),
        )
        knn = {"knn": {"vec": {"vector": [1.0, 0.0], "k": 5}}}
        # The load predates the write: its vectors are the documents' (a
        # load spanning the commit had a vector without a document: KeyError).
        assert set(_hits(reader, knn)) == {"a", "b", "c"}
        assert reader._connection.fired
        assert set(_hits(reader, knn)) == {"a", "b", "c", "d"}
    finally:
        reader.close()


@pytest.mark.parametrize(
    "prefix",
    [
        "SELECT settings FROM indices",  # the result window check
        "SELECT body FROM pipelines",
        "SELECT name, generation FROM indices",  # the index load
        "SELECT id, source FROM docs",  # an id lookup
    ],
)
def test_search_reads_one_snapshot(engine, tmp_path: Path, prefix: str) -> None:
    def blue_green() -> None:
        # Another engine re-indexes into a new index and deletes the old one.
        engine.create_index("things-2", MAPPING)
        engine.bulk_index("things-2", [{"id": "z", "vec": [1.0, 0.0]}])
        engine.update_alias(ALIAS, "things-2", remove_pattern="things-*")
        engine.delete_indices([INDEX])

    reader = SQLiteSearchEngine(tmp_path / "vector.sqlite3")
    try:
        engine_module._INDEX_CACHE.clear()
        reader._connection = _CommitBefore(  # type: ignore[assignment]
            reader.connection, prefix, blue_green
        )
        query = (
            {"ids": {"values": ["a", "z"]}}
            if prefix.startswith("SELECT id, source")
            else {"hybrid": {"queries": [{"match_all": {}}, {"match_all": {}}]}}
        )
        # The search resolved the alias before the swap: it reads the old
        # index whole (it read a deleted index's rows as None: TypeError).
        hits = _hits(reader, query, search_pipeline=PIPELINE)
        assert reader._connection.fired
        assert set(hits) == ({"a"} if "ids" in query else {"a", "b", "c"})
        assert set(_hits(reader, query, search_pipeline=PIPELINE)) == {"z"}
    finally:
        reader.close()


def test_vectors_without_documents_raise_a_store_error(engine, tmp_path: Path) -> None:
    raw = sqlite3.connect(tmp_path / "vector.sqlite3", isolation_level=None)
    try:
        raw.execute("DELETE FROM docs WHERE id = 'a'")
        raw.execute("UPDATE indices SET generation = generation + 1000")
    finally:
        raw.close()
    with pytest.raises(StoreConfigurationError, match="without a document"):
        _hits(engine, {"knn": {"vec": {"vector": [1.0, 0.0], "k": 5}}})


def test_indices_deleted_by_another_process_leave_the_cache(
    engine, tmp_path: Path
) -> None:
    engine.create_index("other-1", MAPPING)
    engine.bulk_index("other-1", DOCS[:1])
    _hits(engine, {"match_all": {}})
    engine.search(index="other-1", body={})
    assert (engine._cache_path, "other-1") in engine_module._INDEX_CACHE
    # Another process deletes it: this one's cache is not told.
    raw = sqlite3.connect(tmp_path / "vector.sqlite3", isolation_level=None)
    try:
        raw.execute("DELETE FROM indices WHERE name = 'other-1'")
    finally:
        raw.close()
    _hits(engine, {"match_all": {}})
    assert (engine._cache_path, "other-1") not in engine_module._INDEX_CACHE
    assert (engine._cache_path, INDEX) in engine_module._INDEX_CACHE


def test_data_persists_after_reopen(engine, tmp_path: Path) -> None:
    engine.close()
    engine_module._INDEX_CACHE.clear()
    reopened = SQLiteSearchEngine(tmp_path / "vector.sqlite3")
    try:
        assert set(_hits(reopened, {"match_all": {}})) == {"a", "b", "c"}
        hybrid = _search(
            reopened,
            {"hybrid": {"queries": [{"match_all": {}}, {"match_all": {}}]}},
            search_pipeline=PIPELINE,
        )
        assert len(hybrid["hits"]["hits"]) == 3
        assert reopened.get_index_name_by_alias(ALIAS) == INDEX
    finally:
        reopened.close()


def test_missing_index_is_opensearch_index_not_found(engine) -> None:
    for call in (
        lambda: engine.search(index="missing", body={}),
        lambda: engine.get_index_mappings("missing"),
        lambda: engine.put_index_meta("missing", {}),
        lambda: engine.bulk_index("missing", [{"id": "x"}]),
        lambda: asyncio.run(engine.aget_mappings("missing")),
        lambda: asyncio.run(engine.asearch(index="missing", body={})),
    ):
        with pytest.raises(NotFoundError) as excinfo:
            call()
        assert is_index_not_found(excinfo.value)
    # A pattern matching nothing is no error.
    assert engine.search(index="missing-*", body={})["hits"]["hits"] == []


def test_locked_database_raises_store_timeout(engine, tmp_path: Path) -> None:
    other = sqlite3.connect(tmp_path / "vector.sqlite3", isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        engine.timeout_seconds = 0.05
        engine.close()  # reopened with the shorter busy timeout
        with pytest.raises(StoreTimeoutError, match="locked"):
            engine.bulk_index(INDEX, [{"id": "x"}])
    finally:
        other.execute("ROLLBACK")
        other.close()


def test_corrupt_database_raises_store_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"this is not a database" * 100)
    engine = SQLiteSearchEngine(path)
    with pytest.raises(StoreConfigurationError):
        engine.get_indices_by_alias(ALIAS)


def test_unwritable_location_raises_store_configuration_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    engine = SQLiteSearchEngine(blocker / "vector.sqlite3")
    with pytest.raises(StoreConfigurationError):
        engine.create_index(INDEX, MAPPING)


def test_large_index_logs_a_warning(engine, monkeypatch, caplog) -> None:
    monkeypatch.setattr(engine_module, "LARGE_INDEX_DOCS", 2)
    engine_module._INDEX_CACHE.clear()
    with caplog.at_level(logging.WARNING):
        _hits(engine, {"match_all": {}})
    assert "loaded 3 documents" in caplog.text


def test_in_memory_engine(tmp_path: Path) -> None:
    engine = SQLiteSearchEngine(":memory:")
    engine.create_index(INDEX, MAPPING)
    engine.bulk_index(INDEX, DOCS[:1])
    assert engine.search(index=INDEX, body={})["hits"]["hits"][0]["_id"] == "a"
    engine.close()
    assert not any(key[0] == engine._cache_path for key in engine_module._INDEX_CACHE)


def test_in_memory_engines_never_share_cached_indices() -> None:
    # A collected engine's id() is reused by the next one: keyed by it, the
    # new engine was served the old one's cached index.
    for _ in range(20):
        old = SQLiteSearchEngine(":memory:")
        old.create_index(INDEX, MAPPING)
        old.bulk_index(INDEX, [{"id": "old", "name": "Vendor"}])
        assert [h["_id"] for h in old.search(index=INDEX, body={})["hits"]["hits"]] == [
            "old"
        ]
        old_key = old._cache_path
        del old  # not closed: collecting it frees its cached index
        gc.collect()
        assert not any(key[0] == old_key for key in engine_module._INDEX_CACHE)
        new = SQLiteSearchEngine(":memory:")
        new.create_index(INDEX, MAPPING)
        new.bulk_index(INDEX, [{"id": "new", "name": "Vendor"}])
        try:
            hits = new.search(index=INDEX, body={})["hits"]["hits"]
            assert [h["_id"] for h in hits] == ["new"]
        finally:
            new.close()


# --------------------------------------------------------------------------- #
# Through OpenSearchIndexer and OpenSearchRetriever
# --------------------------------------------------------------------------- #

_EMBEDDINGS = HashingEmbeddingFactory()


def _artifacts() -> dict[str, list[Any]]:
    return {
        "text_units": [
            TextUnit(
                id="t1",
                text="Vendor supplies widgets to Buyer under a purchase order.",
                attributes={
                    "filters": {"category": "contract"},
                    "category": "contract",
                },
            ),
            TextUnit(id="t2", text="공급업체는 매주 창고에서 위젯을 출하한다."),
        ],
        "entities": [
            Entity(
                id="e-vendor",
                name="Vendor",
                type="ORG",
                description="Vendor supplies widgets",
                text_unit_ids=["t1"],
                rank=3,
                attributes={"filters": {"category": "contract"}},
            ),
            Entity(
                id="e-buyer",
                name="Buyer",
                type="ORG",
                description="Buyer orders widgets",
                text_unit_ids=["t1"],
                rank=1,
            ),
        ],
        "relationships": [
            Relationship(
                id="r1",
                source_id="e-vendor",
                target_id="e-buyer",
                source_name="Vendor",
                target_name="Buyer",
                description="Vendor supplies widgets to Buyer",
                weight=1.0,
                text_unit_ids=["t1"],
            )
        ],
        "claims": [
            Claim(
                id="cl1",
                subject_id="e-vendor",
                subject_name="Vendor",
                object_name="Buyer",
                type="OBLIGATION",
                description="Vendor must deliver widgets",
                source_text="Vendor supplies widgets",
                text_unit_ids=["t1"],
            )
        ],
        "community_reports": [
            CommunityReport(
                id="cr1",
                community_id="c1",
                name="Widget supply",
                summary="Vendor supplies widgets to Buyer.",
                full_content="Widget supply\n\nVendor supplies widgets to Buyer.",
                rank=2.0,
                text_unit_ids=["t1"],
                attributes={"filters": {"category": "contract"}},
            )
        ],
    }


def _index_everything(indexer: OpenSearchIndexer) -> None:
    artifacts = _artifacts()
    assert indexer.initialize()
    for stats in (
        indexer.index_text_units(artifacts["text_units"]),
        indexer.index_entities(artifacts["entities"]),
        indexer.index_relationships(artifacts["relationships"]),
        indexer.index_claims(artifacts["claims"]),
        indexer.index_community_reports(artifacts["community_reports"]),
    ):
        assert stats.failed_items == 0, stats.errors


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("shapes") / "vector.sqlite3"
    config = Config()
    indexer = OpenSearchIndexer(
        config,
        embedding_factory=_EMBEDDINGS,
        opensearch_client=SQLiteSearchEngine(path),
    )
    _index_everything(indexer)
    engine = SQLiteSearchEngine(path)
    retriever = OpenSearchRetriever(config, engine, embedding_factory=_EMBEDDINGS)
    yield config, engine, retriever
    retriever.close()
    indexer.close()


_QUERIES = {
    "empty": SearchQuery(query=""),
    "star": SearchQuery(query="*"),
    "text": SearchQuery(query="Which widgets does Vendor supply?"),
    "dense": SearchQuery(query="창고 위젯"),
    "optional": SearchQuery(query="vendor", optional_keywords=["depot", "buyer"]),
    "long": SearchQuery(query=" ".join(f"word{i}" for i in range(400))),
}
_FILTERS: dict[str, dict[str, Any] | None] = {
    "none": None,
    "fetch_by_id": {"id": ["t1", "e-vendor", "r1", "cl1", "cr1", "missing"]},
    "term": {"type": "ORG"},
    "terms": {"text_unit_ids": ["t1"]},
    "range": {"rank": {"gte": 2}},
    "attr": {"attr_category": "contract"},
    "attributes": {"attributes.category": "contract"},
    "keyword_subfield": {"name.keyword": "Vendor"},
    "two": {"type": ["ORG", "OBLIGATION"], "id": ["e-vendor", "cl1", "t1"]},
}


def _hybrid_has_one_sub_query(body: dict[str, Any]) -> bool:
    hybrid = body["query"].get("hybrid")
    return hybrid is not None and len(hybrid["queries"]) == 1


@pytest.mark.parametrize("filters", list(_FILTERS), ids=list(_FILTERS))
@pytest.mark.parametrize("query", list(_QUERIES), ids=list(_QUERIES))
@pytest.mark.parametrize("search_type", list(SearchType), ids=lambda t: t.value)
def test_every_retriever_request_shape_runs_on_the_engine(
    stack, search_type: SearchType, query: str, filters: str
) -> None:
    config, engine, retriever = stack
    search_query = _QUERIES[query].model_copy(
        update={"search_type": search_type, "filters": _FILTERS[filters]}
    )
    vector = (
        _EMBEDDINGS.get_model("m").embed_query(search_query.query)
        if search_query.query and search_type != SearchType.LEXICAL
        else None
    )
    for prefix, fields in retriever._field_mappings.items():
        index_filters = retriever._filters_for_index(prefix, search_query.filters)
        index_query = search_query.model_copy(update={"filters": index_filters})
        body, params = retriever._build_search_request(
            index_query, search_type, fields["lexical"], fields["vector"], vector
        )
        alias = retriever._get_name(prefix, None)
        if _hybrid_has_one_sub_query(body):
            # A hybrid query of one sub-query against a two-weight pipeline:
            # OpenSearch rejects it ("number of weights ... must match"), and so
            # does the engine.
            with pytest.raises(StoreRequestRejectedError, match="number of weights"):
                engine.search(index=alias, body=body, **params)
            continue
        response = engine.search(index=alias, body=body, **params)
        hits = response["hits"]["hits"]
        assert len(hits) <= body["size"]
        for hit in hits:
            assert hit["_index"].startswith(alias)
            assert not any(k.endswith("_embedding") for k in hit["_source"])
        wanted = (index_filters or {}).get("id")
        if wanted:
            assert {hit["_id"] for hit in hits} <= set(wanted)


def test_retriever_parses_engine_hits(stack) -> None:
    config, _engine, retriever = stack
    for search_type in SearchType:
        results = asyncio.run(
            retriever.aretrieve(
                SearchQuery(query="Vendor supplies widgets", search_type=search_type)
            )
        )
        assert {"t1", "e-vendor"} <= {r.source for r in results}, search_type
        assert all(
            not any(k.endswith("_embedding") for k in r.metadata) for r in results
        )


def test_alias_swap_reaps_stale_indices(tmp_path: Path) -> None:
    config = Config()
    engine = SQLiteSearchEngine(tmp_path / "vector.sqlite3")
    indexer = OpenSearchIndexer(
        config, embedding_factory=_EMBEDDINGS, opensearch_client=engine
    )
    alias = indexer._get_name(config.indexing.opensearch.entities_index_prefix, None)
    # A sibling namespace whose index names also match ``<alias>-*``.
    sibling = f"{alias}-2-20250101000000"
    engine.create_index(sibling, {"mappings": {}})
    entities = _artifacts()["entities"]
    live = []
    for _ in range(3):
        assert indexer.index_entities(entities).failed_items == 0
        live.append(engine.get_index_name_by_alias(alias))
    assert len(set(live)) == 3
    indices = engine.get_aliases_by_index(f"{alias}-*")
    assert indices == {live[-1]: [alias], sibling: []}
    hits = engine.search(index=alias, body={"query": {"match_all": {}}})["hits"]["hits"]
    assert sorted(h["_id"] for h in hits) == ["e-buyer", "e-vendor"]
    assert indexer.clear([config.indexing.additional_suffix or "default"])
    assert engine.get_aliases_by_index(f"{alias}-*") == {sibling: []}
    indexer.close()


def test_check_index_settings_reads_the_engine_meta(tmp_path: Path) -> None:
    config = Config()
    path = tmp_path / "vector.sqlite3"
    indexer = OpenSearchIndexer(
        config,
        embedding_factory=_EMBEDDINGS,
        opensearch_client=SQLiteSearchEngine(path),
    )
    indexer.set_doc_status_tracking(True)
    assert indexer.index_entities(_artifacts()["entities"]).failed_items == 0
    indexer.check_index_settings()  # built for these settings

    other = OpenSearchIndexer(
        config,
        embedding_factory=HashingEmbeddingFactory(dimensions=32),
        opensearch_client=SQLiteSearchEngine(path),
    )
    with pytest.raises(ConfigurationError, match="dimension"):
        other.check_index_settings()

    # An index without recorded settings gets its dimension recorded.
    engine = indexer.opensearch_client
    index = engine.get_index_name_by_alias(
        indexer._get_name(config.indexing.opensearch.entities_index_prefix, None)
    )
    engine.put_index_meta(index, {})
    indexer.check_index_settings()
    assert engine.get_index_mappings(index)["_meta"] == {"embedding_dimension": 64}
    indexer.close()
    other.close()


def test_read_back_through_the_indexer(tmp_path: Path) -> None:
    indexer = OpenSearchIndexer(
        Config(),
        embedding_factory=_EMBEDDINGS,
        opensearch_client=SQLiteSearchEngine(tmp_path / "vector.sqlite3"),
    )
    _index_everything(indexer)
    assert indexer.read_entity_descriptions(["e-vendor", "nope"]) == {
        "e-vendor": "Vendor supplies widgets"
    }
    (claim,) = indexer.read_claims(["cl1"])
    assert claim.text_unit_ids == ["t1"]
    indexer.close()


def test_shapes_cover_every_search_type_and_filter_kind() -> None:
    # Guards the enumeration above against a new filter clause or search type.
    clause_kinds = {
        next(iter(clause))
        for value in _FILTERS.values()
        for clause in OpenSearchRetriever._build_filter_clauses(value)
    }
    assert clause_kinds == {"term", "terms", "range"}
