# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for OpenSearchRetriever query/parse logic.

Complements ``test_opensearch_clause_budget.py`` (batch-size budgeting). Here we
cover the query-body construction (lexical / vector / hybrid), filter-clause
construction, per-index field mappings, search-type support guard, and hit ->
RetrievalResult parsing — all pure, AWS-free logic. The retriever is built via
``__new__`` so its AWS-client ``__init__`` never runs; only the config-derived
attributes the methods touch are set.
"""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.retrieval.token_manager import SectionType
from aws_graphrag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from aws_graphrag.domain.models import Config, SearchQuery, SearchType

pytestmark = pytest.mark.unit


@pytest.fixture
def retriever(config: Config) -> OpenSearchRetriever:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_max_size", config.indexing.opensearch.max_query_size)
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    return inst


# --------------------------------------------------------------------------- #
# field mappings
# --------------------------------------------------------------------------- #


def test_field_mappings_cover_all_indices(retriever, config) -> None:
    o = config.indexing.opensearch
    mappings = retriever._field_mappings
    assert set(mappings) == {
        o.text_units_index_prefix,
        o.entities_index_prefix,
        o.relationships_index_prefix,
        o.claims_index_prefix,
        o.community_reports_index_prefix,
    }


def test_entities_mapping_lexical_and_vector_fields(retriever, config) -> None:
    o = config.indexing.opensearch
    m = retriever._field_mappings[o.entities_index_prefix]
    assert m["lexical"] == ["name", "description"]
    assert m["vector"] == ["name_embedding", "description_embedding"]


def test_text_units_lexical_includes_translated_field(retriever, config) -> None:
    o = config.indexing.opensearch
    lang = config.processing.translation.target_language.value
    m = retriever._field_mappings[o.text_units_index_prefix]
    assert m["lexical"] == ["text", f"translated_text_{lang}"]


# --------------------------------------------------------------------------- #
# _normalize_index_prefixes
# --------------------------------------------------------------------------- #


def test_normalize_index_prefixes_string(retriever) -> None:
    assert retriever._normalize_index_prefixes("graphrag-entities") == [
        "graphrag-entities"
    ]


def test_normalize_index_prefixes_none_returns_all(retriever) -> None:
    out = retriever._normalize_index_prefixes(None)
    assert set(out) == set(retriever._field_mappings.keys())


def test_normalize_index_prefixes_list_passthrough(retriever) -> None:
    assert retriever._normalize_index_prefixes(["a", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# _is_search_type_supported
# --------------------------------------------------------------------------- #


def test_search_type_supported_lexical_requires_lexical_fields(retriever) -> None:
    assert retriever._is_search_type_supported(SearchType.LEXICAL, ["name"], [])
    assert not retriever._is_search_type_supported(SearchType.LEXICAL, [], ["v"])


def test_search_type_supported_vector_requires_vector_fields(retriever) -> None:
    assert retriever._is_search_type_supported(SearchType.VECTOR, [], ["v"])
    assert not retriever._is_search_type_supported(SearchType.VECTOR, ["name"], [])


def test_search_type_supported_hybrid_needs_either(retriever) -> None:
    assert retriever._is_search_type_supported(SearchType.HYBRID, ["name"], [])
    assert retriever._is_search_type_supported(SearchType.HYBRID, [], ["v"])
    assert not retriever._is_search_type_supported(SearchType.HYBRID, [], [])


# --------------------------------------------------------------------------- #
# _build_filter_clauses
# --------------------------------------------------------------------------- #


def test_build_filter_clauses_none(retriever) -> None:
    assert retriever._build_filter_clauses(None) == []


def test_build_filter_clauses_term_terms_range(retriever) -> None:
    clauses = retriever._build_filter_clauses(
        {
            "type": "PERSON",  # scalar -> term
            "id": ["a", "b"],  # list -> terms
            "rank": {"gte": 5},  # dict -> range
        }
    )
    assert {"term": {"type": "PERSON"}} in clauses
    assert {"terms": {"id": ["a", "b"]}} in clauses
    assert {"range": {"rank": {"gte": 5}}} in clauses


# --------------------------------------------------------------------------- #
# _build_lexical_query
# --------------------------------------------------------------------------- #


def test_build_lexical_query_match_all_for_empty(retriever) -> None:
    q = SearchQuery(query="")
    assert retriever._build_lexical_query(q, ["name"]) == {"match_all": {}}


def test_build_lexical_query_star_is_match_all(retriever) -> None:
    q = SearchQuery(query="*")
    assert retriever._build_lexical_query(q, ["name"]) == {"match_all": {}}


def test_build_lexical_query_multi_match_with_fuzziness(retriever) -> None:
    q = SearchQuery(query="alice")
    body = retriever._build_lexical_query(q, ["name", "description"])
    assert body["multi_match"]["query"] == "alice"
    assert body["multi_match"]["fields"] == ["name", "description"]
    assert body["multi_match"]["fuzziness"] == "AUTO"


def test_build_lexical_query_optional_keywords_become_should(retriever) -> None:
    q = SearchQuery(query="alice", optional_keywords=["acme", "seattle"])
    body = retriever._build_lexical_query(q, ["name"])
    assert "bool" in body
    assert body["bool"]["must"][0]["multi_match"]["query"] == "alice"
    assert body["bool"]["should"][0]["multi_match"]["query"] == "acme seattle"


# --------------------------------------------------------------------------- #
# _build_vector_query
# --------------------------------------------------------------------------- #


def test_build_vector_query_single_field(retriever) -> None:
    body = retriever._build_vector_query([0.1, 0.2], 7, ["name_embedding"])
    assert body == {"knn": {"name_embedding": {"vector": [0.1, 0.2], "k": 7}}}


def test_build_vector_query_multiple_fields_should_bool(retriever) -> None:
    body = retriever._build_vector_query(
        [0.1], 5, ["name_embedding", "description_embedding"]
    )
    should = body["bool"]["should"]
    assert len(should) == 2
    assert should[0]["knn"]["name_embedding"]["k"] == 5
    assert should[1]["knn"]["description_embedding"]["vector"] == [0.1]


# --------------------------------------------------------------------------- #
# _build_hybrid_query
# --------------------------------------------------------------------------- #


def test_build_hybrid_query_combines_lexical_and_vector(retriever) -> None:
    q = SearchQuery(query="alice")
    body = retriever._build_hybrid_query(
        q, [0.1], ["name"], ["name_embedding"], 5, None
    )
    queries = body["hybrid"]["queries"]
    # One lexical clause + one vector clause.
    assert len(queries) == 2
    assert "multi_match" in queries[0]
    assert "knn" in queries[1]


def test_build_hybrid_query_empty_yields_match_none(retriever) -> None:
    q = SearchQuery(query="alice")
    # No lexical fields and no vector -> nothing to query.
    body = retriever._build_hybrid_query(q, None, [], [], 5, None)
    assert body == {"match_none": {}}


def test_build_hybrid_query_match_all_with_filters(retriever) -> None:
    q = SearchQuery(query="")  # lexical -> match_all
    filters = [{"term": {"type": "PERSON"}}]
    body = retriever._build_hybrid_query(q, None, ["name"], [], 5, filters)
    queries = body["hybrid"]["queries"]
    assert queries[0]["bool"]["filter"] == filters


# --------------------------------------------------------------------------- #
# _build_main_query / _build_search_request
# --------------------------------------------------------------------------- #


def test_build_main_query_lexical_wraps_filters(retriever) -> None:
    q = SearchQuery(query="alice", search_type=SearchType.LEXICAL)
    filters = [{"term": {"type": "PERSON"}}]
    body = retriever._build_main_query(
        q, SearchType.LEXICAL, ["name"], [], None, 5, filters
    )
    assert body["bool"]["filter"] == filters
    assert body["bool"]["must"][0]["multi_match"]["query"] == "alice"


def test_build_main_query_filter_only_when_no_query(retriever) -> None:
    q = SearchQuery(query="", search_type=SearchType.HYBRID)
    filters = [{"term": {"type": "PERSON"}}]
    body = retriever._build_main_query(
        q, SearchType.HYBRID, ["name"], ["v"], None, 5, filters
    )
    assert body == {"bool": {"filter": filters}}


def test_build_search_request_clamps_size_to_max(retriever) -> None:
    retriever_max = retriever._max_size
    q = SearchQuery(query="x", top_k=retriever_max * 10, retrieval_multiplier=1)
    body, params = retriever._build_search_request(
        q, SearchType.LEXICAL, ["name"], [], None
    )
    assert body["size"] == retriever_max


def test_build_search_request_hybrid_sets_pipeline_param(retriever, config) -> None:
    q = SearchQuery(query="x", search_type=SearchType.HYBRID)
    body, params = retriever._build_search_request(
        q, SearchType.HYBRID, ["name"], ["name_embedding"], [0.1]
    )
    assert (
        params["search_pipeline"]
        == config.indexing.opensearch.hybrid_search_pipeline_name
    )


def test_build_search_request_lexical_no_pipeline(retriever) -> None:
    q = SearchQuery(query="x", search_type=SearchType.LEXICAL)
    body, params = retriever._build_search_request(
        q, SearchType.LEXICAL, ["name"], [], None
    )
    assert params == {}


# --------------------------------------------------------------------------- #
# _parse_hit -> RetrievalResult + section typing
# --------------------------------------------------------------------------- #


def test_parse_hit_basic(retriever) -> None:
    hit = {
        "_index": "graphrag-entities-default-1",
        "_id": "doc1",
        "_score": 1.5,
        "_source": {"id": "e1", "name": "Alice", "description": "researcher"},
    }
    result = retriever._parse_hit(hit)
    assert result.source == "e1"
    assert result.score == 1.5
    assert result.retriever_type == SectionType.ENTITY.value
    assert result.metadata["_search_index"] == "graphrag-entities-default-1"
    assert "Title: Alice" in result.content
    assert "Description: researcher" in result.content


def test_parse_hit_falls_back_to_underscore_id(retriever) -> None:
    hit = {"_index": "graphrag-text-units-default", "_id": "x9", "_source": {}}
    result = retriever._parse_hit(hit)
    assert result.source == "x9"


def test_determine_section_type_per_index(retriever, config) -> None:
    o = config.indexing.opensearch
    assert (
        retriever._determine_section_type(f"{o.text_units_index_prefix}-default")
        == SectionType.TEXT
    )
    assert (
        retriever._determine_section_type(f"{o.entities_index_prefix}-default")
        == SectionType.ENTITY
    )
    assert (
        retriever._determine_section_type(f"{o.relationships_index_prefix}-default")
        == SectionType.RELATIONSHIP
    )
    assert (
        retriever._determine_section_type(f"{o.claims_index_prefix}-default")
        == SectionType.CLAIM
    )
    assert (
        retriever._determine_section_type(f"{o.community_reports_index_prefix}-default")
        == SectionType.COMMUNITY
    )
    assert retriever._determine_section_type("unknown-index") == SectionType.GENERAL


# --------------------------------------------------------------------------- #
# _extract_content
# --------------------------------------------------------------------------- #


def test_extract_content_dedups_parts(retriever) -> None:
    content = retriever._extract_content(
        {"name": "Alice", "description": "researcher", "summary": "a summary"}
    )
    assert "Title: Alice" in content
    assert "Description: researcher" in content
    assert "Summary: a summary" in content


def test_extract_content_translated_suppresses_raw_text(retriever, config) -> None:
    lang = config.processing.translation.target_language.value
    content = retriever._extract_content(
        {f"translated_text_{lang}": "translated", "text": "raw"}
    )
    # When a translation exists, the raw `text` is not also appended.
    assert "translated" in content
    assert "raw" not in content


def test_extract_content_uses_raw_text_when_no_translation(retriever) -> None:
    content = retriever._extract_content({"text": "raw body"})
    assert "raw body" in content
