# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for OpenSearchRetriever query/parse logic.

Covers the query-body construction (lexical / vector / hybrid), filter-clause
construction, per-index field mappings, search-type support guard, and hit ->
RetrievalResult parsing — all pure, AWS-free logic. The retriever is built via
``__new__`` so its AWS-client ``__init__`` never runs; only the config-derived
attributes the methods touch are set.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.domain.models import Config, SearchQuery, SearchType
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.shared.utils.scripts import is_dense_script_char

pytestmark = pytest.mark.unit


@pytest.fixture
def retriever(config: Config) -> OpenSearchRetriever:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_max_size", config.indexing.opensearch.max_query_size)
    object.__setattr__(inst, "_max_result_window", 10000)
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    object.__setattr__(
        inst, "_index_filter_fields", inst._initialize_index_filter_fields()
    )
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


def test_text_units_lexical_includes_additional_languages(config) -> None:
    translation = config.processing.translation
    translation.target_language = LanguageCode.EN
    translation.additional_target_languages = [LanguageCode.KO, LanguageCode.EN]
    retriever = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(retriever, "_config", config)
    object.__setattr__(retriever, "_opensearch_config", config.indexing.opensearch)
    m = retriever._initialize_field_mappings()[
        config.indexing.opensearch.text_units_index_prefix
    ]
    assert m["lexical"] == ["text", "translated_text_en", "translated_text_ko"]


# --------------------------------------------------------------------------- #
# _normalize_index_prefixes
# --------------------------------------------------------------------------- #


def test_normalize_index_prefixes_string(retriever) -> None:
    assert retriever._normalize_index_prefixes("graphrag-entities") == [
        "graphrag-entities"
    ]


def test_normalize_index_prefixes_none_skips_unbuilt_claims(retriever, config) -> None:
    # Default config: claim extraction OFF -> no claims index is ever built, so
    # the default sweep must not query it (it 404s on every query otherwise).
    assert config.processing.claim_extraction.enabled is False
    o = config.indexing.opensearch
    out = retriever._normalize_index_prefixes(None)
    assert o.claims_index_prefix not in out
    assert out == [
        o.text_units_index_prefix,
        o.entities_index_prefix,
        o.relationships_index_prefix,
        o.community_reports_index_prefix,
    ]


def test_normalize_index_prefixes_none_all_when_everything_built(
    retriever, config
) -> None:
    config.processing.claim_extraction.enabled = True
    out = retriever._normalize_index_prefixes(None)
    assert set(out) == set(retriever._field_mappings.keys())


def test_normalize_index_prefixes_none_follows_optional_index_flags(
    retriever, config
) -> None:
    config.processing.claim_extraction.enabled = True
    config.indexing.opensearch.build_relationship_vector_index = False
    config.graph.community_detection.enabled = False
    o = config.indexing.opensearch
    assert retriever._normalize_index_prefixes(None) == [
        o.text_units_index_prefix,
        o.entities_index_prefix,
        o.claims_index_prefix,
    ]


def test_explicit_prefixes_reach_unbuilt_optional_index(retriever, config) -> None:
    # Gating applies only to the default sweep; a pinned prefix is honoured.
    claims = config.indexing.opensearch.claims_index_prefix
    assert retriever._normalize_index_prefixes([claims]) == [claims]


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
    # Each fuzzy term rewrites into at most this many terms per field.
    assert body["multi_match"]["max_expansions"] == 10


def test_build_lexical_query_matches_a_long_question_exactly(retriever) -> None:
    # Fuzziness expands every term (up to 50 terms per field by default):
    # on a 100,000-term index a 15-word question failed the search with
    # too_many_nested_clauses. Past 10 words the query is matched exactly.
    ten = " ".join(f"word{i}" for i in range(10))
    body = retriever._build_lexical_query(SearchQuery(query=ten), ["name"])
    assert body["multi_match"]["fuzziness"] == "AUTO"
    body = retriever._build_lexical_query(SearchQuery(query=ten + " more"), ["name"])
    assert "fuzziness" not in body["multi_match"]
    assert "max_expansions" not in body["multi_match"]


@pytest.mark.parametrize("text", ["가나다상사", "青空海運", "カタカナ", "가나다 Corp"])
def test_build_lexical_query_no_fuzziness_for_dense_scripts(retriever, text) -> None:
    # One Hangul/Han/Kana character is a whole syllable or morpheme, so the
    # one edit AUTO allows on a 3-5 character term matches a different word
    # ("가나라상사" for "가나다상사").
    body = retriever._build_lexical_query(SearchQuery(query=text), ["name"])
    assert "fuzziness" not in body["multi_match"]


def test_build_lexical_query_optional_keywords_become_should(retriever) -> None:
    q = SearchQuery(query="alice", optional_keywords=["acme", "seattle"])
    body = retriever._build_lexical_query(q, ["name"])
    assert "bool" in body
    assert body["bool"]["must"][0]["multi_match"]["query"] == "alice"
    assert body["bool"]["should"][0]["multi_match"]["query"] == "acme seattle"


def _terms(text: str) -> int:
    dense = sum(map(is_dense_script_char, text))
    return dense or len(text.split())


@pytest.mark.parametrize(
    "text",
    [
        " ".join(f"word{i}" for i in range(1100)),
        "倉" * 1100,
        " ".join(["가나다"] * 600),
    ],
    ids=["latin", "han", "hangul"],
)
def test_build_lexical_query_caps_a_long_query(retriever, text) -> None:
    # A multi_match makes one clause per term per field; 1,100 words failed
    # the search with maxClauseCount (400) and 600 with
    # too_many_nested_clauses (500), each answered from no context. The
    # question takes up to 150 terms and the optional keywords the rest of
    # 200: a 150-character Han question with 150 Han keywords failed with
    # too_many_nested_clauses on five fields.
    q = SearchQuery(query=text, optional_keywords=text.split()[:300])
    body = retriever._build_lexical_query(q, ["name"])
    must = body["bool"]["must"][0]["multi_match"]["query"]
    should = body["bool"]["should"][0]["multi_match"]["query"]
    assert text.startswith(must) and text.startswith(should)
    assert (_terms(must), _terms(should)) == (150, 50)


@pytest.mark.parametrize(
    ("fields", "expected"), [(5, (150, 50)), (6, (150, 16)), (7, (142, 0))]
)
def test_build_lexical_query_scales_the_budget_by_field_count(
    retriever, fields: int, expected: tuple[int, int]
) -> None:
    # Text units search one more field per translated language: with a target
    # and four more languages (6-7 fields) 200 terms made 1,200-1,400 clauses,
    # and the search failed with too_many_nested_clauses (500), leaving the
    # text section empty. The 1,000 clauses are spread over the fields.
    text = " ".join(f"word{i}" for i in range(300))
    q = SearchQuery(query=text, optional_keywords=text.split())
    body = retriever._build_lexical_query(q, [f"f{i}" for i in range(fields)])
    if "bool" in body:
        must = body["bool"]["must"][0]["multi_match"]["query"]
        should = body["bool"]["should"][0]["multi_match"]["query"]
    else:
        must, should = body["multi_match"]["query"], ""
    assert (_terms(must), _terms(should) if should else 0) == expected
    assert (_terms(must) + (_terms(should) if should else 0)) * fields <= 1000


def test_build_lexical_query_fuzzy_terms_count_their_expansions(retriever) -> None:
    keywords = [f"kw{i}" for i in range(300)]
    q = SearchQuery(query="alice", optional_keywords=keywords)
    body = retriever._build_lexical_query(q, ["name"])
    should = body["bool"]["should"][0]["multi_match"]["query"]
    # One fuzzy term expands to up to 10, leaving 190 of the 200.
    assert should == " ".join(keywords[:190])


def test_build_lexical_query_keeps_a_dense_name_in_a_long_question(
    retriever,
) -> None:
    # One Hangul name made the whole question count as dense text, cut to
    # 150 characters: the name at the end of a 170-character question was lost.
    text = (
        "Considering the supply agreement, the depot storage fees and the "
        "courier schedule described in the contract, which Korean agent "
        "represents Vendor, and where is 가나다상사 located?"
    )
    assert len(text) > 150
    body = retriever._build_lexical_query(SearchQuery(query=text), ["name"])
    assert body["multi_match"]["query"] == text


def test_build_lexical_query_bounds_mixed_text(retriever) -> None:
    # Words and dense-script characters count toward one term cap.
    words = " ".join(f"word{i}" for i in range(60))
    text = words + " " + "倉" * 300
    capped = retriever._build_lexical_query(SearchQuery(query=text), ["name"])[
        "multi_match"
    ]["query"]
    assert capped == words + " " + "倉" * 90


def test_build_lexical_query_keeps_a_normal_query(retriever) -> None:
    text = " ".join(f"word{i}" for i in range(100))
    body = retriever._build_lexical_query(SearchQuery(query=text), ["name"])
    assert body["multi_match"]["query"] == text


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


def test_build_vector_query_attaches_filters_to_knn(retriever) -> None:
    # A VECTOR query that also carries filters must pre-filter the kNN clause,
    # else it returns the global top-k and leaks docs outside the filter.
    filters = [{"terms": {"community_id": ["c1", "c2"]}}]
    body = retriever._build_vector_query([0.1], 7, ["name_embedding"], filters)
    knn = body["knn"]["name_embedding"]
    assert knn["filter"] == {"bool": {"filter": filters}}


def test_build_vector_query_no_filter_when_none(retriever) -> None:
    body = retriever._build_vector_query([0.1], 7, ["name_embedding"])
    assert "filter" not in body["knn"]["name_embedding"]


# --------------------------------------------------------------------------- #
# _build_search_bodies
# --------------------------------------------------------------------------- #


def _bodies(retriever, query: SearchQuery, vector=None, *, size=5, sort=None):
    return retriever._build_search_bodies(
        query,
        query.search_type,
        ["name"],
        ["name_embedding"],
        vector,
        size=size,
        sort=sort,
    )


def test_hybrid_is_a_lexical_and_a_knn_search(retriever) -> None:
    lexical, knn = _bodies(retriever, SearchQuery(query="alice"), [0.1])
    assert "multi_match" in lexical["query"]
    assert knn["query"]["knn"]["name_embedding"]["vector"] == [0.1]
    # No spare hit: each search returns exactly the hits it is fused over.
    assert lexical["size"] == knn["size"] == 5


def test_hybrid_filters_reach_both_searches(retriever) -> None:
    q = SearchQuery(query="risk", filters={"community_id": ["c1"]})
    lexical, knn = _bodies(retriever, q, [0.1])
    filters = [{"terms": {"community_id": ["c1"]}}]
    assert lexical["query"]["bool"]["filter"] == filters
    assert knn["query"]["knn"]["name_embedding"]["filter"] == {
        "bool": {"filter": filters}
    }


def test_vector_search_is_one_knn_search(retriever) -> None:
    (body,) = _bodies(
        retriever, SearchQuery(query="alice", search_type=SearchType.VECTOR), [0.1]
    )
    assert "knn" in body["query"]


def test_lexical_search_wraps_filters(retriever) -> None:
    q = SearchQuery(
        query="alice", search_type=SearchType.LEXICAL, filters={"type": "PERSON"}
    )
    (body,) = _bodies(retriever, q)
    assert body["query"]["bool"]["filter"] == [{"term": {"type": "PERSON"}}]
    assert body["query"]["bool"]["must"][0]["multi_match"]["query"] == "alice"


@pytest.mark.parametrize("search_type", [SearchType.HYBRID, SearchType.VECTOR])
def test_text_less_search_is_a_filter_browse(retriever, search_type) -> None:
    q = SearchQuery(query="", search_type=search_type, filters={"type": "PERSON"})
    assert [b["query"] for b in _bodies(retriever, q)] == [
        {"bool": {"filter": [{"term": {"type": "PERSON"}}]}}
    ]
    q = SearchQuery(query="", search_type=search_type)
    assert [b["query"] for b in _bodies(retriever, q)] == [{"match_all": {}}]


def test_hit_count_clamps_size_to_max(retriever) -> None:
    retriever_max = retriever._max_size
    q = SearchQuery(query="x", top_k=retriever_max * 10, retrieval_multiplier=1)
    assert retriever._hit_count(q) == retriever_max


def test_hit_count_floors_size_to_terms_filter(retriever) -> None:
    # Fetch-by-id: ids passed as a terms filter with top_k=len(ids). size must
    # not be clamped to max_query_size (default 100) or ids beyond 100 are
    # silently dropped. Floor size at the terms-list length.
    n = retriever._max_size + 50  # 150 > 100
    ids = [f"id-{i}" for i in range(n)]
    q = SearchQuery(query="", top_k=n, retrieval_multiplier=1, filters={"id": ids})
    assert retriever._hit_count(q) == n  # not clamped to _max_size


def test_hit_count_other_list_filters_do_not_raise_size(retriever) -> None:
    # Only the id list names the hits to fetch; a 3,000-value type filter
    # raised size (and the kNN k of every vector field) to 3,000.
    types = [f"t{i}" for i in range(3000)]
    q = SearchQuery(query="x", top_k=10, filters={"type": types})
    assert retriever._hit_count(q) == 10


def test_hit_count_terms_floor_bounded_by_result_window(retriever) -> None:
    # The floor never exceeds max_result_window (the cluster hits ceiling).
    huge = retriever._max_result_window + 5000
    ids = [f"id-{i}" for i in range(huge)]
    q = SearchQuery(query="", top_k=huge, retrieval_multiplier=1, filters={"id": ids})
    assert retriever._hit_count(q) == retriever._max_result_window


class _Client:
    """Answers every search with ``hits`` per index, best first.

    A kNN search gets ``knn_hits`` of that index instead, when given.
    """

    def __init__(
        self,
        hits: dict[str, list[tuple[str, float]]],
        knn_hits: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        self.hits = hits
        self.knn_hits = knn_hits or {}
        self.requests: list[dict] = []

    @property
    def bodies(self) -> list[dict]:
        return [request["body"] for request in self.requests]

    async def asearch(self, **request) -> dict:
        self.requests.append(request)
        index, body = request["index"], request["body"]
        source = self.knn_hits if "knn" in str(body["query"]) else self.hits
        found = next(
            (v for k, v in source.items() if index.startswith(k)),
            next(v for k, v in self.hits.items() if index.startswith(k)),
        )
        return {
            "hits": {
                "hits": [
                    {"_index": f"{index}-1", "_id": i, "_score": s, "_source": {}}
                    for i, s in found[: body["size"]]
                ]
            }
        }


async def test_hybrid_fuses_the_two_searches_client_side(
    retriever, config, mocker
) -> None:
    # OpenSearch 2.13 normalized a hybrid query's lexical sub-query over a
    # document outside its hits; the retriever now fuses two plain searches,
    # each normalized over exactly its own hits, with no search pipeline.
    prefix = config.indexing.opensearch.text_units_index_prefix
    config.search.hybrid.lexical_weight = 0.25
    config.search.hybrid.vector_weight = 0.75
    client = _Client(
        {prefix: [("a", 9.0), ("b", 5.0), ("c", 1.0)]},
        {prefix: [("c", 0.9), ("a", 0.5), ("d", 0.1)]},
    )
    object.__setattr__(retriever, "_opensearch_client", client)
    mocker.patch.object(
        OpenSearchRetriever, "_get_query_vector", mocker.AsyncMock(return_value=[0.1])
    )
    mocker.patch.object(OpenSearchRetriever, "_check_embedding_settings")
    results = await retriever.aretrieve(
        SearchQuery(query="x", top_k=3, index_prefixes=[prefix])
    )
    assert len(client.requests) == 2
    assert all(set(r) == {"index", "body"} for r in client.requests)
    # lexical a=1, b=0.5, c=min; vector c=1, a=0.5, d=min (MIN_SCORE 0.001).
    expected = {
        "c": 0.25 * 0.001 + 0.75 * 1.0,
        "a": 0.25 * 1.0 + 0.75 * 0.5,
        "b": 0.25 * 0.5,
    }
    assert [r.source for r in results] == ["c", "a", "b"]
    for r in results:
        assert r.score == pytest.approx(expected[r.source])


async def test_index_sweep_takes_each_index_in_turn(retriever, config, mocker) -> None:
    # Raw scores of two indices do not compare (here the entity index's sum of
    # two kNN scores tops every text hit); the sweep keeps each index's best.
    o = config.indexing.opensearch
    client = _Client(
        {
            o.text_units_index_prefix: [("t0", 0.9), ("t1", 0.8), ("t2", 0.7)],
            o.entities_index_prefix: [("e0", 1.9), ("e1", 1.8), ("e2", 1.7)],
        }
    )
    object.__setattr__(retriever, "_opensearch_client", client)
    mocker.patch.object(
        OpenSearchRetriever, "_get_query_vector", mocker.AsyncMock(return_value=None)
    )
    results = await retriever.aretrieve(
        SearchQuery(
            query="x",
            search_type=SearchType.LEXICAL,
            top_k=4,
            index_prefixes=[o.text_units_index_prefix, o.entities_index_prefix],
        )
    )
    assert [r.source for r in results] == ["t0", "e0", "t1", "e1"]


def test_text_less_search_sorts_by_importance(retriever, config) -> None:
    o = config.indexing.opensearch
    browse = SearchQuery(
        query="", search_type=SearchType.LEXICAL, filters={"source_id": ["e1"]}
    )
    assert retriever._importance_sort(
        o.relationships_index_prefix, browse, SearchType.LEXICAL
    ) == [
        {"rank": {"order": "desc"}},
        {"weight": {"order": "desc"}},
        {"id": {"order": "asc"}},
    ]
    (body,) = _bodies(retriever, browse, sort=[{"rank": {"order": "desc"}}])
    assert body["sort"] == [{"rank": {"order": "desc"}}]
    assert body["track_scores"] is True
    # Not for a search with text, a fetch by id or an index without rank.
    by_id = SearchQuery(query="", filters={"id": ["r1"]})
    for prefix, query in (
        (o.relationships_index_prefix, SearchQuery(query="supplies")),
        (o.relationships_index_prefix, by_id),
        (o.text_units_index_prefix, browse),
    ):
        assert retriever._importance_sort(prefix, query, SearchType.LEXICAL) is None


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


def test_searches_exclude_embedding_fields(retriever) -> None:
    # Vectors are only needed server-side for kNN scoring; the response must not
    # ship them back (community reports alone carry three per hit).
    for body in _bodies(retriever, SearchQuery(query="x"), [0.1]):
        assert body["_source"] == {"excludes": ["*_embedding"]}


def test_parse_hit_drops_embedding_fields_from_metadata(retriever) -> None:
    # Defensive: even if the cluster/client returns vectors anyway, they never
    # reach RetrievalResult.metadata (and from there the reported sources).
    hit = {
        "_index": "graphrag-community-reports-default-1",
        "_id": "r1",
        "_score": 0.7,
        "_source": {
            "id": "r1",
            "name": "Vendor network",
            "summary": "Vendor supplies Buyer.",
            "full_content": "Vendor supplies Buyer with parts.",
            "name_embedding": [0.1, 0.2],
            "summary_embedding": [0.3, 0.4],
            "full_content_embedding": [0.5, 0.6],
            "document_ids": ["doc-1"],
        },
    }
    result = retriever._parse_hit(hit)
    assert not [k for k in result.metadata if k.endswith("_embedding")]
    assert result.metadata["document_ids"] == ["doc-1"]
    assert result.metadata["summary"] == "Vendor supplies Buyer."


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


def test_extract_content_report_is_its_full_content_once(retriever) -> None:
    # full_content already opens with the title and summary; repeating them
    # spent the budget twice and DRIFT's character cut kept only the repeats.
    full = "# Vendor network\n\nVendor supplies Buyer.\n\n## Parts\nWeekly."
    content = retriever._extract_content(
        {
            "name": "Vendor network",
            "summary": "Vendor supplies Buyer.",
            "full_content": full,
        }
    )
    assert content == full


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


# Relationship hits must name their endpoints, mirroring
# upstream LightRAG's `{"entity1", "entity2", "description"}` relations_context.
# Without the names a relationship line is unresolvable text, and widening the
# relationship stream to upstream's top_k=40 then multiplies noise instead of
# signal.


def test_extract_content_names_relationship_endpoints(retriever) -> None:
    content = retriever._extract_content(
        {
            "source_name": "erik erikson",
            "target_name": "joan erikson",
            "type": "spouse",
            "description": "Marriage relationship",
        }
    )
    assert "Relationship: erik erikson -[spouse]-> joan erikson" in content
    assert "Description: Marriage relationship" in content


def test_extract_content_relationship_without_a_type_still_names_endpoints(
    retriever,
) -> None:
    content = retriever._extract_content(
        {"source_name": "a", "target_name": "b", "description": "d"}
    )
    assert "Relationship: a -> b" in content


def test_extract_content_relationship_with_one_missing_endpoint(retriever) -> None:
    # A half-populated edge is still more useful named than anonymous.
    content = retriever._extract_content({"source_name": "a", "description": "d"})
    assert "Relationship: a -> unknown" in content


def test_extract_content_non_relationship_hit_gets_no_relationship_line(
    retriever,
) -> None:
    content = retriever._extract_content({"name": "Alice", "description": "researcher"})
    assert "Relationship:" not in content


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), 1.0]


async def test_query_embeddings_are_cached_per_text(retriever) -> None:
    # Local search embeds the same entity-focus text for several indices; it
    # must hit the embedding model once.
    from collections import OrderedDict

    from unified_kg_rag.domain.models import SearchType

    embedder = _CountingEmbedder()
    object.__setattr__(retriever, "_embedding_model", embedder)
    object.__setattr__(retriever, "_query_embedding_cache", OrderedDict())
    object.__setattr__(retriever, "_pending_embeddings", {})

    fields = ["text_embedding"]
    first = await retriever._get_query_vector("Vendor Buyer", SearchType.HYBRID, fields)
    again = await retriever._get_query_vector("Vendor Buyer", SearchType.VECTOR, fields)
    other = await retriever._get_query_vector("Invoice", SearchType.HYBRID, fields)

    assert first == again
    assert other != first
    assert embedder.calls == ["Vendor Buyer", "Invoice"]


async def test_concurrent_sub_queries_share_one_embedding_call(retriever) -> None:
    # Sub-queries of one search run concurrently: all but the first used to
    # miss the cache (filled only after the await) and embed again.
    import asyncio
    from collections import OrderedDict

    from unified_kg_rag.domain.models import SearchType

    class _SlowEmbedder(_CountingEmbedder):
        async def aembed_query(self, text: str) -> list[float]:
            await asyncio.sleep(0.01)
            return await super().aembed_query(text)

    embedder = _SlowEmbedder()
    object.__setattr__(retriever, "_embedding_model", embedder)
    object.__setattr__(retriever, "_query_embedding_cache", OrderedDict())
    object.__setattr__(retriever, "_pending_embeddings", {})

    fields = ["text_embedding"]
    vectors = await asyncio.gather(
        *(
            retriever._get_query_vector("Vendor", SearchType.HYBRID, fields)
            for _ in range(4)
        )
    )

    assert embedder.calls == ["Vendor"]
    assert all(v == vectors[0] for v in vectors)
    assert retriever._pending_embeddings == {}


async def test_failed_embedding_with_every_waiter_cancelled_is_not_logged(
    retriever,
) -> None:
    # The shared call is shielded, so cancelling its only waiter leaves it
    # running; its failure then had no one to retrieve it and asyncio logged
    # "Task exception was never retrieved" when the task was collected.
    import asyncio
    import gc
    from collections import OrderedDict

    from unified_kg_rag.domain.models import SearchType

    started = asyncio.Event()

    class _FailingEmbedder:
        async def aembed_query(self, text: str) -> list[float]:
            started.set()
            await asyncio.sleep(0.01)
            raise RuntimeError("embedding failed")

    object.__setattr__(retriever, "_embedding_model", _FailingEmbedder())
    object.__setattr__(retriever, "_query_embedding_cache", OrderedDict())
    object.__setattr__(retriever, "_pending_embeddings", {})
    loop = asyncio.get_running_loop()
    reported: list[dict] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        waiter = asyncio.ensure_future(
            retriever._get_query_vector("Vendor", SearchType.VECTOR, ["v"])
        )
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        while retriever._pending_embeddings:
            await asyncio.sleep(0.005)
        del waiter
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)

    assert reported == []


async def test_query_embedding_cache_is_bounded(retriever, monkeypatch) -> None:
    from collections import OrderedDict

    from unified_kg_rag.domain.models import SearchType

    embedder = _CountingEmbedder()
    object.__setattr__(retriever, "_embedding_model", embedder)
    object.__setattr__(retriever, "_query_embedding_cache", OrderedDict())
    object.__setattr__(retriever, "_pending_embeddings", {})
    monkeypatch.setattr(OpenSearchRetriever, "QUERY_EMBEDDING_CACHE_SIZE", 2)

    fields = ["text_embedding"]
    for text in ["a", "b", "a", "c"]:  # "a" refreshed, so "b" is evicted
        await retriever._get_query_vector(text, SearchType.VECTOR, fields)
    await retriever._get_query_vector("a", SearchType.VECTOR, fields)
    await retriever._get_query_vector("b", SearchType.VECTOR, fields)

    assert embedder.calls == ["a", "b", "c", "b"]
    assert len(retriever._query_embedding_cache) == 2


@pytest.mark.parametrize("key", ["max_result_window", "index.max_result_window"])
def test_index_setting_reads_either_spelling(key) -> None:
    from unified_kg_rag.adapters.retrievers.opensearch_retriever import _index_setting

    assert _index_setting({key: "500"}, "max_result_window", 10000) == 500
    assert _index_setting({}, "max_result_window", 10000) == 10000
