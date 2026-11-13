# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document citations (``search.citations``, docs/design.md §7.1), AWS-free:
the markers in every answer language and in streamed chunks, the numbered
context, the citations of every section type, the lineage read under the
caller's filters and suffix, and the guardrail and abstention edges."""

from __future__ import annotations

import ast
import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.retrieval.token_manager import (
    EMPTY_CONTEXT_PLACEHOLDER,
    TokenManager,
)
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval import citations
from unified_kg_rag.application.retrieval.rag_chain import (
    NO_CONTEXT_ANSWER,
    GraphRAGChain,
    ProcessedQuery,
    RAGInput,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.prompts import retrieval as retrieval_prompts
from unified_kg_rag.domain.prompts.retrieval import CITATION_INSTRUCTIONS
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.citations import (
    CitationMarkerScanner,
    cited_markers,
    neutralize_markers,
    strip_markers,
)
from unified_kg_rag.ports.guardrail import GuardAction, GuardSource, GuardVerdict

pytestmark = pytest.mark.unit


# --- markers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "markers"),
    [
        ("Depot stores the gadgets [1].", [1]),
        ("Depot stores them [1, 3] and Vendor ships [2].", [1, 3, 2]),
        ("Both [2][1] say so; [2] again.", [2, 1]),
        ("A range [1-3] and [5–6].", [1, 2, 3, 5, 6]),
        # Korean, Japanese and Chinese answers with ASCII markers.
        ("창고는 공급사의 부품을 보관합니다 [1].", [1]),
        ("倉庫は供給者の部品を保管する[2]。", [2]),
        ("仓库为供应商保管零件[1][2]。", [1, 2]),
        # Full-width and lenticular brackets, full-width digits, ideographic
        # comma (CJK output).
        ("倉庫は部品を保管する［１］。", [1]),
        ("仓库保管零件【3】。", [3]),
        ("창고 [1、2]", [1, 2]),
        # Not markers.
        ("See [note] and [] and [a1].", []),
        ("A reversed range [3-1] names its two ends.", [3, 1]),
        ("A long range [1-500] is not every number in it.", [1, 500]),
        # A bracketed year is text, as strip_markers reads it, in every
        # language.
        ("In [2020] the vendor shipped [2].", [2]),
        ("2020년 [2020] 공급사가 출하했다 [1].", [1]),
        ("［２０２０］年に出荷した【1】。", [1]),
    ],
)
def test_markers_are_read_in_every_answer_language(
    answer: str, markers: list[int]
) -> None:
    assert cited_markers(answer) == markers


def test_markers_are_stripped_from_the_words_of_an_answer() -> None:
    assert strip_markers("Depot stores them [1, 2].") == "Depot stores them."
    assert strip_markers("倉庫［１］【2】は保管する") == "倉庫は保管する"
    # A bracketed year is text.
    assert strip_markers("In [2020] Depot [3] opened.") == "In [2020] Depot opened."


@pytest.mark.parametrize(
    "text",
    [
        "In [2020] Depot [3] opened [1000].",
        "2020년 [2020] 창고 [12] 개설 [999].",
        "［２０２０］年 倉庫【7】",
    ],
)
def test_the_markers_read_are_the_markers_stripped(text: str) -> None:
    # A marker counted as a citation is one the answer's words lose, and a
    # bracketed number kept as text is never a citation ("[2020]" was an
    # unknown citation marker).
    stripped = strip_markers(text)
    assert cited_markers(stripped) == []
    assert cited_markers(text) and all(len(str(n)) <= 3 for n in cited_markers(text))


def test_only_the_contexts_markers_count() -> None:
    assert cited_markers("In [2020] the vendor [2] shipped.", valid={1, 2}) == [2]


def _scanned(text: str, cuts: list[int], valid: set[int] | None = None) -> list[int]:
    scanner = CitationMarkerScanner(valid)
    found: list[int] = []
    edges = [0, *sorted(set(cuts)), len(text)]
    for start, end in zip(edges, edges[1:], strict=False):
        found.extend(scanner.feed(text[start:end]))
    found.extend(scanner.finish())
    return found


@pytest.mark.parametrize(
    "text",
    [
        "Depot [12] stores [1, 3] gadgets [2][4].",
        "창고는 [1] 부품을 [1-2] 보관합니다【3】.",
        "倉庫［１２］は部品を保管する[2、3]。 [unclosed",
        "[",
        "[1",
    ],
)
def test_the_scanner_reads_the_markers_of_the_whole_text_for_any_chunking(
    text: str,
) -> None:
    whole = cited_markers(text)
    for cut in range(len(text) + 1):
        assert list(dict.fromkeys(_scanned(text, [cut]))) == whole, cut
    single = CitationMarkerScanner()
    for char in text:
        single.feed(char)
    single.finish()
    assert single.markers == whole


@given(
    st.lists(
        st.sampled_from(["Depot ", "[", "1", "2", "]", ", ", "-", "【", "】", "창고 "]),
        max_size=30,
    ),
    st.lists(st.integers(min_value=0, max_value=60), max_size=6),
)
def test_scanning_chunks_equals_reading_the_joined_answer(
    parts: list[str], cuts: list[int]
) -> None:
    text = "".join(parts)
    assert list(dict.fromkeys(_scanned(text, cuts))) == cited_markers(text)


# --- the numbered context --------------------------------------------------


def _token_manager(config: Config) -> TokenManager:
    manager = TokenManager(config, providers=None)
    manager.count_tokens = lambda text: len((text or "").split())  # type: ignore[method-assign]
    manager.count_tokens_many = lambda texts: [  # type: ignore[method-assign]
        len((t or "").split()) for t in texts
    ]
    return manager


def _results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content="Vendor: a supplier.",
            score=0.9,
            source="e-1",
            retriever_type="entity",
            metadata={"name": "Vendor", "text_unit_ids": ["tu-2", "tu-9"]},
        ),
        RetrievalResult(
            content="Depot stores widgets for Vendor.",
            score=0.8,
            source="tu-1",
            retriever_type="text",
            metadata={"id": "tu-1", "document_ids": ["doc-1"]},
        ),
        RetrievalResult(
            content="Key points: Depot stores widgets.",
            score=1.0,
            source="synthesized_key_points",
            retriever_type="general",
            metadata={"synthesized": True},
        ),
        RetrievalResult(
            content="Vendor ships to Buyer.",
            score=0.7,
            source="tu-2",
            retriever_type="text",
            metadata={"id": "tu-2", "document_ids": ["doc-2"]},
        ),
    ]


def test_a_numbered_context_numbers_its_evidence_in_rendered_order(
    config: Config,
) -> None:
    manager = _token_manager(config)
    optimized = manager.optimize_context(_results(), "q", number_sections=True)
    context = manager.build_context_string(optimized)
    # Passages first (their block comes first), then the entity; the
    # synthesized key points carry no marker and no id.
    assert "[1] Depot stores widgets for Vendor." in context
    assert "[2] Vendor ships to Buyer." in context
    assert "[3] Vendor: a supplier." in context
    assert "Key points: Depot stores widgets." in context
    assert "[synthesized_key_points]" not in context
    assert "[tu-1]" not in context
    markers = {s.source_id: s.marker for s in optimized.sections}
    assert markers == {
        "tu-1": 1,
        "tu-2": 2,
        "e-1": 3,
        "synthesized_key_points": None,
    }
    # Each section is counted as rendered (at the widest marker), so the
    # context never exceeds what was packed.
    for line, section in zip(
        manager.ranked_section_lines(optimized),
        sorted(optimized.sections, key=lambda s: s.priority, reverse=True),
        strict=True,
    ):
        assert len(line.split()) <= section.token_count, line


def test_no_section_is_numbered_past_the_widest_marker(config: Config) -> None:
    # Regression (rv5): sections were counted at a three-digit marker and
    # numbered without a cap, so a 1000th section rendered "[1000]", wider
    # than counted and a number no answer marker is read as.
    from unified_kg_rag.adapters.retrieval.token_manager import _MARKER_PLACEHOLDER

    manager = _token_manager(config)
    results = [
        RetrievalResult(
            content=f"Passage {i}.",
            score=1.0 - i / 2000,
            source=f"tu-{i}",
            retriever_type="text",
            metadata={"id": f"tu-{i}"},
        )
        for i in range(_MARKER_PLACEHOLDER + 5)
    ]
    optimized = manager.optimize_context(
        results, "q", max_tokens=10**6, number_sections=True
    )
    markers = [s.marker for s in optimized.sections]
    assert len(markers) == len(results)
    assert max(m for m in markers if m is not None) == _MARKER_PLACEHOLDER
    assert markers.count(None) == 5
    context = manager.build_context_string(optimized)
    assert f"[{_MARKER_PLACEHOLDER}] " in context
    assert f"[{_MARKER_PLACEHOLDER + 1}]" not in context


def test_a_synthesized_section_naming_its_evidence_is_numbered(
    config: Config,
) -> None:
    # Global search's key points and DRIFT's intermediate answers carry the
    # lineage of what they were written from (citable): the answer can cite
    # them. A synthesized section without one stays unnumbered.
    manager = _token_manager(config)
    results = _results()
    results.append(
        RetrievalResult(
            content="Key point from the community reports (relevance 80/100): "
            "Vendor ships widgets.",
            score=0.8,
            source="synthesized_key_points_1",
            retriever_type="general",
            metadata={
                "synthesized": True,
                "citable": True,
                "text_unit_ids": ["tu-7"],
                "document_ids": ["doc-7"],
            },
        )
    )
    optimized = manager.optimize_context(results, "q", number_sections=True)
    context = manager.build_context_string(optimized)
    markers = {s.source_id: s.marker for s in optimized.sections}
    assert markers["synthesized_key_points"] is None
    assert markers["synthesized_key_points_1"] == 4
    assert "[4] Key point from the community reports" in context
    unnumbered = manager.optimize_context(results, "q")
    assert all(s.marker is None for s in unnumbered.sections)


def test_bracketed_numbers_in_retrieved_text_are_not_citation_markers(
    config: Config,
) -> None:
    # Regression (rv23): a footnoted passage ("audited in [2] and [3]")
    # rendered its own brackets next to the real markers, so an answer that
    # copied the phrase cited unrelated sections 2 and 3. In a numbered
    # context a marker-shaped span of retrieved text reads as "(2)", in every
    # bracket form an answer's marker is read in.
    manager = _token_manager(config)
    results = [
        RetrievalResult(
            content="Acme shipped late, as audited in [2] and [3].",
            score=1.0,
            source="tu-1",
            retriever_type="text",
            metadata={"id": "tu-1"},
        ),
        RetrievalResult(
            content="倉庫は監査済み［１］【2、3】。In [2020] it opened.",
            score=0.9,
            source="tu-2",
            retriever_type="text",
            metadata={"id": "tu-2"},
        ),
        RetrievalResult(
            content="Vendor ships to Buyer.",
            score=0.8,
            source="tu-3",
            retriever_type="text",
            metadata={"id": "tu-3"},
        ),
    ]
    optimized = manager.optimize_context(results, "q", number_sections=True)
    context = manager.build_context_string(optimized)
    assert "[1] Acme shipped late, as audited in (2) and (3)." in context
    assert "[2] 倉庫は監査済み（１）（2、3）。In [2020] it opened." in context
    # Only the context's own markers remain readable as citations.
    assert cited_markers(context) == [1, 2, 3]
    copied = "Acme shipped late, as audited in (2) and (3) [1]."
    assert cited_markers(copied, valid={1, 2, 3}) == [1]
    # The section is counted as rendered.
    for section in optimized.sections:
        assert "[" not in section.content.replace("[2020]", "")
    # An unnumbered context keeps the text as written.
    plain = manager.build_context_string(manager.optimize_context(results, "q"))
    assert "audited in [2] and [3]." in plain


def test_marker_shaped_spans_are_neutralized() -> None:
    assert neutralize_markers("see [2], [1-3] and [4, 5]") == (
        "see (2), (1-3) and (4, 5)"
    )
    assert neutralize_markers("［１］【2】") == "（１）（2）"
    # Not markers: a year, a word, empty brackets.
    assert neutralize_markers("[2020] [note] []") == "[2020] [note] []"


def test_an_unnumbered_context_is_as_before(config: Config) -> None:
    manager = _token_manager(config)
    optimized = manager.optimize_context(_results(), "q")
    context = manager.build_context_string(optimized)
    assert "[tu-1] Depot stores widgets for Vendor." in context
    assert "\nVendor: a supplier." in context
    assert all(s.marker is None for s in optimized.sections)


# --- citations of each section type ------------------------------------------


def _entry(metadata: dict[str, Any], source: str) -> dict[str, Any]:
    return {"source": source, "metadata": metadata}


def test_a_passage_cites_its_unit_document_file_and_pages() -> None:
    citation = citations.citation_from_entry(
        4,
        _entry(
            {
                "section_type": "text",
                "chunk_id": "tu-1",
                "id": "tu-1",
                "document_ids": ["doc-1"],
                "attributes": {
                    "file_name": "계약서.pdf",
                    "file_path": "/srv/corpus/legal/계약서.pdf",
                    "relative_path": "legal/계약서.pdf",
                    "page_start": 3,
                    "page_end": 4,
                },
            },
            "tu-1",
        ),
    )
    assert (citation.marker, citation.chunk_id, citation.text_unit_ids) == (
        4,
        "tu-1",
        ["tu-1"],
    )
    assert (citation.document_id, citation.file_name, citation.file_path) == (
        "doc-1",
        "계약서.pdf",
        "legal/계약서.pdf",
    )
    assert (citation.page, citation.page_end) == (3, 4)
    assert citation.title == "계약서.pdf"


@pytest.mark.parametrize(
    "path",
    ["/srv/corpus/a.pdf", "../a.pdf", "a/../../b.pdf", "C:\\corpus\\a.pdf", "s3://b/a"],
)
def test_a_citation_never_names_an_absolute_or_climbing_path(path: str) -> None:
    assert source_fields.safe_relative_path(path) is None


def test_a_citation_names_a_relative_path_with_forward_slashes() -> None:
    assert source_fields.safe_relative_path("legal\\2024\\a.pdf") == "legal/2024/a.pdf"
    assert source_fields.safe_relative_path("./a.pdf") == "a.pdf"


def test_a_seed_only_fact_cites_its_seed_graph_and_iri() -> None:
    citation = citations.citation_from_entry(
        1,
        _entry(
            {
                "section_type": "entity",
                "name": "Depot",
                "text_unit_ids": ["sg-1"],
                "provenance": {
                    "kind": "seed_graph",
                    "seed_graphs": ["warehouses"],
                    "iri": "urn:depot",
                },
            },
            "e-1",
        ),
    )
    assert citation.provenance == {
        "kind": "seed_graph",
        "seed_graphs": ["warehouses"],
        "iri": "urn:depot",
    }
    assert citation.iri == "urn:depot"
    assert citation.title == "Depot"
    assert citation.document_id is None


async def test_graph_citations_resolve_their_lineage_in_context_first() -> None:
    passage = citations.citation_from_entry(
        1,
        _entry(
            {
                "section_type": "text",
                "id": "tu-1",
                "chunk_id": "tu-1",
                "document_ids": ["doc-1"],
                "attributes": {"file_name": "a.txt", "relative_path": "a.txt"},
            },
            "tu-1",
        ),
    )
    relationship = citations.citation_from_entry(
        2,
        _entry(
            {
                "section_type": "relationship",
                "source_name": "Vendor",
                "target_name": "Buyer",
                "text_unit_ids": ["tu-7", "sg-3", "tu-8", "tu-9", "tu-6", "tu-1"],
            },
            "r-1",
        ),
    )
    asked: list[list[str]] = []
    stored = {
        "tu-6": ("doc-3", "c.txt"),
        "tu-7": ("doc-2", "b.txt"),
        "tu-8": ("doc-2", "b.txt"),
        # tu-9 is filtered out (or gone): never named.
    }

    async def fetch(ids: list[str]) -> list[dict[str, Any]]:
        asked.append(ids)
        return [
            {
                "id": i,
                "document_ids": [stored[i][0]],
                "attributes": {"file_name": stored[i][1], "page_start": 2},
            }
            for i in ids
            if i in stored
        ]

    await citations.aresolve_documents([passage, relationship], fetch, limit=4)
    # The context's own unit first, seed units never; up to the limit of
    # units the fetch returns (tu-9 is not, so tu-6 is read next).
    assert asked == [["tu-7", "tu-8", "tu-9"], ["tu-6"]]
    assert relationship.title == "Vendor -> Buyer"
    assert [d.document_id for d in relationship.documents] == [
        "doc-2",
        "doc-1",
        "doc-3",
    ]
    assert relationship.documents[0].text_unit_ids == ["tu-7", "tu-8"]
    assert (relationship.file_name, relationship.page) == ("b.txt", 2)
    assert relationship.documents[1].file_path == "a.txt"


async def test_the_lineage_cap_counts_the_units_the_filters_let_through() -> None:
    # Regression (rv5): the first lineage_units units were fetched and the
    # rest never read, so a filter excluding those named no document though
    # later units pass. The units are read in growing pages until each
    # citation has the limit or its lineage is read.
    report = citations.citation_from_entry(
        1,
        _entry(
            {
                "section_type": "community",
                "text_unit_ids": [f"u{i}" for i in range(40)],
            },
            "c-1",
        ),
    )
    allowed = {"u12": "계약서.ko.txt", "u13": "メモ.ja.txt", "u30": "plan.txt"}
    asked: list[list[str]] = []

    async def fetch(ids: list[str]) -> list[dict[str, Any]]:
        asked.append(ids)
        return [
            {"id": i, "document_ids": [f"doc-{i}"], "attributes": {"file_name": n}}
            for i, n in allowed.items()
            if i in ids
        ]

    await citations.aresolve_documents([report], fetch, limit=2)
    assert [d.file_name for d in report.documents] == ["계약서.ko.txt", "メモ.ja.txt"]
    # Pages of 2, 4, 8 (what it lacks, doubled each time): each unit once.
    assert [len(ids) for ids in asked] == [2, 4, 8]
    assert [u for ids in asked for u in ids] == [f"u{i}" for i in range(14)]


async def test_the_lineage_reads_stop_after_a_bounded_number_of_pages() -> None:
    # Regression (rv6): the pages went on until the lineage was read, so a
    # filter excluding most of a hub's units read all of them.
    report = citations.citation_from_entry(
        1,
        _entry(
            {
                "section_type": "community",
                "text_unit_ids": [f"u{i}" for i in range(500)],
            },
            "c-1",
        ),
    )
    asked: list[list[str]] = []

    async def fetch(ids: list[str]) -> list[dict[str, Any]]:
        asked.append(ids)
        return []

    await citations.aresolve_documents([report], fetch, limit=3)
    assert len(asked) == citations.MAX_LINEAGE_PAGES
    assert sum(len(ids) for ids in asked) == 3 * (2**citations.MAX_LINEAGE_PAGES - 1)
    assert report.documents == []


def test_a_report_cites_the_units_its_findings_cite() -> None:
    # A report's text_unit_ids are its community's membership; the strategy
    # sets the units of the records its findings cite beside them.
    report = citations.citation_from_entry(
        1,
        _entry(
            {
                "section_type": "community",
                "text_unit_ids": ["m-1", "m-2"],
                "document_ids": ["doc-m"],
                "findings_text_unit_ids": ["f-1"],
                "findings_document_ids": ["doc-f"],
            },
            "c-1",
        ),
    )
    legacy = citations.citation_from_entry(
        2, _entry({"section_type": "community", "text_unit_ids": ["m-1"]}, "c-2")
    )
    assert report.text_unit_ids == ["f-1"]
    assert legacy.text_unit_ids == ["m-1"]


async def test_a_fully_known_lineage_needs_no_fetch() -> None:
    passage = citations.citation_from_entry(
        1, _entry({"section_type": "text", "id": "tu-1", "document_ids": ["d"]}, "tu-1")
    )
    entity = citations.citation_from_entry(
        2, _entry({"section_type": "entity", "text_unit_ids": ["tu-1"]}, "e")
    )

    async def fetch(ids: list[str]) -> list[dict[str, Any]]:
        raise AssertionError("no fetch expected")

    await citations.aresolve_documents([passage, entity], fetch, limit=10)
    assert entity.document_id == "d"


async def test_the_lineage_work_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (rv27 perf 1): the lineages and each page's plan scanned a
    # hub entity's thousands of units on the event loop, 16-39 ms of every
    # local and mix query on the 13-hub corpus.
    loop_thread = threading.get_ident()
    threads: list[int] = []
    lineage, attach = citations._lineage, citations._attach

    def recording_lineage(*args: Any) -> list[str]:
        threads.append(threading.get_ident())
        return lineage(*args)

    def recording_attach(*args: Any) -> None:
        threads.append(threading.get_ident())
        attach(*args)

    monkeypatch.setattr(citations, "_lineage", recording_lineage)
    monkeypatch.setattr(citations, "_attach", recording_attach)
    plan = citations._Pages.page

    def recording_page(self: Any, scale: int) -> list[str]:
        threads.append(threading.get_ident())
        return plan(self, scale)

    monkeypatch.setattr(citations._Pages, "page", recording_page)
    entity = citations.citation_from_entry(
        1,
        _entry(
            {"section_type": "entity", "text_unit_ids": [f"u{i}" for i in range(9)]},
            "e",
        ),
    )

    async def fetch(ids: list[str]) -> list[dict[str, Any]]:
        return [{"id": "u5", "document_ids": ["d-5"]}] if "u5" in ids else []

    await citations.aresolve_documents([entity], fetch, limit=2)
    assert entity.document_id == "d-5"
    # The lineages, four pages (the last one empty) and the attach.
    assert len(threads) == 6 and loop_thread not in threads


def test_unknown_markers_are_never_citations() -> None:
    cited = [
        citations.citation_from_entry(
            n, _entry({"section_type": "text", "id": f"tu-{n}"}, f"tu-{n}")
        )
        for n in (1, 2)
    ]
    marked, unknown = citations.mark_cited(cited, "Depot [2] ships [7] [2].")
    assert [c.cited for c in marked] == [False, True]
    assert unknown == [7]


# --- the chain -----------------------------------------------------------------


class _Store:
    """A document retriever: a fixed hit list for searches, the stored text
    units for a fetch by id (honouring a ``team`` filter, and ``top_k`` as
    the port allows: the OpenSearch retriever floors an id fetch at its id
    count, another may not)."""

    UNITS = {
        "tu-1": ("doc-1", "contract.txt", "a"),
        "tu-2": ("doc-2", "logistics.pdf", "a"),
        "tu-3": ("doc-3", "secret.txt", "b"),
    }

    def __init__(self, hits: list[RetrievalResult]) -> None:
        self.hits = hits
        self.fetches: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        filters = query.filters or {}
        if "id" not in filters:
            return [r.model_copy(deep=True) for r in self.hits]
        self.fetches.append(query)
        team = filters.get("team")
        return [
            RetrievalResult(
                content="",
                score=1.0,
                source=unit,
                retriever_type="text",
                metadata={
                    "id": unit,
                    "document_ids": [document],
                    "attributes": {
                        "file_name": name,
                        "relative_path": f"corpus/{name}",
                        "page_start": 5,
                        "page_end": 6,
                    },
                },
            )
            for unit, (document, name, unit_team) in self.UNITS.items()
            if unit in filters["id"] and (team is None or team == unit_team)
        ][: query.top_k]


def _hits() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content="Depot stores widgets for Vendor.",
            score=0.9,
            source="tu-1",
            retriever_type="text",
            metadata={
                "id": "tu-1",
                "document_ids": ["doc-1"],
                "attributes": {"file_name": "contract.txt"},
            },
        ),
        RetrievalResult(
            content="Vendor: a supplier of widgets.",
            score=0.8,
            source="e-1",
            retriever_type="entity",
            metadata={
                "id": "e-1",
                "name": "Vendor",
                "text_unit_ids": ["tu-2", "tu-3"],
            },
        ),
    ]


class _Guard:
    def __init__(self, action: GuardAction = GuardAction.NONE) -> None:
        self.action = action

    async def acheck(self, text: str, source: GuardSource, **kwargs) -> GuardVerdict:
        if source is GuardSource.OUTPUT and self.action is GuardAction.BLOCKED:
            return GuardVerdict(action=GuardAction.BLOCKED, text="Refused.")
        if source is GuardSource.OUTPUT and self.action is GuardAction.MASKED:
            return GuardVerdict(action=GuardAction.MASKED, text="Masked [1].")
        return GuardVerdict(action=GuardAction.NONE, text=text)


def _chain(
    config: Config,
    answer: str = "Depot stores widgets [1]; Vendor supplies them [2] [9].",
    *,
    hits: list[RetrievalResult] | None = None,
    guard: _Guard | None = None,
    on_answer: Callable[[], None] | None = None,
) -> tuple[GraphRAGChain, _Store, list[tuple[str, str]]]:
    config.providers.rerank.kind = "none"
    prompts: list[tuple[str, str]] = []
    store = _Store(_hits() if hits is None else hits)

    def respond(system: str, human: str) -> str:
        if system.startswith("[{"):
            # Content blocks (a prompt-cache marker): the fake joins their repr.
            system = "".join(b["text"] for b in ast.literal_eval(system))
        prompts.append((system, human))
        if on_answer is not None:
            on_answer()
        return answer

    chain = GraphRAGChain(
        config=config,
        model_factory=ScriptedLLMFactory(respond),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: store,
            RetrieverRole.GRAPH: lambda: store,
        },
        guard=guard,
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())  # type: ignore[method-assign]
    return chain, store, prompts


def _query(**fields: Any) -> RAGInput:
    return RAGInput(
        **{
            "query": "Who stores the widgets?",
            "search_strategy": SearchStrategy.SIMPLE,
            "enable_query_processing": False,
            **fields,
        }
    )


async def test_the_answer_cites_numbered_sections_and_their_documents(
    config: Config,
) -> None:
    chain, store, prompts = _chain(config)
    out = await chain.ainvoke(_query(suffix="tenant-a"))
    system, human = prompts[-1]
    assert CITATION_INSTRUCTIONS in system
    assert "[1] Depot stores widgets for Vendor." in human
    assert "[2] Vendor: a supplier of widgets." in human
    passage, entity = out.citations
    assert (passage.marker, passage.cited, passage.file_name) == (
        1,
        True,
        "contract.txt",
    )
    assert (entity.marker, entity.cited, entity.title) == (2, True, "Vendor")
    # The entity resolves through its lineage, in the query's namespace.
    assert [d.file_name for d in entity.documents] == ["logistics.pdf", "secret.txt"]
    assert (entity.file_path, entity.page, entity.page_end) == (
        "corpus/logistics.pdf",
        5,
        6,
    )
    (fetch,) = store.fetches
    assert fetch.suffix == "tenant-a"
    assert fetch.source_fields == ["id", "document_ids", "attributes"]
    # [9] was never in the context: not a citation.
    assert out.metadata["unknown_citation_markers"] == [9]
    # Sources are unchanged.
    assert [s["source"] for s in out.sources] == ["tu-1", "e-1"]


@pytest.mark.parametrize("stream", [False, True])
async def test_each_sections_source_is_built_once_a_query(
    config: Config, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    # perf (rv24): the citations and the sources each built every section's
    # source entry, twice a query over the widest contexts.
    chain, _, _ = _chain(config)
    built: list[str] = []
    section_source = GraphRAGChain._section_source

    def counting(section: Any, results: Any) -> Any:
        built.append(str(section.source_id))
        return section_source(section, results)

    monkeypatch.setattr(GraphRAGChain, "_section_source", staticmethod(counting))
    if stream:
        chunks = [chunk async for chunk in chain.astream(_query())]
        assert chunks
    else:
        out = await chain.ainvoke(_query())
        assert [s["source"] for s in out.sources] == ["tu-1", "e-1"]
        assert [c.marker for c in out.citations] == [1, 2]
    assert sorted(built) == ["e-1", "tu-1"]


# --- citations per strategy ------------------------------------------------

_DEFAULT_OFF = {SearchStrategy.GLOBAL, SearchStrategy.DRIFT}


def test_global_and_drift_answers_do_not_cite_by_default(config: Config) -> None:
    # 2026-11-08: on 50 sensemaking questions the uncited global and DRIFT
    # answers won comprehensiveness (0.68 and 0.75); every other strategy
    # (structured included) keeps citations on.
    assert config.search.citations.enabled is True
    for strategy in SearchStrategy:
        if strategy is SearchStrategy.AUTO:
            continue
        assert citations.citations_enabled(config, strategy) is (
            strategy not in _DEFAULT_OFF
        ), strategy


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        # An entry turns a strategy on; the map merges over the default, so
        # naming global leaves DRIFT off.
        ({"strategies": {"global": True}}, {"global": True, "drift": False}),
        (
            {"strategies": {"global": True, "drift": True}},
            {"global": True, "drift": True},
        ),
        # ... or off, and wins over enabled either way.
        ({"strategies": {"local": False}}, {"local": False, "mix": True}),
        ({"enabled": False}, {"local": False, "global": False, "structured": False}),
        (
            {"enabled": False, "strategies": {"mix": True}},
            {"mix": True, "local": False, "global": False},
        ),
    ],
)
def test_an_explicit_setting_decides_a_strategys_citations(
    settings: dict[str, Any], expected: dict[str, bool]
) -> None:
    config = Config.model_validate({"search": {"citations": settings}})
    for name, cites in expected.items():
        assert citations.citations_enabled(config, name) is cites, name


@pytest.mark.parametrize(
    ("base", "strategies", "cites"),
    [
        ("mix", {}, True),
        ("mix", {"mix": False}, False),
        ("local", {"local": False}, False),
        ("local", {"mix": False}, True),
        # Named itself, iterative is its own.
        ("mix", {"mix": False, "iterative": True}, True),
        ("mix", {"iterative": False}, False),
    ],
)
def test_iterative_follows_its_base_unless_it_is_named(
    base: str, strategies: dict[str, bool], cites: bool
) -> None:
    config = Config.model_validate(
        {
            "search": {
                "iterative_search": {"base": base},
                "citations": {"strategies": strategies},
            }
        }
    )
    assert citations.citations_enabled(config, SearchStrategy.ITERATIVE) is cites


def test_auto_is_no_key_of_the_per_strategy_map() -> None:
    with pytest.raises(ValueError, match="auto resolves"):
        Config.model_validate({"search": {"citations": {"strategies": {"auto": True}}}})


def test_an_unknown_or_missing_strategy_reads_the_switch(config: Config) -> None:
    # A result saved without a strategy, or with one this release lacks.
    assert citations.citations_enabled(config, None) is True
    assert citations.citations_enabled(config, "retired") is True
    config.search.citations.enabled = False
    assert citations.citations_enabled(config, None) is False


def _state(strategy: SearchStrategy) -> dict[str, Any]:
    return {
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        "search_results": SearchResult(
            query=SearchQuery(query="q"),
            results=_results(),
            total_results=4,
            search_strategy=strategy.value,
            processing_time=0.0,
        ),
        "resolved_strategy": strategy,
    }


@pytest.mark.parametrize(
    ("strategy", "strategies", "numbered"),
    [
        (SearchStrategy.GLOBAL, {}, False),
        (SearchStrategy.DRIFT, {}, False),
        (SearchStrategy.LOCAL, {}, True),
        (SearchStrategy.MIX, {}, True),
        (SearchStrategy.STRUCTURED, {}, True),
        (SearchStrategy.ITERATIVE, {}, True),
        (SearchStrategy.ITERATIVE, {"mix": False}, False),
        (SearchStrategy.GLOBAL, {"global": True}, True),
        (SearchStrategy.LOCAL, {"local": False}, False),
    ],
)
def test_the_answer_context_is_numbered_for_a_strategy_that_cites(
    config: Config,
    strategy: SearchStrategy,
    strategies: dict[str, bool],
    numbered: bool,
) -> None:
    config.search.citations.strategies = {
        **config.search.citations.strategies,
        **{SearchStrategy(k): v for k, v in strategies.items()},
    }
    chain, _store, _prompts = _chain(config)
    state = _state(strategy)
    optimized = chain._context_optimization_step(state)
    assert optimized.numbered is numbered
    assert any(s.marker is not None for s in optimized.sections) is numbered
    state["optimized_context"] = optimized
    instructions = chain._answer_instructions(state)
    assert (CITATION_INSTRUCTIONS in instructions) is numbered


@pytest.mark.parametrize("strategy", [SearchStrategy.GLOBAL, SearchStrategy.DRIFT])
def test_global_and_drift_turned_on_get_the_plain_citation_instructions(
    config: Config, strategy: SearchStrategy
) -> None:
    # The "cite where possible" suffix is gone: it gained no
    # comprehensiveness and cost global answers directness (x4 e4), and it
    # departed from MS GraphRAG's reduce prompt.
    config.search.citations.strategies = {strategy: True}
    chain, _store, _prompts = _chain(config)
    state = _state(strategy)
    state["optimized_context"] = chain._context_optimization_step(state)
    assert chain._answer_instructions(state) == CITATION_INSTRUCTIONS
    assert not hasattr(retrieval_prompts, "SYNTHESIS_CITATION_INSTRUCTIONS")
    for name in dir(retrieval_prompts):
        value = getattr(retrieval_prompts, name)
        if isinstance(value, str):
            assert "cite where possible" not in value.casefold(), name
    # Without a numbered context there is nothing to cite.
    config.search.citations.strategies = {strategy: False}
    state["optimized_context"] = chain._context_optimization_step(state)
    assert chain._answer_instructions(state) == ""


async def test_a_strategy_whose_citations_are_off_answers_without_them(
    config: Config,
) -> None:
    config.search.citations.strategies = {SearchStrategy.SIMPLE: False}
    chain, store, prompts = _chain(config, answer="Depot stores widgets [1].")
    out = await chain.ainvoke(_query())
    system, human = prompts[-1]
    assert CITATION_INSTRUCTIONS not in system
    assert "[tu-1] Depot stores widgets for Vendor." in human
    assert out.citations == []
    assert store.fetches == []
    # A bracketed number the model wrote anyway is no unknown marker.
    assert "unknown_citation_markers" not in out.metadata


async def test_auto_follows_the_strategy_it_routes_to(config: Config) -> None:
    # One candidate: no router call, the query runs simple.
    config.search.auto_routable_strategies = [SearchStrategy.SIMPLE]
    config.search.citations.strategies = {SearchStrategy.SIMPLE: False}
    chain, _store, prompts = _chain(config)
    out = await chain.ainvoke(_query(search_strategy=SearchStrategy.AUTO))
    assert out.metadata["search_strategy"] == "simple"
    assert CITATION_INSTRUCTIONS not in prompts[-1][0]
    assert out.citations == []
    config.search.citations.strategies = {SearchStrategy.SIMPLE: True}
    chain, _store, prompts = _chain(config)
    out = await chain.ainvoke(_query(search_strategy=SearchStrategy.AUTO))
    assert CITATION_INSTRUCTIONS in prompts[-1][0]
    assert [c.marker for c in out.citations] == [1, 2]


_KEY_POINT_ANSWERS = {
    "en": "Depot stores widgets [1]; Vendor ships them [3].",
    "ko": "창고가 부품을 보관하고 [1] 공급사가 출하합니다 [3].",
    "ja": "倉庫が部品を保管し［１］、供給者が出荷する【3】。",
}


@pytest.mark.parametrize("language", list(_KEY_POINT_ANSWERS))
async def test_an_answer_citing_a_key_point_names_its_reports_documents(
    config: Config, language: str
) -> None:
    # Regression: the key points had no marker, so a global answer cited
    # nothing; now a point's citation names the documents of the reports it
    # credits, read from their lineage, even with the reports themselves
    # out of the context. It is still never a source.
    point = RetrievalResult(
        content="Key point from the community reports (relevance 90/100): "
        "Vendor ships the widgets.",
        score=0.9,
        source="synthesized_key_points_1",
        retriever_type="general",
        metadata={
            "synthesized": True,
            "citable": True,
            "report_ids": ["c-1"],
            "text_unit_ids": ["tu-2"],
            "document_ids": ["doc-2"],
        },
    )
    chain, store, prompts = _chain(
        config, _KEY_POINT_ANSWERS[language], hits=[*_hits(), point]
    )
    out = await chain.ainvoke(_query())
    assert "[3] Key point from the community reports" in prompts[-1][1]
    cited = {c.marker: c for c in out.citations}
    assert cited[3].cited is True
    assert cited[3].source_id == "synthesized_key_points_1"
    assert cited[3].text_unit_ids == ["tu-2"]
    assert [d.file_name for d in cited[3].documents] == ["logistics.pdf"]
    assert "unknown_citation_markers" not in out.metadata
    assert "synthesized_key_points_1" not in [s["source"] for s in out.sources]


async def test_a_source_names_the_document_its_text_unit_stores(
    config: Config,
) -> None:
    # Regression: a text unit stores its document id in its attributes
    # (attributes.document_id); the reported source named none.
    passage = RetrievalResult(
        content="倉庫は部品を保管する。",
        score=0.9,
        source="tu-1",
        retriever_type="text",
        metadata={"id": "tu-1", "attributes": {"document_id": "doc-ja"}},
    )
    chain, _, _ = _chain(config, "倉庫が保管する［１］。", hits=[passage])
    out = await chain.ainvoke(_query())
    (source,) = out.sources
    assert source["metadata"]["document_ids"] == ["doc-ja"]
    assert out.citations[0].document_id == "doc-ja"


async def test_a_filtered_query_never_names_a_document_its_filters_exclude(
    config: Config,
) -> None:
    chain, store, _ = _chain(config)
    out = await chain.ainvoke(_query(filters={"team": "a"}))
    entity = out.citations[1]
    assert [d.file_name for d in entity.documents] == ["logistics.pdf"]
    assert store.fetches[0].filters == {"team": "a", "id": ["tu-2", "tu-3"]}


async def test_a_failed_lineage_read_leaves_the_answer_and_passages(
    config: Config,
) -> None:
    chain, store, _ = _chain(config)

    async def broken(query: SearchQuery) -> list[RetrievalResult]:
        if "id" in (query.filters or {}):
            raise RuntimeError("store down")
        return _hits()

    store.aretrieve = broken  # type: ignore[method-assign]
    out = await chain.ainvoke(_query())
    assert out.answer.startswith("Depot stores widgets [1]")
    # Recorded in the query's ledger as an unresolved read (rv8: a state
    # flag reported as True, beside the ledger).
    assert out.metadata["citations_unresolved"] == [
        {
            "step": "citation_lineage",
            "reason": "graph citations name no document",
            "error_type": "RuntimeError",
        }
    ]
    assert "degraded" not in out.metadata
    assert out.citations[0].file_name == "contract.txt"
    assert out.citations[1].documents == []


async def test_the_lineage_read_runs_alongside_the_answer(config: Config) -> None:
    # The read needs only the context: it no longer waits for the answer.
    # The answer model (run in a worker thread) waits for the read to start;
    # run one after the other it would wait out its timeout.
    started = threading.Event()
    seen: list[bool] = []
    chain, store, _ = _chain(
        config, on_answer=lambda: seen.append(started.wait(timeout=5))
    )
    fetch = store.aretrieve

    async def recording(query: SearchQuery) -> list[RetrievalResult]:
        if "id" in (query.filters or {}):
            started.set()
        return await fetch(query)

    store.aretrieve = recording  # type: ignore[method-assign]
    out = await chain.ainvoke(_query())
    assert seen == [True]
    assert [d.file_name for d in out.citations[1].documents] == [
        "logistics.pdf",
        "secret.txt",
    ]


async def test_a_failed_answer_cancels_the_lineage_read(config: Config) -> None:
    config.processing.ignore_errors = False

    def broken() -> None:
        raise RuntimeError("model down")

    chain, store, _ = _chain(config, on_answer=broken)
    cancelled = asyncio.Event()
    fetch = store.aretrieve

    async def hanging(query: SearchQuery) -> list[RetrievalResult]:
        if "id" not in (query.filters or {}):
            return await fetch(query)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    store.aretrieve = hanging  # type: ignore[method-assign]
    with pytest.raises(Exception, match="model down"):
        await chain.ainvoke(_query())
    assert cancelled.is_set()


async def test_citations_off_restores_the_earlier_context_and_prompt(
    config: Config,
) -> None:
    config.search.citations.enabled = False
    chain, store, prompts = _chain(config)
    out = await chain.ainvoke(_query())
    system, human = prompts[-1]
    assert CITATION_INSTRUCTIONS not in system
    assert "[tu-1] Depot stores widgets for Vendor." in human
    assert out.citations == []
    assert store.fetches == []
    assert "unknown_citation_markers" not in out.metadata


async def test_an_empty_context_has_no_citations(config: Config) -> None:
    chain, _, prompts = _chain(config, hits=[])
    out = await chain.ainvoke(_query())
    assert out.answer == NO_CONTEXT_ANSWER
    assert out.citations == []
    assert prompts == []
    assert EMPTY_CONTEXT_PLACEHOLDER


async def test_a_blocked_answer_returns_no_citations(config: Config) -> None:
    config.aws.bedrock.guardrail.check_output = True
    chain, _, _ = _chain(config, guard=_Guard(GuardAction.BLOCKED))
    out = await chain.ainvoke(_query())
    assert out.answer == "Refused."
    assert (out.sources, out.citations) == ([], [])
    assert "unknown_citation_markers" not in out.metadata


async def test_a_masked_answer_is_marked_from_its_screened_text(
    config: Config,
) -> None:
    config.aws.bedrock.guardrail.check_output = True
    chain, _, _ = _chain(config, guard=_Guard(GuardAction.MASKED))
    out = await chain.ainvoke(_query())
    assert out.answer == "Masked [1]."
    assert [c.cited for c in out.citations] == [True, False]
    assert "unknown_citation_markers" not in out.metadata


async def test_a_marked_decline_keeps_its_citations_uncited(config: Config) -> None:
    config.search.abstention.mode = "mark"
    chain, _, _ = _chain(config, "[NO_ANSWER] 컨텍스트에 정보가 없습니다.")
    out = await chain.ainvoke(_query())
    assert out.metadata["abstention_reason"] == "marker"
    assert [c.cited for c in out.citations] == [False, False]


async def test_a_streamed_answer_carries_its_markers(config: Config) -> None:
    chain, store, prompts = _chain(config, "倉庫は部品を保管する［１］【2】。")
    chunks = [chunk async for chunk in chain.astream(_query())]
    scanner = CitationMarkerScanner(valid={1, 2})
    for chunk in chunks:
        scanner.feed(chunk)
    scanner.finish()
    assert scanner.markers == [1, 2]
    assert CITATION_INSTRUCTIONS in prompts[-1][0]
    # A streamed answer reads no lineage.
    assert store.fetches == []


@pytest.mark.parametrize("cites", [True, False])
async def test_kg_answer_shows_markers_only_for_a_strategy_that_cites(
    config: Config, cites: bool
) -> None:
    from unified_kg_rag.application.cli.mcp_tools import KnowledgeGraphTools

    config.search.citations.strategies = {SearchStrategy.SIMPLE: cites}
    chain, _store, prompts = _chain(config, answer="Depot stores widgets [1].")
    result = await KnowledgeGraphTools(chain).answer(  # type: ignore[arg-type]
        "Who stores the widgets?", strategy="simple"
    )
    assert (CITATION_INSTRUCTIONS in prompts[-1][0]) is cites
    markers = [(r.citation, r.cited) for r in result.records]
    if cites:
        assert (1, True) in markers
    else:
        assert all(m == (None, None) for m in markers)
