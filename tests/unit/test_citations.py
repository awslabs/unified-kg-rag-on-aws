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
    RAGInput,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.prompts.retrieval import (
    CITATION_INSTRUCTIONS,
    SYNTHESIS_CITATION_INSTRUCTIONS,
)
from unified_kg_rag.domain.retrieval.citations import (
    CitationMarkerScanner,
    cited_markers,
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
    assert citations.safe_relative_path(path) is None


def test_a_citation_names_a_relative_path_with_forward_slashes() -> None:
    assert citations.safe_relative_path("legal\\2024\\a.pdf") == "legal/2024/a.pdf"
    assert citations.safe_relative_path("./a.pdf") == "a.pdf"


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
    units for a fetch by id (honouring a ``team`` filter)."""

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
        ]


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
        query="Who stores the widgets?",
        search_strategy=SearchStrategy.SIMPLE,
        enable_query_processing=False,
        **fields,
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


@pytest.mark.parametrize(
    ("strategy", "softened"),
    [
        (SearchStrategy.GLOBAL, True),
        (SearchStrategy.DRIFT, True),
        (SearchStrategy.LOCAL, False),
        (SearchStrategy.MIX, False),
    ],
)
def test_global_and_drift_answers_cite_where_possible(
    config: Config, strategy: SearchStrategy, softened: bool
) -> None:
    # rv5: with citations on, pairwise judges preferred the uncited global
    # and DRIFT answers on comprehensiveness: "after each statement, cite"
    # read as leaving out what has no marker. Their answers are told to
    # keep every relevant point and cite where possible; the grounding
    # rules (the prompt's own) still decide what may be stated.
    chain, _store, _prompts = _chain(config)
    optimized = _token_manager(config).optimize_context(
        _results(), "q", number_sections=True
    )
    instructions = chain._answer_instructions(
        {"optimized_context": optimized, "resolved_strategy": strategy}
    )
    assert instructions.startswith(CITATION_INSTRUCTIONS)
    assert (SYNTHESIS_CITATION_INSTRUCTIONS in instructions) is softened
    assert "Cite where possible" in SYNTHESIS_CITATION_INSTRUCTIONS
    # Without a numbered context there is nothing to cite.
    unnumbered = _token_manager(config).optimize_context(_results(), "q")
    assert (
        chain._answer_instructions(
            {"optimized_context": unnumbered, "resolved_strategy": strategy}
        )
        == ""
    )


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
    assert out.metadata["citations_unresolved"] is True
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
