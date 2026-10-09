# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LightRAG dual-keyword JSON parsing and mode detection (M3)."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.search_strategies.lightrag_search import (
    LightRAGSearchStrategy,
)
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    ProcessedQuery,
)
from unified_kg_rag.domain.models import Config, SearchQuery, SearchStrategy, SearchType
from unified_kg_rag.domain.retrieval.strategy_registry import (
    QueryInput,
    get_strategy_spec,
)
from unified_kg_rag.shared import LanguageModelError

pytestmark = pytest.mark.unit


class _StubChain:
    """Stands in for a setup_chain Runnable, returning a canned LLM output."""

    def __init__(self, output: str) -> None:
        self._output = output

    async def ainvoke(self, _inputs: dict, config=None) -> str:
        return self._output


class TestRegisteredQueryInputs:
    """Query-side extractions are declared on the strategy registration."""

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            (SearchStrategy.LOCAL, {QueryInput.ENTITIES}),
            (SearchStrategy.DRIFT, {QueryInput.ENTITIES}),
            (SearchStrategy.MIX, {QueryInput.DUAL_KEYWORDS}),
            (SearchStrategy.HYBRID, {QueryInput.DUAL_KEYWORDS}),
            (SearchStrategy.NAIVE, set()),
            (SearchStrategy.GLOBAL, set()),
            (SearchStrategy.SIMPLE, set()),
        ],
    )
    def test_query_inputs(
        self, strategy: SearchStrategy, expected: set[QueryInput]
    ) -> None:
        assert get_strategy_spec(strategy).query_inputs == expected
        state = {"resolved_strategy": strategy}
        assert GraphRAGChain._needs_query_entities(state) is (
            QueryInput.ENTITIES in expected
        )
        assert GraphRAGChain._needs_dual_keywords(state) is (
            QueryInput.DUAL_KEYWORDS in expected
        )

    def test_unresolved_strategy_extracts_entities_only(self) -> None:
        assert GraphRAGChain._needs_query_entities({}) is True
        assert GraphRAGChain._needs_dual_keywords({}) is False


@pytest.fixture
def chain(config: Config) -> GraphRAGChain:
    return GraphRAGChain(config=config)


class TestExtractDualKeywords:
    async def test_maps_and_coerces(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            chain,
            "_get_chain_for_prompt",
            lambda *a, **k: _StubChain(
                '{"high_level_keywords": ["theme", ""], "low_level_keywords": ["Alice"]}'
            ),
        )
        hl, ll = await chain._extract_dual_keywords("q", "English")
        assert hl == ["theme"]  # falsy "" filtered out
        assert ll == ["Alice"]

    async def test_ignore_errors_returns_empty(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chain.ignore_errors = True
        monkeypatch.setattr(
            chain,
            "_get_chain_for_prompt",
            lambda *a, **k: _StubChain("not json"),
        )
        assert await chain._extract_dual_keywords("q", "English") == ([], [])

    async def test_raises_when_not_ignoring(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        chain.ignore_errors = False
        monkeypatch.setattr(
            chain,
            "_get_chain_for_prompt",
            lambda *a, **k: _StubChain("not json"),
        )
        with pytest.raises(json.JSONDecodeError):
            await chain._extract_dual_keywords("q", "English")

    @pytest.mark.parametrize(
        "raw",
        [
            '{"high_level_keywords": "theme", "low_level_keywords": ["Alice"]}',
            '{"high_level_keywords": null, "low_level_keywords": ["Alice"]}',
            '{"high_level_keywords": [1, {"k": 2}], "low_level_keywords": []}',
            '["theme", "Alice"]',
            "{}",
        ],
    )
    async def test_wrong_shape_raises_when_not_ignoring(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        chain.ignore_errors = False
        monkeypatch.setattr(
            chain, "_get_chain_for_prompt", lambda *a, **k: _StubChain(raw)
        )
        with pytest.raises(LanguageModelError, match="keyword"):
            await chain._extract_dual_keywords("q", "English")

    async def test_wrong_shape_degrades_to_empty_when_ignoring(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A bare string must never be split into single-character keywords.
        chain.ignore_errors = True
        monkeypatch.setattr(
            chain,
            "_get_chain_for_prompt",
            lambda *a, **k: _StubChain(
                '{"high_level_keywords": "theme", "low_level_keywords": ["Alice"]}'
            ),
        )
        assert await chain._extract_dual_keywords("q", "English") == ([], [])

    async def test_numeric_keywords_are_kept_as_text(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A year or a quantity is a legitimate keyword; the model emits it as a
        # JSON number, which must not fail the query.
        chain.ignore_errors = False
        monkeypatch.setattr(
            chain,
            "_get_chain_for_prompt",
            lambda *a, **k: _StubChain(
                '{"high_level_keywords": [2024, "Acme"], '
                '"low_level_keywords": [3.5, "Vendor"]}'
            ),
        )
        assert await chain._extract_dual_keywords("q", "English") == (
            ["2024", "Acme"],
            ["3.5", "Vendor"],
        )

    @pytest.mark.parametrize("item", ["null", "true", '["x"]', '{"k": 1}'])
    async def test_non_scalar_keyword_raises(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch, item: str
    ) -> None:
        chain.ignore_errors = False
        monkeypatch.setattr(
            chain,
            "_get_chain_for_prompt",
            lambda *a, **k: _StubChain(
                f'{{"high_level_keywords": ["Acme", {item}], "low_level_keywords": []}}'
            ),
        )
        with pytest.raises(LanguageModelError, match="keyword"):
            await chain._extract_dual_keywords("q", "English")

    async def test_missing_level_is_empty(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chain.ignore_errors = False
        monkeypatch.setattr(
            chain,
            "_get_chain_for_prompt",
            lambda *a, **k: _StubChain('{"low_level_keywords": ["Alice"]}'),
        )
        assert await chain._extract_dual_keywords("q", "English") == ([], ["Alice"])


class TestSearchStepThreading:
    async def test_lightrag_mode_threaded_into_search_query(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        class _Strategy:
            async def asearch(self, query: SearchQuery, config=None):
                captured["query"] = query
                return "result"

        monkeypatch.setattr(chain, "_get_strategy_instance", lambda _s: _Strategy())
        state = {
            "resolved_strategy": SearchStrategy.HYBRID,
            "processed_query": ProcessedQuery(
                original_query="q",
                final_query="q",
                hl_keywords=["t"],
                ll_keywords=["e"],
            ),
            "search_type": SearchType.HYBRID,
        }
        await chain._search_step(state)
        sq = captured["query"]
        assert sq.metadata["search_strategy"] == "hybrid"
        assert sq.hl_keywords == ["t"] and sq.ll_keywords == ["e"]

    async def test_graphrag_mode_threaded_into_search_query(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        class _Strategy:
            async def asearch(self, query: SearchQuery, config=None):
                captured["query"] = query
                return "result"

        monkeypatch.setattr(chain, "_get_strategy_instance", lambda _s: _Strategy())
        state = {
            "resolved_strategy": SearchStrategy.LOCAL,
            "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        }
        await chain._search_step(state)
        assert captured["query"].metadata == {"search_strategy": "local"}


class TestLightragModeFromQuery:
    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            ({"search_strategy": "naive"}, "naive"),
            ({"lightrag_mode": "hybrid"}, "hybrid"),
            ({}, "mix"),
        ],
    )
    def test_mode(self, metadata: dict, expected: str) -> None:
        strategy = LightRAGSearchStrategy.__new__(LightRAGSearchStrategy)
        query = SearchQuery(query="q", metadata=metadata)
        assert strategy._mode(query) == expected
