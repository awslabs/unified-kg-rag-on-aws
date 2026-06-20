# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for LightRAG dual-keyword JSON parsing and mode detection (M3)."""

from __future__ import annotations

import pytest

from aws_graphrag.application.retrieval.rag_chain import (
    LIGHTRAG_STRATEGIES,
    GraphRAGChain,
    ProcessedQuery,
)
from aws_graphrag.domain.models import Config, SearchQuery, SearchStrategy, SearchType

pytestmark = pytest.mark.unit


class _StubChain:
    """Stands in for a setup_chain Runnable, returning a canned LLM output."""

    def __init__(self, output: str) -> None:
        self._output = output

    async def ainvoke(self, _inputs: dict) -> str:
        return self._output


class TestParseKeywordJson:
    def test_plain_json(self) -> None:
        out = GraphRAGChain._parse_keyword_json(
            '{"high_level_keywords": ["a"], "low_level_keywords": ["b"]}'
        )
        assert out["high_level_keywords"] == ["a"]
        assert out["low_level_keywords"] == ["b"]

    def test_json_in_code_fence(self) -> None:
        raw = '```json\n{"high_level_keywords": [], "low_level_keywords": ["x"]}\n```'
        out = GraphRAGChain._parse_keyword_json(raw)
        assert out["low_level_keywords"] == ["x"]

    def test_json_wrapped_in_prose(self) -> None:
        raw = 'Here are the keywords: {"high_level_keywords": ["t"], "low_level_keywords": []} done.'
        out = GraphRAGChain._parse_keyword_json(raw)
        assert out["high_level_keywords"] == ["t"]

    def test_empty_object(self) -> None:
        out = GraphRAGChain._parse_keyword_json(
            '{"high_level_keywords": [], "low_level_keywords": []}'
        )
        assert out == {"high_level_keywords": [], "low_level_keywords": []}

    def test_invalid_json_raises(self) -> None:
        import json

        with pytest.raises(json.JSONDecodeError):
            GraphRAGChain._parse_keyword_json("not json at all")


class TestLightragModeDetection:
    def test_lightrag_strategies_membership(self) -> None:
        assert LIGHTRAG_STRATEGIES == {
            SearchStrategy.MIX,
            SearchStrategy.HYBRID,
            SearchStrategy.NAIVE,
        }

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            (SearchStrategy.MIX, True),
            (SearchStrategy.HYBRID, True),
            (SearchStrategy.NAIVE, True),
            (SearchStrategy.LOCAL, False),
            (SearchStrategy.GLOBAL, False),
            (SearchStrategy.DRIFT, False),
            (SearchStrategy.SIMPLE, False),
        ],
    )
    def test_is_lightrag_mode(self, strategy: SearchStrategy, expected: bool) -> None:
        assert (
            GraphRAGChain._is_lightrag_mode({"resolved_strategy": strategy}) is expected
        )


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


class TestSearchStepThreading:
    async def test_lightrag_mode_threaded_into_search_query(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        class _Strategy:
            async def asearch(self, query: SearchQuery):
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
        assert sq.metadata["lightrag_mode"] == "hybrid"
        assert sq.hl_keywords == ["t"] and sq.ll_keywords == ["e"]

    async def test_graphrag_mode_has_no_lightrag_metadata(
        self, chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        class _Strategy:
            async def asearch(self, query: SearchQuery):
                captured["query"] = query
                return "result"

        monkeypatch.setattr(chain, "_get_strategy_instance", lambda _s: _Strategy())
        state = {
            "resolved_strategy": SearchStrategy.LOCAL,
            "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        }
        await chain._search_step(state)
        assert "lightrag_mode" not in captured["query"].metadata
