# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP tools end to end on the embedded stores: no container, no AWS.

One ``backend: local`` ingest (the fake Ollama server, English, Korean and
Japanese documents) serves every test: each tool over the real chain and
the SQLite stores, the protocol through the SDK's in-memory client, and
``run-mcp-server`` as a stdio child process. Every boto3 client and Bedrock
factory is patched to fail while the index is built.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
import yaml

from tests.fixtures.extras import requires_mcp
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import (
    DIM,
    FakeCrossEncoder,
    _config,
    _errors,
    _ingest,
    _providers,
)
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.aws import token_counter as token_counter_module
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.cli import mcp_tools
from unified_kg_rag.application.cli.mcp_tools import KnowledgeGraphTools, describe_error
from unified_kg_rag.application.retrieval import methodology_check
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    IndexArtifact,
    PipelineStageStatus,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import McpServerConfig
from unified_kg_rag.shared.exceptions import (
    GraphRAGException,
    MethodologyNotIndexedError,
)

pytestmark = pytest.mark.integration

NAMES = ("Vendor", "Buyer", "Depot", "한빛물류", "東京倉庫")
ANSWER = "Depot stores widgets for Vendor."
DOCUMENTS = {
    "contract.txt": "Vendor supplies widgets to Buyer.",
    "logistics.txt": "Depot stores widgets for Vendor.",
    "korea.txt": "한빛물류 moves widgets for Depot.",
    "japan.txt": "東京倉庫 keeps spare widgets for 한빛물류.",
}


def _respond(system: str, human: str) -> str:
    if "knowledge graph extraction expert" in system:
        return stages._extraction(stages._source_text(human), NAMES)
    if "search strategy selector" in system:
        return '{"strategy": "local"}'
    if "specialized entity extraction expert" in system:
        return "Depot, Vendor"
    if "expert keyword extractor" in system:
        return json.dumps(
            {
                "high_level_keywords": ["widget storage"],
                "low_level_keywords": ["Depot", "Vendor"],
            }
        )
    return stages._respond(system, human)


class Index:
    def __init__(
        self, config: Config, providers: Providers, root: Path, server: FakeOllama
    ) -> None:
        self.config = config
        self.providers = providers
        self.root = root
        self.server = server

    def chain(self, **kwargs: Any) -> GraphRAGChain:
        return GraphRAGChain(self.config, providers=self.providers, **kwargs)


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Index]:
    root = tmp_path_factory.mktemp("mcp")
    config = _config(root / "stores", root / "viz")
    server = FakeOllama(_respond, ANSWER, DIM)
    providers = _providers(config, server, FakeCrossEncoder("synthetic-reranker"))
    source = root / "corpus"
    source.mkdir()
    for name, text in DOCUMENTS.items():
        (source / name).write_text(text, encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        for target, name in (
            (bedrock_module.BaseBedrockModelFactory, "__init__"),
            (token_counter_module.BedrockTokenCounter, "__init__"),
        ):
            patch.setattr(target, name, _forbidden(f"{target}.{name}"))
        patch.setattr(boto3.Session, "client", _forbidden("boto3.Session.client"))
        context = _ingest(config, providers, source, root / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    yield Index(config, providers, root, server)


def _forbidden(what: str) -> Any:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(f"{what} called")

    return fail


@pytest.fixture
async def tools(index: Index) -> Any:
    chain = index.chain()
    try:
        yield KnowledgeGraphTools(chain)
    finally:
        await chain.aclose()


async def test_kg_search_returns_chunks_with_their_documents(
    tools: KnowledgeGraphTools,
) -> None:
    result = await tools.search(
        "Who stores the widgets for Vendor?", strategy="local", top_k=10
    )
    assert result.strategy == "local" and result.namespace == "default"
    # The id the server's log lines of the call carry (no trace: tracing off).
    assert result.query_id and len(result.query_id) == 12
    assert result.trace_id is None
    texts = [r for r in result.records if r.kind == "text"]
    assert texts, [r.kind for r in result.records]
    assert {r.document_name for r in texts} <= set(DOCUMENTS)
    assert all(r.chunk_id and r.source_id == r.chunk_id for r in texts)
    entities = [r for r in result.records if r.kind == "entity"]
    assert entities and all(r.documents for r in entities)
    assert {e.name for r in entities for e in r.entities} <= set(NAMES)
    # The query's model calls: the local strategy's query extraction.
    assert result.usage["llm_calls"] >= 1


async def test_kg_answer_answers_with_its_sources(tools: KnowledgeGraphTools) -> None:
    result = await tools.answer("Who stores the widgets for Vendor?", strategy="mix")
    assert result.answer == ANSWER
    assert result.records and not result.abstained
    assert result.usage["llm_calls"] >= 2  # keywords and the answer
    assert {r.document_name for r in result.records if r.kind == "text"} <= set(
        DOCUMENTS
    )


@pytest.mark.parametrize("name", ["Depot", "한빛물류", "東京倉庫"])
async def test_kg_get_entity_finds_an_exact_name_in_any_script(
    tools: KnowledgeGraphTools, name: str
) -> None:
    result = await tools.get_entity(name=name)
    assert result.match == "name"
    (record,) = result.records
    assert record.name == name and record.kind == "entity"
    assert record.documents and set(record.documents) <= set(DOCUMENTS)
    assert result.usage["llm_calls"] == 0
    # By the id it returned, too.
    again = await tools.get_entity(entity_id=record.source_id)
    assert (again.match, [r.name for r in again.records]) == ("id", [name])


async def test_kg_get_entity_falls_back_to_a_name_search(
    tools: KnowledgeGraphTools,
) -> None:
    result = await tools.get_entity(name="the Depot warehouse")
    assert result.match == "search"
    assert "Depot" in [r.name for r in result.records]
    nothing = await tools.get_entity(name="Zyzzyva")
    assert (nothing.match, nothing.records) == ("none", [])


async def test_kg_neighbors_walks_the_graph_with_relationships(
    tools: KnowledgeGraphTools,
) -> None:
    result = await tools.neighbors(name="Depot", max_hops=1, limit=10)
    assert result.entity is not None and result.entity.name == "Depot"
    neighbours = {r.name: r for r in result.records if r.kind == "entity"}
    assert {"Vendor", "한빛물류"} <= set(neighbours)
    assert all(r.attributes["hops"] == 1 for r in neighbours.values())
    relationships = [r for r in result.records if r.kind == "relationship"]
    assert relationships
    for relationship in relationships:
        ends = {e.name for e in relationship.entities}
        assert "Depot" in ends or ends <= set(neighbours)
        assert relationship.text_unit_ids and relationship.documents
    two = await tools.neighbors(name="Depot", max_hops=2, limit=10)
    hops = {r.name: r.attributes["hops"] for r in two.records if r.kind == "entity"}
    assert hops.get("Buyer") == 2 or hops.get("東京倉庫") == 2, hops


async def test_kg_neighbors_reads_graph_edges_without_relationship_vectors(
    tools: KnowledgeGraphTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = await methodology_check.namespace_artifacts(tools.chain, None)
    without = frozenset(held - {IndexArtifact.RELATIONSHIP_VECTORS})

    async def artifacts(*_args: Any, **_kwargs: Any) -> frozenset[IndexArtifact]:
        return without

    monkeypatch.setattr(methodology_check, "namespace_artifacts", artifacts)
    result = await tools.neighbors(name="Depot")
    edges = [r for r in result.records if r.kind == "relationship"]
    assert edges and all("weight" in r.attributes for r in edges)
    assert any(e.name == "Depot" for r in edges for e in r.entities)


async def test_a_namespace_without_a_graph_refuses_the_graph_tools(
    tools: KnowledgeGraphTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def artifacts(*_args: Any, **_kwargs: Any) -> frozenset[IndexArtifact]:
        return frozenset({IndexArtifact.TEXT_UNITS, IndexArtifact.SUMMARY_TREE})

    monkeypatch.setattr(methodology_check, "namespace_artifacts", artifacts)
    with pytest.raises(MethodologyNotIndexedError, match="holds raptor"):
        await tools.get_entity(name="Depot")


async def test_kg_describe_schema_lists_what_the_namespace_holds(
    tools: KnowledgeGraphTools,
) -> None:
    result = await tools.describe_schema()
    assert {"graphrag", "lightrag"} <= set(result.methodologies)
    assert {"auto", "local", "mix", "global"} <= set(result.strategies)
    # What auto routes among is search.auto_routable_strategies, not every
    # strategy the namespace serves.
    assert set(result.auto_routes) < set(result.strategies)
    assert "naive" in result.strategies and "naive" not in result.auto_routes
    assert result.build is not None and result.build["pipeline_id"] == "run-1"
    assert "name.keyword" in result.filter_keys
    assert result.extraction_schema is not None
    assert result.namespaces == ["default"]
    assert result.exists


async def test_kg_describe_schema_of_a_namespace_nothing_was_ingested_under(
    index: Index, caplog: pytest.LogCaptureFixture
) -> None:
    # It listed the config's methodologies and 11 strategies for a typo'd
    # namespace, with no exists field; a search on it is index_not_found.
    import logging

    from unified_kg_rag.shared.exceptions import IndexNotFoundError

    config = index.config.model_copy(deep=True)
    config.mcp.allowed_namespaces = ["nosuch"]
    chain = GraphRAGChain(config, providers=index.providers)
    try:
        tools = KnowledgeGraphTools(chain)
        result = await tools.describe_schema(namespace="nosuch")
        # A graph tool checks the namespace first, as a query does: no
        # retrieval failure is logged with its traceback for it.
        with caplog.at_level(logging.ERROR):
            with pytest.raises(IndexNotFoundError, match="nosuch"):
                await tools.get_entity(name="Depot", namespace="nosuch")
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    finally:
        await chain.aclose()
    assert result.namespace == "nosuch" and not result.exists
    assert (result.methodologies, result.artifacts, result.strategies) == ([], [], [])
    assert result.auto_routes == []


# --- the strategy allowlist ------------------------------------------------


async def test_auto_runs_only_an_offered_strategy(index: Index) -> None:
    # The router would answer "local"; with mix the only offered strategy
    # auto may route to, auto runs mix and asks no router.
    offered = [SearchStrategy.AUTO, SearchStrategy.MIX]
    chain = index.chain(allowed_strategies=offered)
    try:
        tools = KnowledgeGraphTools(chain, settings=McpServerConfig(strategies=offered))
        calls = len(index.server.chat_models)
        result = await tools.search("Who stores the widgets for Vendor?")
        assert result.strategy == "mix"
        prompts = len(index.server.chat_models) - calls
        assert prompts == result.usage["llm_calls"] == 1  # the keywords only
    finally:
        await chain.aclose()


@pytest.mark.parametrize("ignore_errors", [False, True])
@pytest.mark.parametrize(
    ("offered", "routable"),
    [
        # tree reads the summary tree, which the namespace does not hold.
        (["auto", "tree"], None),
        # No configured route serves the namespace: the routing fallback
        # (local) is not offered, so auto is refused rather than run it.
        (["auto", "mix"], ["tree"]),
    ],
)
async def test_auto_with_no_offered_strategy_served_is_refused_before_any_call(
    index: Index, offered: list[str], routable: list[str] | None, ignore_errors: bool
) -> None:
    config = index.config.model_copy(deep=True)
    config.processing.ignore_errors = ignore_errors
    if routable is not None:
        # Unvalidated: tree stands for a route the namespace does not serve
        # (search.auto_routable_strategies refuses it at load).
        config.search = config.search.model_copy(
            update={"auto_routable_strategies": [SearchStrategy(s) for s in routable]}
        )
    strategies = [SearchStrategy(s) for s in offered]
    chain = GraphRAGChain(
        config, providers=index.providers, allowed_strategies=strategies
    )
    try:
        tools = KnowledgeGraphTools(
            chain, settings=McpServerConfig(strategies=strategies)
        )
        calls = len(index.server.chat_models), len(index.server.embed_models)
        # Under processing.ignore_errors the chain absorbs the refusal into
        # its error answer; the caller still gets its message.
        with pytest.raises(GraphRAGException, match="auto may run only") as raised:
            await tools.answer("Who stores the widgets for Vendor?")
        assert describe_error(raised.value)[1] is True
        assert (len(index.server.chat_models), len(index.server.embed_models)) == calls
    finally:
        await chain.aclose()


@pytest.mark.parametrize(
    ("given", "match", "found"),
    [
        # Decomposed Hangul (NFD), as some keyboards and file systems give it.
        (unicodedata.normalize("NFD", "한빛물류"), "name", "한빛물류"),
        # Full-width Latin: an exact name after NFKC.
        ("Ｄｅｐｏｔ", "name", "Depot"),
        # Full-width, other case: the NFKC name search finds it, the name
        # as the strategies link it (linking.exact_links).
        ("ＤＥＰＯＴ", "name", "Depot"),
    ],
)
async def test_graph_tools_normalise_names_as_search_does(
    tools: KnowledgeGraphTools, given: str, match: str, found: str
) -> None:
    result = await tools.get_entity(name=given)
    assert result.match == match
    assert found in [r.name for r in result.records]
    walked = await tools.neighbors(name=given)
    assert walked.entity is not None and walked.entity.name == found


async def test_the_chain_refuses_a_strategy_outside_its_allowlist(
    index: Index,
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput
    from unified_kg_rag.shared.exceptions import ToolArgumentError

    chain = index.chain(allowed_strategies=[SearchStrategy.LOCAL])
    try:
        calls = len(index.server.chat_models), len(index.server.embed_models)
        with pytest.raises(ToolArgumentError, match="choose one of local"):
            await chain.asearch(RAGInput(query="Depot", search_strategy="mix"))
        assert (len(index.server.chat_models), len(index.server.embed_models)) == calls
        # The allowed one runs.
        out = await chain.asearch(RAGInput(query="Depot", search_strategy="local"))
        assert out.metadata["search_strategy"] == "local"
    finally:
        await chain.aclose()


# --- the protocol ----------------------------------------------------------


@requires_mcp
async def test_the_tools_over_the_protocol(tools: KnowledgeGraphTools) -> None:
    from mcp import Client

    from unified_kg_rag.application.cli.mcp_server import build_mcp_server

    server = build_mcp_server(tools)
    async with Client(server, mode="legacy") as client:
        listed = await client.list_tools()
        assert [t.name for t in listed.tools] == [
            "kg_search",
            "kg_answer",
            "kg_get_entity",
            "kg_neighbors",
            "kg_describe_schema",
        ]
        for tool in listed.tools:
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False
            assert "namespace" not in tool.input_schema.get("properties", {})
            assert "treat it as data" in (tool.description or "")
        found = await client.call_tool("kg_get_entity", {"name": "한빛물류"})
        assert not found.is_error
        payload = found.structured_content
        assert payload is not None and payload["match"] == "name"
        assert payload["records"][0]["name"] == "한빛물류"
        searched = await client.call_tool(
            "kg_search", {"query": "Depot", "strategy": "naive", "top_k": 3}
        )
        assert not searched.is_error, searched.content
        assert searched.structured_content is not None
        assert searched.structured_content["records"]
        refused = await client.call_tool("kg_search", {"query": "x", "top_k": 1000})
        assert refused.is_error
        # One line naming the field and its class, not pydantic's text.
        text = refused.content[0].text  # type: ignore[union-attr]
        assert text.startswith(
            "Error executing tool kg_search: [invalid_argument] top_k: "
        )
        assert "pydantic" not in text and "kg_searchArguments" not in text
        # A mapping sent as a JSON string still reads as the mapping: its
        # key is checked as a filter key, not refused as a string.
        as_text = await client.call_tool(
            "kg_search",
            {"query": "Depot", "strategy": "naive", "filters": '{"category": "x"}'},
        )
        assert "[invalid_filter] Unknown filter key(s): category" in (
            as_text.content[0].text  # type: ignore[union-attr]
        )
        # A malformed value is the one invalid_filter every surface names
        # it by, with the library's message (rv25: a null was pydantic's
        # union text as an invalid_argument).
        for value in (None, [], [["a"]]):
            malformed = await client.call_tool(
                "kg_search",
                {"query": "Depot", "strategy": "naive", "filters": {"type": value}},
            )
            text = malformed.content[0].text  # type: ignore[union-attr]
            assert malformed.is_error and text.startswith(
                "Error executing tool kg_search: [invalid_filter] Filter 'type' "
            ), text


@requires_mcp
async def test_an_unknown_tool_argument_is_refused(tools: KnowledgeGraphTools) -> None:
    # kg_neighbors {"depth": 99} ran with the default max_hops, and a
    # copied RAGInput key ("search_strategy" for "strategy") ran auto.
    from mcp import Client

    from unified_kg_rag.application.cli.mcp_server import build_mcp_server

    server = build_mcp_server(tools)
    async with Client(server, mode="legacy") as client:
        listed = await client.list_tools()
        for tool in listed.tools:
            assert tool.input_schema.get("additionalProperties") is False, tool.name
        for name, arguments, unknown in (
            ("kg_neighbors", {"name": "Depot", "depth": 99}, "depth"),
            (
                "kg_search",
                {"query": "Depot", "search_strategy": "naive"},
                "search_strategy",
            ),
            ("kg_describe_schema", {"suffix": "other"}, "suffix"),
        ):
            refused = await client.call_tool(name, arguments)
            assert refused.is_error, name
            text = refused.content[0].text  # type: ignore[union-attr]
            assert text.startswith(
                f"Error executing tool {name}: [invalid_argument] {unknown}: "
            ), text


@requires_mcp
async def test_a_store_failure_reaches_the_caller_without_its_details(
    tools: KnowledgeGraphTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp import Client

    from unified_kg_rag.application.cli.mcp_server import build_mcp_server
    from unified_kg_rag.shared.exceptions import StoreConnectionError

    async def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise StoreConnectionError("cannot reach https://vpc-secret.internal:8182")

    monkeypatch.setattr(mcp_tools.graph_lookup, "find_entities", broken)
    async with Client(build_mcp_server(tools)) as client:
        result = await client.call_tool("kg_get_entity", {"name": "Depot"})
    assert result.is_error
    text = " ".join(getattr(c, "text", "") for c in result.content)
    assert "vpc-secret" not in text and mcp_tools.INTERNAL_ERROR_MESSAGE in text


@requires_mcp
async def test_an_unexpected_failure_is_logged_under_the_ids_the_caller_gets(
    tools: KnowledgeGraphTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp import Client
    from structlog.contextvars import get_contextvars

    from unified_kg_rag.application.cli import mcp_server
    from unified_kg_rag.application.cli.mcp_server import build_mcp_server
    from unified_kg_rag.shared.exceptions import tag_query_failure

    async def broken(*_args: Any, **_kwargs: Any) -> Any:
        error = RuntimeError("internal detail")
        tag_query_failure(error, "q-traceback", "t-traceback")
        raise error

    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(mcp_tools.graph_lookup, "find_entities", broken)
    monkeypatch.setattr(
        mcp_server.logger,
        "error",
        lambda *_args, **_kwargs: logged.append(get_contextvars()),
    )
    async with Client(build_mcp_server(tools)) as client:
        result = await client.call_tool("kg_get_entity", {"name": "Depot"})
    assert result.is_error
    text = " ".join(getattr(c, "text", "") for c in result.content)
    assert "q-traceback" in text
    # The traceback's line names the id the caller was told to look for.
    assert logged == [{"query_id": "q-traceback", "trace_id": "t-traceback"}]


class _Metrics:
    def __init__(self) -> None:
        self.rows: list[tuple[dict[str, float], dict[str, str]]] = []

    def emit(
        self,
        namespace: str,
        metrics: dict[str, float],
        dimensions: dict[str, str] | None = None,
    ) -> None:
        self.rows.append((dict(metrics), dict(dimensions or {})))


async def test_a_kg_answer_past_the_deadline_is_recorded_by_the_chain(
    index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from unified_kg_rag.shared.exceptions import QueryTimeoutError, query_failure_ids

    metrics = _Metrics()
    monkeypatch.setattr(index.providers, "metrics_sink", metrics)
    real_screen = GraphRAGChain._screen_input

    async def slow_screen(self: GraphRAGChain, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(3)
        return await real_screen(self, *args, **kwargs)

    monkeypatch.setattr(GraphRAGChain, "_screen_input", slow_screen)
    config = index.config.updated({"search.query_timeout_seconds": 0.5})
    chain = GraphRAGChain(config, providers=index.providers)
    try:
        with pytest.raises(QueryTimeoutError) as raised:
            await KnowledgeGraphTools(chain).answer(
                "가젯을 보관하는 곳은?", strategy="local"
            )
    finally:
        await chain.aclose()
    # The chain's own deadline fired (the call's, passed in), not the
    # server's backstop: the query is counted, with its class and its ids.
    assert "The query did not finish within 0.5 s" in str(raised.value)
    assert query_failure_ids(raised.value).get("query_id")
    ((values, dimensions),) = metrics.rows
    assert values["query_failed"] == 1.0
    assert dimensions["error_class"] == "timeout"


@requires_mcp
async def test_only_the_configured_tools_are_listed(index: Index) -> None:
    from mcp import Client

    from unified_kg_rag.application.cli.mcp_server import build_mcp_server

    config = index.config.model_copy(deep=True)
    config.mcp.tools = ["kg_search", "kg_get_entity"]
    config.mcp.allowed_namespaces = ["other"]
    chain = GraphRAGChain(config, providers=index.providers)
    try:
        async with Client(build_mcp_server(chain)) as client:
            listed = await client.list_tools()
            assert [t.name for t in listed.tools] == ["kg_search", "kg_get_entity"]
            schema = listed.tools[0].input_schema["properties"]["namespace"]
            assert {"default", "other"} <= set(json.dumps(schema).split('"'))
            refused = await client.call_tool("kg_answer", {"query": "q"})
            assert refused.is_error
    finally:
        await chain.aclose()


@requires_mcp
@pytest.mark.slow
@pytest.mark.timeout(240)
async def test_run_mcp_server_serves_over_stdio(index: Index, tmp_path: Path) -> None:
    from mcp import Client, StdioServerParameters

    path = _stdio_config(index, tmp_path)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "unified_kg_rag.application.cli.run_mcp_server",
            "--config-path",
            str(path),
        ],
        env={"AWS_EC2_METADATA_DISABLED": "true", "PYTHONPATH": os.getcwd()},
        cwd=str(tmp_path),
    )
    async with Client(parameters, mode="legacy", read_timeout_seconds=120) as client:
        found = await client.call_tool("kg_get_entity", {"name": "東京倉庫"})
        assert not found.is_error, found.content
        assert found.structured_content is not None
        assert found.structured_content["records"][0]["name"] == "東京倉庫"
        described = await client.call_tool("kg_describe_schema", {})
        assert not described.is_error, described.content
        assert described.structured_content is not None
        assert "lightrag" in described.structured_content["methodologies"]


def _stdio_config(index: Index, tmp_path: Path) -> Path:
    settings = json.loads(
        json.dumps(index.config.as_input(), default=lambda v: getattr(v, "value", v))
    )
    settings["logging"] = {"log_to_file": False}
    settings["providers"] = {
        **settings.get("providers", {}),
        "llm": {"base_url": "http://127.0.0.1:9"},
        "embedding": {"base_url": "http://127.0.0.1:9"},
    }
    path = tmp_path / "mcp.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return path


@requires_mcp
@pytest.mark.slow
@pytest.mark.timeout(240)
def test_stdout_carries_only_protocol_messages_and_eof_exits_0(
    index: Index, tmp_path: Path
) -> None:
    import subprocess

    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "kg_get_entity", "arguments": {"name": "Depot"}},
        },
    ]
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "AWS_EC2_METADATA_DISABLED": "true",
        "PYTHONPATH": os.getcwd(),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "unified_kg_rag.application.cli.run_mcp_server",
            "--config-path",
            str(_stdio_config(index, tmp_path)),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for message in messages:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        replies: dict[Any, Any] = {}
        while 2 not in replies:
            line = process.stdout.readline()
            assert line, "the server closed stdout"
            reply = json.loads(line)  # every line is a protocol message
            replies[reply.get("id")] = reply
        # The client closes stdin: the server ends, exit code 0.
        process.stdin.close()
        rest = process.stdout.read()
        code = process.wait(timeout=60)
        stderr = process.stderr.read() if process.stderr else ""
    finally:
        if process.poll() is None:
            process.kill()
    assert code == 0, stderr[-2000:]
    assert [json.loads(line) for line in rest.splitlines() if line.strip()] == []
    assert set(replies) == {1, 2}
    assert replies[2]["result"]["structuredContent"]["records"][0]["name"] == "Depot"
    assert "Serving namespace 'default'" in stderr


# --- HTTP ------------------------------------------------------------------

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@requires_mcp
def test_a_mounted_app_checks_the_hosts_it_is_given(index: Index) -> None:
    from starlette.testclient import TestClient

    from unified_kg_rag.application.cli.mcp_server import build_mcp_app

    chain = index.chain()

    proxied = {"Host": "mcp.internal.example"}

    def post(client: Any, headers: dict[str, str]) -> int:
        return int(
            client.post("/mcp", json=_INIT, headers={**_HEADERS, **headers}).status_code
        )

    try:
        # Behind a proxy (no allowed_hosts): its forwarded Host is served.
        with TestClient(build_mcp_app(chain)) as client:
            assert post(client, proxied) == 200
        checked = build_mcp_app(chain, allowed_hosts=["mcp.internal.example"])
        with TestClient(checked) as client:
            assert [
                post(client, headers)
                for headers in (
                    proxied,
                    {"Host": "evil.example"},
                    {**proxied, "Origin": "http://evil.example"},
                    {**proxied, "Origin": "https://mcp.internal.example"},
                )
            ] == [200, 421, 403, 200]
            # Stateless: no event stream to open, no session to delete.
            assert client.get("/mcp", headers=proxied).status_code == 405
            assert client.delete("/mcp", headers=proxied).status_code == 405
    finally:
        chain.close()


def _free_port(host: str) -> int:
    import socket

    with socket.socket() as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


@requires_mcp
@pytest.mark.slow
@pytest.mark.timeout(240)
def test_http_on_any_loopback_address_refuses_foreign_hosts_and_origins(
    index: Index, tmp_path: Path
) -> None:
    import socket
    import subprocess
    import time

    import httpx

    # 127.0.0.2: a loopback address the SDK's own check does not cover.
    host = "127.0.0.2"
    port = _free_port(host)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "AWS_EC2_METADATA_DISABLED": "true",
        "PYTHONPATH": os.getcwd(),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "unified_kg_rag.application.cli.run_mcp_server",
            "--config-path",
            str(_stdio_config(index, tmp_path)),
            "--transport",
            "streamable-http",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env=environment,
    )
    try:
        for _ in range(120):
            try:
                socket.create_connection((host, port), timeout=1).close()
                break
            except OSError:
                assert process.poll() is None, process.communicate()[1][-2000:]
                time.sleep(0.5)
        url = f"http://{host}:{port}/mcp"

        def post(**headers: str) -> int:
            return httpx.post(
                url, json=_INIT, headers={**_HEADERS, **headers}, timeout=30
            ).status_code

        assert post() == 200
        assert post(Host="evil.example") == 421
        assert post(Origin="http://evil.example") == 403
        assert post(Origin=f"http://{host}:{port}") == 200
        assert httpx.get(url, timeout=10).status_code == 405
    finally:
        process.terminate()
        out, err = process.communicate(timeout=60)
    assert process.returncode == 143, err[-2000:]
    # The access log goes to stderr: stdout stays empty.
    assert out == ""
    assert "POST /mcp" in err
