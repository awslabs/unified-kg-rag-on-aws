# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public API exported from ``unified_kg_rag``.

The pinned list makes a change to the public surface deliberate: adding or
removing a name means editing this test and, for a removal or rename, a
CHANGELOG "Breaking" entry.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import unified_kg_rag

pytestmark = pytest.mark.unit

PUBLIC_API = sorted(
    [
        "__version__",
        # Using the framework
        "Citation",
        "CitationMarkerScanner",
        "CitedDocument",
        "Config",
        "ConversationKey",
        "ConversationTurn",
        "DataIngestionPipeline",
        "EvaluationManager",
        "GraphRAGChain",
        "InputDocument",
        "MemoryManager",
        "Methodology",
        "MetricsSink",
        "ModelPurpose",
        "ModelRole",
        "PipelineConfig",
        "PipelineContext",
        "PipelineStageStatus",
        "PipelineStageType",
        "ProcessedQuery",
        "RAGInput",
        "RAGOutput",
        "SearchOutput",
        "SearchResult",
        "SearchStrategy",
        "SearchType",
        "Source",
        "build_mcp_app",
        "build_mcp_server",
        "configure_event_loop",
        "describe_namespace",
        "export_graph",
        "get_config",
        "setup_logging",
        # Errors
        "ConfigurationError",
        "ConfirmationDeclinedError",
        "DocumentNotFoundError",
        "EmbeddingModelError",
        "ErrorClass",
        "GraphRAGException",
        "IndexNotFoundError",
        "InvalidFilterError",
        "LanguageModelError",
        "MethodologyNotIndexedError",
        "ModelError",
        "ModelUnreachableError",
        "NamespaceLeaseError",
        "PipelineExecutionError",
        "PipelineInterrupted",
        "QueryTimeoutError",
        "RerankModelError",
        "SchemaSourceError",
        "SeedGraphError",
        "StoreConnectionError",
        "StoreRequestRejectedError",
        "classify_error",
        # Extending it
        "BaseGraphRAGRetriever",
        "BaseRenderer",
        "BaseSearchStrategy",
        "ContentGuardPort",
        "DocStatusPort",
        "EmbeddingFactoryPort",
        "GraphIndexer",
        "GraphQueryPort",
        "GuardAction",
        "GuardSource",
        "GuardVerdict",
        "LLMFactoryPort",
        "ModelFactoryPort",
        "ParserFactory",
        "ProviderRole",
        "Providers",
        "QueryInput",
        "RerankFactoryPort",
        "RetrieverRole",
        "SchemaDocument",
        "SchemaSourceContext",
        "SeedEdge",
        "SeedGraph",
        "SeedNode",
        "SeedSourceContext",
        "SeedSourceOptions",
        "TokenCounterPort",
        "VectorIndexer",
        "build_providers",
        "register_provider",
        "register_renderer",
        "register_schema_source",
        "register_seed_source",
        "register_strategy",
    ]
)


def test_the_public_api_is_pinned() -> None:
    assert sorted(unified_kg_rag.__all__) == PUBLIC_API


@pytest.mark.parametrize("name", [n for n in PUBLIC_API if n != "__version__"])
def test_every_public_name_resolves_to_its_defining_module(name: str) -> None:
    value = getattr(unified_kg_rag, name)
    module = unified_kg_rag._EXPORTS[name]
    assert value is getattr(sys.modules[module], name)


def test_the_type_checking_imports_match_the_exports() -> None:
    # Type checkers read the TYPE_CHECKING block; the runtime reads _EXPORTS.
    tree = ast.parse(Path(unified_kg_rag.__file__).read_text(encoding="utf-8"))
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING"
    )
    imported = {
        alias.name: node.module
        for node in block.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imported == unified_kg_rag._EXPORTS


def test_an_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'GraphRagChain'"):
        unified_kg_rag.GraphRagChain  # noqa: B018


def test_importing_the_package_loads_none_of_it() -> None:
    # The facade is lazy: `import unified_kg_rag` (as the CLIs do for
    # __version__) imports no subpackage and no model, AWS or store SDK.
    script = (
        "import json, sys\n"
        "import unified_kg_rag\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith(('unified_kg_rag.', 'langchain', 'boto3', 'pydantic', "
        "'opensearchpy', 'gremlin_python')))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# Top-level modules of the optional extras the public API must not need.
_EXTRA_MODULES = (
    # viz
    "pyvis",
    # eval
    "datasets",
    "langchain_community",
    "langchain_openai",
    "pandas",
    "ragas",
    # mcp
    "mcp",
)


@pytest.mark.slow
def test_the_public_api_loads_no_optional_extra() -> None:
    # Every public name (BaseRenderer, register_renderer and
    # EvaluationManager included) works on a core install: the built-in
    # renderers load pyvis only when they render, the manager the
    # judges' packages only when they are enabled, and the CLIs import too.
    script = (
        "import json, sys\n"
        "import unified_kg_rag\n"
        "for name in unified_kg_rag.__all__:\n"
        "    getattr(unified_kg_rag, name)\n"
        "import unified_kg_rag.adapters.renderers\n"
        "import unified_kg_rag.application.cli.run_evaluation\n"
        "import unified_kg_rag.application.cli.run_visualization\n"
        "import unified_kg_rag.application.cli.run_mcp_server\n"
        f"extras = {_EXTRA_MODULES!r}\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} "
        "& set(extras))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
