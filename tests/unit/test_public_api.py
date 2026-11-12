# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public API exported from ``unified_kg_rag``.

The pinned list makes a change to the public surface deliberate: adding or
removing a name means editing this test and, for a removal or rename, a
CHANGELOG "Breaking" entry.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

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
        "InvalidArgumentError",
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
        "RunRecord",
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
        # The types the extension surface names
        "Claim",
        "Community",
        "CommunityReport",
        "DocStatusRecord",
        "DocumentDelta",
        "Entity",
        "ExtractionSettings",
        "FilterFields",
        "GraphEdge",
        "GraphProfile",
        "GraphQueryRequest",
        "GraphQueryResult",
        "GroundingContext",
        "IndexArtifact",
        "IndexBuildRecord",
        "IndexLimits",
        "IndexingStats",
        "LanguageModelInfo",
        "Relationship",
        "RenderContext",
        "RetrievalResult",
        "SearchQuery",
        "StoredAnnotation",
        "SummaryNode",
        "TextUnit",
        "TransientRetryConfig",
        "load_seed_graphs",
        "resolve_extraction_schema",
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


def _module_names(module: ModuleType) -> dict[str, Any]:
    """The names an annotation in ``module`` can use: its globals and the
    names its ``TYPE_CHECKING`` blocks import."""
    names = dict(vars(module))
    tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING":
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom):
                    source = importlib.import_module(
                        "." * statement.level + (statement.module or ""),
                        module.__package__,
                    )
                    for alias in statement.names:
                        if hasattr(source, alias.name):
                            names[alias.asname or alias.name] = getattr(
                                source, alias.name
                            )
    return names


def _extension_points(value: Any) -> list[Callable[..., Any]]:
    """The functions of an exported extension point whose signatures an
    extender reads: a registry function itself, or every public method (and
    ``__init__``) a port, base class or ``Providers`` defines in the package."""
    if inspect.isfunction(value):
        return [value]
    found = []
    for klass in value.__mro__:
        if not klass.__module__.startswith("unified_kg_rag."):
            continue
        for name, member in vars(klass).items():
            function = getattr(member, "__func__", member)
            if inspect.isfunction(function) and (
                name == "__init__" or not name.startswith("_")
            ):
                found.append(function)
    return found


def _named_types(function: Callable[..., Any]) -> set[str]:
    """The package's classes ``function``'s annotations name; a ``@final``
    helper's return type is the caller's to pass on, not to name."""
    namespace = _module_names(sys.modules[function.__module__])
    annotations = dict(function.__annotations__)
    if getattr(function, "__final__", False):
        annotations.pop("return", None)
    found = set()
    for annotation in annotations.values():
        text = annotation.__name__ if inspect.isclass(annotation) else str(annotation)
        for token in re.findall(r"[A-Za-z_]\w*", text):
            value = namespace.get(token)
            if inspect.isclass(value) and value.__module__.startswith(
                "unified_kg_rag."
            ):
                found.add(value.__name__)
    return found


def test_the_extension_surface_names_only_exported_types() -> None:
    # A plugin implements a port or a base class from the root package
    # alone (docs/guides/extend.md): every type a port's, a base class's or
    # a registry's signature names is exported too.
    points = [
        value
        for name in unified_kg_rag._EXPORTS
        if (value := getattr(unified_kg_rag, name)) is unified_kg_rag.Providers
        or (inspect.isfunction(value) and name.startswith("register_"))
        or (
            inspect.isclass(value)
            and (inspect.isabstract(value) or getattr(value, "_is_protocol", False))
        )
    ]
    assert unified_kg_rag.VectorIndexer in points
    assert unified_kg_rag.DocStatusPort in points
    missing = {
        f"{function.__qualname__}: {name}"
        for point in points
        for function in _extension_points(point)
        for name in _named_types(function)
        if name not in unified_kg_rag._EXPORTS
    }
    assert not missing, sorted(missing)
