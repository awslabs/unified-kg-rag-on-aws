# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each shared concept has one owner module; no copy of it grows back.

The rv7 and rv8 rounds folded concepts implemented several ways into one
place each (the degradation ledger, the model-output reader, the answer
view, the script table, the stored-field readers, the atomic write, the
language detector, the filter comparisons, the edge-list codec, the lineage
merge). Each guard below names the pattern a copy takes, its owner and the
other places allowed to hold it, with why (``tests/fixtures/source_guards``):
a new copy fails it, and so does an allowlist entry nothing needs any more.
"""

from __future__ import annotations

import ast
import re

import pytest

from tests.fixtures.source_guards import (
    PACKAGE,
    SourceGuard,
    any_of,
    calls,
    key_reads,
    lines_matching,
    matches,
    nodes,
    string_literals,
)

pytestmark = pytest.mark.unit

# The directories that read retrieved or stored items to answer, rank, cite
# or show them.
_READ_PATHS = (
    "adapters/retrieval",
    "adapters/retrievers",
    "adapters/search_strategies",
    "application/retrieval",
    "application/presentation",
    "application/cli",
    "domain/retrieval",
    "evaluation",
)
# The surfaces that show an answer: the console, the MCP tools, run-rag.
_SURFACES = (
    "application/cli/console/ask.py",
    "application/cli/mcp_tools.py",
    "application/cli/run_rag_chain.py",
)


def _rewrap(node: ast.AST) -> bool:
    """``x if isinstance(x, list) else [x]``: a value-or-list made a list."""
    return (
        isinstance(node, ast.IfExp)
        and isinstance(node.test, ast.Call)
        and getattr(node.test.func, "id", None) == "isinstance"
        and len(node.test.args) == 2
        and "list" in ast.unparse(node.test.args[1])
        and isinstance(node.orelse, ast.List | ast.Set | ast.Tuple)
        and len(node.orelse.elts) == 1
    )


def _comparison_table(node: ast.AST) -> bool:
    """A mapping of range operators to how they compare (callables)."""
    if not isinstance(node, ast.Dict):
        return False
    pairs = {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
    }
    compared = [pairs[op] for op in ("gt", "gte", "lt", "lte") if op in pairs]
    return len(compared) >= 2 and all(
        isinstance(value, ast.Lambda | ast.Attribute) for value in compared
    )


def _lineage_dedup(node: ast.AST) -> bool:
    """``dict.fromkeys(...)`` over text unit ids: a lineage merged by hand
    (a literal tuple of field names is a catalog, not a lineage)."""
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "dict.fromkeys"
        and bool(node.args)
        and not isinstance(node.args[0], ast.Tuple | ast.List)
        and "text_unit" in ast.unparse(node.args[0])
    )


def _sources_read(node: ast.AST) -> bool:
    """``x.sources`` of something that is not the answer view."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "sources"
        and "view" not in ast.unparse(node.value)
    )


GUARDS = [
    SourceGuard(
        name="G1 degradation ledger keys",
        use="DegradationLedger to write them, DegradationReport.of to read them",
        owners=("domain/retrieval/degradation.py",),
        find=string_literals("degraded", "degraded_error", "citations_unresolved"),
        allowed={
            "adapters/search_strategies/iterative_search.py::"
            "IterativeSearchStrategy.asearch": "the iterative run record's own "
            "'degraded' flag (IterativeRun.degraded), not the ledger",
            "evaluation/iterative_statistics.py::iterative_statistics": "counts "
            "that run-record flag",
        },
    ),
    SourceGuard(
        name="G2 model JSON parsing",
        use="shared.utils.model_output.parse_model_object (or the chain's "
        "json parser, which wraps it)",
        owners=("shared/utils/model_output.py",),
        find=calls("json.loads", "ast.literal_eval"),
        within=(
            "adapters/search_strategies",
            "adapters/retrieval",
            "application/retrieval",
            "adapters/llm",
            "adapters/ingestion",
            "adapters/evaluators",
            "evaluation",
        ),
        allowed={
            "adapters/search_strategies/global_search.py::_json_string": "decodes "
            "one salvaged string body of a broken map answer (MS GraphRAG's "
            "json_repair recovery), not an answer",
            "adapters/search_strategies/global_search.py::_unparsed_shape": "names "
            "the parser's complaint for the log; the answer was read already",
            "adapters/search_strategies/structured_search.py::"
            "StructuredSearchStrategy._entity_result": "a vertex's stored filters "
            "JSON, not model output",
            "adapters/ingestion/json_loader.py::JsonTextLoader.lazy_load": "a "
            "corpus JSON file",
            "adapters/llm/response_cache.py::LLMResponseCache.open": "the cache's "
            "own index file",
            "evaluation/benchmarks/musique.py::_items": "the downloaded dataset",
            "evaluation/benchmarks/structural.py::_max_employees": "the benchmark's "
            "generated gold data",
            "evaluation/evaluation_manager.py::EvaluationManager.build_run_manifest": "the "
            "dataset file, hashed by content",
            "evaluation/saved_results.py::_read": "saved result files",
        },
    ),
    SourceGuard(
        name="G3 surfaces read the answer view",
        use="AnswerView (application/presentation/answer_view.py): its sources, "
        "written sections and QueryOutcome",
        owners=("application/presentation/answer_view.py",),
        find=any_of(
            nodes(_sources_read),
            key_reads("sources", "error_class"),
            string_literals("degraded", "degraded_error", "citations_unresolved"),
        ),
        within=_SURFACES,
        allowed={
            "application/cli/run_rag_chain.py::RAGChainRunner._run_query": "reads "
            "the view's own error_class from the JSON payload it built",
        },
    ),
    SourceGuard(
        name="G4 Unicode script ranges",
        use="shared.utils.scripts (script_of, letter_class, the Script table)",
        owners=("shared/utils/scripts.py",),
        find=lines_matching(
            re.compile(
                r"\\u[0-9A-Fa-f]{4}-\\u[0-9A-Fa-f]{4}"
                r"|\\U[0-9A-Fa-f]{8}-\\U[0-9A-Fa-f]{8}"
                r"|0x[0-9A-Fa-f]{4,5}\s*,\s*0x[0-9A-Fa-f]{4,5}"
                r"|[가-힣ぁ-ゟァ-ヿ一-鿿]-[가-힣ぁ-ゟァ-ヿ一-鿿]"
            )
        ),
        allowed={
            "adapters/local/search_engine/analyzers.py::<module>": "ports of "
            "Lucene's analyzer tables (folding, Hebrew, the scanned planes): what "
            "the store does, not a script classification",
            "domain/models/seed_graph.py::<module>": "refuses lone surrogates, an "
            "encoding error",
            "evaluation/answer_match_evaluator.py::<module>": "circled list "
            "numbers (①-⑳), a list marker",
        },
    ),
    SourceGuard(
        name="G5 stored file name and path",
        use="domain.retrieval.source_fields.file_name / safe_relative_path",
        owners=(
            "domain/retrieval/source_fields.py",
            "application/retrieval/citations.py",
        ),
        find=key_reads("file_name", "file_path"),
        allowed={
            "application/cli/console/stages.py::_failed_items": "a stage's failed "
            "file entries (pipeline metrics), not a stored item",
            "application/ingestion/summary_tree_stage.py::SummaryTreeStage._tree_of": "a per-document tree is keyed by the stored path: two files of one "
            "name in other directories are two documents",
            "evaluation/citation_scoring.py::_file_names": "reads a Citation's own "
            "documents (CitedDocument.file_name), built by citations.py",
        },
    ),
    SourceGuard(
        name="G6 stored id lists made lists",
        use="domain.retrieval.source_fields.id_list / listed",
        owners=("domain/retrieval/source_fields.py", "domain/retrieval/filters.py"),
        find=nodes(_rewrap),
        within=_READ_PATHS,
        allowed={
            "adapters/retrieval/base.py::scope_filters": "filter values (a value or "
            "a list), intersected with a scope's",
            "adapters/retrievers/neptune_retriever.py::NeptuneRetriever._hydrate": "a "
            "Gremlin valueMap, whose values are lists (the codec's reading)",
            "adapters/search_strategies/structured_search.py::"
            "StructuredSearchStrategy._request": "a plan's allowed ids argument",
            "application/retrieval/rag_chain.py::_model_texts": "the model-written "
            "texts the output guardrail screens",
            "domain/retrieval/graph_query.py::_check_value_kind": "a structured "
            "plan's operand",
        },
    ),
    SourceGuard(
        name="G7 atomic file replace",
        use="shared.utils.files.atomic_write_text / atomic_write_bytes",
        owners=("shared/utils/files.py",),
        find=calls("os.replace", "os.rename"),
        allowed={
            "application/cli/console/server.py::create_app.upload": "moves an "
            "upload's staged directory into place (a directory, not a file)",
        },
    ),
    SourceGuard(
        name="G8 script counting",
        use="domain.languages (one detector for queries and corpora)",
        owners=("domain/languages", "shared/utils/scripts.py"),
        find=lines_matching(re.compile(r"\bScriptCounts\b|\bcount_letters\(")),
    ),
    SourceGuard(
        name="G9 filter comparison tables",
        use="domain.retrieval.filters (Term.holds, Range.holds) or a store's "
        "filter compiler",
        owners=("domain/retrieval/filters.py",),
        find=nodes(_comparison_table),
        allowed={
            "adapters/graph_query/gremlin.py::GremlinQuery._condition": "the "
            "structured graph query's Gremlin compiler (P.gt, ...), not a filter",
            "adapters/local/search_engine/query.py::<module>": "the embedded "
            "engine's OpenSearch range query, the store itself",
        },
    ),
    SourceGuard(
        name="G10 edge list decoding",
        use="adapters.storage.neptune_codec.decode_edge_list (a stored edge), "
        "domain.retrieval.source_fields.id_list (a result's metadata)",
        owners=(
            "adapters/storage/neptune_codec.py",
            "domain/retrieval/source_fields.py",
        ),
        find=calls("json.loads"),
        within=(
            "adapters/retrievers",
            "adapters/search_strategies",
            "adapters/retrieval",
            "application/retrieval",
            "application/presentation",
            "domain/retrieval",
        ),
        allowed={
            "adapters/search_strategies/global_search.py::_json_string": "a "
            "salvaged map answer string (G2)",
            "adapters/search_strategies/global_search.py::_unparsed_shape": "a "
            "map answer's shape for the log (G2)",
            "adapters/search_strategies/structured_search.py::"
            "StructuredSearchStrategy._entity_result": "a vertex's stored filters "
            "JSON, a mapping, not an edge list",
            "domain/retrieval/graph_query.py::seed_properties": "a seed entity's "
            "properties JSON, a mapping",
        },
    ),
    SourceGuard(
        name="G11 lineage merging",
        use="domain.retrieval.lineage.merge (first_seen or support order)",
        owners=("domain/retrieval/lineage.py",),
        find=nodes(_lineage_dedup),
        within=_READ_PATHS,
        allowed={
            "evaluation/source_resolver.py::TextUnitFileResolver.aresolve": "the "
            "ids to look up, deduplicated before one batched read",
        },
    ),
]


@pytest.mark.parametrize("guard", GUARDS, ids=lambda g: g.name.split(" ", 1)[0])
def test_no_copy_outside_the_owner(guard: SourceGuard) -> None:
    copies = [str(hit) for hit in guard.copies()]
    assert not copies, f"{guard.name}: use {guard.use}: {copies}"


@pytest.mark.parametrize("guard", GUARDS, ids=lambda g: g.name.split(" ", 1)[0])
def test_every_allowlist_entry_is_still_needed(guard: SourceGuard) -> None:
    assert guard.stale() == [], f"{guard.name}: drop these allowlist entries"


@pytest.mark.parametrize("guard", GUARDS, ids=lambda g: g.name.split(" ", 1)[0])
def test_every_owner_and_allowed_module_exists(guard: SourceGuard) -> None:
    # A renamed owner would leave the guard reading no module as it.
    for path in (*guard.owners, *(key.split("::", 1)[0] for key in guard.allowed)):
        assert (PACKAGE / path).exists(), f"{guard.name}: {path}"


@pytest.mark.parametrize(
    ("finder", "source", "found"),
    [
        (string_literals("degraded"), 'x = m.get("degraded")', ["'degraded'"]),
        (string_literals("degraded"), 'def f():\n    """degraded"""\n', []),
        (calls("json.loads"), "import json\njson.loads(s)\n", ["json.loads"]),
        (calls("os.replace"), "import os\nos.replace(a, b)\n", ["os.replace"]),
        (key_reads("file_name"), 'a.get("file_name")\nb["file_name"]\n', 2),
        (key_reads("file_name"), 'b["file_name"] = 1\n', []),
        (nodes(_rewrap), "y = x if isinstance(x, list) else [x]\n", 1),
        (nodes(_comparison_table), 'T = {"gt": lambda a: a, "lt": op.lt}\n', 1),
        (nodes(_comparison_table), 'q = {"gte": start, "lt": end}\n', []),
        (nodes(_lineage_dedup), "u = dict.fromkeys(r.text_unit_ids)\n", 1),
        (nodes(_lineage_dedup), 'F = dict.fromkeys(("id", "text_unit_ids"))\n', []),
        (nodes(_sources_read), "a = output.sources\nb = view.sources\n", 1),
        (GUARDS[3].find, 'R = "[\\u3040-\\u309f]"\n', 1),
        (GUARDS[3].find, 'R = re.compile("(?<![가-힣])없")\n', 1),
        (GUARDS[3].find, "S = ((0x0E00, 0x0E7F),)\n", 1),
    ],
)
def test_each_pattern_catches_the_forms_copies_took(
    finder: object, source: str, found: list[str] | int
) -> None:
    hits = matches(finder, source)  # type: ignore[arg-type]
    assert (len(hits) if isinstance(found, int) else hits) == found


def test_a_guard_reports_copies_and_entries_nothing_needs() -> None:
    owner = "domain/retrieval/degradation.py"
    finder = string_literals("citations_unresolved")
    # Its owner holds the key: no copy, and a stale entry is reported.
    guard = SourceGuard(
        name="probe",
        use="-",
        owners=(owner,),
        find=finder,
        allowed={"nowhere.py::f": "gone"},
    )
    assert guard.copies() == [] and guard.stale() == ["nowhere.py::f"]
    # Without its owner, the owner's own use is a copy, keyed by scope.
    bare = SourceGuard(name="probe", use="-", owners=(), find=finder)
    assert {hit.key for hit in bare.copies()} == {f"{owner}::<module>"}
    # A directory owner covers the modules under it.
    folder = SourceGuard(
        name="probe", use="-", owners=("domain/retrieval",), find=finder
    )
    assert folder.copies() == []
