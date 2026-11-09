# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each shared concept has one owner module; no copy of it grows back.

The rv7 and rv8 rounds folded concepts implemented several ways into one
place each (the degradation ledger, the model-output reader, the answer
view, the script table, the stored-field readers, the atomic write, the
language detector, the filter comparisons, the edge-list codec, the lineage
merge). Each guard below names the pattern a copy takes, its owner and the
other places allowed to hold it, with how many matches each holds and why
(``tests/fixtures/source_guards``): a new copy fails it, one more in an
allowed place too, and so does an entry that counts more than its place
holds, an owner the pattern does not match, or a path that does not exist.
CONTRIBUTING.md ("Single-owner guards") says how to add one.
"""

from __future__ import annotations

import ast
import re

import pytest

from tests.fixtures.source_guards import (
    SourceGuard,
    _scopes,
    any_of,
    calls,
    calls_mentioning,
    key_reads,
    lines_matching,
    matches,
    module_nodes,
    nodes,
    references,
    resolved,
    string_literals,
    string_values,
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
# The surfaces that show an answer: the console, the MCP server, run-rag.
_SURFACES = (
    "application/cli/console",
    "application/cli/mcp_server.py",
    "application/cli/mcp_tools.py",
    "application/cli/run_rag_chain.py",
)


def _isinstance_test(node: ast.AST) -> bool:
    """``isinstance(x, ...)`` or ``not isinstance(x, ...)``."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node = node.operand
    return (
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "isinstance"
        and len(node.args) == 2
    )


def _one_item(node: ast.AST | None) -> bool:
    return isinstance(node, ast.List | ast.Set | ast.Tuple) and len(node.elts) == 1


def _rewrap(node: ast.AST) -> bool:
    """A value-or-list made a list: ``x if isinstance(x, list) else [x]``,
    either way round, or ``if not isinstance(x, list): x = [x]`` (or a
    ``return [x]``)."""
    if isinstance(node, ast.IfExp):
        return _isinstance_test(node.test) and (
            _one_item(node.body) or _one_item(node.orelse)
        )
    if not (
        isinstance(node, ast.If)
        and _isinstance_test(node.test)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Assign | ast.Return)
    ):
        return False
    value = node.body[0].value
    return _one_item(value) or (
        isinstance(value, ast.IfExp)
        and (_one_item(value.body) or _one_item(value.orelse))
    )


_RANGE_OPS = ("gt", "gte", "lt", "lte")


def _comparison_table(node: ast.AST, tree: ast.Module) -> bool:
    """A mapping of range operators to how they compare: lambdas or the
    ``operator`` module's functions, however imported."""
    if not isinstance(node, ast.Dict):
        return False
    pairs = {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
    }
    compared = [pairs[op] for op in _RANGE_OPS if op in pairs]
    return len(compared) >= 2 and all(
        isinstance(value, ast.Lambda)
        or (resolved(value, tree) or "").startswith("operator.")
        for value in compared
    )


def _operator_branch(node: ast.AST) -> bool:
    """A branch on a range operator's name: ``op == "gte"``, ``case "gt":``."""
    if isinstance(node, ast.Compare):
        return (
            len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and any(
                isinstance(side, ast.Constant) and side.value in _RANGE_OPS
                for side in (node.left, *node.comparators)
            )
        )
    return (
        isinstance(node, ast.MatchValue)
        and isinstance(node.value, ast.Constant)
        and node.value.value in _RANGE_OPS
    )


_LINEAGE = re.compile(r"(?<![a-z])units?(?:_ids?)?\b|lineage")


def _mentions_lineage(node: ast.AST) -> bool:
    """Whether ``node`` reads unit ids (``text_unit_ids``, ``units``, a
    ``lineage``), not a word that only holds the letters (``community``)."""
    return _LINEAGE.search(ast.unparse(node)) is not None


def _lineage_dedup(node: ast.AST) -> bool:
    """Unit ids merged by hand: ``dict.fromkeys(...)`` over them (a literal
    tuple of field names is a catalog, not a lineage), or a loop that
    appends each one not yet seen."""
    if isinstance(node, ast.Call):
        return (
            ast.unparse(node.func) == "dict.fromkeys"
            and bool(node.args)
            and not isinstance(node.args[0], ast.Tuple | ast.List)
            and _mentions_lineage(node.args[0])
        )
    if not (isinstance(node, ast.For) and _mentions_lineage(node.iter)):
        return False
    for child in ast.walk(node):
        if (
            isinstance(child, ast.If)
            and isinstance(child.test, ast.Compare)
            and isinstance(child.test.ops[0], ast.NotIn)
            and any(
                isinstance(b, ast.Expr)
                and isinstance(b.value, ast.Call)
                and getattr(b.value.func, "attr", None) == "append"
                for b in child.body
            )
        ):
            return True
    return False


_VIEW = re.compile(r"(?:^|[._])view$")


def _sources_read(node: ast.AST) -> bool:
    """``x.sources`` of something that is not the answer view (``view``,
    ``self.view``, ``answer_view``)."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "sources"
        and not _VIEW.search(ast.unparse(node.value))
    )


# Past it, a code point is another script than ASCII's and Latin-1's (an
# ``ord(ch) < 128`` is an ASCII test, not a script range).
_LATIN_1 = 0xFF


def _codepoint_compare(node: ast.AST, tree: ast.Module) -> bool:
    """A script test by comparison: ``"\\u3040" <= ch <= "\\u309f"``, or
    ``0xAC00 <= ord(ch)``."""
    if not (
        isinstance(node, ast.Compare)
        and all(isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE) for op in node.ops)
    ):
        return False
    sides = (node.left, *node.comparators)
    if any(
        isinstance(s, ast.Constant)
        and isinstance(s.value, str)
        and len(s.value) == 1
        and ord(s.value) > _LATIN_1
        for s in sides
    ):
        return True
    return any(
        isinstance(s, ast.Constant)
        and isinstance(s.value, int)
        and not isinstance(s.value, bool)
        and s.value > _LATIN_1
        for s in sides
    ) and any(
        isinstance(s, ast.Call) and resolved(s.func, tree) == "ord" for s in sides
    )


# One end of a character range: a non-ASCII character, or its escape
# (``\x`` escapes name ASCII and Latin-1 controls, no script).
_END = (
    r"(?:[^\x00-\x7f]|\\u(?!00[0-7])[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}" r"|\\N\{[^}]+\})"
)
_SCRIPT_RANGE = re.compile(rf"\[(?:\\.|[^\]\\])*?{_END}-{_END}")
_DEGRADATION = "unified_kg_rag.domain.retrieval.degradation"
_SCRIPTS = "unified_kg_rag.shared.utils.scripts"
_WRITE_MODES = re.compile(r"[wax]")


def _file_write(node: ast.AST) -> bool:
    """A write that is not atomic: ``.write_text``/``.write_bytes``, or an
    ``open`` / ``Path.open`` for writing, appending or creating."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in ("write_text", "write_bytes"):
        return True
    if isinstance(func, ast.Name) and func.id == "open":
        positional = node.args[1] if len(node.args) > 1 else None
    elif isinstance(func, ast.Attribute) and func.attr == "open":
        positional = node.args[0] if node.args else None
    else:
        return False
    mode = next((k.value for k in node.keywords if k.arg == "mode"), positional)
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and bool(_WRITE_MODES.search(mode.value))
    )


def _path_move(node: ast.AST) -> bool:
    """``path.replace(target)`` / ``path.rename(target)``: one positional
    argument and none named (``str.replace`` takes two)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("replace", "rename")
        and len(node.args) == 1
        and not node.keywords
        and not isinstance(node.args[0], ast.Constant)
    )


GUARDS = [
    SourceGuard(
        name="G1 degradation ledger keys",
        use="DegradationLedger to write them, DegradationReport.of to read them",
        owners=("domain/retrieval/degradation.py",),
        find=any_of(
            string_literals("degraded", "degraded_error", "citations_unresolved"),
            references(
                f"{_DEGRADATION}.DEGRADED_KEY",
                f"{_DEGRADATION}.DEGRADED_ERROR_KEY",
                f"{_DEGRADATION}.UNRESOLVED_KEY",
            ),
        ),
        allowed={
            "adapters/search_strategies/iterative_search.py::"
            "IterativeSearchStrategy.asearch": (
                1,
                "the iterative run record's own 'degraded' flag "
                "(IterativeRun.degraded), not the ledger",
            ),
            "evaluation/iterative_statistics.py::iterative_statistics": (
                2,
                "counts that run-record flag",
            ),
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
            "application/ingestion",
            "application/prompts",
            "adapters/llm",
            "adapters/ingestion",
            "adapters/evaluators",
            "evaluation",
        ),
        allowed={
            "adapters/search_strategies/global_search.py::_json_string": (
                1,
                "decodes one salvaged string body of a broken map answer (MS "
                "GraphRAG's json_repair recovery), not an answer",
            ),
            "adapters/search_strategies/global_search.py::_unparsed_shape": (
                1,
                "names the parser's complaint for the log; the answer was read "
                "already",
            ),
            "adapters/search_strategies/structured_search.py::"
            "StructuredSearchStrategy._entity_result": (
                1,
                "a vertex's stored filters JSON, not model output",
            ),
            "adapters/ingestion/json_loader.py::JsonTextLoader.lazy_load": (
                1,
                "a corpus JSON file",
            ),
            "adapters/llm/response_cache.py::LLMResponseCache.open": (
                1,
                "the cache's own index file",
            ),
            "application/ingestion/cache_manager.py::CacheManager._deserialize_data": (
                1,
                "a stage cache entry the pipeline wrote",
            ),
            "application/ingestion/cache_manager.py::"
            "CacheManager._load_single_file_data": (
                1,
                "a stage cache file the pipeline wrote",
            ),
            "application/ingestion/cache_manager.py::CacheManager._load_chunked_data": (
                1,
                "a chunked stage cache file the pipeline wrote",
            ),
            "evaluation/benchmarks/musique.py::_items": (1, "the downloaded dataset"),
            "evaluation/evaluation_manager.py::EvaluationManager.build_run_manifest": (
                1,
                "the dataset file, hashed by content",
            ),
            "evaluation/saved_results.py::_read": (1, "saved result files"),
        },
    ),
    SourceGuard(
        name="G3 surfaces read the answer view",
        use="AnswerView (application/presentation/answer_view.py): its sources, "
        "written sections and QueryOutcome",
        owners=("application/presentation/answer_view.py",),
        find=any_of(
            nodes(_sources_read),
            key_reads("sources", "error_class", "error_type"),
            # The answer's error read off its raw metadata (QueryOutcome
            # reads it once).
            key_reads("error", of=re.compile(r"metadata")),
            string_literals("degraded", "degraded_error", "citations_unresolved"),
        ),
        within=_SURFACES,
        allowed={},
    ),
    SourceGuard(
        name="G4 Unicode script ranges",
        use="shared.utils.scripts (script_of, letter_class, the Script table)",
        owners=("shared/utils/scripts.py",),
        find=any_of(
            string_values(_SCRIPT_RANGE),
            module_nodes(_codepoint_compare),
            lines_matching(
                re.compile(
                    r"0x[0-9A-Fa-f]{4,5}\s*,\s*0x[0-9A-Fa-f]{4,5}"
                    r"|\\U[0-9A-Fa-f]{8}-\\U[0-9A-Fa-f]{8}"
                )
            ),
        ),
        allowed={
            # Ports of Lucene's analyzer tables: what the store does, not a
            # script classification.
            "adapters/local/search_engine/analyzers.py::_ALPHABETIC_SYMBOLS": (
                4,
                "the enclosed and squared letters Lucene's ASCII folding maps",
            ),
            "adapters/local/search_engine/analyzers.py::_HEBREW": (
                1,
                "the Hebrew letters Lucene's standard tokenizer keeps whole",
            ),
            "adapters/local/search_engine/analyzers.py::_SCANNED": (
                1,
                "the planes the standard tokenizer scans",
            ),
            "adapters/local/search_engine/analyzers.py::_ASTRAL": (
                2,
                "the supplementary planes, one token each",
            ),
            "domain/models/seed_graph.py::_SURROGATE": (
                1,
                "refuses lone surrogates, an encoding error",
            ),
            "domain/retrieval/citations.py::_RAW_NUMBER": (
                1,
                "a citation marker's digits, full-width ones too",
            ),
            "evaluation/answer_match_evaluator.py::_LIST_ITEM": (
                1,
                "circled list numbers (①-⑳), a list marker",
            ),
        },
    ),
    SourceGuard(
        name="G5 stored file name and path",
        use="domain.retrieval.source_fields.file_name / safe_relative_path",
        owners=("domain/retrieval/source_fields.py",),
        find=key_reads("file_name", "file_path"),
        allowed={
            "application/cli/console/stages.py::_failed_items": (
                1,
                "a stage's failed file entries (pipeline metrics), not a stored "
                "item",
            ),
            "application/ingestion/summary_tree_stage.py::SummaryTreeStage._tree_of": (
                2,
                "keys a document's tree by its stored file_path; that host path "
                "is the trees' partition and hashed into their node ids, so "
                "reading source_fields.relative_path instead re-keys every stored "
                "tree (left to the wave-2 re-key bundle)",
            ),
            "adapters/doc_status/dynamodb.py::DynamoDBDocStatusStore._serialize": (
                1,
                "a registry record's own fields, written to its item",
            ),
            "evaluation/citation_scoring.py::_file_names": (
                4,
                "reads a Citation's own documents (CitedDocument.file_name), "
                "built by citations.py",
            ),
        },
    ),
    SourceGuard(
        name="G6 stored id lists made lists",
        use="domain.retrieval.source_fields.id_list / listed",
        owners=("domain/retrieval/source_fields.py", "domain/retrieval/filters.py"),
        find=nodes(_rewrap),
        within=(*_READ_PATHS, "adapters/storage"),
        allowed={
            "adapters/retrieval/base.py::scope_filters": (
                2,
                "filter values (a value or a list), intersected with a scope's",
            ),
            "adapters/retrieval/base.py::"
            "BaseSearchStrategy._expand_over_passing_relationships": (
                1,
                "a caller's id filter value",
            ),
            "adapters/storage/neptune_codec.py::property_values": (
                1,
                "the codec's reading of a Gremlin valueMap (lists by design)",
            ),
            "adapters/storage/sqlite_graph_indexer.py::vertex_value_map": (
                1,
                "the SQLite graph's valueMap, as Gremlin returns it",
            ),
            "application/retrieval/rag_chain.py::_string_leaves": (
                1,
                "the model-written strings the output guardrail screens",
            ),
            "domain/retrieval/graph_query.py::_refuse_hidden_tokens": (
                1,
                "a structured plan's operand (one value or several)",
            ),
            "domain/retrieval/graph_query.py::literal_conditions": (
                1,
                "a structured plan's operand (one value or several)",
            ),
            "evaluation/answer_match_evaluator.py::answer_aliases": (
                1,
                "a dataset's answer_aliases (a string or a list of strings), "
                "read once for the evaluator and the run's language check",
            ),
            "adapters/search_strategies/structured_search.py::"
            "StructuredSearchStrategy._request": (
                1,
                "a plan's allowed ids argument",
            ),
            "application/retrieval/rag_chain.py::_model_texts": (
                1,
                "the model-written texts the output guardrail screens",
            ),
            "domain/retrieval/graph_query.py::_check_value_kind": (
                1,
                "a structured plan's operand",
            ),
        },
    ),
    SourceGuard(
        name="G7 atomic file writes",
        use="shared.utils.files.atomic_write_text",
        owners=("shared/utils/files.py",),
        find=any_of(
            calls("os.replace", "os.rename", "shutil.move"),
            nodes(_path_move),
            nodes(_file_write),
        ),
        allowed={
            "application/cli/console/server.py::create_app.upload": (
                2,
                "streams each uploaded file into a temporary sibling and replaces "
                "the file with it once every file is staged (one upload, all or "
                "none)",
            ),
            "application/cli/console/jobs.py::JobManager._launch": (
                1,
                "a job's append-only log, which the console tails while it runs",
            ),
        },
    ),
    SourceGuard(
        name="G8 script counting",
        use="domain.languages (one detector for queries and corpora)",
        owners=("domain/languages", "shared/utils/scripts.py"),
        find=any_of(
            lines_matching(re.compile(r"\bScriptCounts\b|\bcount_letters\(")),
            calls_mentioning(
                "collections.Counter",
                f"{_SCRIPTS}.script_of",
                f"{_SCRIPTS}.letter_class",
            ),
        ),
    ),
    SourceGuard(
        name="G9 filter comparisons",
        use="domain.retrieval.filters (Term.holds, Range.holds, bound_holds) or "
        "a store's filter compiler",
        owners=("domain/retrieval/filters.py",),
        find=any_of(module_nodes(_comparison_table), nodes(_operator_branch)),
        allowed={
            "adapters/local/search_engine/query.py::_RANGE_OPERATORS": (
                1,
                "the embedded engine's OpenSearch range query, the store itself",
            ),
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
            "application/storage",
            "domain/retrieval",
        ),
        allowed={
            "adapters/search_strategies/global_search.py::_json_string": (
                1,
                "a salvaged map answer string (G2)",
            ),
            "adapters/search_strategies/global_search.py::_unparsed_shape": (
                1,
                "a map answer's shape for the log (G2)",
            ),
            "adapters/search_strategies/structured_search.py::"
            "StructuredSearchStrategy._entity_result": (
                1,
                "a vertex's stored filters JSON, a mapping, not an edge list",
            ),
            "domain/retrieval/graph_query.py::seed_properties": (
                1,
                "a seed entity's properties JSON, a mapping",
            ),
        },
    ),
    SourceGuard(
        name="G11 lineage merging",
        use="domain.retrieval.lineage.merge (first_seen or support order)",
        owners=("domain/retrieval/lineage.py",),
        find=nodes(_lineage_dedup),
        within=_READ_PATHS,
        allowed={
            "evaluation/source_resolver.py::TextUnitFileResolver.aresolve": (
                1,
                "the ids to look up, deduplicated before one batched read",
            ),
            "application/retrieval/graph_lookup.py::text_unit_file_names": (
                1,
                "the ids asked, deduplicated before batched reads",
            ),
            "adapters/search_strategies/lightrag_search.py::"
            "LightRAGSearchStrategy._passing_linked_chunk_ids": (
                1,
                "one check round's ids: each lineage up to past its deepest "
                "unchecked chunk, not a merged lineage",
            ),
            "domain/retrieval/ppr.py::PassageGraph.link_passages": (
                1,
                "one entity's passages, each linked by one edge",
            ),
        },
    ),
]


def _id(guard: SourceGuard) -> str:
    return guard.name.split(" ", 1)[0]


@pytest.mark.parametrize("guard", GUARDS, ids=_id)
def test_no_copy_outside_the_owner(guard: SourceGuard) -> None:
    copies = [str(hit) for hit in guard.copies()]
    assert not copies, f"{guard.name}: use {guard.use}: {copies}"


@pytest.mark.parametrize("guard", GUARDS, ids=_id)
def test_every_allowlist_entry_is_still_needed(guard: SourceGuard) -> None:
    assert guard.stale() == [], f"{guard.name}: shrink or drop these entries"


@pytest.mark.parametrize("guard", GUARDS, ids=_id)
def test_the_pattern_matches_its_own_owners(guard: SourceGuard) -> None:
    # A finder that misses the owner's own form would miss a copy of it.
    assert guard.unmatched_owners() == [], f"{guard.name}: widen the finder"


@pytest.mark.parametrize("guard", GUARDS, ids=_id)
def test_every_owner_within_and_allowed_path_exists(guard: SourceGuard) -> None:
    # A renamed owner or a mistyped within path would leave the guard
    # reading no module as it.
    assert guard.missing_paths() == [], guard.name


_G4 = next(g for g in GUARDS if g.name.startswith("G4 ")).find


@pytest.mark.parametrize(
    ("finder", "source", "found"),
    [
        (string_literals("degraded"), 'x = m.get("degraded")', ["'degraded'"]),
        (string_literals("degraded"), 'def f():\n    """degraded"""\n', []),
        # The key read through its constant, however imported.
        (
            references(f"{_DEGRADATION}.DEGRADED_KEY"),
            f"from {_DEGRADATION} import DEGRADED_KEY as K\nm.get(K)\n",
            ["K"],
        ),
        (calls("json.loads"), "import json\njson.loads(s)\n", ["json.loads"]),
        (calls("json.loads"), "import json as j\nj.loads(s)\n", ["j.loads"]),
        (calls("json.loads"), "from json import loads as l\nl(s)\n", ["l"]),
        (calls("os.replace"), "import os\nos.replace(a, b)\n", ["os.replace"]),
        (calls("os.replace"), "from os import replace\nreplace(a, b)\n", 1),
        (calls("shutil.move"), "import shutil\nshutil.move(a, b)\n", 1),
        (nodes(_path_move), "tmp.replace(path)\n", 1),
        (nodes(_path_move), 's.replace("a", "b")\n', []),
        (nodes(_file_write), 'p.write_text(t)\nopen(p, "w")\np.open("ab")\n', 3),
        (nodes(_file_write), 'open(p)\np.open()\nf(mode="after")\n', []),
        (key_reads("file_name"), 'a.get("file_name")\nb["file_name"]\n', 2),
        (key_reads("file_name"), 'b["file_name"] = 1\n', []),
        (key_reads("file_name"), 'getattr(d, "file_name", None)\n', 1),
        (key_reads("file_name"), 'for k in ("file_name", "x"):\n    m.get(k)\n', 1),
        (
            key_reads("error", of=re.compile("metadata")),
            'out.metadata.get("error")\nresult["error"]\n',
            1,
        ),
        (nodes(_rewrap), "y = x if isinstance(x, list) else [x]\n", 1),
        (nodes(_rewrap), "y = [x] if isinstance(x, str) else list(x)\n", 1),
        (nodes(_rewrap), "if not isinstance(x, list):\n    x = [x]\n", 1),
        (nodes(_rewrap), "y = x if isinstance(x, list) else list(x)\n", []),
        (
            module_nodes(_comparison_table),
            'import operator as op\nT = {"gt": lambda a: a, "lt": op.lt}\n',
            1,
        ),
        (
            module_nodes(_comparison_table),
            'from operator import ge, le\nT = {"gte": ge, "lte": le}\n',
            1,
        ),
        (module_nodes(_comparison_table), 'q = {"gte": start, "lt": end}\n', []),
        (nodes(_operator_branch), 'if op == "gt":\n    pass\n', 1),
        (nodes(_operator_branch), 'match op:\n    case "lte":\n        pass\n', 1),
        (nodes(_lineage_dedup), "u = dict.fromkeys(r.text_unit_ids)\n", 1),
        (nodes(_lineage_dedup), "u = dict.fromkeys(units)\n", 1),
        (nodes(_lineage_dedup), 'F = dict.fromkeys(("id", "text_unit_ids"))\n', []),
        (nodes(_lineage_dedup), "c = dict.fromkeys(r.community for r in rs)\n", []),
        (
            nodes(_lineage_dedup),
            "for u in r.text_unit_ids:\n    if u not in out:\n        out.append(u)\n",
            1,
        ),
        (nodes(_sources_read), "a = output.sources\nb = view.sources\n", 1),
        (
            nodes(_sources_read),
            "a = preview.sources\nb = self.answer_view.sources\n",
            1,
        ),
        (_G4, 'R = "[\\u3040-\\u309f]"\n', 1),
        (_G4, 'R = "[\u0400-\u04ff]"\n', 1),
        (_G4, 'R = re.compile("[ก-๙]+")\n', 1),
        (_G4, 'R = re.compile("(?<![가-힣])없")\n', 1),
        (_G4, 'R = r"[#\\u3040-\\u309f]"\n', 1),
        (_G4, "S = ((0x0E00, 0x0E7F),)\n", 1),
        (_G4, "ok = 0xAC00 <= ord(ch) <= 0xD7A3\n", 1),
        (_G4, 'ok = "\\u3040" <= ch <= "\\u309f"\n', 1),
        (_G4, 'C = "[\\x00-\\x1f\\x7f]"\nok = ord(ch) < 128 and ch <= "z"\n', []),
        (
            calls_mentioning("collections.Counter", f"{_SCRIPTS}.script_of"),
            f"from collections import Counter as C\nfrom {_SCRIPTS} import script_of"
            " as so\nC(so(c) for c in t)\n",
            1,
        ),
    ],
)
def test_each_pattern_catches_the_forms_copies_took(
    finder: object, source: str, found: list[str] | int
) -> None:
    hits = matches(finder, source)  # type: ignore[arg-type]
    assert (len(hits) if isinstance(found, int) else hits) == found


def test_a_guard_counts_copies_and_reports_entries_nothing_needs() -> None:
    owner = "domain/retrieval/degradation.py"
    finder = string_literals("citations_unresolved")
    # Its owner holds the key: no copy; an entry no match needs is stale.
    guard = SourceGuard(
        name="probe",
        use="-",
        owners=(owner,),
        find=finder,
        allowed={"nowhere.py::f": (1, "gone")},
    )
    assert guard.copies() == []
    assert guard.stale() == ["nowhere.py::f: 1 allowed, 0 found"]
    assert guard.missing_paths() == ["nowhere.py"]
    # Without its owner, the owner's own use is a copy, keyed by the name
    # its module-level assignment binds.
    bare = SourceGuard(name="probe", use="-", owners=(), find=finder)
    keys = {hit.key for hit in bare.copies()}
    assert keys == {f"{owner}::UNRESOLVED_KEY"}
    # An entry admits as many matches as it counts: one more is a copy, one
    # fewer a stale count.
    (key,) = keys
    assert SourceGuard("probe", "-", (), finder, {key: (1, "-")}).copies() == []
    assert SourceGuard("probe", "-", (), finder, {key: (0, "-")}).copies()
    assert SourceGuard("probe", "-", (), finder, {key: (2, "-")}).stale()
    # A directory owner covers the modules under it; an owner the pattern
    # misses is reported.
    folder = SourceGuard(
        name="probe", use="-", owners=("domain/retrieval",), find=finder
    )
    assert folder.copies() == [] and folder.unmatched_owners() == []
    blind = SourceGuard(name="probe", use="-", owners=(owner,), find=calls("x.y"))
    assert blind.unmatched_owners() == [owner]
    typo = SourceGuard(name="probe", use="-", owners=(), find=finder, within=("nope",))
    assert typo.missing_paths() == ["nope"]


def test_module_scopes_are_keyed_by_the_name_they_bind() -> None:
    source = (
        "A = 1\n"
        "B: int = (\n    2\n)\n"
        "class C:\n    D = 3\n    def m(self):\n        e = 4\n"
        "print(5)\n"
    )
    assert _scopes(ast.parse(source)) == {
        1: "A",
        2: "B",
        3: "B",
        4: "B",
        5: "C",
        6: "C.D",
        7: "C.m",
        8: "C.m",
    }
