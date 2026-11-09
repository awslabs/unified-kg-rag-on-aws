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
The guards are tripwires for the forms of a copy they know, not coverage: a
copy spelled another way passes them, and review owns it.
CONTRIBUTING.md ("Single-owner guards") says how to add one.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

import pytest

from tests.fixtures.source_guards import (
    PACKAGE,
    Source,
    SourceGuard,
    _bindings,
    _names_in,
    _scopes,
    any_of,
    calls,
    calls_mentioning,
    imported,
    key_reads,
    lines_matching,
    matches,
    method_calls,
    module_nodes,
    nodes,
    references,
    resolved,
    string_literals,
    string_values,
    uses,
)
from unified_kg_rag.shared.utils.scripts import SCRIPTS

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


def _type_test(node: ast.AST) -> bool:
    """``isinstance(x, ...)``, ``type(x) is ...`` (``==``, ``is not``,
    ``!=``), or its ``not``."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node = node.operand
    if isinstance(node, ast.Call):
        return getattr(node.func, "id", None) == "isinstance" and len(node.args) == 2
    return (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Is | ast.IsNot | ast.Eq | ast.NotEq)
        and any(
            isinstance(side, ast.Call) and getattr(side.func, "id", None) == "type"
            for side in (node.left, *node.comparators)
        )
    )


def _one_item(node: ast.AST | None) -> bool:
    return isinstance(node, ast.List | ast.Set | ast.Tuple) and len(node.elts) == 1


# The types a value-or-list is told apart by.
_SEQUENCES = re.compile(
    r"\b(?:list|tuple|set|frozenset|str|bytes|Sequence|Iterable|Collection)\b"
)


def _tested(test: ast.AST) -> str | None:
    """The value a type test against a sequence or string type reads (``v``
    of ``isinstance(v, str)``; none of ``isinstance(v, dict)``)."""
    if isinstance(test, ast.UnaryOp):
        test = test.operand
    if isinstance(test, ast.Call):
        value, kind = test.args
    else:
        assert isinstance(test, ast.Compare)
        typed = [s for s in (test.left, *test.comparators) if isinstance(s, ast.Call)]
        others = [s for s in (test.left, *test.comparators) if s not in typed]
        if not (typed and typed[0].args and others):
            return None
        value, kind = typed[0].args[0], others[0]
    return ast.unparse(value) if _SEQUENCES.search(ast.unparse(kind)) else None


def _wraps(statement: ast.stmt, value: str | None) -> bool:
    """``x = [v]``, ``return [v]``, or either of ``... if ... else [v]``,
    for the tested ``value`` ``v``."""
    if not isinstance(statement, ast.Assign | ast.AnnAssign | ast.Return):
        return False
    bound = statement.value
    options = (bound.body, bound.orelse) if isinstance(bound, ast.IfExp) else (bound,)
    return any(
        isinstance(option, ast.List | ast.Set | ast.Tuple)
        and len(option.elts) == 1
        and ast.unparse(option.elts[0]) == value
        for option in options
    )


def _rewrap(node: ast.AST) -> bool:
    """A value-or-list made a list on a type test: ``x if isinstance(x,
    list) else [x]`` either way round, an ``if`` on the test that binds or
    returns ``[x]`` in either branch, or a ``case str():`` that does."""
    if isinstance(node, ast.IfExp):
        return _type_test(node.test) and (
            _one_item(node.body) or _one_item(node.orelse)
        )
    if isinstance(node, ast.If):
        value = _tested(node.test) if _type_test(node.test) else None
        return value is not None and any(
            _wraps(s, value) for s in (*node.body, *node.orelse)
        )
    if not isinstance(node, ast.Match):
        return False
    subject = ast.unparse(node.subject)
    return any(
        isinstance(case.pattern, ast.MatchClass)
        and any(_wraps(s, subject) for s in case.body)
        for case in node.cases
    )


_RANGE_OPS = ("gt", "gte", "lt", "lte")
# The ``operator`` functions a range operator names.
_COMPARISONS = ("gt", "ge", "lt", "le")


def _compares(value: ast.AST, source: Source) -> bool:
    """A comparison: a lambda, an ``operator`` function however imported,
    or the name of one (``"ge"``, for ``getattr(operator, ...)``)."""
    return (
        isinstance(value, ast.Lambda)
        or (resolved(value, source) or "").startswith("operator.")
        or (isinstance(value, ast.Constant) and value.value in _COMPARISONS)
    )


def _comparison_table(node: ast.AST, source: Source) -> bool:
    """A mapping of range operators to how they compare, as a literal or a
    ``dict(gt=...)`` call."""
    if isinstance(node, ast.Dict):
        pairs = {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant)
        }
    elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        pairs = {k.arg: k.value for k in node.keywords if k.arg}
    else:
        return False
    compared = [pairs[op] for op in _RANGE_OPS if op in pairs]
    return len(compared) >= 2 and all(_compares(v, source) for v in compared)


def _names_an_operator(node: ast.AST) -> bool:
    """A range operator's name, or a literal collection holding one."""
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return any(map(_names_an_operator, node.elts))
    return isinstance(node, ast.Constant) and node.value in _RANGE_OPS


def _operator_branch(node: ast.AST) -> bool:
    """A branch on a range operator's name: ``op == "gte"``, ``op in ("gt",
    "gte")``, ``case "gt":``."""
    if isinstance(node, ast.Compare):
        return (
            len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq | ast.NotEq | ast.In | ast.NotIn)
            and any(map(_names_an_operator, (node.left, *node.comparators)))
        )
    return (
        isinstance(node, ast.MatchValue)
        and isinstance(node.value, ast.Constant)
        and node.value.value in _RANGE_OPS
    )


_LINEAGE = re.compile(r"(?<![a-z])(?:units?(?:_ids?)?|chunks|chunk_ids)\b|lineage")


def _mentions_lineage(node: ast.AST) -> bool:
    """Whether ``node`` reads unit or chunk ids (``text_unit_ids``,
    ``units``, ``source_chunks``, a ``lineage``), not a word that only
    holds the letters (``community``)."""
    return _LINEAGE.search(ast.unparse(node)) is not None


def _grows(statement: ast.AST) -> bool:
    """``out.append(u)`` or ``seen.add(u)``."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and getattr(statement.value.func, "attr", None) in ("append", "add")
    )


def _keeps_first_seen(loop: ast.For) -> bool:
    """A loop that keeps each id once: ``if u not in out: out.append(u)``,
    or ``if u in seen: continue`` before an append or add."""
    grows = any(map(_grows, ast.walk(loop)))
    for child in ast.walk(loop):
        if not (isinstance(child, ast.If) and isinstance(child.test, ast.Compare)):
            continue
        op = child.test.ops[0]
        if isinstance(op, ast.NotIn) and any(map(_grows, child.body)):
            return True
        if (
            isinstance(op, ast.In)
            and grows
            and any(isinstance(b, ast.Continue) for b in child.body)
        ):
            return True
    return False


def _over_lineage(node: ast.DictComp | ast.SetComp, element: ast.AST) -> bool:
    """Whether a comprehension's ``element`` is the id it draws from a
    lineage (``u`` of ``for u in r.text_unit_ids``)."""
    return any(
        _mentions_lineage(g.iter) and ast.unparse(g.target) == ast.unparse(element)
        for g in node.generators
    )


def _lineage_support(node: ast.AST, source: Source) -> bool:
    """Unit ids counted to rank them by support: ``Counter(...)`` over a
    lineage, or ``.update(...)`` of one into a name bound to a Counter."""
    if not (isinstance(node, ast.Call) and node.args):
        return False
    if resolved(node.func, source) == "collections.Counter":
        return _mentions_lineage(node.args[0])
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "update"
        and isinstance(func.value, ast.Name)
        and any(
            isinstance(v, ast.Call)
            and resolved(v.func, source) == "collections.Counter"
            for v in _bindings(source).get(func.value.id, ())
        )
        and _mentions_lineage(node.args[0])
    )


def _lineage_dedup(node: ast.AST) -> bool:
    """Unit ids merged by hand: ``fromkeys`` over them (``dict``'s or
    ``OrderedDict``'s; a literal tuple of field names is a catalog, not a
    lineage), ``{u: None ...}`` over them or a set of them made a list
    (``sorted({...})``), or a loop that keeps each one not yet seen."""
    if isinstance(node, ast.Call) and node.args:
        first = node.args[0]
        if ast.unparse(node.func) in ("sorted", "list"):
            # A set of the ids made a list again.
            return isinstance(first, ast.SetComp) and _over_lineage(first, first.elt)
        return (
            ast.unparse(node.func).endswith("fromkeys")
            and not isinstance(first, ast.Tuple | ast.List)
            and _mentions_lineage(first)
        )
    if isinstance(node, ast.DictComp):
        # ``{u: None for ... in lineage}``: dict.fromkeys spelled out.
        return isinstance(node.value, ast.Constant) and _over_lineage(node, node.key)
    return (
        isinstance(node, ast.For)
        and _mentions_lineage(node.iter)
        and _keeps_first_seen(node)
    )


_VIEW = re.compile(r"(?:^|[._])view$")


def _sources_read(node: ast.AST) -> bool:
    """``x.sources`` of something that is not the answer view (``view``,
    ``self.view``, ``answer_view``)."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "sources"
        and not _VIEW.search(ast.unparse(node.value))
    )


# The script table's letter blocks: a code point in (or one past the end
# of) one is a script's. Past Latin-1 only: an ``ord(ch) < 128`` is an
# ASCII test, ``"a" <= ch <= "z"`` not a script range. A bare number below
# the Latin and IPA blocks' end is as likely an HTTP status (``400 <=
# status < 500``): only a character spells a code point there.
_LATIN_1 = 0xFF
_SMALL_NUMBERS = 0x2FF


def _bounded_scripts(cp: int, floor: int = _LATIN_1) -> frozenset[str]:
    """The scripts whose letters ``cp`` bounds: in one of the script's
    blocks, or one before or past it (``< 0xD7A4``, ``> 0xABFF``), and past
    ``floor``."""
    if cp <= floor:
        return frozenset()
    return frozenset(
        script.name
        for script in SCRIPTS
        if any(lo - 1 <= cp <= hi + 1 for lo, hi in script.ranges)
    )


def _a_script_bound(cp: int, floor: int = _LATIN_1) -> bool:
    """Whether ``cp`` bounds a script's letters (:func:`_bounded_scripts`)."""
    return bool(_bounded_scripts(cp, floor))


def _script_bounds(bounds: list[int], of_characters: bool) -> bool:
    """Whether ``bounds`` are a script range's ends. Bounds a character is
    held to are, when each bounds some script's letters, though not one
    script's (``0x3040 <= ord(ch) <= 0x9FFF``, kana to Han: the "is it
    CJK" check). Bare numbers must each bound the same script's: a script
    range's ends are one script's (``0xAC00 <= cp <= 0xD7A3``), while two
    round numbers each near some block are a number range (``1024 <= port
    <= 49151``, ``1000 <= year <= 2999``).

    Known limits, both of bare numbers: a number range whose ends both fall
    in (or one past) one script's blocks is flagged though it is no script
    range: sizes ``1024 <= size <= 8192`` (one past Greek and Greek
    Extended), ``(16384, 131072)``, a sample rate ``44100 <= rate <= 48000``
    (Han and Hangul are wide). A real one in the package gets an
    ``allowed`` entry with its reason; the ends cannot be told apart from
    a script's without the Unicode block-edge table, which ``SCRIPTS`` is
    not. And a pure-number span across scripts (``(0x3040, 0x9FFF)``,
    ``range(0x3040, 0xA000)``) is missed: spelled with characters or
    ``ord`` it is caught."""
    if of_characters:
        return all(map(_a_script_bound, bounds))
    return bool(frozenset.intersection(*map(_bounded_scripts, bounds)))


def _code_point(node: ast.AST, source: Source) -> int | None:
    """The code point a constant names: a one-character string, ``chr(int)``,
    ``ord(char)``, or an int past :data:`_SMALL_NUMBERS`; ``-1`` for a
    smaller int (a constant, but no code point)."""
    if isinstance(node, ast.Call) and len(node.args) == 1:
        if resolved(node.func, source) in ("chr", "ord"):
            inner = node.args[0]
            if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
                return inner.value
            return _code_point(inner, source)
        return None
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        return None
    if isinstance(node.value, int):
        return node.value if node.value > _SMALL_NUMBERS else -1
    if isinstance(node.value, str) and len(node.value) == 1:
        return ord(node.value)
    return None


def _a_character(node: ast.AST, source: Source) -> bool:
    """Whether ``node`` is a character's code point or the character
    itself: ``ord(...)``, a name bound to one, a one-character string."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and len(node.value) == 1
    if isinstance(node, ast.Name):
        return any(_a_character(v, source) for v in _bindings(source).get(node.id, ()))
    return isinstance(node, ast.Call) and resolved(node.func, source) in (
        "ord",
        "chr",
    )


def _code_point_range(node: ast.AST, source: Source) -> bool:
    """A script range spelled in code: a comparison, a pair or a
    ``range(...)`` whose bounds are a script range's ends
    (:func:`_script_bounds`: ``0xAC00 <= cp <= 0xD7A3``, ``(12352, 12447)``,
    ``range(44032, 55204)``, ``chr(0x3040) <= ch``, ``0x3040 <= ord(ch) <=
    0x9FFF``), or a character compared with one script bound
    (``0xAC00 <= ord(ch)``, ``ch >= "\\u3040"``)."""
    if isinstance(node, ast.Compare):
        if not all(
            isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE) for op in node.ops
        ):
            return False
        sides = (node.left, *node.comparators)
        bounds = [cp for s in sides if (cp := _code_point(s, source)) is not None]
        if len(bounds) == 1:
            return _a_script_bound(bounds[0]) and any(
                _a_character(s, source) for s in sides if _code_point(s, source) is None
            )
        return len(bounds) > 1 and _script_bounds(
            bounds, any(_a_character(s, source) for s in sides)
        )
    if isinstance(node, ast.Tuple | ast.List) and len(node.elts) >= 2:
        # A block's bounds first: ``(0x3040, 0x309F)``, ``(1024, 1327, "cyrillic")``.
        pair = node.elts[:2]
    elif (
        isinstance(node, ast.Call)
        and resolved(node.func, source) == "range"
        and len(node.args) >= 2
    ):
        pair = node.args[:2]
    else:
        return False
    lo, hi = (_code_point(end, source) for end in pair)
    return (
        lo is not None
        and hi is not None
        and lo < hi
        and _script_bounds([lo, hi], any(_a_character(e, source) for e in pair))
    )


# A character class of a pattern, and a range in it whose ends are a
# character or its escape (``\x`` escapes name ASCII and Latin-1 only).
_CLASS = re.compile(r"\[(?:\\.|[^\]\\])*\]")
_END = r"(\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}|\\N\{[^}]+\}|[^\x00-\x7f])"
_CLASS_RANGE = re.compile(rf"{_END}-{_END}")


def _escaped(end: str) -> int | None:
    if end.startswith(("\\u", "\\U")):
        return int(end[2:], 16)
    if end.startswith("\\N{"):
        try:
            return ord(unicodedata.lookup(end[3:-1]))
        except KeyError:
            return None
    return ord(end)


def _script_class(match: re.Match[str]) -> bool:
    """A character class with a range both of whose ends bound a script's
    letters (``[\\u3040-\\u309f]``, ``[가-힣]``; ``[\\u200b-\\u200f]``, a
    format-character range, is none)."""
    return any(
        all(
            (cp := _escaped(end)) is not None and _a_script_bound(cp)
            for end in pair.groups()
        )
        for pair in _CLASS_RANGE.finditer(match.group(0))
    )


_DEGRADATION = "unified_kg_rag.domain.retrieval.degradation"
_DEGRADATION_KEYS = tuple(
    f"{_DEGRADATION}.{key}"
    for key in ("DEGRADED_KEY", "DEGRADED_ERROR_KEY", "UNRESOLVED_KEY")
)
_IDENTITY = "unified_kg_rag.domain.ingestion.document_identity"
_SCRIPTS = "unified_kg_rag.shared.utils.scripts"
_WRITE_MODES = re.compile(r"[wax+]")
# A mode string, not a member's name (``ZipFile.open("word/document.xml")``).
_MODE = re.compile(r"[rwxabtU+]+")
# The module-level ``open``s: each takes its file first, its mode second.
_OPENERS = (
    "open",
    "builtins.open",
    "io.open",
    "codecs.open",
    "gzip.open",
    "bz2.open",
    "lzma.open",
    "tarfile.open",
    "zipfile.ZipFile",
    "os.fdopen",
)
_COPIES = ("shutil.copyfile", "shutil.copy", "shutil.copy2")
_OS_WRITE_FLAGS = {f"os.{flag}" for flag in ("O_WRONLY", "O_RDWR", "O_APPEND")}


def _writes(mode: ast.AST | None) -> bool:
    """Whether a file opened in ``mode`` may be written: a literal mode
    that writes, appends, creates or updates (``+``), or one chosen at run
    time; reading is the default."""
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return bool(_WRITE_MODES.search(mode.value))
    return True


def _file_write(node: ast.AST, source: Source) -> bool:
    """A write that is not atomic: ``.write_text``/``.write_bytes``, a copy
    into place, an ``os.open`` for writing, or a file opened to write by a
    module-level ``open`` (mode second) or a value's ``.open`` (a path's:
    mode first; another's, as a zip member's, after its name)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in ("write_text", "write_bytes"):
        return True
    name = resolved(func, source)
    if name in _COPIES:
        return True
    if name == "os.open":
        return bool(
            _OS_WRITE_FLAGS & set().union(*(_names_in(a, source) for a in node.args))
        )
    keyword = next((k.value for k in node.keywords if k.arg == "mode"), None)
    args = node.args
    if name in _OPENERS:
        return _writes(keyword or (args[1] if len(args) > 1 else None))
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "open"
        and not imported(func.value, source)
    ):
        return False
    first = args[0] if args else None
    if keyword is None and not (
        first is None
        or (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and _MODE.fullmatch(first.value)
        )
    ):
        return _writes(args[1] if len(args) > 1 else None)
    return _writes(keyword or first)


def _path_move(node: ast.AST, source: Source) -> bool:
    """``path.replace(target)`` / ``path.rename(target)``: one positional
    argument and none named (``str.replace`` takes two), on a value, not a
    module (``dataclasses.replace(cfg)``)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("replace", "rename")
        and len(node.args) == 1
        and not node.keywords
        and not isinstance(node.args[0], ast.Constant)
        and not imported(node.func.value, source)
    )


def _counted_into(node: ast.AST, source: Source) -> bool:
    """A count kept by hand, keyed by a script: ``counts[script_of(c)] +=
    1``."""
    return (
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Subscript)
        and bool(_names_in(node.target.slice, source) & set(_SCRIPT_READERS))
    )


_SCRIPT_READERS = (f"{_SCRIPTS}.script_of", f"{_SCRIPTS}.letter_class")
# The decoders of a string into data (G2, G10): a denylist of the modules
# that only decode, whole, and of the decoding members of those that also
# encode, so a dump, an exception type or an annotation is no decoder
# (``yaml.safe_dump``, ``json.decoder.JSONDecodeError``, ``yaml.Node``).
# Any use of one in a guarded directory is a copy or an allowlist entry;
# the list names decoders, not the forms a call of one takes, so it grows
# only with a new decoding library.
_DECODERS = (
    "json.loads",
    "json.JSONDecoder",
    *(
        f"json.decoder.{name}"
        for name in (
            "JSONDecoder",
            "JSONObject",
            "JSONArray",
            "scanstring",
            "py_scanstring",
        )
    ),
    "json.scanner",
    "ast.literal_eval",
    *(
        f"yaml.{kind}load{every}"
        for kind in ("", "safe_", "full_", "unsafe_")
        for every in ("", "_all")
    ),
    # ruamel's decoder is the round-trip object's ``.load``: a method no
    # dotted name reaches, so the object it is a member of.
    "ruamel.yaml.YAML",
    "orjson.loads",
    "ujson.loads",
    "ujson.load",
    "simplejson.loads",
    "simplejson.JSONDecoder",
    "rapidjson.loads",
    "rapidjson.Decoder",
    *(
        f"msgspec.{form}.{name}"
        for form in ("json", "yaml")
        for name in ("decode", "Decoder")
    ),
    "json5.loads",
    "hjson.loads",
    "demjson3.decode",
    "json_repair",
    "pydantic_core.from_json",
    "langchain_core.output_parsers.json",
    "langchain_core.output_parsers.pydantic",
    "langchain_core.output_parsers.JsonOutputParser",
    "langchain_core.output_parsers.SimpleJsonOutputParser",
    "langchain_core.output_parsers.PydanticOutputParser",
    "langchain_core.utils.json",
    "langchain.output_parsers",
)
# Pydantic's decoders of a JSON string, called on a model or an adapter.
_DECODER_METHODS = ("model_validate_json", "validate_json", "parse_raw")
_DECODING = any_of(
    uses(*_DECODERS),
    # json.load reads a file; over a string wrapped as one, a decoder.
    calls_mentioning(("json.load",), "io.StringIO", "io.BytesIO"),
    method_calls(*_DECODER_METHODS),
)


GUARDS = [
    SourceGuard(
        name="G1 degradation ledger keys",
        use="DegradationLedger to write them, DegradationReport.of to read them",
        owners=("domain/retrieval/degradation.py",),
        find=any_of(
            string_literals("degraded", "degraded_error", "citations_unresolved"),
            references(*_DEGRADATION_KEYS),
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
        find=_DECODING,
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
            key_reads("error", "error_message", of=re.compile(r"metadata")),
            string_literals("degraded", "degraded_error", "citations_unresolved"),
            references(*_DEGRADATION_KEYS),
        ),
        within=_SURFACES,
        allowed={},
    ),
    SourceGuard(
        name="G4 Unicode script ranges",
        use="shared.utils.scripts (script_of, letter_class, the Script table)",
        owners=("shared/utils/scripts.py",),
        find=any_of(
            string_values(_CLASS, _script_class),
            # A regex Unicode property (``\p{Han}``, ``\p{L}``).
            string_values(re.compile(r"\\[pP]\{")),
            module_nodes(_code_point_range),
            # A script read off a character's Unicode name ("HANGUL ...").
            calls("unicodedata.name"),
        ),
        allowed={
            # A port of Lucene's analyzer table: what the store does, not a
            # script classification.
            "adapters/local/search_engine/analyzers.py::_HEBREW": (
                2,
                "the Hebrew letters Lucene's standard tokenizer keeps whole",
            ),
        },
    ),
    SourceGuard(
        name="G5 stored file name and path",
        use="domain.retrieval.source_fields.file_name / safe_relative_path",
        owners=("domain/retrieval/source_fields.py",),
        find=any_of(
            key_reads("file_name", "file_path", "relative_path"),
            # A loaded Document holds its path under RELATIVE_PATH_KEY (its
            # owner is document_identity); a stored item under its
            # attributes, which the owner reads.
            key_reads(
                constants=(f"{_IDENTITY}.RELATIVE_PATH_KEY",),
                of=re.compile(r"attributes"),
            ),
        ),
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
            module_nodes(_path_move),
            module_nodes(_file_write),
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
            calls_mentioning(("collections.Counter", "sum", "len"), *_SCRIPT_READERS),
            module_nodes(_counted_into),
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
        find=_DECODING,
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
        find=any_of(nodes(_lineage_dedup), module_nodes(_lineage_support)),
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
            "adapters/search_strategies/lightrag_search.py::"
            "LightRAGSearchStrategy._lineages": (
                1,
                "one stored lineage sorted, so the weighted pick does not "
                "depend on how the graph was built; not merged with another",
            ),
            "evaluation/evaluation_manager.py::EvaluationManager._files_by_support": (
                1,
                "counts the files a source's units come from, to rank files, "
                "not units",
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
        # Any decoder of the denylist, however it is reached; once a chain.
        (_DECODING, "import orjson\norjson.loads(s)\n", ["orjson.loads"]),
        (_DECODING, "from json_repair import repair_json as r\nr(s)\n", ["r"]),
        (_DECODING, "from json.decoder import JSONDecoder\nJSONDecoder()\n", 1),
        (_DECODING, "import json\nmap(json.loads, rows)\n", ["json.loads"]),
        (_DECODING, "import yaml\nyaml.safe_load(s)\n", ["yaml.safe_load"]),
        (_DECODING, "TypeAdapter(list[int]).validate_json(s)\n", 1),
        (_DECODING, "import json\njson.dumps(v)\njson.load(fp)\n", []),
        # A module that also encodes: its decoders, not its encoders,
        # exception types or annotations.
        (_DECODING, "import yaml\nyaml.load(s, Loader=L)\nyaml.full_load_all(s)\n", 2),
        (_DECODING, "import msgspec\nmsgspec.json.decode(b)\n", 1),
        (_DECODING, "from pydantic_core import from_json\nfrom_json(s)\n", 1),
        (
            _DECODING,
            "import yaml, orjson, msgspec, pydantic_core\n"
            "yaml.safe_dump(d)\norjson.dumps(v)\nmsgspec.json.encode(v)\n"
            "pydantic_core.to_json(v)\ndef g(n: yaml.Node) -> None: ...\n"
            "try:\n    pass\nexcept yaml.YAMLError:\n    pass\n",
            [],
        ),
        (
            _DECODING,
            "from json.decoder import JSONDecodeError\n"
            "from pydantic_core import PydanticUndefined, ValidationError\n"
            "x = (JSONDecodeError, PydanticUndefined, ValidationError)\n",
            [],
        ),
        (calls("os.replace"), "import os\nos.replace(a, b)\n", ["os.replace"]),
        (calls("os.replace"), "from os import replace\nreplace(a, b)\n", 1),
        (calls("shutil.move"), "import shutil\nshutil.move(a, b)\n", 1),
        (module_nodes(_path_move), "tmp.replace(path)\n", 1),
        (module_nodes(_path_move), 's.replace("a", "b")\n', []),
        (module_nodes(_file_write), 'p.write_text(t)\nopen(p, "w")\np.open("ab")\n', 3),
        (module_nodes(_file_write), 'open(p)\np.open()\nf(mode="after")\n', []),
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
        # A character held to two scripts' bounds (kana to Han, Greek to
        # Cyrillic): the usual "is it CJK" range, not a number range.
        (_G4, "ok = 0x3040 <= ord(ch) <= 0x9FFF\n", 1),
        (_G4, 'ok = "\\u3040" <= ch <= "\\u9fff"\n', 1),
        (_G4, "ok = 0x0370 <= ord(ch) <= 0x052F\n", 1),
        (_G4, 'r = range(ord("\\u3040"), ord("\\u9fff"))\n', 1),
        (_G4, 'C = "[\\x00-\\x1f\\x7f]"\nok = ord(ch) < 128 and ch <= "z"\n', []),
        # Round numbers each near some script's block, not one script's
        # ends: a number range.
        (_G4, "ok = 1024 <= port <= 49151\nok = 1000 <= year <= 2999\n", []),
        (_G4, "T = (1000, 30000)\nr = range(1000, 5000, 1000)\n", []),
        (_G4, "S = range(0x3040, 0x30A0)\nok = 0x0400 <= cp <= 0x04FF\n", 2),
        (
            calls_mentioning(("collections.Counter",), f"{_SCRIPTS}.script_of"),
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


# Planted copies, tuning data (a reviewer's set and the ``tuned_`` one, held
# out until the finders were tuned on it), each naming its guard, the module
# it stands in and what the guard must do with it: ``copy`` (find it), ``clean`` (find
# nothing: code near the concept that is not a copy), ``missed: why`` (a
# form it does not see yet) or ``flagged: why`` (a false positive it still
# reports). A guard that starts seeing a missed form, or stops flagging a
# clean one, fails here until the header says so.
_GUARD_COPIES = Path(__file__).parents[1] / "fixtures" / "guard_copies"
_HEADER = re.compile(
    r"# guard: (?P<guard>G\d+)\n# as: (?P<path>\S+)\n"
    r"# expect: (?P<expect>copy|clean|missed|flagged)(?:: .+)?\n"
)
_BY_ID = {_id(guard): guard for guard in GUARDS}


def _planted() -> list[object]:
    params = []
    for file in sorted(_GUARD_COPIES.glob("*.py")):
        text = file.read_text(encoding="utf-8")
        header = _HEADER.match(text)
        assert header, f"{file.name}: no guard/as/expect header"
        params.append(
            pytest.param(*header.group("guard", "path", "expect"), text, id=file.stem)
        )
    return params


@pytest.mark.parametrize(("guard_id", "path", "expect", "text"), _planted())
def test_each_planted_copy_meets_its_guard(
    guard_id: str, path: str, expect: str, text: str
) -> None:
    guard = _BY_ID[guard_id]
    assert (PACKAGE / path).is_file() and guard.scans(path), (guard.name, path)
    found = [str(hit) for hit in guard.hits_in(path, text)]
    if expect in ("copy", "flagged"):
        assert found, f"{guard.name} misses it"
    else:
        assert not found, f"{guard.name} reports {found}"
