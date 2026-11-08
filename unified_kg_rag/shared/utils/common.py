# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
from collections.abc import Collection, Iterable
from itertools import zip_longest
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, TypeVar

from .scripts import drop_optional_marks

_T = TypeVar("_T")

# Fraction of CPUs to use for the default process/thread-pool worker count.
_DEFAULT_WORKER_CPU_FRACTION = 0.8


def _available_cpu_count() -> int:
    """CPUs actually available to this process, respecting container limits.

    ``multiprocessing.cpu_count()`` returns the host's physical core count and
    ignores cgroup CPU quotas, so inside a container with ``--cpus 2`` (e.g. a
    2-vCPU Fargate task) it can report the host's 7+ cores — over-sizing pools —
    or, combined with the 0.8 fraction, mis-size them. We prefer, in order:

    1. the cgroup v2 / v1 CPU quota (the real CFS ceiling the task runs under),
    2. the CPU affinity mask (``os.sched_getaffinity``, respects cpusets),
    3. ``cpu_count()`` as a last resort.
    """
    quota = _cgroup_cpu_quota()
    if quota is not None and quota >= 1:
        return quota
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity >= 1:
                return affinity
        except OSError:
            pass
    return cpu_count()


def _cgroup_cpu_quota() -> int | None:
    """Whole CPUs permitted by the cgroup CFS quota, or None if unlimited/absent.

    Reads cgroup v2 (``cpu.max`` = "<quota> <period>") then cgroup v1
    (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``). Returns ``ceil(quota/period)``
    rounded to at least 1, or None when no quota is set (``max`` / ``-1``) or the
    files are unreadable (non-Linux, no cgroup mount).
    """
    try:
        v2 = Path("/sys/fs/cgroup/cpu.max")
        if v2.is_file():
            quota_str, _, period_str = v2.read_text().strip().partition(" ")
            if quota_str != "max":
                quota, period = int(quota_str), int(period_str or "100000")
                if quota > 0 and period > 0:
                    return max(1, -(-quota // period))  # ceil division
            return None
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, -(-quota // period))
    except (OSError, ValueError):
        return None
    return None


def default_max_workers() -> int:
    """Default pool size for the ingestion stages' executors.

    Single source of truth for the ``* 0.8`` heuristic that was duplicated across
    the loader/gleaner/claim-extractor/resolver/indexing manager. Uses the
    container-aware CPU count (:func:`_available_cpu_count`) so a 2-vCPU Fargate
    task does not size pools off the host's core count. Always at least 1.
    """
    return max(1, int(_available_cpu_count() * _DEFAULT_WORKER_CPU_FRACTION))


# Strip punctuation/symbols but KEEP letters, marks and digits of ANY script.
# The previous [^a-z0-9\s] deleted all non-ASCII text, which collapsed every
# CJK/Cyrillic/accented entity name to "" — and since entity ids are hashes of
# the normalized name, that zeroed out the graph for non-English corpora (a
# core multilingual-support bug). \w alone is not enough: it holds no
# combining mark, so a Devanagari or Thai vowel sign split its word
# ("राहुल" -> "र ह ल"). Underscore is treated as a separator (handled before
# this runs).
RE_INVALID_CHARS = re.compile(r"[^\w\s]", re.UNICODE)


def _drop_symbols(text: str) -> str:
    """``text`` with every character but a letter, digit, mark or space
    replaced by a space."""
    if not RE_INVALID_CHARS.search(text):
        return text
    return "".join(
        (
            ch
            if ch.isalnum() or ch.isspace() or unicodedata.category(ch)[0] == "M"
            else " "
        )
        for ch in text
    )


RE_EXTRA_SPACES = re.compile(r"\s+")


def interleave(lists: Iterable[Iterable[_T]]) -> list[_T]:
    """The items of several ranked lists in one: their first items, then
    their second ones, and so on (what an equal-weight rank fusion of the
    lists keeps when one list's scores do not compare with another's)."""
    return [
        item
        for same_rank in zip_longest(*(list(items) for items in lists))
        for item in same_rank
        if item is not None
    ]


def compute_hash(data: str, length: int = 16) -> str:
    """Compute a truncated content hash for dedup / cache-key / id purposes.

    SHA-256 is used for all hashing (no MD5) — these are content fingerprints,
    not security digests, but standardizing on SHA-256 avoids weak-algorithm
    findings and keeps one code path.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]


def text_digest(text: str | None) -> str:
    """Log-safe stand-in for user, corpus or model text: length + short hash.

    INFO-and-above log records carry this instead of the text itself (queries,
    entity names and model output are DEBUG-only); the hash lets one query be
    correlated across records without revealing it.
    """
    if not text:
        return "len=0"
    return f"len={len(text)} sha={compute_hash(text, length=8)}"


def error_digest(error: BaseException) -> str:
    """Log-safe stand-in for an error at INFO and above: its type and
    ``text_digest`` of its message, which may quote a model's reply, an
    answer, a query or corpus text. Log the error itself at DEBUG."""
    return f"{type(error).__name__} ({text_digest(str(error))})"


def ensure_list(data: Any, inner_key: str | None = None) -> list[Any]:
    if isinstance(data, dict):
        data = data.get(inner_key, []) if inner_key else data

    if not isinstance(data, list):
        return [data] if data else []

    return data


def generate_stable_id(
    content: str, namespace_key: str = "unified-kg-rag-on-aws"
) -> str:
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, namespace_key)
    return str(uuid.uuid5(namespace, content))


def id_content(kind: str, *parts: str) -> str:
    """The content a stable id of ``kind`` over ``parts`` hashes
    (:func:`generate_stable_id`): ``kind:part:part...``, as ids were
    always made, unless a part holds a ``:``, when two different part lists
    could join into one string ("A:B" + "C" and "A" + "B:C"); those are
    joined as a JSON list instead, which no other part list gives."""
    if any(":" in part for part in parts):
        return json.dumps([kind, *parts], ensure_ascii=False)
    return ":".join((kind, *parts))


def normalize_name(name: str | None) -> str:
    """Normalize a name for token-level fuzzy similarity (not identity).

    Entity ids and exact-name matching use :func:`entity_key`, which keeps
    meaningful symbols; this stricter form drops all punctuation so token
    overlap ignores it.

    Unicode-aware: casefold + NFKC normalization, treat _/- as separators, drop
    punctuation/symbols but keep letters/marks/digits of any script, except
    the marks a script writes at will (Arabic short vowels, Hebrew points:
    ``drop_optional_marks``). If the
    cleaned result is empty (e.g. a name made only of punctuation/emoji) fall
    back to the casefolded original so a non-empty input never collapses to ""
    (which would merge unrelated entities under the same id).
    """
    if not name:
        return ""

    # NFKC unifies compatibility forms (full-width, ligatures); casefold is the
    # Unicode-correct lowercasing (handles ß, Turkish İ, Greek, etc.).
    normalized = unicodedata.normalize("NFKC", name).casefold()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = drop_optional_marks(_drop_symbols(normalized))
    normalized = RE_EXTRA_SPACES.sub(" ", normalized).strip()

    return normalized or unicodedata.normalize("NFKC", name).casefold().strip()


# Quote marks and thousands separators carry no identity ("McDonald's" vs
# "McDonalds", "1,000" vs "1000"), so entity_key drops them.
_RE_KEY_DROP_CHARS = re.compile("[\"'`\u2018\u2019\u201c\u201d,]")
# Sentence punctuation the LLM sometimes leaves on the end of a name
# ("Acme Corp." vs "Acme Corp"). Only stripped at the end, so "4.2", ".NET"
# and "C++" keep their symbols.
_RE_KEY_TRAILING_PUNCT = re.compile(r"[.;:!?]+$")


def clean_display_name(name: str | None) -> str:
    """Return the display form of an entity name: trimmed, whitespace collapsed.

    This is what ``Entity.name`` (and relationship ``source_name`` /
    ``target_name``) carries, so retrieval context and community reports quote
    the surface form the source used ("$1,000 penalty", "Section 4.2", "C++")
    instead of a normalized key. Identity lives in :func:`entity_key`.
    """
    if not name:
        return ""
    return " ".join(name.split())


def entity_key(name: str | None) -> str:
    """Return the identity key of an entity name (used for ids and exact match).

    NFKC + casefold, ``_``/``-`` treated as spaces, quote marks and commas
    removed, whitespace collapsed, and trailing sentence punctuation stripped.
    Unlike :func:`normalize_name` it keeps every other symbol, so names that
    differ only in meaningful punctuation get distinct keys ("C++" vs "C#",
    "Section 4.2" vs "Section 42") while case/whitespace/quote variants of one
    name ("ACME  Corp." vs "Acme Corp") share a key. Falls back to the
    casefolded input so a non-empty name never yields an empty key.

    Entity and relationship ids are hashes of this key; changing it changes
    every id, which requires a full re-index of existing graphs.
    """
    if not name:
        return ""
    folded = unicodedata.normalize("NFKC", name).casefold()
    key = folded.replace("_", " ").replace("-", " ")
    key = _RE_KEY_DROP_CHARS.sub("", key)
    key = " ".join(key.split())
    key = _RE_KEY_TRAILING_PUNCT.sub("", key).rstrip()
    return key or folded.strip()


def safe_float_parse(value: Any, default_value: float | None = None) -> float | None:
    if value is None:
        return default_value

    try:
        return float(value)
    except (ValueError, TypeError):
        return default_value


def parse_llm_score(value: Any) -> int:
    """A 0-100 integer score from untrusted LLM output: clamped, and 0 for
    what is not a finite number (``parse_llm_json`` accepts ``Infinity`` and
    ``NaN``, and ``"1e999"`` overflows to infinity, which ``int`` rejects)."""
    number = safe_float_parse(str(value), default_value=0.0)
    if number is None or not math.isfinite(number):
        return 0
    return max(0, min(100, int(number)))


# A `{` that plausibly starts a JSON object: a key's quote, or the end of a
# response cut right after it.
_RE_OPENS_OBJECT = re.compile(r'\{\s*(?:"|\Z)')
# The message of the strict parse's error for a response cut inside its
# last object (``parse_model_object`` tells it from other parse errors).
UNCLOSED_JSON_MESSAGE = "The last JSON object is unclosed (cut off?)"


def parse_llm_json(
    raw: str, *, strict: bool = False, answer_keys: Collection[str] = ()
) -> dict[str, Any]:
    """Parse a JSON object from a noisy LLM response.

    LLMs wrap JSON in markdown fences or surround it with prose; every call
    site that reads LLM JSON uses this one forgiving extraction. The span
    from the first ``{`` to the last ``}`` is parsed. When it does not parse
    because it holds several objects, as a model that corrects itself writes
    (an answer, "Correction: ... final output:", the answer again), the last
    top-level object that parses is the answer: the one the model ends on.
    With ``answer_keys``, the last one holding any of them is, so an example
    object the model writes before or after its answer is not taken for it.
    Non-object JSON yields ``{}``. Unparseable text yields ``{}`` so callers
    degrade gracefully, or raises ``json.JSONDecodeError`` when ``strict`` is
    set, for callers that must surface a broken response. Under ``strict`` a
    response whose last object is unclosed (cut off) raises too, even when
    an earlier object parses: a self-correction ends on a complete object.
    An unclosed ``{`` counts only when it starts JSON (``{"`` or a ``{`` the
    response ends on): prose after the answer may hold a lone brace.
    """
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            pass
        else:
            # A `{` after the last `}` that starts JSON (a key, or the cut)
            # opens an object left unclosed; prose may hold a lone brace.
            if not (strict and _RE_OPENS_OBJECT.search(text, end + 1)):
                return parsed if isinstance(parsed, dict) else {}
        objects, cut_at = _top_level_json_objects(text)
        if cut_at is not None and not _RE_OPENS_OBJECT.search(text, cut_at):
            # Lone braces in prose after the answer, no cut-off object.
            cut_at = None
        if strict and cut_at is not None:
            raise json.JSONDecodeError(UNCLOSED_JSON_MESSAGE, text, cut_at)
        if objects:
            return next(
                (o for o in reversed(objects) if any(k in o for k in answer_keys)),
                objects[-1],
            )
    # Strip a leading/trailing markdown code fence if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    # Isolate the outermost JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except ValueError:
        if strict:
            raise
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _top_level_json_objects(text: str) -> tuple[list[dict[str, Any]], int | None]:
    """The top-level ``{...}`` spans of ``text`` that parse as JSON objects,
    in order, and where an unclosed one starts (``None`` when every one is
    closed). Braces inside JSON strings do not count; an object nested in
    one that does not parse is not top-level, so it is never taken for the
    answer, nor is one inside an object left unclosed."""
    objects: list[dict[str, Any]] = []
    position = 0
    while (start := text.find("{", position)) != -1:
        depth, in_string, escaped, end = 0, False, False, -1
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end == -1:
            # Unclosed (cut off): what follows is inside it, not top-level.
            return objects, start
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            objects.append(parsed)
        position = end + 1
    return objects, None


# Every vector field the indexers write ends in this suffix (``text_embedding``,
# ``name_embedding``, ``full_content_embedding``, ...).
EMBEDDING_FIELD_SUFFIX = "_embedding"


def strip_embedding_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``data`` without vector fields.

    Drops ``embedding`` and ``*_embedding`` keys. Vectors are only needed for
    scoring; carried into result metadata or exports they bloat every payload
    (a community report holds three) and leak into reported sources.
    """
    return {
        k: v
        for k, v in data.items()
        if k != "embedding" and not k.endswith(EMBEDDING_FIELD_SUFFIX)
    }
