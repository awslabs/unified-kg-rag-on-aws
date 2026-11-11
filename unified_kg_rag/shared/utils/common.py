# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
from collections.abc import Iterable
from itertools import zip_longest
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, TypeVar

from .scripts import (
    APOSTROPHE_STAND_INS,
    INVISIBLE_FORMAT_CHARS,
    QUOTE_MARKS,
    WORD_DASHES,
    drop_optional_marks,
    join_dense_middle_dots,
)

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


def _invisible(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) == "Cf"


def strip_invisible(text: str) -> str:
    """``text`` without the whitespace and the invisible format characters
    (zero-width space and joiners, a byte-order mark, direction marks) at
    its ends: what ``str.strip`` leaves of a text that shows nothing
    ("\u200b") is not empty, this is."""
    start, end = 0, len(text)
    while start < end and _invisible(text[start]):
        start += 1
    while end > start and _invisible(text[end - 1]):
        end -= 1
    return text[start:end]


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
# "McDonalds", "1,000" vs "1000", «Ромашка» vs Ромашка), so entity_key drops
# them: the apostrophe, the grave accent, the comma and every quote mark
# (``QUOTE_MARKS``).
_RE_KEY_DROP_CHARS = re.compile(f"[{re.escape(QUOTE_MARKS)}'`,]")
# Read before NFKC: what a writer types for an apostrophe as one
# (``APOSTROPHE_STAND_INS``), and the invisible format characters dropped
# (``INVISIBLE_FORMAT_CHARS``: "Wi\u00adFi" is "WiFi").
_KEY_BEFORE_NFKC = str.maketrans(
    {
        **dict.fromkeys(APOSTROPHE_STAND_INS, "'"),
        **dict.fromkeys(INVISIBLE_FORMAT_CHARS),
    }
)
# Dashes join words as "-" does, so entity_key reads them as spaces
# (``WORD_DASHES``), as it reads "_", except a dash opening the name before a
# digit, a minus sign ("−40 °C" is not "40 °C"), read as "-". Only opening
# the name: after a space, a dense-script name's optional spacing would make
# one dash a sign or a hyphen ("영하 -40" and "영하-40").
_RE_KEY_DASHES = re.compile(
    f"(?P<sign>^\\s*[{re.escape(WORD_DASHES)}](?=\\d))|[{re.escape(WORD_DASHES)}_]"
)
# Sentence punctuation the LLM sometimes leaves on the end of a name
# ("Acme Corp." vs "Acme Corp"), and the ideographic full stop ("君の名は。",
# as the query entity list strips it). Only stripped at the end, so "4.2",
# ".NET" and "C++" keep their symbols.
_RE_KEY_TRAILING_PUNCT = re.compile(r"[\s.;:!?\u3002]+$")


def _key_dash(match: re.Match[str]) -> str:
    return "-" if match["sign"] is not None else " "


def strip_trailing_punctuation(key: str) -> str:
    """``key`` without the sentence punctuation and whitespace at its end
    (:func:`entity_key`'s last step): what a key keeps of a name it was cut
    from (a legal form after "Co.") ends as a key does."""
    return _RE_KEY_TRAILING_PUNCT.sub("", key)


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

    The invisible format characters dropped (a soft hyphen, a zero-width
    space, direction marks: ``INVISIBLE_FORMAT_CHARS``) and an acute accent,
    modifier apostrophe or prime read as an apostrophe
    (``APOSTROPHE_STAND_INS``), then NFKC + casefold, quote marks of every
    script, apostrophes and commas removed, the marks a script writes at
    will dropped (Arabic short vowels and tatweel, Hebrew points:
    ``drop_optional_marks``), ``_`` and every dash treated as spaces but a
    minus sign opening the name before a digit ("−40" is "-40"), a middle dot between Han,
    Hangul or Kana letters removed ("ジョン・スミス" is "ジョンスミス",
    "한ㆍ미" "한미": ``join_dense_middle_dots``), whitespace collapsed, and
    trailing sentence punctuation stripped (an ideographic full stop too).
    A key is its own key.
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
    # Normalized again after casefolding, which can leave a decomposed
    # sequence ("ß" + U+0301 folds to "ss" + U+0301), so a key is its own key.
    folded = unicodedata.normalize(
        "NFKC",
        unicodedata.normalize("NFKC", name.translate(_KEY_BEFORE_NFKC)).casefold(),
    )
    key = _RE_KEY_DROP_CHARS.sub("", drop_optional_marks(folded))
    key = join_dense_middle_dots(_RE_KEY_DASHES.sub(_key_dash, key))
    key = strip_trailing_punctuation(" ".join(key.split()))
    # A name of what a key drops keys as itself folded, or (only invisible
    # characters) as written.
    return key or folded.strip() or unicodedata.normalize("NFKC", name).strip()


def safe_float_parse(value: Any, default_value: float | None = None) -> float | None:
    if value is None:
        return default_value

    try:
        return float(value)
    except (ValueError, TypeError):
        return default_value


def parse_llm_score(value: Any) -> int:
    """A 0-100 integer score from untrusted LLM output: clamped, and 0 for
    what is not a finite number (the JSON reader accepts ``Infinity`` and
    ``NaN``, and ``"1e999"`` overflows to infinity, which ``int`` rejects)."""
    number = safe_float_parse(str(value), default_value=0.0)
    if number is None or not math.isfinite(number):
        return 0
    return max(0, min(100, int(number)))


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
