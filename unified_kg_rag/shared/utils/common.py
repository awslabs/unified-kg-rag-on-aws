# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import math
import os
import re
import socket
import ssl
import traceback
import unicodedata
import uuid
from collections.abc import Iterable
from itertools import zip_longest
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, TypeVar
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit

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
    or, combined with the 0.8 fraction, mis-size them. Both container limits
    count, each when present, and the lower wins (``docker run --cpus=6
    --cpuset-cpus=0,1`` runs on two CPUs, ``--cpus=2`` on a 0-7 cpuset on two):

    1. the cgroup v2 / v1 CPU quota (the real CFS ceiling the task runs under),
    2. the CPU affinity mask (``os.sched_getaffinity``, respects cpusets);

    ``cpu_count()`` when neither is known.
    """
    limits: list[int] = []
    quota = _cgroup_cpu_quota()
    if quota is not None and quota >= 1:
        limits.append(quota)
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity >= 1:
                limits.append(affinity)
        except OSError:
            pass
    return min(limits) if limits else cpu_count()


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


def safe_url(url: str) -> str:
    """``url`` without userinfo, query or fragment (they may hold a key)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        # An IPv6 literal keeps its brackets ("http://[::1]:4318").
        host = f"[{host}]"
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def http_status(error: BaseException) -> int | None:
    """The HTTP status an error carries as ``status_code`` (``openai``'s,
    ``ollama.ResponseError``), ``response.status_code`` (``httpx``'s) or,
    for ``urllib``'s ``HTTPError``, ``code``."""
    if isinstance(error, HTTPError):
        return error.code if isinstance(error.code, int) else None
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def reported_status(error: BaseException) -> int | None:
    """:func:`http_status`, or the one botocore's ``ClientError`` reports
    in its response metadata."""
    if (status := http_status(error)) is not None:
        return status
    response = getattr(error, "response", None)
    metadata = response.get("ResponseMetadata") if isinstance(response, dict) else None
    found = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return found if isinstance(found, int) else None


def error_code(error: BaseException) -> str:
    """The service's error code an error carries: a botocore
    ``ClientError``'s (``ThrottlingException``), the error type an
    OpenSearch response names (``index_not_found_exception``, from its
    ``info``; its ``error`` alone may be the raw response body), the engine
    error a Gremlin server's JSON status message names (Neptune's
    ``ConcurrentModificationException``; the rest of it may quote the
    query), an ``openai`` error's ``code`` (``rate_limit_exceeded``); else
    the error's type name. Each only when it is an identifier
    (:data:`_CODE`): a ``code`` may be free text (``SystemExit``'s)."""
    response = getattr(error, "response", None)
    if isinstance(response, dict) and (
        code := _identifier((response.get("Error") or {}).get("Code"))
    ):
        return code
    if code := _status_message_code(getattr(error, "status_message", None)):
        return code
    info = getattr(error, "info", None)
    if isinstance(info, dict) and isinstance(info.get("error"), dict):
        if kind := _identifier(info["error"].get("type")):
            return kind
    if code := _identifier(getattr(error, "code", None)):
        return code
    return type(error).__name__


# An error code: an identifier, never text (a message, a query) an error
# carries where a code usually is.
_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _identifier(value: object) -> str | None:
    """``value`` when it is an error code (:data:`_CODE`), else ``None``."""
    return value if isinstance(value, str) and _CODE.fullmatch(value) else None


def _status_message_code(message: object) -> str | None:
    """The ``code`` a JSON status message (a Gremlin server's) names."""
    if not isinstance(message, str) or not message.lstrip().startswith("{"):
        return None
    try:
        body = json.loads(message)
    except ValueError:
        return None
    return _identifier(body.get("code")) if isinstance(body, dict) else None


def _data_free_facts(error: BaseException) -> str:
    """What an error whose text is the runtime's, not data, says: an
    ``OSError``'s errno (named by the OS, ``os.strerror``) and the paths it
    names (a path is where the run reads and writes, not the content), a
    decode error's position and reason. Empty for any other error: its text
    may quote a model's reply, a query or corpus text."""
    if isinstance(error, ssl.SSLError):
        # Its errno is the SSL library's error number, not an OS one.
        parts = (getattr(error, "library", None), getattr(error, "reason", None))
        return " ".join(str(part) for part in parts if part)
    if isinstance(error, socket.gaierror | socket.herror):
        # Its errno is the resolver's (EAI_*), named by the resolver.
        return (
            f"[Errno {error.errno}] {error.strerror}"
            if isinstance(error.strerror, str)
            else ""
        )
    if isinstance(error, OSError) and isinstance(error.errno, int):
        facts = f"[Errno {error.errno}] {os.strerror(error.errno)}"
        paths = [
            repr(os.fsdecode(path))
            for path in (error.filename, error.filename2)
            if isinstance(path, str | bytes | os.PathLike)
        ]
        return f"{facts}: {' -> '.join(paths)}" if paths else facts
    if isinstance(error, json.JSONDecodeError):
        # ``msg`` is the decoder's own ("Expecting value"), never the document.
        return f"{error.msg}: line {error.lineno} column {error.colno}"
    if isinstance(error, UnicodeDecodeError | UnicodeEncodeError):
        return f"{error.encoding!r} {error.reason} at position {error.start}"
    return ""


# What a status an error reports is, by the top-level package of its type:
# a Gremlin server's response status (597, a script evaluation error) is
# not an HTTP one.
_STATUS_KINDS = {"gremlin_python": "Gremlin status"}


def error_name(error: BaseException, *, _depth: int = 0) -> str:
    """Log-safe name of ``error``, never its message: its type, the
    service's error code and the HTTP status it reports
    (``ClientError ThrottlingException, HTTP 429``; ``ResponseError, HTTP
    400``) and what an error whose text holds no data says
    (``PermissionError [Errno 13] Permission denied: '/out'``,
    ``JSONDecodeError Expecting value: line 1 column 1``). A package error
    that reports none of its own and was raised from another (a
    ``LanguageModelError`` wrapping a provider's) names that one too
    (``LanguageModelError from ResponseError, HTTP 500``)."""
    name = type(error).__name__
    code = error_code(error)
    head = name if code == name else f"{name} {code}"
    if facts := _data_free_facts(error):
        head = f"{head} {facts}"
    status = reported_status(error)
    if status is not None:
        package = type(error).__module__.partition(".")[0]
        return f"{head}, {_STATUS_KINDS.get(package, 'HTTP')} {status}"
    cause = error.__cause__
    if (
        cause is not None
        and head == name
        and type(error).__module__.partition(".")[0] == __name__.partition(".")[0]
        and _depth < 3
    ):
        return f"{head} from {error_name(cause, _depth=_depth + 1)}"
    return head


def error_digest(error: BaseException) -> str:
    """Log-safe stand-in for an error at INFO and above: :func:`error_name`
    (its type, code, HTTP status) and ``text_digest`` of its message, which
    may quote a model's reply, an answer, a query or corpus text. Log the
    error itself at DEBUG."""
    return f"{error_name(error)} ({text_digest(str(error))})"


def error_locator(error: BaseException, *, frames: int = 4, depth: int = 4) -> str:
    """Where an unexpected ``error`` came from, without its message: the type
    chain along ``__cause__``/``__context__`` (outermost first; not the
    context a ``raise ... from None`` suppressed), each by :func:`error_name`
    with its last ``frames`` ``file:line function`` frames (the package's
    files by path from the package, others by their last two parts, never
    an absolute path). Frames are code
    locations (``traceback.extract_tb``), never the exception text, so it is
    safe at INFO and above and locates a failure the surface only says
    "the server log has the details" of. Empty for an error never raised."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(parts) < depth:
        seen.add(id(current))
        located = []
        for frame in traceback.extract_tb(current.__traceback__)[-frames:]:
            path = Path(frame.filename).parts
            keep = (
                path[path.index("unified_kg_rag") :]
                if "unified_kg_rag" in path
                else path[-2:]
            )
            located.append(f"{'/'.join(keep)}:{frame.lineno} {frame.name}")
        parts.append(f"{error_name(current)} at {' > '.join(located) or '?'}")
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return " <- ".join(parts) if error.__traceback__ is not None else ""


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


# A decimal digit of any script (str patterns match Unicode digits).
_RE_DIGIT = re.compile(r"\d")


def _ascii_digit(match: re.Match[str]) -> str:
    return str(unicodedata.decimal(match[0]))


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
    ``drop_optional_marks``), the decimal digits of every script read as
    ASCII digits ("PO١٠٠٢" is "po1002", as NFKC reads a full-width "１"),
    ``_`` and every dash treated as spaces but a
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
    if not folded.isascii():
        folded = _RE_DIGIT.sub(_ascii_digit, folded)
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
