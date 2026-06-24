# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import re
import unicodedata
import uuid
from multiprocessing import cpu_count
from typing import Any

# Fraction of CPUs to use for the default process/thread-pool worker count.
_DEFAULT_WORKER_CPU_FRACTION = 0.8


def default_max_workers() -> int:
    """Default pool size for the ingestion stages' executors.

    Single source of truth for the ``int(cpu_count() * 0.8)`` heuristic that was
    duplicated across the loader/gleaner/claim-extractor/resolver/indexing
    manager. Always at least 1.
    """
    return max(1, int(cpu_count() * _DEFAULT_WORKER_CPU_FRACTION))


# Strip punctuation/symbols but KEEP letters, marks and digits of ANY script
# (\w is Unicode-aware in Python 3). The previous [^a-z0-9\s] deleted all
# non-ASCII text, which collapsed every CJK/Cyrillic/accented entity name to ""
# — and since entity ids are hashes of the normalized name, that zeroed out the
# graph for non-English corpora (a core multilingual-support bug). Underscore is
# treated as a separator (handled before this runs).
RE_INVALID_CHARS = re.compile(r"[^\w\s]", re.UNICODE)
RE_EXTRA_SPACES = re.compile(r"\s+")


def compute_hash(data: str, algorithm: str = "sha256", length: int = 16) -> str:
    """Compute a truncated content hash for dedup / cache-key / id purposes.

    SHA-256 is used for all hashing (no MD5) — these are content fingerprints,
    not security digests, but standardizing on SHA-256 avoids weak-algorithm
    findings and keeps one code path. ``algorithm`` is accepted for backward
    compatibility; only "sha256" is supported.
    """
    if algorithm.lower() != "sha256":
        raise ValueError(f"Unsupported algorithm: '{algorithm}'")
    hash_obj = hashlib.sha256(data.encode("utf-8"))
    return hash_obj.hexdigest()[:length]


def ensure_list(data: Any, inner_key: str | None = None) -> list[Any]:
    if isinstance(data, dict):
        data = data.get(inner_key, []) if inner_key else data

    if not isinstance(data, list):
        return [data] if data else []

    return data


def generate_stable_id(content: str, namespace_key: str = "aws-graphrag") -> str:
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, namespace_key)
    return str(uuid.uuid5(namespace, content))


def normalize_name(name: str | None) -> str:
    """Normalize an entity/relationship name for id-hashing and matching.

    Unicode-aware: casefold + NFKC normalization, treat _/- as separators, drop
    punctuation/symbols but keep letters/marks/digits of any script. If the
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
    normalized = RE_INVALID_CHARS.sub(" ", normalized)
    normalized = RE_EXTRA_SPACES.sub(" ", normalized).strip()

    return normalized or unicodedata.normalize("NFKC", name).casefold().strip()


def safe_float_parse(value: Any, default_value: float | None = None) -> float | None:
    if value is None:
        return default_value

    try:
        return float(value)
    except (ValueError, TypeError):
        return default_value
