# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import re
import unicodedata
import uuid
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

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


def generate_stable_id(
    content: str, namespace_key: str = "unified-kg-rag-on-aws"
) -> str:
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


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object from a noisy LLM response.

    LLMs wrap JSON in markdown fences or surround it with prose; several call
    sites (DRIFT primer, global-search map, keyword extraction, prompt tuning)
    need the same forgiving extraction. Strips a leading ```/```json fence,
    isolates the outermost ``{...}``, and parses it. Returns ``{}`` on any
    failure (or non-object JSON) so callers degrade gracefully rather than crash.
    """
    if not raw:
        return {}
    text = raw.strip()
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
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
