# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuSiQue as a corpus plus an evaluation dataset.

MuSiQue (Trivedi et al., TACL 2022) composes 2-4 single-hop questions into
one multi-hop question; each item ships 20 paragraphs, of which the
supporting ones answer it and the rest are distractors. The data is
distributed under CC BY 4.0 (https://github.com/StonyBrookNLP/musique) and
is never committed here: ``prepare`` reads the official JSONL — a local copy
from the official download, or the answerable dev split fetched at run time
from a pinned revision of a public mirror, checked against its SHA-256.

``prepare`` writes one text file per distinct paragraph (title, blank line,
text) under ``corpus/`` and ``eval_data.json`` in the ``run-eval`` format:
``answer`` and ``answer_aliases`` for ``answer_match``, the supporting
paragraphs' file names as ``reference_sources`` for ``retrieval``, and the
hop count (``2hop``, ``3hop``, ``4hop``) as ``category``. Every selected question's
distractors are part of the corpus, as in the benchmark's own setting.

``unanswerable`` (opt-in) adds that many unanswerable items of a MuSiQue full
file (``musique_full_v1.0_*.jsonl``, downloaded the same way): each is the
question of an answerable twin with one or more supporting paragraphs
replaced. An item is left out when its twin is in the answerable sample, or
when the exact text of a removed paragraph is in the pooled corpus (MuSiQue
reuses paragraphs across questions): see ``select_unanswerable``. The
answerable questions still come from the answerable file
(``unanswerable_source`` is the full one), so they are the MuSiQue-Ans
sample the seed draws without ``unanswerable``; a full file given alone
supplies both, its answerable half labelled ``musique-full-answerable``.
``hops`` filters by hop count or draws an answerable quota per hop.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from unified_kg_rag.evaluation.retrieval_evaluator import reference_key
from unified_kg_rag.shared import EvaluationException, get_logger
from unified_kg_rag.shared.utils.files import atomic_open, atomic_write_text

logger = get_logger(__name__)

LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
CITATION = (
    "Trivedi, Balasubramanian, Khot, Sabharwal. MuSiQue: Multihop Questions "
    "via Single-hop Question Composition. TACL 2022."
)
# The answerable dev split, byte-identical to the official
# musique_ans_v1.0_dev.jsonl, at a pinned revision of a public mirror.
DEV_URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/"
    "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321/musique_ans_v1.0_dev.jsonl"
)
DEV_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"  # pragma: allowlist secret
# The full dev split (each question answerable and as an unanswerable twin),
# at a pinned revision of a public mirror whose answerable split is the one
# above byte for byte; a second, independent mirror serves the same bytes.
FULL_DEV_URL = (
    "https://huggingface.co/datasets/bdsaglam/musique/resolve/"
    "22873a405dd809893b22ada0b499299fb612d2df/musique_full_v1.0_dev.jsonl"
)
FULL_DEV_SHA256 = "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a"  # pragma: allowlist secret
FULL_DEV_FILE = "musique_full_v1.0_dev.jsonl"

_HOPS = re.compile(r"(\d+)hop")
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def download_dev(
    destination: Path, *, timeout: float = 120.0, full: bool = False
) -> Path:
    """Fetch the answerable dev split (``full``: the full one, with the
    unanswerable twins) to ``destination`` and verify it."""
    url, expected = (FULL_DEV_URL, FULL_DEV_SHA256) if full else (DEV_URL, DEV_SHA256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MuSiQue dev (%s) from %s", LICENSE, url)
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected - constant https URL
    with urllib.request.urlopen(  # nosec B310 - constant https URL
        url, timeout=timeout
    ) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise EvaluationException(
            f"MuSiQue dev download has SHA-256 {digest}, expected {expected}; "
            "pass the official file with --data-path instead."
        )
    with atomic_open(destination, "wb") as out:
        out.write(data)
    return destination


# metadata.benchmark of the rows: MuSiQue-Ans, or the answerable half of a
# full file (not the same benchmark).
BENCHMARK_NAME = "musique"
FULL_ANSWERABLE = "musique-full-answerable"
FULL_UNANSWERABLE = "musique-full-unanswerable"
# Appended to an unanswerable item's id, so its query id never equals its
# twin's in another sample.
UNANSWERABLE_ID_SUFFIX = "__unanswerable"
HOP_COUNTS = (2, 3, 4)


class InvalidBenchmarkDataError(EvaluationException):
    """The benchmark file is not MuSiQue JSONL, or cannot supply the
    requested sample (an input error, exit 2)."""


def parse_hops(value: str) -> dict[int, int | None]:
    """``--hops``: ``"3,4"`` (a filter: ``{3: None, 4: None}``) or
    ``"2=50,3=50,4=100"`` (exact answerable quotas per hop).

    Raises:
        ValueError: An unknown hop count, a repeated one, a quota below 1,
            or quotas on some hop counts but not others.
    """
    hops: dict[int, int | None] = {}
    for part in value.split(","):
        hop_text, _, quota_text = part.strip().partition("=")
        try:
            hop = int(hop_text)
            quota = int(quota_text) if quota_text else None
        except ValueError:
            raise ValueError(
                f"'{part.strip()}' is not a hop count or hop=quota"
            ) from None
        if hop not in HOP_COUNTS:
            raise ValueError(f"hop count {hop} is not one of 2, 3, 4")
        if hop in hops:
            raise ValueError(f"hop count {hop} is given twice")
        if quota is not None and quota < 1:
            raise ValueError(f"the quota of {hop} hops must be at least 1")
        hops[hop] = quota
    if len({quota is None for quota in hops.values()}) > 1:
        raise ValueError("give a quota for every hop count or for none")
    return hops


def has_unanswerable(source: Path) -> bool:
    """Whether the MuSiQue JSONL ``source`` holds an unanswerable item (a
    full file)."""
    return any(not item.get("answerable", True) for item in _items(source))


def hop_count(item_id: str) -> int | None:
    """The hop count an item id starts with (``3hop1__...`` is 3)."""
    match = _HOPS.match(item_id)
    return int(match.group(1)) if match else None


def _items(source: Path) -> Iterator[dict[str, Any]]:
    seen: set[tuple[str, bool]] = set()
    with source.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise InvalidBenchmarkDataError(
                    f"{source}:{number} is not a JSON line: {e}"
                ) from e
            if not _is_item(item):
                raise InvalidBenchmarkDataError(
                    f"{source}:{number} is not a MuSiQue item (an object with "
                    "an 'id', a non-empty string 'question', a string "
                    "'answer', a list of string 'answer_aliases', a boolean "
                    "'answerable' and a 'paragraphs' list of objects with a "
                    "string 'paragraph_text')"
                )
            if item.get("answerable", True) or "id" in item:
                # Query ids key the results: a repeated one would merge two.
                # The full files pair every question with an unanswerable
                # twin of the same id, so ids are unique per answerability.
                key = (str(item["id"]), bool(item.get("answerable", True)))
                if key in seen:
                    raise InvalidBenchmarkDataError(
                        f"{source}:{number} repeats the id {item['id']!r}"
                    )
                seen.add(key)
            yield item


def _is_item(item: Any) -> bool:
    if not isinstance(item, dict) or not isinstance(item.get("answerable", True), bool):
        return False
    if item.get("answerable", True) is False:
        # Checked in full only when sampled (select_unanswerable).
        return True
    return _is_complete(item)


def _is_complete(item: dict[str, Any]) -> bool:
    paragraphs = item.get("paragraphs")
    aliases = item.get("answer_aliases") or []
    question = item.get("question")
    return (
        "id" in item
        and isinstance(question, str)
        and bool(question.strip())
        and isinstance(item.get("answer"), str)
        and isinstance(aliases, list)
        and all(isinstance(alias, str) for alias in aliases)
        and isinstance(paragraphs, list)
        and all(
            isinstance(p, dict) and isinstance(p.get("paragraph_text", ""), str)
            for p in paragraphs
        )
    )


def _file_name(title: str, taken: set[str]) -> str:
    # No leading dot: the parser skips hidden files.
    stem = " ".join(_UNSAFE.sub(" ", title).split())[:150].lstrip(". ") or "paragraph"
    name, n = f"{stem}.txt", 2
    # Distinct under the retrieval evaluator's matching (case, NFKC, "_" as
    # " "), or two paragraphs would count as one file.
    while reference_key(name) in taken:
        name, n = f"{stem} ({n}).txt", n + 1
    taken.add(reference_key(name))
    return name


def check_output_directory(output_dir: Path) -> None:
    """Raise unless ``output_dir``'s ``corpus/`` is new or empty: files of
    another sample would be ingested with this one."""
    corpus = output_dir / "corpus"
    if corpus.exists() and any(corpus.iterdir()):
        raise EvaluationException(
            f"'{corpus}' is not empty; choose a new --output-directory"
        )


def _text(paragraph: dict[str, Any]) -> str:
    """A paragraph's text, whitespace-normalized for exact-text matching."""
    return " ".join(str(paragraph.get("paragraph_text") or "").split())


def removed_paragraphs(twin: dict[str, Any], item: dict[str, Any]) -> set[str]:
    """The supporting paragraphs of the answerable ``twin`` that the
    unanswerable ``item`` lacks (their normalized text)."""
    present = {_text(p) for p in item["paragraphs"]}
    return {
        text
        for p in twin["paragraphs"]
        if p.get("is_supporting") and (text := _text(p)) not in present
    }


def sample_answerable(
    items: list[dict[str, Any]],
    rng: random.Random,
    *,
    limit: int | None = None,
    hops: dict[int, int | None] | None = None,
) -> list[dict[str, Any]]:
    """``limit`` items drawn with ``rng`` (all when ``None``), from the hop
    counts ``hops`` names; with quotas, exactly that many per hop count.

    Raises:
        InvalidBenchmarkDataError: Quotas with a ``limit``, or a quota above
            the items of its hop count.
    """
    if hops:
        items = [i for i in items if hop_count(str(i["id"])) in hops]
    if hops and None not in hops.values():
        if limit is not None:
            raise InvalidBenchmarkDataError(
                "--hops quotas set the sample size; drop --limit"
            )
        sample: list[dict[str, Any]] = []
        for hop, quota in sorted(hops.items()):
            group = [i for i in items if hop_count(str(i["id"])) == hop]
            if quota is not None and quota > len(group):
                raise InvalidBenchmarkDataError(
                    f"{quota} {hop}-hop questions requested, {len(group)} available"
                )
            sample.extend(rng.sample(group, quota or 0))
        return sample
    if limit is not None and limit < len(items):
        return rng.sample(items, limit)
    return items


def select_unanswerable(
    candidates: list[dict[str, Any]],
    twins: dict[str, dict[str, Any]],
    answerable: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """``count`` unanswerable items, each with its answerable twin, that stay
    unanswerable in the pooled corpus; and the counts of the selection.

    Candidates are visited in a ``rng`` shuffle. One is excluded when its
    twin is in ``answerable`` (``excluded_twin``: the twin's paragraphs
    would give the removed ones back), when the file has no twin to tell
    which paragraphs were removed (``excluded_no_twin``), or
    (``excluded_contaminated``) when the exact text of a removed paragraph is
    in the pool so far (the answerable sample's paragraphs and those of the
    items kept) or one of its own paragraphs is a removed paragraph of an
    item kept. A kept item's paragraphs join the pool. ``examined`` counts
    the candidates visited before ``count`` were kept.

    Raises:
        InvalidBenchmarkDataError: A visited candidate is not a complete
            MuSiQue item, or fewer than ``count`` pass.
    """
    sampled = {str(i["id"]) for i in answerable}
    pool = {_text(p) for item in answerable for p in item["paragraphs"]}
    removed_kept: set[str] = set()
    counts = {
        "requested": count,
        "kept": 0,
        "examined": 0,
        "excluded_twin": 0,
        "excluded_no_twin": 0,
        "excluded_contaminated": 0,
    }
    order = list(candidates)
    rng.shuffle(order)
    kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in order:
        if len(kept) == count:
            break
        if not _is_complete(item):
            raise InvalidBenchmarkDataError(
                f"the unanswerable item {item.get('id')!r} is not a complete "
                "MuSiQue item (question, answer, answer_aliases, paragraphs)"
            )
        counts["examined"] += 1
        item_id = str(item["id"])
        twin = twins.get(item_id)
        if item_id in sampled:
            counts["excluded_twin"] += 1
            continue
        removed = removed_paragraphs(twin, item) if twin is not None else set()
        if not removed:
            counts["excluded_no_twin"] += 1
            continue
        own = {_text(p) for p in item["paragraphs"]}
        if removed & pool or own & removed_kept:
            counts["excluded_contaminated"] += 1
            continue
        assert twin is not None  # removed is empty without one
        kept.append((item, twin))
        pool |= own
        removed_kept |= removed
    counts["kept"] = len(kept)
    if len(kept) < count:
        raise InvalidBenchmarkDataError(
            f"only {len(kept)} of the {count} requested unanswerable items "
            f"stay unanswerable in the pooled corpus ({counts}); request "
            "fewer, or fewer answerable questions"
        )
    return kept, counts


def prepare(
    source: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    unanswerable: int | None = None,
    hops: dict[int, int | None] | None = None,
    unanswerable_source: Path | None = None,
) -> dict[str, Any]:
    """Write ``corpus/`` and ``eval_data.json`` for (a sample of) ``source``.

    ``limit`` answerable questions are drawn with ``seed`` (all when
    ``None``), from the hop counts ``hops`` names (with quotas: exactly that
    many per hop count). Unanswerable items are skipped unless
    ``unanswerable`` asks for that many (:func:`select_unanswerable`, same
    seed, same hop counts), drawn from the full file
    ``unanswerable_source`` (with their twins, its answerable half), or from
    ``source`` itself when it is a full file and ``unanswerable_source`` is
    not given; they are written with ``metadata.answerable: false``. The
    answerable half of a full file is not MuSiQue-Ans (its paragraphs and
    answer aliases differ), so answerable rows drawn from one are labelled
    ``musique-full-answerable`` instead of ``musique``, with a WARNING. The
    answerable sample is the one ``source`` and ``seed`` draw without
    ``unanswerable``.

    Raises:
        InvalidBenchmarkDataError: ``unanswerable`` with no file that has
            unanswerable items, or a sample the files cannot supply.
    """
    check_output_directory(output_dir)
    corpus = output_dir / "corpus"
    every_item = list(_items(source))
    items = [i for i in every_item if i.get("answerable", True)]
    full_items = (
        list(_items(unanswerable_source))
        if unanswerable and unanswerable_source is not None
        else every_item
    )
    unanswerable_items = [i for i in full_items if not i.get("answerable", True)]
    if unanswerable and not unanswerable_items:
        named = unanswerable_source or source
        raise InvalidBenchmarkDataError(
            f"{named} has no unanswerable items: --unanswerable needs a "
            f"MuSiQue full file ({FULL_DEV_FILE}, the default download with "
            "--unanswerable)"
        )
    benchmark = FULL_ANSWERABLE if len(items) < len(every_item) else BENCHMARK_NAME
    if benchmark == FULL_ANSWERABLE:
        logger.warning(
            "%s is a MuSiQue full file: its answerable items are written as "
            "'%s', whose paragraphs and answer aliases differ from "
            "MuSiQue-Ans, so the scores are not comparable to MuSiQue-Ans "
            "results (use musique_ans_v1.0_*.jsonl, or the default download, "
            "for those).",
            source,
            FULL_ANSWERABLE,
        )
    rng = random.Random(seed)
    items = sample_answerable(items, rng, limit=limit, hops=hops)
    declined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    selection: dict[str, int] = {}
    if unanswerable:
        declined, selection = select_unanswerable(
            [
                i
                for i in unanswerable_items
                if not hops or hop_count(str(i.get("id", ""))) in hops
            ],
            # Twins come from the whole full file, sampled or not: its
            # answerable half tells which paragraphs were removed.
            {str(i["id"]): i for i in full_items if i.get("answerable", True)},
            items,
            unanswerable,
            rng,
        )
        logger.info(
            "MuSiQue unanswerable selection: %s (exact-text contamination "
            "filter over the pooled corpus)",
            selection,
        )
    corpus.mkdir(parents=True, exist_ok=True)
    files: dict[tuple[str, str], str] = {}
    taken: set[str] = set()

    def write(item: dict[str, Any]) -> list[str]:
        supporting: list[str] = []
        for paragraph in item["paragraphs"]:
            title = str(paragraph.get("title") or "")
            text = str(paragraph.get("paragraph_text") or "")
            key = (title, text)
            if key not in files:
                files[key] = _file_name(title, taken)
                atomic_write_text(corpus / files[key], f"{title}\n\n{text}\n")
            if paragraph.get("is_supporting"):
                supporting.append(files[key])
        return supporting

    dataset: list[dict[str, Any]] = []
    for item in items:
        supporting = write(item)
        item_id = str(item["id"])
        dataset.append(
            {
                "query_id": item_id,
                "question": item["question"],
                "answer": item["answer"],
                "answer_aliases": list(item.get("answer_aliases") or []),
                "reference_sources": supporting,
                "category": _category(item_id),
                "metadata": {"benchmark": benchmark},
            }
        )
    for item, twin in declined:
        write(item)
        item_id = str(item["id"])
        dataset.append(
            {
                "query_id": f"{item_id}{UNANSWERABLE_ID_SUFFIX}",
                "question": item["question"],
                # No gold answer and no supporting files: scored only on
                # whether the chain declines.
                "answer": "",
                "answer_aliases": [],
                "reference_sources": [],
                "category": _category(item_id),
                "metadata": {
                    "benchmark": FULL_UNANSWERABLE,
                    "answerable": False,
                    # For auditing an answer that names it anyway (a
                    # near-duplicate paragraph or the model's own knowledge).
                    "twin_answer": twin["answer"],
                    "twin_answer_aliases": list(twin.get("answer_aliases") or []),
                },
            }
        )
    atomic_write_text(
        output_dir / "eval_data.json", json.dumps(dataset, ensure_ascii=False, indent=2)
    )
    counts: dict[str, Any] = {
        "benchmark": benchmark,
        "questions": len(dataset),
        "paragraphs": len(files),
    }
    if hops:
        counts["hops"] = {
            f"{hop}hop": sum(1 for i in items if hop_count(str(i["id"])) == hop)
            for hop in sorted(hops)
        }
    if unanswerable:
        counts["answerable"] = len(items)
        counts["unanswerable"] = len(declined)
        counts["unanswerable_filter"] = selection
    return counts


def _category(item_id: str) -> str:
    hops = hop_count(item_id)
    return f"{hops}hop" if hops is not None else item_id.split("__", 1)[0]
