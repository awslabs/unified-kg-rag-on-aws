# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark assertions``: questions and assertions generated from a corpus.

A native reimplementation of the approach of Microsoft's BenchmarkQED (AutoQ
question classes and AutoE-style assertions; code MIT, Copyright (c)
Microsoft Corporation, https://github.com/microsoft/benchmark-qed). The
prompts are written here (``domain/prompts/benchmark.py``) and none of
BenchmarkQED's datasets is used. The pure decision steps (claim and
assertion parsing, the assertion ranking, the validation rule, the cluster
sampling) follow BenchmarkQED's; docs/design.md#93-length-controlled-judging-assertions-and-the-closed-book-arm
lists where and why this differs.

1. Text units: each ``.txt``/``.md`` file of the corpus directory is split
   into units of about ``unit_tokens`` estimated tokens (paragraphs packed;
   an overlong paragraph is cut by the script-aware estimate, so CJK text
   without spaces splits too), with ids ``<file>#<n>`` and character offsets.
   At most ``max_units`` are sampled with the seed and embedded.
2. Seeded k-means (k-means++ start) into ``clusters`` groups; each cluster's
   sample is the unit nearest its centroid and that unit's nearest
   untaken neighbours by cosine distance (BenchmarkQED's centroid
   representative sampling), clusters by size.
3. Questions, by class (``QUESTION_CLASSES``), with the evaluation role:
   ``data_local`` from one cluster's sample (``oversample`` x the wanted
   number, spread over the clusters); ``data_global`` from local questions
   of several clusters that share a topic label; ``data_linked`` from two
   local questions of different clusters that name a common entity;
   ``activity_local`` / ``activity_global`` from generated personas and
   tasks with one cluster's sample or the corpus description.
4. Assertions for the data classes: claims extracted from the question's
   source units (map; each claim cites unit ids), merged into at most
   ``max_assertions`` assertions citing claims (reduce; batched map-reduce
   for a global question with many claims), validated against the cited
   unit texts (grounding, relevance, verifiability 1-5; kept when every
   score is at least ``min_validation_score``), then ranked (BenchmarkQED's
   weighted reciprocal rank fusion of importance and source count). A data
   question left without an assertion is dropped.
5. ``eval_data.json`` (the run-eval format), ``text_units.jsonl`` (every
   unit's id, file, offsets and text, for review) and
   ``generation_report.json`` (seed, models, counts, drops, usage).

Every prompt names the output language. Sampling, clustering, the call
order and the selection depend only on the seed and the corpus: with the
same model replies, the same files are written. A failed or malformed call
drops the item it was for and is counted in the report.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from unified_kg_rag.adapters.evaluators.base import error_digest
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.llm.langchain import LLMJsonOutputParser
from unified_kg_rag.adapters.llm.token_estimate import (
    estimate_token_count,
    truncate_to_estimate,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, ModelPurpose, ModelRole, language_name
from unified_kg_rag.domain.models.config import model_id_value
from unified_kg_rag.domain.prompts import (
    QUESTION_CLASS_GUIDANCE,
    BasePrompt,
    BenchmarkActivityContextPrompt,
    BenchmarkAssertionsPrompt,
    BenchmarkClaimsPrompt,
    BenchmarkQuestionsPrompt,
    BenchmarkValidationPrompt,
)
from unified_kg_rag.shared import EvaluationException, get_logger
from unified_kg_rag.shared.usage import merge_usage, usage_scope

logger = get_logger(__name__)

LICENSE_NOTE = (
    "Generated from your corpus; approach after Microsoft BenchmarkQED "
    "(MIT, https://github.com/microsoft/benchmark-qed), no BenchmarkQED data used."
)
BENCHMARK_NAME = "autoq"
QUESTION_CLASSES = (
    "data_local",
    "data_global",
    "data_linked",
    "activity_local",
    "activity_global",
)
DATA_CLASSES = frozenset({"data_local", "data_global", "data_linked"})
CORPUS_SUFFIXES = frozenset({".txt", ".md"})
OUTPUT_FILES = ("eval_data.json", "text_units.jsonl", "generation_report.json")
# BenchmarkQED's assertion ranking: reciprocal rank fusion of importance and
# source count, k 60, weights 0.3 / 0.7.
RRF_K = 60
RRF_SCORE_WEIGHT = 0.3
RRF_SOURCE_WEIGHT = 0.7
# A global question's claims go to the reduce step in batches of this many
# (map: partial assertions per batch; reduce: final assertions over them).
CLAIM_BATCH = 40
_EMBED_BATCH = 64
_KMEANS_ITERATIONS = 100
_PARAGRAPH = re.compile(r"\S[\s\S]*?(?=\n[ \t\r\f\v]*\n|\Z)")


class TextUnit(BaseModel):
    """One unit of the corpus: a span of one file."""

    model_config = ConfigDict(frozen=True)

    id: str
    file: str
    start: int
    end: int
    text: str


class Claim(BaseModel):
    """A claim the source units make that helps answer a question."""

    statement: str
    sources: list[str]
    score: int


class GeneratedAssertion(BaseModel):
    """An assertion with the unit ids and claims it rests on."""

    statement: str
    sources: list[str]
    claims: list[str] = Field(default_factory=list)
    score: int = 5


class Candidate(BaseModel):
    """A generated question and what it was made from."""

    question_class: str
    question: str
    topic: str = ""
    entities: list[str] = Field(default_factory=list)
    clusters: list[int] = Field(default_factory=list)
    source_units: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    assertions: list[GeneratedAssertion] = Field(default_factory=list)
    persona: str | None = None
    task: str | None = None


class GenerationSettings(BaseModel):
    """The knobs of one generation run (``prepare-benchmark assertions``)."""

    questions_per_class: int = Field(default=10, ge=1)
    classes: list[str] = Field(default_factory=lambda: list(QUESTION_CLASSES))
    clusters: int = Field(default=20, ge=1)
    units_per_cluster: int = Field(default=8, ge=1)
    unit_tokens: int = Field(default=600, ge=20)
    max_units: int = Field(default=3000, ge=1)
    max_assertions: int = Field(default=10, ge=1)
    oversample: float = Field(default=2.0, ge=1.0)
    min_validation_score: int = Field(default=3, ge=1, le=5)
    personas: int = Field(default=5, ge=1)
    tasks_per_persona: int = Field(default=3, ge=1)
    seed: int = 0
    language: str = "en"


# --- Text units ---------------------------------------------------------


def split_units(file: str, text: str, max_tokens: int) -> list[TextUnit]:
    """``text`` as units of at most about ``max_tokens`` estimated tokens:
    paragraphs packed together, an overlong paragraph cut by the estimate."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    for match in _PARAGRAPH.finditer(text):
        p_start, p_end = match.start(), match.end()
        if estimate_token_count(text[p_start:p_end]) > max_tokens:
            if start is not None:
                spans.append((start, end))
                start = None
            spans.extend(_cut(text, p_start, p_end, max_tokens))
            continue
        if start is not None and estimate_token_count(text[start:p_end]) > max_tokens:
            spans.append((start, end))
            start = None
        if start is None:
            start = p_start
        end = p_end
    if start is not None:
        spans.append((start, end))
    return [
        TextUnit(id=f"{file}#{n}", file=file, start=s, end=e, text=text[s:e].strip())
        for n, (s, e) in enumerate(spans, start=1)
        if text[s:e].strip()
    ]


def _cut(text: str, start: int, end: int, max_tokens: int) -> Iterator[tuple[int, int]]:
    while start < end:
        piece = truncate_to_estimate(text[start:end], max_tokens) or text[start]
        yield start, start + len(piece)
        start += len(piece)
        while start < end and text[start].isspace():
            start += 1


def load_corpus(directory: Path, unit_tokens: int) -> list[TextUnit]:
    """The units of every ``.txt``/``.md`` file under ``directory`` (hidden
    files and directories skipped), files in path order.

    Raises:
        EvaluationException: No such file, or none holds text.
    """
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in CORPUS_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )
    units = [
        unit
        for path in files
        for unit in split_units(
            path.relative_to(directory).as_posix(),
            path.read_text(encoding="utf-8", errors="replace"),
            unit_tokens,
        )
    ]
    if not units:
        raise EvaluationException(
            f"'{directory}' holds no .txt or .md file with text to generate from"
        )
    return units


def sample_units(units: Sequence[TextUnit], limit: int, seed: int) -> list[TextUnit]:
    """At most ``limit`` units, drawn with ``seed``, in corpus order."""
    if len(units) <= limit:
        return list(units)
    picked = sorted(random.Random(seed).sample(range(len(units)), limit))
    return [units[i] for i in picked]


# --- Clustering ----------------------------------------------------------


def kmeans(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Cluster labels of ``vectors`` (rows) into at most ``k`` clusters:
    k-means++ start seeded with ``seed``, Lloyd iterations to convergence."""
    count = len(vectors)
    k = max(1, min(k, count))
    rng = np.random.default_rng(seed)
    squared = np.einsum("ij,ij->i", vectors, vectors)
    centers = [vectors[int(rng.integers(count))]]
    nearest = np.sum((vectors - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(nearest.sum())
        if total <= 0:
            break
        center = vectors[int(rng.choice(count, p=nearest / total))]
        centers.append(center)
        nearest = np.minimum(nearest, np.sum((vectors - center) ** 2, axis=1))
    center_array = np.asarray(centers, dtype=float)
    labels = np.zeros(count, dtype=int)
    for iteration in range(_KMEANS_ITERATIONS):
        # ||x - c||^2 without the (units x clusters x dimensions) array.
        distances = (
            squared[:, None]
            - 2.0 * vectors @ center_array.T
            + np.einsum("ij,ij->i", center_array, center_array)[None, :]
        )
        new_labels = distances.argmin(axis=1)
        if iteration and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(len(center_array)):
            members = vectors[labels == c]
            if len(members):
                center_array[c] = members.mean(axis=0)
    # Dense labels 0..n-1 in order of first appearance.
    remap = {old: new for new, old in enumerate(dict.fromkeys(labels.tolist()))}
    return np.asarray([remap[label] for label in labels.tolist()], dtype=int)


def cluster_samples(
    vectors: np.ndarray, labels: np.ndarray, per_cluster: int
) -> list[tuple[int, list[int]]]:
    """One sample per cluster, largest cluster first (ties by first
    appearance): the member nearest the cluster's centroid (Euclidean), then
    its ``per_cluster - 1`` nearest units by cosine distance among all units
    not yet taken, reps excluded (BenchmarkQED's ``KmeansTextSampler`` with
    centroid representatives). Returns ``(label, unit indices)`` pairs."""
    by_cluster: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_cluster.setdefault(int(label), []).append(index)
    ordered = sorted(by_cluster.items(), key=lambda item: -len(item[1]))
    reps: list[int] = []
    for _, members in ordered:
        member_vectors = vectors[members]
        to_centroid = np.linalg.norm(
            member_vectors - member_vectors.mean(axis=0), axis=1
        )
        reps.append(members[int(np.argmin(to_centroid))])
    if per_cluster == 1:
        return [(label, [rep]) for (label, _), rep in zip(ordered, reps, strict=True)]
    taken = set(reps)
    corpus = [i for i in range(len(vectors)) if i not in taken]
    if not corpus:
        return [(label, [rep]) for (label, _), rep in zip(ordered, reps, strict=True)]
    rep_vectors = vectors[reps]
    corpus_vectors = vectors[corpus]
    rep_norms = np.linalg.norm(rep_vectors, axis=1, keepdims=True)
    corpus_norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
    similarities = (rep_vectors / np.where(rep_norms == 0, 1.0, rep_norms)) @ (
        corpus_vectors / np.where(corpus_norms == 0, 1.0, corpus_norms)
    ).T
    distances = 1.0 - similarities
    available = np.ones(len(corpus), dtype=bool)
    neighbours = per_cluster - 1
    samples: list[tuple[int, list[int]]] = []
    for row, ((label, _), rep) in enumerate(zip(ordered, reps, strict=True)):
        distance_row = distances[row].copy()
        distance_row[~available] = np.inf
        if neighbours >= int(available.sum()):
            top = np.where(available)[0]
        else:
            top = np.argpartition(distance_row, neighbours)[:neighbours]
        top = top[np.argsort(distance_row[top])][:neighbours]
        available[top] = False
        samples.append((label, [rep, *(corpus[i] for i in top)]))
    return samples


# --- Parsing and ranking (BenchmarkQED's decision rules) ------------------


def parse_claims(output: Any, short_ids: dict[str, str]) -> list[Claim]:
    """The claims of a map reply whose ``sources`` cite given units.

    As BenchmarkQED's local claim extractor: a claim needs a non-empty
    statement and an integer score above 0; its sources are read as
    strings, those not among ``short_ids`` (shown id to unit id) dropped,
    and a claim left without one dropped.
    """
    entries = output.get("claims") if isinstance(output, dict) else None
    if not isinstance(entries, list):
        return []
    claims: list[Claim] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        statement = str(entry.get("statement") or "").strip()
        score = entry.get("score", 0)
        if not statement or not isinstance(score, int) or isinstance(score, bool):
            continue
        if score <= 0:
            continue
        sources = entry.get("sources")
        if not isinstance(sources, list):
            continue
        units = list(
            dict.fromkeys(
                short_ids[str(s).strip()]
                for s in sources
                if str(s).strip() in short_ids
            )
        )
        if units:
            claims.append(Claim(statement=statement, sources=units, score=score))
    return claims


def claims_context(claims: Sequence[Claim]) -> str:
    """Claims as the reduce prompt reads them, ids ``claim_<n>``."""
    if not claims:
        return "No claims provided."
    return "\n".join(
        f"ID: claim_{n}\nStatement: {claim.statement}\nImportance Score: {claim.score}\n"
        for n, claim in enumerate(claims, start=1)
    )


def parse_assertions(output: Any, claims: Sequence[Claim]) -> list[GeneratedAssertion]:
    """The assertions of a reduce reply, each with the units of its claims.

    As BenchmarkQED's local claim assertion generator: a statement, an
    integer score in 1..10 (5 when missing) and a non-empty source list are
    required; sources are claim ids (``claim_<n>``), unknown ones dropped,
    and an assertion with no known claim (or claims without a unit) dropped.
    """
    entries = output.get("assertions") if isinstance(output, dict) else None
    if not isinstance(entries, list):
        return []
    by_id = {f"claim_{n}": claim for n, claim in enumerate(claims, start=1)}
    parsed: list[GeneratedAssertion] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        statement = str(entry.get("statement") or "").strip()
        score = entry.get("score", 5)
        sources = entry.get("sources") or []
        if not statement or not isinstance(score, int) or isinstance(score, bool):
            continue
        if not 1 <= score <= 10 or not isinstance(sources, list) or not sources:
            continue
        cited = [str(s).strip() for s in sources if str(s).strip() in by_id]
        units = list(dict.fromkeys(u for c in cited for u in by_id[c].sources))
        if cited and units:
            parsed.append(
                GeneratedAssertion(
                    statement=statement, sources=units, claims=cited, score=score
                )
            )
    return parsed


def competition_ranks(values: Sequence[float]) -> list[int]:
    """1-based ranks, higher values first, ties sharing the best rank and the
    next rank skipping them ([10, 8, 8, 6] -> [1, 2, 2, 4])."""
    ordered = sorted(values, reverse=True)
    first: dict[float, int] = {}
    for position, value in enumerate(ordered, start=1):
        first.setdefault(value, position)
    return [first[value] for value in values]


def rank_assertions(
    assertions: Sequence[GeneratedAssertion], limit: int | None
) -> list[GeneratedAssertion]:
    """The assertions by BenchmarkQED's weighted reciprocal rank fusion of
    importance score (0.3) and source count (0.7), k 60, best first (stable),
    at most ``limit``."""
    if not assertions:
        return []
    score_ranks = competition_ranks([a.score for a in assertions])
    source_ranks = competition_ranks([len(a.sources) for a in assertions])
    fused = [
        RRF_SCORE_WEIGHT / (RRF_K + s) + RRF_SOURCE_WEIGHT / (RRF_K + c)
        for s, c in zip(score_ranks, source_ranks, strict=True)
    ]
    order = sorted(range(len(assertions)), key=lambda i: -fused[i])
    ranked = [assertions[i] for i in order]
    return ranked if limit is None else ranked[:limit]


def clamp_score(value: Any) -> int | None:
    """A validation score as an integer in 1..5 (clamped), or ``None``."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(5, number))


def parse_validation(output: Any, count: int, min_score: int) -> list[bool]:
    """Per assertion (shown 1..``count``), whether it passed validation:
    grounding, relevance and verifiability each at least ``min_score``
    (BenchmarkQED's rule). An assertion the reply does not score, or scores
    with a value that is not a number, fails (BenchmarkQED would default it
    to 3, a pass)."""
    entries = output.get("scores") if isinstance(output, dict) else None
    passed = [False] * count
    if not isinstance(entries, list):
        return passed
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(str(entry.get("id")).strip())
        except ValueError:
            continue
        if not 1 <= number <= count:
            continue
        scores = [
            clamp_score(entry.get(key))
            for key in ("grounding", "relevance", "verifiability")
        ]
        passed[number - 1] = all(s is not None and s >= min_score for s in scores)
    return passed


def _questions_of(output: Any) -> list[dict[str, Any]]:
    entries = output.get("questions") if isinstance(output, dict) else None
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("question") or "").strip()
        key = _question_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        entities = entry.get("entities")
        questions.append(
            {
                "question": text,
                "topic": str(entry.get("topic") or "").strip(),
                "entities": list(
                    dict.fromkeys(
                        str(e).strip()
                        for e in (entities if isinstance(entities, list) else [])
                        if str(e).strip()
                    )
                ),
            }
        )
    return questions


def _numbered_units(units: Sequence[TextUnit]) -> tuple[str, dict[str, str]]:
    """Units as ``Source <n>:`` blocks, and the shown ids to unit ids."""
    short = {str(n): unit.id for n, unit in enumerate(units, start=1)}
    text = "\n\n".join(
        f"Source {n} (id {n}):\n{unit.text}" for n, unit in enumerate(units, start=1)
    )
    return text, short


# --- Generation ----------------------------------------------------------


class AssertionBenchmarkGenerator:
    """Generates a question and assertion set from a corpus (module doc)."""

    def __init__(
        self,
        config: Config,
        settings: GenerationSettings,
        providers: Providers | None = None,
    ) -> None:
        unknown = [c for c in settings.classes if c not in QUESTION_CLASSES]
        if unknown or not settings.classes:
            raise EvaluationException(
                f"unknown question class {', '.join(unknown) or '(none)'} "
                f"(choose from {', '.join(QUESTION_CLASSES)})"
            )
        self.config = config
        self.settings = settings
        self.providers = Providers.resolve(config, providers)
        self.language = language_name(settings.language)
        self.model_id = config.model_id(ModelRole.EVALUATION)
        self._chains: dict[type[BasePrompt], Runnable[Any, Any]] = {}
        self._semaphore = asyncio.Semaphore(config.processing.max_concurrency)
        self.dropped: Counter[str] = Counter()
        self.usage: dict[str, dict[str, Any]] = {}
        # The sampled units by id (the validation pass reads cited texts).
        self.units: dict[str, TextUnit] = {}

    def _chain(self, prompt: type[BasePrompt], keys: tuple[str, ...]) -> Any:
        if prompt not in self._chains:
            self._chains[prompt] = setup_chain(
                self.providers.llm_factory,
                self.model_id,
                prompt,
                LLMJsonOutputParser(answer_keys=keys),
                custom_prompts=self.config.custom_prompts,
                model_purpose=ModelPurpose.EVALUATION,
            )
        return self._chains[prompt]

    async def _call(
        self,
        step: str,
        prompt: type[BasePrompt],
        keys: tuple[str, ...],
        inputs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """One model call; ``None`` (counted under ``step``) when it fails."""
        try:
            async with self._semaphore:
                output = await self._chain(prompt, keys).ainvoke(
                    {**inputs, "language": self.language}
                )
        except Exception as e:  # noqa: BLE001 - a failed call drops its item
            # The error may quote the model's reply: a digest at WARNING.
            logger.warning(
                "prepare-benchmark assertions: %s call failed: %s",
                step,
                error_digest(e),
            )
            logger.debug("prepare-benchmark assertions: %s call error: %s", step, e)
            self.dropped[f"{step}_call_failed"] += 1
            return None
        return output if isinstance(output, dict) else {}

    async def _step(
        self, name: str, calls: Sequence[Callable[[], Awaitable[Any]]]
    ) -> list[Any]:
        """Run ``calls`` concurrently in order, metering them as ``name``."""
        with usage_scope() as meter:
            results = await asyncio.gather(*(call() for call in calls))
        snapshot = meter.snapshot()
        if name in self.usage:
            snapshot = merge_usage([self.usage[name], snapshot])
        self.usage[name] = snapshot
        return list(results)

    async def _embed(self, units: Sequence[TextUnit]) -> np.ndarray:
        model = self.providers.embedding_factory.get_model(
            model_id=self.config.embedding_model_id,
            model_purpose=ModelPurpose.EVALUATION,
        )
        rows: list[list[float]] = []
        for start in range(0, len(units), _EMBED_BATCH):
            batch = [u.text for u in units[start : start + _EMBED_BATCH]]
            rows.extend(await model.aembed_documents(batch))
        return np.asarray(rows, dtype=float)

    async def agenerate(self, corpus_directory: Path) -> dict[str, Any]:
        """Generate; returns ``rows`` (the run-eval dataset), ``units`` (the
        units the rows cite) and ``report``."""
        settings = self.settings
        all_units = load_corpus(corpus_directory, settings.unit_tokens)
        units = sample_units(all_units, settings.max_units, settings.seed)
        self.units = {u.id: u for u in units}
        vectors = await self._embed(units)
        labels = kmeans(vectors, settings.clusters, settings.seed)
        samples = cluster_samples(vectors, labels, settings.units_per_cluster)
        clusters = [(label, [units[i] for i in idx]) for label, idx in samples]
        wanted = set(settings.classes)
        needs_local = bool(wanted & DATA_CLASSES)
        local = await self._local_candidates(clusters) if needs_local else []
        selected: dict[str, list[Candidate]] = {}
        if "data_local" in wanted:
            selected["data_local"] = self._round_robin(_distinct(local))
        if "data_global" in wanted:
            selected["data_global"] = await self._global_candidates(local)
        if "data_linked" in wanted:
            selected["data_linked"] = await self._linked_candidates(local)
        if wanted & {"activity_local", "activity_global"}:
            selected.update(await self._activity_candidates(clusters))
        by_id = {u.id: u for u in units}
        rows: list[dict[str, Any]] = []
        cited: set[str] = set()
        for question_class in QUESTION_CLASSES:
            unique = _distinct(selected.get(question_class, []))
            self.dropped[f"{question_class}_duplicate"] += len(
                selected.get(question_class, [])
            ) - len(unique)
            for n, candidate in enumerate(unique, start=1):
                row = self._row(question_class, n, candidate, by_id)
                cited.update(row["metadata"]["source_text_units"])
                rows.append(row)
        report = {
            "benchmark": BENCHMARK_NAME,
            "license": LICENSE_NOTE,
            "seed": settings.seed,
            "settings": settings.model_dump(),
            "language": self.language,
            "models": {
                "generation": model_id_value(self.model_id),
                "embedding": model_id_value(self.config.embedding_model_id),
            },
            "corpus": {
                "text_units": len(all_units),
                "sampled_units": len(units),
                "clusters": len(clusters),
            },
            "questions": dict(Counter(r["category"] for r in rows)),
            "assertions": sum(len(r.get("assertions", [])) for r in rows),
            "dropped": {k: v for k, v in sorted(self.dropped.items()) if v},
            "usage": self.usage,
        }
        return {
            "rows": rows,
            "units": [u for u in units if u.id in cited],
            "report": report,
        }

    def _row(
        self,
        question_class: str,
        n: int,
        candidate: Candidate,
        by_id: dict[str, TextUnit],
    ) -> dict[str, Any]:
        sources = candidate.source_units
        if candidate.assertions:
            sources = list(
                dict.fromkeys(u for a in candidate.assertions for u in a.sources)
            )
        row: dict[str, Any] = {
            "query_id": f"{BENCHMARK_NAME}-{question_class.replace('_', '-')}-{n:03d}",
            "question": candidate.question,
            "category": question_class,
            "metadata": {
                "benchmark": BENCHMARK_NAME,
                "question_class": question_class,
                "topic": candidate.topic,
                "entities": candidate.entities,
                "clusters": candidate.clusters,
                "source_text_units": sources,
                **(
                    {"persona": candidate.persona, "task": candidate.task}
                    if candidate.persona
                    else {}
                ),
            },
        }
        if question_class in DATA_CLASSES:
            row["assertions"] = [
                {"text": a.statement, "kind": "positive", "source": a.sources}
                for a in candidate.assertions
            ]
            row["metadata"]["assertion_importance"] = [
                a.score for a in candidate.assertions
            ]
            row["reference_sources"] = list(
                dict.fromkeys(Path(by_id[u].file).name for u in sources if u in by_id)
            )
        return row

    def _per_cluster(self, clusters: int) -> int:
        candidates = math.ceil(
            self.settings.questions_per_class * self.settings.oversample
        )
        return max(1, math.ceil(candidates / max(1, clusters)))

    async def _local_candidates(
        self, clusters: Sequence[tuple[int, list[TextUnit]]]
    ) -> list[Candidate]:
        """Data-local candidates with claims and validated assertions, in
        cluster order then generation order; candidates without claims or
        assertions dropped."""
        per_cluster = self._per_cluster(len(clusters))

        def ask(units: list[TextUnit]) -> Callable[[], Awaitable[Any]]:
            text, _ = _numbered_units(units)
            return lambda: self._call(
                "data_local_questions",
                BenchmarkQuestionsPrompt,
                ("questions",),
                {
                    "question_class": "data_local",
                    "class_guidance": QUESTION_CLASS_GUIDANCE["data_local"],
                    "context": text,
                    "num_questions": per_cluster,
                },
            )

        replies = await self._step("questions", [ask(u) for _, u in clusters])
        candidates: list[Candidate] = []
        sources: list[list[TextUnit]] = []
        for (label, units), reply in zip(clusters, replies, strict=True):
            for item in _questions_of(reply)[:per_cluster]:
                candidates.append(
                    Candidate(
                        question_class="data_local",
                        clusters=[label],
                        source_units=[u.id for u in units],
                        **item,
                    )
                )
                sources.append(units)
        claims = await self._step(
            "claims",
            [
                self._claims_call(c.question, units)
                for c, units in zip(candidates, sources, strict=True)
            ],
        )
        kept: list[tuple[Candidate, list[TextUnit]]] = []
        for candidate, units, found in zip(candidates, sources, claims, strict=True):
            if not found:
                self.dropped["data_local_no_claims"] += 1
                continue
            candidate.claims = found
            kept.append((candidate, units))
        await self._assert_all("data_local", [c for c, _ in kept], batched=False)
        return [c for c, _ in kept if c.assertions]

    def _claims_call(
        self, question: str, units: Sequence[TextUnit]
    ) -> Callable[[], Awaitable[list[Claim]]]:
        text, short = _numbered_units(units)

        async def call() -> list[Claim]:
            reply = await self._call(
                "claims",
                BenchmarkClaimsPrompt,
                ("claims",),
                {"question": question, "text_units": text},
            )
            return parse_claims(reply, short) if reply is not None else []

        return call

    async def _assert_all(
        self, question_class: str, candidates: Sequence[Candidate], *, batched: bool
    ) -> None:
        """Reduce each candidate's claims to assertions, validate and rank
        them (in place); ``batched`` maps claims in batches first."""
        reduced = await self._step(
            "assertions",
            [self._reduce_call(c, batched=batched) for c in candidates],
        )
        verdicts = await self._step(
            "validation",
            [
                self._validation_call(c.question, found)
                for c, found in zip(candidates, reduced, strict=True)
            ],
        )
        for candidate, found, passed in zip(candidates, reduced, verdicts, strict=True):
            valid = [a for a, ok in zip(found, passed, strict=True) if ok]
            self.dropped[f"{question_class}_assertions_invalid"] += len(found) - len(
                valid
            )
            candidate.assertions = rank_assertions(valid, self.settings.max_assertions)
            if not candidate.assertions:
                self.dropped[f"{question_class}_no_assertions"] += 1

    def _reduce_call(
        self, candidate: Candidate, *, batched: bool
    ) -> Callable[[], Awaitable[list[GeneratedAssertion]]]:
        async def reduce(claims: Sequence[Claim]) -> list[GeneratedAssertion]:
            reply = await self._call(
                "assertions",
                BenchmarkAssertionsPrompt,
                ("assertions",),
                {
                    "question": candidate.question,
                    "claims": claims_context(claims),
                    "max_assertions": self.settings.max_assertions,
                },
            )
            return parse_assertions(reply, claims) if reply is not None else []

        async def call() -> list[GeneratedAssertion]:
            claims = candidate.claims
            if not batched or len(claims) <= CLAIM_BATCH:
                return await reduce(claims)
            # Map: partial assertions per batch; reduce: final ones over the
            # partial assertions, read as claims citing their units.
            partial = await asyncio.gather(
                *(
                    reduce(claims[i : i + CLAIM_BATCH])
                    for i in range(0, len(claims), CLAIM_BATCH)
                )
            )
            merged = [
                Claim(statement=a.statement, sources=a.sources, score=a.score)
                for group in partial
                for a in group
            ]
            return await reduce(merged) if merged else []

        return call

    def _validation_call(
        self, question: str, assertions: Sequence[GeneratedAssertion]
    ) -> Callable[[], Awaitable[list[bool]]]:
        async def call() -> list[bool]:
            if not assertions:
                return []
            cited = list(dict.fromkeys(u for a in assertions for u in a.sources))
            short = {unit: str(n) for n, unit in enumerate(cited, start=1)}
            units = self.units
            missing = [u for u in cited if u not in units]
            if missing:
                raise EvaluationException(f"unknown text units {missing}")
            reply = await self._call(
                "validation",
                BenchmarkValidationPrompt,
                ("scores",),
                {
                    "question": question,
                    "assertions": "\n".join(
                        f"{n}. {a.statement} (sources: "
                        f"{', '.join(short[u] for u in a.sources)})"
                        for n, a in enumerate(assertions, start=1)
                    ),
                    "sources": "\n\n".join(
                        f"Source {short[u]}:\n{units[u].text}" for u in cited
                    ),
                },
            )
            if reply is None:
                return [False] * len(assertions)
            return parse_validation(
                reply, len(assertions), self.settings.min_validation_score
            )

        return call

    def _round_robin(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """``questions_per_class`` candidates, one per cluster in turn (in
        cluster order, each cluster's in generation order)."""
        queues: dict[int, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            queues[candidate.clusters[0] if candidate.clusters else -1].append(
                candidate
            )
        order = list(
            dict.fromkeys(c.clusters[0] if c.clusters else -1 for c in candidates)
        )
        picked: list[Candidate] = []
        while len(picked) < self.settings.questions_per_class and any(
            queues[label] for label in order
        ):
            for label in order:
                if queues[label] and len(picked) < self.settings.questions_per_class:
                    picked.append(queues[label].pop(0))
        return picked

    async def _global_candidates(self, local: Sequence[Candidate]) -> list[Candidate]:
        """Data-global questions from topic groups of local candidates that
        span two or more clusters (largest first; with too few, one group of
        every cluster's first candidate), assertions from their claims."""
        groups: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in local:
            groups[" ".join(candidate.topic.casefold().split())].append(candidate)
        spanning = sorted(
            (
                (topic, members)
                for topic, members in groups.items()
                if topic and len({c.clusters[0] for c in members}) > 1
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )
        wanted = math.ceil(self.settings.questions_per_class * self.settings.oversample)
        chosen = [members for _, members in spanning[:wanted]]
        if len(chosen) < wanted:
            firsts = list({c.clusters[0]: c for c in reversed(local)}.values())[::-1]
            if len({c.clusters[0] for c in firsts}) > 1:
                chosen.append(sorted(firsts, key=lambda c: c.clusters[0]))
        per_group = max(1, math.ceil(wanted / max(1, len(chosen))))

        def ask(members: list[Candidate]) -> Callable[[], Awaitable[Any]]:
            context = "\n".join(f"- {c.question}" for c in members)
            return lambda: self._call(
                "data_global_questions",
                BenchmarkQuestionsPrompt,
                ("questions",),
                {
                    "question_class": "data_global",
                    "class_guidance": QUESTION_CLASS_GUIDANCE["data_global"],
                    "context": context,
                    "num_questions": per_group,
                },
            )

        replies = await self._step("questions", [ask(m) for m in chosen])
        candidates: list[Candidate] = []
        for members, reply in zip(chosen, replies, strict=True):
            claims = _merged_claims(members)
            for item in _questions_of(reply)[:per_group]:
                candidates.append(
                    Candidate(
                        question_class="data_global",
                        clusters=sorted({c.clusters[0] for c in members}),
                        source_units=list(
                            dict.fromkeys(u for c in members for u in c.source_units)
                        ),
                        claims=claims,
                        **item,
                    )
                )
        await self._assert_all("data_global", candidates, batched=True)
        kept = _distinct([c for c in candidates if c.assertions])
        return kept[: self.settings.questions_per_class]

    async def _linked_candidates(self, local: Sequence[Candidate]) -> list[Candidate]:
        """Data-linked questions from pairs of local candidates of different
        clusters that name a common entity (by entity, then cluster order),
        assertions from both candidates' claims (no map-reduce)."""
        by_entity: dict[str, list[Candidate]] = defaultdict(list)
        names: dict[str, str] = {}
        for candidate in local:
            for entity in dict.fromkeys(
                " ".join(e.casefold().split()) for e in candidate.entities
            ):
                by_entity[entity].append(candidate)
        for candidate in local:
            for entity in candidate.entities:
                names.setdefault(" ".join(entity.casefold().split()), entity)
        wanted = math.ceil(self.settings.questions_per_class * self.settings.oversample)
        pairs: list[tuple[str, Candidate, Candidate]] = []
        used: set[tuple[int, int]] = set()
        index = {id(c): n for n, c in enumerate(local)}
        for entity in sorted(by_entity):
            members = by_entity[entity]
            for i, first in enumerate(members):
                for second in members[i + 1 :]:
                    key = (index[id(first)], index[id(second)])
                    if first.clusters[0] == second.clusters[0] or key in used:
                        continue
                    used.add(key)
                    pairs.append((entity, first, second))
        pairs = pairs[:wanted]

        def ask(
            entity: str, a: Candidate, b: Candidate
        ) -> Callable[[], Awaitable[Any]]:
            context = (
                f"Common entity: {names[entity]}\n\n"
                f"Question 1: {a.question}\nClaims 1:\n"
                + "\n".join(f"- {c.statement}" for c in a.claims)
                + f"\n\nQuestion 2: {b.question}\nClaims 2:\n"
                + "\n".join(f"- {c.statement}" for c in b.claims)
            )
            return lambda: self._call(
                "data_linked_questions",
                BenchmarkQuestionsPrompt,
                ("questions",),
                {
                    "question_class": "data_linked",
                    "class_guidance": QUESTION_CLASS_GUIDANCE["data_linked"],
                    "context": context,
                    "num_questions": 1,
                },
            )

        replies = await self._step("questions", [ask(*p) for p in pairs])
        candidates: list[Candidate] = []
        for (entity, a, b), reply in zip(pairs, replies, strict=True):
            for item in _questions_of(reply)[:1]:
                item["entities"] = list(
                    dict.fromkeys([names[entity], *item["entities"]])
                )
                candidates.append(
                    Candidate(
                        question_class="data_linked",
                        clusters=sorted({a.clusters[0], b.clusters[0]}),
                        source_units=list(
                            dict.fromkeys([*a.source_units, *b.source_units])
                        ),
                        claims=_merged_claims([a, b]),
                        **item,
                    )
                )
        await self._assert_all("data_linked", candidates, batched=False)
        kept = _distinct([c for c in candidates if c.assertions])
        return kept[: self.settings.questions_per_class]

    async def _activity_candidates(
        self, clusters: Sequence[tuple[int, list[TextUnit]]]
    ) -> dict[str, list[Candidate]]:
        """Activity questions: personas and tasks from a cross-cluster sample
        (one call), then per (persona, task) in turn one question about a
        cluster's sample (local) or the corpus description (global)."""
        settings = self.settings
        sample = "\n\n".join(
            truncate_to_estimate(units[0].text, settings.unit_tokens // 2)
            for _, units in clusters
            if units
        )
        (context,) = await self._step(
            "activity_context",
            [
                lambda: self._call(
                    "activity_context",
                    BenchmarkActivityContextPrompt,
                    ("description", "personas"),
                    {
                        "context": sample,
                        "num_personas": settings.personas,
                        "num_tasks": settings.tasks_per_persona,
                    },
                )
            ],
        )
        description = str((context or {}).get("description") or "").strip()
        tasks: list[tuple[str, str]] = []
        for persona in (context or {}).get("personas") or []:
            if not isinstance(persona, dict):
                continue
            role = str(persona.get("persona") or "").strip()
            for task in persona.get("tasks") or []:
                if role and str(task).strip():
                    tasks.append((role, str(task).strip()))
        if not tasks:
            self.dropped["activity_no_personas"] += 1
            return {}
        out: dict[str, list[Candidate]] = {}
        for question_class in ("activity_local", "activity_global"):
            if question_class not in settings.classes:
                continue
            jobs: list[tuple[str, str, int, list[TextUnit]]] = []
            for n in range(settings.questions_per_class):
                role, task = tasks[n % len(tasks)]
                label, units = clusters[n % len(clusters)]
                jobs.append((role, task, label, units))
            replies = await self._step(
                "questions",
                [
                    self._activity_call(question_class, description, *job)
                    for job in jobs
                ],
            )
            candidates: list[Candidate] = []
            for (role, task, label, units), reply in zip(jobs, replies, strict=True):
                items = _questions_of(reply)[:1]
                if not items:
                    self.dropped[f"{question_class}_no_question"] += 1
                    continue
                local = question_class == "activity_local"
                candidates.append(
                    Candidate(
                        question_class=question_class,
                        clusters=[label] if local else [],
                        source_units=[u.id for u in units] if local else [],
                        persona=role,
                        task=task,
                        **items[0],
                    )
                )
            out[question_class] = candidates
        return out

    def _activity_call(
        self,
        question_class: str,
        description: str,
        role: str,
        task: str,
        label: int,
        units: list[TextUnit],
    ) -> Callable[[], Awaitable[Any]]:
        if question_class == "activity_local":
            body, _ = _numbered_units(units)
        else:
            body = f"Corpus description: {description}"
        context = f"Persona: {role}\nTask: {task}\n\n{body}"
        return lambda: self._call(
            f"{question_class}_questions",
            BenchmarkQuestionsPrompt,
            ("questions",),
            {
                "question_class": question_class,
                "class_guidance": QUESTION_CLASS_GUIDANCE[question_class],
                "context": context,
                "num_questions": 1,
            },
        )


def _question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def _distinct(candidates: Sequence[Candidate]) -> list[Candidate]:
    """``candidates`` without a repeated question (case and spacing folded)."""
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = _question_key(candidate.question)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _merged_claims(members: Sequence[Candidate]) -> list[Claim]:
    """The claims of ``members``, each once (by statement and sources)."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    claims: list[Claim] = []
    for member in members:
        for claim in member.claims:
            key = (claim.statement, tuple(claim.sources))
            if key not in seen:
                seen.add(key)
                claims.append(claim)
    return claims


def check_output_directory(output_dir: Path) -> None:
    """Raise when ``output_dir`` already holds a generated file: a second run
    must not overwrite a reviewed set."""
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing:
        raise EvaluationException(
            f"'{output_dir}' already holds {', '.join(existing)}; choose a new "
            "--output-directory"
        )


def write_outputs(output_dir: Path, generated: dict[str, Any]) -> None:
    """Write ``eval_data.json``, ``text_units.jsonl`` and
    ``generation_report.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_data.json").write_text(
        json.dumps(generated["rows"], ensure_ascii=False, indent=2), "utf-8"
    )
    (output_dir / "text_units.jsonl").write_text(
        "".join(
            json.dumps(unit.model_dump(), ensure_ascii=False) + "\n"
            for unit in generated["units"]
        ),
        "utf-8",
    )
    (output_dir / "generation_report.json").write_text(
        json.dumps(generated["report"], ensure_ascii=False, indent=2), "utf-8"
    )
