# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automatic prompt tuning (MS GraphRAG ``prompt_tune`` ported AWS-native).

Samples a corpus, asks a Bedrock model to profile its domain/language/persona/
entity-types, and grounds few-shot extraction examples by running the real
``GraphExtractor`` over sampled chunks (capturing genuine input→output pairs,
as MS GraphRAG does), then emits ``custom_prompts`` overrides adapted to that
profile. The output is a YAML-ready dict the user pastes under ``custom_prompts``
in their config — keeping prompt tuning a deliberate, reviewable step rather
than opaque runtime behaviour.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    ModelPurpose,
    Relationship,
    TextUnit,
)
from unified_kg_rag.domain.prompts import (
    CommunityReportPrompt,
    CorpusProfilePrompt,
    GraphExtractionPrompt,
)
from unified_kg_rag.shared import LanguageModelError, get_logger
from unified_kg_rag.shared.utils import generate_stable_id, parse_llm_json

logger = get_logger(__name__)

# Sentence or line boundaries for picking a verbatim evidence span. Latin
# sentence-final punctuation needs following whitespace (so "3.5" or "e.g."
# mid-token is not a boundary); CJK terminators (。！？) end a sentence on
# their own, since CJK text has no space between sentences.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*|\n+")
# Target length of an evidence span cut from a long sentence.
_MAX_EVIDENCE_CHARS = 200
# Hard cap when the names themselves lie further apart than the target; a
# sentence whose names cannot fit in this many characters gives no span.
_MAX_EVIDENCE_SPAN_CHARS = 400


def _escape_braces(text: str) -> str:
    """Make ``text`` literal inside a LangChain f-string prompt template."""
    return text.replace("{", "{{").replace("}", "}}")


@dataclass
class CorpusProfile:
    """Structured corpus characterization produced during tuning."""

    domain: str = "general knowledge"
    language: str = "English"
    persona: str = "You are an expert knowledge-graph extraction specialist."
    entity_types: list[str] = field(default_factory=list)
    few_shot_examples: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CorpusProfile:
        entity_types = payload.get("entity_types") or []
        if isinstance(entity_types, str):
            # A model may answer "PERSON, ORGANIZATION" instead of a list.
            entity_types = re.split(r"[,;\n]", entity_types)
        elif not isinstance(entity_types, list):
            entity_types = []
        return cls(
            domain=str(payload.get("domain") or cls.domain).strip(),
            language=str(payload.get("language") or cls.language).strip(),
            persona=str(payload.get("persona") or cls.persona).strip(),
            entity_types=[
                str(t).strip().upper() for t in entity_types if str(t).strip()
            ],
        )


class PromptTuner:
    """Generate domain-adapted ``custom_prompts`` from a corpus sample."""

    MAX_SAMPLE_CHARS = 8000

    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        *,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        self.providers = Providers.resolve(config, providers, boto_session)
        self.boto_session = self.providers.boto_session
        self.factory = self.providers.llm_factory

    def sample_corpus(self, texts: list[str]) -> str:
        """Concatenate document texts up to the sampling budget."""
        sample, total = [], 0
        for text in texts:
            if not text:
                continue
            remaining = self.MAX_SAMPLE_CHARS - total
            if remaining <= 0:
                break
            snippet = text[:remaining]
            sample.append(snippet)
            total += len(snippet)
        return "\n\n---\n\n".join(sample)

    async def profile_corpus(self, texts: list[str]) -> CorpusProfile:
        """Run the profiling LLM over a corpus sample."""
        corpus_sample = self.sample_corpus(texts)
        if not corpus_sample:
            logger.warning("Empty corpus sample; returning default profile")
            return CorpusProfile()

        chain = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.config.search.entity_extraction_model_id,
            prompt_class=CorpusProfilePrompt,
            parser=StrOutputParser(),
            custom_prompts=self.config.custom_prompts,
        )
        raw = await chain.ainvoke({"corpus_sample": corpus_sample})
        try:
            payload = parse_llm_json(raw, strict=True)
        except json.JSONDecodeError as e:
            # A default profile here would yield generic prompts presented as
            # tuned ones; fail so the user can retry or fix the model.
            raise LanguageModelError(
                f"Corpus profiling returned no JSON profile: {str(raw)[:200]!r}"
            ) from e
        if not payload:
            logger.warning("Corpus profile is empty; using the default profile")
        return CorpusProfile.from_payload(payload)

    MAX_EXAMPLES = 3
    EXAMPLE_CHUNK_CHARS = 1200

    def _sample_chunks(self, texts: list[str]) -> list[str]:
        """Slice the corpus into up to ``MAX_EXAMPLES`` example-sized chunks.

        Few-shot grounding wants a handful of *real, self-contained* passages,
        not the whole concatenated sample. We take the leading slice of distinct
        documents (then fall back to slicing within one document) so the worked
        examples span the corpus rather than repeating one document's opening.
        """
        chunks: list[str] = []
        for text in texts:
            stripped = (text or "").strip()
            if not stripped:
                continue
            chunks.append(stripped[: self.EXAMPLE_CHUNK_CHARS])
            if len(chunks) >= self.MAX_EXAMPLES:
                break
        # Single long document: slice it into multiple non-overlapping windows so
        # we still get several distinct examples.
        if len(chunks) == 1 and len(texts) == 1:
            whole = (texts[0] or "").strip()
            chunks = [
                whole[i : i + self.EXAMPLE_CHUNK_CHARS]
                for i in range(0, len(whole), self.EXAMPLE_CHUNK_CHARS)
            ][: self.MAX_EXAMPLES]
        return [c for c in chunks if c.strip()]

    async def generate_examples(self, profile: CorpusProfile, texts: list[str]) -> str:
        """Generate corpus-grounded few-shot extraction examples.

        MS GraphRAG's prompt-tune step grounds few-shots in the real corpus by
        running actual entity/relationship extraction over sampled chunks and
        embedding those genuine input→output pairs into the tuned prompt (rather
        than asking the model to invent a representative example). We do the
        same: run the real ``GraphExtractor`` over a few sampled chunks and
        render each ``(chunk text → extracted entities/relationships)`` pair in
        the exact XML shape the extraction prompt teaches. Returns an empty
        string if the corpus is empty or extraction yields nothing (extraction
        still works without examples, so this degrades gracefully).
        """
        chunks = self._sample_chunks(texts)
        if not chunks:
            return ""

        text_units = [
            TextUnit.model_validate(
                {
                    "id": generate_stable_id(f"tune-example:{idx}:{chunk}"),
                    "text": chunk,
                }
            )
            for idx, chunk in enumerate(chunks)
        ]
        try:
            extractor = GraphExtractor(self.config, providers=self.providers)
            extractor.show_progress = False
            entities, relationships, _ = extractor.extract_from_text_units(text_units)
        except Exception as exc:  # noqa: BLE001 - examples are best-effort
            logger.warning("Corpus-grounded example extraction failed: %s", exc)
            return ""

        # Group the real extraction output back by source chunk so each rendered
        # example pairs a genuine passage with what was actually extracted from it.
        rendered: list[str] = []
        for unit in text_units:
            unit_entities = [e for e in entities if unit.id in (e.text_unit_ids or [])]
            unit_relationships = [
                r for r in relationships if unit.id in (r.text_unit_ids or [])
            ]
            if not unit_entities and not unit_relationships:
                continue
            rendered.append(
                self._render_example(unit.text, unit_entities, unit_relationships)
            )
            if len(rendered) >= self.MAX_EXAMPLES:
                break

        return "\n\n".join(rendered).strip()

    @staticmethod
    def _evidence_span(text: str, *names: str) -> str | None:
        """Return a short verbatim span of ``text`` that mentions every name.

        The extractor strips the model's ``source_text`` once grounding has
        checked it, so the example re-derives one: the first sentence (or line)
        that contains all ``names``. A long sentence is cut to a window that
        covers one mention of every name, padded to ``_MAX_EVIDENCE_CHARS``
        and at most ``_MAX_EVIDENCE_SPAN_CHARS`` long. ``None`` when no
        sentence mentions them all within that bound.
        """
        wanted = [n.strip() for n in names if n and n.strip()]
        if not wanted:
            return None
        folded_wanted = [n.casefold() for n in wanted]
        for sentence in _SENTENCE_SPLIT.split(text):
            span = sentence.strip()
            folded = span.casefold()
            if not span or not all(n in folded for n in folded_wanted):
                continue
            if len(span) <= _MAX_EVIDENCE_CHARS:
                return span
            window = PromptTuner._covering_window(span, wanted)
            if window is not None:
                return window
        return None

    @staticmethod
    def _casefold_with_origin(text: str) -> tuple[str, list[int]]:
        """Casefold ``text`` and map each folded character to its source index.

        ``str.casefold`` can change length ("ß" folds to "ss"), so offsets in
        the folded copy are not offsets in ``text``; ``origin[i]`` is the index
        in ``text`` of the character that produced folded character ``i``.
        """
        parts: list[str] = []
        origin: list[int] = []
        for index, char in enumerate(text):
            folded = char.casefold()
            parts.append(folded)
            origin.extend([index] * len(folded))
        return "".join(parts), origin

    @staticmethod
    def _covering_window(span: str, names: list[str]) -> str | None:
        """Shortest slice of ``span`` holding one mention of each of ``names``.

        Padded with surrounding text up to ``_MAX_EVIDENCE_CHARS``; ``None``
        when the mentions are more than ``_MAX_EVIDENCE_SPAN_CHARS`` apart.
        Names are matched casefolded, the same test _evidence_span filters
        sentences with, so a name whose casefold changes length ("Straße" vs
        "STRASSE") is still located.
        """
        folded, origin = PromptTuner._casefold_with_origin(span)
        mentions = [
            [
                (origin[m.start()], origin[m.end() - 1] + 1)
                for m in re.finditer(re.escape(name.casefold()), folded)
            ]
            for name in names
        ]
        if not all(mentions):
            return None
        best: tuple[int, int] | None = None
        # The shortest covering window starts at some mention; for each start,
        # take every other name's first mention at or after it.
        for start, _ in sorted({m for occ in mentions for m in occ}):
            ends = [
                min((e for s, e in occ if s >= start), default=-1) for occ in mentions
            ]
            if -1 in ends:
                continue
            end = max(ends)
            if best is None or end - start < best[1] - best[0]:
                best = (start, end)
        if best is None or best[1] - best[0] > _MAX_EVIDENCE_SPAN_CHARS:
            return None
        lo, hi = best
        pad = max(0, _MAX_EVIDENCE_CHARS - (hi - lo))
        lo = max(0, lo - pad // 2)
        hi = min(len(span), max(hi, lo + _MAX_EVIDENCE_CHARS))
        lo = max(0, min(lo, hi - _MAX_EVIDENCE_CHARS))
        return span[lo:hi].strip()

    @classmethod
    def _render_example(
        cls, text: str, entities: list[Entity], relationships: list[Relationship]
    ) -> str:
        """Render one (text → extraction) pair in the GraphExtractionPrompt shape.

        Confidence is stored normalized (0.0-1.0) and weight as the summed
        1-10 strength, while the extraction prompt teaches a 1-10 scale for
        both, so both are mapped back onto it for the demonstration.
        Every record carries a verbatim ``<source_text>`` span from the example
        text, as the extraction rules require; a record no sentence of the text
        supports is left out rather than shown without evidence.
        """

        def _esc(value: str | None) -> str:
            return (value or "").strip()

        lines = [f"EXAMPLE TEXT:\n{text.strip()}", "", "<entities>"]
        for entity in entities:
            span = cls._evidence_span(text, entity.name or "")
            if span is None:
                continue
            confidence_1_10 = round(
                (entity.confidence if entity.confidence else 1.0) * 10
            )
            lines.extend(
                [
                    "<entity>",
                    f"<name>{_esc(entity.name)}</name>",
                    f"<type>{_esc(entity.type) or 'ENTITY'}</type>",
                    f"<description>{_esc(entity.description)}</description>",
                    f"<confidence>{confidence_1_10}</confidence>",
                    f"<source_text>{span}</source_text>",
                    "</entity>",
                ]
            )
        lines.append("</entities>")
        lines.append("")
        lines.append("<relationships>")
        for rel in relationships:
            span = cls._evidence_span(
                text, rel.source_name or "", rel.target_name or ""
            )
            if span is None:
                continue
            # Weight is the summed 1-10 strength of the relationship's
            # extractions; clamp it back to the scale the prompt teaches.
            strength_1_10 = min(10, max(1, round(rel.weight or 1.0)))
            lines.extend(
                [
                    "<relationship>",
                    f"<source>{_esc(rel.source_name)}</source>",
                    f"<target>{_esc(rel.target_name)}</target>",
                    f"<type>{_esc(rel.type) or 'RELATED_TO'}</type>",
                    f"<description>{_esc(rel.description)}</description>",
                    f"<strength>{strength_1_10}</strength>",
                    f"<source_text>{span}</source_text>",
                    "</relationship>",
                ]
            )
        lines.append("</relationships>")
        return "\n".join(lines)

    @staticmethod
    def build_custom_prompts(profile: CorpusProfile) -> dict[str, str]:
        """Turn a profile into ``custom_prompts`` override strings.

        Adapts both extraction-side prompts (graph extraction with persona,
        entity-type guidance, and any generated few-shot example) and the
        community-report persona, so the whole indexing pipeline speaks the
        corpus's domain — not just entity extraction.

        A ``*_system`` override replaces the whole system prompt, and the
        built-in one is where the XML output schema and the extraction rules
        (verbatim ``source_text`` grounding, no invented entities) live. So each
        tuned system prompt is a domain-adapted preamble followed by the
        prompt's built-in ``output_rules`` verbatim, then any examples. The
        result is a complete, self-contained override.

        The overrides are LangChain f-string templates. Every corpus- or
        model-derived field has its braces doubled (a JSON sample would
        otherwise break formatting and an ``{input_text}`` in it would be
        substituted); the built-in rules are appended unescaped, so their
        variables (``{entity_types}``) are still filled at call time.
        """
        persona = _escape_braces(profile.persona)
        domain = _escape_braces(profile.domain)
        language = _escape_braces(profile.language)
        entity_types = _escape_braces(", ".join(profile.entity_types))
        examples = _escape_braces(profile.few_shot_examples)
        entity_guidance = (
            f" Entity types common in this corpus: {entity_types}."
            if profile.entity_types
            else ""
        )
        examples_block = (
            "\n\n# DOMAIN EXAMPLES\n"
            "Worked examples from this corpus. They illustrate the format; the "
            "rules above take precedence.\n\n"
            f"{examples}"
            if examples
            else ""
        )
        graph_extraction_system = (
            f"{persona}\n\n"
            f"You extract entities and relationships from {domain} documents "
            f"written in {language}.{entity_guidance} Follow the extraction "
            "rules and the output format below exactly.\n\n"
            f"{GraphExtractionPrompt.output_rules}"
            f"{examples_block}"
        )
        community_report_system = (
            f"{persona}\n\n"
            f"You analyze communities of entities and relationships extracted from "
            f"{domain} documents and write reports in {language}. Follow the "
            "report requirements and the output format below exactly.\n\n"
            f"{CommunityReportPrompt.output_rules}"
        )
        return {
            "graph_extraction_system": graph_extraction_system,
            "community_report_system": community_report_system,
        }

    async def tune(self, texts: list[str]) -> dict[str, Any]:
        """End-to-end: profile the corpus, generate examples, return overrides."""
        profile = await self.profile_corpus(texts)
        profile.few_shot_examples = await self.generate_examples(profile, texts)
        logger.info(
            "Corpus profile: domain='%s', language='%s', %d entity types, examples=%s",
            profile.domain,
            profile.language,
            len(profile.entity_types),
            "yes" if profile.few_shot_examples else "no",
        )
        return {
            "profile": {
                "domain": profile.domain,
                "language": profile.language,
                "persona": profile.persona,
                "entity_types": profile.entity_types,
            },
            "custom_prompts": self.build_custom_prompts(profile),
        }
