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

from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit
from unified_kg_rag.domain.prompts import CorpusProfilePrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import generate_stable_id, parse_llm_json

logger = get_logger(__name__)


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
        return cls(
            domain=str(payload.get("domain") or cls.domain).strip(),
            language=str(payload.get("language") or cls.language).strip(),
            persona=str(payload.get("persona") or cls.persona).strip(),
            entity_types=[str(t).strip().upper() for t in entity_types if t],
        )


class PromptTuner:
    """Generate domain-adapted ``custom_prompts`` from a corpus sample."""

    MAX_SAMPLE_CHARS = 8000

    def __init__(
        self, config: Config, boto_session: boto3.Session | None = None
    ) -> None:
        self.config = config
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name
        )
        self.factory = BedrockLanguageModelFactory(
            config=config,
            boto_session=self.boto_session,
            region_name=config.aws.bedrock.region_name,
        )

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
            factory=self.factory,
            model_id=self.config.search.entity_extraction_model_id,
            prompt_class=CorpusProfilePrompt,
            parser=StrOutputParser(),
            custom_prompts=self.config.custom_prompts,
        )
        raw = await chain.ainvoke({"corpus_sample": corpus_sample})
        return CorpusProfile.from_payload(self._parse_json(raw))

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        # Shared degrade-to-{} LLM-JSON parser: a non-JSON/malformed response
        # yields {} so tuning falls back to the default profile rather than
        # crashing the whole run.
        return parse_llm_json(raw)

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
            extractor = GraphExtractor(self.config, boto_session=self.boto_session)
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
    def _render_example(
        text: str, entities: list[Entity], relationships: list[Relationship]
    ) -> str:
        """Render one (text → extraction) pair in the GraphExtractionPrompt shape.

        Confidence/weight are stored normalized (0.0-1.0) but the extraction
        prompt teaches a 1-10 scale, so scale back up for the demonstration.
        """

        def _esc(value: str | None) -> str:
            return (value or "").strip()

        lines = [f"EXAMPLE TEXT:\n{text.strip()}", "", "<entities>"]
        for entity in entities:
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
                    "</entity>",
                ]
            )
        lines.append("</entities>")
        lines.append("")
        lines.append("<relationships>")
        for rel in relationships:
            strength_1_10 = round((rel.weight if rel.weight else 1.0) * 10)
            lines.extend(
                [
                    "<relationship>",
                    f"<source>{_esc(rel.source_name)}</source>",
                    f"<target>{_esc(rel.target_name)}</target>",
                    f"<type>{_esc(rel.type) or 'RELATED_TO'}</type>",
                    f"<description>{_esc(rel.description)}</description>",
                    f"<strength>{strength_1_10}</strength>",
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
        """
        entity_guidance = (
            f"Focus on these domain entity types: {', '.join(profile.entity_types)}."
            if profile.entity_types
            else ""
        )
        examples_block = (
            f"\n\n# DOMAIN EXAMPLE\n{profile.few_shot_examples}"
            if profile.few_shot_examples
            else ""
        )
        graph_extraction_system = (
            f"{profile.persona}\n\n"
            f"You extract entities and relationships from {profile.domain} documents "
            f"written in {profile.language}. {entity_guidance}\n\n"
            "Follow the output format exactly as specified in the human message."
            f"{examples_block}"
        )
        community_report_system = (
            f"{profile.persona}\n\n"
            f"You analyze communities of entities and relationships extracted from "
            f"{profile.domain} documents and write reports in {profile.language}. "
            "Follow the output format exactly as specified in the human message."
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
