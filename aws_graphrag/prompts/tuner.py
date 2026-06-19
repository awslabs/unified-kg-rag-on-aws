# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Automatic prompt tuning (MS GraphRAG ``prompt_tune`` ported AWS-native).

Samples a corpus, asks a Bedrock model to profile its domain/language/persona/
entity-types, then emits ``custom_prompts`` overrides adapted to that profile.
The output is a YAML-ready dict the user pastes under ``custom_prompts`` in
their config — keeping prompt tuning a deliberate, reviewable step rather than
opaque runtime behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser

from aws_graphrag.aws import BedrockLanguageModelFactory
from aws_graphrag.core import get_logger
from aws_graphrag.models import Config
from aws_graphrag.prompts import CorpusProfilePrompt
from aws_graphrag.utils import setup_chain

logger = get_logger(__name__)


@dataclass
class CorpusProfile:
    """Structured corpus characterization produced during tuning."""

    domain: str = "general knowledge"
    language: str = "English"
    persona: str = "You are an expert knowledge-graph extraction specialist."
    entity_types: list[str] = field(default_factory=list)

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
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def build_custom_prompts(profile: CorpusProfile) -> dict[str, str]:
        """Turn a profile into ``custom_prompts`` override strings.

        Currently adapts the graph-extraction system prompt with the domain
        persona and entity-type guidance; the structure makes it easy to extend
        to other prompts.
        """
        entity_guidance = (
            f"Focus on these domain entity types: {', '.join(profile.entity_types)}."
            if profile.entity_types
            else ""
        )
        graph_extraction_system = (
            f"{profile.persona}\n\n"
            f"You extract entities and relationships from {profile.domain} documents "
            f"written in {profile.language}. {entity_guidance}\n\n"
            "Follow the output format exactly as specified in the human message."
        )
        return {"graph_extraction_system": graph_extraction_system}

    async def tune(self, texts: list[str]) -> dict[str, Any]:
        """End-to-end: profile the corpus and return a custom_prompts dict."""
        profile = await self.profile_corpus(texts)
        logger.info(
            "Corpus profile: domain='%s', language='%s', %d entity types",
            profile.domain,
            profile.language,
            len(profile.entity_types),
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
