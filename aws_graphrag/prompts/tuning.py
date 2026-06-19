# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Prompts for automatic prompt tuning.

Ports the intent of Microsoft GraphRAG's ``prompt_tune/generator`` (domain
detection, persona, entity types) into a single corpus-analysis prompt that
emits a structured JSON profile. That profile then parameterizes domain-adapted
``custom_prompts`` fragments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import BasePrompt

if TYPE_CHECKING:
    from ..models.config import CustomPromptConfig


@dataclass(frozen=True)
class CorpusProfilePrompt(BasePrompt):
    """Analyze a corpus sample and emit a domain/persona/entity-type profile."""

    input_variables = ["corpus_sample"]
    output_variables = ["domain", "language", "persona", "entity_types"]

    @classmethod
    def _get_custom_prompts(
        cls, custom_prompts: CustomPromptConfig
    ) -> tuple[str | None, str | None]:
        return (
            custom_prompts.corpus_profile_system,
            custom_prompts.corpus_profile_human,
        )

    system_prompt_template = """You are an expert corpus analyst configuring a knowledge-graph
extraction system for a specific domain. Analyze the provided document sample and produce a
structured profile that will adapt the extraction prompts to this corpus.

Output MUST be a single valid JSON object and nothing else (no markdown fences, no prose). It
must contain exactly these keys:
- "domain": a short phrase naming the subject domain (e.g. "clinical oncology research").
- "language": the dominant natural language of the documents (e.g. "English").
- "persona": a one-sentence expert persona best suited to extract entities/relationships from
  this corpus (e.g. "You are a medical informatics specialist...").
- "entity_types": an array of 4-10 UPPERCASE entity-type labels most relevant to this domain
  (e.g. ["DRUG", "DISEASE", "GENE", "CLINICAL_TRIAL"]).

Base every field ONLY on evidence in the sample. The first character must be {{ and the last }}."""

    human_prompt_template = """Document sample:
\"\"\"
{corpus_sample}
\"\"\"

Produce the corpus profile JSON:"""
