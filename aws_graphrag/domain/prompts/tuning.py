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
    pass


@dataclass(frozen=True)
class CorpusProfilePrompt(BasePrompt):
    prompt_key = "corpus_profile"
    """Analyze a corpus sample and emit a domain/persona/entity-type profile."""

    input_variables = ["corpus_sample"]
    output_variables = ["domain", "language", "persona", "entity_types"]

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


@dataclass(frozen=True)
class ExtractionExamplesPrompt(BasePrompt):
    prompt_key = "extraction_examples"
    """Generate domain-adapted few-shot extraction examples from a sample.

    Ports MS GraphRAG ``prompt_tune`` few-shot generation: grounding the
    extraction prompt in 1-2 worked examples drawn from the corpus itself
    measurably improves entity/relationship recall on specialized domains.
    """

    input_variables = ["domain", "persona", "entity_types", "corpus_sample"]
    output_variables = ["examples"]

    system_prompt_template = """You are building few-shot examples for a knowledge-graph extraction
prompt specialized to the "{domain}" domain. {persona}

Given a short excerpt from the corpus, produce ONE worked extraction example that demonstrates how
entities and relationships should be extracted from this kind of text, using ONLY these entity
types: {entity_types}.

Output MUST follow this exact structure and contain nothing else (no prose, no markdown fences):

EXAMPLE TEXT:
<a 2-4 sentence excerpt-style passage representative of the domain>

<entities>
<entity>
<name>EXACT_ENTITY_NAME</name>
<type>ONE_OF_THE_ALLOWED_TYPES</type>
<description>One concise sentence grounded in the example text.</description>
<confidence>8</confidence>
</entity>
</entities>

<relationships>
<relationship>
<source>SOURCE_ENTITY_NAME</source>
<target>TARGET_ENTITY_NAME</target>
<type>SPECIFIC_RELATIONSHIP_TYPE</type>
<description>One concise sentence explaining the connection.</description>
<strength>8</strength>
</relationship>
</relationships>

Keep it realistic for the domain, self-consistent (every relationship endpoint must appear as an
entity), and compact (3-6 entities, 2-4 relationships)."""

    human_prompt_template = """Domain: {domain}
Allowed entity types: {entity_types}

Corpus excerpt to base the example on:
\"\"\"
{corpus_sample}
\"\"\"

Produce one worked extraction example:"""
