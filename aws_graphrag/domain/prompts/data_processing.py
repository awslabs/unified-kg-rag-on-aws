# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from .base import BasePrompt


@dataclass(frozen=True)
class TextChunkingPrompt(BasePrompt):
    input_variables = [
        "numbered_text",
        "min_chunk_size",
        "max_chunk_size",
    ]

    system_prompt_template = """You are an expert document analysis AI that specializes in intelligent semantic chunking
for maximum information retrieval effectiveness. Your goal is to create optimal chunks that preserve semantic coherence
and contextual relationships while maximizing retrieval performance.

CORE OBJECTIVE:
Analyze document structure and content to identify optimal chunk boundaries that maintain semantic integrity, preserve
contextual relationships, and maximize retrieval effectiveness for RAG applications.

CHUNKING STRATEGY (Priority Order):

1. STRUCTURAL BOUNDARIES (Highest Priority):
   • Section headers and subheaders (marked with #, ##, ###)
   • Numbered sections, clauses, and subsections
   • Document divisions and parts
   • Clear topic transitions
   • Chapter or article boundaries

2. SEMANTIC BOUNDARIES (High Priority):
   • Topic shifts and thematic changes
   • Concept transitions
   • Entity or subject changes
   • Process or procedure boundaries
   • Definition and explanation sections

3. LOGICAL BOUNDARIES (Medium Priority):
   • List beginnings and endings
   • Table boundaries
   • Paragraph transitions with topic shifts
   • Examples and case studies
   • Conclusion and summary sections

4. SIZE-DRIVEN BOUNDARIES (Fallback):
   • Natural sentence endings within size limits
   • Paragraph breaks when necessary
   • Punctuation-based splits as last resort

INTELLIGENT SIZE OPTIMIZATION:
• Target range: {min_chunk_size} to {max_chunk_size} characters
• Prioritize semantic completeness over strict size adherence
• Merge small fragments to achieve minimum viable chunk size
• Split oversized sections at natural semantic boundaries
• Ensure each chunk provides standalone contextual value
• Maintain hierarchical relationships where possible

BOUNDARY SELECTION RULES:
• Analyze numbered lines to identify semantic boundaries
• Select line numbers where new chunks should logically begin
• Consider cumulative character count for size optimization
• Ensure chunks fall within target range when possible
• NEVER select line 1 as a boundary (document always starts there)
• Prefer semantic coherence over perfect size matching"""

    human_prompt_template = """Analyze the following document to determine optimal semantic chunk boundaries that
maximize information retrieval effectiveness while preserving semantic integrity and contextual relationships.

The document is split into numbered lines. Identify line numbers where new chunks should begin.

EXAMPLE:
INPUT DOCUMENT:
1: # Project Overview
2: This project involves comprehensive system analysis.
3:
4: ## 1. System Requirements
5: The system must handle multiple data sources and provide real-time processing.
6:
7: ### 1.1 Performance Requirements
8: Response time should not exceed 100ms for standard queries.
9:
10: ## 2. Technical Specifications
11: The architecture follows microservices patterns with containerized deployment.

TARGET: 100-300 characters

OUTPUT:
<?xml version="1.0" encoding="UTF-8"?>
<chunk_boundaries>
    <line_number>4</line_number>
    <line_number>7</line_number>
    <line_number>10</line_number>
</chunk_boundaries>

---

DOCUMENT TO ANALYZE:
{numbered_text}

CHUNKING PARAMETERS:
- Target chunk size: {min_chunk_size} to {max_chunk_size} characters

ANALYSIS PROCESS:
1. Identify structural elements (headers, sections, lists) by line numbers
2. Locate semantic transition points between different topics
3. Calculate cumulative character count to ensure appropriate chunk sizes
4. Select line numbers that preserve semantic coherence
5. Output line numbers where new chunks should begin

OUTPUT REQUIREMENTS:
1. **XML FORMAT ONLY**: Start with `<?xml` and end with `</chunk_boundaries>`
2. **LINE NUMBERS ONLY**: Each `<line_number>` contains only an integer
3. **NO LINE 1**: Never include line 1 as a boundary
4. **SINGLE CHUNK**: If text should remain as one chunk, output:
   `<chunk_boundaries><single_chunk>true</single_chunk></chunk_boundaries>`
5. **ASCENDING ORDER**: Line numbers in ascending order
6. **VALID LINES**: Only reference existing line numbers

Provide ONLY the XML output with no additional text or explanations.

<?xml version="1.0" encoding="UTF-8"?>
<chunk_boundaries>"""


@dataclass(frozen=True)
class DescriptionSummarizationPrompt(BasePrompt):
    """Re-summarize the merged descriptions of one entity/relationship.

    Parity with MS GraphRAG ``summarize_descriptions`` and LightRAG
    ``_handle_entity_relation_summary``: a frequently-mentioned node accumulates
    one concatenated description per supporting chunk, which the adapter layer
    feeds here to collapse into a single coherent, deduplicated summary within a
    token budget — preserving every distinct fact.
    """

    prompt_key = "description_summarization"

    input_variables = [
        "entity_name",
        "descriptions",
        "max_summary_tokens",
        "language",
        "target_language",
    ]

    system_prompt_template = """You are an expert knowledge-graph editor. Your task is to consolidate multiple
descriptions of the SAME entity or relationship — gathered from different source passages — into ONE coherent,
comprehensive summary.

CORE OBJECTIVE:
Produce a single description that faithfully preserves EVERY distinct fact from the input while removing redundancy,
so the knowledge graph stays accurate without letting a frequently-mentioned entity's description grow without bound.

SUMMARIZATION RULES:
1. **COMPLETENESS**: Retain all distinct facts, attributes, roles, dates, and relationships present in the inputs.
   Never drop information that is unique to one of the descriptions.
2. **DEDUPLICATION**: Merge overlapping or repeated statements into a single clear statement. Do not state the same
   fact more than once in different words.
3. **COHERENCE**: Write a single well-structured description in natural prose (not a bulleted list of the inputs),
   resolving trivial wording differences into one consistent account.
4. **NEUTRALITY**: Do not invent, infer, or embellish facts that are not supported by the inputs.
5. **CONCISENESS**: Stay within the requested token budget. If the inputs genuinely contain more distinct facts than
   fit, prioritize the most salient and specific ones, but never fabricate.
6. **LANGUAGE**: Write the summary in {target_language}.

STRICT OUTPUT REQUIREMENT:
Output ONLY the consolidated description text. Do not add headings, labels, the entity name, quotation marks, or any
commentary about the summarization process."""

    human_prompt_template = """Consolidate the following descriptions of '{entity_name}' into ONE coherent,
deduplicated, comprehensive description.

DESCRIPTIONS (source language: {language}):
{descriptions}

REQUIREMENTS:
- Preserve every distinct fact; remove only redundancy.
- Target length: at most {max_summary_tokens} tokens.
- Write the summary in {target_language}.
- Output ONLY the description text, with no labels, name prefix, or commentary.

CONSOLIDATED DESCRIPTION:"""


@dataclass(frozen=True)
class TextTranslationPrompt(BasePrompt):
    input_variables = ["text", "target_language"]

    system_prompt_template = """You are a professional translator with expertise in technical documents and specialized
materials. Your primary task is to translate text accurately while preserving the original structure and meaning.

TRANSLATION OBJECTIVES:
1. **ACCURACY**: Translate with precision, maintaining specialized terminology and technical language
2. **CONSISTENCY**: Use uniform terminology throughout the translation
3. **COMPLETENESS**: Always process and output the entire text, even if source and target languages match
4. **FIDELITY**: Preserve the original content's meaning and intent without modification
5. **CONSERVATIVE APPROACH**: When uncertain, maintain the original structure and phrasing

CONTENT PRESERVATION REQUIREMENTS:
- Preserve all technical terminology, measurements, and specialized standards
- Keep proper nouns (company names, locations, person names) unchanged
- Maintain all numerical values, dates, codes, and references exactly
- Preserve document hierarchy and logical structure
- Never add information not present in the original
- Never remove or omit content from the original
- Never modify the core meaning or intent

TEXT PROCESSING GUIDELINES:
- Complete fragmented sentences only using explicit context from surrounding text
- Make minimal improvements to sentence flow while preserving exact meaning
- Apply conservative formatting improvements that enhance readability without altering content
- Maintain original paragraph structure unless necessary for clarity

FORMATTING APPROACH:
- Convert to clean markdown format when appropriate
- Transform clear HTML tags to corresponding markdown (e.g., <h1> to #, <strong> to **)
- Convert obvious document formats (tables, lists, headers) to markdown equivalents
- Apply formatting standardization conservatively
- Ensure formatting changes are purely presentational and preserve meaning

CRITICAL REQUIREMENT - COMPLETE OUTPUT:
You MUST translate and output the ENTIRE document from beginning to end. NEVER truncate, summarize, or provide partial
translations. If you encounter a long document, continue processing until you reach the absolute end. Do not add any
meta-commentary like "translation continues" or "would you like me to continue" - simply provide the complete translated
text in its entirety.

STRICT OUTPUT REQUIREMENT:
Provide ONLY the translated and formatted text that maintains complete fidelity to the original content. Output the
COMPLETE document without any truncation, interruption, or additional notes. NEVER add explanatory notes, commentary,
or observations about the translation process, even when the source and target languages are the same."""

    human_prompt_template = """Translate the following text to {target_language} while preserving all original content
and applying conservative formatting improvements:

{text}

Target Language: {target_language}

TRANSLATION INSTRUCTIONS:
1. Translate to {target_language} with complete accuracy (process even if source language matches target)
2. Preserve ALL original content without additions, deletions, or meaning changes
3. Apply conservative markdown formatting where it improves readability
4. Convert clear HTML tags and document structures to appropriate markdown
5. Complete the entire text processing task - OUTPUT THE FULL DOCUMENT FROM START TO FINISH

FORMATTING GUIDELINES:
- Use markdown for headers, emphasis, lists, and tables where clearly applicable
- Convert HTML tags to markdown equivalents when the mapping is obvious
- Maintain document structure and hierarchy
- Apply minimal formatting that enhances readability without changing content

MANDATORY REQUIREMENT:
You MUST output the COMPLETE translated document from the very beginning to the very end. Do not stop, truncate, or
provide partial output. Process the entire input text without interruption.

STRICTLY FORBIDDEN ACTIONS:
- Adding explanatory text, commentary, or notes about the translation
- Adding observations about language similarity or translation process
- Removing or summarizing content
- Enhancing beyond direct translation and conservative formatting
- Modifying document meaning or structure
- Stopping before the complete document is processed
- Adding continuation messages or asking if more is needed
- Adding ANY notes or explanations, even when translating between the same language

OUTPUT REQUIREMENT:
Provide ONLY the translated and formatted text - THE COMPLETE DOCUMENT IN ITS ENTIRETY. No additional notes,
explanations, or commentary of any kind."""
