# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``processing`` section: parsing, chunking, translation and extraction."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from unified_kg_rag.domain.languages import LanguageCode

from ..retrieval import (
    validate_safe_name,
)
from ..schema import (
    EntityTypeField,
    EntityTypeItem,
    ExtractionSchema,
    RelationshipPatternField,
    RelationshipTypeField,
    SchemaMode,
    relationship_pattern,
    relationship_type_spec,
)
from ._common import ChunkingStrategy, ChunkLengthUnit, ResolutionMethod
from ._seed import SeedGraphConfig


class DocumentParsingConfig(BaseModel):
    # index_value set in code is checked too (see IndexingConfig).
    model_config = ConfigDict(validate_assignment=True)

    source_directory: str | Path = Field(
        default="source",
        description=(
            "Directory to load documents from, for library "
            "callers (null = the default). run-ingestion requires "
            "--source-directory (or UKR_SOURCE_DIRECTORY) "
            "instead."
        ),
    )
    target_directory: str | Path | None = Field(
        default=None,
        description=(
            "Directory to export each parsed document to as <stem>.json for "
            "inspection (run-ingestion --output-directory). null = no export. "
            "Must not be the source directory."
        ),
    )
    index_value: str | None = Field(
        default=None,
        description=(
            "Index suffix the run writes: every OpenSearch index and Neptune "
            "label is named <prefix>-<index_value>. null = 'default'. run-rag "
            "and run-eval query it too unless --suffix names another "
            "(RAGInput.suffix in Python, which does not read it). "
            "run-ingestion --suffix sets it for one run. Lowercase letters, "
            "digits, '-' and '_'."
        ),
    )
    source_scope: str | None = Field(
        default=None,
        description=(
            "Identity of the corpus source for incremental indexing: a run only "
            "treats registry documents of its own index suffix AND source scope "
            "as deleted. None = the resolved source directory. Set it when the "
            "local directory is a staging copy, e.g. the S3 URI a container "
            "syncs from (env UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE). "
            "Document and text-unit ids derive from it, so set it for ids that "
            "are reproducible across paths and machines."
        ),
    )

    @field_validator("source_directory", mode="before")
    @classmethod
    def _unset_source_directory(cls, value: Any) -> Any:
        # The 0.2.3 template left the key empty (null); run-ingestion takes
        # the directory from its flag, so null reads as unset.
        if value is None:
            return cls.model_fields["source_directory"].default
        return value

    @field_validator("index_value")
    @classmethod
    def _check_index_value(cls, value: str | None) -> str | None:
        # Part of every index name: fail at config load, not at indexing
        # after the extraction was paid for.
        return value if value is None else validate_safe_name(value, "index_value")


class ChunkingConfig(BaseModel):
    # Settings set in code are checked too: a string length_unit would pick
    # the character splitter while the cache key reads tokens.
    model_config = ConfigDict(validate_assignment=True)

    chunker_type: ChunkingStrategy = Field(
        default=ChunkingStrategy.INTELLIGENT,
        description=(
            "intelligent lets the chunking model pick semantic boundaries (one"
            " fast LLM call per pre-chunk); simple splits by size with no LLM "
            "call."
        ),
    )
    length_unit: ChunkLengthUnit = Field(
        default=ChunkLengthUnit.CHARS,
        description=(
            "What the six chunk size settings below count. chars (default) "
            "counts characters. tokens counts a local token estimate (a "
            "Han/Hangul/Kana character ~1 token, other text ~4 characters per "
            "token, at least one per word; no model call), so one setting "
            "keeps chunks inside the embedding and reranker token limits in "
            "any language. Switching re-chunks and re-extracts the corpus; "
            "set the sizes in tokens when you switch (e.g. min 250, max "
            "2000, overlap 125, fallback 1200, pre-chunk 12500 and its "
            "overlap 125)."
        ),
    )
    min_chunk_size: int = Field(
        default=1000,
        ge=1,
        description=(
            "Minimum chunk size (length_unit, characters by default); shorter "
            "pieces are merged into a "
            "neighbour. Default 1,000 (~250 English tokens) keeps headings and "
            "short trailing paragraphs from becoming chunks of their own."
        ),
    )
    max_chunk_size: int = Field(
        default=8000,
        ge=1,
        description=(
            "Maximum chunk size (length_unit, characters by default). Every "
            "chunk is embedded whole "
            "and reranked as one document, so it must fit the embedding "
            "model's input (Titan Text Embeddings V2: 8,192 tokens) and the "
            "reranker's per-document limit (Cohere Rerank 3.5: 4,096 tokens), "
            "or its tail is never embedded or reranked. In characters, "
            "English averages ~4 characters per token and a Han/Hangul/Kana "
            "character ~1, so for CJK corpora with reranking use about half "
            "the default sizes, or set length_unit: tokens."
        ),
    )
    chunk_overlap: int = Field(
        default=500,
        ge=0,
        description=(
            "Overlap between consecutive chunks the size-based splitter cuts "
            "(the simple chunker, and the intelligent chunker's size "
            "fallbacks), in length_unit (characters by default). The "
            "intelligent chunker's model-chosen boundaries cut between lines "
            "and share no text."
        ),
    )
    pre_chunk_size: int = Field(
        default=50000,
        ge=1,
        description=(
            "Pre-chunk size (length_unit, characters by default): the window "
            "the intelligent chunker's "
            "LLM picks boundaries in. Not embedded itself (its chunks are capped "
            "by max_chunk_size), so it is sized for the chunking model's prompt."
        ),
    )
    pre_chunk_overlap: int = Field(
        default=500,
        ge=0,
        description="Pre-chunk overlap (length_unit, characters by default)",
    )
    fallback_chunk_size: int = Field(
        default=4800,
        ge=1,
        description=(
            "Target chunk size for the size-based splitter (the simple "
            "chunker, and intelligent chunking when the LLM fails or a chunk "
            "exceeds max_chunk_size). It splits at paragraphs, then lines, "
            "then after CJK sentence terminators, then after a sentence's "
            "'.', '!', '?' or '…' before a space, then spaces "
            "(length_unit, characters by default). 4,800 characters is ~1,200 "
            "English tokens, the default chunk size of MS GraphRAG and "
            "LightRAG."
        ),
    )
    max_marker_miss_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum allowed boundary marker miss rate",
    )

    @model_validator(mode="after")
    def validate_chunk_sizes(self, info: ValidationInfo) -> "ChunkingConfig":
        if info.field_name in type(self).model_fields:
            # An assignment (one field at a time): the sizes are checked
            # together where they are given together, so code can set
            # several in any order. Validated as a nested section, the
            # field name is the parent's (``chunking``) and the sizes are
            # checked.
            return self
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")
        if self.chunk_overlap >= self.min_chunk_size:
            raise ValueError("chunk_overlap must be less than min_chunk_size")
        if self.chunk_overlap >= self.fallback_chunk_size:
            # The size-based splitter overlaps its chunks by chunk_overlap.
            raise ValueError("chunk_overlap must be less than fallback_chunk_size")
        if self.pre_chunk_overlap >= self.pre_chunk_size:
            raise ValueError("pre_chunk_overlap must be less than pre_chunk_size")
        return self


# The chunk sizes processing.chunking.length_unit counts.
_CHUNK_SIZE_FIELDS = (
    "min_chunk_size",
    "max_chunk_size",
    "chunk_overlap",
    "pre_chunk_size",
    "pre_chunk_overlap",
    "fallback_chunk_size",
)


class TranslationConfig(BaseModel):
    # An assigned code ("ko") is checked and read as its LanguageCode, as a
    # loaded one is: the index language (cache_keys.index_language) reads
    # its value.
    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(
        default=True,
        description=(
            "Run the translation stage. It is skipped at no cost when "
            "source_language equals target_language and there are no "
            "additional_target_languages."
        ),
    )
    source_language: LanguageCode = Field(
        default=LanguageCode.EN,
        description="Predominant source language of the corpus. Selects the "
        "OpenSearch analyzer of the original chunk text and drives the "
        "same-language no-op skip; translation itself is content-driven. With "
        "translation disabled the corpus is indexed in it: extraction, "
        "descriptions, claims, community reports and the summary tree are "
        "written in it, the graph indexes are analyzed for it, and a query "
        "without a requested language is processed into it.",
    )
    target_language: LanguageCode = Field(
        default=LanguageCode.EN,
        description=(
            "Language the corpus is translated into. When translation runs, "
            "the corpus is indexed in it: extraction, embeddings and the "
            "answer context use it, and descriptions, claims, community "
            "reports and the summary tree are written in it (with "
            "translation disabled, source_language takes this role)."
        ),
    )
    additional_target_languages: list[LanguageCode] | None = Field(
        default=None,
        description=(
            "Extra languages to translate into. Each is indexed in its own "
            "translated_text_<language> field with that language's analyzer "
            "and searched lexically (useful with a per-query "
            "RAGInput.target_language); extraction, embeddings and the answer "
            "context use target_language only. Each language is one more LLM "
            "call per chunk."
        ),
    )

    @property
    def translated_languages(self) -> list[str]:
        """Language codes with a translated text field: target first, no repeats."""
        languages = [self.target_language, *(self.additional_target_languages or [])]
        return list(dict.fromkeys(language.value for language in languages))

    @property
    def is_noop(self) -> bool:
        """True when translation would do nothing useful (same-language skip)."""
        return (
            self.source_language == self.target_language
            and not self.additional_target_languages
        )


class DescriptionSummarizationConfig(BaseModel):
    """LLM re-summarization of merged entity/relationship descriptions.

    Without this, merging an entity that appears in many chunks simply
    concatenates every chunk's description, so a popular entity's description
    grows unbounded — bloating prompts/embeddings and degrading quality. After
    merge, any description over the token budget is re-summarized by a cheap
    LLM into one coherent, deduplicated text; items below the threshold skip
    the LLM entirely. A cost variant of MS GraphRAG ``summarize_descriptions``
    (every item with more than one description) and LightRAG
    ``_handle_entity_relation_summary`` (also at ``force_llm_summary_on_merge``
    fragments), not parity: only the token budget triggers it.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Re-summarize merged descriptions longer than "
            "force_summary_threshold_tokens with an LLM (a cost variant of MS "
            "GraphRAG and LightRAG, which also summarize short merges). Off, "
            "descriptions are only concatenated and grow without bound for "
            "frequently mentioned entities."
        ),
    )
    force_summary_threshold_tokens: int = Field(
        default=600,
        ge=1,
        description="Re-summarize a merged description only when its estimated "
        "token count exceeds this threshold. Descriptions at or below it are left "
        "as-is, so cheap entities never incur an LLM call.",
    )
    max_summary_tokens: int = Field(
        default=256,
        ge=1,
        description="Target length (in tokens) of the produced summary; injected "
        "into the summarization prompt as the budget.",
    )


class EntityGroundingConfig(BaseModel):
    """Provenance grounding for extracted entities.

    The extraction prompt asks the model to emit a verbatim ``source_text`` span
    for every entity. When enabled, an entity whose evidence span does not occur
    in its source chunk (verbatim or by token overlap) is treated as a
    hallucination — the model invented it from domain priors rather than the
    document — and is either dropped or confidence-penalized. OFF by default:
    it relies on the model emitting spans and is conservative, but turning it on
    is the defense against corpus-absent entities polluting the index.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Hallucination guard: drop (or, with action 'penalize', "
            "down-weight) entities and relationships whose verbatim "
            "source_text span is not found in their chunk. Punctuation is "
            "ignored and Han/Hangul/Kana spans are compared by character "
            "bigrams, so a changed Korean particle still counts as present. It"
            " applies to gleaning additions too. Requires an extraction model "
            "that emits a source_text span per item."
        ),
    )
    action: str = Field(
        default="drop",
        pattern="^(drop|penalize)$",
        description="What to do with an ungrounded entity: 'drop' removes it; "
        "'penalize' multiplies its confidence by penalty_factor so the "
        "confidence threshold can filter it without hard deletion. An "
        "ungrounded relationship is likewise dropped, or kept at its weight "
        "times penalty_factor; 'penalize' still drops one naming an entity its "
        "chunk did not extract, which it would otherwise make cite the chunk. "
        "Gleaning additions follow the same action.",
    )
    penalty_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence multiplier applied to ungrounded entities when "
        "action='penalize'.",
    )
    min_span_tokens: int = Field(
        default=4,
        ge=1,
        description="Evidence spans shorter than this (after normalization) are "
        "treated as grounded — too short to judge, so the gate never deletes "
        "short legitimate names on weak signal. A Han/Hangul/Kana character "
        "counts as one token.",
    )
    min_overlap_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="When the span is not a verbatim substring, the fraction of "
        "its tokens that must appear in the chunk to still count as grounded "
        "(handles light paraphrase / whitespace edits).",
    )


INLINE_SCHEMA_SOURCE = "inline"


class SchemaSourceConfig(BaseModel):
    """Where the extraction schema comes from: the inline keys, or a
    registered schema source that fills them at start-up."""

    model_config = ConfigDict(validate_assignment=True)

    kind: str = Field(
        default=INLINE_SCHEMA_SOURCE,
        pattern=r"^[a-z][a-z0-9_]*$",
        description=(
            "'inline' (default): the schema is the entity_types, "
            "relationship_types and relationship_patterns keys. Another kind "
            "names a registered schema source ('file': a YAML or JSON file of "
            "those keys; 'owl': an OWL or RDFS ontology, with the ontology "
            "extra) loaded once at start-up into those keys; the loaded "
            "content, not its location, keys the extraction caches. A file "
            "that sets entity_types replaces the config's, which must then "
            "be left unset."
        ),
    )
    path: str | None = Field(
        default=None,
        description=(
            "Location the schema source reads. For 'file', a local file "
            "path or an S3 object (s3://bucket/key.yaml); for 'owl', a local "
            "file or directory, an S3 object, or an S3 prefix read as a "
            "directory (s3://bucket/prefix/, the trailing '/' required). A "
            "local path may be written as a file:// URL. S3 is read with the run's AWS session "
            "(s3:GetObject, and s3:ListBucket for a prefix). A relative path "
            "is resolved against the config file's directory (the working "
            "directory without one); an HTTP location needs a source "
            "registered for it, which receives a URL as written. Logs and "
            "errors show it without userinfo, query or fragment. Must be "
            "unset for 'inline'."
        ),
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Settings of the registered source, read by its loader ('file' "
            "takes none; 'owl' takes what is selected, how it is labelled and "
            "the caps). Not keyed: the schema the source loads is. Must be "
            "empty for 'inline'."
        ),
    )
    secret_arn: str | None = Field(
        default=None,
        pattern=r"^arn:aws[a-z-]*:secretsmanager:",
        description=(
            "ARN of a Secrets Manager secret holding the credentials a "
            "registered source reads at load time ('file' and 'owl' take "
            "none). Never logged. Must be unset for 'inline'."
        ),
    )

    @model_validator(mode="after")
    def _check_path(self) -> "SchemaSourceConfig":
        if self.kind == INLINE_SCHEMA_SOURCE:
            given = [
                name
                for name, value in (
                    ("path", self.path),
                    ("options", self.options or None),
                    ("secret_arn", self.secret_arn),
                )
                if value is not None
            ]
            if given:
                raise ValueError(
                    f"{', '.join(f'schema_source.{name}' for name in given)} "
                    f"{'is' if len(given) == 1 else 'are'} set but "
                    "schema_source.kind is 'inline'; set the kind of the "
                    "source that reads it"
                )
        return self


class GraphExtractionConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    max_entities_per_chunk: int = Field(
        default=50,
        ge=1,
        description=(
            "Most entities the extraction prompt asks for "
            "per text chunk; it also sizes the extraction "
            "call's output-token floor."
        ),
    )
    max_relationships_per_chunk: int = Field(
        default=50, ge=1, description="Maximum relationships per text chunk"
    )
    entity_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Drop an entity from each chunk that scored it below this "
            "confidence, with that chunk's relationships naming it (per chunk,"
            " so the result does not depend on which documents are indexed "
            "together). Compared against the normalized confidence: the "
            "prompt's 1-10 score / 10. 0.0 keeps everything."
        ),
    )
    entity_types: list[EntityTypeField] = Field(
        default_factory=lambda: list[EntityTypeItem](
            [
                "PERSON: Names, individuals, roles, titles",
                "ORGANIZATION: Companies, institutions, departments, groups",
                "LOCATION: Places, addresses, geographic areas, facilities",
                "CONCEPT: Ideas, theories, methodologies, frameworks, principles",
                "OBJECT: Documents, tools, products, systems, technologies",
                "EVENT: Meetings, projects, activities, processes, incidents",
                "TEMPORAL: Dates, time periods, schedules, deadlines",
            ]
        ),
        description=(
            "Entity categories injected into the extraction prompt, each "
            "'LABEL: short description' (the description is optional), e.g. "
            "['GENE: ...', 'DISEASE: ...'], or {label, description, iri} "
            "(iri: the ontology class the type maps to, stored as type_iri). "
            "The cheapest and "
            "most effective domain-adaptation knob; an empty list lets the "
            "model choose any relevant types. Labels must differ in more "
            "than case and word separators."
        ),
    )
    relationship_types: list[RelationshipTypeField] = Field(
        default_factory=list,
        description=(
            "Relationship types of the extraction schema, each 'LABEL: short "
            "description' or {label, description, iri} (iri: the ontology "
            "property the type maps to, stored as predicate_iri). The "
            "extraction prompt lists them (and"
            " relationship_patterns) instead of its example types, and unless "
            "schema_mode is off a type naming one, or an entity type naming a "
            "declared one, is written as declared. Empty: none declared, the "
            "prompt shows its examples."
        ),
    )
    relationship_patterns: list[RelationshipPatternField] = Field(
        default_factory=list,
        description=(
            "Relationship types allowed between entity types, each "
            "'SOURCE -RELATION-> TARGET' (e.g. 'PERSON -WORKS_FOR-> "
            "ORGANIZATION') or {source_type, relation, target_type}; every "
            "label must be a declared type. A relationship type with patterns "
            "is allowed only between the entity types they name, in their "
            "direction; one without is not restricted. Empty: no restriction."
        ),
    )
    schema_mode: SchemaMode = Field(
        default="align",
        description=(
            "How extraction applies the schema: 'off' (the prompt only), "
            "'align' (keep every item, write in-schema labels as declared and "
            "count the rest; with entity types alone it changes nothing) or "
            "'strict' (also drop, per text unit, entities of an undeclared or "
            "empty type with their relationships, relationships of an "
            "undeclared type, and relationships a pattern does not allow "
            "between two entities of declared types). Labels match by case "
            "and word separators only, never by similarity. Where the schema "
            "applies, indexing stores in_schema on entities and "
            "relationships, and unless the mode is off it stores a declared "
            "type's iri (type_iri, predicate_iri) on the items of that type, "
            "with entity types alone too; both are filter keys."
        ),
    )
    schema_source: SchemaSourceConfig = Field(
        default_factory=SchemaSourceConfig,
        description="Where the extraction schema comes from",
    )
    description_summarization: DescriptionSummarizationConfig = Field(
        default_factory=DescriptionSummarizationConfig,
        description="LLM re-summarization of over-long merged descriptions",
    )
    entity_grounding: EntityGroundingConfig = Field(
        default_factory=EntityGroundingConfig,
        description="Provenance grounding guard against hallucinated entities",
    )

    @field_validator("relationship_types", "relationship_patterns")
    @classmethod
    def _check_items(cls, items: list[Any], info: ValidationInfo) -> list[Any]:
        """Parse each string item, naming the ones that do not."""
        parse: Callable[[Any], Any] = (
            relationship_type_spec
            if info.field_name == "relationship_types"
            else relationship_pattern
        )
        errors = []
        for position, item in enumerate(items):
            try:
                parse(item)
            except ValueError as e:
                errors.append(f"item {position}: {_schema_message(e)}")
        if errors:
            raise ValueError("; ".join(errors))
        return items

    @model_validator(mode="after")
    def _check_schema(self) -> "GraphExtractionConfig":
        """Reject an inconsistent schema, and inline relationship types or
        patterns beside a schema source (which fills them)."""
        errors: list[str] = []
        source = self.schema_source
        if source.kind != INLINE_SCHEMA_SOURCE and (
            self.relationship_types or self.relationship_patterns
        ):
            errors.append(
                f"schema_source.kind '{source.kind}' fills relationship_types "
                "and relationship_patterns; leave them unset here"
            )
        try:
            ExtractionSchema.from_config(self)
        except ValueError as e:
            errors.append(_schema_message(e))
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def extraction_schema(self) -> ExtractionSchema:
        """The schema these settings hold inline.

        Raises:
            ConfigurationError: ``schema_source`` names a source that has not
                been loaded into the inline keys yet; the inline keys alone
                would be the wrong schema. Pass the config through
                ``application.bootstrap.resolve_extraction_schema`` first.
        """
        if self.schema_source.kind != INLINE_SCHEMA_SOURCE:
            # Imported here: shared.logging imports this module.
            from unified_kg_rag.shared.exceptions import ConfigurationError

            raise ConfigurationError(
                f"processing.graph_extraction.schema_source.kind "
                f"'{self.schema_source.kind}' is not loaded yet; resolve the "
                "config with application.bootstrap.resolve_extraction_schema "
                "before reading its extraction schema"
            )
        return ExtractionSchema.from_config(self)


def _schema_message(error: ValueError) -> str:
    """The messages of a schema error, without pydantic's framing."""
    if isinstance(error, ValidationError):
        return "; ".join(
            str(detail["msg"]).removeprefix("Value error, ")
            for detail in error.errors(include_url=False, include_input=False)
        )
    return str(error)


class GleaningConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description=(
            "Extra extraction passes per text unit that ask the "
            "model for the entities and relationships it missed."
        ),
    )
    max_rounds: int = Field(
        default=3,
        ge=1,
        description=(
            "Maximum gleaning rounds per text unit. Each round after the first"
            " re-sends only the text units whose previous answer added a new "
            "entity or relationship, so a unit stops once an answer adds "
            "nothing new. On MuSiQue, round 1 added most of gleaning's "
            "relationships to bridge entities and rounds 2-3 little more, "
            "gleaning was the largest share of the ingestion cost, and the "
            "answers favoured neither 1 nor 3 rounds (see "
            "docs/concepts/defaults-and-evidence.md#gleaning-rounds). 1 is MS "
            "GraphRAG's default and the cost lever."
        ),
    )
    max_entities_per_prompt: int = Field(
        default=100, ge=1, description="Maximum entities per gleaning prompt"
    )
    max_relationships_per_prompt: int = Field(
        default=100,
        ge=1,
        description="Maximum relationships per gleaning prompt",
    )


class ClaimExtractionConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description=(
            "Extract claims (covariates): one more LLM call per text unit. "
            "When on, local search injects the matching claims into its "
            "context and simple search includes the claims index in its sweep."
            " Claim subjects and objects are not linked into the entity graph."
            " Turned on for an incremental namespace built without claims, the "
            "next run rebuilds it once from its stored extractions; that run, "
            "and one turning it off for a namespace that holds claims, is "
            "refused while the namespace holds other source scopes' documents."
        ),
    )
    max_entities_per_prompt: int = Field(
        default=100,
        ge=0,
        description="Maximum entities per claim extraction prompt",
    )


class ProcessingConfig(BaseModel):
    max_concurrency: int = Field(
        default=40,
        ge=1,
        description=(
            "LLM calls an ingestion stage keeps in flight: the next item "
            "starts as soon as one finishes, so this is both the peak and the "
            "steady concurrency (embedding requests: at most "
            "processing.embedding_concurrency of them). "
            "These stages are I/O-bound, so this can be "
            "well above the CPU count. Bedrock quotas differ per account, "
            "model and Region: lower it if the logs show ThrottlingException "
            "retries, raise it if quota allows. The backend: local preset sets"
            " 20, since a local model server queues the calls it cannot run at"
            " once."
        ),
    )
    embedding_concurrency: int = Field(
        default=16,
        ge=1,
        description=(
            "Embedding requests the indexing stage keeps in flight at once "
            "(also capped by processing.max_concurrency). In a real-AWS run, "
            "16 embedded 1,600 texts in half the time 8 took, without "
            "throttling; lower it if the logs show embedding "
            "ThrottlingException retries or a local embedding server queues."
        ),
        json_schema_extra={"evidence": "operational"},
    )
    embedding_batch_size: int = Field(
        default=50,
        ge=1,
        description=(
            "Texts per embedding request. The indexing stage embeds and writes "
            "embedding_batch_size x embedding_concurrency items at a time (at "
            "most 1,000, one read of their stored digests). Keep it within "
            "what the embedding endpoint takes per request."
        ),
        json_schema_extra={"evidence": "operational"},
    )
    io_workers: int = Field(
        default=64,
        ge=1,
        description=(
            "Threads for blocking I/O on the query path (the Bedrock calls "
            "LangChain runs in the event loop's default executor, Neptune "
            "traversals, reranking) in the CLIs and GraphRAGChain's sync "
            "methods. Python's default is min(32, CPUs + 4), six on a 2-vCPU "
            "task. An async host calls "
            "unified_kg_rag.configure_event_loop(asyncio.get_running_loop(), "
            "config.processing.io_workers) at startup."
        ),
    )
    ignore_errors: bool = Field(
        default=False,
        description=(
            "When a whole step fails (a batch call, a translation language, an"
            " evaluator, global map/reduce, the DRIFT primer, the auto "
            "router), log it and continue with an empty or degraded result "
            "instead of raising. Items whose own LLM call fails are always "
            "skipped and reported, whatever this says."
        ),
    )
    deduplicate: bool = Field(
        default=False,
        description=(
            "Drop near-duplicate documents before extraction. "
            "Two documents are near-duplicates when the Jaccard"
            " similarity of their character 3-gram sets, "
            "estimated with a 128-permutation MinHash, is at "
            "least 0.9 (fixed parameters); of each pair the "
            "first is kept, and an incremental run keeps the "
            "already-indexed copy."
        ),
    )
    resolution_method: ResolutionMethod = Field(
        default=ResolutionMethod.MINHASH,
        description=(
            "Name-similarity measure of fuzzy entity resolution "
            "(fuzzy_entity_resolution): minhash or sequence_matcher."
        ),
    )
    fuzzy_entity_resolution: bool = Field(
        default=False,
        description="Also merge entities whose names are only similar "
        "(resolution_method at similarity_threshold). An incremental merge "
        "folds each new entity into its one best-matching stored entity and "
        "never joins stored entities with each other, so where a full build "
        "chains similar names into one group it can keep several, until "
        "indexing.reset regroups the corpus. Off by default: names are merged when they "
        "are equal after normalization, legal-form removal ('Acme Inc' / "
        "'Acme') and dense-script spacing, because name similarity also joins "
        "different organizations that share most of their name ('... "
        "University' / '... University Hospital').",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity threshold for fuzzy entity resolution "
            "(fuzzy_entity_resolution) and for matching claim subjects and "
            "objects to entity names. Raise it if distinct entities merge."
        ),
    )
    document_parsing: DocumentParsingConfig = Field(
        default_factory=DocumentParsingConfig,
        description="Document parsing configuration",
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Text chunking configuration"
    )
    translation: TranslationConfig = Field(
        default_factory=TranslationConfig, description="Text translation configuration"
    )
    graph_extraction: GraphExtractionConfig = Field(
        default_factory=GraphExtractionConfig,
        description="Graph extraction configuration",
    )
    gleaning: GleaningConfig = Field(
        default_factory=GleaningConfig,
        description="Iterative extraction refinement configuration",
    )
    claim_extraction: ClaimExtractionConfig = Field(
        default_factory=ClaimExtractionConfig,
        description="Claim extraction configuration",
    )
    seed_graphs: list[SeedGraphConfig] = Field(
        default_factory=list,
        description=(
            "Knowledge graphs imported as virtual documents of this run's "
            "corpus and merged with the extracted graph (each item: "
            "processing.seed_graphs[] below). Documents are canonical: a "
            "seed fills types, names and confidence only where no document "
            "states them, and adds description lines, properties, IRIs and "
            "edges. Loaded once at start-up, after the extraction schema; "
            "what a seed converts to keys every stage, where it is read "
            "from does not. As an environment variable, a JSON list: "
            'UKR__PROCESSING__SEED_GRAPHS=\'[{"name": "procure", '
            '"kind": "jsonl", "path": "seed/procure.jsonl"}]\'.'
        ),
    )

    @field_validator("seed_graphs")
    @classmethod
    def _unique_seed_names(cls, value: list[SeedGraphConfig]) -> list[SeedGraphConfig]:
        names = [seed.name for seed in value]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"seed graph names must be unique: {', '.join(repeated)} "
                "appear more than once"
            )
        return value
