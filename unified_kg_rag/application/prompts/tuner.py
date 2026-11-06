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

On request it also drafts an extraction schema (:class:`SchemaDraft`): the
profile's entity types and the relationship types the grounded examples
use, as a file for ``schema_source: {kind: file}``. Over a schema another
source loads (an ontology), the draft keeps its types and says which
relationship types the source lacks.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import yaml
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ValidationError

from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.schema_sources import redact_location
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.domain.ingestion.contributions import unit_entities
from unified_kg_rag.domain.ingestion.schema_conformance import unit_entity_types
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    ModelPurpose,
    ModelRole,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.config import INLINE_SCHEMA_SOURCE, merge_over
from unified_kg_rag.domain.models.schema import (
    EntityTypeItem,
    EntityTypeSpec,
    ExtractionSchema,
    RelationshipPattern,
    RelationshipTypeSpec,
    entity_type_spec,
    schema_key,
)
from unified_kg_rag.domain.prompts import (
    CommunityReportPrompt,
    CorpusProfilePrompt,
    GraphExtractionPrompt,
)
from unified_kg_rag.shared import LanguageModelError, get_logger
from unified_kg_rag.shared.utils import (
    entity_key,
    generate_stable_id,
    parse_llm_json,
    text_digest,
)
from unified_kg_rag.shared.utils.sentences import sentence_ends
from unified_kg_rag.shared.utils.validation import format_validation_errors

logger = get_logger(__name__)

# Target length of an evidence span cut from a long sentence.
_MAX_EVIDENCE_CHARS = 200
# Hard cap when the names themselves lie further apart than the target; a
# sentence whose names cannot fit in this many characters gives no span.
_MAX_EVIDENCE_SPAN_CHARS = 400


# The relationship types a schema draft lists at most (the most frequent).
MAX_DRAFT_RELATIONSHIP_TYPES = 12
# How often the examples must state a (source type, relation, target type)
# before a schema draft lists it as a pattern.
MIN_PATTERN_SUPPORT = 2
_LABEL_SEPARATORS = re.compile(r"[\s_]+")


def _escape_braces(text: str) -> str:
    """Make ``text`` literal inside a LangChain f-string prompt template."""
    return text.replace("{", "{{").replace("}", "}}")


# The keys a corpus profile answer holds (CorpusProfile.from_payload).
_PROFILE_KEYS = ("domain", "language", "persona", "entity_types")


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


def _draft_label(label: str) -> str:
    """``label`` as a schema draft writes it: NFKC, trimmed, each run of
    whitespace and ``_`` one ``_``, uppercase (``works for`` → ``WORKS_FOR``)."""
    return _LABEL_SEPARATORS.sub(
        "_", unicodedata.normalize("NFKC", label).strip()
    ).upper()


def _type_item(spec: EntityTypeSpec | RelationshipTypeSpec) -> str | dict[str, str]:
    """``spec`` as a schema file writes it: ``LABEL`` or ``LABEL:
    description``, or a mapping when it has an IRI or its label a ':'
    (which would read as ``LABEL: description``)."""
    if spec.iri is None and ":" not in spec.label:
        return f"{spec.label}: {spec.description}" if spec.description else spec.label
    return spec.model_dump(mode="json", exclude_defaults=True)


def _parses_back(pattern: RelationshipPattern) -> bool:
    try:
        return RelationshipPattern.parse(str(pattern)) == pattern
    except ValueError:
        return False


class SchemaDraft(BaseModel):
    """An extraction schema drafted by prompt tuning, for review.

    ``entity_types`` are the corpus profile's, or the configured ones;
    ``relationship_types`` the configured ones, then those the grounded
    examples state most often (at most ``MAX_DRAFT_RELATIONSHIP_TYPES``
    more; ``relationship_type_counts`` how often each is stated). A
    configured type keeps its description and IRI, and a configured
    pattern is kept. A relationship type the config gives no pattern gets
    patterns only when the examples state it between two of the entity
    types and every such (source type, target type) at least
    ``MIN_PATTERN_SUPPORT`` times, so the draft never flags its own
    examples as off pattern; one the config gives patterns gets no other.
    Written as a ``schema_source`` file (:meth:`to_yaml`) and meant for
    ``schema_mode: align``. ``schema_source`` is the ``(kind, location)`` of
    the schema source the configured types came from when it is not a file,
    and ``added_relationship_types`` the types the examples add to them;
    the draft then freezes that source's schema: a type it declares gets
    no pattern but the source's, so with nothing added the draft loads as
    the same schema (same digest).
    """

    entity_types: list[EntityTypeSpec] = Field(default_factory=list)
    relationship_types: list[RelationshipTypeSpec] = Field(default_factory=list)
    relationship_patterns: list[RelationshipPattern] = Field(default_factory=list)
    relationship_type_counts: dict[str, int] = Field(default_factory=dict)
    added_relationship_types: list[str] = Field(default_factory=list)
    schema_source: tuple[str, str] | None = None

    @classmethod
    def from_examples(
        cls,
        entity_types: Sequence[EntityTypeItem],
        records: list[TextUnitExtraction],
        *,
        relationship_types: Sequence[RelationshipTypeSpec] = (),
        relationship_patterns: Sequence[RelationshipPattern] = (),
        schema_source: tuple[str, str] | None = None,
    ) -> SchemaDraft:
        """The draft of the ``entity_types`` (a profile's or the
        configured ones), the configured ``relationship_types`` and
        ``relationship_patterns`` (loaded from ``schema_source`` when that
        is not a file), and the example extraction ``records``; the same
        input gives the same draft."""
        entities: dict[str, EntityTypeSpec] = {}
        for item in entity_types:
            spec = entity_type_spec(item)
            if spec is not None:
                entities.setdefault(schema_key(spec.label), spec)

        declared: dict[str, RelationshipTypeSpec] = {}
        for rel_spec in relationship_types:
            declared.setdefault(schema_key(rel_spec.label), rel_spec)
        counts: Counter[str] = Counter()
        spellings: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for rel in record.relationships:
                if label := _draft_label(rel.type or ""):
                    counts[schema_key(label)] += 1
                    spellings[schema_key(label)][label] += 1
        ranked = sorted(
            (key for key in counts if key not in declared),
            key=lambda key: (-counts[key], min(spellings[key])),
        )[:MAX_DRAFT_RELATIONSHIP_TYPES]
        relations: dict[str, RelationshipTypeSpec] = {
            **declared,
            **{
                key: RelationshipTypeSpec(
                    label=min(spellings[key], key=lambda s: (-spellings[key][s], s))
                )
                for key in ranked
            },
        }
        # A relation the config gives patterns keeps only those: its
        # examples in another direction are counted off pattern instead.
        patterned = {schema_key(pattern.relation) for pattern in relationship_patterns}
        if schema_source is not None:
            # Over another source the draft freezes its schema: a type the
            # source declares keeps the source's patterns, none included.
            patterned |= declared.keys()

        # Relation key -> (source key, target key) -> statements.
        triples: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        for record in records:
            types = unit_entity_types(record.entities)
            for rel in record.relationships:
                relation = schema_key(_draft_label(rel.type or ""))
                source = schema_key(types.get(entity_key(rel.source_name or ""), ""))
                target = schema_key(types.get(entity_key(rel.target_name or ""), ""))
                if relation in relations and source in entities and target in entities:
                    triples[relation][(source, target)] += 1
        patterns = list(relationship_patterns) + [
            RelationshipPattern(
                source_type=entities[source].label,
                relation=relations[relation].label,
                target_type=entities[target].label,
            )
            for relation in relations
            if relation not in patterned
            and triples[relation]
            and min(triples[relation].values()) >= MIN_PATTERN_SUPPORT
            for source, target in sorted(triples[relation])
        ]
        draft = cls(
            entity_types=list(entities.values()),
            relationship_types=list(relations.values()),
            relationship_patterns=patterns,
            relationship_type_counts={
                spec.label: counts[key] for key, spec in relations.items()
            },
            added_relationship_types=[relations[key].label for key in ranked],
            schema_source=schema_source,
        )
        draft.extraction_schema()  # A draft is a valid schema (raises if not).
        return draft

    def extraction_schema(self) -> ExtractionSchema:
        """The draft as the schema ``schema_mode: align`` would apply.

        Raises:
            ValueError: The draft is inconsistent (``schema_problems``).
        """
        return ExtractionSchema.from_items(
            self.entity_types, self.relationship_types, self.relationship_patterns
        )

    def document(self) -> dict[str, Any]:
        """The keys of the schema file. Without entity types the config's
        stay, so the file then leaves ``entity_types`` out."""
        document: dict[str, Any] = {}
        if self.entity_types:
            document["entity_types"] = [_type_item(spec) for spec in self.entity_types]
        document["relationship_types"] = [
            _type_item(spec) for spec in self.relationship_types
        ]
        # A pattern whose labels hold its separators would not parse back:
        # a mapping.
        document["relationship_patterns"] = [
            (str(pattern) if _parses_back(pattern) else pattern.model_dump(mode="json"))
            for pattern in self.relationship_patterns
        ]
        return document

    def to_yaml(self, path: str = "schema.yaml") -> str:
        """The schema file: a comment on how to use it, then the keys."""
        if self.schema_source is None:
            lines = [
                "# Extraction schema draft written by run-prompt-tuning. Review it",
                "# (rename, merge or drop types, add descriptions and iri), then load it:",
            ]
        else:
            kind, location = self.schema_source
            added = len(self.added_relationship_types)
            keys = {schema_key(label) for label in self.added_relationship_types}
            new_patterns = sum(
                schema_key(pattern.relation) in keys
                for pattern in self.relationship_patterns
            )
            lines = [
                "# Extraction schema draft written by run-prompt-tuning: copied from",
                f"# {kind} '{location}' plus {added} relationship type"
                f"{'' if added == 1 else 's'} the source lacks"
                + (f" ({', '.join(self.added_relationship_types)})" if added else "")
                + (
                    f" and {new_patterns} pattern{'' if new_patterns == 1 else 's'}"
                    f" for {'it' if added == 1 else 'them'}"
                    if new_patterns
                    else ""
                )
                + ".",
                f"# Add them to the source (keep kind: {kind}), or review this file",
                "# and load it with kind: file to freeze the mapping:",
            ]
        lines += [
            "#   processing:",
            "#     graph_extraction:",
            f"#       schema_source: {{kind: file, path: {json.dumps(path)}}}",
            "#       schema_mode: align",
            "# Remove entity_types from the config, or from this file, first: both",
            "# may not set them. align keeps every extracted item; strict would",
            "# drop what this sample did not show.",
        ]
        if self.relationship_type_counts:
            lines.append("# Relationship types by statements in the tuning examples:")
            lines.extend(
                f"#   {label}: {count}"
                for label, count in self.relationship_type_counts.items()
            )
        body = str(
            yaml.safe_dump(
                self.document(), sort_keys=False, allow_unicode=True, width=1000
            )
        )
        return "\n".join(lines) + "\n" + body


class PromptTuner:
    """Generate domain-adapted ``custom_prompts`` from a corpus sample."""

    MAX_SAMPLE_CHARS = 8000

    def __init__(
        self,
        config: Config,
        *,
        providers: Providers | None = None,
    ) -> None:
        """Load the config's extraction schema source, as ingestion does.

        Raises:
            SchemaSourceError: The extraction schema source cannot be loaded.
        """
        # The examples and the draft are built on the schema the corpus will
        # be extracted with.
        source = config.processing.graph_extraction.schema_source
        self.schema_source: tuple[str, str] | None = (
            None
            if source.kind in (INLINE_SCHEMA_SOURCE, "file")
            else (source.kind, redact_location(source.path))
        )
        self.config = resolve_extraction_schema(config, providers=providers)
        self.providers = Providers.resolve(self.config, providers)
        self.factory = self.providers.llm_factory

    @property
    def schema_configured(self) -> bool:
        """Whether the config declares the entity types (or patterns, which
        name them): the examples are then extracted with them, else with the
        corpus profile's (:meth:`example_entity_types`)."""
        extraction = self.config.processing.graph_extraction
        return "entity_types" in extraction.model_fields_set or bool(
            extraction.relationship_patterns
        )

    def example_entity_types(self, profile: CorpusProfile) -> list[str]:
        """The entity types the examples are extracted with and a schema
        draft lists: the configured ones when :attr:`schema_configured`,
        else the profile's (one per :func:`schema_key`; none keeps the
        default ones).

        The profile's types replace the defaults so that the examples state
        the types the draft lists, and the draft can infer patterns; a
        configured schema is honoured instead.
        """
        if self.schema_configured:
            schema = self.config.processing.graph_extraction.extraction_schema()
            return [spec.label for spec in schema.entity_types]
        types: dict[str, str] = {}
        for item in profile.entity_types:
            spec = entity_type_spec(item)
            if spec is not None:
                types.setdefault(schema_key(spec.label), spec.label)
        return list(types.values())

    def _example_config(self, entity_types: list[str] | None) -> Config:
        """The config the examples are extracted with: this one, with
        ``entity_types`` in place of its entity types when given and not
        already configured."""
        if not entity_types or self.schema_configured:
            return self.config
        data = merge_over(
            self.config.as_input(),
            {"processing": {"graph_extraction": {"entity_types": entity_types}}},
        )
        try:
            return Config.model_validate(data)
        except ValidationError as e:
            logger.warning(
                "Extracting the examples with the configured entity types: "
                "the profile's are not a valid schema (%s)",
                format_validation_errors("", e),
            )
            return self.config

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
            model_id=self.config.model_id(ModelRole.GRAPH_EXTRACTION),
            prompt_class=CorpusProfilePrompt,
            parser=StrOutputParser(),
            custom_prompts=self.config.custom_prompts,
        )
        raw = await chain.ainvoke({"corpus_sample": corpus_sample})
        try:
            payload = parse_llm_json(raw, strict=True, answer_keys=_PROFILE_KEYS)
        except json.JSONDecodeError as e:
            # A default profile here would yield generic prompts presented as
            # tuned ones; fail so the user can retry or fix the model. The
            # model output stays out of the message (it is logged at ERROR)
            # and is logged only at DEBUG.
            logger.debug("Unparseable corpus profile output: %r", raw)
            raise LanguageModelError(
                "Corpus profiling returned no JSON profile (model output "
                f"{text_digest(str(raw))}; logged at DEBUG)"
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

    def extract_examples(
        self, texts: list[str], entity_types: list[str] | None = None
    ) -> tuple[list[TextUnit], list[TextUnitExtraction]]:
        """The sampled example chunks and the real ``GraphExtractor``'s
        records of them (MS GraphRAG's prompt tuning grounds its few-shots in
        the corpus so too, rather than asking the model to invent one), extracted with ``entity_types`` when given and the
        config does not declare its own (:attr:`schema_configured`); no
        records if extraction fails (examples are best-effort)."""
        chunks = self._sample_chunks(texts)
        if not chunks:
            return [], []

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
            extractor = GraphExtractor(
                self._example_config(entity_types), providers=self.providers
            )
            extractor.show_progress = False
            records, _ = extractor.extract_units(text_units)
        except Exception as exc:  # noqa: BLE001 - examples are best-effort
            logger.warning("Corpus-grounded example extraction failed: %s", exc)
            return text_units, []
        return text_units, records

    @classmethod
    def render_examples(
        cls, text_units: list[TextUnit], records: list[TextUnitExtraction]
    ) -> str:
        """Each unit with a non-empty record, rendered as a worked example
        (:meth:`_render_example`), up to ``MAX_EXAMPLES``."""
        # Each rendered example pairs a genuine passage with what was actually
        # extracted from it (its unit's record).
        by_unit = {record.text_unit_id: record for record in records}
        rendered: list[str] = []
        for unit in text_units:
            record = by_unit.get(unit.id)
            if record is None or not (record.entities or record.relationships):
                continue
            rendered.append(
                cls._render_example(
                    unit.text, unit_entities(record), record.relationships
                )
            )
            if len(rendered) >= cls.MAX_EXAMPLES:
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
        # Sentences or lines (``sentence_ends``: a CJK stop ends one on its
        # own, a Latin one only before whitespace and not in "3.5" or "e.g.").
        starts = [0, *sentence_ends(text)]
        for start, end in zip(starts, [*starts[1:], len(text)], strict=True):
            span = text[start:end].strip()
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

        Types are shown UPPERCASE, as the extraction rules ask: the merge step
        stores the most frequent entity type lower-cased.
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
                    f"<type>{_esc(entity.type).upper() or 'ENTITY'}</type>",
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
                    f"<type>{_esc(rel.type).upper() or 'RELATED_TO'}</type>",
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
            f"{domain} documents written in {language}. Follow the "
            "report requirements and the output format below exactly.\n\n"
            f"{CommunityReportPrompt.output_rules}"
        )
        return {
            "graph_extraction_system": graph_extraction_system,
            "community_report_system": community_report_system,
        }

    async def tune(
        self, texts: list[str], *, schema_draft: bool = False
    ) -> dict[str, Any]:
        """End-to-end: profile the corpus, generate examples, return overrides.

        The examples are extracted with the profile's entity types, or with
        the configured schema when there is one (:meth:`example_entity_types`).
        With ``schema_draft`` the result also holds ``"schema_draft"``, the
        :class:`SchemaDraft` of those entity types and the examples' records.
        """
        profile = await self.profile_corpus(texts)
        entity_types = self.example_entity_types(profile)
        text_units, records = self.extract_examples(texts, entity_types)
        profile.few_shot_examples = self.render_examples(text_units, records)
        logger.info(
            "Corpus profile: domain='%s', language='%s', %d entity types, examples=%s",
            profile.domain,
            profile.language,
            len(profile.entity_types),
            "yes" if profile.few_shot_examples else "no",
        )
        result: dict[str, Any] = {
            "profile": {
                "domain": profile.domain,
                "language": profile.language,
                "persona": profile.persona,
                "entity_types": profile.entity_types,
            },
            # A configured schema's entity types, not the profile's, are the
            # ones the tuned prompt recommends.
            "custom_prompts": self.build_custom_prompts(
                replace(profile, entity_types=entity_types)
                if self.schema_configured
                else profile
            ),
        }
        if schema_draft:
            result["schema_draft"] = self.schema_draft(entity_types, records)
        return result

    def schema_draft(
        self, entity_types: list[str], records: list[TextUnitExtraction]
    ) -> SchemaDraft:
        """The :class:`SchemaDraft` of the examples' ``entity_types`` and
        ``records`` and of every type and pattern the config declares, each
        with its description and IRI: a draft written over a reviewed
        schema drops nothing it declares."""
        schema = self.config.processing.graph_extraction.extraction_schema()
        return SchemaDraft.from_examples(
            list(schema.entity_types) if self.schema_configured else entity_types,
            records,
            relationship_types=schema.relationship_types,
            relationship_patterns=schema.relationship_patterns,
            schema_source=self.schema_source,
        )
