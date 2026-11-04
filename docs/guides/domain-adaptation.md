# Domain adaptation

> 🇰🇷 한국어판: [domain-adaptation.ko.md](./domain-adaptation.ko.md)

Four levers specialize the pipeline for a domain (medical, legal, finance,
...): the entity types and the prompts, on this page, and the extraction
schema and seed graphs, in the
[User Guide](../user-guide.md#9-domain-adaptation). `run-prompt-tuning`
drafts the first two from a corpus sample. A change to the entity types or
to an extraction prompt re-extracts the documents extracted with other
settings on the next incremental run
([Incremental indexing: Workflows](./incremental.md#workflows)). How the
prompts and tuning work inside:
[Design §11](../design.md#11-prompts-and-prompt-tuning).

## Recommended flow

`run-prompt-tuning` → review → merge the useful prompts and tune
`entity_types` → re-ingest.

## Entity types

`processing.graph_extraction.entity_types` is the cheapest lever with the
highest impact. It sets the categories the extraction prompt asks for, with
no prompt rewrite:

```yaml
processing:
  graph_extraction:
    entity_types:
      - "GENE: Genes, gene products, loci"
      - "DISEASE: Disorders, syndromes, conditions"
      - "DRUG: Medications, compounds, dosages"
      - "TRIAL: Clinical trials, studies, cohorts"
```

## Prompt tuning (`run-prompt-tuning`)

Samples documents from a directory, profiles the corpus (domain, language,
persona, entity types) with `models.roles.graph_extraction`, and writes a
domain-adapted `custom_prompts` YAML fragment to review and merge into
`config.yaml`. Every flag with its default:
[CLI Reference](../reference/cli.md#run-prompt-tuning).

```bash
run-prompt-tuning --source-directory ./source --output-path tuned_prompts.yaml --config-path config.yaml
```

The output holds a `custom_prompts` block and a `profile` with the detected
domain and entity types. **Review it**, then copy the prompts you want under
`custom_prompts:`.

- The tuned `graph_extraction_system` and `community_report_system` are a
  domain preamble followed by the built-in rules and output format, verbatim;
  generated few-shot examples come last.
- Entity categories still come from `processing.graph_extraction.entity_types`
  (`{entity_types}`): copy the profile's `entity_types` there.
- The few-shot examples are extracted with the profile's entity types, so
  they show the types the profile found. A config that declares its own
  entity types or patterns (inline or from a `schema_source`, which tuning
  loads as ingestion does; one that cannot be loaded exits 2) is honoured
  instead: the examples, and the entity types the tuned prompt names, are
  the configured ones.
- `--schema-out schema.yaml` also writes an
  [extraction schema](../user-guide.md#c-extraction-schema) draft: the
  entity types the examples were extracted with (above), the relationship
  types the tuning examples state most often (at most 12, with their counts
  in a comment) and a pattern only where every pair of endpoint types of a
  relationship type appears at least twice. Every type and pattern the
  config declares stays, with its description and IRI, and a relationship
  type with configured patterns gets no other. It is a
  `schema_source: {kind: file}` file, to load with `schema_mode: align`
  after review (its comment names the path relative to the `--config-path`
  file's directory); the same model answers give the same file. It never
  replaces your config or schema: naming the `--config-path` file or the
  schema file the config loads (with `--schema-out` or `--output-path`)
  exits 2, even with `--force`, and so does naming an existing file with
  either unless you add `--force`. Reviewing and loading the draft:
  [Extraction schema](../user-guide.md#c-extraction-schema), *Drafting a
  schema with prompt tuning*.
- Non-text formats go through the ingestion loaders (custom
  `ParserFactory.register_loader` ones included); unparsable files are skipped
  with a warning.
- If the model returns no JSON profile, the command exits non-zero and writes
  nothing.

## Custom prompts

Override a prompt's `*_system` / `*_human` text under `custom_prompts`
(`null`, the default, keeps the built-in prompt in
`unified_kg_rag/domain/prompts/`; five prompts have no override, listed at
the end of this page). Override what you need and leave the rest `null`.
Keep the `{braces}` variables, which the framework fills:

```yaml
custom_prompts:
  # A human-only override keeps the built-in system prompt and its output format.
  graph_extraction_human: |
    Extract medical entities and relationships from this clinical text:
    {input_text}
    Extraction Limits:
    - Maximum Entities: {max_entities_per_chunk}
    - Maximum Relationships: {max_relationships_per_chunk}

  entity_extraction_system: |
    You are a financial expert. Extract companies, instruments, markets, and metrics
    from user queries.
```

### Override keys

Each key has a `_system` and a `_human` field: `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `query_rewrite`, `entity_extraction`,
`drift_hyde`, `drift_primer`, `drift_action`, `strategy_selection`,
`keywords_extraction`, `global_map`, `pairwise_comparison` (the
`--pairwise-judge` judge; keep its JSON verdict format, with an `overall`
verdict under `evaluation.pairwise_overall`), `answer_span_extraction` (the
`evaluation.answer_span: extracted` span extractor) and the prompt-tuning
`corpus_profile`. Every key with its default:
[Configuration Reference](../reference/configuration.md#custom_prompts).

### Validation at load

Overrides are checked when the config loads; a mistake fails the load naming
the key:

- A `{name}` the prompt is not called with is an error (it would raise
  `KeyError` on every call). Write a literal brace as `{{` / `}}`:
  `{{"name": "Vendor"}}`. `run-prompt-tuning` output is already escaped.
- Each prompt's data variables must appear in its system or human template
  (`{input_text}` for `graph_extraction` and `claim_extraction`, `{query}` and
  `{context}` for `answer_generation`, ...); limits such as
  `{max_entities_per_chunk}`, `{entity_types}` and `{relationship_types}`
  (the declared relationship types, else the built-in examples) may be left
  out.

### Replacing a system prompt

A `*_system` override replaces the whole built-in system prompt, output
instructions included. For `graph_extraction` and `community_report` that is
the XML format the parser reads (and, for extraction, the verbatim
`<source_text>` rule the hallucination guard checks), so keep it: start from
`run-prompt-tuning` output or copy the format from
`unified_kg_rag/domain/prompts/graph_extraction.py`. An override of these two
that drops a tag the parser reads (`<entities>`, `<relationships>`,
`<source_text>`; `<community_name>`, `<summary>`, `<rating>`,
`<rating_explanation>`, `<findings>`) logs a warning naming the tags.

### Prompts without an override

Every prompt has a `custom_prompts.*_system` / `*_human` override except
the prompts of the intelligent chunker (`TextChunkingPrompt`), document
translation (`TextTranslationPrompt`), query translation
(`TranslationPrompt`), global search's community rating
(`CommunityRelevancePrompt`) and reduce step (`MapReduceSummaryPrompt`),
which are not overridable.

## Extraction schema and seed graphs

The relationship types and patterns extraction follows, declared or read
from an OWL/RDFS ontology, are in
[User Guide: Extraction schema](../user-guide.md#c-extraction-schema). A
knowledge graph you already have joins the extracted one as a seed graph:
[User Guide: Seed graphs](../user-guide.md#d-seed-graphs-bring-your-own-knowledge-graph).
