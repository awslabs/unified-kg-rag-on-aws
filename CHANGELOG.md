# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial public release preparation. This is an AWS-native, open-source reference
framework that unifies the GraphRAG and LightRAG retrieval methodologies on
Amazon Bedrock, Neptune, OpenSearch, and DynamoDB.

### Added
- Two selectable retrieval methodologies: GraphRAG community-summary
  (`auto`/`drift`/`global`/`local`/`simple`) and LightRAG dual-level keyword
  (`mix`/`hybrid`/`naive`), sharing one ingestion/indexing/caching/hybrid-search
  stack.
- Incremental indexing via a DynamoDB document-status registry (content-hash
  diff, idempotent upserts, per-document lineage for deletion).
- Triple-hybrid retrieval (lexical + semantic + graph) with RRF fusion and
  Bedrock reranking; multilingual ingestion and retrieval.
- CLIs: `run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
  `run-prompt-tuning`.
- CDK deployment stack (`iac/`) with Well-Architected security defaults.
- Community reports now carry `text_unit_ids` and `document_ids`, indexed as
  keyword fields, so reports can be filtered by source document. These are the
  candidate sources associated with the community's members (community
  membership provenance): they may include material `max_entities_per_report`
  or the token budget kept out of the report prompt, and they do not establish
  sentence-level citation support. The schema alone does not backfill existing
  reports: those index both fields as empty lists until report generation and
  indexing are re-run (for example `run-ingestion --force-rebuild`).

### Fixed
- RRF fusion now accumulates a cross-store match. The fusion key was derived
  from a hash of the rendered content, and the graph and vector stores render
  the same artifact differently, so an entity present in both produced two keys
  and graph/vector rank agreement was never rewarded. An artifact now counts
  once per fusion bucket, at its best rank there: DRIFT concatenates every
  iteration into one bucket and dedupes by rendered content, so an entity
  reached over two paths arrived twice and its repetition outranked rank 1.
- Ingestion stage-result cache keys now carry a fingerprint of the inputs that
  determine a stage's output (its config subtree, model id, and prompt
  overrides, cumulative over the upstream stages it consumes). A changed prompt,
  model, or processing rule previously resumed from stale stage output unless
  the run used `--force-rebuild`, a new `pipeline_id`, or hit a TTL expiry.
  The resume point follows the cache, not the recorded status: an auto resume
  starts at the first completed stage whose output the cache no longer holds
  under the current inputs and recomputes it and every stage downstream; an
  explicit `--resume-from-stage` whose prerequisite is missing fails before any
  stage runs, naming the stage to resume from. Caches written before this
  change (keys without a fingerprint) carry no record of their inputs and are
  treated as a miss: the run logs the legacy key it found and recomputes from
  that stage once. A completed stage that produced an empty output now caches
  it, so "produced nothing" is distinguishable from "not cached".
- The gleaner now applies the `ENTITY_CORRECTION` and `RELATIONSHIP_CORRECTION`
  issues `GraphRefinementPrompt` asks for (wrong entity name or type, wrong
  relationship type or direction). Issue dispatch branched only on the two
  `MISSING_*` types, so every correction the model returned was discarded, and
  gleaning could only ever add to the graph, never fix it. The prompt now also
  specifies the `<details>` shape for both correction types, which it previously
  requested without defining. After a round's corrections and duplicate merge,
  every relationship's `source_name` / `target_name` is rewritten from the final
  entity id-to-name mapping, so a renamed or merged entity is named the same on
  its edges (which the indexers read) as on the node.
