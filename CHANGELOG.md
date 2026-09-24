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
