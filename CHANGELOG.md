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

### Fixed
- Ingestion stage-result cache keys now carry a fingerprint of the inputs that
  determine a stage's output (its config subtree, model id, and prompt
  overrides, cumulative over the upstream stages it consumes). A changed prompt,
  model, or processing rule previously resumed from stale stage output unless
  the run used `--force-rebuild`, a new `pipeline_id`, or hit a TTL expiry.
