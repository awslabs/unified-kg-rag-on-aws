# Unified Knowledge Graph RAG on AWS — User Guide

> 🇰🇷 한국어판: [docs/user-guide.ko.md](./user-guide.ko.md)

The user guide is now split into one page per task, concept and
reference, listed in the [documentation hub](./index.md). This page stays
as an index of where each of its sections went, so older links, including
the `--help` output of earlier versions, still land: each heading below
keeps the fragment (heading id) of the section it replaces and points to
the new page.

## Contents

1. Prerequisites & Installation: [Getting started on AWS](./getting-started/aws.md) (including [Required models](./getting-started/aws.md#required-models)), [Getting started locally](./getting-started/local.md) and [Local mode](./guides/local-mode.md)
2. Configuration: [Configuration](./guides/configure.md) (every key: [Configuration Reference](./reference/configuration.md)), [Model selection notes](./reference/models.md#model-selection-notes), [Guardrail](./guides/guardrail.md), [Observability](./guides/observability.md) and the [Telemetry reference](./reference/telemetry.md)
3. Ingestion: [Ingestion](./guides/ingest.md), [Multilingual corpora](./guides/multilingual.md) and [Scopes and namespaces](./concepts/scopes-and-namespaces.md)
4. Querying: [Querying](./guides/query.md), [Retrieval](./concepts/retrieval.md) and the [Filters reference](./reference/filters.md)
5. Incremental indexing: [Incremental indexing](./guides/incremental.md)
6. Evaluation: [Evaluation](./guides/evaluate.md), the [Evaluation output reference](./reference/evaluation-output.md) and [Benchmarks](./guides/benchmarks.md)
7. Visualization: [Visualization](./guides/visualize.md)
8. Prompt tuning: [Domain adaptation: Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning)
9. Domain adaptation: [Domain adaptation](./guides/domain-adaptation.md), [Extraction schema](./guides/extraction-schema.md), [Ontology](./guides/ontology.md) and [Seed graphs](./guides/seed-graphs.md)
10. Operations & troubleshooting: [Troubleshooting](./troubleshooting.md), [CLI exit codes and JSON output](./reference/exit-codes-and-json.md) and the [Operator Runbook](./operations.md)
11. Using the library from Python: [Using the library from Python](./guides/python-library.md)
12. Limitations: [Overview: Limitations](./concepts/overview.md#limitations)

## Moved sections

### 1. Prerequisites & Installation

Moved to [Getting started on AWS](./getting-started/aws.md) and
[Getting started locally](./getting-started/local.md).

### Runtime

Moved to [Getting started on AWS: Runtime](./getting-started/aws.md#runtime).

### AWS services

Moved to [Getting started on AWS: AWS services](./getting-started/aws.md#aws-services).

### Required models

Moved to [Getting started on AWS: Required models](./getting-started/aws.md#required-models).

### Install

Moved to [Getting started on AWS: Install](./getting-started/aws.md#install).

### Authentication

Moved to [Getting started on AWS: Authentication](./getting-started/aws.md#authentication).

### IAM permissions

Moved to [Getting started on AWS: IAM permissions](./getting-started/aws.md#iam-permissions).

### Local mode (no AWS)

Moved to [Getting started locally](./getting-started/local.md) (a first
run) and [Local mode](./guides/local-mode.md) (providers, differences from
AWS mode, containers).

### Differences from AWS mode

Moved to [Local mode: Differences from AWS mode](./guides/local-mode.md#differences-from-aws-mode).

### Provider flags

Moved to [Local mode: Provider flags](./guides/local-mode.md#provider-flags).

### Local stores in containers

Moved to [Local mode: Local stores in containers](./guides/local-mode.md#local-stores-in-containers).

### Local stores with Bedrock models

Moved to [Local mode: Local stores with Bedrock models](./guides/local-mode.md#local-stores-with-bedrock-models).

### 2. Configuration

Moved to [Configuration](./guides/configure.md).

### How configuration is loaded

Moved to [Configuration: How configuration is loaded](./guides/configure.md#how-configuration-is-loaded).

### Settings most runs change

Moved to [Configuration: Settings most runs change](./guides/configure.md#settings-most-runs-change).

### Model selection notes

Moved to [Model Catalog: Model selection notes](./reference/models.md#model-selection-notes).

### Guardrail

Moved to [Guardrail](./guides/guardrail.md).

### S3 cache encryption

Moved to [Configuration: S3 cache encryption](./guides/configure.md#s3-cache-encryption).

### What the logs contain

Moved to [Configuration: What the logs contain](./guides/configure.md#what-the-logs-contain).

### Custom prompts

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts).

### Environment variable overrides

Moved to [Configuration: Environment variable overrides](./guides/configure.md#environment-variable-overrides).

### Observability

Moved to [Observability](./guides/observability.md) (settings and
recipes) and the [Telemetry reference](./reference/telemetry.md) (trace
shape, usage counters).

### 3. Ingestion (`run-ingestion`)

Moved to [Ingestion](./guides/ingest.md).

### The 12 pipeline stages

Moved to [Ingestion: The 12 pipeline stages](./guides/ingest.md#the-12-pipeline-stages).

### Index suffix

Moved to [Scopes and namespaces: Index suffix](./concepts/scopes-and-namespaces.md#index-suffix).

### Multilingual ingestion

Moved to [Multilingual corpora](./guides/multilingual.md).

### 4. Querying (`run-rag`)

Moved to [Querying](./guides/query.md) and [Retrieval](./concepts/retrieval.md).

### Search strategies — when to use which

Moved to [Retrieval: Search strategies](./concepts/retrieval.md#search-strategies--when-to-use-which).

### Answer style

Moved to [Retrieval: Answer style](./concepts/retrieval.md#answer-style).

### Context budget

Moved to [Retrieval: Context budget](./concepts/retrieval.md#context-budget).

### Rerank status

Moved to [Querying: Rerank status](./guides/query.md#rerank-status).

### Query deadline

Moved to [Querying: Query deadline](./guides/query.md#query-deadline).

### Filters

Moved to [Querying: Filters](./guides/query.md#filters) (how to use them)
and the [Filters reference](./reference/filters.md) (every field, value
typing, per-strategy applicability).

### Interactive mode & conversation memory

Moved to [Querying: Interactive mode & conversation memory](./guides/query.md#interactive-mode--conversation-memory).

### 5. Incremental indexing

Moved to [Incremental indexing](./guides/incremental.md).

### Enable it

Moved to [Incremental indexing: Enable it](./guides/incremental.md#enable-it).

### Workflows

Moved to [Incremental indexing: Workflows](./guides/incremental.md#workflows)
(rebuilding: [Rebuild](./guides/incremental.md#rebuild-indexingreset) and
[Rebuild signals](./guides/incremental.md#rebuild-signals)).

### Deletion scope

Moved to [Incremental indexing: Deletion scope](./guides/incremental.md#deletion-scope).

### Failed documents

Moved to [Incremental indexing: Failed documents](./guides/incremental.md#failed-documents).

### Quarantined documents

Moved to [Incremental indexing: Quarantined documents](./guides/incremental.md#quarantined-documents).

### Concurrent runs

Moved to [Incremental indexing: Concurrent runs](./guides/incremental.md#concurrent-runs).

### Document size limit

Moved to [Incremental indexing: Document size limit](./guides/incremental.md#document-size-limit).

### Cross-run merge

Moved to [Incremental indexing: Cross-run merge](./guides/incremental.md#cross-run-merge).

### 6. Evaluation (`run-eval`)

Moved to [Evaluation](./guides/evaluate.md) (evaluators, dataset format,
comparisons, rebuilding and gating), the
[Evaluation output reference](./reference/evaluation-output.md) (result
files, summary, run manifest, comparison fields, evaluator details) and
[Benchmarks](./guides/benchmarks.md).

### Evaluators

Moved to [Evaluation: Evaluators](./guides/evaluate.md#evaluators).

### `ragas` contexts

Moved to [Evaluation output: `ragas` contexts](./reference/evaluation-output.md#ragas-contexts).

### `graph_aware`

Moved to [Evaluation output: `graph_aware`](./reference/evaluation-output.md#graph_aware).

### `retrieval`

Moved to [Evaluation output: `retrieval`](./reference/evaluation-output.md#retrieval).

### `answer_match`

Moved to [Evaluation output: `answer_match`](./reference/evaluation-output.md#answer_match).

### Eval data format

Moved to [Evaluation: Eval data format](./guides/evaluate.md#eval-data-format).

### Examples

The examples of sections 3, 4 and 6 moved to their guides:

- Ingestion: [Ingestion: Examples](./guides/ingest.md#examples).
- <a id="examples-1"></a>Querying: [Querying: Examples](./guides/query.md#examples).
- <a id="examples-2"></a>Evaluation: [Evaluation: Examples](./guides/evaluate.md#examples)
  (what the result files hold: [Evaluation output](./reference/evaluation-output.md)).

### Comparing strategies

Moved to [Evaluation: Comparing strategies](./guides/evaluate.md#comparing-strategies)
(every field: [Evaluation output: Comparison](./reference/evaluation-output.md#comparison)).

### Rebuilding, re-judging and gating a comparison

Moved to [Evaluation: Rebuilding, re-judging and gating a comparison](./guides/evaluate.md#rebuilding-re-judging-and-gating-a-comparison).

### Benchmarks (MuSiQue)

Moved to [Benchmarks](./guides/benchmarks.md).

### 7. Visualization (`run-visualization`)

Moved to [Visualization](./guides/visualize.md).

### 8. Prompt tuning (`run-prompt-tuning`)

Moved to [Domain adaptation: Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning).

### A. `entity_types` (cheapest, highest-impact)

Moved to [Domain adaptation: Entity types](./guides/domain-adaptation.md#entity-types).

### B. `custom_prompts` overrides

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts)
(the five prompts without an override:
[Prompts without an override](./guides/domain-adaptation.md#prompts-without-an-override)).

### 9. Domain adaptation

Moved to [Domain adaptation](./guides/domain-adaptation.md) (entity types,
prompts, prompt tuning), [Extraction schema](./guides/extraction-schema.md),
[Ontology](./guides/ontology.md) and [Seed graphs](./guides/seed-graphs.md).

### C. Extraction schema

Moved to [Extraction schema](./guides/extraction-schema.md).

### From an OWL/RDFS ontology

Moved to [Ontology](./guides/ontology.md) (install, configure, preview,
freeze) and the [Ontology mapping reference](./reference/ontology-mapping.md)
(options, what is read, mapping rules, worked example).

- <a id="ontology-load-cost"></a>Load cost: [Ontology: Load cost](./guides/ontology.md#load-cost).
- <a id="ontology-what-a-change-re-runs"></a>What a change re-runs: [Ontology: What a change re-runs](./guides/ontology.md#what-a-change-re-runs).
- <a id="ontology-preview"></a>Preview: [Ontology: Preview](./guides/ontology.md#preview).

### D. Seed graphs (bring your own knowledge graph)

Moved to [Seed graphs](./guides/seed-graphs.md) and the
[Seed graph formats reference](./reference/seed-graph-formats.md) (JSONL,
CSV, RDF).

- <a id="seed-precedence"></a>Precedence: [Seed graphs: Precedence](./guides/seed-graphs.md#precedence-documents-decide).
- <a id="seed-lifecycle"></a>Lifecycle: [Seed graphs: Lifecycle](./guides/seed-graphs.md#lifecycle).
- <a id="seed-cost-and-limits"></a>Cost and limits: [Seed graphs: Cost and limits](./guides/seed-graphs.md#cost-and-limits).

### 10. Operations & troubleshooting

Moved to [Troubleshooting](./troubleshooting.md) (an FAQ and every error
message with its meaning and action) and
[CLI exit codes and JSON output](./reference/exit-codes-and-json.md);
[IAM permissions](./getting-started/aws.md#iam-permissions) are in Getting
started on AWS, and the procedures in the
[Operator Runbook](./operations.md).

### CLI exit codes and JSON output

Moved to [CLI exit codes and JSON output](./reference/exit-codes-and-json.md).

### Large / multilingual / heterogeneous corpora

Moved to [Ingestion: Large and heterogeneous corpora](./guides/ingest.md#large-and-heterogeneous-corpora).

### Cost notes

Moved to [Ingestion: Cost levers](./guides/ingest.md#cost-levers).

### 11. Using the library from Python

Moved to [Using the library from Python](./guides/python-library.md).

### 12. Limitations

Moved to [Overview: Limitations](./concepts/overview.md#limitations).
