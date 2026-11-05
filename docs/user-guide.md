# Unified Knowledge Graph RAG on AWS — User Guide

> 🇰🇷 한국어판: [docs/user-guide.ko.md](./user-guide.ko.md)

The user guide is split into one page per task, concept and reference,
listed in the [documentation hub](./index.md). This page keeps the headings
of the user guide of the 0.2.3 release, so a link into one of its sections
still lands: each heading below keeps that section's fragment (heading id)
and points to the page that covers it.

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

Moved to [Getting started on AWS](./getting-started/aws.md),
[Getting started locally](./getting-started/local.md) and
[Local mode](./guides/local-mode.md).

### Runtime

Moved to [Getting started on AWS: Runtime](./getting-started/aws.md#runtime).

### AWS services

Moved to [Getting started on AWS: AWS services](./getting-started/aws.md#aws-services)
(the models to enable: [Required models](./getting-started/aws.md#required-models)).

### Install

Moved to [Getting started on AWS: Install](./getting-started/aws.md#install).

### Authentication

Moved to [Getting started on AWS: Authentication](./getting-started/aws.md#authentication).

### 2. Configuration

Moved to [Configuration](./guides/configure.md) (how configuration loads,
the settings most runs change, environment overrides) and the
[Configuration Reference](./reference/configuration.md) (every key).

### 2.1 `aws` — service endpoints & credentials

Moved to [Configuration Reference: `aws`](./reference/configuration.md#aws).

### 2.2 `fixing` — auto-repair malformed model output

Output fixing has no settings section: it is always on, and
`models.roles.fixing` picks its model
([Configuration Reference: `models.roles`](./reference/configuration.md#modelsroles);
the former keys: [Removed keys](./reference/configuration.md#removed-keys)).

### 2.3 `processing` — concurrency, chunking, translation, extraction

Moved to [Configuration Reference: `processing`](./reference/configuration.md#processing).

### 2.4 `graph` — analysis, community detection, visualization

Moved to [Configuration Reference: `graph`](./reference/configuration.md#graph).

### 2.5 `indexing` — OpenSearch & Neptune write side

Moved to [Configuration Reference: `indexing`](./reference/configuration.md#indexing).

### 2.6 `search` — retrieval, fusion, reranking, per-strategy knobs

Moved to [Configuration Reference: `search`](./reference/configuration.md#search).

### 2.7 `memory`, `cache`, `logging`

Moved to the Configuration Reference: [`memory`](./reference/configuration.md#memory),
[`cache`](./reference/configuration.md#cache) and
[`logging`](./reference/configuration.md#logging) (what the logs contain:
[Configuration](./guides/configure.md#what-the-logs-contain)).

### 2.8 `evaluation`

Moved to [Configuration Reference: `evaluation`](./reference/configuration.md#evaluation).

### 2.9 `custom_prompts`

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts)
(every key: [Configuration Reference: `custom_prompts`](./reference/configuration.md#custom_prompts)).

### 3. Ingestion (`run-ingestion`)

Moved to [Ingestion](./guides/ingest.md) (index namespaces:
[Scopes and namespaces](./concepts/scopes-and-namespaces.md)).

### CLI flags (verified)

Moved to [CLI Reference: `run-ingestion`](./reference/cli.md#run-ingestion).

### The 12 pipeline stages

Moved to [Ingestion: The 12 pipeline stages](./guides/ingest.md#the-12-pipeline-stages).

### Examples

Moved to [Ingestion: Examples](./guides/ingest.md#examples).

### Multilingual ingestion

Moved to [Multilingual corpora](./guides/multilingual.md).

### 4. Querying (`run-rag`)

Moved to [Querying](./guides/query.md) and [Retrieval](./concepts/retrieval.md)
(every filter field: [Filters reference](./reference/filters.md)).

### CLI flags (verified)

Moved to [CLI Reference: `run-rag`](./reference/cli.md#run-rag).

### Search strategies — when to use which

Moved to [Retrieval: Search strategies](./concepts/retrieval.md#search-strategies--when-to-use-which).

### Examples

Moved to [Querying: Examples](./guides/query.md#examples).

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

### Cross-run merge

Moved to [Incremental indexing: Cross-run merge](./guides/incremental.md#cross-run-merge).

### 6. Evaluation (`run-eval`)

Moved to [Evaluation](./guides/evaluate.md) (evaluators, dataset format,
comparisons, rebuilding and gating), the
[Evaluation output reference](./reference/evaluation-output.md) (result
files, summary, run manifest, comparison fields, evaluator details) and
[Benchmarks](./guides/benchmarks.md).

### CLI flags (verified)

Moved to [CLI Reference: `run-eval`](./reference/cli.md#run-eval).

### Evaluators

Moved to [Evaluation: Evaluators](./guides/evaluate.md#evaluators).

### Eval data format

Moved to [Evaluation: Eval data format](./guides/evaluate.md#eval-data-format).

### Examples

Moved to [Evaluation: Examples](./guides/evaluate.md#examples) (what the
result files hold: [Evaluation output](./reference/evaluation-output.md)).

### 7. Visualization (`run-visualization`)

Moved to [Visualization](./guides/visualize.md).

### CLI flags (verified)

Moved to [CLI Reference: `run-visualization`](./reference/cli.md#run-visualization).

### 8. Prompt tuning (`run-prompt-tuning`)

Moved to [Domain adaptation: Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning).

### CLI flags (verified)

Moved to [CLI Reference: `run-prompt-tuning`](./reference/cli.md#run-prompt-tuning).

### 9. Domain adaptation

Moved to [Domain adaptation](./guides/domain-adaptation.md) (entity types,
prompts, prompt tuning), [Extraction schema](./guides/extraction-schema.md),
[Ontology](./guides/ontology.md) and [Seed graphs](./guides/seed-graphs.md).

### A. `entity_types` (cheapest, highest-impact)

Moved to [Domain adaptation: Entity types](./guides/domain-adaptation.md#entity-types).

### B. `custom_prompts` overrides

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts)
(the five prompts without an override:
[Prompts without an override](./guides/domain-adaptation.md#prompts-without-an-override)).

### 10. Operations & troubleshooting

Moved to [Troubleshooting](./troubleshooting.md) (an FAQ and every error
message with its meaning and action),
[CLI exit codes and JSON output](./reference/exit-codes-and-json.md) and the
[Operator Runbook](./operations.md) (procedures).

### IAM permissions

Moved to [Getting started on AWS: IAM permissions](./getting-started/aws.md#iam-permissions).

### Common errors

Moved to [Troubleshooting: Common errors](./troubleshooting.md#common-errors).

### Large / multilingual / heterogeneous corpora

Moved to [Ingestion: Large and heterogeneous corpora](./guides/ingest.md#large-and-heterogeneous-corpora).

### Cost notes

Moved to [Ingestion: Cost levers](./guides/ingest.md#cost-levers).
