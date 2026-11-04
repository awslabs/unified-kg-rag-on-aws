# Documentation

> 🇰🇷 한국어판: [docs/index.ko.md](./index.ko.md)

The documentation of `unified-kg-rag-on-aws`, by what you want to do. New to
the project? Read the [Overview](./concepts/overview.md) first.

## Start here

| You are | Goal | Read, in order | Done when |
|---|---|---|---|
| **First-time evaluator** | See it work without AWS | [Getting started locally](./getting-started/local.md) → [Overview](./concepts/overview.md) → [Search strategies](./concepts/retrieval.md#search-strategies--when-to-use-which) → [Evaluators](./user-guide.md#evaluators) | Answers from both methodologies on a sample corpus, plus a judge-free `run-eval` |
| **Builder on AWS** | Ingest a real corpus and query it | [Getting started on AWS](./getting-started/aws.md) (existing stores, [required models](./getting-started/aws.md#required-models)) or [`iac/README.md`](../iac/README.md#quick-start) (CDK) → [Configuration](./guides/configure.md) → [Ingestion](./guides/ingest.md) → [Querying](./guides/query.md) → [Incremental indexing](./guides/incremental.md) → [Domain adaptation](./guides/domain-adaptation.md) | Incremental ingestion on, queries with filters, a tuned schema or prompts |
| **Operator** | Run, monitor, recover, upgrade | [`iac/README.md`](../iac/README.md) → [Operator Runbook](./operations.md) ([before a run](./operations.md#before-a-run), [when a run fails](./operations.md#when-a-run-fails), [alarms](./operations.md#alarms), [common errors](./operations.md#common-errors)) → [CHANGELOG](../CHANGELOG.md) | Every alarm and error leads to a procedure |
| **Evaluator or researcher** | Measure quality and cost, reproduce the defaults, compare with the papers | [Evaluation](./user-guide.md#6-evaluation-run-eval) → [Comparing strategies](./user-guide.md#comparing-strategies) → [Benchmarks](./user-guide.md#benchmarks-musique) → [Defaults and evidence](./concepts/defaults-and-evidence.md) → [Beyond the papers](./concepts/overview.md#beyond-the-papers) | A paired comparison with intervals and a gate, on MuSiQue or your own data |
| **Extender** | Add a backend, strategy, parser, schema or seed source | [Overview](./concepts/overview.md) → [Design Doc §2](./design.md#2-hexagonal-architecture-ports--adapters) → [§15 Extension Guide](./design.md#15-extension-guide) → [Extraction schema](./guides/extraction-schema.md) / [Seed graphs](./guides/seed-graphs.md) | A registered plugin that passes the fake-port conformance tests |
| **Contributor** | Change code and docs without breaking the gate | [CONTRIBUTING.md](../CONTRIBUTING.md) → [Design Doc §2](./design.md#2-hexagonal-architecture-ports--adapters) and [§13](./design.md#13-testing-strategy) → [Documentation rules](../CONTRIBUTING.md#documentation) | The pull request passes CI, including the docs checks |

## Page map

| Type | Page | Covers |
|---|---|---|
| Tutorial | [README Quickstart](../README.md#quickstart) | Choose how to run, install, first ingestion, query and evaluation |
| Tutorial | [Getting started locally](./getting-started/local.md), [Getting started on AWS](./getting-started/aws.md) | Step-by-step first runs: a sample corpus without AWS (both methodologies, a judge-free evaluation); prerequisites, required models, install, authentication and IAM for existing Neptune and OpenSearch |
| Explanation | [Overview](./concepts/overview.md) | Architecture, pipeline, the two methodologies, what the framework adds beyond the papers, limitations |
| Explanation | [Defaults and evidence](./concepts/defaults-and-evidence.md) | The measured defaults, how they were measured, dated prices, how to reproduce |
| Explanation | [Design Doc](./design.md) | Hexagonal architecture, algorithms, data model, extension guide, further reading |
| Explanation | [Retrieval](./concepts/retrieval.md), [Scopes and namespaces](./concepts/scopes-and-namespaces.md) | The search strategies and when to use which, hybrid scoring and rerank, answer style, context budget; index suffix, source scope, namespace, and why neither the suffix nor filters are access control |
| How-to | [User Guide](./user-guide.md) | The tasks not yet split out: evaluation and benchmarks, visualization, exit codes, library use |
| How-to | [`iac/README.md`](../iac/README.md) | CDK stacks, topology, cost drivers, `-c key=value` options, Guardrail deployment, after-deploy commands |
| How-to | [Local mode](./guides/local-mode.md), [Configuration](./guides/configure.md) | Local providers, differences from AWS mode, stores in containers; how configuration loads, the settings most runs change, environment variables, what the logs contain |
| How-to | [Ingestion](./guides/ingest.md), [Multilingual corpora](./guides/multilingual.md), [Querying](./guides/query.md) | The 12 stages, resume and force-rebuild, S3 sync, the response cache, large corpora and cost levers; translation, analyzers and chunk sizes for CJK text; running queries, filters, rerank status, the query deadline, interactive mode and memory |
| How-to | [Incremental indexing](./guides/incremental.md) | Enabling the registry, what each corpus or setting change does, rebuilds and the two rebuild signals, deletion scope and mass deletions, moving a corpus, failed and quarantined documents, concurrent runs |
| How-to | [Domain adaptation](./guides/domain-adaptation.md) | Entity types, prompt tuning, custom prompt overrides and how they are checked, the prompts without an override |
| How-to | [Extraction schema](./guides/extraction-schema.md), [Ontology](./guides/ontology.md), [Seed graphs](./guides/seed-graphs.md) | Schema modes, declaring types and patterns, the stored schema fields and their filters, what a schema change re-runs; reading the schema from an OWL/RDFS ontology; merging a knowledge graph you already have, how seeds meet the schema and the documents, their lifecycle and cost |
| How-to | [Guardrail](./guides/guardrail.md), [Observability](./guides/observability.md) | Screening queries and answers, the contextual grounding check; tracing with Jaeger, Langfuse, Phoenix or CloudWatch, export limits and privacy |
| How-to | More task guides (planned) | One page per task, split out of the User Guide |
| Reference | [CLI Reference](./reference/cli.md) | Every flag of every CLI (generated from the parsers) |
| Reference | [Configuration Reference](./reference/configuration.md) | Every configuration key: type, default, description (generated from the code) |
| Reference | [Model Catalog](./reference/models.md) | Supported Bedrock models, per-provider request differences, model selection notes (tiers, roles, the judge, output caps) |
| Reference | [Filters reference](./reference/filters.md) | Every filterable field per store, value typing and dates, per-strategy applicability, OpenSearch strictness, filtered graph expansion |
| Reference | [Ontology mapping reference](./reference/ontology-mapping.md), [Seed graph formats](./reference/seed-graph-formats.md) | The `owl` source's options, what it reads and its mapping rules, with a worked example; the JSONL, CSV and RDF seed formats and their options |
| Reference | [Telemetry reference](./reference/telemetry.md) | Every span and attribute, vendor attribute names, known limits, model-usage counters |
| Runbook | [Operator Runbook](./operations.md) | Capacity, re-ingestion, incremental runs, caches, backup and restore, upgrades, failed and quarantined documents, alarms, query deadlines, tenant isolation, common errors |
| Project | [CONTRIBUTING.md](../CONTRIBUTING.md), [CHANGELOG.md](../CHANGELOG.md), [SECURITY.md](../SECURITY.md) | Development workflow, release notes, reporting security issues |

## I want to

| I want to | Go to |
|---|---|
| Try it without an AWS account | [Getting started locally](./getting-started/local.md) |
| Deploy the whole stack on AWS | [`iac/README.md`](../iac/README.md#quick-start) |
| Enable the Bedrock models it needs | [Required models](./getting-started/aws.md#required-models) |
| Change a model or a setting | [Configuration](./guides/configure.md), [Model selection notes](./reference/models.md#model-selection-notes), [Configuration Reference](./reference/configuration.md) |
| Screen queries and answers | [Guardrail](./guides/guardrail.md) |
| Trace queries and ingestion runs | [Observability](./guides/observability.md) |
| Ingest a corpus, resume or rebuild | [Ingestion](./guides/ingest.md) |
| Index a Korean or mixed-language corpus | [Multilingual corpora](./guides/multilingual.md) |
| Pick a search strategy | [Search strategies](./concepts/retrieval.md#search-strategies--when-to-use-which) |
| Filter results by document or attribute | [Querying: Filters](./guides/query.md#filters), [Filters reference](./reference/filters.md) |
| Update the index as documents change or are deleted | [Incremental indexing](./guides/incremental.md) |
| Adapt extraction to my domain | [Domain adaptation](./guides/domain-adaptation.md), [Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning) |
| Import an ontology or an existing knowledge graph | [Ontology](./guides/ontology.md), [Seed graphs](./guides/seed-graphs.md) |
| Compare strategies or gate a change in CI | [Comparing strategies](./user-guide.md#comparing-strategies), [Rebuilding, re-judging and gating](./user-guide.md#rebuilding-re-judging-and-gating-a-comparison) |
| Know why a default is what it is | [Defaults and evidence](./concepts/defaults-and-evidence.md) |
| Use it from Python | [Using the library from Python](./user-guide.md#11-using-the-library-from-python) |
| Fix an error or a failed run | [Common errors](./operations.md#common-errors), [When a run fails](./operations.md#when-a-run-fails) |
| Add a backend, strategy or parser | [Design Doc §15](./design.md#15-extension-guide) |

## Glossary

| Term | Meaning |
|---|---|
| Ingestion | The 12-stage pipeline (`run-ingestion`) that turns documents into the graph and the indexes |
| Text unit | A chunk of a document; extraction, citations and deletion all work per text unit |
| Entity, relationship | A node and an edge of the extracted knowledge graph |
| Claim | An optional factual statement extracted per text unit (MS GraphRAG covariates) |
| Gleaning | Extra extraction rounds that ask the model what the first pass missed |
| Community, community report | A Leiden cluster of entities, and the LLM-written summary that `global` and `drift` read |
| Methodology, strategy | GraphRAG or LightRAG, and the concrete retrieval algorithm (`local`, `mix`, ...) chosen per query |
| Index suffix | The `<suffix>` in every index and graph label name (`processing.document_parsing.index_value`); one per tenant or corpus version |
| Doc-status registry | The DynamoDB (or SQLite) table of each document's hash and text units that makes ingestion incremental |
| Namespace | The indexes one index suffix names; one writing run per namespace at a time (a write lease) |
| Source scope | The corpus source a run diffs and deletes within (by default the source directory); document and text-unit ids derive from it |
| Quarantine | A document set aside after runs repeatedly died processing it; skipped until its file changes or the limit is raised |
| Seed graph | A knowledge graph you already have, merged into the extracted one as virtual documents |
| Extraction schema | The entity types, relationship types and patterns extraction follows, declared or read from an ontology |
