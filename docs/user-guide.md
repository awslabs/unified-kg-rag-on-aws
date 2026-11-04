# Unified Knowledge Graph RAG on AWS — User Guide

> 🇰🇷 한국어판: [docs/user-guide.ko.md](./user-guide.ko.md)

This is the practical, how-to-use guide for **unified-kg-rag-on-aws** — an AWS-native
knowledge-graph RAG framework that builds knowledge graphs from large,
multilingual document corpora and answers questions over them. It reimplements
two retrieval methodologies on one stack: **Microsoft GraphRAG**
(community-summary) and **LightRAG** (dual-level keyword), selectable per query.

- For the *what / why* and a one-minute quickstart, see [README.md](../README.md).
- For a map of every page, see the [documentation hub](./index.md); for the
  concepts and what the framework adds beyond the papers, the
  [Overview](./concepts/overview.md).
- For *internals / architecture* (hexagonal layers, ports & adapters, the
  dependency rule), see [docs/design.md](./design.md).

The six console entry points (defined as `pyproject` scripts) all start
through `application.cli.launch`, which then loads the module below:

| Script | Module | Purpose |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | Build / update the knowledge graph |
| `run-rag` | `application.cli.run_rag_chain` | Query the graph |
| `run-eval` | `application.cli.run_evaluation` | Evaluate retrieval + generation |
| `run-visualization` | `application.cli.run_visualization` | Render an exported graph (no ingestion) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | Generate domain-adapted prompts |
| `prepare-benchmark` | `application.cli.prepare_benchmark` | Turn a public benchmark (MuSiQue) into a corpus + `run-eval` dataset |

Every CLI's `--help` ends with an example invocation and a link to its section here; every flag is in the [CLI Reference](./reference/cli.md).

## Contents

1. [Prerequisites & Installation](#1-prerequisites--installation) (including [Required models](#required-models))
2. [Configuration](#2-configuration) (every key: [Configuration Reference](./reference/configuration.md))
3. [Ingestion (`run-ingestion`)](#3-ingestion-run-ingestion)
4. [Querying (`run-rag`)](#4-querying-run-rag)
5. [Incremental indexing](#5-incremental-indexing)
6. [Evaluation (`run-eval`)](#6-evaluation-run-eval)
7. [Visualization (`run-visualization`)](#7-visualization-run-visualization)
8. [Prompt tuning (`run-prompt-tuning`)](#8-prompt-tuning-run-prompt-tuning)
9. [Domain adaptation](#9-domain-adaptation)
10. [Operations & troubleshooting](#10-operations--troubleshooting)
11. [Using the library from Python](#11-using-the-library-from-python)
12. [Limitations](#12-limitations)

Related pages: [CLI Reference](./reference/cli.md) · [Configuration Reference](./reference/configuration.md) · [Model Catalog](./reference/models.md) · [Operator Runbook](./operations.md) ·
[Design Doc](./design.md) · [`iac/README.md`](../iac/README.md)

---

## 1. Prerequisites & Installation

### Runtime

- **Python 3.11 – 3.12**
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager; `pip` works too)

### AWS services

AWS mode only: [Local mode](#local-mode-no-aws) replaces each of them with a
local component.

| Service | Required? | Used for |
|---|---|---|
| **Amazon Bedrock** | Yes | Every LLM call, embeddings and reranking (enable model access for the model ids you configure) |
| **Amazon Neptune** | Yes | The knowledge graph and multi-hop traversal at query time |
| **Amazon OpenSearch** | Yes | Vector and BM25 lexical indices (text units, entities, community reports, relationships, claims) |
| **Amazon S3** | Only for `--s3-sync` or `s3://` paths | Stage-cache sync, `s3://` datasets and the CDK corpus |
| **Amazon DynamoDB** | Only for incremental indexing | The document-status registry that diffs the corpus by content hash |

### Required models

Enable access to these Bedrock models (the code defaults; a model you
configure replaces the one in its row) before the first run:

| Role | Default model id | Config key |
|---|---|---|
| Default tier (extraction, gleaning, community reports, answers) | `anthropic.claude-sonnet-5-5` | `models.default` |
| `run-eval` LLM judges (only when an LLM judge runs) | `anthropic.claude-opus-5-5` | `models.roles.evaluation` |
| Fast tier (chunking, translation, summarization, routing, map steps) | `anthropic.claude-haiku-5-5` | `models.fast` |
| Embeddings (indexing, queries and evaluation) | `amazon.titan-embed-text-v2:0` | `models.embedding.model_id` |
| Reranking | `cohere.rerank-v3-5:0` | `models.rerank` |

- **Region.** Every Bedrock call goes to `aws.bedrock.region_name`
  (default: `aws.region_name`, `us-west-2`, which offers all four). Where
  rerank models are missing (for example `ap-northeast-2`), point it at a
  region that has them or set `search.reranking.enabled: false`; a private
  VPC whose only Bedrock route is its VPC endpoint must keep its own region.
- **Inference profiles.** Claude Sonnet 5.5 and Haiku 5.5 are invocable only
  through a cross-region inference profile: keep
  `aws.bedrock.enable_global_profile: true` and grant
  `bedrock:ListInferenceProfiles` (if no profile resolves, model creation
  fails with a message naming this remedy).
- **Permissions.** See [IAM permissions](#iam-permissions); reranking needs a
  statement of its own.

The [Model Catalog](./reference/models.md) lists every curated model and its limits.

The framework connects to services that **already exist** and never creates
them. Run it in [local mode](#local-mode-no-aws), against your own services
(record their endpoints in `config.yaml`, [§2](#2-configuration)), or on the
bundled CDK app in [`iac/`](../iac/README.md), which provisions the whole
stack ([Quick start](../iac/README.md#quick-start)); its stores are VPC-only,
so the CLIs run there as tasks that receive the endpoints as environment
variables ([After deploy](../iac/README.md#after-deploy)).

### Install

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
uv sync --extra eval --extra viz  # or: pip install -e '.[eval,viz]'
```

`uv sync` alone installs ingestion and queries. Two extras add the rest;
without one, the command that needs it stops with exit `2`, naming it:

- **`eval`**: the LangChain and RAGAS judges of `run-eval` (ragas, datasets,
  pandas, langchain-openai), both enabled by default ([§6](#evaluators)).
  The LLM-free evaluators need no extra.
- **`viz`**: pyvis and Bokeh for `run-visualization` ([§7](#7-visualization-run-visualization)).

**Markdown and HTML** need the `unstructured` extra
(`uv sync --extra unstructured`; it requires `unstructured>=0.24.1`, which
fixes URL-partitioning SSRF and unescaped markup). Without it, only `.pdf`,
`.txt`, `.csv` and `.json` are parsed and ingestion skips `.md` / `.html` files with one warning
per extension.

The extra tokenizes with a spaCy model (`en_core_web_sm`) that it downloads
from GitHub at the first Markdown or HTML file. Without network access,
install it beforehand, once, where the network is reachable:
`python -c "from unstructured.nlp.tokenize import sent_tokenize; sent_tokenize('Ready.')"`.
Otherwise the first such file waits up to 120 s and fails, and every later
one of the run fails at once with the same error.

The deployed image (`docker/Dockerfile`) leaves the extra out (it adds about
200 MB), so a Step Functions ingest of `.md` / `.html` files fails with "No
supported files found": convert the files, or build with
`--build-arg UV_EXTRAS="--extra unstructured"`, which also installs the spaCy
model in the image (a task in the private VPC cannot download it). The image's
`docker/config.yaml` holds no endpoints; the CDK compute stack injects them.

**OWL/RDFS ontologies** as the extraction schema need the `ontology` extra
(`uv sync --extra ontology`; [From an OWL/RDFS ontology](#from-an-owlrdfs-ontology)).

### Authentication

1. **AWS credentials** come from the standard credential chain:
   `aws.profile_name` names a profile, `null` uses the default chain (env
   vars, instance role, ...). Neptune signs with SigV4 when
   `aws.neptune.use_iam: true`.
2. **OpenSearch** uses IAM (`aws.opensearch.use_iam: true`) or, with
   `use_iam: false`, a username and password in a `.env` file (copy
   `.env-template`), which the `run-*` CLIs load:

   ```bash
   UKR__AWS__OPENSEARCH__USERNAME=your_opensearch_username
   UKR__AWS__OPENSEARCH__PASSWORD=your_opensearch_password
   ```

### IAM permissions

The principal running the CLIs needs Bedrock (`InvokeModel`,
`InvokeModelWithResponseStream` and `CountTokens` for your model ids and
embeddings; without `CountTokens` token counts are estimated), Neptune
(connect / SigV4 with `use_iam: true`), OpenSearch (read and write the
configured indices), S3 (the configured bucket) and DynamoDB (with incremental
indexing). Three Bedrock actions need statements of their own:

- **`bedrock:Rerank` on `Resource: "*"`.** The Rerank API authorizes against
  another resource shape than `InvokeModel`, so a statement scoped to model or
  inference-profile ARNs denies it (the CDK task role grants it separately).
  Without it every query logs `Reranking failed: ...` at ERROR and returns
  the fused results unreranked (`metadata.rerank.status: failed`); if you
  cannot grant it, set `search.reranking.enabled: false`.
- **`bedrock:ListInferenceProfiles` on `Resource: "*"`**, an account-level
  read the model resolver makes.
- **`bedrock:ApplyGuardrail`** on the guardrail's ARN
  (`arn:aws:bedrock:<bedrock region>:<account>:guardrail/<id>`), when a
  guardrail is set.

### Local mode (no AWS)

Local mode runs ingestion, querying and evaluation on one host without an AWS
account, for development and evaluation (AWS stays the default). By default
the graph store, the vector/lexical store and the registry are SQLite files, so
the only service to run is [Ollama](https://ollama.com) at
`http://localhost:11434`:

```bash
uv sync --extra local
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
uv run run-ingestion --backend local --source-directory ./docs-in
uv run run-rag --backend local --query "..."
```

- **Files.** Without a config file the three stores live in
  `.unified-kg-rag/` of the working directory, so run every command from the
  same directory, or use a config file with `backend: local` (relative paths
  resolve against it). To start over, delete the three files together or run
  once with `indexing.reset: true`.
- **Incremental indexing** is off in the preset; turn it on with
  `providers.doc_status.enabled: true` (the run logs the registry's absolute
  path).
- **CPU-only PyTorch.** On Linux `uv sync --extra local` installs the locked
  CUDA build (several GB). For the CPU build run `uv sync` then
  `uv pip install --torch-backend cpu -e '.[local]'` (a later plain `uv sync`
  removes it).

`backend: local` (in the file, or `--backend local`) fills in these providers
for every key the config leaves unset; a key the file sets always wins
(`providers.<role>.kind`, an `aws.neptune` / `aws.opensearch` connection key,
an embedding or rerank model id). `config-template.yaml` leaves them
commented out, so a copy switches with `backend: local` alone. Roles can be
mixed, for example local stores with Bedrock models:

| Role | `providers.<role>.kind` | Local default |
|---|---|---|
| Language models | `llm`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `qwen2.5:7b-instruct` at `http://localhost:11434` |
| Embeddings | `embedding`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `bge-m3` (1024 dimensions) |
| Reranking | `rerank`: `bedrock` \| `cross_encoder` \| `none` | `cross_encoder`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Graph store | `graph`: `neptune` \| `gremlin` \| `sqlite` | `sqlite` at `providers.graph.sqlite_path` (default `.unified-kg-rag/graph.sqlite3`) |
| Vector/lexical store | `vector`: `opensearch` \| `sqlite` | `sqlite` at `providers.vector.sqlite_path` (default `.unified-kg-rag/vector.sqlite3`) |
| Doc-status registry | `doc_status`: `dynamodb` \| `sqlite` | `sqlite` at `providers.doc_status.sqlite_path` (default `.unified-kg-rag/doc-status.sqlite3`) |

`gremlin` (a TinkerPop Gremlin Server) and `opensearch` (a local OpenSearch)
are reached through the `aws.neptune` and `aws.opensearch` keys, which the
preset points at `localhost` ([containers](#local-stores-in-containers)).

- **Models.** With a non-Bedrock LLM, `models.default` is required (the
  preset sets it) and `models.fast` defaults to it.
  `providers.llm.context_window_size` and `max_output_tokens` describe the
  model (the window goes to Ollama as `num_ctx`); the preset fits
  `search.token_manager.max_context_tokens` (22000) to the default
  32,768-token window, only for a non-Bedrock LLM left at that window (a
  Bedrock LLM or an explicit `context_window_size` keeps `30000`). An
  OpenAI-compatible server (vLLM, llama.cpp server, LM Studio) takes
  `base_url` (its `/v1` root) and, if needed, `api_key`.
- **Embeddings.** A non-Bedrock provider needs `models.embedding.model_id`
  and `dimension`. The dimension is fixed into the vector mappings, so
  changing either needs a full re-index; another model than `bge-m3` needs its
  own dimension. The local providers trust the configured dimension (it is
  not checked against the model at start): a wrong value fails at the first
  vector write or search.
- **Reranking.** The cross-encoder downloads its model from Hugging Face on
  the first query. Without the `[local]` extra (or with `kind: none`) queries
  keep the fused order, logged once and reported as
  `metadata.rerank.status` `unavailable` (`disabled` for `kind: none`).
- **Embedded stores.** Each `sqlite` store is one file created on first use
  (a relative `sqlite_path` resolves like the other paths); the
  `indexing.neptune` and `indexing.opensearch` settings apply to them.
- **Token counting** uses the script-aware estimate
  (`providers.llm.tokenizer: tiktoken` for a BPE encoding).
- **Custom kinds** register a builder with
  `adapters.provider_registry.register_provider`
  ([design §15](./design.md#15-extension-guide)).

#### Differences from AWS mode

- **Quality.** Small local models extract noticeably worse (more failed
  chunks, weaker resolution and reports): use the largest model the hardware
  runs. The `langchain` / `ragas` judges score less reliably.
- **Model behaviour.** No prompt caching or reasoning effort; a response cut
  at the output limit is not detected (keep `max_output_tokens` at what the
  model can generate); token counts and so context budgets are estimates. A
  server that refuses a call (HTTP 4xx) or stays unreachable fails the query
  with the client's error instead of answering "could not find".
- **Guardrail.** With `aws.bedrock.guardrail.identifier` set, the query
  chain still screens queries and answers through Bedrock `ApplyGuardrail`
  (AWS credentials needed).

**The embedded vector store** differs from OpenSearch:

- Text mapped to `nori`, `kuromoji` or `smartcn` is split into overlapping
  CJK character bigrams instead of morphological analysis: a name still
  matches inside a longer word (홍길동 in 홍길동으로부터), but ranking differs.
- `english` stems only with `snowballstemmer` from the `[local]` extra (else
  one warning); emoji are not indexed. There is no fuzzy matching, so a typo,
  and under `standard` an inflected form ("vendors"), matches only through
  the vector side.
- kNN is exact (every filtered document is scored), so cost grows linearly.
- Each searched index is held in memory (about documents x dimension x 4
  bytes per vector field), sized for about 100,000 documents (a larger one
  logs a warning: use OpenSearch). The first query after a write reloads the
  index, and the first lexical one re-analyzes the searched field: about 10 s
  for 50,000 documents of 120 words, about three times slower under
  `english` above about 65,000 distinct words.

**SQLite files** allow one writer at a time: run one ingestion per store file
(queries can read alongside); a writer waiting longer than
`search.store_timeout_seconds` fails with "stayed locked". Keep the files on
a local disk, not NFS, SMB or Amazon EFS
([Common errors](./operations.md#local-mode)).

#### Provider flags

`run-ingestion`, `run-rag`, `run-eval`, `run-prompt-tuning` and
`run-visualization` take the provider flags (`--backend`, `--llm-provider`,
`--llm-model`, `--embedding-provider`, `--embedding-model`,
`--embedding-dimension`, `--rerank-provider`, `--model-base-url`). Each
overrides its key of the loaded config, a key the file sets included; an
omitted flag leaves the config as it is (`run-visualization` calls no model
or store and only accepts them). The keys each flag sets are in the
[CLI Reference](./reference/cli.md#provider-flags).

#### Local stores in containers

[`docker/compose.local.yaml`](../docker/compose.local.yaml) runs the stores
as servers: a Gremlin Server (in-memory TinkerGraph), a single-node
OpenSearch 2.13 with `analysis-nori` (security plugin off) and, under the
`models` profile, Ollama. OpenSearch is the engine of Amazon OpenSearch
Service, so use it to check ranking before an AWS deployment
([Multilingual ingestion](#multilingual-ingestion) covers `nori` on AWS).
[`docker/config.local.yaml`](../docker/config.local.yaml) selects
`providers.graph.kind: gremlin` and `providers.vector.kind: opensearch` and
turns on incremental indexing. It needs Docker Compose 2.23.1 or later.

```bash
uv sync --extra local
docker compose -f docker/compose.local.yaml --profile models up -d --wait
docker compose -f docker/compose.local.yaml exec ollama ollama pull qwen2.5:7b-instruct
docker compose -f docker/compose.local.yaml exec ollama ollama pull bge-m3
uv run run-ingestion --config-path docker/config.local.yaml --source-directory ./docs-in
uv run run-rag --config-path docker/config.local.yaml --query "..."
docker compose -f docker/compose.local.yaml down -v
```

- **`down -v`** deletes the stores' data but not the SQLite registry
  (`.unified-kg-rag/doc-status.sqlite3` at the repository root): delete it or
  set `indexing.reset: true` before ingesting into fresh stores, or the
  incremental run stops before writing anything.
- No model is pulled automatically; the preset connects with plain `ws://`
  and `http://`, without SigV4 or authentication.
- Smoke test without models:
  `LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py`.

Differences from Neptune:

- TinkerGraph is in memory: after a Gremlin Server restart the next
  incremental run stops before writing and asks for `indexing.reset: true`
  (or deleting the registry).
- A list property written at vertex creation keeps duplicate values.
- TinkerGraph has no transactions: keep `indexing.neptune.index_concurrency`
  at 1 (the default). With concurrent writers a retried edge rewrite can
  leave a duplicate edge.
- Writes look elements up by their `id` property, which the compose file's
  server indexes (without the index it slows down as the graph grows), or
  with `indexing.neptune.vertex_ids: custom` vertices by their vertex id,
  which the compose file's graph accepts as a string. Use it for
  development-sized corpora.
- The stores have no authentication or TLS; the compose file binds them to
  `127.0.0.1` and drops all Linux capabilities.

#### Local stores with Bedrock models

Set `providers.llm.kind`, `providers.embedding.kind` and
`providers.rerank.kind` to `bedrock` in a config with `backend: local`
(Bedrock model ids then validate against the catalog) and use AWS credentials
with Bedrock access. For the containers, start only the stores
(`docker compose -f docker/compose.local.yaml up -d --wait`) and edit a copy
of `docker/config.local.yaml`.

---

## 2. Configuration

Copy the template and point every CLI at it with `--config-path config.yaml`:

```bash
cp config-template.yaml config.yaml
```

A config file holds only the keys you change; the others keep their
defaults. Every key, its type, default and description is in the generated
[Configuration Reference](./reference/configuration.md); this section covers what needs
more than a table row.

### How configuration is loaded

Each layer overrides the one before:

1. **Built-in defaults** (`unified_kg_rag/domain/models/config.py`). Without
   `--config-path`, a CLI runs on these plus the environment.
2. **The `backend` preset**: `backend: local` (or `--backend local`) fills in
   the local providers and stores for every key the file and the environment
   leave unset ([Local mode](#local-mode-no-aws)).
3. **Your YAML file**. A dict-valued key (such as `logging.library_levels`)
   replaces the whole default dict, it does not merge.
4. **Environment variables** ([below](#environment-variable-overrides)).
5. **Provider flags** (`--backend`, `--llm-provider`, ...), applied last. The
   config is rebuilt from what the file and the environment set, so switching
   `--backend` drops only what the old preset filled in, never a value you set.

The file is validated when it loads. A wrong type or value
(`Configuration validation error: ...`), invalid YAML, a missing
`--config-path` file and an unknown key (in the file or as a `UKR__` variable)
are usage errors (exit 2). Unknown keys are refused because a misspelled key
silently keeps its default; `--allow-unknown-keys`
(`allow_unknown_keys=True` for `get_config`) logs a WARNING per key instead.
A moved key fails naming its replacement
([Moved keys](./reference/configuration.md#moved-keys)).

Retries are configured in one section, `retry`: `max_attempts` for model
calls (ingestion items, Bedrock embeddings and query-time calls) with its
backoff, and an attempt count for the callers whose defaults differ (model
servers, the graph-store indexer, OpenSearch queries, RAGAS judges). Every
attempt count includes the first attempt; `1` disables the retry.

### Settings most runs change

| Key | Default | Change it when |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | Stores in another region (Bedrock follows unless `aws.bedrock.region_name` is set) |
| `aws.neptune.endpoint`, `aws.opensearch.endpoint` | `null` | Always on AWS (the CDK stack injects them) |
| `providers.doc_status.enabled` | `false` | Incremental indexing ([§5](#5-incremental-indexing)) |
| `models.default`, `models.fast` | Claude Sonnet 5.5, Claude Haiku 5.5 | Other models ([notes](#model-selection-notes)) |
| `processing.max_concurrency` | `40` | Lower on `ThrottlingException` retries, raise if quota allows |
| `processing.graph_extraction.entity_types` | 7 generic types | Domain adaptation ([§9](#9-domain-adaptation)) |
| `processing.translation.target_language` | `"en"` | Another corpus or answer language ([Multilingual](#multilingual-ingestion)) |
| `processing.claim_extraction.enabled` | `false` | Claims in `local` search (one more LLM call per text unit) |
| `processing.document_parsing.index_value` | `null` (`default`) | Separate corpora or tenants ([Index suffix](#index-suffix)) |
| `graph.community_detection.enabled` | `true` | LightRAG-only, lighter ingestion (`global` and `drift` need it) |
| `search.reranking.enabled` | `true` | No rerank model in the region, or no reranking wanted |
| `search.token_manager.max_context_tokens` | `30000` | More (or cheaper, less) context ([Context budget](#context-budget)) |
| `indexing.reset` | `false` | One run that rebuilds the stores from the whole corpus (quarantined documents skipped) |
| `aws.bedrock.guardrail.identifier` | `null` | Screen queries and answers ([Guardrail](#guardrail)) |
| `observability.enabled` | `false` | OpenTelemetry traces ([Observability](#observability)) |

### Model selection notes

The `models` section names the model of each role; ids are checked against
the family's provider (`providers.llm.kind`, `providers.embedding.kind`,
`providers.rerank.kind`). Every LLM role runs on one of two tiers, so one line
switches the pipeline; a `models.roles` entry (a model id, or `default` /
`fast` for the other tier) wins over its tier:

| Tier | Roles |
| --- | --- |
| `default` | `fixing`, `graph_extraction`, `gleaning`, `claim_extraction`, `community_report`, `entity_extraction`, `answer_generation`, `evaluation` (with a provider other than Bedrock; see below) |
| `fast` | `chunking`, `translation`, `description_summary`, `query_translation`, `strategy_selection`, `query_rewrite`, `community_relevance`, `global_map`, `global_reduce`, `drift_primer`, `drift_action`, `answer_span` |

```yaml
models:
  default: "openai.gpt-6-sol"          # every default-tier role
  roles:
    answer_generation: "anthropic.claude-opus-5-5"   # one role pinned
    query_rewrite: default                          # moved to the other tier
```

- **The judge is its own model.** On Bedrock an unset
  `models.roles.evaluation` is `anthropic.claude-opus-5-5`, not the
  default tier: an LLM judge favours its own answers
  ([measurement and dated list prices](./concepts/defaults-and-evidence.md#the-evaluation-judge)).
  Opus 5.5 costs more per token than Sonnet 5.5, and only `run-eval`'s LLM
  judges call it. When the
  answer model is Opus 5.5 itself, the unset judge falls back to the default
  tier's model, or to Sonnet 5.5 if that is the answer model too; `run-eval`
  logs the judge it chose, and why, at INFO. With `--from-results` or
  `--baseline-results` the saved runs' answer models (their manifests'
  `models.answer_generation`) count as answer models too. Set
  `models.roles.evaluation: default` for the earlier judge (the answer
  model; `run-eval` then warns that it judges its own answers, as it does
  for any configured judge that is the answer model), or another model when
  Opus 5.5 is not enabled in your account or region. With another provider
  the judge stays on the `default` tier (and is warned about when it judges
  its own answers). The CDK stack grants every Bedrock model by default; a
  scoped `bedrock_model_arns` must list the judge model too.
- The embedding model (`models.embedding.model_id`) serves the indices, the
  queries and RAGAS; changing it or its
  `dimension` needs a re-index. On Bedrock, embedding and rerank ids are a
  closed list ([Model Catalog](./reference/models.md#model-ids-and-capability-records));
  other providers take any model name.
- Settings of an unselected provider (`providers.llm.temperature` with
  Bedrock models) are ignored with a WARNING.
- **Output cap.** `aws.bedrock.default_max_output_tokens` is 16384 because
  Bedrock reserves input + `max_tokens` against the tokens-per-minute quota
  when a request starts; asking for the model maximum throttles concurrent
  ingestion. Long-output prompts derive a higher floor that wins (by default
  23192 for graph extraction and gleaning, 18992 for document translation,
  29792 for claims), rising with their per-chunk limits. The short-answer
  query calls ask for less (`search.output_ceilings`, 4096 each: the
  router, the query entity and keyword extractions, the dynamic-selection
  report ratings; `evaluation.answer_span_output_ceiling` for the answer-span
  extractor), never less than the call's thinking budget plus 1024 (an
  effort of low budgets 2048, high 8192), so an `auto` query reserves about
  29k output tokens instead of 49k.
- Curated models, request shapes, prompt caching and
  `aws.bedrock.model_overrides` are in the [Model Catalog](./reference/models.md).

### Context budget

`search.token_manager.max_context_tokens` (`30000`, upstream LightRAG's
budget) is the retrieved context the answer prompt gets, clamped with a
warning to what `models.roles.answer_generation` accepts beside its output
reservation; `null` derives it from that model's window. Per-type shares and
`local` slot quotas: [reference](./reference/configuration.md#searchtoken_manager).

### Guardrail

Setting `aws.bedrock.guardrail.identifier` screens two texts of every query
with the `ApplyGuardrail` API; the guardrail must exist in
`aws.bedrock.region_name`.

- **What is screened.** The raw user query, once, before memory, rewriting,
  routing and retrieval (`check_input`), and the final answer (`check_output`;
  a streamed answer then arrives as one chunk). Only these are billed.
- **What happens.**
  - A blocked query gets the guardrail's message, retrieves nothing, writes
    no memory and sets `RAGOutput.metadata.blocked`. A masked query continues
    with the masked text only.
  - A blocked or masked answer is replaced by the guardrail's text (blocked:
    no `sources` either). Masking covers the answer, not its sources (corpus
    text as retrieved): keep text a caller must not see out of the corpus.
  - Interventions are in `RAGOutput.metadata.guardrail`, the WARNING log and
    the trace. A guardrail API error fails the query, or with
    `processing.ignore_errors` gives the degraded answer with
    `metadata.error`: never unscreened.
- **Not on model calls.** Model calls inside a query run unguarded: attached
  to a call, a guardrail evaluates the whole prompt, so a `PROMPT_ATTACK`
  filter judges retrieved text as user input (false blocks, cost growing
  with context)
  ([prompt attacks](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html)).
  The contextual grounding check is opt-in and runs on the answer's check
  only (below).
- **Ingestion and the evaluation judges run unguarded** (`run-eval`'s answers
  come from the screened query path). A PII guardrail that anonymizes
  `NAME` would rewrite entity names to `{NAME}` (merging distinct people), and
  a `PROMPT_ATTACK` filter can block instruction-like corpus text. On the
  query path, `NAME` anonymization also strips the name a question retrieves
  by, so the baseline guardrail `iac/` creates anonymizes email, phone and
  card numbers but not `NAME` ([iac/README.md](../iac/README.md#guardrail)).
- **Contextual grounding check (opt-in; English, French and Spanish
  only).** `aws.bedrock.guardrail.contextual_grounding.enabled: true` sends
  the context the answer model was given (`grounding_source`) and the
  question it answered (`query`) with the answer's check, so the guardrail's
  [contextual grounding policy](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html)
  scores whether the answer is grounded in that context and relevant to the
  question, and blocks it below its threshold (`iac/`:
  `-c guardrail_contextual_grounding=true`, `guardrail_grounding_threshold`,
  default 0.7). Needs `check_output`.
  - **Language.** The Classic tier (the default, and the one `iac/` creates)
    scores English, French and Spanish only: answers in other languages,
    Korean included, score low and are blocked. The Standard tier covers
    more languages but needs a cross-Region guardrail profile and is offered
    in fewer regions. Measure the threshold on your own questions first.
  - A blocked answer takes the refusal path above, with
    `reason: contextual_grounding` in its `metadata.guardrail` record. Each
    checked answer carries `metadata.grounding` (`checked`, `scores`,
    `failed`, `blocked`) and the `kg_rag.guardrail` span its scores. An answer that
    fails without being blocked (a detect-only policy), and a guardrail that
    returns no scores (no grounding policy), are logged at WARNING.
  - **Limits.** Bedrock takes 100,000 characters of reference, 1,000 of
    question and 5,000 of answer. A longer context keeps its highest-ranked
    sections and a longer question its head. A longer answer is screened by
    the other policies only: `metadata.grounding.skipped_reason` is
    `answer_too_long`, with a WARNING.
  - **Cost.** Billed per text unit (1,000 characters) of reference, question
    and answer, on top of the other policies: a full context is about 100
    text units per answer
    ([pricing](https://aws.amazon.com/bedrock/pricing/)). `ApplyGuardrail`'s
    per-second text-unit quota (Service Quotas, guardrail region) throttles
    concurrent queries with long contexts first.
- **Custom code** calling `setup_chain` or `get_model` for non-query work
  passes `model_purpose=ModelPurpose.INGESTION` (or `EVALUATION`;
  `from unified_kg_rag import ModelPurpose`). Unmarked
  calls default to `QUERY`: the query-time transient-error retry, and neither
  the response cache nor the ingestion effort.

### S3 cache encryption

`aws.s3.encryption.encryption_type: BUCKET_DEFAULT` sends no encryption
header, so the bucket default applies (the CMK with the CDK stack's
`use_cmk=true`; forcing `AES256` bypasses it). On a reused SSE-KMS bucket the
writer needs `kms:GenerateDataKey` and `kms:Decrypt` on its key.

### What the logs contain

At `INFO` and above, user and corpus text is logged only as lengths, counts,
ids and a stable short hash (`query: len=42 sha=1a2b3c4d`). Query text,
rewritten queries, entity names and raw model output are logged only at
`DEBUG`, and exception messages never carry model output. Treat `DEBUG` logs
(`logging.level: DEBUG`, `UKR__LOGGING__LEVEL=DEBUG` or `--verbose`), and
libraries set to `DEBUG` in `logging.library_levels` (`botocore` logs request
bodies), as containing user and corpus data.

With `logging.log_to_file: true` the CLIs also write `logging.log_file_path`
(`logs/log.txt`, dated `log_YYYYMMDD.txt`, relative to the working directory,
which must be writable); it is off by default. As a library the
package configures no handler: the host's logging applies, with the
structured fields (`pipeline_id`, `trace_id`, ...) as record attributes.

### Custom prompts

Every prompt has a `custom_prompts.*_system` / `*_human` override (default
`null` = the built-in prompt in `unified_kg_rag/domain/prompts/`) except
the prompts of the intelligent chunker (`TextChunkingPrompt`), document
translation (`TextTranslationPrompt`), query translation
(`TranslationPrompt`), global search's community rating
(`CommunityRelevancePrompt`) and reduce step (`MapReduceSummaryPrompt`), which
are not overridable. Override what you need and leave the rest `null`
([§9](#9-domain-adaptation)). Overrides are validated at load: unknown
`{variables}` and missing data variables (such as `{input_text}`) are errors,
and a literal brace is written `{{` / `}}`.

### Environment variable overrides

Any config key can be set from the environment as `UKR__` followed by its
path, with double underscores between the segments, in any case:
`UKR__AWS__NEPTUNE__ENDPOINT` sets `aws.neptune.endpoint`, and a segment past
a free-form mapping names one of its keys
(`UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`). A variable wins over the
config file and the defaults. Values are converted and validated as the same
value in the file would be (`true`/`1`/`yes`/`on` or `false`/`0`/`no`/`off`,
in any case, for booleans; JSON for a list or mapping, e.g. `UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES='["local", "mix"]'`).
An invalid value, or a `UKR__` variable that names no key, is a usage error
(exit 2) naming the variable; `--allow-unknown-keys` turns the unknown name
into a WARNING.

- `AWS_PROFILE` and `AWS_REGION` fill `aws.profile_name` and
  `aws.region_name` only when the file leaves them unset, so a shell that
  exports `AWS_REGION` (SSO helpers and CloudShell do) does not move a file's
  region; `AWS_REGION` then also moves Bedrock calls while
  `aws.bedrock.region_name` is `null`. `AWS_DEFAULT_REGION` is not read.
- `UKR_SOURCE_DIRECTORY`, `UKR_PIPELINE_ID` (one underscore: not config
  keys) and `UKR_METRICS_ENVIRONMENT` are read by `run-ingestion`
  ([CLI Reference](./reference/cli.md#run-ingestion)). The variables the CDK tasks get
  are listed in [`iac/README.md`](../iac/README.md#2-build-and-push-the-app-image).
- The CLIs load `./.env` from the current working directory (python-dotenv),
  never one from a parent directory (a `~/.env` setting `AWS_PROFILE` would
  otherwise switch accounts unnoticed); a variable already set wins. Importing
  the package loads no `.env` and never changes `os.environ`: a Python
  caller loads one itself (`dotenv.load_dotenv()` before `get_config()`).

> **LangSmith tracing uploads content.** With `LANGSMITH_TRACING=true` (or
> `LANGCHAIN_TRACING_V2=true`), LangChain sends every traced run to
> LangSmith, prompts, retrieved text and model outputs included. The CLIs log
> a WARNING at startup (from `setup_logging`) but leave it on; unset the
> variable before running on a confidential corpus.

### Observability

Tracing is off by default. With `observability.enabled: true`, every ingestion
run and every query becomes an OpenTelemetry trace that any OTLP-capable
backend (Jaeger, Grafana Tempo, an OpenTelemetry Collector, Langfuse, Phoenix)
can show. It needs the `otel` extra (`uv sync --extra otel`); enabling tracing
without it fails at start-up with a `ConfigurationError` naming the extra, and
with tracing off nothing OpenTelemetry-related is imported.

- `observability.backend`: `none` (the process's global tracer provider,
  which a host application or `opentelemetry-instrument` sets up), `otlp` (an
  OTLP exporter the package installs), or `langfuse` / `phoenix` (that
  exporter pointed at the vendor). The other keys (`capture_content`,
  `redact_patterns`, `sample_rate`, `service_name`, `export_timeout_seconds`)
  are in the [reference](./reference/configuration.md#observability).
- The exporter is configured by the standard OpenTelemetry environment
  variables, never by the config file, so a collector key never lands in it:
  `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`),
  `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TIMEOUT` and
  `OTEL_EXPORTER_OTLP_PROTOCOL` (`http/protobuf`, or `grpc` for port 4317).
- Every span of a query carries `session.id` (`RAGInput.conversation_id`)
  and, when set, `user.id` (`RAGInput.user_id`), which Langfuse and Phoenix
  group traces by; pass an opaque user id, not an e-mail address. Log records
  written inside a span carry its `trace_id` and `span_id`.
- Spans carry no query text, passages or model output unless
  `capture_content: true`.

Model usage is counted with tracing on or off: every chat model the factories
build records each call (the tokens its response reports) for the query or
ingestion stage that made it. `RAGOutput.metadata["usage"]` (and
`SearchOutput.metadata["usage"]`) holds the query's `llm_calls`,
`input_tokens`, `output_tokens`, `cache_hits`, `estimated_calls`,
`cache_read_input_tokens`, `cache_write_input_tokens`, `request_retries`
and `failed_calls`, and the same counts
`by_role` (the model's purpose: `query`, `ingestion`, `evaluation`) and
`by_model` (the model id the factory resolved). `run-ingestion
--output-format json` reports `token_usage` (the stages this run executed,
with `stages.<stage>`), `--metrics-sink cloudwatch` emits it per stage
([runbook](./operations.md#alarms)), and `run-eval` records it per answer
(`EvaluationResult.usage`) and per run (`summary.usage`). A response the
ingestion response cache served, a chunk-boundary or community report
answer the extraction store served, a description summary a full build took
from its stored copy, or a query entity or keyword extraction the chain's
extraction cache served (`search.query_extraction_cache_size`), counts as
a cache hit, not a call; a failed
call is not in `llm_calls` but in `failed_calls`; retries are calls.
`input_tokens` includes the prompt-cache reads and writes, which are also
counted apart (`cache_read_input_tokens`, `cache_write_input_tokens`, from
Bedrock's `cacheReadInputTokens` / `cacheWriteInputTokens`) since they are
billed at other rates. `request_retries` counts the requests botocore resent
inside one call (its `RetryAttempts`): the call counts once in `llm_calls`,
but a request resent after a read timeout or a server error is a Bedrock
invocation of its own (a resent throttled one is not). So Bedrock's
CloudWatch `Invocations` can exceed `llm_calls` by up to `request_retries +
failed_calls`, plus any calls made outside a usage scope (the `run-eval`
warm-up queries). A call that has not answered when the
usage is read (an `auto` query's unread extraction, still running in the
background) or was cancelled in flight (by `search.query_timeout_seconds`,
say) was sent and is billed, so it counts in `llm_calls` and in
`estimated_calls`, at an estimate of its input tokens and no output tokens;
one that answers later is settled in `run-eval`'s run usage. `stream` / `astream` yield text only and report no usage.

The [runbook](./operations.md#observability) has the Jaeger, Langfuse and
Phoenix recipes, the [trace shape](./operations.md#trace-shape) (every span
and attribute, the vendor attribute names and the known limits) and the
export limits; CloudWatch tracing on the CDK stacks is in
[iac/README.md](../iac/README.md#cloudwatch-tracing-optional).

---

## 3. Ingestion (`run-ingestion`)

Ingestion turns a directory of documents into a knowledge graph in the graph
and vector stores. `run-ingestion --source-directory <dir>` runs the 12 stages ([Design §4](./design.md#4-ingestion-pipeline) shows them),
checkpoints each one, and resumes after the last valid stage when re-run with
the same `--pipeline-id`. With the doc-status registry on it is incremental
([§5](#5-incremental-indexing)).

Every flag with its default: [CLI Reference](./reference/cli.md#run-ingestion).

### The 12 pipeline stages

In run order; `--enabled-stages` / `--resume-from-stage` take the **names**
(case-insensitive):

1. **`document_parsing`**: text per format (`.pdf`, `.txt`, `.csv`, `.json`;
   `.md`/`.html` with the `unstructured` extra).
2. **`document_loading`**: the run's corpus (MinHash dedup, incremental
   filter); without `document_parsing`, pre-parsed `Document` `.json` files.
3. **`text_chunking`**: text units (`processing.chunking`).
4. **`translation`** (optional): to `target_language`; a no-op when source
   equals target and there are no extra targets.
5. **`graph_extraction`**: entities and relationships per chunk (LLM).
6. **`gleaning`** (optional): refinement passes (`processing.gleaning`).
7. **`graph_resolution`**: merges duplicate entities and relationships (name
   variants; similar names with `processing.fuzzy_entity_resolution`).
8. **`claim_extraction`** (optional, off by default): factual claims.
9. **`claim_resolution`** (optional): dedupes the claims.
10. **`graph_analysis`**: builds the graph, logs `graph.analysis.statistics`.
11. **`community_detection`**: Leiden clustering and LLM community reports.
12. **`indexing`**: writes to OpenSearch and Neptune (and the registry when
    enabled).

### Examples

```bash
# Full build
run-ingestion --source-directory ./documents --config-path config.yaml

# Keep the stage cache in S3 so a fresh task can resume
run-ingestion --source-directory ./documents --config-path config.yaml \
  --s3-sync --s3-bucket-name your-bucket

# Resume an interrupted run from a stage
run-ingestion --source-directory ./documents --config-path config.yaml \
  --pipeline-id <id> --resume-from-stage graph_extraction

# Run only specific stages
run-ingestion --source-directory ./documents --config-path config.yaml \
  --enabled-stages DOCUMENT_PARSING,TEXT_CHUNKING,GRAPH_EXTRACTION
```

**Resume vs. force-rebuild.** Without `--force-rebuild`, completed stages are
cached and skipped on a re-run:

- `--pipeline-id <id>` resumes that run at the first failed or incomplete
  stage (within `--enabled-stages`); a stage switched on since (gleaning,
  claims) counts as incomplete. Without `--pipeline-id` a run gets a new id.
- If the corpus or config changed, the CLI lists the completed stages whose
  output is no longer cached and asks before recomputing them (`--yes`
  without a terminal).
- With incremental indexing, a resume whose extraction settings changed since
  its delta was computed restarts at `document_loading`, so the delta follows
  the new settings.
- `--force-rebuild` recomputes every stage and rewrites the cache; it also
  asks the model for every pre-chunk's boundaries, description summary and
  community report again and embeds every text again (see below).

**S3 sync** (`--s3-sync`) keeps the stage cache in
`s3://<bucket>/<prefix>/...`, uploaded after every stage, so a fresh task
resumes there. On SIGTERM (ECS `StopTask`, a Step Functions timeout) the run
uploads the running stage's checkpoint and exits 143. A failed upload between
stages is retried by the next one; a failed initial download or final upload
fails the run with `CacheSyncError`. Expiry and the other caches:
[Caches and prefixes](./operations.md#caches-and-prefixes). Across runs,
the doc-status registry (`providers.doc_status.enabled`) keeps unchanged
documents from being embedded again: an incremental run embeds only what its
delta changes. Within what a run writes, a text embedded before with the same
embedding provider, model and dimension keeps its stored vector (each vector
stores a digest of its text), so a changed document re-embeds only the
chunks, descriptions and reports whose text changed, and a reset of an
unchanged corpus embeds again only what it regenerates. The intelligent
chunker's boundary answers are kept per pre-chunk in the extraction store,
so a reset or an edit asks the model only for pre-chunks whose text or
chunking settings changed, and an unchanged pre-chunk keeps its boundaries
even with a model that answers otherwise each time. Only an answer the
chunker used is kept: a pre-chunk split by size instead (the answer named
lines that do not exist) asks the model again on the next run. A full build
(a reset, a first run, or a run without the registry) does the same for
the corpus-wide answers: a description over the summarization threshold
whose stored copy was summarized from the same text with the same
summarization settings keeps that summary, and a community whose report
input (its members, descriptions, edges and claims as packed, its roll-up
text and name) and report settings are unchanged takes the stored answer
(the extraction store keeps it, and the report names it in
`attributes.report_record`). A community's name (`Level 0 Community
1a2b3c4d`) derives from its members, so a document that adds a community
leaves the others' names, and their stored answers, as they were.

**LLM response cache** (`cache.llm_responses`, on by default). Every
ingestion model response is kept under `llm-cache/` in the run's cache
directory (synced with it), so a re-run with the same pipeline id repays only
the calls a stopped or crashed stage had not made. The key is the rendered
prompt plus the model id, max tokens, temperature and reasoning.

- Only earlier runs' responses are served, each at most once per run; a
  response cut off at the output-token limit is not kept. Above temperature 0
  a re-run replays the earlier sample. The responses of a call that failed
  every attempt (one that never parsed), or whose graph or claims record
  failed to build, are dropped, so a re-run asks the model again for those
  units only; the late answer of a call abandoned on its timeout is not kept
  either.
- `--force-rebuild` serves nothing; query and evaluation calls are never
  cached.
- The cache grows by a few KB per model call and is read into memory at
  start; it is written in segments of `cache.llm_responses.flush_every`
  responses, so a killed task loses at most one segment. A stop also ends
  the retries of the model calls still running.

### Index suffix

Every index and graph label of a corpus is named `<prefix>-<suffix>`
(plus `-<additional_suffix>` with `indexing.additional_suffix`); `<suffix>` is
the **index suffix**. Ingestion takes it from
`processing.document_parsing.index_value` (no flag); `run-rag` and `run-eval`
query the same `index_value` of their config unless `--suffix` names another,
and in Python `RAGInput.suffix` selects it (the chain does not read
`index_value`). All default to `default`. Use one index suffix, and one config file, per
tenant or corpus version, with the same `indexing.additional_suffix` on both
sides; incremental indexing and deletion are scoped to it (§5).

The suffix separates names, not access: whoever can query can name any
suffix. A multi-tenant service must derive it from the authenticated caller
on the server; see
[Operator Runbook: Query access and tenant isolation](./operations.md#query-access-and-tenant-isolation).

### Multilingual ingestion

Set the corpus's predominant language and the target you want to index in:

```yaml
processing:
  translation:
    enabled: true
    source_language: "ko"
    target_language: "en"
    additional_target_languages: ["ja"]   # index additional languages too
```

With `source_language == target_language` and no additional targets, the
translation stage is skipped at no cost, even with `enabled: true`.

- **Analyzers** come from `indexing.opensearch.language_analyzers` (e.g.
  `ko: nori`; others use `default_analyzer`). Chunk `text` is analyzed for
  `source_language`, so set it to the corpus language even without
  translation: with `source_language: en` a Korean corpus keeps words with
  their particles ("홍길동으로부터") as single tokens. Changing it changes the
  mapping: re-index the text units.
- **Unicode.** Text and queries are NFC-normalized, so decomposed Hangul (NFD)
  matches its composed form.
- **Chunk sizes.** The chunk sizes count characters by default, and a
  Hangul, Han or Kana character is about one token, so an 8,000-character
  Korean chunk is ~6,000 tokens: over the 4,096-token document limit of Cohere
  Rerank 3.5, which then never sees its tail. Halve the sizes, or count them
  in tokens with `processing.chunking.length_unit: tokens` (a local estimate,
  no model call) and give every size in tokens:

  ```yaml
  processing:
    chunking:
      length_unit: tokens
      min_chunk_size: 250
      max_chunk_size: 2000
      chunk_overlap: 125
      fallback_chunk_size: 1200
      pre_chunk_size: 12500
      pre_chunk_overlap: 125
  ```

  Switching the unit re-chunks the corpus and re-extracts its documents (an
  incremental run re-extracts every document of the scope).
- **The analyzer must exist on the cluster**, or the indexing stage fails
  with `the cluster has no analyzer '<name>'`. `nori` comes from the
  `analysis-nori` plugin: the
  [local OpenSearch container](#local-stores-in-containers) has it, an Amazon
  OpenSearch Service domain only once its Nori package is associated (a
  blue/green deployment). The
  [CDK deployment](../iac/README.md) therefore maps `ko` to the built-in `cjk`
  analyzer (character bigrams); `-c opensearch_nori_package=true`
  associates the package and maps `ko` to `nori`
  ([Korean with Nori](./operations.md#korean-with-nori)).
- **Failures.** A text unit whose translation fails is extracted in its
  source language; a failed call for a whole target language fails the stage
  unless `processing.ignore_errors: true`. An incremental run records the
  documents of the `failed_units` as failed, so the next run retries them.

---

## 4. Querying (`run-rag`)

`run-rag --query "..."` answers one question, `--interactive` holds a
conversation. Pick a strategy with `--search-strategy` (default `auto`, an LLM
router); every answer returns the sources the answer model actually saw.
[Design §6](./design.md#6-retrieval-two-methodologies) shows how each
strategy retrieves.

Every flag with its default: [CLI Reference](./reference/cli.md#run-rag).

### Search strategies — when to use which

GraphRAG strategies excel at *summarization and thematic synthesis* over a
corpus; LightRAG strategies are faster and lean on *dual-level keyword*
retrieval, good for keyword-driven lookups and as a low-cost baseline. Both
share the hybrid scorer (BM25, vector, graph, RRF, rerank).

**GraphRAG (community-summary):**

| Strategy | Use when | How it works |
|---|---|---|
| `simple` | Fast factual lookups | OpenSearch vector + keyword retrieval, no graph traversal; the claims index too when claims are on. Fastest. |
| `local` | Questions about specific entities | Query entities → graph traversal to neighbors and relationships, with vector/keyword hits; claims when enabled. Each query entity is looked up on its own: the entity of its name seeds the traversal, else its top hit if it scores at least `search.local_search.seed_score_ratio` (0.6) of the best one (on a hybrid search rescaled to the weights, so a hit only the vector search found never seeds beside a name match; a vector search's raw similarities are closer, so the floor filters less there). Community reports: the most similar to the query entities by default; `search.local_search.report_selection: membership` reads those of the communities the traversed entities belong to instead (MS GraphRAG local), usually fewer of them; it answered measurably worse, so similarity is the default ([evidence](./concepts/defaults-and-evidence.md#local-report-selection)). |
| `global` | Broad, thematic questions ("what are the main themes") | Maps the community reports of one cut of the hierarchy (`search.global_search.community_depth`, default 2; a cut over `max_reports_per_level` falls back to the deepest shallower cut that fits, or under a report-key filter to its most similar reports) into scored key points on the fast model, one call per `map_context_tokens` of reports, and answers from the ranked points. Dynamic selection walks down from the roots instead. |
| `drift` | Multi-part questions needing the broad picture and specific facts | Reports similar to a hypothetical answer (HyDE) draft intermediate answers and follow-ups; each follow-up runs a local search and raises more, for `n_depth` rounds; the answer model reduces the results. The most LLM calls (about `primer_folds + k_followups * n_depth`). |
| `auto` | General use (the default) | An LLM router (`models.roles.strategy_selection`) picks among `search.auto_routable_strategies` (default local, mix, global; drift, at up to 27 LLM calls a query, only when added). The entity and keyword extractions the candidates read, and the query translation, run alongside the router (`search.auto_speculative_extraction`), so `auto` routed to local or mix takes as long as local or mix asked for directly; the extraction the route does not read keeps running in the background and is billed (one fast-model call a query with the defaults, two when routed to global): the query does not wait for it, counts it in its usage from its start (`estimated_calls` while it runs), and it fills the extraction cache; it is cancelled at `search.query_timeout_seconds` or when the chain closes. |

**LightRAG (dual-level keyword):**

| Strategy | Use when | How it works |
|---|---|---|
| `mix` | General LightRAG use | Low-level keywords → entity index, high-level keywords → relationship index, one-hop expansion to incident relationships and endpoint entities (Neptune multi-hop is opt-in: `search.lightrag_search.enable_graph_expansion`), **plus** naive chunk retrieval blended in. |
| `hybrid` | Keyword-driven graph questions | `mix` without the naive chunk blend. |
| `naive` | Fast baseline, comparison eval | Chunk retrieval only (`search.lightrag_search.chunk_stream_top_k`, default 20). The default `--search-type hybrid` adds BM25; `--search-type vector` approximates upstream LightRAG's pure-vector baseline, and MS GraphRAG's `basic` search. |

`mix` and `hybrid` read the relationships index ingestion builds; a short
query that yields no keywords is used as a low-level keyword itself (up to
`search.lightrag_search.raw_query_fallback_max_len`).

Both add the source chunks the matched entities cite and, as a stream of its
own, those the matched relationships cite (each up to
`search.lightrag_search.chunk_stream_top_k`; fusion interleaves them and the
merged chunks are cut to that width after the rerank).
`search.lightrag_search.kg_chunk_pick_method` decides which of an item's
chunks are taken: `weight` (the default) gives each matched item
`related_chunk_number` chunks on a decreasing gradient; `vector` takes the
cited chunks that best match the query, one search a side, as upstream
LightRAG does by default. `weight` stays the default, and MMR's
`search.fusion.diversity_lambda` keeps 0.5: the alternatives measured no
significant difference ([evidence](./concepts/defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity)).

### Answer style

`global` answers with the comprehensive prompt by default, every other
strategy with the concise one. The concise prompt writes a one-sentence
direct answer, then brief support (`custom_prompts.answer_generation_*`);
the comprehensive one a multi-paragraph markdown answer that explains every
key point, modelled on MS GraphRAG's global reduce prompt
(`custom_prompts.comprehensive_answer_*`). `search.answer_prompt_style` is
merged over that default, so naming one strategy leaves the others as they
are:

```yaml
search:
  answer_prompt_style: {drift: comprehensive}   # global stays comprehensive
  query_timeout_seconds: 300                    # see below
# answer_prompt_style: {global: concise}        # concise global answers
```

Why. Comprehensive global answers cover more than concise ones but are less
direct, and pairwise LLM judges favour longer answers: a user who wants a
short answer should keep `{global: concise}`. `drift` stays concise by
default because the comprehensive answer brings its latency close to the
180 s `search.query_timeout_seconds`. With `{drift: comprehensive}`, raise
that deadline (300 s leaves room;
[runbook](./operations.md#query-deadlines-and-model-timeouts)). The
measurements: [Defaults and evidence](./concepts/defaults-and-evidence.md#answer-style).

`auto` uses the style of the strategy it routes to. `answer_match` scores
read the first line that is not a heading (`evaluation.answer_span`), so a
comprehensive answer scores lower on exact match whatever its quality.

Global search hands the answer model its ranked key points followed by the
reports they cite. `search.global_search.evidence_tokens` caps that report
text (whole reports, best point first) at 4000 tokens by default, which
cut the cost of a global query by about 40% against no cap with no measured
quality difference, and the answer keeps its sources
([evidence](./concepts/defaults-and-evidence.md#global-evidence-cap)). `null` (no cap) keeps as much as the
context budget holds, which on a large cut is most of a global query's input
tokens; `0` (cheaper still) sends the key points only, as MS GraphRAG's reduce step reads
them, and the answer then reports no sources (the key points are model
output), so the `retrieval` evaluator skips its rank metrics for a global
query (`no_sources_by_config`) instead of scoring a miss.

### Examples

```bash
# Single query (auto strategy, hybrid search)
run-rag --query "What are the main themes in the documents?" --config-path config.yaml

# Pick a strategy and search type
run-rag --query "How does entity X relate to Y?" \
  --search-strategy local --search-type hybrid --config-path config.yaml

# Retrieval only, JSON output
run-rag --query "..." --mode search --output-format json --config-path config.yaml

# Verbose (entities, top sources, metrics) with attribute filters
run-rag --query "..." --verbose --filters attr_category:research type:person \
  --config-path config.yaml
```

### Rerank status

Every query reports what reranking did in `metadata.rerank` of the
`SearchResult` and the RAG output (and `--output-format json`): `status`,
`reason`, and `stages`, one entry per rerank step with its `status`, `scope`
(`all`, or the reranked result types, e.g. `text` for `local`, `drift`, `mix`
and `hybrid`) and `reason`.

| Status | Meaning |
|---|---|
| `applied` | The rerank model reordered the candidates. |
| `disabled` | `search.reranking.enabled: false` or `providers.rerank.kind: none`. |
| `unavailable` | The rerank model could not be built (initialization failed, logged at ERROR, or the cross-encoder's `[local]` extra is missing). |
| `skipped` | Nothing to rerank: no candidates, none of a rerank-eligible type, or no query text. |
| `failed` | The call raised (denied `bedrock:Rerank`, a region without the model, an unreachable endpoint) or returned nothing usable; the fused order was kept. |

- A `failed` or `unavailable` reason is the AWS error code or the exception
  type, without the message.
- Built-in strategies rerank once per query. A caller that fuses several
  times into one trace gets one stage per fusion; the top-level `status` is
  the worst, in the order `failed` > `unavailable` > `applied` > `skipped` >
  `disabled`. `metadata.reranked` marks each reordered result.
- A Bedrock permission or region problem shows as `failed` at the first
  query, not `unavailable`. `search.reranking.required: true` fails such
  queries with `RerankModelError` instead of answering from the fused order.

### Query deadline

A query (`run-rag`, `invoke`/`ainvoke`, `search`/`asearch`, a stream) may take
`search.query_timeout_seconds` (default 180) in all. Past it the query fails
with `QueryTimeoutError` (exported from `unified_kg_rag`; also a builtin
`TimeoutError`) and its model and store calls in flight are cancelled:

- `run-rag` prints the message and exits `1`; `run-eval` counts the query as
  failed (`error_message` set, not scored) and goes on with the others.
- With `processing.ignore_errors` the query returns the error answer instead,
  with the message in `search_results.metadata.error`.
- A streamed answer must also send its first chunk within
  `search.model_timeout_seconds` of its model call; once chunks flow, the
  overall deadline applies.
- `null` turns both deadlines off.

The query's model calls (answer generation and the other query-time LLM
calls, query embeddings, Bedrock rerank) use their own clients:
`search.model_timeout_seconds` (default 90) for each request's answer and
`search.model_max_attempts` (default 2) botocore attempts, so a stalled
Bedrock call gives up after about 3 minutes. Ingestion and evaluation-judge
calls keep the 330 s read timeout and 6 attempts. On an OpenAI-compatible or
Ollama server the query calls' request timeout is
`providers.llm.timeout_seconds` / `providers.embedding.timeout_seconds`,
capped at `search.model_timeout_seconds`; `backend: local` presets 300 s
model and 900 s query timeouts, since a local server queues requests (a
`null` there is replaced by the preset: set a large number instead).

A non-streamed answer arrives in one piece, so `search.model_timeout_seconds`
also caps how long the model may generate it. Raise it (and the query
timeout) for long answers or high reasoning effort. The defaults come from
measured query times on real AWS ([evidence](./concepts/defaults-and-evidence.md#query-deadlines-and-model-timeouts)):
90 s covers the slowest whole query in one request with room to spare, and
180 s lets a stalled first attempt be retried once. Tuning is in the
[Operator Runbook](./operations.md#query-deadlines-and-model-timeouts).

### Filters

`--filters key:value ...` (or `RAGInput.filters`) narrows what retrieval ranks;
a pair without a key, a `:` or a value, a key given twice and a key no index the
strategy reads declares are usage errors (exit 2).
Filters compile to `term`/`terms`/`range` clauses on OpenSearch and `has`
steps on Neptune. Each store accepts the fields its indexer writes:

| Store | Filterable fields |
|---|---|
| Text units | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.filters.<key>` |
| Entities | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `in_schema`, `type_iri`, `iri`, `source_kinds`, `attr_<key>`, `attributes.<path>` |
| Relationships | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids`, `in_schema`, `predicate_iri`, `source_kinds`, `attr_<key>`, `attributes.filters.<key>` |
| Claims | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `start_date`, `end_date`, `description`, `source_text`, `text_unit_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Community reports | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `level`, `depth`, `max_cut_depth`, `parent_community_id`, `child_community_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Neptune entity vertices | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `in_schema`, `type_iri`, `iri`, `source_kinds`, `attr_<key>` (where present) |

- On OpenSearch, `attr_<key>` is entry `<key>` of a document's `filters`
  metadata (an item from several documents carries all their values); on
  Neptune entity vertices it is the entity's own attribute (`attr_role`).
- Entity and claim `type` and claim `status` are stored lowercase and `term`
  matches exactly: filter `type:person`, not `type:PERSON`. Relationships
  take no `type` filter (their type is not stored on the relationships
  index). `in_schema`, `type_iri` and `predicate_iri` are the
  [extraction schema](#c-extraction-schema) annotations; `iri` and
  `source_kinds` (`document`, `seed_graph`) come from
  [seed graphs](#d-seed-graphs-bring-your-own-knowledge-graph). Prefer keyword or
  numeric fields; a `term` on an analyzed field such as `description` matches
  single lowercase tokens. A range filter takes `{"gte": ..., "lte": ...}`
  through the Python API. `attr_<key>`, `attributes.filters.<key>` and the
  claim dates are keywords, a date-like value too, so a range on them
  compares strings, which orders ISO dates (`YYYY-MM-DD`) correctly. A claim
  date that reads as a calendar date (`2024-01-15`, `2024-1-5`,
  `January 15th, 2024`, `Sept. 3, 2024`) is stored as `YYYY-MM-DD`, and a
  coarser period as its first day in `start_date` and as its last day in
  `end_date` (as the claim prompt asks: `YYYY-12-31` for the end of a
  year): an ISO week (`2024-W03`) as its Monday or Sunday (an ISO week
  with its weekday, `2024-W03-2`, as that day), a quarter (`Q3 2022`) as
  2022-07-01 or 2022-09-30, a month (`March 2024`, `2024-03`) as
  2024-03-01 or 2024-03-31 and a year (`2024`) as 2024-01-01 or
  2024-12-31. So a `start_date` of `{"gte": "2022-07-01", "lte":
  "2022-09-30"}` matches `Q3 2022` but `{"gte": "2022-08-01"}` does not,
  and an `end_date` of `{"lte": "2024-06-30"}` does not match "until
  2024". Any other value (`10/08/2024`, whose day and month order is
  unknown, or `early 2024`) is stored as written and compares as text, so
  only an exact filter matches it meaningfully: a range can match it
  falsely (`10/08/2024` sorts before every ISO date, so `{"lte":
  "2000-01-01"}` matches it). An unknown date (`UNKNOWN`, `TBD`,
  `present`) is not stored, so it matches no filter, a range included.
- Boolean and numeric fields take their type's values: `in_schema` a
  boolean; `rank`, `confidence`, `weight` and `rating` a number; `n_tokens`,
  `level`, `depth` and `max_cut_depth` an integer. A string for one of them
  (every `--filters` and `set-filter` value is one, and a dataset or
  `RAGInput.filters` may hold one) is converted before the search, in a list
  and in range bounds too, so every store matches it as it matches the typed
  value: `in_schema:false` is `False` (a boolean takes `true`/`false`,
  `1`/`0` or `yes`/`no`, in any case) and `rank:2` is `2`. A string its
  type cannot take (`in_schema:maybe`, `rank:high`, `level:1.5`) raises
  `InvalidFilterError` (a usage error, exit 2, on the command line), and so
  does a typed value of another type: a number for a boolean field
  (`in_schema: 1`), a boolean for a numeric one (`rank: true`) or a
  fractional number for an integer one (`level: 2.5`; `2.0` is `2`). The
  values of every other field are matched as given, `attr_<key>` and
  `attributes.<path>` included.
- A filter applies only to the stores that declare its field
  (`type:person` leaves text units unfiltered). A key no store the
  selected strategy reads declares (e.g. `category`, or `type` with `naive`
  or `global`) raises `InvalidFilterError`, listing the filterable keys
  (`unified_kg_rag/adapters/storage/filter_schema.py`). `naive` reads text
  units; `hybrid` and `mix` entities, relationships and text units (and
  Neptune with `search.lightrag_search.enable_graph_expansion`); `local` and
  `drift` every index the pipeline builds and Neptune; `global` community
  reports (and text units with `search.global_search.text_unit_slots`,
  scoped to the selected reports' communities); `simple` every index the
  pipeline builds. `global` refuses the keys it scopes its own searches by:
  `depth` and `max_cut_depth` (with
  `search.global_search.use_dynamic_selection`, `depth` and
  `community_id`), and `community_ids` with `text_unit_slots`. `id`,
  `source_id` and `target_id` narrow what `local`, `drift`, `mix` and
  `hybrid` fetch by id or walk to (only artifacts with a listed value come
  back); a range on them is refused with every strategy (`naive` and
  `global` too, which do not scope them, so a filter is valid whatever the
  strategy), as is an empty list of values on any key. `auto` routes
  only among the strategies it may route to that apply every key (with the
  default set, `type:person` leaves `local` and `mix`) and refuses the
  filters, before routing, only when none does. Neptune filters entity
  vertices only.
- OpenSearch applies `attr_<key>` and `attributes.<path>` strictly
  (fail-closed): an item without the attribute is excluded. Relationships,
  claims and community reports carry the filter values of their text units'
  documents, so each matches when one of them does. A vector or hybrid search
  skips an index none of whose documents carry the attribute (OpenSearch 2.13
  rejects a kNN filter on a field the index does not map).
- A document's removal clears its filter values from the entities,
  relationships and claims that outlive it: a key no remaining document
  gives is stored as null, which matches no filter.
- `attr_<key>` and `attributes.filters.<key>` name document attributes. A
  graph store holds none, so graph expansion keeps an expanded entity (the
  seeds aside) only when its entities-index document, which carries the
  filters of every document the entity comes from, passes them; under two or
  more keys, one of its text units (one document's) must also pass them all.
  It must also be reached from a seed, within `max_hops`, over relationships
  that pass the filter (an edge only excluded documents state is not
  walked), and its `Path:` follows those relationships: the walk runs over
  the relationships index, one hop at a time, so it is cut after these
  checks and it passes through entities a vertex filter (`type`) leaves out
  of the results. Without that index
  (`indexing.opensearch.build_relationship_vector_index: false`, or the
  index missing at query time, logged as a WARNING) the graph is walked and
  the entities-index check alone decides.
  Under a filter on text units, local and mix also fill their chunk quota
  with the entity-linked chunks that pass, not the top-ranked ones only.
  Other
  `attributes.<path>` keys name an item's own attributes and apply to the
  entities index only; a graph walk's entities, LightRAG's seeds included
  (`local`'s and `drift`'s were found under the filters), are checked
  against that index for them and for `name.keyword`, which entity
  vertices do not store. The check sees what the walk kept after its own
  per-hop and `top_k` cuts, so a passing neighbour ranked below them is
  not reached. On Neptune, `attr_<key>` is also an entity vertex
  property (`attr_role`), applied where present (a vertex without it passes),
  but the entities-index check above still applies: `attr_role:buyer` selects
  nothing unless documents carry a `role` filter. Every other key is strict
  on both stores.

**Filters are not an access-control boundary**: a store that does not declare
a key returns its content unfiltered, and an entity, relationship or
community report drawn from several documents passes when one of them
matches, so it can carry excluded documents' content. An item from several
documents also matches each key of a multi-key filter independently: the
seeds, relationships, claims and community reports carry each key's values of
all their documents, so `attr_category:x attr_grp:b` matches one whose
documents are `{category: x, grp: a}` and `{category: y, grp: b}`, although
no document is both. Separate tenants or users by
[index suffix](#index-suffix), chosen server-side.

### Interactive mode & conversation memory

```bash
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

Interactive mode turns memory on. Commands: `help`, `new` (a fresh
conversation id), `set-filter key:value`, `clear-filters`, `show-config`,
`quit` / `exit`. A command is the whole line (only `set-filter` takes an
argument): `help me find ...` is a question. `set-filter` refuses a key no
index of the search strategy declares, as `--filters` does (with
`--interactive`, before the first question: exit code 2). With `auto` it
refuses a key only when, with the filters already set, no strategy `auto`
may route to applies them all.

Conversation memory lives in the process (`MemoryManager`, limits in the
`memory` section):

- It carries over between the turns of one session, or between calls of one
  `GraphRAGChain` reusing a `conversation_id` with `use_memory=True` and the
  same `suffix` and `user_id` (other tenants' or users' turns never reach
  it). Chains share it only through one `MemoryManager(config)` passed as
  `memory_manager=`.
- Separate `run-rag --query` invocations start with no history, so
  `--use-memory` and `--conversation-id` do nothing there (`run-rag` logs a
  WARNING); use `--interactive` for multi-turn from the CLI.
- A follow-up is rewritten from the history as a standalone question before
  retrieval (one `models.roles.query_rewrite` call per turn with history;
  prompt key `query_rewrite`), so "what about its suppliers?" names the
  entity from the earlier turn. The rewrite reads the last
  `memory.max_messages_per_conversation` messages, each earlier answer cut to
  1,000 characters, and runs even with `enable_query_processing` off (that
  flag covers translation and entity/keyword extraction). Memory adds no
  other LLM call.

---

## 5. Incremental indexing

Incremental (delta) indexing re-indexes only documents that are **new or
changed** since the last run, and merges them into the live graph — instead of
rebuilding everything.

### Enable it

```yaml
providers:
  doc_status:
    enabled: true          # kind: dynamodb (the default) or sqlite
aws:
  dynamodb:
    table_name: "unified-kg-rag-on-aws-doc-status-v2"
    create_table_if_missing: true
```

With this on, each `run-ingestion` diffs the corpus against the DynamoDB
document-status registry by **content hash**. In local mode the registry is a
SQLite file instead (`providers.doc_status.kind: sqlite`; see
[Local mode](#local-mode-no-aws)).

### Workflows

- **Add a document:** drop it into the source directory and re-run
  `run-ingestion`. Only the new file is parsed, extracted and indexed; its
  entities and relationships merge into the existing graph.
- **Modify a document:** edit it and re-run. Its old artifacts are removed and
  the new version is indexed. Only its changed chunks go to the model: the
  extraction store (`indexing.opensearch.extraction_records_index_prefix`)
  serves every chunk whose text and extraction settings it already holds a
  record for, in any document (`--force-rebuild` re-extracts them all).
- **Delete a document:** remove it and re-run. What only its chunks stated
  (entities, relationships, claims) is deleted; shared artifacts are kept
  without its chunks, and a community or report goes once none of its chunks
  is left.
- **Change the extraction settings** (a tuned `custom_prompts` set,
  `entity_types`, an extraction, gleaning or chunking model, chunk sizes,
  translation, resolution): the next run re-extracts the unchanged documents
  extracted with other settings and logs a WARNING with their count and the
  settings that changed. Settings are compared field by field, so a setting a
  newer release adds is no change while it holds its default. A re-extracted
  chunk still takes its stored record when the settings that shape a record
  (models, extraction, gleaning and claim settings, the target language, the
  prompt templates) are unchanged, e.g. after a resolution or chunking change.
  Edits to the built-in prompt templates (code) are not seen: reset after
  editing one. A description summary setting (`description_summarization`,
  its `custom_prompts` or `models.roles.description_summary`) is not an
  extraction setting: changing it re-extracts nothing. The descriptions a
  run merges are summarized the new way, and `indexing.reset: true`
  summarizes every one again (a stored summary records the settings it was
  made with and is reused only under the same ones).
- **Change the embedding model or its dimension**
  (`models.embedding.model_id` / `dimension`): each OpenSearch index records
  the model and dimension it was built with, and an incremental run stops
  before its first model call (exit 2) when they differ from the run's (an
  index that records none is refused too). Re-run once with
  `indexing.reset: true`. `run-rag` and
  `run-eval` with another model or dimension fail with a `ConfigurationError`
  instead of answering from empty search results.
- **Change the corpus or translation languages** (`processing.translation`
  languages, `indexing.opensearch.language_analyzers`): live text fields keep
  their analyzers, so an incremental run stops before its first model call
  (exit 2); re-run once with `indexing.reset: true`. Dropping a language does
  not stop it.
- **Index a suffix with the registry off** (as the `backend: local` preset
  does): a later incremental run on that suffix stops before its first model
  call.
  Re-run once with `indexing.reset: true` and the registry on, or keep it off
  for that suffix.
- **Rebuild** with `indexing.reset: true`: the whole corpus is indexed again
  without taking the suffix offline, except quarantined documents, which are
  skipped and keep their records; a failed reset drops nothing, so it
  is safe to re-run (how it works:
  [Operator Runbook: Re-ingest](./operations.md#re-ingest)). The stored
  chunk boundaries, extraction records and vectors of unchanged text are
  reused, and so are the stored description summaries and community report
  answers whose input is unchanged, so a rebuild with the same models and
  settings asks the model only for what the corpus changed. A rebuild after
  incremental runs still asks for the reports of the communities it
  clusters differently (an incremental run clusters only its delta); after
  a summarization or report setting change it asks again for every summary
  or report, and after an embedding model or dimension change every text
  is embedded again. Add `--force-rebuild` to ask and embed everything
  again.
- **When to rebuild:** an incremental run clusters only its delta and
  appends its communities. After 20 incremental runs since the last full
  build, or once the suffix holds 1.5 times the communities that build
  clustered, the run logs a WARNING (`Communities of '<suffix>' are due for
  re-clustering: ...`) that recommends `indexing.reset: true`; nothing is
  reset automatically. Set the thresholds in
  `indexing.community_compaction` (`null` turns one off); the counters are
  in the build record, and the `community_growth_ratio` and
  `runs_since_clustering` metrics
  ([Operator Runbook: Incremental runs](./operations.md#incremental-runs)).
  A suffix whose build record holds no counters starts counting at its next
  full build.

### Deletion scope

A run's **scope** is its index suffix (`document_parsing.index_value`, plus
`indexing.additional_suffix`) and its corpus source
(`document_parsing.source_scope`, by default the resolved source directory;
the container entrypoint sets it to the S3 URI it syncs from). A document's
registry key is its scope and its path relative to the source directory, and a
run only diffs, and treats as deleted, the documents of its own scope. So
another tenant's run (another `index_value`) never deletes this tenant's
documents, a subfolder run as its own source directory never deletes the rest
of the corpus (its files register as separate documents: do not index the
same files from two roots into one suffix), and two sources indexed into one
suffix stay separate even for the same relative path. Document and text-unit
ids derive from the scope too, so set `source_scope` for ids that are
reproducible across paths and machines (the same corpus indexed from another
directory or host otherwise gets other ids). A file that fails to
parse or load, or that the run skips because no loader reads its type (an
optional extra such as `unstructured` not installed), is reported as
`failed` and keeps its indexed content until a run reads it again.

**Mass deletions.** A run that would delete more than
`indexing.max_deletion_share` (default `0.5`) of the documents its scope
holds stops before it writes and asks (`This run deletes N of the M
documents ...; continue?`): a partial sync or a missing volume looks like
files removed on purpose. `run-ingestion` asks on a terminal;
`--confirm-deletions` (or `UKR_CONFIRM_DELETIONS=1`) answers yes to this
question only, `--yes` to every question, and a no, or no terminal, stops
the run (exit 1). The share counts documents only: seed graph shards
follow `processing.seed_graphs`, so removing or resharding a seed graph
never asks, and its shards do not dilute the share. Scopes of fewer than 10
documents, a retired scope's records, a reset and a run with the registry
off are not checked; `1.0` or `null` turns the check off. From Python, pass
`DataIngestionPipeline(..., confirm_deletions=callable)`; without it, or on
a no, `run` raises `ConfirmationDeclinedError`.

**Moving a corpus** with the default scope changes its scope, and no run
deletes another scope's documents (a directory missing here may be another
host's corpus), so the old content stays indexed. Either:

- set `document_parsing.source_scope` to a stable name before moving it; or
- run once from the new location with
  `--retire-source-scope <old source directory>` (or
  `indexing.retire_source_scopes`). That run treats every record of the old
  scope in its suffix as deleted and indexes the new location as new
  documents. Give the scope as the registry stores it: the absolute
  directory (trailing and repeated slashes ignored) or the former
  `source_scope`, `s3://` URIs included; a non-URI value is also tried as a
  path resolved like `--source-directory`. A bare value (`old`) is a name or a
  relative path: the reading that has records is retired, and when both do
  the run stops before any write (give `./old` or an absolute path). Naming
  the run's own scope fails before the registry is read, a bare value one of
  whose readings is the run's own included: when the own scope is its name,
  give the directory as `./<value>`; when it is the run's own directory, the
  name could be another corpus's `source_scope`, so retire that name from a
  working directory where the value is not the run's directory, with the
  same configuration (the same stores and registry, e.g. the same
  `--config-path`). A scope without records logs a WARNING.

A local run with the default scope and new documents logs a WARNING naming the
other local source directories with records in its suffix; it deletes
nothing. Set `source_scope` per corpus to silence it when they are separate
corpora.

The registry is partitioned by index namespace (the `<namespace>#` prefix of
each document key), so a run reads only its own namespace's records. A table
of the earlier `doc_id`-only key schema is rejected at first use: switch to a
new table and run once with `indexing.reset: true`
([Operator Runbook: Incremental runs](./operations.md#incremental-runs)).

### Failed documents

A document is recorded `FAILED` when one of its text units failed in
translation, extraction, gleaning or claim extraction, when a base-level
community covering it got no report, when an artifact stage crashed in a
`--continue-on-error` run, when writing one of its artifacts failed, or when
the extraction record of one of its text units could not be stored. The next
run processes it again, even if the file did not change, until
`indexing.max_document_failures` (default `3`) consecutive failures of the
same content; editing the file or raising the limit retries it after that.
Claims are optional: when only claim extraction failed, for an error of the
unit's own, the document is indexed without those claims, counted for none,
and the next run extracts them once more.
See [Operator Runbook: Failed documents](./operations.md#failed-documents).

### Quarantined documents

A document whose processing kills the task (out of memory, a crashing parser)
would stop every later run the same way. After
`indexing.max_document_attempts` (default `3`) runs in a row died processing
its unchanged content, the last of them with the document alone, the next run
removes what the attempts wrote, records it `QUARANTINED`, logs a WARNING
naming the file and indexes the rest; it is skipped, by a reset too, until its
file changes or the limit is raised.

- Only deaths count: a run that finishes, fails or is stopped (SIGTERM,
  Ctrl-C) clears its in-progress mark, and a full rebuild counts no attempt.
- When runs died processing several documents, the next runs halve the
  suspects until the culprit is alone, deferring the other new and changed
  documents meanwhile (`total_documents_deferred`).

See [Operator Runbook: Quarantined documents](./operations.md#quarantined-documents).

### Concurrent runs

With the registry, a run holds its index namespace's write lease from its
first stage to its end; a second run on the same namespace stops with
`NamespaceLeaseError` before it reads the stores or writes anything. See
[Operator Runbook: Concurrent runs](./operations.md#concurrent-runs).

### Document size limit

The DynamoDB registry stores a document's text-unit (chunk) ids in one item,
limited to 400 KB (roughly 10,000 ids, about half that for a document you
edit, whose in-progress record holds old and new ids). A document over it
fails the indexing stage, naming the file, before anything is written: split
it. The SQLite registry of local mode has no such limit.

### Cross-run merge

A delta run reads the entities, relationships and claims it touches back from
the stores and merges the delta into them, so an entity shared with unchanged
documents keeps their descriptions and chunk lineage (which `mix` follows);
merged descriptions over the summarization budget are re-summarized. When a
document changes or goes, what it shared loses its chunks and is derived again
from the chunks left, and an entity takes the id of its current winning name,
as a full build would give it (stored communities and reports keep the old id
until a rebuild). The merge is always on: the removed `indexing.cross_run_merge`
key logs a WARNING and is ignored. The rules are in
[Design §5.4](./design.md#54-identity-derived-content-and-merge-rules).

---

## 6. Evaluation (`run-eval`)

`run-eval --data-path <dataset.json>` answers every question of a dataset and
scores the answers and their sources. Three evaluators need no LLM and cost
nothing; two use an LLM judge. Several strategies (or `--repeats`) turn the run
into a paired comparison.

Every flag with its default: [CLI Reference](./reference/cli.md#run-eval).

### Evaluators

`evaluation.enabled_evaluators` selects them; all five are on by default. The
deterministic ones skip a query that lacks their dataset fields; for a
judge-free run set `enabled_evaluators: [answer_match, retrieval, graph_aware]`.

| Evaluator | LLM | Needs in the dataset | Main metrics |
|---|---|---|---|
| `answer_match` | no | `answer` (+ `answer_aliases`) | `answer_contains` (headline), `exact_match`, `token_f1` |
| `retrieval` | no | `reference_sources` | `hit_at_k`, `recall_at_k`, `mrr` (k = `evaluation.retrieval_k`, default 5) |
| `graph_aware` | no | `expected_entities` / `expected_relationships` | entity and relationship coverage (recall), in the answer and in the retrieved sources |
| `langchain` | judge | `answer` | `correctness`, `partial_correctness` |
| `ragas` | judge | `answer` | `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, `faithfulness` |

The two judges need the `eval` extra (`uv sync --extra eval`, [§1](#install));
without it `run-eval` stops before the first query with exit `2`, naming the
extra, even with `processing.ignore_errors: true`. The judge-free run above
needs no extra. Any other enabled evaluator that cannot be built (no Bedrock
access for the judge, say) or rejects its configuration stops the run; with
`processing.ignore_errors: true` it is dropped and listed in
`run_manifest.dropped_evaluators`. Scoring starts after every answer is
generated, so an evaluator error never discards the run: the metric is
recorded as failed for that query (as is a judge reply without a usable
score, never 0), and `--max-failure-rate` decides the exit code.

#### `ragas` contexts

- `context_precision` makes one judge call per context, so it scores at most
  `evaluation.ragas_max_contexts` (default 20; `null` = no cap) top-ranked
  sources (`context_precision@N`); without the cap, strategies that report
  100+ sources hit `ragas_timeout`.
- `faithfulness` and `context_recall` see every source within
  `max_context_tokens`. By default (`evaluation.max_context_tokens: null`)
  that is the answer model's budget (`search.token_manager.max_context_tokens`),
  so the answer is graded against the context it was given; a smaller value
  logs a warning.
- Each report records what the judge saw (`judge_contexts`,
  `judge_context_tokens`, ...); none of this changes what the answer model
  saw.
- ragas usage telemetry is off (`RAGAS_DO_NOT_TRACK=true` by default); set
  `RAGAS_DO_NOT_TRACK=false` in the environment to opt back in.

#### `graph_aware`

The share of the expected graph artifacts that appear in the generated answer
(coverage = recall), with the phrase matcher of `answer_contains`. A
relationship given as `{"source": "A", "target": "B"}` or `"A -> B"` counts
when the answer mentions both endpoints; any other string must appear as a
phrase. Precision and F1 are not emitted: the entities of a free-text answer
cannot be enumerated reliably.

Answer-text coverage rises with answer length, so compare strategies on
`retrieved_entity_coverage` / `retrieved_relationship_coverage` (the same
expectations matched against the entities and relationship endpoints of the
sources the answer model was given) and read `answer_length` next to the
answer coverage. Retrieved coverage is skipped, not 0, when the context had
no source of that type (`global` by design), so check the metric's `pairs` /
`unpaired` in a comparison and `metric_outcomes`' skipped count in one run.

#### `retrieval`

Did the sources the answer model saw include the gold documents?

- **Attribution.** Text-unit sources name their file; entity, relationship
  and community-report sources are attributed to the files of the text units
  they cite, the file holding most of them first.
- **k counts files, not sources**: the sources are flattened into one list of
  distinct files in rank order, so a community report attributed to many
  files cannot inflate hit@k.
- **Per source type.** `by_source_type` (aggregated in
  `grouped_statistics.source_type`) has the same metrics per section type; its
  `text` row, the text units the answer model read verbatim, is the fairest
  cross-strategy comparison. `attributable_fraction` (attributable / reported
  sources) shows how much of the context the metrics cover.
- **Matching.** NFKC-normalized and case-insensitive on the file name, a
  reference's directories ignored: `docs/Terms.pdf`, `terms.pdf` and `terms`
  match the source `Terms.pdf`, `terms.docx` does not, and a version suffix
  is no extension (`notes.v1` does not match `notes.v2`). A title such as
  `AC/DC` is compared whole.
- **Skips.** Skipped when a query has no `reference_sources` or none of its
  sources is attributable; a query that retrieved no sources scores 0.

#### `answer_match`

Scores against `answer` and the optional `answer_aliases` (a string or a
list, top level or under `metadata`, which wins), taking the maximum.

- **`answer_contains`** is the headline: 1.0 when the gold answer or an alias
  appears in the generated answer as a whole-word phrase. Long RAG answers
  rarely equal a short gold span, so it tracks correctness far better than
  exact match. A possessive is read as its word (`Depot` matches `Depot's`;
  `Depots` is another word); `graph_aware` matches the same way.
- **Korean and CJK.** Korean particles are tolerated after a word (`AWS`
  matches `AWS는`); a bare number may carry a Korean unit (`2024` matches
  `2024년에`) but not a magnitude numeral (`3` does not match `3억`); spacing
  next to CJK letters is ignored (`3억 원` = `3억원`); a single-word CJK gold
  is matched as a substring, but not inside a longer number or Latin word
  (`2년` does not match `12년`).
- **Yes/no golds** (`yes`/`no`/`nope`, Korean `예`/`네`/`아니요`/`아니오`/
  `아닙니다`, Chinese `是`/`对`/`否`/`不是`/`没有`, Japanese `はい`/`いいえ`) are
  scored by the polarity of the first yes/no word that answers the question,
  as in HotpotQA (`there is no clause`, `네 가지`, `(예: …)` and `No. 5` are
  not answers, so `네, 맞습니다` answers yes but `네 맞습니다` does not).
- **`exact_match` and `token_f1`** score the part of the answer
  `evaluation.answer_span` names. SQuAD, HotpotQA and MuSiQue score a short
  predicted answer (a span such as `Depot`) against the gold answer, while
  the answer prompt writes a one-sentence answer and then its support, so
  the whole answer would score its length as much as its correctness:
  - `first_line` (default, free): the first line that is neither empty nor a
    Markdown heading, the one-sentence answer (`Depot stores widgets for
    Vendor.` scores token F1 1/3 against `Depot`; EM still under-counts).
  - `extracted`: the `models.roles.answer_span` model (fast tier; prompt
    `custom_prompts.answer_span_extraction_*`) copies the short span out of
    the answer, one call per distinct question and answer, reused across
    repeats and strategies. A failed call, an empty reply or a span that is
    not in the answer is scored on the first line instead and counted in
    `answer_span_statistics.fallbacks`. This is the setting for numbers
    close to the papers'.
  - `full`: the whole answer.

  Each report records `answer_span` and the `scored_span`. The normalization
  is SQuAD v1.1's (NFKC, lowercase, punctuation and English articles
  removed, whitespace collapsed), keeping punctuation that changes meaning
  (`-5`, `1.5`, `C++`); `evaluation.answer_normalization: official_squad` is
  the official script (MuSiQue's evaluation uses it too) and `hotpotqa` adds
  HotpotQA's rule that a `yes`/`no`/`noanswer` span or gold earns token F1
  only when the two are equal. Token F1 splits on whitespace, so for Chinese
  and Japanese it degrades to exact match, and Korean particles make both
  under-count.

### Eval data format

A JSON array of objects. Only `question` is required; `expected_entities` /
`expected_relationships` are needed only by `graph_aware`.

```json
[
  {
    "id": "q1",
    "question": "What are the main themes discussed in the documents?",
    "answer": "The main themes include AI, machine learning, and data processing.",
    "category": "general",
    "difficulty": "easy",
    "reference_sources": ["doc1.pdf", "doc2.txt"],
    "expected_entities": ["AI", "machine learning", "data processing"],
    "expected_relationships": [
      { "source": "AI", "target": "machine learning" },
      "machine learning -> data processing"
    ],
    "metadata": { "search_strategy": "global", "answer_aliases": ["AI and ML"] }
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

- Per-item `metadata` (e.g. `search_strategy`) overrides the CLI defaults for
  that question. `id` may also be given as `query_id`.
- Mark a question the corpus cannot answer with
  `"metadata": {"answerable": false}`: no evaluator grades it; it counts only
  in `abstention_statistics` (below).
- The file is validated before any query runs. A file that cannot be read,
  an empty or non-array dataset, a missing `question`, a duplicate id, a field
  of the wrong type (a non-list `expected_entities`, an `answer_aliases` that
  is not a string or a list of strings) or `metadata` the RAG chain rejects
  (an unknown `search_strategy`, or `filters` it refuses: a malformed value
  such as an empty list, or a key the item's strategy refuses, checked under
  every compared strategy in a comparison) stops the run with
  `Error: invalid evaluation dataset:`, the item index and id, and exit code 2.

### Examples

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

Either path can be an `s3://` URI, so `run-eval` also runs as a one-off task on
the CDK stack ([After deploy](../iac/README.md#5-evaluate-from-inside-the-vpc)).
Results are uploaded at the end with the `aws.s3.encryption` settings; a
failed upload exits non-zero and keeps the local files (path printed). S3 is
reached with the config's `aws.profile_name` / `aws.region_name`; the shell's
`AWS_PROFILE` / `AWS_REGION` fill them only when the config leaves them unset.
The principal needs `s3:GetObject` on the dataset and `s3:PutObject` on the
prefix.

Results are written as `evaluation_{results,reports,summary}_<timestamp>.json`
(`..._<strategy>_<timestamp>.json` when every query asked for, or used, one
strategy; an `auto` run is `..._auto_...`). `<timestamp>` is
`YYYYMMDD_HHMMSS_<8 hex characters>` (runs started in the same second do not
collide): find the latest run by sorting the names, not by building one.
The summary holds per-metric statistics (`metric_statistics`) and
scored/failed/skipped counts (`metric_outcomes`), plus:

- `grouped_statistics`: the same split by `search_strategy` (the strategy
  actually used, which varies under `auto`), `category`, `difficulty` and
  `source_type` (the retrieval metrics per source section type).
- `abstention_statistics`: how often the chain returned its fixed no-context
  reply ("I could not find relevant information…"): `abstained`, `answered`,
  `abstention_rate`, `per_strategy`, and an `unanswerable` block for
  `answerable: false` items. The chain returns that reply only when retrieval
  finds nothing, and a model declining in its own words is not detected, so
  on a populated index the `unanswerable` rate is a lower bound on correct
  declines. Each result carries `abstained`.
- `answer_span_statistics`: the `answer_span` exact match and token F1
  read, the reports `scored` and the extracted spans that fell back to the
  first line (`fallbacks`).
- `rerank_status_counts`: queries per [rerank status](#rerank-status); each
  result carries `rerank_status` (`null` when answer generation failed
  first).
- `guardrail_blocked_counts`: blocked queries per `input` / `output` (a
  blocked item is scored on the blocked message; a blocked answer drops its
  sources, so its retrieval metrics and the RAGAS context metrics
  (`context_precision`, `context_recall`, `faithfulness`) are skipped as
  `guardrail_blocked`. A query blocked on input retrieved nothing, so it is a
  retrieval miss and scores 0 on context precision and recall, while
  `faithfulness` is skipped, as for any answer without context:
  `no_contexts`); each result carries `guardrail_blocked`.
- `run_manifest`: what was measured, so two runs can be compared:
  - the CLI arguments (URLs without credentials), versions, `git_sha` /
    `git_dirty`, the dataset (path, hashes, query count), a UTC timestamp and
    `dropped_evaluators`;
  - the model ids and `judge_is_answer_model` (a judge that is the answer
    model inflates judge scores; the default Bedrock judge, Opus 5.5, is not
    the default answer model, and falls back to another when the answer
    model is Opus 5.5; a configured `models.roles.evaluation` that is the
    answer model, `default` say, is kept and warned about). The `--pairwise-judge` of a
    comparison counts as a judge, and `pairwise_judge` records its settings;
  - `answer_scoring`: the `answer_span` and `answer_normalization` exact
    match and token F1 used (with `answer_match`), and `models.answer_span`
    the extracting model; runs scored differently are not comparable on
    those two metrics;
  - `retrieval_config_sha256`, a hash of every configuration setting that
    can change an answer or a score (model ids and ingestion settings
    included; regions, endpoints, credentials, paths, timeouts and other
    operational knobs left out), so the same evaluation hashes the same in
    every environment. The `run-eval` query flags (`--suffix`, `--top-k`,
    `--search-type`, `--retrieval-multiplier`, `--enable-thinking`) are not
    configuration: they are in the CLI arguments, not in the hash;
  - `index`: per queried suffix, the ingestion run that last wrote its vector
    indices (`pipeline_id`, `mode`, `built_at`, `documents`, the extraction
    settings, for an incremental run the `full_build` it was applied on,
    and the `communities` counters, see §5); best-effort, `null` without
    index metadata.

  `EvaluationManager.evaluate_dataset` builds it for library callers too
  (pass `dataset_path=` / `cli_args=`).

Per-query reports list each evaluator's `metrics` (no overall score: an
average of unrelated metrics means nothing) and record, per reported source in
rank order, `retrieved_source_ids` (the files it is attributed to, `[]` when
unattributable), `retrieved_source_types`, `retrieved_entities` and
`retrieved_relationships`.

### Comparing strategies

`run-eval --search-strategy local,mix,global --repeats 3` answers and scores
the same queries with each strategy, `--repeats` times each (the listed
strategy overrides a per-item `search_strategy`; list them in one flag, as a
second `--search-strategy` is a usage error), saves every run's files and
adds `evaluation_comparison_<timestamp>.json`:

- `strategies.<s>.metrics.<evaluator>/<metric>`: the mean over queries of the
  per-query mean over repeats, and `repeat_stdev` (run-to-run noise);
  `routed_strategies` shows how `auto` routed. The chain extracts each
  question's entities and keywords once (`search.query_extraction_cache_size`,
  1024 questions), so the repeats and the strategies that read them (`mix`
  and `hybrid` the keywords, `local` the entities) search with the same
  ones, and the run-to-run noise is the retrieval's and the answer's.
- `deltas_vs_baseline.<s>.<metric>`: paired against the first strategy over
  the queries both scored: `mean_delta`, a 95% paired bootstrap interval
  (`ci_low`/`ci_high`; 2,000 resamples, seed 0), `wins`/`ties`/`losses`, the
  per-query deltas and `unpaired`. A query whose answer generation failed
  scores 0 on every metric the other run scored (`answer_length` excepted)
  that its own strategy scores on some query, so failing cannot help a
  strategy, and a metric it never produces (`global` reports no entities,
  so no `retrieved_entity_coverage`) is not invented as a loss; a strategy
  that scored nothing is filled on every metric. The rule reads the data, so
  a metric that only the failed query could score (one item with
  `expected_relationships`, say) is not filled either
  (`failed_answers` / `failed_queries`;
  `guardrail_blocked_counts` counts blocks; `warnings` names strategies that
  failed, or had blocked, different numbers, runs whose exact match and
  token F1 read different `answer_span`s, and, with `answer_span:
  extracted`, each strategy whose extracted spans fell back to the first
  line (`answer_span_fallbacks`) and fallback rates that differ). A
  difference is
  supported when its interval excludes 0; a 10-point difference on binary
  metrics is detected in 15% of runs with 50 queries and 59% with 200, so
  compare on a few hundred queries.
- `deltas_by_category.<s>.<category>.<metric>`: the same deltas over each
  dataset `category` (MuSiQue's hop count, say). Each interval is per
  category and not corrected for the number of categories and metrics read,
  so treat a lone excluded 0 among many as exploratory.
- `pairwise_vs_baseline.<s>` (with `--pairwise-judge`): the GraphRAG /
  LightRAG head-to-head LLM judge on comprehensiveness, diversity,
  empowerment and directness (plus LightRAG's `overall` with
  `evaluation.pairwise_overall: true`), using `models.roles.evaluation`
  (prompt: `custom_prompts.pairwise_comparison_*`) on the first repeat's
  answers. Each pair is judged in both orders and a criterion counts as a
  win only when both agree; `position_consistency` (overall and per
  criterion) is the share of judged pairs whose two orders agreed, and a
  value under 0.5 is warned about: such a judge decides by position, and its
  ties say nothing about the answers. `candidate_win_rate` scores each
  decided pair 1, a tie 0.5 and a loss 0, with a bootstrap interval
  (`ci_low`/`ci_high`) over those per-query scores; it is `null` when the
  judge decided no pair. A pair where one answer failed is not judged: the
  other side wins (`candidate_errors`, `baseline_errors`; both failed:
  `both_errors`). `answerable: false` items are not judged
  (`unanswerable_skipped`). `failed` counts pairs the judge could not decide
  (a failed call, or a verdict other than `1`, `2` or `tie`), and
  `--max-failure-rate` applies to them. A judge that is the answer model is
  warned about.
- `cost.strategies.<s>`: the answers' model use per query, as `mean` /
  `p50` / `p95` over every answer of every repeat: `llm_calls_per_query`,
  `input_tokens_per_query`, `output_tokens_per_query`,
  `total_tokens_per_query`, `latency_seconds`, the `totals` and
  `cache_hits`, and with `evaluation.prices` set `usd_per_query` and
  `usd_total`. `cost.deltas_vs_baseline.<s>` pairs them against the
  baseline like the metrics (`usage/llm_calls`, `usage/input_tokens`,
  `usage/output_tokens`, `usage/total_tokens`, `usage/cost_usd`,
  `latency/response_time_seconds`, each query's mean over its repeats, with
  the bootstrap interval). A failed or timed-out answer counts with what it
  spent before failing: each answer runs in its own usage scope and timer,
  so its result keeps the calls and tokens of its attempts and the time from
  its first attempt's start to its last one's end (`failed_answers` counts
  them; a warning says so). A strategy is then not cheaper or faster for
  failing, though a fast failure does cost less: read the cost next to
  `failed_answers` and the failure warning. Saved answers without a usage
  record or a latency are counted in `answers_without_usage` /
  `answers_without_latency` and warned about,
  and a `--fail-if` on a `usage/*` or `latency/*` measure fails for them.
  The arms and repeats of one run share the chain's query extraction cache,
  so a later one that reuses an extraction pays no call for it (counted in
  `cache_hits`): its cost depends on what ran before it, and a strategy whose
  repeat count or cache hits per answer differ from its baseline's is warned
  about. To compare cold-cache cost, set `search.query_extraction_cache_size:
  0`; the quality deltas do not depend on it. Before each strategy's first
  run, `run-eval` answers one synthetic query with it, untimed, unscored and
  not counted (`evaluation.warm_up_query`, default `true`), so the first
  strategy's latency does not include opening the store connections and the
  models' first calls; it costs one extra query (its model calls) per
  strategy and run, which is in neither the run's usage nor its USD
  estimate (Bedrock bills it all the same), and `false` turns it off.
- `run_manifest` of the first run; with `--pairwise-judge`,
  `pairwise_judge_model` and the judge's `pairwise_usage`.

The comparison is also written as `evaluation_comparison_<timestamp>.csv`
(one row per non-baseline strategy, measure and query: `baseline`,
`baseline_value`, `value`, `delta`, `category`) and `.md` (the metric and
cost tables, the pairwise judge's win rates, the gates and the warnings, for
a pull request or a CI job summary).

No model prices ship with the package. For a USD estimate, fill in the
current [Amazon Bedrock prices](https://aws.amazon.com/bedrock/pricing/) of
your region, per million tokens, keyed by the model id the usage records
name (`usage.by_model`; an inference profile id such as `global.<model>` when
the factory resolved one). Answers that used a model without a price get no
estimate, and the comparison warns. `input_tokens` includes the prompt-cache
reads and writes; give their rates (`cache_read_per_million_tokens`,
`cache_write_per_million_tokens`) and they are priced apart, otherwise at
the input rate (an upper bound for reads, which cost a tenth of input on
Claude):

```yaml
evaluation:
  prices:
    "global.<answer-model-id>":
      input_per_million_tokens: 0.0
      output_per_million_tokens: 0.0
      cache_read_per_million_tokens: 0.0
      cache_write_per_million_tokens: 0.0
    "global.<fast-model-id>": {input_per_million_tokens: 0.0, output_per_million_tokens: 0.0}
```

The estimate covers the calls the usage records: botocore's resent requests
(`request_retries`), failed calls (`failed_calls`) and the warm-up queries
are not priced, so CloudWatch's invocation count and the bill can run a
little higher (in the package's November 2026 benchmark runs: Sonnet 5.5
13% more invocations than recorded calls, input tokens within 4.6%, output
within 0.9%; Haiku 5.5 within 0.2%).

The console prints the same summary; the failure budget applies to every run.

### Rebuilding, re-judging and gating a comparison

`run-eval --from-results <dir|files>` rebuilds the comparison from the files
earlier runs saved (`evaluation_results_*` with the `evaluation_reports_*`
and `evaluation_summary_*` next to them; `evaluation.save_detailed_results:
true`, the default) without answering a query or opening a store. The runs
are grouped by strategy and compared in the order the oldest run listed them
(`--search-strategy` picks and orders them instead), so a comparison that was
stopped part-way is finished by running only the missing strategies into the
same directory, with the same `--repeats`, and rebuilding: its deltas are
those of an uninterrupted run. A strategy whose runs were all lost runs again
with `--repeats N`; one with some repeats saved runs only the missing ones
(`--repeats` of the difference), or delete its partial files and run all N.

```bash
# Stopped during naive (run with --repeats 3): run it alone into the same
# directory with the same --repeats, then rebuild.
run-eval --data-path eval.json --search-strategy naive --repeats 3 --output-directory out/
run-eval --from-results out/ --output-directory out/

# Re-judge the saved answers with another judge model (no answer call).
run-eval --from-results out/ --pairwise-judge --judge-model <judge-model-id>

# CI: compare this run with an earlier one and fail on a supported regression.
run-eval --data-path eval.json --search-strategy local \
  --baseline-results ci/previous/ --output-directory ci/current/ \
  --fail-if "answer_match/token_f1:ci_high<-0.02" \
  --fail-if "usage/total_tokens:mean_delta>500"
```

- The saved runs of one strategy are pooled as its repeats. The rebuild
  warns when they differ in a query flag (`suffix`, `top_k`, `search_type`,
  `retrieval_multiplier`, `enable_thinking`), `retrieval_config_sha256`,
  `git_sha`, the `index` build or the answer model (keep runs of other
  settings in another directory, or list the result files), when the
  strategies ran different numbers of repeats, and when a run has no
  `evaluation_summary_*` file (its manifest, so none of these checks, nor
  the answer-scoring and self-preference ones, can read it).
- `--pairwise-judge` with `--from-results` judges the saved answers again
  (`--judge-model` sets `models.roles.evaluation` for the run, here and in a
  normal run; without `--pairwise-judge` and an LLM-judge evaluator it
  judges nothing, which is warned about); the self-preference warning
  compares that judge with the saved runs' answer model.
- `--baseline-results` makes an earlier run the baseline: each of its
  strategies is labelled `baseline:<strategy>`, and each strategy of this
  run, or of `--from-results`, is compared with the baseline of the same
  strategy (`deltas_vs_baseline.<s>` and `cost.deltas_vs_baseline.<s>`
  against `baseline:<s>`; `baselines` maps each strategy to its baseline).
  A strategy the earlier run did not run is compared with its first
  strategy, with a warning. Run the baseline with the same `--repeats` as
  the runs compared with it: repeats reuse the query extraction cache, so
  different counts shift the usage deltas (warned about). Save each run in
  a directory of its own: an `--output-directory` that is the
  `--baseline-results` directory is warned about, since the next run that
  reads it as its baseline pools this run in as repeats. Runs whose exact
  match and token F1 read the answer differently (`answer_scoring` in the
  run manifest: `evaluation.answer_span`, `answer_normalization`) are
  refused (exit 2), checked before any answer is generated; runs of different
  datasets are warned about (only their shared query ids pair).
- `--fail-if <measure>:<stat><op><value>` (repeatable) exits 1 when a
  compared strategy's paired delta meets the condition: `<measure>` is a key
  of `deltas_vs_baseline` (`<evaluator>/<metric>`) or of
  `cost.deltas_vs_baseline`, `<stat>` is `mean_delta`, `ci_low` or `ci_high`.
  `ci_high<-0.02` fails only when the whole interval is below -0.02, a
  regression the query sample supports. A strategy with no delta of the
  measure fails the gate too, and so does a cost measure (`usage/*`,
  `latency/*`) that some answers of the strategy or of its baseline have no
  value of, or when the strategy and its baseline failed different numbers
  of answers (each counts with what it spent before failing, so the delta
  compares the failures too). The result is in `comparison.gates` and the
  JSON `gate_failures`. It needs a comparison: two strategies, or
  `--baseline-results`.

### Benchmarks (MuSiQue)

`prepare-benchmark musique` turns
[MuSiQue](https://github.com/StonyBrookNLP/musique) (answerable, multi-hop,
CC BY 4.0; Trivedi et al., TACL 2022) into a corpus and a `run-eval` dataset.
Nothing of it is in this repository: by default it downloads the answerable
dev split (2,417 questions) from a pinned revision of a public mirror and
checks its SHA-256; `--data-path` reads a local `musique_ans_v1.0_*.jsonl`
instead. A full file (`musique_full_v1.0_*.jsonl`) is read too, but its
answerable half is not MuSiQue-Ans (other paragraphs and answer aliases):
its rows are labelled `metadata.benchmark: musique-full-answerable` instead
of `musique`, with a WARNING that the scores are not comparable to MuSiQue-Ans
results. The label is in its output (console and `--output-format json`),
and `run-eval` counts it in the summary (`benchmarks`, also printed under a
comparison of several strategies or repeats) and the run manifest
(`dataset.benchmarks`). It writes `corpus/` (one text file per distinct paragraph; each
question brings 2-4 supporting paragraphs and up to 18 distractors) and
`eval_data.json` (`answer`, `answer_aliases`, the supporting paragraphs as
`reference_sources`, the hop count as `category`). A `corpus/` that is not
empty is a usage error (exit 2), reported before anything is downloaded.

```bash
# 1. 100 questions (about 1,900 paragraphs), sampled with seed 0.
prepare-benchmark musique --output-directory bench/musique --limit 100

# 2. Ingest the paragraphs under their own suffix.
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# 3. Paired comparison of both methodologies, EM / F1 on the short answer span.
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy local,mix,global,naive --repeats 2 --config-path musique.yaml
```

with `musique.yaml` holding your usual config plus:

```yaml
processing:
  document_parsing:
    index_value: musique          # keep the benchmark out of your own index
evaluation:
  answer_span: extracted                 # EM / F1 on the short span the answer gives
  answer_normalization: official_squad   # MuSiQue's evaluation script normalization
  enabled_evaluators: [answer_match, retrieval]   # judge-free; add langchain / ragas if wanted
```

MuSiQue (like HotpotQA and SQuAD) scores the short answer a reader model
predicts against the gold answer and its aliases, with EM and token F1 under
the official normalization. A RAG answer is a sentence and its support, so
`answer_span: extracted` (one fast-model call per answer) has a model copy
the span out of it first; the numbers are then close to, but not the same
as, a paper's, whose readers were trained to output spans. The default
`first_line` costs nothing but scores the one-sentence answer, so its EM is
far below a paper's. The GraphRAG and LightRAG papers report no EM / F1;
they use the pairwise LLM judge (`--pairwise-judge`).

Read `answer_contains` / `token_f1` and `retrieval` (per `category`, and
`deltas_by_category` in a comparison).
Ingestion cost grows with the paragraph count (about 19 per question). Cite
the paper when you publish results.

---

## 7. Visualization (`run-visualization`)

A **standalone** renderer that draws from an exported visualization-data
JSON; it does **not** re-run ingestion or touch AWS. Every flag with its
default: [CLI Reference](./reference/cli.md#run-visualization).

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

It needs the `viz` extra (`uv sync --extra viz`, [§1](#install)). The two
renderers are **`interactive`** (pyvis) and **`static`** (Bokeh), configured
under `graph.visualization` (`interactive.*`, `static.*`, `centrality.*`).
Ingestion itself renders nothing.

- **The input.** The community-detection stage of `run-ingestion` writes
  `visualization_data.json` on every run whose graph covers the whole corpus:
  a full build, a reset, or an incremental run in which every document is new
  or changed. An incremental run that skips unchanged documents writes
  nothing (it logs an INFO line) and leaves an earlier file in place; run one
  full build (`indexing.reset: true`) to visualize the whole graph.
- **Where.** `graph.visualization.outputs_directory`, by default
  `<cache directory>/<pipeline_id>/visualization/`. S3 cache sync uploads the
  `.json` only; re-render the HTML locally.
- **What.** Nodes (colored by their level-0 `community_id`), edges and the
  community hierarchy, without vectors. Writing it calls no model;
  `run-visualization` computes a seeded spring layout from the graph
  topology and the `centrality` metrics when it renders.
- **Size.** The interactive graph keeps the top `interactive.max_nodes` nodes
  by degree (default 2000) and lays out only those.
- Edge width and opacity are log-scaled to the graph's own weight range.
- Only files a renderer wrote are reported (an earlier file of the same name
  is removed first); when nothing is rendered (an empty graph) the command
  exits `1`.

---

## 8. Prompt tuning (`run-prompt-tuning`)

Samples documents from a directory, profiles the corpus (domain, language,
persona, entity types) with `models.roles.graph_extraction`, and writes a
domain-adapted `custom_prompts` YAML fragment to review and merge into
`config.yaml`. Every flag with its default:
[CLI Reference](./reference/cli.md#run-prompt-tuning).

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
  [extraction schema](#c-extraction-schema) draft: the entity types the
  examples were extracted with (above), the relationship types the tuning
  examples state most often (at most 12, with their counts in a comment)
  and a pattern only where every pair of endpoint types of a relationship
  type appears at least twice. Every type and pattern the config declares
  stays, with its description and IRI, and a relationship type with
  configured patterns gets no other. It is a `schema_source: {kind: file}`
  file, to load with `schema_mode: align` after review (its comment names
  the path relative to the `--config-path` file's directory); the same
  model answers give the same file. It never replaces your config or
  schema: naming the `--config-path` file or the schema file the config
  loads (with `--schema-out` or `--output-path`) exits 2, even with
  `--force`, and so does naming an existing file with either unless you
  add `--force`.
- Non-text formats go through the ingestion loaders (custom
  `ParserFactory.register_loader` ones included); unparsable files are skipped
  with a warning.
- If the model returns no JSON profile, the command exits non-zero and writes
  nothing.

---

## 9. Domain adaptation

Three levers specialize the pipeline for a domain (medical, legal, finance,
...): the entity types (A), the prompts (B) and the extraction schema (C).
A knowledge graph you already have joins the extracted one as a seed graph
(D).

### A. `entity_types` (cheapest, highest-impact)

Set the categories the extraction prompt asks for, with no prompt rewrite:

```yaml
processing:
  graph_extraction:
    entity_types:
      - "GENE: Genes, gene products, loci"
      - "DISEASE: Disorders, syndromes, conditions"
      - "DRUG: Medications, compounds, dosages"
      - "TRIAL: Clinical trials, studies, cohorts"
```

### B. `custom_prompts` overrides

Override a prompt's `*_system` / `*_human` text (`null` keeps the built-in
prompt; five prompts have no override, see [Custom prompts](#custom-prompts)).
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

Override keys (each `_system` + `_human`): `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `query_rewrite`, `entity_extraction`,
`drift_hyde`, `drift_primer`, `drift_action`, `strategy_selection`,
`keywords_extraction`, `global_map`, `pairwise_comparison` (the
`--pairwise-judge` judge; keep its JSON verdict format, with an `overall`
verdict under `evaluation.pairwise_overall`), `answer_span_extraction` (the
`evaluation.answer_span: extracted` span extractor) and the prompt-tuning
`corpus_profile`.

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

A `*_system` override replaces the whole built-in system prompt, output
instructions included. For `graph_extraction` and `community_report` that is
the XML format the parser reads (and, for extraction, the verbatim
`<source_text>` rule the hallucination guard checks), so keep it: start from
`run-prompt-tuning` output or copy the format from
`unified_kg_rag/domain/prompts/graph_extraction.py`. An override of these two
that drops a tag the parser reads (`<entities>`, `<relationships>`,
`<source_text>`; `<community_name>`, `<summary>`, `<rating>`,
`<rating_explanation>`, `<findings>`) logs a warning naming the tags.

**Recommended flow:** `run-prompt-tuning` → review → merge the useful prompts
and tune `entity_types` → re-ingest.

### C. Extraction schema

Entity types, relationship types and the patterns allowed between them form
the extraction schema. With the defaults it is the `entity_types` list
alone, and extraction works as it always has. Declare relationship types
(and, if you like, patterns and ontology IRIs) when the graph should use
one fixed vocabulary: the same relationship written `works for`,
`Works_For` and `WORKS_FOR` then becomes one edge label, and every stored
item records whether it fits the schema.

**Which mode.** `schema_mode` chooses what extraction does with the schema:

| Mode | Keeps | Use it when |
|---|---|---|
| `off` | everything, as written | the schema should only shape the prompt |
| `align` (default) | everything; declared types written as declared | you want consistent labels and can live with, or want to review, items outside the schema |
| `strict` | only items of declared types and allowed patterns | downstream users of the graph accept only the ontology's types |

`strict` costs recall: whatever the model types outside the schema is lost,
and with an entity go the unit's relationships naming it. Start with
`align`, look at what falls outside the schema (`filters={"in_schema":
False}`, below), widen the schema, and switch to `strict` only when little
falls outside. To measure the cost on your questions, ingest the corpus
under two index suffixes (one per mode, [Index suffix](#index-suffix)) and
run the same `run-eval` dataset against each.

**Declaring the schema.** Inline, in `processing.graph_extraction`:

```yaml
processing:
  graph_extraction:
    entity_types:
      - "PERSON: People"
      - {label: ORGANIZATION, description: "Companies", iri: "https://schema.org/Organization"}
    relationship_types:
      - "WORKS_FOR: Employment"
      - {label: PARTNERS_WITH, iri: "https://example.org/onto#partnersWith"}
    relationship_patterns:
      - "PERSON -WORKS_FOR-> ORGANIZATION"
      - {source_type: ORGANIZATION, relation: PARTNERS_WITH, target_type: ORGANIZATION}
    schema_mode: align     # off | align (default) | strict
```

- A type is `"LABEL: description"` (the description is optional) or a
  `{label, description, iri}` mapping; `iri` names the ontology class (entity
  type) or property (relationship type) it maps to. A pattern is
  `"SOURCE -RELATION-> TARGET"` or a `{source_type, relation, target_type}`
  mapping, and must name declared types. A misspelled mapping key fails the
  load.
- Labels match ignoring case, surrounding spaces and the choice of space or
  `_` between words (`works for` = `WORKS_FOR`), never by similarity:
  `PERSONS` is not `PERSON`. Two labels of one kind may not match each other.
- From the environment the lists are JSON, e.g.
  `UKR__PROCESSING__GRAPH_EXTRACTION__RELATIONSHIP_TYPES='["WORKS_FOR: Employment"]'`
  and `UKR__PROCESSING__GRAPH_EXTRACTION__SCHEMA_MODE=strict`
  ([Environment variable overrides](#environment-variable-overrides)).
- Or keep the schema in a file: `schema_source: {kind: file, path:
  schema.yaml}` reads `relationship_types`, `relationship_patterns` and,
  optionally, `entity_types` from a YAML or JSON file of the same shape (a
  relative path is resolved against the config file). Leave those keys out
  of the config then: a file and a config that both set `entity_types` fail
  the load. The path is a local file (or a `file://` URL) or an S3 object
  (`s3://bucket/schemas/schema.yaml`, at most 16 MiB), read with the run's
  AWS session; an `https://` location fails the load (it needs a source
  registered for it). `kind: owl` reads an OWL or RDFS ontology the same
  way: a local file or directory, an S3 object, or an S3 prefix read as a
  directory (`s3://bucket/ontology/`, the trailing `/` required). Declare
  `@base` (or `xml:base`) in an ontology with relative IRIs: without it they
  resolve against the file's location, so moving the file renames them.
  See [From an OWL/RDFS ontology](#from-an-owlrdfs-ontology). Other formats
  (SHACL, a schema service) plug in as a registered schema source
  ([Design Doc §15](./design.md#15-extension-guide)).
- **On AWS** the ingestion task reads the config baked into the image
  (`docker/Dockerfile` copies only `docker/config.yaml`, to
  `/app/config.yaml`), so a local `kind: file` schema (or `kind: owl`
  ontology) must be in the image too: add a `COPY` of the file next to it and rebuild, keep the schema in
  S3 (the task roles' permissions: [Operations Guide](./operations.md#schema-sources-in-s3)),
  or declare it inline in that config or through the
  `UKR__PROCESSING__GRAPH_EXTRACTION__*` variables above.

**What extraction does.**

- **Prompt.** In every mode the declared relationship types and patterns
  replace the example relationship types of the extraction prompt. With no
  relationship type declared, the prompt keeps its examples.
- **`align`** keeps every entity and relationship. A type that names a
  declared one is written as declared (`works for` becomes `WORKS_FOR`)
  before ids are built, so its spellings give one edge label and one
  relationship id. Off-schema and off-pattern items are counted. With entity
  types alone (no relationship types, as in the default config) labels stay
  as written.
- **`strict`** does the same and drops, per text unit:
  - an entity whose type is empty or undeclared, with the unit's
    relationships naming it. The unit records the rejection, so gleaning
    cannot bring the entity back through a relationship; a statement of the
    same entity with a declared type keeps it.
  - a relationship of an undeclared type.
  - a relationship a pattern does not allow, when the unit lists both
    endpoints with declared types. An endpoint the unit does not list passes,
    and a reversed relationship is dropped, not flipped. A relationship type
    without patterns is not restricted.

  Gleaning follows the same rules: it does not add an off-schema entity or
  relationship, and ignores a correction to an undeclared type or to a
  direction no pattern allows.
- An endpoint's type is the unit's merged type of the entity (its most
  frequent type). Being declared does not make a type win: under `align` an
  entity stated twice as `VENDOR` and once as `ORGANIZATION` stays a
  `vendor`.
- Drops are not failures: a text unit whose items were all dropped is
  extracted (with nothing), counts as a success and is not retried.
- Claims are not checked against the schema, but a claim's subject must be
  an entity its text unit states: under `strict`, a claim about an entity
  dropped as off schema in that unit is dropped with it. An object outside
  the graph is kept as the claim's literal value, as without a schema.
- The graph extraction stage reports `entities_off_schema`,
  `entities_dropped_off_schema`, `relationships_off_schema`,
  `relationships_dropped_off_schema`,
  `relationships_dropped_off_schema_endpoint`, `relationships_off_pattern`
  and `relationships_dropped_off_pattern` in its `extraction_stats`. The
  gleaning stage reports the same counters for what gleaning proposes in
  its `gleaning_stats` (kept and counted under `align`, dropped under
  `strict`), with `corrections_ignored_off_schema`. Both are in the stage
  metrics of `<cache directory>/<pipeline_id>/pipeline_metadata.json`, and
  each stage logs one summary line; the off-schema labels themselves are
  logged at DEBUG only.
- A relationship of an undeclared type is counted in
  `relationships_off_schema` whatever its endpoints, so the count is the
  same under `align` and `strict`. Each strict drop is counted once: an
  undeclared type first, then an endpoint dropped as off schema
  (`relationships_dropped_off_schema_endpoint`), then a pattern. The counters
  count statements per text unit; an `in_schema: false` listing shows the
  merged items, one per entity or relationship.

**What is stored.** Indexing writes these fields on entities and
relationships, in the graph and the vector stores:

| Field | On | Value | Written |
|---|---|---|---|
| `in_schema` | entities, relationships | whether the stored type is a declared type of that kind (an empty type is not); `true` for every item of a kind with no declared type (nothing of it is off schema) | under `strict`, and under `align` once relationship types are declared, on both kinds |
| `type_iri` | entities | the `iri` of the declared entity type the stored type names | whenever such a type has an `iri` and the mode is not `off`, with entity types alone too |
| `predicate_iri` | relationships | the `iri` of the declared relationship type | as `type_iri` |

- The values follow the stored (merged) type, so a full build and an
  incremental run store the same ones. Whether a relationship fits a
  pattern is not stored: it depends on the endpoints' types, which other
  documents can change.
- An entity named only as a relationship endpoint has no type and is off
  schema when entity types are declared, under `strict` too.
- A field that does not apply is not written at all, so a config without
  relationship types or IRIs (the default included) stores its items
  without these fields. Ids never change because of them.

**Filters.** The fields are filter keys on the entities and relationships
indices and on Neptune entity vertices (graph expansion), like any other
field ([Filters](#filters)):

```python
RAGInput(query="...", search_strategy="local", filters={"in_schema": False})
RAGInput(query="...", search_strategy="mix",
         filters={"type_iri": "https://schema.org/Organization"})
```

The first returns only off-schema entities (and, with `mix`, `hybrid`,
`local` or `drift`, off-schema relationships): the review list of what
`align` kept outside the schema. An item stored without the field matches
no value of it, and so does an index that does not map the field yet (the
next ingestion run maps it; no query fails for it).

A filter on a field the configured schema writes on no item is refused
before any model call (`InvalidFilterError`), as it could only empty the
results: `in_schema` under `off` or under `align` with entity types alone,
`type_iri` when no entity type declares an `iri` (or the mode is `off`),
`predicate_iri` likewise for relationship types. The check reads the
query's config, its schema source included; when the source cannot be
read, the key is not checked and a warning is logged once.

**Changing the schema.** What the next incremental run does depends on
whether the model sees the change:

- **Re-extracts its scope**: a change the prompt or the schema check
  reads. That is a type's label or description, a pattern, `strict` on or
  off, and `off` against `align` once relationship types are declared.
  Items of other index scopes keep what their own runs extracted (labels as
  the model wrote them, off-schema items `strict` would drop) until those
  scopes run again or `indexing.reset: true` rebuilds the corpus.
- **Re-annotates only**: an IRI added, changed or removed, `off` against
  `align` with entity types alone, or another form of the same item
  (`"GENE: A gene"` or `{label: GENE, description: A gene}`; a pattern as a
  string or a mapping). Nothing is extracted or embedded again. The run
  rewrites `in_schema`, `type_iri` and `predicate_iri` of every stored
  entity and relationship of the index suffix, of every scope, from their
  stored types, and logs one INFO line with the counts; the stores then
  hold what a full build writes. The registry records which annotations
  the stores of each index namespace hold, once for all its scopes: the
  rewrite runs once per namespace, whichever scope's run finds the
  schema changed (a schema reverted on one scope after another scope's run
  is rewritten too), and a new scope of an annotated namespace reads
  nothing. A namespace without this record is read once.
  The stores are read in bounded pages. A rewrite that does not finish (a
  killed run, a failed write, a store that cannot be read) warns, leaves
  the run going, and is made again by the next run, under any schema. A
  custom store adapter that cannot list its items is skipped with a
  warning; `indexing.reset: true` rewrites it.
- **Changes nothing**: moving a schema file without editing it (the content
  is keyed, not the path). Upgrading with the defaults, or with entity
  types written as strings and no IRI, changes no cache key, extraction
  record or stored item; an incremental run maps the new fields on its live
  OpenSearch indices before its first write.

**Drafting a schema with prompt tuning.** `run-prompt-tuning --schema-out
schema.yaml` also writes a schema file draft (see [§8](#8-prompt-tuning-run-prompt-tuning)):

1. Run it over a representative sample of the corpus.
2. Review the draft: it lists the profile's entity types (the configured
   ones when the config declares them) and the relationship types the
   tuning examples state most often, with a count
   each, and a pattern only where the examples support it. Merge synonyms,
   drop noise, add descriptions and IRIs. To refine a schema the config
   already loads, write the draft to another file (the command refuses to
   overwrite the loaded one): the draft keeps every declared type and
   pattern with its description and IRI, so you merge only what it adds.
3. Load it with `schema_source: {kind: file, path: schema.yaml}` and
   `schema_mode: align` (the draft comes from a few chunks, so `strict`
   would drop what they did not show), and remove `entity_types` from the
   config if the draft sets them.
4. Ingest, review the off-schema items with `filters={"in_schema": False}`
   and the stage counters, extend the file, re-run.

**Local backend.** `backend: local` behaves the same: the SQLite graph
store and the embedded vector store hold the same fields, the same filters
apply (graph expansion included), and live embedded indices gain the
fields the same way.

#### From an OWL/RDFS ontology

`schema_source: {kind: owl}` fills the schema from an OWL or RDFS ontology:
its classes become entity types, its object properties relationship types
(each with its IRI, so `type_iri` and `predicate_iri` name the ontology's
terms) and its domains and ranges the patterns. The mapping translates, it
does not reason: it applies a fixed set of RDFS-style rules once at
start-up, with no model call and no OWL reasoner.

**Install.** The source needs the `ontology` extra (rdflib):

```bash
uv sync --extra ontology    # or: pip install -e '.[ontology]'
```

Without it, every CLI run with such a config (`run-ingestion`, `run-rag`,
`run-eval`, `run-prompt-tuning`) exits 2 before any model call, naming the
extra; `run-rag` needs it too, as queries read the schema to check filter
keys. The deployed image leaves the extra out: build it with
`--build-arg UV_EXTRAS="--extra ontology"` and copy the ontology into the
image next to the config
([Operator Runbook: Ontology schema in the image](./operations.md#ontology-schema-in-the-image)),
or keep it in S3 ([Schema sources in S3](./operations.md#schema-sources-in-s3)).

**Configure.**

```yaml
processing:
  graph_extraction:
    schema_source:
      kind: owl
      path: ontology/procure.ttl   # a file, or a directory of ontology files
    schema_mode: align
```

- Leave `entity_types` out of the config: the ontology sets them.
- `path` is a local file or directory, relative to the config file (a
  `file://` URL works too), or an `s3://` object or `s3://bucket/prefix/`
  read as a directory ([Schema sources in S3](./operations.md#schema-sources-in-s3)).
  A directory is read recursively as one ontology (files sorted, names
  starting with `.` skipped, an OASIS XML catalog such as Protégé's
  `catalog-v001.xml` skipped). Any other URL, and a `secret_arn`, fail the
  load.
- The format follows the suffix: `.ttl` Turtle, `.rdf` / `.owl` / `.xml`
  RDF/XML, `.jsonld` / `.json-ld` JSON-LD, `.nt` N-Triples, `.nq` N-Quads,
  `.trig` TriG; other files of a directory are skipped. `options.format`
  reads every file in one format instead, whatever its suffix (in a
  directory too, so keep only ontology files there).

**Options** go under `schema_source.options`; an unknown option or a bad
value fails the load naming `schema_source.options.<name>`. A term is a full
IRI or a CURIE (`ex:Vendor`) whose prefix comes from `prefixes`, then the
well-known `rdf`, `rdfs`, `owl`, `xsd`, `skos`, `schema` and `dcterms`,
then the ontology's own prefix bindings (a prefix bound to two namespaces
must be set in `prefixes`). A term that names nothing in the ontology fails
the load, so a typo does not silently select nothing.

| Option | Default | Meaning |
|---|---|---|
| `namespaces` | `[]`: every namespace but RDF, RDFS, OWL, XSD, SKOS and Dublin Core | IRI prefixes of the classes and properties to map |
| `classes` | `[]`: every class | Root classes, each mapped with its subclasses |
| `exclude_classes` | `[]` | Classes left out, with their subclasses |
| `max_depth` | `null`: every level | Subclass levels below the roots (`0`: the roots only; without `classes`, the roots are the top-level classes) |
| `properties` | `[]`: every object property | Root properties, each mapped with its sub-properties |
| `exclude_properties` | `[]` | Properties left out, with their sub-properties |
| `include_deprecated` | `false` | Map terms marked `owl:deprecated true` too |
| `prefixes` | `{}` | Prefix → namespace IRI for the CURIEs of these options |
| `label_source` | `auto` | `auto`: `skos:prefLabel`, then `rdfs:label`; `label`: `rdfs:label` only; `local_name`: the IRI's local name only. A term without a label gets its local name |
| `label_case` | `upper_snake` | `upper_snake` (`supplyChain` → `SUPPLY_CHAIN`, `공급 업체` → `공급_업체`) or `preserve` (as written, punctuation made spaces) |
| `languages` | `[en, ""]` | Preferred language tags of labels and descriptions, in order (`""`: untagged; `en` also matches `en-US`, after an exact `en`); any other tag comes after them |
| `labels` | `{}` | IRI or CURIE → the label to use, as written |
| `on_collision` | `error` | Two terms of one kind with one label: fail, or `qualify`: suffix each with its namespace's prefix (`BANK_EX`, `BANK_GEO`): one `prefixes` binds, else a well-known one, else one the files bind, else a short hash of the namespace. Pin the colliding namespaces in `prefixes` (see [*What a change re-runs*](#ontology-what-a-change-re-runs)) |
| `description_properties` | `[skos:definition, rdfs:comment]` | Annotation properties a description is read from, the first in the preferred language. Another property named here must be a property of the ontology or a literal annotation of one of its terms, or the load fails |
| `max_description_chars` | `300` | Longer descriptions are cut at a word boundary with "…" (`0`: no description from the ontology) |
| `hierarchy_hints` | `true` | Append "Kind of X.", "Narrower than P." and "Inverse of Q." to the descriptions |
| `patterns` | `closure` | `closure`: patterns from the domains and ranges; `none`: no patterns |
| `max_entity_types` | `64` | Fail when more classes are selected |
| `max_relationship_types` | `64` | Fail when more properties are mapped |
| `max_patterns` | `256` | Fail when more patterns result |
| `format` | `auto` | `auto` (by suffix), `turtle`, `xml`, `json-ld`, `nt`, `nquads` or `trig` |
| `max_bytes` | `67108864` (64 MiB) | Fail when the files hold more bytes together (see [*Load cost*](#ontology-load-cost)) |

**What is read.** Only named terms of the files' statements:

- Classes: what is typed `owl:Class` or `rdfs:Class`, the ends of
  `rdfs:subClassOf` and `owl:equivalentClass`, and the classes named in a
  relationship type's domain or range. A class named only in a datatype
  property's domain is none.
- Parents: a class's named `rdfs:subClassOf` objects, and the named
  members of an `owl:intersectionOf` it is a subclass of or equivalent to
  (`ex:Vendor owl:equivalentClass [owl:intersectionOf (ex:Organization
  [a owl:Restriction ...])]` puts `ex:Vendor` below `ex:Organization`). A
  class equivalent to an `owl:unionOf` covers it: the union's named
  members are below it.
- Relationship types: object properties (`owl:ObjectProperty` or one of
  its characteristics such as `owl:SymmetricProperty`, an end of
  `owl:inverseOf`, or a sub-, super- or equivalent property of an object
  property), an `rdf:Property` with a range that is not a literal type,
  and a sub-, super- or equivalent property of such an `rdf:Property`.
  Literal types are the XSD types, `rdfs:Literal`, `rdf:langString` and
  the other RDF literal types, an `rdfs:Datatype`, and the schema.org data
  types (`schema:Text`, `URL`, `Number`, `Integer`, `Float`, `Boolean`,
  `Date`, `DateTime`, `Time`, `CssSelectorType`, `PronounceableText`,
  `XPathType`, and anything typed `schema:DataType`, over `http` or
  `https`); they are never entity types.
- Left out: datatype properties (`owl:DatatypeProperty`, or an
  `rdf:Property` whose ranges are all literal types) and annotation
  properties (reported as skipped), a property typed only
  `owl:FunctionalProperty` or with only a domain or range (an INFO line
  names them: `N properties not read ...`), individuals, the RDF, RDFS,
  OWL and XSD vocabularies, and anonymous classes other than the class
  expressions above. Labels and descriptions are string literals.

**Mapping rules.**

- **Closure.** `rdfs:subClassOf` is transitive over the whole ontology,
  unselected classes included. A `subClassOf` cycle or an
  `owl:equivalentClass` makes one type, whose `iri` is its smallest
  selected IRI (properties likewise, with `owl:equivalentProperty`). A
  property's domain is what its own `rdfs:domain` statements, its
  super-properties' and its inverses' ranges all allow; the range mirrors
  it. Each statement is a class, an `owl:unionOf` of classes or all of the
  property's `schema:domainIncludes` together, and allows any of its
  classes. `owl:Thing` and `rdfs:Resource` allow anything. A symmetric
  property, or one that is its own inverse, allows its domain and range at
  both ends; with statements on one end only it takes them for the other,
  and its sub-properties and inverses inherit both (decided on the whole
  closure, so a range an inverse supplies is not overridden).
  Transitivity, functionality, cardinality and restrictions are ignored.
- **Selection.** Namespaces and deprecation first, then the `classes`
  subtrees down to `max_depth`, minus the `exclude_classes` subtrees; then
  properties the same way. Selecting a class does not select its
  ancestors; the closure still goes through unselected ones.
- **Flat hierarchy, push-down and lift.** Every selected class is a type of
  its own; no type inherits another's patterns at extraction time.
  Instead the patterns are written out per class: a property's sources are
  the selected classes that are, or are below, a class every domain
  statement allows (pushed down). When no selected class is, they are the
  nearest selected ancestors of the allowed classes (lifted). When neither
  exists, the property is dropped and the log says why. Targets follow the
  range the same way, and the patterns are every source with every target.
  A property with neither a domain nor a range gets no pattern, so even
  `strict` does not restrict it.
- **Inverses and symmetry.** An inverse pair is two relationship types,
  each with patterns in its own direction; nothing is flipped. Leave one
  out with `exclude_properties` if the graph should hold one direction. A
  symmetric property between n classes gives n × n patterns.
- **Order.** Types follow a depth-first walk of the selected hierarchy,
  siblings by label then IRI; patterns sort by relationship type, source,
  target. Only the statements decide it, never the file or its
  serialisation.

**Hierarchy and the stored annotations.** Because the types are flat:

- `in_schema` is true for an entity whose stored type is a selected class.
  With `classes: ["ex:Organization"]` and `max_depth: 0`, an entity the
  model types `VENDOR` is off schema (and `strict` drops it).
- `type_iri` is the IRI of the class the stored type names, the one the
  model wrote, not its ancestors: an entity stored as `VENDOR` has
  `type_iri` `https://example.org/procure#Vendor`, not `#Organization`.
  To fetch a subtree, filter on the list of its IRIs:

  ```python
  EX = "https://example.org/procure#"
  RAGInput(query="...", search_strategy="local",
           filters={"type_iri": [EX + "Organization", EX + "Vendor", EX + "Buyer"]})
  ```

  The command line cannot say this: `run-rag --filters` takes each key
  once and `set-filter` one value, and a comma is read as part of the
  value. Use `RAGInput.filters` or the `filters` of a `run-eval` dataset
  item for a list.

- The pattern check is not hierarchy-aware: under `strict`, an
  `ORGANIZATION` pattern does not admit a `VENDOR` endpoint, which is why
  the mapping writes the subclass patterns out. An entity's stored type is
  still its most frequent one: stated twice as `VENDOR` and once as
  `ORGANIZATION`, it stays a vendor.

**Worked example.** `tests/fixtures/ontologies/procure.ttl` (`ex:` is
`https://example.org/procure#`) declares `ex:Organization` with the
subclasses `ex:Vendor` and `ex:Buyer`, and `ex:Person`; `ex:tradesWith`,
symmetric, Organization → Organization; `ex:supplies`, a sub-property of
`tradesWith`, Vendor → Buyer; `ex:suppliedBy`, the inverse of `supplies`;
`ex:employs`, Organization → Person; `ex:worksFor`, the inverse of
`employs`; and `ex:name`, a datatype property. With the default options it
maps to:

| Kind | Types (description) |
|---|---|
| Entity types | `ORGANIZATION`; `BUYER` ("Kind of ORGANIZATION."); `VENDOR` ("An organization that sells goods. Kind of ORGANIZATION."); `PERSON` |
| Relationship types | `EMPLOYS` ("Inverse of WORKS_FOR."); `SUPPLIED_BY` ("Inverse of SUPPLIES."); `TRADES_WITH`; `SUPPLIES` ("Narrower than TRADES_WITH. Inverse of SUPPLIED_BY."); `WORKS_FOR` ("Inverse of EMPLOYS.") |

and 17 patterns:

- `EMPLOYS`: `ORGANIZATION`, `BUYER` and `VENDOR` → `PERSON` (3, pushed
  down from Organization).
- `SUPPLIED_BY`: `BUYER` → `VENDOR` (1, from the inverse's domain and
  range).
- `TRADES_WITH`: every pair of `ORGANIZATION`, `BUYER` and `VENDOR` (9,
  symmetric).
- `SUPPLIES`: `VENDOR` → `BUYER` (1).
- `WORKS_FOR`: `PERSON` → `ORGANIZATION`, `BUYER` and `VENDOR` (3).

`ex:name` is skipped. Under `strict`, `VENDOR -SUPPLIES-> BUYER` passes and
`BUYER -SUPPLIES-> VENDOR` is dropped. Narrowing the selection:

```yaml
    schema_source:
      kind: owl
      path: ontology/procure.ttl
      options:
        classes: ["ex:Organization"]   # ORGANIZATION, BUYER, VENDOR
        max_depth: 0                   # ORGANIZATION only
```

- `classes: ["ex:Organization"]` alone maps `ORGANIZATION`, `BUYER` and
  `VENDOR`; `EMPLOYS` and `WORKS_FOR` are dropped (`PERSON` is not
  selected), leaving 11 patterns.
- With `max_depth: 0` only `ORGANIZATION` is left, and `SUPPLIES` and
  `SUPPLIED_BY` are lifted to `ORGANIZATION -> ORGANIZATION` (3 patterns),
  which under `strict` admits either direction between two organizations.

**Prompt budget and caps.** The types and patterns are in the prompt of
every graph extraction call: about 5 tokens per type plus its description
(a 300-character one is about 75) and about 13 per pattern. At the default
caps that is up to about 3,300 tokens of patterns, plus about 10,000 for
128 types with 300-character descriptions. Patterns multiply: a property between m
source and n target classes gives m × n, so a broad symmetric property
dominates. A selection over a cap fails the load (exit 2) instead of being
cut, since cutting would change the prompt arbitrarily, and the message
says what to narrow; for patterns it names the five properties that yield
the most:

```text
schema_source.options: the selection yields 17 patterns, more than max_patterns (10); the most come from https://example.org/procure#tradesWith (9), ... Narrow classes or properties, exclude some, set patterns: none, or raise max_patterns
```

Narrow with `namespaces`, `classes`, `max_depth` and the `exclude_*`
options, set `patterns: none` (types only; `strict` then checks types but
no pattern), or raise the cap knowing the prompt grows.

<a id="ontology-load-cost"></a>**Load cost.** The files are parsed into memory once per process, at
start-up (`run-rag`: at the first filtered query), and the mapping then
runs in time and memory about linear in the ontology's size, however deep
its hierarchy. Parsing dominates: a 14 MB Turtle file of 110,000 classes
and 500 properties took about 45 s and 1.2 GB peak memory (parse 30 s,
reading the classes and properties 10 s, mapping 5 s), that is about 3 s
and 85 MB per MB of Turtle; a 20,000-deep `subClassOf` chain maps in
under a second. At the default `max_bytes` (64 MiB) that would be about
5 GB, within the default 8 GiB task but not alongside a large ingestion,
so lower `max_bytes` to what you load, give the task memory to match, or
map a module of the ontology (its own files) instead of all of it.

**Korean labels.** For an ontology labelled in Korean:

```yaml
    schema_source:
      kind: owl
      path: ontology/procure.ttl
      options:
        label_source: label        # rdfs:label only
        languages: [ko]            # Korean first; then untagged, then any other tag
        label_case: preserve       # labels as written
        labels:
          "ex:suppliedBy": 공급받는다  # no Korean label in the ontology
```

maps the example to `사람`, `조직`, `공급업체`, `구매자` and `거래한다`,
`공급한다`, `고용한다`, `공급받는다`, `근무한다`, with the same 17 patterns.

- A term without a Korean label takes an untagged one, then another
  language, then its local name (percent-decoded, so a Hangul IRI gives a
  Hangul label). Under `preserve` that local name stays as written
  (`suppliedBy`; `upper_snake` would give `SUPPLIED_BY`), so give such
  terms a `labels` entry.
- Hangul has no case, so for a Korean label the two cases differ only in
  the separator (`공급 업체` → `공급_업체` or `공급 업체`), which label
  matching ignores; they differ for Latin labels and local names.
- Descriptions follow `languages` too (here the English comment, the only
  one). The hierarchy hints are always English: "Kind of 조직."

<a id="ontology-what-a-change-re-runs"></a>**What a change re-runs.** The mapped schema is keyed, not the files, so
an incremental run reacts to what the mapping yields
([Changing the schema](#c-extraction-schema)):

- **Re-extracts**: a different label, description or pattern, or a new
  order of the types. That is an edit to a selected term's label or
  comment, a class or property entering or leaving the selection, and an
  option change that alters any of these (`hierarchy_hints` included). Settle
  the options before the first full ingest; if the ontology's editors
  relabel often, use `label_source: local_name` or pin `labels`.
- **Re-annotates only**: an IRI that changes while the labels stay
  (`ex:Vendor` renamed `ex:Supplier`, its label kept), including an
  equivalent class with a smaller IRI, which then names the type
  (`ex:Supplier owl:equivalentClass ex:Vendor` makes `type_iri`
  `ex:Supplier`). Only while the label stays, though: a merged type takes
  the best label of all its members, or under `label_source: local_name`
  (or with no label) the local name of its smallest IRI, so a new
  equivalent class with a label of its own, or with a smaller IRI under
  `local_name`, can relabel the type and re-extract. Pin such a type in
  `labels`.
- **Nothing**: re-serialising (Turtle to RDF/XML), moving or renaming the
  files (with absolute IRIs, below), `owl:versionInfo`, `rdfs:seeAlso`,
  a datatype property's label or comment, edits to terms outside the
  selection that the closure does not use, and adding statements the
  closure already infers. With `on_collision: qualify`, bind every
  colliding namespace in `options.prefixes`: otherwise the suffix may come
  from the prefixes the files declare, which re-serialising can rename or
  drop (turning `BANK_EX` into a hash), and a relative IRI's namespace,
  and so its hash, follows the file's location.

**Declare a base IRI.** Write absolute IRIs, or declare a base: `@base`
in Turtle and TriG, `xml:base` in RDF/XML, `@base` in a JSON-LD context. A
relative IRI (`<#Vendor>`) resolves against the file's own location
(`file:///app/ontology/procure.ttl#Vendor`, or its `s3://` URL), so moving
the file, between paths or between a path and S3, or reading
it from another path renames every such term: their `type_iri` and
`predicate_iri` change and the next run re-annotates every stored item.
Two files of a directory that both write `<#Vendor>` name two classes. The
load warns `N terms have IRIs relative to their file`.

**Not followed, not read.**

- `owl:imports` is not followed. Put the imported ontology's files in the
  same directory (a directory is one ontology); an import that no file
  declares is logged once as a warning.
- Nothing is fetched beyond `path` itself. A JSON-LD `@context` or
  `@import` naming a document
  is refused (inline the context, or convert the file to Turtle); XML
  external entities are not resolved (internal ones are expanded).
- OWL/XML (`.owx`, or an RDF/XML suffix whose root element is
  `owl:Ontology`), OWL functional (`.ofn`) and Manchester (`.omn`) syntax,
  OBO (`.obo`) and Notation3 (`.n3`) are refused. Convert them first:
  `robot convert --input onto.owx --output onto.ttl` (ROBOT), or Protégé's
  *File > Save as > Turtle*.
- Shapes the parsers take quadratic time over are refused before parsing,
  naming the limit: RDF/XML whose entities expand past the XML parser's
  amplification limit, an RDF/XML element of more than 10,000 pieces of
  text (`MAX_XML_TEXT_CHUNKS`: each line break and entity reference is
  one), more than 1,000 elements inside `rdf:parseType="Literal"` values
  of a file (`MAX_XML_LITERAL_ELEMENTS`), and an N-Triples or N-Quads line
  over 256 KiB (`MAX_LINE_BYTES`). Convert such a file to Turtle. An IRI
  with whitespace, which RDF/XML lets through, and a cyclic `rdf:List`
  fail the load too. A parse error names the line and what is wrong,
  never the file's text.

<a id="ontology-preview"></a>**Preview.** What extraction will use, with no model call:

```python
import yaml

from unified_kg_rag import get_config
from unified_kg_rag.application.bootstrap import resolve_extraction_schema

extraction = resolve_extraction_schema(get_config("config.yaml")).processing.graph_extraction
print(yaml.safe_dump(
    extraction.model_dump(
        mode="json",
        include={"entity_types", "relationship_types", "relationship_patterns"},
        exclude_none=True,
    ),
    allow_unicode=True,
    sort_keys=False,
))
```

Each load also logs one INFO line, `Ontology '<path>' (N files): ... entity
types, ... relationship types and ... patterns from ... classes and ...
properties; ... properties dropped, ... ends lifted, ...`; DEBUG names each
dropped, lifted, merged, skipped and qualified term. A property the
reader cannot type (only a domain, range or `owl:FunctionalProperty`)
gets its own INFO line, `Ontology: N properties not read ...`, naming the
first five.

**Freezing the mapping.** Two ways to stop ontology edits from reaching
the graph:

- Save the preview's output as `schema.yaml` and load it with
  `schema_source: {kind: file, path: schema.yaml}`. It is the same schema,
  so nothing re-extracts, and the extra is no longer needed.
- `run-prompt-tuning --schema-out draft.yaml` over the `kind: owl` config
  ([Drafting a schema](#c-extraction-schema)) writes the ontology's types,
  patterns and IRIs plus the relationship types the sample states that
  the ontology lacks, and its header names them and counts the patterns
  it adds for them. A type the ontology declares keeps exactly the
  ontology's patterns (none under `patterns: none`), so with nothing added
  the draft loads as the same schema and nothing re-extracts. Add the new
  types to the ontology (keep `kind: owl`), or review the draft and load
  it with `kind: file` (the added types change the schema, so that
  re-extracts).

### D. Seed graphs (bring your own knowledge graph)

A seed graph is a knowledge graph you already have (a supplier master, a
product catalog, the instances of an ontology) merged into the graph that
extraction builds from the documents. Each seed is split into virtual
documents of the run's own corpus (`seed_graphs/<name>/<k>-of-<n>.jsonl`),
so incremental runs, removal, resume and the stores treat it like any
file. No model reads a seed: its statements are parsed, never chunked by a
model, translated, extracted, gleaned or mined for claims.

Use one when the documents name entities a curated source already
describes and answers should use its descriptions, properties and IRIs,
or when the graph needs edges the text does not state (an org chart, who
supplies whom) for local search and graph expansion to walk. It does not
override the documents ([*Precedence*](#seed-precedence), below), and it brings instances, not
types: the vocabulary of types is the extraction schema (C).

**Configure.** Seeds are declared under `processing.seed_graphs`, a list:

```yaml
processing:
  graph_extraction:
    entity_types:
      - {label: VENDOR, description: "Companies that sell goods", iri: "https://example.org/procure#Vendor"}
      - {label: BUYER, description: "Companies that buy goods", iri: "https://example.org/procure#Buyer"}
    relationship_types:
      - {label: SUPPLIES, iri: "https://example.org/procure#supplies"}
    relationship_patterns:
      - "VENDOR -SUPPLIES-> BUYER"
    schema_mode: align
  seed_graphs:
    - name: procure
      kind: jsonl
      path: seed/procure.jsonl         # relative to this file
      filters: {department: purchasing}
      description_properties: [country]
```

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Names the shard documents and the `seed_graph` document filter of everything the seed states. Up to 64 lowercase letters, digits, `-` and `_`, starting with a letter or digit; unique. Renaming it imports the seed again |
| `kind` | required | `jsonl`, `csv`, `rdf` (with the `ontology` extra), or a kind a deployment registers ([Design Doc §15](./design.md#15-extension-guide)) |
| `path` | required | A local file or directory, relative to the config file (also when set from the environment; a `file://` URL works too), an S3 object (`s3://bucket/key`) or an S3 prefix read as a directory (`s3://bucket/seed/`, the trailing `/` required), read with the run's AWS session. Other URLs fail |
| `options` | `{}` | Settings of the kind (below). An unknown option fails the import |
| `secret_arn` | `null` | A Secrets Manager secret a registered service source reads; `jsonl`, `csv` and `rdf` refuse one |
| `filters` | `{}` | Document filters of the seed's items, as a document's metadata filters (string, number or boolean values). `seed_graph: <name>` is always added, so the key `seed_graph` is reserved |
| `on_name_collision` | `error` | Two nodes of the seed that are one entity by name: `error`, `merge` or `qualify` ([*Precedence*](#seed-precedence)) |
| `description_properties` | `[]` | Property keys written as `key: value` lines after the description, so answers can use them |
| `confidence` | `1.0` | Confidence of the seed's statements; a node's own `confidence` wins |
| `shards` | `16` (1 to 1,024) | Virtual documents the seed is split into, by a stable hash of each node's id |
| `max_items_per_unit` | `50` (1 to 500) | Node and edge statements per text unit, the unit of lineage |
| `max_nodes` | `100000` | Fail above this many nodes |
| `max_edges` | `500000` | Fail above this many edges |
| `max_bytes` | `268435456` (256 MiB); `33554432` (32 MiB) for `rdf` | Fail when the files hold more bytes together (checked before they are read, and while reading). An RDF parser holds the whole graph before `max_nodes` is checked, from about 25 (N-Triples) up to about 150 (dense Turtle) times the file size in memory (32 MiB: up to about 5 GB): size the task before raising it for `rdf` |
| `max_property_keys` | `64` (0 to 512) | Fail when the seed uses more distinct property keys (each is a key of `attributes.seed_properties`) |
| `max_description_chars` | `1000` (at least 50) | Cut descriptions here, at a word boundary with "…", so an item only the seed states never needs the description summarizer |

- The caps fail the import; nothing is truncated, as a cut would change the
  graph arbitrarily. A shard over 2,000 text units fails too, naming its
  largest node: raise `shards`, or, for a hub (a node and its outgoing
  edges stay in one shard, so more shards do not split it),
  `max_items_per_unit`. So does a JSONL line or CSV row over 1 MiB.
- Every name, IRI and type is bounded so that a store can index it as a
  keyword (at most 32,766 UTF-8 bytes): a node `name` (a qualified one
  included) holds at most 1,000 characters, an `iri`, `type_iri` or
  `predicate_iri` 4,000, a `type` 256. Text that is not valid Unicode (a
  JSON `\ud800` escape: a lone surrogate) fails in any field, naming the
  file and line, or the RDF subject.
- Every kind takes two options: `dangling` (an edge naming an unknown
  node: `error`, the default, or `skip`, which leaves it out and counts
  it) and `qualifier_property` (for `on_name_collision: qualify`).
- From the environment the list is JSON:
  `UKR__PROCESSING__SEED_GRAPHS='[{"name": "procure", "kind": "jsonl", "path": "seed/procure.jsonl"}]'`.
- `run-ingestion` loads every seed once at start-up, after the extraction
  schema and before any model is built: a seed that cannot be imported
  exits 2, naming the seed and, for a bad record, the file and line, as
  does a configured kind whose extra is missing (`rdf`). The other CLIs
  do not load seeds (`run-prompt-tuning` profiles the documents only) and
  need no seed kind's extra: a query image answers over an index an `rdf`
  seed was imported into without the `ontology` extra.
- The corpus may be empty when seeds are configured: an empty
  `--source-directory` is a seed-only run ([*Lifecycle*](#seed-lifecycle)) for a source scope
  the registry holds no documents of (a new one, or a seed-only scope).
  For a scope with documents (under any spelling of it: `s3://bkt/corpus`
  and `s3://bkt/corpus/` are one scope) an empty corpus is far likelier a
  volume not mounted yet or an S3 sync that copied nothing, so the run
  fails at `document_loading` before it writes (`No source document
  reached this run, only its seed graphs, but the doc-status registry
  holds 2 documents of source scope ...`, exit 1) rather than delete them.
  With the registry off, see [*Lifecycle*](#seed-lifecycle).
- **On AWS** the seed files must be in the image next to the config, or in
  S3 where the ingestion task role reads
  ([Operator Runbook: Seed graphs in the image or S3](./operations.md#seed-graphs-in-the-image-or-s3)).
  Keep them out of the ingestion source prefix: a `.csv` there is also
  parsed as a document.

**JSONL** (`kind: jsonl`): one file (any name), or a directory whose
`*.jsonl` files are read recursively in sorted order (names starting with
`.` skipped). Each non-blank line is a JSON object whose `"kind"` is
`node` or `edge`:

```json
{"kind": "node", "id": "v1", "name": "Acme Supplies", "type": "VENDOR", "iri": "https://example.org/procure#acme", "description": "Industrial valve vendor.", "properties": {"country": "DE"}}
{"kind": "node", "id": "b1", "name": "Globex Buyer", "type_iri": "https://example.org/procure#Buyer"}
{"kind": "edge", "source": "v1", "target": "b1", "type": "SUPPLIES", "description": "Acme supplies Globex with valves.", "weight": 2}
```

- A node takes `id` (required, unique), `name` (default: the id), `type`
  (a label), `type_iri` (a class IRI the schema maps), `description`,
  `iri` (the node's own IRI), `confidence` and `properties`.
- An edge takes `source` and `target` (node ids), `type` or a
  `predicate_iri` a relationship type declares (one of them is required),
  `description`, `weight` (default 1) and `properties`.
- Property values are strings, numbers or booleans; `null` is dropped. An
  unknown key, a repeated node id, a non-scalar property, `NaN`, a line
  that is not a JSON object and one nested too deeply to decode fail the
  import, naming the file and line.
- A self-loop is dropped (a full build draws none). Edges with the same
  endpoints and type merge: weights add, descriptions are unioned.

**CSV** (`kind: csv`): `path` is a directory (or an S3 prefix ending in
`/`) holding `nodes.csv` and `edges.csv`. The first row names the columns;
the node and edge keys above are columns, and any other column is a
string property:

```text
id,name,type,type_iri,description,iri,confidence,country
v1,Acme Supplies,VENDOR,,Industrial valve vendor.,https://example.org/procure#acme,,DE
b1,Globex Buyer,,https://example.org/procure#Buyer,,,,
```

```text
source,target,type,predicate_iri,description,weight
v1,b1,SUPPLIES,,Acme supplies Globex with valves.,2
```

```yaml
  seed_graphs:
    - name: procure
      kind: csv
      path: seed/procure               # a directory holding nodes.csv and edges.csv
      filters: {department: purchasing}
      description_properties: [country]
      options:
        property_columns: [country]
```

These files state the same graph as the JSONL example. An empty cell is an
unset value. `options.nodes` and `options.edges` name other files in the
directory (`edges: null` reads nodes only) and `options.property_columns`
keeps only the listed extra columns. UTF-8 with an optional byte-order
mark; a row with more cells than columns, a repeated column, a missing
`id` (`source`, `target`) column and a field over Python's CSV limit
(131,072 characters) fail the import.

**RDF** (`kind: rdf`): the instances of an ontology (an A-Box) in Turtle,
N-Triples, N-Quads, TriG, RDF/XML or JSON-LD. It needs the `ontology` extra
(the image too: `--build-arg UV_EXTRAS="--extra ontology"`), and reads the
files exactly as the `owl` schema source does (a file, a directory, an
`s3://` object or prefix; `options.format`; nothing fetched, `owl:imports`
not followed, the same refusals: [Not followed, not read](#from-an-owlrdfs-ontology)),
within the seed's `max_bytes` (32 MiB unless set: the parsed graph is held
in memory, up to about 150 times the file size, about 5 GB). With the procure ontology as the
schema:

```yaml
processing:
  graph_extraction:
    schema_source: {kind: owl, path: ontology/procure.ttl}
    schema_mode: strict
  seed_graphs:
    - name: procure
      kind: rdf
      path: seed/procure-abox.ttl
      options:
        languages: [en, ""]            # the default; [ko, en, ""] names Acme in Korean
```

```turtle
@prefix ex: <https://example.org/procure#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:acme a ex:Organization, ex:Vendor ;
    rdfs:label "Acme Supplies"@en, "애크미 서플라이"@ko ;
    skos:definition "Industrial valve vendor."@en ;
    ex:country "DE" ;
    ex:rating "4"^^xsd:integer ;
    ex:supplies ex:globex ;
    ex:headquarters ex:berlin .

ex:globex a ex:Buyer ;
    rdfs:label "Globex Buyer" .

ex:berlin rdfs:label "Berlin"@en .
```

gives two nodes and one edge: `Acme Supplies` (`VENDOR`, the most specific
of its two classes, with the properties `ex:country` = `DE` and
`ex:rating` = 4), `Globex Buyer` (`BUYER`) and `Acme Supplies -SUPPLIES->
Globex Buyer` with `predicate_iri` `ex:supplies`. `ex:berlin` has no type,
so it is no instance and the `ex:headquarters` edge is skipped (counted as
`objects_not_instances`).

- **Instances** are named subjects with an `rdf:type` outside the RDF,
  RDFS, OWL, XSD and SKOS vocabularies that are not classes or properties
  themselves. A node's `id` and `iri` are its IRI.
- **Names** come from `name_properties` (default `skos:prefLabel`,
  `rdfs:label`, `schema:name`) in the order of `languages` (default `[en,
  ""]`, `""` = untagged; any other tag after them), else the IRI's
  percent-decoded local name as written. **Descriptions** likewise from
  `description_properties` (default `skos:definition`, `rdfs:comment`,
  `schema:description`).
- **Types.** Each `rdf:type` is looked up among the schema's entity type
  IRIs; when several map, the one the schema declares last wins. An `owl`
  schema lists a class before its subclasses, so that is the most specific
  class (`VENDOR` above, not `ORGANIZATION`). A class no type declares
  keeps its IRI as `type_iri` with the label `labels` gives it, else its
  local name in `UPPER_SNAKE`: `strict` then drops the node, `align` and
  `off` keep it.
- **Edges** are triples whose object is an instance, typed by the
  relationship type whose IRI is the predicate, else `labels`, else the
  predicate's local name in `UPPER_SNAKE`. An object that is not an
  instance is skipped and counted (`dangling` does not apply).
- **Properties** are the other literals, keyed by CURIE through
  `options.prefixes`, the well-known prefixes, then the files' own
  bindings, and by the full IRI when none binds the namespace (N-Triples
  binds none, so its keys are full IRIs unless `options.prefixes` binds
  them). Typed numbers and booleans become numbers and booleans. A
  predicate with several values keeps the preferred string by language,
  else the smallest value. `properties` keeps only the listed predicates,
  `exclude_properties` drops some.
- **Not read**: blank nodes, reified statements, RDF-star and
  `owl:sameAs` are skipped and counted, never followed. Nothing is
  inferred: no inverse, symmetric or sub-property edge, no type a
  subclass statement implies. One INFO line per seed gives the counts.
- **IRIs relative to their file** (no `@base`) are warned about: moving
  the file renames the nodes, which then re-import.

| Option | Default | Meaning |
|---|---|---|
| `prefixes` | `{}` | Prefix → namespace for the CURIEs of these options and of the property keys |
| `name_properties` | `skos:prefLabel`, `rdfs:label`, `schema:name` (`https` and `http`) | Literals a name is read from |
| `description_properties` | `skos:definition`, `rdfs:comment`, `schema:description` (`https` and `http`) | Literals a description is read from |
| `languages` | `[en, ""]` | Preferred language tags of names, descriptions and string properties |
| `labels` | `{}` | Class or property (IRI or CURIE) no schema type declares → the type label to use |
| `properties` | `null`: every other literal | The literal predicates kept as properties |
| `exclude_properties` | `[]` | Literal predicates never kept |
| `format` | `auto` | `auto` (by suffix), `turtle`, `xml`, `json-ld`, `nt`, `nquads` or `trig` |

**How seeds meet the schema.** Each seed is converted against the run's
resolved extraction schema (C), the whole seed at once, so a pattern check
always sees both endpoint types. Under `on_name_collision: qualify`, nodes
sharing a name are checked as the distinct entities they become, each by
its own type; under `merge`, as the one entity, as a full build sees it:

- A `type_iri` (or `predicate_iri`) names the declared type with that
  `iri`; one no type declares is counted (`nodes_unmapped_type_iri`) and
  the `type` label is used instead. A `type` label is written as
  extraction writes it (`vendor` becomes `VENDOR` once relationship types
  are declared).
- `off` checks nothing. `align` keeps everything and counts what falls
  outside the schema. `strict` drops off-schema nodes with their edges,
  edges of undeclared types and edges a pattern does not allow. Adding a
  `DISTRIBUTOR` node with a `SUPPLIES` edge, and a `BUYER -SUPPLIES->
  VENDOR` edge, to the JSONL example counts `entities_off_schema=1` and
  `relationships_off_pattern=1` under `align`, and under `strict` also
  drops the node and both edges.
- The counts are in the INFO line `Seed graph 'procure': ...`, with the
  counter names extraction uses. Seed items get `in_schema`, `type_iri`
  and `predicate_iri` like extracted ones; `iri` is the node's own IRI.

<a id="seed-precedence"></a>**Precedence: documents decide.** A seed node and a document mention whose
names are one entity (equal after normalization, or a legal-form variant,
as extraction merges names) become one entity:

| What | Rule |
|---|---|
| Type, name, confidence | The documents' statements decide whenever any states the entity. A seed's fill only a gap: an entity no document states, or one documents name only as a relationship endpoint, with no type |
| Description | Unioned, document lines first, the seed's lines after them |
| Properties | Kept under `attributes.seed_properties`, never over an extracted attribute. Stored, not indexed: any key and any string, number or boolean value can be written, and no filter or search matches them |
| IRI | The smallest one stated; an entity stated with several is counted (`entities_with_conflicting_iris`, INFO) |
| Edges | Added. A seed edge and a document edge with the same endpoints and type are one relationship, their weights added |
| Ranks | Count document-backed edges only (an edge is seed-only when every text unit it cites is a seed's): a seed never makes an entity more important, and an entity only seed edges join has rank 0. Leiden still clusters every edge at its full weight. `indexing.neptune.entity_importance_source: degree` counts the edges at query time, seed edges included |
| Communities | A community whose members only seeds state is stored without a report (`communities_seed_only` in the stage metrics); a report on a mixed community lists the document-ranked entities first. On an incremental run a community counts as mixed when a stored member is document-backed, though the run's own graph states it only from a seed (a seed added beside unchanged documents) |

- **Homonyms.** Identity is the name: an unrelated seed node "Mercury" (a
  planet) and the documents' "Mercury" (an element) are one entity, which
  gains the seed's description line and IRI but keeps the documents'
  type. `on_name_collision` covers names within one seed only. Curate the
  names, or qualify them in the seed.
- **Spelling variants.** The rules hold per name. Names that resolve to
  one entity without being equal ("Acme" in a document, "Acme Inc" in the
  seed) still let the seed take part: where two document spellings tie,
  the seed's units can decide the name, and the variants' description
  lines interleave. With `processing.fuzzy_entity_resolution` on, a seed
  spelling can also join two document spellings.
- **`on_name_collision`**, for two nodes of one seed whose names are one
  entity but that share neither an id nor an IRI: `error` (the default)
  fails the import listing them (`1 names are shared by distinct nodes
  (ids v1, v2); set on_name_collision to merge or qualify`); `merge` makes
  them one entity (the smallest IRI kept); `qualify` appends
  ` (<qualifier>)` to every name but the one of the smallest id: the
  `qualifier_property` value, else the IRI's local name, else the id
  (`ACME supplies (v2)`). A qualified name no longer links to a document
  mention.

**Retrieval.** Seed facts reach answers through the entities and
relationships, in every strategy that reads them (vector and keyword
search, graph expansion over seed edges), never as text chunks: the seed's
text units are stored for their lineage, every text search leaves them
out, and `local` and `mix` spend no chunk slot on them. `global` and the
`drift` primer read community reports: a community whose members only
seeds state has none, but the report of a mixed community describes its
seed members too (their descriptions are in its prompt), so seed facts
reach those strategies through the communities documents also state.
`naive` reads text units only.

- **Citations.** A source a seed states carries `provenance` in its
  metadata, with no chunk (`chunk_id` is `None`):

  ```python
  {"kind": "seed_graph",          # or "document+seed_graph"
   "seed_graphs": ["procure"],
   "iri": "https://example.org/procure#acme"}
  ```

  A source no seed states has no `provenance` key.
  `local` (and `drift` and `auto`, which run it) reports the graph's copy
  of an entity the entities index matched, which takes that hit's document
  filters and so names its seed graphs; an entity reached only by a graph
  walk has `"seed_graphs": []`, as graph vertices hold no document
  filters. Its metadata `source_kinds` is a list in every store.
- **Filters** ([Filters](#filters)):
  - `source_kinds` (entities, relationships, Neptune entity vertices):
    `seed_graph` keeps what a seed states, `document` what a document
    states (also an entity both state, with its seed description lines);
    a list matches either. Any case. A community report carries no source
    kind and summarizes every member of its community, seed-only ones
    included, so under a filter that keeps one kind `local`, `drift` and
    `simple` read no report (DRIFT then runs one local search on the
    question itself; `simple` sweeps every other index; `auto` routes to one
    of these) and `global`, which reads reports only, refuses the key.
    `mix` and `hybrid` read no report, and `naive`, which reads text chunks
    only, refuses the key too.
  - `iri` (entities, Neptune entity vertices): the exact IRI.
  - `attr_seed_graph: procure` keeps what that seed states, through the
    document filters every index carries. A tenant filter such as
    `attr_department: legal` excludes what only a seed states unless the
    seed's `filters` declare the key, as for documents.
  - Seed property keys are not filterable: the vector store keeps
    `attributes.seed_properties` unmapped, so a property of any type,
    length or key name never fails a write. To filter on a value, put it in
    the seed's `filters` (one value for the whole seed), or list the key in
    `description_properties` so text and vector search see it.

```python
RAGInput(query="...", search_strategy="local", filters={"source_kinds": "document"})
RAGInput(query="...", search_strategy="mix", filters={"attr_seed_graph": "procure"})
```

**What re-indexes.** The converted statements are keyed, not the files:

| Change | Next incremental run |
|---|---|
| Edit a node or edge | Imports again only the shards whose statements changed (a removal and an add, no model call). No document is extracted again |
| Change `shards` | Imports the whole seed again: every shard path changes |
| Rename the seed, change `filters` | Imports the whole seed again: every shard names them |
| Change `description_properties`, `confidence`, `max_description_chars`, `max_items_per_unit` or `on_name_collision` | Imports again the shards whose statements change |
| A schema change that changes the converted statements (a label, `schema_mode`, an IRI a `type_iri`, `predicate_iri` or RDF class maps through) | Imports again the affected shards (and extracts the documents again, as for any schema change) |
| A model or extraction setting | Extracts the documents again; the seed's shards stay unchanged, as no setting changes what they state |
| Move or re-serialise the files, change a cap the seed fits | Nothing |

A resumed run whose seed changed since the crash recomputes its stages:
the seeds' content is part of every stage's cache key.

<a id="seed-lifecycle"></a>**Lifecycle.** Seeds need the doc-status registry
(`providers.doc_status.enabled`) for these; without it every run is a full
build of the corpus and the configured seeds, and nothing tells an empty
corpus from a missing one. So with the registry off, an empty corpus
builds from the seeds alone without asking only while the index holds no
entity, relationship or text unit yet (a first seed-only build: nothing
to lose).
Once it holds some, the run asks first (`... rebuild the index 'default'
from the seed graphs alone, replacing what it holds; continue?`), as a
reset does below: `--confirm-reset`, `UKR_CONFIRM_RESET=1` or `--yes`
answers yes, and a no, or no terminal, stops the run before it writes
(exit 1). A store that cannot tell, or cannot be read, counts as holding
items.

- **Removal.** Drop the seed from the config: the next incremental run
  deletes its shards, so what only it stated goes, and entities documents
  state lose its description lines, properties, IRIs and edges.
- **Reset.** `indexing.reset: true` rebuilds the namespace from the run's
  corpus, its configured seeds included. The run asks first when the
  reset would drop what is easy to lose unawares: another source scope's
  seed documents (a seed-only scope, below), or, from a corpus of seeds
  alone, another scope's documents (a reset of a seed-only scope rebuilds
  the whole namespace from its seeds). `run-ingestion` asks on the
  terminal; `--confirm-reset` (or `UKR_CONFIRM_RESET=1`) answers yes to
  this question only, `--yes` to every question. A no, or no terminal,
  stops the run before it writes (`Pipeline stopped before it wrote
  anything: ...`, exit 1). Two resets from an empty corpus fail, whatever
  the answer: of a scope that holds documents (above), and of a scope that
  holds no record at all while other scopes hold documents (`... holds no
  record of source scope ...`): a new, moved or mistyped prefix, not a
  seed-only scope, whose earlier runs recorded its seeds. To rebuild a
  namespace from seeds alone, run the seed-only scope once without
  `indexing.reset`, then reset it. From Python, pass
  `DataIngestionPipeline(..., confirm=callable)`; without it, or on a no,
  `run` raises `ConfirmationDeclinedError` (exported by `unified_kg_rag`).
- **A seed-only scope.** A run with seeds, an empty `--source-directory`
  and a fixed `processing.document_parsing.source_scope` (say
  `seeds-procure`) keeps a seed in a scope of its own, with no model call.
  Other scopes' runs on the namespace leave it alone, except a reset.
- **One seed, two corpora.** A seed belongs to the scope of the run that
  configures it. Configured on two scopes of one namespace, it is imported
  twice: the same entities, but its edge weights count twice. Configure a
  shared seed on one scope, or give it a seed-only scope. A scope's first
  run warns when another scope already holds a seed of the same name
  (`Seed graph 'procure' is already imported by source scope(s) ...`): a
  moved or mistyped source directory looks the same.

<a id="seed-cost-and-limits"></a>**Cost and limits.**

- No extraction call (extraction, gleaning, claims, translation) reads
  a seed's text units, but seed descriptions do reach a model in three
  places: the description summarizer, for an entity documents also state
  whose merged description outgrows its threshold; the report of a
  community documents also state, which describes its seed members; and
  the answer prompt, whose context holds the retrieved entities and
  relationships. Write seed descriptions as data a model may read. A
  community only seeds state gets no report, and an item only a seed
  states is cut to `max_description_chars`, so it never needs the
  summarizer.
- Its items are embedded: two vectors per node (name and description),
  one per edge (with
  `indexing.opensearch.build_relationship_vector_index`, the default) and
  one per text unit, about 2 × nodes + edges + (nodes + edges) / 50 at the
  defaults. The 100,000-node, 500,000-edge caps are about 712,000
  embeddings. The whole seed is held in memory and in the stage cache.
- **Delta runs cluster the delta** (§5): a seed added by an incremental run
  is clustered on its own shards' entities until a reset re-clusters the
  graph. Those communities are reported when a stored member is
  document-backed, from the stored members' descriptions followed by the
  seed's lines, as the run indexes them: an edited seed's earlier lines
  are left out, and a description over
  `description_summarization.force_summary_threshold_tokens` is
  summarized first (once: the indexing reuses that summary). A full build
  summarizes all of a member's lines at once, so the incremental text is
  an equivalent summary rather than the same one; with summarization off
  both read the lines whole.
- **Translation.** Seed statements are never translated, while the
  documents' entities are extracted from the translated text: write seed
  names in `processing.translation.target_language` for them to link.
- **Local backend.** The SQLite graph and the embedded vector store hold
  the same fields and apply the same filters.

**Preview** what the seeds convert to, with no model call and no store:

```python
from unified_kg_rag import get_config
from unified_kg_rag.application.bootstrap import load_seed_graphs, resolve_extraction_schema

config = resolve_extraction_schema(get_config("config.yaml"))
for conversion in load_seed_graphs(config):
    print(conversion.name, conversion.stats.model_dump(exclude_defaults=True))
```

For the JSONL example this prints `procure {'nodes_read': 2, 'edges_read':
1, 'nodes': 2, 'edges': 1, 'units': 2, 'documents': 2}`.

---

## 10. Operations & troubleshooting

[IAM permissions](#iam-permissions) are in §1; error messages and their fix
are in [Operator Runbook: Common errors](./operations.md#common-errors).

### CLI exit codes and JSON output

Every CLI (`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`) exits with:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The run failed: a failed stage or query (in `run-ingestion` also a `ConfigurationError` a stage meets, or an unreachable store), a breached `--max-failure-rate`, a failed `run-eval --fail-if` gate, a failed upload, nothing rendered |
| `2` | Usage or configuration error: a bad flag, a missing or invalid input (a `run-eval` dataset, a `run-visualization` data file, a `prepare-benchmark` data file), an invalid config file or override, an unset store endpoint, a missing extra the selected providers need (`local`) (all before any work), or a `ConfigurationError` found while running outside a stage, such as `run-ingestion`'s pre-run refusal before its first model call (the search engine refused to create an index the run needs, or an incremental run's live indices, a deletion-only run's included, were built for other embedding, language or registry settings) |
| `130` | Interrupted with Ctrl-C (SIGINT) |
| `143` | Stopped by SIGTERM (ECS `StopTask`, a Step Functions timeout); `run-ingestion` checkpoints its finished stages first, and `run-eval` uploads the finished runs' results to an `s3://` output (or keeps them locally and logs the path) |

A run stopped with `130` or `143` ends its process at once: model calls in flight are abandoned rather than waited for, the logs, tracing and stdout are flushed first, and a process pool's workers are ended with it. A SIGTERM or Ctrl-C cancels a query CLI's work at its next await, so its cleanup (closing store connections, `run-eval`'s upload) runs (at the interactive prompt a SIGTERM ends the session at once and a Ctrl-C the conversation); a query CLI stopped by a SIGTERM, or by a Ctrl-C while its event loop runs, that is still running 100 seconds later is ended then (inside the 120 s StopTimeout of the app container), with the status document if none was printed, and a second signal ends it at once. `run-ingestion` stops at a SIGTERM or Ctrl-C after the checkpoint or metadata write in progress instead, bounded by the StopTimeout. A CLI that finished ends its process with its exit code as soon as its exit handlers ran, so a signal while it exits cannot turn `0`, `1` or `2` into `-15` or `-2`.

Flags must be spelled out in full: a prefix such as `--output` is rejected.

`--output-format json` prints one JSON document to stdout and moves the
banner, progress and logs to stderr, so the output pipes into `jq` (not with
`--metrics-sink cloudwatch`, which also writes to stdout): the answer or
search results (`run-rag`), the pipeline id, status, the number of documents
the run recorded `FAILED` (`failed_documents`; a model outage that fails
every unit records none, one that fails some units records their documents,
retried by the next run without counting toward
`indexing.max_document_failures`), the number of text units a stage
failed on (`units_failed`: indexed without that stage's output and
extracted again by the next run; counted in a full build too, where no
registry records a document `FAILED` and `failed_documents` stays 0; the
run ends with a WARNING naming them by stage and cause), the number of documents failing
another consecutive run for an error not their own (`documents_retrying`:
an outage or a setting never excludes them; also warned of with each file
and its run count), the number of documents indexed without the claims of
some text units, whose claim extraction failed for an error of their own
(`documents_without_claims`: counted for none, also in the summary), the
number of documents left out because they failed
`indexing.max_document_failures` consecutive runs unchanged, at the same
extraction settings (a settings change starts the count over)
(`documents_excluded`, also in the summary), the model calls and tokens of
the stages the run executed (`token_usage`) and
per-stage results (`run-ingestion`), each run's summary, the comparison, the failure-budget
breaches and the failed gates (`gate_failures`) (`run-eval`), the written files (`run-visualization`), the output
path and profile (`run-prompt-tuning`), or the counts written
(`prepare-benchmark`). A run stopped by a
signal before it printed prints `{"status": "interrupted", "exit_code": 143}`
(`130` for Ctrl-C), and one that fails before it printed prints
`{"status": "failed", "exit_code": 1}` (or `2`) with the `error` it failed
with (an argparse error included), so stdout is always JSON. That holds
from the first moment: a SIGTERM while the CLI still loads, or after it
printed its result while the process exits, exits `143` too, and a Ctrl-C
while it loads `130` (the status document only while loading; a SIGINT the
parent ignores, as `nohup` does, stays ignored). A
`run-ingestion` result whose run a stage failed carries the same `error` and
`exit_code`; `run-ingestion` adds the `pipeline_id` to resume with (once the run resolved it) and, when a stopped
run's next run on the namespace must wait for writes possibly still in
flight (this run's, or those of the run whose expired lease it took over),
`next_run_writes_from` (UTC, ISO 8601).

### Large / multilingual / heterogeneous corpora

- **Large:** raise `processing.max_concurrency` as far as the Bedrock quota
  allows; keep `indexing.neptune.index_concurrency` at `1` unless you measured
  a gain (concurrent write batches conflict on shared vertices and retry);
  the doc-status registry
  (`providers.doc_status.enabled`) makes later runs pay only for changed
  documents. On Neptune, `indexing.neptune.vertex_ids: custom` looks vertices
  up by vertex id rather than by the `id` property. Switching an existing
  graph needs one full build ([Custom Neptune vertex ids](operations.md#custom-neptune-vertex-ids)).
  Graph expansion takes a hub's neighbours in id order
  (`indexing.neptune.deterministic_order: true`, the default), so a rebuild
  returns the same results, at the cost of reading every neighbour of each
  expanded hub (on Gremlin Server, 6.8 to 23.2 ms per query for
  2,500-neighbour hubs; unmeasured on Neptune). If hub-heavy queries are
  too slow, `false` takes the first neighbours Neptune returns instead.
- **Community reports:** a report's prompt packs the community's
  relationships edge by edge, each with its two entities (and their claims
  with claim extraction on), into
  `graph.community_detection.report_generation.max_report_context_tokens`
  (4000). MS GraphRAG uses 8000 with `max_cluster_size` 10: if reports of
  large communities cover too few relationships, raise the budget (more input
  tokens per report call) or lower `max_cluster_size` (more, smaller
  communities and report calls).
- **Multilingual:** see [Multilingual ingestion](#multilingual-ingestion).
- **Heterogeneous domains:** tune `entity_types` to the union of the domains,
  or give each its own [index suffix](#index-suffix).

### Cost notes

LLM calls dominate cost: extraction (one or more calls per chunk), gleaning
(up to `max_rounds` per chunk), community reports, claims (one call per text
unit, off by default) and one answer per query. Levers: Haiku-class models for
the mechanical roles (the default for chunking, translation, map-reduce and
summarization), fewer gleaning rounds, claims off, the caches and incremental
indexing.

Gleaning is the largest of these levers. `processing.gleaning.max_rounds`
stays 3 because 3 rounds found more of a corpus's graph than 1, for about
38% more ingestion cost, with no significant difference in the answers
([evidence](./concepts/defaults-and-evidence.md#gleaning-rounds)). Set `max_rounds: 1` (MS GraphRAG's
default) when ingestion cost matters more than graph recall.

---

## 11. Using the library from Python

The CLIs wrap two public classes. Import them, and everything else, from the
package root: the names in `unified_kg_rag.__all__` are the public API, other
module paths are internal and may change. `get_config` applies the same file
and environment layers as the CLIs (§2).

```python
from unified_kg_rag import (
    DataIngestionPipeline,
    GraphRAGChain,
    PipelineConfig,
    PipelineStageStatus,
    RAGInput,
    get_config,
)

config = get_config("config.yaml")

# Ingest. PipelineConfig carries the run-ingestion run options (pipeline_id,
# cache directory, S3 sync, enabled stages, force_rebuild). run() ingests the
# directory it is given: parsing, loading and the registry scope all use it.
pipeline = DataIngestionPipeline(config, PipelineConfig(pipeline_id="run-001"))
try:
    context = pipeline.run("./source")
    if context.status == PipelineStageStatus.FAILED:
        raise SystemExit("ingestion failed")
finally:
    pipeline.close()  # releases the indexers' Neptune/OpenSearch connections

# Query. One chain serves many queries, including concurrent ones.
chain = GraphRAGChain(config)
try:
    output = chain.invoke(RAGInput(query="How are Alice and Acme related?", search_strategy="mix"))
    print(output.answer)
    for source in output.sources:  # only what the answer model saw
        print(source)
finally:
    chain.close()
```

- A failed stage sets the returned `PipelineContext`'s `status` to `FAILED`
  rather than raising, so check it as above.
- The errors to tell apart are exported from the package root:
  `ConfigurationError` (a setting to fix; `StoreRequestRejectedError` among
  them), `InvalidFilterError`, `IndexNotFoundError`, `StoreConnectionError`,
  `NamespaceLeaseError`, `PipelineExecutionError` and `PipelineInterrupted`
  (a `KeyboardInterrupt`); all but the last are `GraphRAGException`s.
  `DataIngestionPipeline(...)` raises `SeedGraphError` (a
  `ConfigurationError`) for a seed graph it cannot import.
- `DataIngestionPipeline(..., confirm=callable)` answers the questions a run
  asks (`callable(question) -> bool`), today only a reset that would drop
  another source scope's seed graph documents; without it such a run fails
  ([Seed graphs](#d-seed-graphs-bring-your-own-knowledge-graph)).
  `DataIngestionPipeline.run` raises `ConfigurationError`,
  `StoreConnectionError` and `NamespaceLeaseError` as they are, and wraps
  any other crash outside a stage in `PipelineExecutionError`. A
  model client's own transport error (`httpx.ConnectError` from a local
  model server, say) is raised as is.
- `RAGInput` takes the `run-rag` options as fields (`search_strategy`,
  `search_type`, `top_k`, `suffix` ([index suffix](#index-suffix)),
  `filters`, `conversation_id`, `use_memory`, `target_language`); a plain dict
  works too.
- `chain.search(...)` (`await chain.asearch(...)`) retrieves without
  answering and returns a `SearchOutput` (`search_results`,
  `processed_query`, `metadata`).
- In async code use `await chain.ainvoke(...)` and `await chain.aclose()`, and
  size the loop's executor once at startup:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`.
- Always `close()` / `aclose()` what you build: both classes open connections
  on first use and keep them until garbage collection otherwise. A sync call
  still running on another thread when the chain closes raises
  `StoreConnectionError`; a later call reconnects.
- Importing the package configures no logging handler;
  `unified_kg_rag.setup_logging(config)` gives the CLI's logging. Without it,
  structured fields (including `structlog.contextvars` bindings) reach your
  handlers as record attributes, a trailing `_` added to one named like a
  `LogRecord` attribute (`name`, `msg`, ...). `setup_logging` changes
  process-wide state, so call it only where the package owns logging: it
  replaces the structlog configuration, sets the root logger's level to
  `logging.level` and the `logging.library_levels` loggers' levels, and adds
  a stdout handler (plus the dated file with `logging.log_to_file`) to the
  root logger. Handlers already on the root logger stay: they receive every
  record too, so the console shows each line twice when one of them writes
  there, and for the package's records they get the plain message and the
  exception, not the structured fields. A repeated call replaces only the
  handlers it added.

Extensions import from the package root too: the ports (`LLMFactoryPort`,
`GraphIndexer`, `DocStatusPort`, ...), `Providers`, `register_provider`,
`BaseSearchStrategy` with `register_strategy`, `ParserFactory` and
`register_renderer`. Both constructors take injected backends, either as a
`providers=Providers(config, ...)` bundle or as individual arguments
(`doc_status=`, `vector_indexer=`, `model_factory=`, ...), not both; see
[Design Doc §15](./design.md#15-extension-guide). An indexer passed as
`vector_indexer=` / `graph_indexer=` stays yours to close: the pipeline never
closes it.

---

## 12. Limitations

- **Extraction can hallucinate** entities or relationships the text does not
  support, which then reach retrieval.
  `processing.graph_extraction.entity_grounding.enabled` (off by default)
  drops or down-weights items whose quoted `source_text` is not in the chunk.
- **Runs are not reproducible**: LLM outputs (extraction, reports, the
  `auto` router, answers) vary, so compare configurations over several runs.
  Given the same model responses and the same source scope, everything else
  is deterministic in any process, whatever its `PYTHONHASHSEED`. With the
  doc-status registry on, text-unit ids derive from the source scope, which
  defaults to the absolute source directory: the same corpus ingested from
  another directory gets other ids. Set
  `processing.document_parsing.source_scope` for ids that do not depend on
  where the corpus sits.
- **Reranking is regional** ([Required models](#required-models)); without it
  results keep the fusion order, as `metadata.rerank` shows
  ([Rerank status](#rerank-status)).
- **Shipped backends.** Models: Bedrock (default), Ollama or
  OpenAI-compatible servers; reranking: Bedrock or a sentence-transformers
  cross-encoder ([Local mode](#local-mode-no-aws)). Stores: Neptune (or a
  Gremlin Server) with OpenSearch, or the embedded SQLite stores (one host,
  about 100,000 documents per index). Others need a `register_provider`
  builder or custom adapters ([Design Doc §15](./design.md#15-extension-guide)).
- **No serving layer**: no HTTP API, end-user authentication or rate limiting.
- **Filters are not access control** (§4): separate tenants by index suffix.
- **A lighter ingestion limits the strategies.** Without community detection
  (`graph.community_detection.enabled: false`) `global` and `drift` have no
  reports; without the relationship vector index
  (`indexing.opensearch.build_relationship_vector_index: false`) `mix` and
  `hybrid` lose high-level keywords, `local` its relationship section, and
  graph expansion under a document filter keeps entities on the
  entities-index check alone, not on the relationships that reach them.
- **Incremental indexing approximates** (§5): delta runs append communities
  instead of re-clustering, and stored communities and reports keep an
  entity's old id after it took its current name's; a full rebuild
  (`indexing.reset: true`) refreshes both. With DynamoDB one document's chunk
  ids must fit one 400 KB item ([*Document size limit*](#document-size-limit)).
- **Concurrent runs need the registry**, whose write lease stops a second run
  on a namespace; without it (or through the direct `IncrementalIndexer` API)
  nothing locks, and two ingestions on one suffix can delete each other's
  index ([Concurrent runs](./operations.md#concurrent-runs)).
- **Isolating a document that kills the task takes runs**
  ([*Quarantined documents*](#quarantined-documents)): about `log2(n) + 1` for one culprit among `n` documents that
  died together, while the deferred ones wait.
- **Markdown and HTML** need the `unstructured` extra, which
  the container image leaves out by default.
- **Seed graphs link by name only** ([*Precedence*](#seed-precedence),
  [§9 D](#d-seed-graphs-bring-your-own-knowledge-graph)): a seed node and an
  unrelated document entity of the same name are one entity. IRIs are
  stored and filterable but are no identity, `owl:sameAs` is not followed,
  and nothing is inferred.
- **Changing the embedding model** or dimension needs `indexing.reset: true`;
  an incremental run refuses to write into another model's indices
  ([Workflows](#workflows)).

---

*See also: [README.md](../README.md) for the overview and quickstart, and
[docs/design.md](./design.md) for architecture and internals.*
