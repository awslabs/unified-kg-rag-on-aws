# Unified Knowledge Graph RAG on AWS — User Guide

> 🇰🇷 한국어판: [docs/user-guide.ko.md](./user-guide.ko.md)

This is the practical, how-to-use guide for **unified-kg-rag-on-aws** — an AWS-native
knowledge-graph RAG framework that builds knowledge graphs from large,
multilingual document corpora and answers questions over them. It reimplements
two retrieval methodologies on one stack: **Microsoft GraphRAG**
(community-summary) and **LightRAG** (dual-level keyword), selectable per query.

- For the *what / why* and a one-minute quickstart, see [README.md](../README.md).
- For *internals / architecture* (hexagonal layers, ports & adapters, the
  dependency rule), see [docs/design.md](./design.md).

The six console entry points (defined as `pyproject` scripts) are:

| Script | Module | Purpose |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | Build / update the knowledge graph |
| `run-rag` | `application.cli.run_rag_chain` | Query the graph |
| `run-eval` | `application.cli.run_evaluation` | Evaluate retrieval + generation |
| `run-visualization` | `application.cli.run_visualization` | Render an exported graph (no ingestion) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | Generate domain-adapted prompts |
| `prepare-benchmark` | `application.cli.prepare_benchmark` | Turn a public benchmark (MuSiQue) into a corpus + `run-eval` dataset |

Every CLI's `--help` ends with an example invocation and a link to its section here; every flag is in the [CLI Reference](./cli.md).

## Contents

1. [Prerequisites & Installation](#1-prerequisites--installation) (including [Required models](#required-models))
2. [Configuration](#2-configuration) (every key: [Configuration Reference](./configuration.md))
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

Related pages: [CLI Reference](./cli.md) · [Configuration Reference](./configuration.md) · [Model Catalog](./models.md) · [Operator Runbook](./operations.md) ·
[Design Doc](./design.md) · [`iac/README.md`](../iac/README.md)

---

## 1. Prerequisites & Installation

### Runtime

- **Python 3.10 – 3.12**
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager; `pip` works too)

### AWS services

AWS mode only: [Local mode](#local-mode-no-aws) replaces each of them with a
local component.

| Service | Required? | Used for |
|---|---|---|
| **Amazon Bedrock** | Yes | Every LLM call, embeddings and reranking (enable model access for the model ids you configure) |
| **Amazon Neptune** | Yes | The knowledge graph and multi-hop traversal at query time |
| **Amazon OpenSearch** | Yes | Vector and BM25 lexical indices (text units, entities, community reports, relationships, claims) |
| **Amazon S3** | Only for `--s3-sync`, `indexing.opensearch.persist_embedding_cache` or `s3://` paths | Stage-cache sync, the embedding cache, `s3://` datasets and the CDK corpus |
| **Amazon DynamoDB** | Only for incremental indexing | The document-status registry that diffs the corpus by content hash |

### Required models

Enable access to these Bedrock models (the code defaults; a model you
configure replaces the one in its row) before the first run:

| Role | Default model id | Config key |
|---|---|---|
| Default tier (extraction, gleaning, community reports, answers, judges) | `anthropic.claude-sonnet-5-5` | `models.default` |
| Fast tier (chunking, translation, summarization, routing, map steps) | `anthropic.claude-haiku-5-5` | `models.fast` |
| Embeddings (indexing, queries, evaluation and visualization layout) | `amazon.titan-embed-text-v2:0` | `models.embedding.model_id` |
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

The [Model Catalog](./models.md) lists every curated model and its limits.

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
uv sync              # or: pip install -e .
```

**Markdown and HTML** need the `unstructured` extra
(`uv sync --extra unstructured`, Python 3.11 or 3.12; it requires
`unstructured>=0.24.1`, which fixes URL-partitioning SSRF and unescaped
markup). Without it, or on Python 3.10, only `.pdf`, `.txt`, `.csv` and
`.json` are parsed and ingestion skips `.md` / `.html` files with one warning
per extension.

The deployed image (`docker/Dockerfile`) leaves the extra out (it adds about
200 MB), so a Step Functions ingest of `.md` / `.html` files fails with "No
supported files found": convert the files, or build with
`--build-arg UV_EXTRAS="--extra unstructured"`. The image's
`docker/config.yaml` holds no endpoints; the CDK compute stack injects them.

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
  own dimension.
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
[CLI Reference](./cli.md#provider-flags).

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
- Writes look elements up by their `id` property, which the compose file's
  server indexes (without the index it slows down as the graph grows). Use it
  for development-sized corpora.
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
[Configuration Reference](./configuration.md); this section covers what needs
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
([Moved keys](./configuration.md#moved-keys)).

Every `max_attempts` key counts total attempts, the first included; `1`
disables the retry.

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
| `default` | `fixing`, `graph_extraction`, `gleaning`, `claim_extraction`, `community_report`, `entity_extraction`, `answer_generation`, `evaluation` |
| `fast` | `chunking`, `translation`, `description_summary`, `query_translation`, `strategy_selection`, `query_rewrite`, `community_relevance`, `global_map`, `global_reduce`, `drift_primer`, `drift_action` |

```yaml
models:
  default: "openai.gpt-6-sol"          # every default-tier role
  roles:
    answer_generation: "anthropic.claude-opus-5-5"   # one role pinned
    query_rewrite: default                          # moved to the other tier
```

- The embedding model (`models.embedding.model_id`) serves the indices, the
  queries, RAGAS and the visualization layout; changing it or its
  `dimension` needs a re-index. On Bedrock, embedding and rerank ids are a
  closed list ([Model Catalog](./models.md#model-ids-and-capability-records));
  other providers take any model name.
- Settings of an unselected provider (`providers.llm.temperature` with
  Bedrock models) are ignored with a WARNING.
- **Output cap.** `aws.bedrock.default_max_output_tokens` is 16384 because
  Bedrock reserves input + `max_tokens` against the tokens-per-minute quota
  when a request starts; asking for the model maximum throttles concurrent
  ingestion. Long-output prompts derive a higher floor that wins (by default
  23192 for graph extraction and gleaning, 18992 for document translation,
  29792 for claims), rising with their per-chunk limits.
- Curated models, request shapes, prompt caching and
  `aws.bedrock.model_overrides` are in the [Model Catalog](./models.md).

### Context budget

`search.token_manager.max_context_tokens` (`30000`, upstream LightRAG's
budget) is the retrieved context the answer prompt gets, clamped with a
warning to what `models.roles.answer_generation` accepts beside its output
reservation; `null` derives it from that model's window. Per-type shares and
`local` slot quotas: [reference](./configuration.md#searchtoken_manager).

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
  Contextual grounding checks are not run.
- **Ingestion and the evaluation judges run unguarded** (`run-eval`'s answers
  come from the screened query path). A PII guardrail that anonymizes
  `NAME` would rewrite entity names to `{NAME}` (merging distinct people), and
  a `PROMPT_ATTACK` filter can block instruction-like corpus text. On the
  query path, `NAME` anonymization also strips the name a question retrieves
  by, so the baseline guardrail `iac/` creates anonymizes email, phone and
  card numbers but not `NAME` ([iac/README.md](../iac/README.md#guardrail)).
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

The CLIs also write `logging.log_file_path` (`logs/log.txt`, dated
`log_YYYYMMDD.txt`) unless `logging.log_to_file` is `false`. As a library the
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
value in the file would be (`true`/`1`/`yes`/`on` for booleans, JSON for a list
or mapping, e.g. `UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES='["local", "mix"]'`).
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
  ([CLI Reference](./cli.md#run-ingestion)). The variables the CDK tasks get
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
  are in the [reference](./configuration.md#observability).
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

Every flag with its default: [CLI Reference](./cli.md#run-ingestion).

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
- `--force-rebuild` recomputes every stage and rewrites the cache.

**S3 sync** (`--s3-sync`) keeps the stage cache in
`s3://<bucket>/<prefix>/...`, uploaded after every stage, so a fresh task
resumes there. On SIGTERM (ECS `StopTask`, a Step Functions timeout) the run
uploads the running stage's checkpoint and exits 143. A failed upload between
stages is retried by the next one; a failed initial download or final upload
fails the run with `CacheSyncError`. Expiry and the other caches:
[Caches and prefixes](./operations.md#caches-and-prefixes); set
`indexing.opensearch.persist_embedding_cache: true` to avoid re-embedding
unchanged text across runs.

**LLM response cache** (`cache.llm_responses`, on by default). Every
ingestion model response is kept under `llm-cache/` in the run's cache
directory (synced with it), so a re-run with the same pipeline id repays only
the calls a stopped or crashed stage had not made. The key is the rendered
prompt plus the model id, max tokens, temperature and reasoning.

- Only earlier runs' responses are served, each at most once per run; a
  response cut off at the output-token limit is not kept. Above temperature 0
  a re-run replays the earlier sample.
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
`processing.document_parsing.index_value` (no flag); queries select it with `--suffix` (`RAGInput.suffix` in Python).
Both default to `default`. Use one index suffix, and one config file, per
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
- **The analyzer must exist on the cluster**, or the indexing stage fails
  with `the cluster has no analyzer '<name>'`. `nori` comes from the
  `analysis-nori` plugin: the
  [local OpenSearch container](#local-stores-in-containers) has it, an Amazon
  OpenSearch Service domain only once its Nori package is associated (a
  blue/green deployment). The
  [CDK deployment](../iac/README.md) therefore maps `ko` to the built-in `cjk`
  analyzer (character bigrams); switching a domain to Nori is the
  `opensearch_korean_analyzer` row of its
  [configuration](../iac/README.md#configuration-cdkjson-context---c-keyvalue).
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

Every flag with its default: [CLI Reference](./cli.md#run-rag).

### Search strategies — when to use which

GraphRAG strategies excel at *summarization and thematic synthesis* over a
corpus; LightRAG strategies are faster and lean on *dual-level keyword*
retrieval, good for keyword-driven lookups and as a low-cost baseline. Both
share the hybrid scorer (BM25, vector, graph, RRF, rerank).

**GraphRAG (community-summary):**

| Strategy | Use when | How it works |
|---|---|---|
| `simple` | Fast factual lookups | OpenSearch vector + keyword retrieval, no graph traversal; the claims index too when claims are on. Fastest. |
| `local` | Questions about specific entities | Query entities → graph traversal to neighbors and relationships, with vector/keyword hits; claims when enabled. |
| `global` | Broad, thematic questions ("what are the main themes") | Maps the community reports of one cut of the hierarchy (`search.global_search.community_depth`, default 2; a cut over `max_reports_per_level` falls back to the deepest shallower cut that fits) into scored key points on the fast model, one call per `map_context_tokens` of reports, and answers from the ranked points. Dynamic selection walks down from the roots instead. |
| `drift` | Multi-part questions needing the broad picture and specific facts | Reports similar to a hypothetical answer (HyDE) draft intermediate answers and follow-ups; each follow-up runs a local search and raises more, for `n_depth` rounds; the answer model reduces the results. The most LLM calls (about `primer_folds + k_followups * n_depth`). |
| `auto` | General use (the default) | An LLM router (`models.roles.strategy_selection`) picks among `search.auto_routable_strategies` (default local, mix, global; drift, at up to 27 LLM calls a query, only when added). |

**LightRAG (dual-level keyword):**

| Strategy | Use when | How it works |
|---|---|---|
| `mix` | General LightRAG use | Low-level keywords → entity index, high-level keywords → relationship index, one-hop expansion to incident relationships and endpoint entities (Neptune multi-hop is opt-in: `search.lightrag_search.enable_graph_expansion`), **plus** naive chunk retrieval blended in. |
| `hybrid` | Keyword-driven graph questions | `mix` without the naive chunk blend. |
| `naive` | Fast baseline, comparison eval | Chunk retrieval only (`search.lightrag_search.chunk_stream_top_k`, default 20). The default `--search-type hybrid` adds BM25; `--search-type vector` reproduces upstream LightRAG's pure-vector baseline. |

`mix` and `hybrid` read the relationships index ingestion builds; a short
query that yields no keywords is used as a low-level keyword itself (up to
`search.lightrag_search.raw_query_fallback_max_len`).

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

### Filters

`--filters key:value ...` (or `RAGInput.filters`) narrows what retrieval ranks.
Filters compile to `term`/`terms`/`range` clauses on OpenSearch and `has`
steps on Neptune. Each store accepts the fields its indexer writes:

| Store | Filterable fields |
|---|---|
| Text units | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.filters.<key>` |
| Entities | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `attr_<key>`, `attributes.<path>` |
| Relationships | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Claims | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `description`, `source_text`, `text_unit_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Community reports | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `level`, `depth`, `max_cut_depth`, `parent_community_id`, `child_community_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Neptune entity vertices | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `attr_<key>` (where present) |
| Neptune community vertices | `id`, `name`, `level`, `parent`, `size`, `period`, `children` |

- On OpenSearch, `attr_<key>` is entry `<key>` of a document's `filters`
  metadata (an item from several documents carries all their values); on
  Neptune entity vertices it is the entity's own attribute (`attr_role`).
- Entity and claim `type` and claim `status` are stored lowercase and `term`
  matches exactly: filter `type:person`, not `type:PERSON`. Prefer keyword or
  numeric fields; a `term` on an analyzed field such as `description` matches
  single lowercase tokens. A range filter takes `{"gte": ..., "lte": ...}`
  through the Python API.
- A filter applies only to the stores that declare its field
  (`type:person` leaves text units unfiltered). A key no store of the
  selected strategy declares (e.g. `category`) raises `InvalidFilterError`,
  listing the filterable keys
  (`unified_kg_rag/adapters/storage/filter_schema.py`).
- OpenSearch applies `attr_<key>` and `attributes.<path>` strictly
  (fail-closed): an item without the attribute is excluded. Relationships,
  claims and community reports carry the filter values of their text units'
  documents, so each matches when one of them does. A vector or hybrid search
  skips an index none of whose documents carry the attribute (OpenSearch 2.13
  rejects a kNN filter on a field the index does not map). An index built
  before relationships and claims carried their documents' filters returns
  none of them under an attribute filter until it is rebuilt
  (`indexing.reset: true`).
- A document's removal clears its filter values from the entities,
  relationships and claims that outlive it: a key no remaining document
  gives is stored as null, which matches no filter. An index that saw
  removals before this was fixed (#370) keeps the removed documents' values
  on items not rewritten since, so its filters still match them, until it
  is rebuilt (`indexing.reset: true`). An index an incremental run wrote
  after that fix but before #376, with a removal that pruned a community,
  lost the document filters of the text units it kept, which then match no
  filter: rebuild it too.
- `attr_<key>` and `attributes.filters.<key>` name document attributes. A
  graph store holds none, so graph expansion keeps an expanded entity (the
  seeds aside) only when its entities-index document, which carries the
  filters of every document the entity comes from, passes them; under two or
  more keys, one of its text units (one document's) must also pass them all.
  It must also be reached from a seed, within `max_hops`, over relationships
  that pass the filter (an edge only excluded documents state is not
  walked), and its `Path:` follows those relationships. The walk under a
  document filter goes four times wider and is cut after these checks, and
  it passes through entities a vertex filter (`type`) leaves out of the
  results.
  Under a filter on text units, local and mix also fill their chunk quota
  with the entity-linked chunks that pass, not the top-ranked ones only.
  Other
  `attributes.<path>` keys name an item's own attributes and apply to the
  entities index only. On Neptune, `attr_<key>` is also an entity vertex
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
`quit` / `exit`.

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
  editing one.
- **Change the embedding model or its dimension**
  (`models.embedding.model_id` / `dimension`): each OpenSearch index records
  the model and dimension it was built with, and the indexing stage stops
  before any write when they differ from the run's (an index that records
  none is refused too). Re-run once with `indexing.reset: true`. `run-rag` and
  `run-eval` with another model or dimension fail with a `ConfigurationError`
  instead of answering from empty search results.
- **Change the corpus or translation languages** (`processing.translation`
  languages, `indexing.opensearch.language_analyzers`): live text fields keep
  their analyzers, so the indexing stage stops before any write; re-run once
  with `indexing.reset: true`. Dropping a language does not stop it.
- **Index a suffix with the registry off** (as the `backend: local` preset
  does): a later incremental run on that suffix stops before any write.
  Re-run once with `indexing.reset: true` and the registry on, or keep it off
  for that suffix.
- **Rebuild** with `indexing.reset: true`: the whole corpus is indexed again
  without taking the suffix offline, except quarantined documents, which are
  skipped and keep their records; a failed reset drops nothing, so it
  is safe to re-run (how it works:
  [Operator Runbook: Re-ingest](./operations.md#re-ingest)).

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
suffix stay separate even for the same relative path. A file that fails to
parse or load is reported as `failed` and keeps its indexed content until a
run reads it again.

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
See [Operator Runbook: Failed documents](./operations.md#failed-documents).

### Quarantined documents

A document whose processing kills the task (out of memory, a crashing parser)
would stop every later run the same way. After
`indexing.max_document_attempts` (default `3`) runs in a row that died
processing it alone, the next run removes what the attempts wrote, records it
`QUARANTINED`, logs a WARNING naming the file and indexes the rest; it is
skipped until its file changes or the limit is raised.

- Only deaths count: a run that finishes, fails or is stopped by SIGTERM
  clears its in-progress mark, and a full rebuild counts no attempt.
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

Every flag with its default: [CLI Reference](./cli.md#run-eval).

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

An enabled evaluator that cannot be built (no Bedrock access for the judge,
say) or rejects its configuration stops the run; with
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
  exact match.
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
- **`exact_match` and `token_f1`** follow the SQuAD v1.1 normalization
  (NFKC, lowercase, punctuation and English articles removed, whitespace
  collapsed), keeping punctuation that changes meaning (`-5`, `1.5`, `C++`),
  so scores are close to, not comparable with, leaderboard numbers: set
  `evaluation.answer_normalization: official_squad` for the official script.
  Token F1 splits on whitespace, so for Chinese and Japanese it degrades to
  exact match, and Korean particles make both under-count.

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
  (an unknown `search_strategy`) stops the run with
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
    model, the default, inflates judge scores: set `models.roles.evaluation`
    to another model when comparing systems);
  - `retrieval_config_sha256`, a hash of every setting that can change an
    answer or a score (model ids and ingestion settings included; regions,
    endpoints, credentials, paths, timeouts and other operational knobs left
    out), so the same evaluation hashes the same in every environment;
  - `index`: per queried suffix, the ingestion run that last wrote its vector
    indices (`pipeline_id`, `mode`, `built_at`, `documents`, the extraction
    settings and, for an incremental run, the `full_build` it was applied
    on); best-effort, `null` without index metadata.

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
strategy overrides a per-item `search_strategy`), saves every run's files and
adds `evaluation_comparison_<timestamp>.json`:

- `strategies.<s>.metrics.<evaluator>/<metric>`: the mean over queries of the
  per-query mean over repeats, and `repeat_stdev` (run-to-run noise);
  `routed_strategies` shows how `auto` routed.
- `deltas_vs_baseline.<s>.<metric>`: paired against the first strategy over
  the queries both scored: `mean_delta`, a 95% paired bootstrap interval
  (`ci_low`/`ci_high`; 2,000 resamples, seed 0), `wins`/`ties`/`losses`, the
  per-query deltas and `unpaired`. A query whose answer generation failed
  scores 0 on every metric the other run scored (`answer_length` excepted),
  so failing cannot help a strategy (`failed_answers` / `failed_queries`;
  `guardrail_blocked_counts` counts blocks; `warnings` names strategies that
  failed, or had blocked, different numbers). A difference is
  supported when its interval excludes 0; a 10-point difference on binary
  metrics is detected in 15% of runs with 50 queries and 59% with 200, so
  compare on a few hundred queries.
- `pairwise_vs_baseline.<s>` (with `--pairwise-judge`): the GraphRAG /
  LightRAG head-to-head LLM judge on comprehensiveness, diversity,
  empowerment and directness, using `models.roles.evaluation` (prompt:
  `custom_prompts.pairwise_comparison_*`) on the first repeat's answers. Each
  pair is judged in both orders and a criterion counts as a win only when
  both agree; `candidate_win_rate` counts a tie as half. A pair where one
  answer failed is not judged: the other side wins (`candidate_errors`,
  `baseline_errors`; both failed: `both_errors`). `failed` counts pairs the
  judge could not decide (a failed call, or a verdict other than `1`, `2` or
  `tie`), and `--max-failure-rate` applies to them.
- `run_manifest` of the first run.

The console prints the same summary; the failure budget applies to every run.

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
results. It writes `corpus/` (one text file per distinct paragraph; each
question brings 2-4 supporting paragraphs and up to 18 distractors) and
`eval_data.json` (`answer`, `answer_aliases`, the supporting paragraphs as
`reference_sources`, the hop count as `category`). A `corpus/` that is not
empty is a usage error (exit 2), reported before anything is downloaded.

```bash
# 1. 100 questions (about 1,900 paragraphs), sampled with seed 0.
prepare-benchmark musique --output-directory bench/musique --limit 100

# 2. Ingest the paragraphs under their own suffix.
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# 3. Paired comparison of both methodologies, scored like the papers.
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy local,mix,global,naive --repeats 2 --config-path musique.yaml
```

with `musique.yaml` holding your usual config plus:

```yaml
processing:
  document_parsing:
    index_value: musique          # keep the benchmark out of your own index
evaluation:
  answer_normalization: official_squad   # EM / F1 as MuSiQue reports them
  enabled_evaluators: [answer_match, retrieval]   # judge-free; add langchain / ragas if wanted
```

Read `answer_contains` / `token_f1` and `retrieval` (per `category`).
Ingestion cost grows with the paragraph count (about 19 per question). Cite
the paper when you publish results.

---

## 7. Visualization (`run-visualization`)

A **standalone** renderer that draws from an exported visualization-data
JSON; it does **not** re-run ingestion or touch AWS. Every flag with its
default: [CLI Reference](./cli.md#run-visualization).

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

The two renderers are **`interactive`** (pyvis) and **`static`** (Bokeh),
configured under `graph.visualization` (`interactive.*`, `static.*`,
`embedding_method`, `layout_method`).

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
  community hierarchy, without vectors. By default writing it calls no model
  and `run-visualization` computes a seeded spring layout when it renders;
  with `graph.visualization.enabled: true` the stage also renders and stores an
  embedding-based layout (one embedding call per node per ingest).
- **Size.** The interactive graph keeps the top `interactive.max_nodes` nodes
  by degree (default 2000) and lays out only those.
- **Layout.** `embedding_method: "none"` is a spring layout with no
  embedding client; `"description"` embeds each node's `name: description`
  (`models.embedding`) and reduces it with `layout_method`. A failed embedding
  fails the visualization step (not the ingestion) unless
  `processing.ignore_errors: true`, which logs an ERROR and falls back to a
  spring layout; a failed reduction always falls back. A fallback records
  `"layout_degraded": true`.
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
[CLI Reference](./cli.md#run-prompt-tuning).

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
- Non-text formats go through the ingestion loaders (custom
  `ParserFactory.register_loader` ones included); unparsable files are skipped
  with a warning.
- If the model returns no JSON profile, the command exits non-zero and writes
  nothing.

---

## 9. Domain adaptation

Two levers specialize the pipeline for a domain (medical, legal, finance, ...):

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
`--pairwise-judge` judge; keep its JSON verdict format) and the prompt-tuning
`corpus_profile`.

Overrides are checked when the config loads; a mistake fails the load naming
the key:

- A `{name}` the prompt is not called with is an error (it would raise
  `KeyError` on every call). Write a literal brace as `{{` / `}}`:
  `{{"name": "Vendor"}}`. `run-prompt-tuning` output is already escaped.
- Each prompt's data variables must appear in its system or human template
  (`{input_text}` for `graph_extraction` and `claim_extraction`, `{query}` and
  `{context}` for `answer_generation`, ...); limits such as
  `{max_entities_per_chunk}` and `{entity_types}` may be left out.

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
| `1` | The run failed: a failed stage or query (in `run-ingestion` also a `ConfigurationError` found while running, e.g. an incremental run into an index built with the registry off), a breached `--max-failure-rate`, a failed upload, nothing rendered |
| `2` | Usage or configuration error: a bad flag, a missing or invalid input (a `run-eval` dataset, a `run-visualization` data file, a `prepare-benchmark` data file), an invalid config file or override, an unset store endpoint (all before any work), or, outside `run-ingestion`, a `ConfigurationError` found while running |
| `130` | Interrupted with Ctrl-C (SIGINT) |
| `143` | Stopped by SIGTERM (ECS `StopTask`, a Step Functions timeout); `run-ingestion` checkpoints its finished stages first, and `run-eval` uploads the finished runs' results to an `s3://` output (or keeps them locally and logs the path) |

A run stopped with `130` or `143` ends its process at once: model calls in flight are abandoned rather than waited for, the logs, tracing and stdout are flushed first, and a process pool's workers are ended with it.

Flags must be spelled out in full: a prefix such as `--output` is rejected.

`--output-format json` prints one JSON document to stdout and moves the
banner, progress and logs to stderr, so the output pipes into `jq` (not with
`--metrics-sink cloudwatch`, which also writes to stdout): the answer or
search results (`run-rag`), the pipeline id, status and per-stage results
(`run-ingestion`), each run's summary, the comparison and the failure-budget
breaches (`run-eval`), the written files (`run-visualization`), the output
path and profile (`run-prompt-tuning`), or the counts written
(`prepare-benchmark`). A run stopped by a
signal before it printed prints `{"status": "interrupted", "exit_code": 143}`
(`130` for Ctrl-C), and one that fails before it printed prints
`{"status": "failed", "exit_code": 1}` (or `2`), so stdout is always JSON;
`run-ingestion` adds the `pipeline_id` to resume with.

### Large / multilingual / heterogeneous corpora

- **Large:** raise `processing.max_concurrency` as far as the Bedrock quota
  allows; keep `indexing.neptune.index_concurrency` at `1` unless you measured
  a gain (concurrent write batches conflict on shared vertices and retry);
  `indexing.opensearch.persist_embedding_cache` with `--s3-sync` saves
  re-runs the embedding work; the doc-status registry
  (`providers.doc_status.enabled`) makes later runs pay only for changed
  documents.
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
  on first use and keep them until garbage collection otherwise.
- Importing the package configures no logging handler;
  `unified_kg_rag.setup_logging(config)` gives the CLI's logging. Without it,
  structured fields (including `structlog.contextvars` bindings) reach your
  handlers as record attributes, a trailing `_` added to one named like a
  `LogRecord` attribute (`name`, `msg`, ...).

Extensions import from the package root too: the ports (`LLMFactoryPort`,
`GraphIndexer`, `DocStatusPort`, ...), `Providers`, `register_provider`,
`BaseSearchStrategy` with `register_strategy`, `ParserFactory` and
`register_renderer`. Both constructors take injected backends, either as a
`providers=Providers(config, ...)` bundle or as individual arguments
(`doc_status=`, `vector_indexer=`, `model_factory=`, ...), not both; see
[Design Doc §15](./design.md#15-extension-guide).

---

## 12. Limitations

- **Extraction can hallucinate** entities or relationships the text does not
  support, which then reach retrieval.
  `processing.graph_extraction.entity_grounding.enabled` (off by default)
  drops or down-weights items whose quoted `source_text` is not in the chunk.
- **Runs are not reproducible**: LLM outputs (extraction, reports, the
  `auto` router, answers) vary, so compare configurations over several runs.
  Given the same model responses, everything else is deterministic in any
  process, whatever its `PYTHONHASHSEED`.
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
  `hybrid` lose high-level keywords and `local` its relationship section.
- **Incremental indexing approximates** (§5): delta runs append communities
  instead of re-clustering, and stored communities and reports keep an
  entity's old id after it took its current name's; a full rebuild
  (`indexing.reset: true`) refreshes both. With DynamoDB one document's chunk
  ids must fit one 400 KB item (*Document size limit*).
- **Concurrent runs need the registry**, whose write lease stops a second run
  on a namespace; without it (or through the direct `IncrementalIndexer` API)
  nothing locks, and two ingestions on one suffix can delete each other's
  index ([Concurrent runs](./operations.md#concurrent-runs)).
- **Isolating a document that kills the task takes runs** (*Quarantined
  documents*): about `log2(n) + 1` for one culprit among `n` documents that
  died together, while the deferred ones wait.
- **Markdown and HTML** need the `unstructured` extra (Python 3.11+), which
  the container image leaves out by default.
- **Changing the embedding model** or dimension needs `indexing.reset: true`;
  an incremental run refuses to write into another model's indices
  ([Workflows](#workflows)).

---

*See also: [README.md](../README.md) for the overview and quickstart, and
[docs/design.md](./design.md) for architecture and internals.*
