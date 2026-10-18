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

Every CLI's `--help` ends with an example invocation and a link to its section here.

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

Related pages: [Configuration Reference](./configuration.md) · [Model Catalog](./models.md) · [Operator Runbook](./operations.md) ·
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
| **Amazon Bedrock** | Yes | All LLM calls (chunking, extraction, gleaning, community reports, answer generation), embeddings, and reranking. Enable model access for the model IDs you configure. |
| **Amazon Neptune** | Yes | The knowledge graph (entities, relationships, communities) and multi-hop traversal at query time. |
| **Amazon OpenSearch** | Yes | Vector + BM25 lexical indices (text units, entities, community reports, relationships, claims). |
| **Amazon S3** | Only for `--s3-sync`, `indexing.opensearch.persist_embedding_cache` or `s3://` paths | Stage-cache sync, embedding-cache persistence, and `s3://` evaluation datasets and outputs. The CDK pipeline also stages the corpus in S3. |
| **Amazon DynamoDB** | Only for incremental indexing | Document-status registry that diffs the corpus by content hash. |

### Required models

Enable access to these Bedrock models before the first run. They are the code
defaults; a model you configure instead replaces the one in its row.

| Role | Default model id | Config key |
|---|---|---|
| Default tier (extraction, gleaning, community reports, answers, judges) | `anthropic.claude-sonnet-5-5` | `models.default` |
| Fast tier (chunking, translation, summarization, routing, map steps) | `anthropic.claude-haiku-5-5` | `models.fast` |
| Embeddings (indexing, queries, evaluation and visualization layout) | `amazon.titan-embed-text-v2:0` | `models.embedding.model_id` |
| Reranking | `cohere.rerank-v3-5:0` | `models.rerank` |

- **Region.** Every Bedrock call, the rerank call included, goes to
  `aws.bedrock.region_name`, which defaults to `aws.region_name` (`us-west-2`).
  All four default models are offered there. Rerank models are not offered in
  every region (for example not in `ap-northeast-2`): in such a region, set
  `aws.bedrock.region_name` to a region that offers them, or set
  `search.reranking.enabled: false`. A private VPC whose only Bedrock route is
  its VPC endpoint must keep Bedrock in its own region.
- **Inference profiles.** Claude Sonnet 5.5 and Haiku 5.5 are invocable only
  through a cross-region inference profile. Keep
  `aws.bedrock.enable_global_profile: true` and grant
  `bedrock:ListInferenceProfiles`; if no profile resolves, model creation fails with a message that names this remedy.
- **Permissions.** See [§10 IAM permissions](#iam-permissions); reranking needs
  its own statement.

The [Model Catalog](./models.md) lists every curated model and its limits.

The framework connects to services that **already exist** — it never creates
them. Choose one of three ways to run it:

- **Local mode (development).** Run every component locally (Ollama models
  and embedded SQLite stores, or Gremlin Server and OpenSearch in
  containers), or only the stores with Bedrock models; see
  [Local mode](#local-mode-no-aws) below.

- **Bring your own services.** If Neptune and OpenSearch (and, when you use
  them, S3 and DynamoDB) are already running, just record their endpoints in `config.yaml`
  (§2 below) and skip ahead. Make sure Bedrock model access is enabled for the
  model IDs you configure.
- **Provision everything with the bundled CDK app.** The repo ships an optional,
  Well-Architected AWS CDK app in [`iac/`](../iac/README.md) that stands up the
  whole stack in one command — networking (VPC + endpoints), the Neptune cluster,
  the OpenSearch domain, the DynamoDB doc-status table, the S3 cache bucket, an
  ECS Fargate data plane, a Step Functions ingestion pipeline, and CloudWatch
  observability, plus an optional region-pinned Bedrock Guardrail.
  [`iac/README.md`](../iac/README.md) has the deploy commands
  ([Usage](../iac/README.md#usage)), every `-c key=value` option and the
  production-hardening checklist. Neptune and OpenSearch are VPC-only, so the
  CLIs run inside the VPC as a task of the deployed task definition, which
  receives the endpoints as environment variables
  ([After deploy](../iac/README.md#after-deploy)).

### Install

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws

# uv (recommended)
uv sync

# or pip
pip install -e .
```

Optional extra: parsing **Markdown (.md)** and **HTML (.html)** requires the
`unstructured` package. Without it, only `.pdf`, `.txt`, `.csv`, `.json` are
parsed: ingestion skips `.md`/`.html` files and logs one warning per
extension with the install command (`Skipping 2 '.md' file(s) ... install the
optional 'unstructured' extra`). On Python 3.11 or 3.12, install the patched
parser with
`uv sync --extra unstructured` or `pip install -e '.[unstructured]'`.
The extra requires `unstructured>=0.24.1`, which fixes URL-partitioning SSRF
and unescaped markup in generated HTML. On Python 3.10 the extra does not install a parser;
use PDF/TXT/CSV/JSON, or upgrade Python for Markdown/HTML support.

The deployed container image (`docker/Dockerfile`) leaves this extra out by
default because it adds about 200 MB (spaCy and friends), so a Step Functions
ingest of `.md`/`.html` files fails with "No supported files found". Either
convert those files to a supported format, or build the image with the extra:
`docker build --build-arg UV_EXTRAS="--extra unstructured" -f docker/Dockerfile .`.
The image bakes in `docker/config.yaml`, which holds no endpoints; the CDK
compute stack injects them as environment variables.

### Authentication

Two independent auth concerns:

1. **AWS credentials** — supplied through the standard credential chain. Set
   `aws.profile_name` in `config.yaml` to use a named profile, or leave it
   `null` to use the default chain (env vars, instance role, etc.). Neptune
   uses SigV4 when `aws.neptune.use_iam: true`.

2. **OpenSearch auth** — either IAM (`aws.opensearch.use_iam: true`) or
   username/password. For username/password, set `use_iam: false` and create a
   `.env` file (copy `.env-template`):

   ```bash
   # .env — only needed when aws.opensearch.use_iam is false
   UKR__AWS__OPENSEARCH__USERNAME=your_opensearch_username
   UKR__AWS__OPENSEARCH__PASSWORD=your_opensearch_password
   ```

   The `run-*` CLIs load the `.env` file automatically. When `use_iam: true`, no `.env` is needed.

### IAM permissions

Grant the principal running the CLIs access to: Bedrock (InvokeModel /
InvokeModelWithResponseStream / CountTokens for your model IDs, plus
embeddings; without CountTokens token counts fall back to estimates), Neptune
(connect / SigV4 for `use_iam: true`), OpenSearch (read/write the configured
indices), S3 (the configured bucket), and DynamoDB (when incremental indexing is
on).

Three Bedrock actions need statements of their own:

- **`bedrock:Rerank` on `Resource: "*"`.** The Rerank API authorizes against a
  different resource shape than `InvokeModel`, so a statement scoped to
  foundation-model or inference-profile ARNs denies it (the CDK task role
  grants it separately, `iac/stacks/compute_stack.py`). Without it every
  query logs `Reranking failed: ...` at ERROR, reports
  `metadata.rerank.status: failed` and returns the fused results unreranked
  (or fails, with `search.reranking.required: true`). Reranking is on by
  default; if you cannot grant the action, set
  `search.reranking.enabled: false`.
- **`bedrock:ListInferenceProfiles` on `Resource: "*"`**, an account-level
  read the model resolver makes.
- **`bedrock:ApplyGuardrail`** on the guardrail's ARN
  (`arn:aws:bedrock:<bedrock region>:<account>:guardrail/<id>`), when a
  guardrail is set.


### Local mode (no AWS)

Local mode runs ingestion, querying and evaluation on a laptop or a single
host without an AWS account: each role is served by a local component instead
of Bedrock, Neptune, OpenSearch or DynamoDB. AWS stays the default and the
primary target; local mode is for development and evaluation. By default the
graph store, the vector/lexical store and the incremental-indexing registry
are SQLite files, so the only service to run is [Ollama](https://ollama.com)
(installed natively, serving `http://localhost:11434`):

```bash
uv sync --extra local
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
uv run run-ingestion --backend local --source-directory ./docs-in
uv run run-rag --backend local --query "..."
```

Without a config file the stores are created in `.unified-kg-rag/` in the
working directory (`graph.sqlite3`, `vector.sqlite3`, `doc-status.sqlite3`),
so run every command from the same directory, or put the settings in a config
file with `backend: local` (relative paths then resolve against the file).
Incremental indexing is off in the preset; turn it on with
`providers.doc_status.enabled: true`. To start over, delete the three files
together (the registry describes what the stores hold) or run once with
`indexing.reset: true`. To run Gremlin Server and OpenSearch in containers
instead, see [Local stores in containers](#local-stores-in-containers).

On Linux, `uv sync --extra local` installs the CUDA build of PyTorch that the
lockfile pins (several GB) for the cross-encoder, GPU or not. To get the CPU
build instead, install the extra through uv's pip interface:

```bash
uv sync
uv pip install --torch-backend cpu -e '.[local]'
```

`uv run` keeps these packages; a later plain `uv sync` removes them, and
`uv sync --extra local` brings back the locked CUDA build.

`backend: local` (in a config file, or `--backend local`, see
[Provider flags](#provider-flags)) fills in these providers for every key the
config leaves unset. A key the file sets always wins over the preset: a
config that sets `providers.<role>.kind`, an `aws.neptune` or
`aws.opensearch` connection key (`port`, `use_iam`, `use_ssl`, ...) or an
embedding or rerank model id keeps that value under `backend: local`.
`config-template.yaml` leaves these keys commented out, so a copy of it
switches with `backend: local` alone. Each role is chosen under
`providers:`, so roles can be mixed, for example local stores with Bedrock
models:

| Role | `providers.<role>.kind` | Local default |
|---|---|---|
| Language models | `llm`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `qwen2.5:7b-instruct` at `http://localhost:11434` |
| Embeddings | `embedding`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `bge-m3` (1024 dimensions) |
| Reranking | `rerank`: `bedrock` \| `cross_encoder` \| `none` | `cross_encoder`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Graph store | `graph`: `neptune` \| `gremlin` \| `sqlite` | `sqlite` at `providers.graph.sqlite_path` (default `.unified-kg-rag/graph.sqlite3`) |
| Vector/lexical store | `vector`: `opensearch` \| `sqlite` | `sqlite` at `providers.vector.sqlite_path` (default `.unified-kg-rag/vector.sqlite3`) |
| Doc-status registry | `doc_status`: `dynamodb` \| `sqlite` | `sqlite` at `providers.doc_status.sqlite_path` (default `.unified-kg-rag/doc-status.sqlite3`) |

`gremlin` is a TinkerPop Gremlin Server and `opensearch` a local OpenSearch,
both reached through the `aws.neptune` and `aws.opensearch` keys the preset
points at `localhost` (see [Local stores in containers](#local-stores-in-containers)).

- **Models.** With a non-Bedrock LLM, `models.default` is required (the
  preset sets it) and `models.fast` defaults to it; a `models.roles` entry
  still wins for its role. `providers.llm.context_window_size`
  and `max_output_tokens` describe the model (Bedrock models take theirs from
  the catalog), and the window is sent to Ollama as `num_ctx`. The preset
  also sets `search.token_manager.max_context_tokens` to 22000, what the
  default 32,768-token window accepts beside the 8,192-token output
  reservation (the AWS default of 30000 would be clamped on every query).
  It applies only to a non-Bedrock LLM left at the default window: set
  `context_window_size` (or `kind: bedrock`) and the regular default
  applies, clamped to the model's window. An
  OpenAI-compatible server (vLLM, llama.cpp server, LM Studio) takes
  `base_url` (its `/v1` root) and, if it needs one, `api_key`.
- **Embeddings.** `models.embedding.model_id` and `dimension` are required
  with a non-Bedrock provider; the dimension is fixed into the vector store's
  mappings, so changing either needs a full re-index. The preset's dimension
  (1024) belongs to its model (`bge-m3`): naming another model needs its
  dimension too.
- **Reranking.** The cross-encoder downloads its model from Hugging Face on the
  first query. Without the `[local]` extra (or with `kind: none`) queries keep
  the fused order; this is logged once and visible as
  `kg_rag.rerank_enabled=false` on the fusion span and as
  `metadata.rerank.status` `unavailable` (or `disabled` for `kind: none`).
- **Embedded stores.** Each `sqlite` store is one file, created on first use.
  A relative `sqlite_path` is resolved against the config file's directory
  (the working directory without a config file). The `indexing.neptune` and
  `indexing.opensearch` settings (index prefixes, batch sizes, embedding
  settings) apply to them as to Neptune and OpenSearch, and a write waits up
  to `search.store_timeout_seconds` for another process's lock.
- **Incremental indexing** uses the SQLite registry when
  `providers.doc_status.enabled: true` (or `aws.dynamodb.enabled: true`, the
  original switch); [`docker/config.local.yaml`](../docker/config.local.yaml)
  turns it on. A relative `sqlite_path` is resolved like the stores', and the
  run logs the database's absolute path.
- **Token counting** uses the script-aware estimate (the same one Bedrock falls
  back to); `providers.llm.tokenizer: tiktoken` counts with a BPE encoding
  instead, which downloads its encoding file on first use.
- **Custom kinds** register a builder with
  `adapters.provider_registry.register_provider`; see [design §15](./design.md#15-extension-guide).

#### Differences from AWS mode

**Quality and models**

- Extraction with small local models is noticeably weaker: more malformed
  output (retried, then recorded as failed chunks), weaker entity resolution
  and thinner community reports. Prefer the largest model the hardware runs.
- No Bedrock prompt caching, and reasoning-effort settings do not apply.
- A local model's response cut at its output limit is not detected as
  truncated (Bedrock reports the stop reason); keep `max_output_tokens` at
  what the model can generate.
- Token counts are estimates, so context budgets are approximate.
- The LLM judges of `run-eval` (`langchain`, `ragas`) run on the local model
  and score less reliably; the deterministic evaluators are unaffected.
- The local providers never attach a guardrail to a model call. With
  `aws.bedrock.guardrail.identifier` set, the query chain still screens
  queries and answers through Bedrock `ApplyGuardrail` (AWS credentials
  needed).
- A model server that refuses a query's calls (a wrong key or model name, HTTP
  4xx) or cannot be reached after the client's retries fails the query with
  the client's error rather than answering "could not find".

**The embedded vector store** answers the queries the OpenSearch adapter
builds, with these differences:

- Text mapped to `nori` (Korean), `kuromoji` (Japanese) or `smartcn`
  (Chinese) is split into overlapping CJK character bigrams instead of being
  analyzed morphologically: a name still matches inside a longer word
  (홍길동 in 홍길동으로부터), but ranking differs from OpenSearch's.
- The `english` analyzer stems only with `snowballstemmer` from the `[local]`
  extra (without it, a warning is logged once). Emoji are not indexed.
- There is no fuzzy matching (OpenSearch's `fuzziness: AUTO`), so a
  misspelled query term, and under the non-stemming `standard` analyzer an
  inflected form ("vendors" for "vendor"), matches only through the vector
  side.
- kNN search is exact: every document that passes the filters is scored,
  which is accurate but grows linearly with the index. Hybrid search is fused
  by the retriever the same way on both stores, so it differs only where the
  lexical and kNN searches do.
- Each searched index is held in memory (about documents x dimension x 4
  bytes per vector field). It is sized for about 100,000 documents per index
  and logs a warning when it loads a larger one; use OpenSearch beyond that.
  The first query after a write reloads the index, and the first lexical
  query re-analyzes the searched field: about 10 s for 50,000 documents of
  120 words (about three times slower under `english` above about 65,000
  distinct words).

**SQLite files** allow one writer at a time: run one ingestion per store file
at a time (queries can read alongside it). A writer that waits longer than
`search.store_timeout_seconds` fails with "stayed locked". Keep the files on a
local disk; network filesystems (NFS, SMB, Amazon EFS) are not supported.
With the [containers](#local-stores-in-containers), Korean lexical analysis
depends on the `nori` plugin (installed by the compose file) or an equivalent
analyzer in your OpenSearch. The messages are in
[Common errors](./operations.md#local-mode).

#### Provider flags

`run-ingestion`, `run-rag`, `run-eval`, `run-prompt-tuning` and
`run-visualization` take these flags (`run-visualization` calls no model or
store, so it only accepts them, for one command line across the CLIs). Each one overrides its key of the loaded config, a key the file sets
included; an omitted flag leaves the config as it is.

| Flag | Sets |
|---|---|
| `--backend {aws,local}` | `backend`. The preset still fills in only the keys the config leaves unset. `--backend aws` over a `backend: local` file drops the values the local preset filled in, so those keys return to the AWS defaults; a value the file or the environment sets is kept, even one equal to the preset's |
| `--llm-provider KIND` | `providers.llm.kind`: `bedrock`, `openai_compatible` or `ollama` |
| `--llm-model MODEL` | `models.default` |
| `--embedding-provider KIND` | `providers.embedding.kind`: `bedrock`, `openai_compatible` or `ollama` |
| `--embedding-model MODEL` | `models.embedding.model_id` |
| `--embedding-dimension N` | `models.embedding.dimension` |
| `--rerank-provider KIND` | `providers.rerank.kind`: `bedrock`, `cross_encoder` or `none` |
| `--model-base-url URL` | `providers.llm.base_url` and `providers.embedding.base_url` |

#### Local stores in containers

[`docker/compose.local.yaml`](../docker/compose.local.yaml) runs the stores
as servers instead: a TinkerPop Gremlin Server (in-memory TinkerGraph), a
single-node OpenSearch 2.13 with the security plugin disabled and the
`analysis-nori` plugin installed (the Korean mappings use `nori`; an Amazon
OpenSearch Service domain has it only once its optional package is associated,
see [Multilingual ingestion](#multilingual-ingestion)), and, under the `models` profile,
Ollama. OpenSearch is the same query engine as Amazon OpenSearch Service, so
use it to check ranking (morphological analyzers, fuzzy matching) before an
AWS deployment. [`docker/config.local.yaml`](../docker/config.local.yaml)
selects `providers.graph.kind: gremlin` and `providers.vector.kind:
opensearch` and turns on incremental indexing. It needs Docker with Compose
2.23.1 or later (the compose file defines its Gremlin Server init script
inline, which older releases reject).

```bash
uv sync --extra local
docker compose -f docker/compose.local.yaml --profile models up -d --wait
docker compose -f docker/compose.local.yaml exec ollama ollama pull qwen2.5:7b-instruct
docker compose -f docker/compose.local.yaml exec ollama ollama pull bge-m3
uv run run-ingestion --config-path docker/config.local.yaml --source-directory ./docs-in
uv run run-rag --config-path docker/config.local.yaml --query "..."
docker compose -f docker/compose.local.yaml down -v
```

`down -v` deletes the stores' data but not the SQLite registry
(`.unified-kg-rag/doc-status.sqlite3` at the repository root with
`docker/config.local.yaml`). Before the next ingestion into fresh stores,
delete that file or set `indexing.reset: true`; otherwise the incremental run
stops before writing anything (see the Gremlin Server note below).

No model is pulled automatically. The `backend: local` preset points the
`gremlin` and `opensearch` kinds at the containers with
`aws.neptune.use_ssl: false` and `use_iam: false` (plain `ws://`, no SigV4)
and `aws.opensearch.allow_anonymous: true` with `use_ssl: false` (plain
`http://`, no auth).

To check the stores without any model, run the smoke test, which uses a
hashing embedding provider and a fake model server:
`LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py`.

Known differences from Neptune:

- TinkerGraph keeps the graph in memory only, so restarting Gremlin Server
  empties it while the SQLite registry keeps its records. An incremental run
  then stops before writing anything and asks for `indexing.reset: true` (or
  deleting the registry), instead of treating the lost documents as
  unchanged.
- A list property written when a vertex is created keeps duplicate values
  (Neptune stores them once).
- Every write looks its vertex or edge up by its `id` property first, so a
  retry does not duplicate it. The compose file's Gremlin Server indexes that
  property at start-up; one started without the index scans every vertex or
  edge per lookup and slows down as the graph grows (10,000 entities and their
  communities took about 60 seconds instead of 20). Use it for
  development-sized corpora.
- The containers drop all Linux capabilities, but the stores have no
  authentication or TLS; the compose file binds them to `127.0.0.1` only.

#### Local stores with Bedrock models

To keep Bedrock for the models and run only the stores locally, set
`providers.llm.kind`, `providers.embedding.kind` and `providers.rerank.kind`
to `bedrock` in a config with `backend: local` (Bedrock model ids then
validate against the catalog), with AWS credentials that have Bedrock access.
The embedded stores need nothing else; for the containers, start the stores
alone (`docker compose -f docker/compose.local.yaml up -d --wait`, no
`models` profile) and make the change in a copy of `docker/config.local.yaml`.

---

## 2. Configuration

Copy the template and point every CLI at it with `--config-path config.yaml`:

```bash
cp config-template.yaml config.yaml
```

A config file holds only the keys you change; every other key keeps its
built-in default. The [Configuration Reference](./configuration.md) lists
every key with its type, default and description (it is generated from the
code, so it is always current). This section covers how the configuration is
loaded, the settings most runs change, and the topics that need more than a
table row: models, the guardrail, logs, environment variables and tracing.

### How configuration is loaded

Values are resolved in layers, each overriding the one before:

1. **Built-in defaults**: the Pydantic models in
   `unified_kg_rag/domain/models/config.py`. Without `--config-path`, a CLI runs
   on these defaults plus environment overrides.
2. **The `backend` preset**: `backend: local` (or `--backend local`) fills in
   the local providers and stores for every key the file and the environment
   leave unset ([Local mode](#local-mode-no-aws)).
3. **Your YAML file**: every key you set replaces its default; every key you
   omit keeps it. A dict-valued key (for example `logging.library_levels` or
   `graph.visualization.interactive`) replaces the whole default dict rather
   than merging into it.
4. **Environment variables** ([below](#environment-variable-overrides)): on
   top of the file.
5. **Provider flags** (`--backend`, `--llm-provider`, `--rerank-provider` ...):
   applied last. The config is rebuilt from what the file and the environment
   set (and, from Python, every change made to the `Config` object since), so
   switching `--backend` drops only what the old preset filled in, never a
   value you set, even one equal to the preset's.

The file is validated when it loads. A wrong type or an unsupported value
stops the CLI with `Configuration validation error: ...`, naming each invalid
key. That, a file that is not valid YAML, a missing `--config-path` file and
an unknown key (in the file or as a `UKR__` variable) are usage errors: every
CLI prints one error line and exits with code 2. Unknown keys are refused
because a misspelled key would leave its setting at the default without a word
(a guardrail identifier under a wrong key leaves queries unscreened);
`--allow-unknown-keys` (`allow_unknown_keys=True` for `get_config` /
`ConfigLoader`) logs each one at WARNING instead. A key that moved fails with
the key that replaces it ([Moved keys](./configuration.md#moved-keys)).

Every `max_attempts` key counts total attempts, including the first; `1`
disables the retry.

### Settings most runs change

| Key | Default | Change it when |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | Your stores are in another region. Bedrock follows it unless `aws.bedrock.region_name` is set |
| `aws.neptune.endpoint`, `aws.opensearch.endpoint` | `null` | Always, for the AWS stores (the CDK stack injects them) |
| `aws.dynamodb.enabled` | `false` | You want incremental indexing ([§5](#5-incremental-indexing)) |
| `models.default`, `models.fast` | Claude Sonnet 5.5, Claude Haiku 5.5 | You use other models ([Model selection notes](#model-selection-notes)) |
| `processing.max_concurrency` | `40` | The logs show `ThrottlingException` retries (lower it), or your Bedrock quota allows more |
| `processing.graph_extraction.entity_types` | 7 generic types | You adapt extraction to a domain ([§9](#9-domain-adaptation)) |
| `processing.translation.target_language` | `"en"` | The corpus or the answers should be in another language ([Multilingual ingestion](#multilingual-ingestion)) |
| `processing.claim_extraction.enabled` | `false` | You want claims in `local` search (one more LLM call per text unit) |
| `processing.document_parsing.index_value` | `null` (`default`) | You keep several corpora or tenants apart ([Index suffix](#index-suffix)) |
| `graph.community_detection.enabled` | `true` | You query only with LightRAG and want a lighter ingestion (`global` and `drift` need it) |
| `search.reranking.enabled` | `true` | Your Bedrock region has no rerank model, or you do not want reranking |
| `search.token_manager.max_context_tokens` | `30000` | Answers need more (or cheaper, less) retrieved context ([Context budget](#context-budget)) |
| `indexing.reset` | `false` | You rebuild the stores from the whole corpus, for one run |
| `aws.bedrock.guardrail.identifier` | `null` | You screen queries and answers with a Bedrock guardrail ([Guardrail](#guardrail)) |
| `observability.enabled` | `false` | You want OpenTelemetry traces ([Observability](#observability)) |

### Model selection notes

The top-level `models` section says which model serves each role; the
provider of each family is `providers.llm.kind`, `providers.embedding.kind`
and `providers.rerank.kind`, and the ids are checked against it. Every LLM
role runs on one of two tiers, so switching the whole pipeline to another
model is one line, and a `models.roles` entry (a model id, or `default` /
`fast` to move the role to the other tier) wins over its tier:

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

The embedding model (`models.embedding.model_id`) serves the indices, the
queries, RAGAS and the visualization layout; changing it, or its `dimension`,
needs a re-index. Embedding and rerank model ids are a closed list
(`amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1`,
`cohere.embed-v4:0`, `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`;
`cohere.rerank-v3-5:0`, `amazon.rerank-v1:0`); any other id fails validation.
A setting of a provider the config does not select (say
`providers.llm.temperature` while the language models are Bedrock's) is
ignored with a WARNING at load.

The curated models, the request differences between Claude generations and
OpenAI GPT models, prompt caching, and how to describe a model the package
does not know (`aws.bedrock.model_overrides`) are in the
[Model Catalog](./models.md).

**Output cap.** `aws.bedrock.default_max_output_tokens` is 16384 because
Bedrock reserves input + `max_tokens` against the tokens-per-minute quota when
a request starts, so asking for the model maximum (128K on Claude 5.x)
throttles concurrent ingestion long before real usage does. Thinking tokens
count toward it. Prompts with long outputs derive a higher floor from their
own limits that wins: the largest answer those limits allow plus 8192 tokens
of reasoning headroom. Graph extraction and gleaning (and their output fixer)
budget 150 tokens per record at `max_entities_per_chunk` +
`max_relationships_per_chunk` (23192 by default); document translation budgets
1.35 tokens per character of `max_chunk_size` (18992), claim extraction twice
that (29792); community reports stay under the default cap. Raising those
limits raises the floor with them.

### Context budget

`search.token_manager.max_context_tokens` (`30000`) is the retrieved context
the answer prompt gets. It matches upstream LightRAG's total context budget
(MS GraphRAG uses 12000): a budget derived from a 1M-token window (~785K)
never binds, so the per-type budgets would never trim anything. The value is
always clamped to what `models.roles.answer_generation` accepts alongside its
output reservation (`aws.bedrock.default_max_output_tokens`), with a warning
when that happens. With `null`, the budget is derived from that model's window
minus the output reservation and `context_window_headroom_ratio`;
`aws.bedrock.enable_1m_context` widens it on models where 1M is a beta. The
per-section-type shares (`search.token_manager.type_budgets`) and the `local`
slot quotas (`search.local_search.type_quota`) are in the reference.

### Guardrail

Setting `aws.bedrock.guardrail.identifier` screens two texts of every query
with the `ApplyGuardrail` API; the guardrail must exist in
`aws.bedrock.region_name`.

- **What is screened.** The raw user query, once, before conversation memory,
  the follow-up rewrite, routing and retrieval (`check_input`), and the final
  answer (`check_output`; a streamed answer is then delivered as one chunk
  after the check). Only those texts are evaluated and billed.
- **What happens.** A blocked query is answered with the guardrail's blocked
  message; nothing is retrieved or written to memory, and
  `RAGOutput.metadata.blocked` is `true`. A masked query (PII anonymized)
  continues with the masked text, so retrieval, the models and memory never
  see the original. A blocked or masked answer is replaced by the guardrail's
  text. Every intervention is listed in `RAGOutput.metadata.guardrail`
  (source, action, policies), logged at WARNING with a running count, and
  recorded on a `kg_rag.guardrail` span. A guardrail API error fails the
  query (with `processing.ignore_errors`, the standard error answer); it is
  never answered unscreened.
- **Why not on the model calls.** The model calls inside a query (rewrite,
  routing, keyword extraction, global/DRIFT map steps, answer generation) run
  without a guardrail. Attached to a model call, a guardrail evaluates the
  whole prompt: on the Converse API that is the retrieved corpus text and the
  developer instructions, which a `PROMPT_ATTACK` filter judges as user input
  (false blocks), at a cost that grows with context size times calls; on the
  InvokeModel API the `PROMPT_ATTACK` filter does nothing unless the user
  input is wrapped in input tags
  ([Bedrock Guardrails: prompt attacks](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html)).
  Contextual grounding checks are not run: they need the grounding source and
  query as qualified content, which the answer screen does not send.
- **Ingestion and evaluation.** Ingestion models, the prompt tuner and the
  evaluation judges run unguarded by default (`apply_to: "query"`): a PII
  guardrail that anonymizes `NAME` rewrites extracted entity names to a
  placeholder such as `{NAME}` (distinct people merge into one node), and a
  `PROMPT_ATTACK` filter can block instruction-like corpus text.
  `apply_to: "all"` attaches the guardrail to those calls; use it only with a
  policy that is safe for extraction. On the query path `NAME` anonymization
  also hurts: a question about a person loses the name it retrieves by, and
  answers show `{NAME}`. The baseline guardrail that `iac/` creates therefore
  anonymizes email, phone and card numbers but not `NAME`.
- **Custom code** that builds chains with `setup_chain` or calls `get_model`
  for non-query work should pass `model_purpose=ModelPurpose.INGESTION` (or
  `EVALUATION`). Unmarked calls default to `QUERY`: they get the query-time
  transient-error retry, and never an attached guardrail.

### S3 cache encryption

`aws.s3.encryption.encryption_type: BUCKET_DEFAULT` sends no encryption
header, so the bucket's default applies (with the CDK stack's `use_cmk=true`,
its customer-managed KMS key; forcing `AES256` would bypass it). When you reuse
a bucket whose default is SSE-KMS, the writer needs `kms:GenerateDataKey` and
`kms:Decrypt` on that key.

### What the logs contain

At `INFO` and above the package logs only lengths, counts, ids and a short
hash for user and corpus text, for example `query: len=42 sha=1a2b3c4d`. The
hash is stable, so one query can be followed across records without revealing
it. Query text, rewritten (DRIFT and translated) queries, entity names and raw
model output are logged only at `DEBUG`, and exception messages never carry
model output. Treat `DEBUG` logs (`logging.level: DEBUG`,
`UKR__LOGGING__LEVEL=DEBUG` or `--verbose`) as containing user and corpus
data: do not enable them where logs are shipped to shared storage. Libraries
set to `DEBUG` in `logging.library_levels` (for example `botocore`) can log
request bodies too.

The CLIs also write their logs to `logging.log_file_path` (`logs/log.txt`,
dated as `log_YYYYMMDD.txt`) unless `logging.log_to_file` is `false`.
Imported as a library, the package configures no handler or file: the host
application's logging setup applies, its formatter shows the plain message,
and the structured fields (`logger`, `level`, `pipeline_id`, `trace_id`, ...)
are attributes of the log record.

### Custom prompts

Every prompt has a `custom_prompts.*_system` / `*_human` override (default
`null` = the built-in prompt in `unified_kg_rag/domain/prompts/`). Override
what you need and leave the rest `null` ([§9](#9-domain-adaptation)).
Overrides are validated at load: unknown `{variables}` and missing data
variables (such as `{input_text}`) are errors, and a literal brace is written
`{{` / `}}`.

### Environment variable overrides

Any config key can be set from the environment as `UKR__` followed by its
path, with double underscores between the segments, in any case:
`UKR__AWS__NEPTUNE__ENDPOINT` sets `aws.neptune.endpoint`,
`UKR__SEARCH__RERANKING__TOP_K` sets `search.reranking.top_k`, and a segment
past a free-form mapping names one of its keys
(`UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`). A variable wins over
the config file and the built-in defaults. Values are converted to the field's
type (`true`/`1`/`yes`/`on` count as true for booleans; a list or mapping
setting takes JSON, e.g. `UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES='["local", "mix"]'`)
and checked against the field's constraints, as the same value in the file
would be. An invalid value, or a `UKR__` variable that names no config key, is
a usage error: the CLI prints one line naming the variable and exits 2
(`--allow-unknown-keys` turns the unknown name into a WARNING).

The AWS SDK's own `AWS_PROFILE` and `AWS_REGION` fill `aws.profile_name` and
`aws.region_name` only when the config file leaves them unset: as in the SDKs,
an explicit setting wins over the environment, so a shell that exports
`AWS_REGION` (SSO helpers and CloudShell do) does not move a file's region.
`AWS_DEFAULT_REGION` is not read; `UKR__AWS__REGION_NAME` overrides the file
like any other `UKR__` variable.

The variables a deployment usually sets:

| Variable | Sets | Notes |
|---|---|---|
| `AWS_PROFILE` / `AWS_REGION` | `aws.profile_name` / `aws.region_name` | Only when the file leaves them unset. `AWS_REGION` also moves Bedrock calls while `aws.bedrock.region_name` is `null` |
| `UKR__AWS__BEDROCK__REGION_NAME` | `aws.bedrock.region_name` | |
| `UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER` / `__VERSION` | `aws.bedrock.guardrail.identifier` / `version` | |
| `UKR__AWS__NEPTUNE__ENDPOINT` | `aws.neptune.endpoint` | |
| `UKR__AWS__OPENSEARCH__ENDPOINT` | `aws.opensearch.endpoint` | |
| `UKR__AWS__OPENSEARCH__USERNAME` / `__PASSWORD` | `aws.opensearch.username` / `password` | Basic auth when `aws.opensearch.use_iam: false`; the password stays masked in logs |
| `UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS` | `indexing.opensearch.index_settings.number_of_replicas` | Applies to indices created afterwards |
| `UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO` | `indexing.opensearch.language_analyzers.ko` | Applies to indices created afterwards |
| `UKR__AWS__S3__BUCKET_NAME` | `aws.s3.bucket_name` | |
| `UKR__AWS__DYNAMODB__TABLE_NAME` / `__CREATE_TABLE_IF_MISSING` | `aws.dynamodb.table_name` / `create_table_if_missing` | |
| `UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE` | `processing.document_parsing.source_scope` | Set by the container entrypoint to the S3 URI it synced |
| `UKR__OBSERVABILITY__ENABLED` / `__BACKEND` | `observability.enabled` / `backend` | |
| `UKR__LOGGING__LEVEL` / `__LOG_FORMAT` | `logging.level` / `log_format` | |

`run-ingestion` also reads `UKR_SOURCE_DIRECTORY` and `UKR_PIPELINE_ID` (one
underscore: they are not config keys) as the defaults of `--source-directory`
and `--pipeline-id`; an explicit flag wins. `UKR_METRICS_ENVIRONMENT` is the
`Environment` dimension of `--metrics-sink cloudwatch`. The variables the CDK
compute stack injects are listed in
[`iac/README.md`](../iac/README.md#2-build-and-push-the-app-image); incremental
indexing there still needs `aws.dynamodb.enabled: true` in the config file
(the image's config sets it).

The CLIs also load a `.env` file with python-dotenv: only `./.env` in the
current working directory, never one in a parent directory (a `~/.env` setting
`AWS_PROFILE` would otherwise switch accounts unnoticed). A variable already
set in the environment wins over `.env`. Importing the package loads no `.env`
and never changes `os.environ`; a Python caller that wants one loads it itself
(for example `dotenv.load_dotenv()` before `get_config()`).

> **LangSmith tracing uploads content.** When `LANGSMITH_TRACING=true` (or the
> older `LANGCHAIN_TRACING_V2=true`) is set, LangChain sends every traced run to
> LangSmith, including prompts, retrieved document text and model outputs. The
> CLIs log a WARNING at startup when tracing is on but do not turn it off, since
> you may want it. Unset the variable before running on a confidential corpus.
> The warning comes from `setup_logging`, so library code that does not call it
> gets none.

### Observability

Tracing is off by default. With `observability.enabled: true`, every ingestion
run and every query becomes an OpenTelemetry trace that any OTLP-capable
backend (Jaeger, Grafana Tempo, an OpenTelemetry Collector, ...) can show. It
needs the `otel` extra:

```bash
uv sync --extra otel          # or: pip install -e '.[otel]'
```

Enabling tracing without the extra fails at start-up with a
`ConfigurationError` that names the extra; with tracing off nothing
OpenTelemetry-related is imported. `observability.backend` picks where spans
go (`none`: the process's global tracer provider, which a host application or
`opentelemetry-instrument` sets up; `otlp`: an OTLP exporter the package
installs; `langfuse` / `phoenix`: that exporter pointed at the vendor, below),
and `capture_content`, `redact_patterns`, `sample_rate`, `service_name` and
`export_timeout_seconds` are in the
[reference](./configuration.md#observability).

The exporter is configured by the standard OpenTelemetry environment
variables, never by the config file, so a collector API key never lands in it:
`OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`),
`OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TIMEOUT` and
`OTEL_EXPORTER_OTLP_PROTOCOL` (`http/protobuf` by default, `grpc` for port
4317). Inject headers that carry credentials from Secrets Manager or task
secrets at run time.

**Langfuse and Phoenix.** Both ingest OTLP, so these backends need only the
`otel` extra, no vendor SDK. Their settings also come from the environment
only; a standard `OTEL_EXPORTER_OTLP_*` variable wins over the vendor one.

| Backend | Endpoint | Credentials |
|---|---|---|
| `langfuse` | `LANGFUSE_HOST` (e.g. `http://localhost:3000`; the exporter posts to `<host>/api/public/otel/v1/traces`). Required: there is no default host. With `OTEL_EXPORTER_OTLP_ENDPOINT` instead, use `<host>/api/public/otel`. HTTP only; gRPC fails at start-up. | `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (required), sent as a Basic `Authorization` header with `x-langfuse-ingestion-version: 4`. |
| `phoenix` | `PHOENIX_COLLECTOR_ENDPOINT` (default `http://localhost:6006`, or `:4317` with `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`). | `PHOENIX_API_KEY`, when the server needs one, as a Bearer `authorization` header. |

Spans keep their `gen_ai.*` and `kg_rag.*` attributes and gain the vendor's
names on export:
Langfuse `langfuse.observation.type` (`generation` for model calls,
`embedding`, `retriever` for retrieval and rerank, `span` otherwise) and, with
`capture_content`, `langfuse.observation.input`/`output`; Phoenix
(OpenInference) `openinference.span.kind`, `llm.model_name`,
`llm.provider`, `llm.token_count.*` and, with `capture_content`, `input.value`/`output.value`.

**Sessions and users.** Every span of a query carries `session.id` (the
`RAGInput.conversation_id`) and, when set, `user.id` (`RAGInput.user_id`), the
names both Langfuse and Phoenix group traces by. `user_id` is recorded only on
spans and exported as given, so pass an opaque id, not an e-mail address.

**Trace shape.**

- Ingestion: `kg_rag.ingestion.run` (status `ERROR` when the run failed) →
  `kg_rag.ingestion.stage` (one per stage, with input/output counts, status,
  the stage's numeric metrics as `kg_rag.stage.<metric>`, such as dropped
  units, and `error.type` plus the exception when it crashed) → `kg_rag.batch`
  (item and failure counts) → one `chat <model>` span per model call and
  `embeddings <model>` spans.
- Query: `kg_rag.query` (with `capture_content`, the redacted query as
  `input.value` and, outside `stream` / `astream`, the redacted answer as
  `output.value`, which Langfuse and Phoenix show in their trace lists) →
  `kg_rag.strategy` (`kg_rag.rerank_status`, the
  query's [rerank status](#rerank-status)) → `kg_rag.retrieve` (hit counts per
  result type and up to 50 source ids) → `kg_rag.fusion` (in/out counts, the
  step's `kg_rag.rerank_status`) →
  `kg_rag.rerank` (`kg_rag.rerank.status` = `ok`, `empty` or `error`, document
  counts) → `kg_rag.context_build` (tokens used vs the token budget) → the
  answer's `chat <model>` span. `global` adds `kg_rag.global.select` (per
  community level read: `kg_rag.depth`, `kg_rag.reports`, `kg_rag.dynamic`
  with dynamic selection) and `kg_rag.global.map` (`kg_rag.reports`,
  `kg_rag.map_calls`, `kg_rag.report_tokens`) under `kg_rag.strategy`;
  `drift` adds `kg_rag.drift.primer` (`kg_rag.reports`, `kg_rag.answers`) and
  one `kg_rag.drift.actions` per round (`kg_rag.depth`, `kg_rag.actions`),
  whose local searches nest under it. With a guardrail, each screened query and
  answer adds a `kg_rag.guardrail` span under `kg_rag.query`
  (`aws.bedrock.guardrail.id`, `.source` = `input` or `output`, `.action` =
  `none`, `masked` or `blocked`, `.intervened`, `.policies`), and the query
  span carries `kg_rag.blocked`.
- Evaluation (`run-eval`): one `kg_rag.batch` root for answer generation (item
  and failure counts only) → one `kg_rag.query` per question, which carries
  that question's input and output; then one `kg_rag.evaluate` root per
  evaluator (`kg_rag.evaluator`, `kg_rag.items`) → the judge model's
  `chat <model>` spans (`langchain`, `ragas`, and `pairwise_judge` for
  `--pairwise-judge`).
- Model spans follow the OpenTelemetry GenAI semantic conventions (span
  kind `CLIENT`, as for embedding and Bedrock rerank spans):
  `gen_ai.request.model`, `gen_ai.request.max_tokens` / `temperature`,
  `gen_ai.usage.input_tokens` / `output_tokens`,
  `gen_ai.usage.cache_read.input_tokens` / `cache_creation.input_tokens` when
  Bedrock reports prompt-cache use, finish reasons (Bedrock, OpenAI-compatible
  and Ollama), `error.type` on a failed call and
  `aws.bedrock.guardrail.intervened`. Cost is not recorded.

Log records written while a span is active carry its `trace_id` and `span_id`,
so a log line can be found from its trace and the other way round.

**Privacy and export limits.** Spans carry no query text, passages or model
output unless `capture_content: true`. What is exported, how credentials are
handled, and what an unreachable collector costs are in the runbook's
[Export limits, private VPCs and privacy](./operations.md#export-limits-private-vpcs-and-privacy).

Known limits:

- The streamed answer of `stream` / `astream` is traced as its own trace
  (retrieval and context building are in the `kg_rag.query` trace); its
  model span takes `session.id` / `user.id` from the answer step's run
  metadata, so it still groups with its session.
- The async batch path records one span per batch, not per chunk.
- A model call cancelled during `ainvoke` (for example by `asyncio.wait_for`)
  leaves no model span: LangChain reports no callback for it. A stream the
  caller stops reading early ends its span normally.
- With `otlp`, `langfuse` or `phoenix` the first backend the process installs
  also becomes OpenTelemetry's global tracer provider, which can be set only
  once: after a later configuration with different settings, the package's
  spans use the new exporter, but spans that other libraries emit through the
  global OpenTelemetry API are dropped.

For CloudWatch, run an ADOT or OpenTelemetry Collector next to the task and
use `backend: otlp`; on the CDK stacks `-c enable_cloudwatch_tracing=true`
adds that collector and sets the backend (recipe, Transaction Search and the
Bedrock invocation-logging backstop in the
[runbook](operations.md#cloudwatch-through-the-adot-collector-sidecar)). Local
Langfuse and Phoenix recipes are in the [runbook](operations.md#observability)
too.

---

## 3. Ingestion (`run-ingestion`)

Ingestion turns a directory of documents into a knowledge graph in the graph
store and the search and vector store. `run-ingestion --source-directory <dir>`
runs the 12 stages ([Design §4](./design.md#4-ingestion-pipeline) shows them),
checkpoints each one, and resumes after the last valid stage when re-run with
the same `--pipeline-id`. With the doc-status registry on it is incremental
([§5](#5-incremental-indexing)).

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--source-directory` | `$UKR_SOURCE_DIRECTORY` | Directory of source documents. Required to run; if the flag is omitted it falls back to the `UKR_SOURCE_DIRECTORY` environment variable. |
| `--output-directory` | none (no export) | Export parsed documents as JSON here for inspection (must not be the source directory) |
| `--cache-directory` | `cache` | Pipeline cache + intermediate results |
| `--force-rebuild` | off | Recompute every stage, ignoring the cached stage outputs; the cache is rewritten with the new outputs |
| `--s3-sync` | off | Sync cache to S3 (requires `--s3-bucket-name`) |
| `--s3-bucket-name` | — | S3 bucket for cache sync |
| `--s3-prefix` | `pipeline-runs` | S3 key prefix for cache files |
| `--pipeline-id` | `$UKR_PIPELINE_ID` | Existing run to resume/inspect. If the flag is omitted it falls back to the `UKR_PIPELINE_ID` environment variable. The id names the run's cache directory and S3 prefix, so it accepts letters, digits, `.`, `_` and `-` only: up to 128 characters, starting with a letter or digit, with no `..`. |
| `--resume-from-stage` | — | Stage to resume from, by name, case-insensitive (requires `--pipeline-id`); an unknown name is a usage error (exit 2) |
| `--verify-metadata` | off | Verify pipeline metadata integrity (needs `--pipeline-id`); exits non-zero when it is corrupt or the run is not found |
| `--yes`, `-y` | off | Answer yes to confirmation prompts (continuing after a failed integrity check). Without a terminal on stdin a prompt cannot be asked, so the run fails unless `--yes` is passed |
| `--continue-on-error` | off | Keep going when a stage errors. When an artifact stage (translation through community detection) crashes, every document of the run is recorded FAILED, so the next incremental run re-processes them |
| `--enabled-stages` | all | Comma-separated stage list to run |
| `--metrics-sink` | `none` | `none`, or `cloudwatch` (emits CloudWatch EMF — Embedded Metric Format — metrics to stdout, with an `Environment` dimension when `UKR_METRICS_ENVIRONMENT` is set (the CDK task sets its `env_name`) and none otherwise; `pipeline_id` is recorded as a log property, not a dimension, so runs add no new metric series). Each stage's metrics are emitted when the stage finishes, and only by a run that executed it, so a resumed run does not repeat the counts of the stages it restored |
| `--retire-source-scope SCOPE` | — | Incremental runs: treat every registry record of the former source scope `SCOPE` (in this run's index suffix) as deleted, removing its exclusive content. Repeatable; added to `indexing.retire_source_scopes`. Fails when `SCOPE` is the run's own scope. See §5 Deletion scope. |
| `--output-format` | `text` | `text`, or `json` for one JSON result on stdout (see [Exit codes and JSON output](#cli-exit-codes-and-json-output)) |
| `--config-path` | — | Path to `config.yaml`; a missing file is a usage error (exit 2) |
| `--allow-unknown-keys` | off | Warn on unknown config keys (and `UKR__` variables) and go on, instead of failing (see [How configuration is loaded](#how-configuration-is-loaded)) |
| `--backend`, `--llm-provider`, ... | — | Provider selection; see [Provider flags](#provider-flags) |

### The 12 pipeline stages

Run order (`DataIngestionPipeline.STAGE_CLASSES`). Use the stage **names**
(case-insensitive) with `--enabled-stages` / `--resume-from-stage`:

1. **`document_parsing`** — extract text per format (`.pdf`, `.txt`, `.csv`,
   `.json`; `.md`/`.html` with the `unstructured` extra).
2. **`document_loading`** — build the run's corpus from the parsed documents
   (MinHash dedup, incremental filter). If `document_parsing` is disabled, it
   instead loads pre-parsed `Document` `.json` files from the source directory.
3. **`text_chunking`** — split documents into text units (`processing.chunking`).
4. **`translation`** — optional; translate to `target_language` (no-op when
   source == target and no extra targets).
5. **`graph_extraction`** — LLM extracts entities + relationships per chunk.
6. **`gleaning`** — optional iterative refinement passes (`processing.gleaning`).
7. **`graph_resolution`** — merge duplicate entities/relationships (name variants; similar names with `processing.fuzzy_entity_resolution`).
8. **`claim_extraction`** — optional; extract factual claims (off by default).
9. **`claim_resolution`** — optional; dedupe extracted claims.
10. **`graph_analysis`** — builds the graph and logs its statistics (`graph.analysis.statistics`).
11. **`community_detection`** — Leiden clustering + LLM community reports.
12. **`indexing`** — write everything to OpenSearch + Neptune (and DynamoDB
    registry when enabled).

### Examples

```bash
# Full build
run-ingestion --source-directory ./documents --config-path config.yaml

# With S3 cache sync
run-ingestion --source-directory ./documents --config-path config.yaml \
  --s3-sync --s3-bucket-name your-bucket

# Force a clean rebuild (ignore cache)
run-ingestion --source-directory ./documents --config-path config.yaml --force-rebuild

# Resume an interrupted run from a stage
run-ingestion --source-directory ./documents --config-path config.yaml \
  --pipeline-id <id> --resume-from-stage graph_extraction

# Run only specific stages
run-ingestion --source-directory ./documents --config-path config.yaml \
  --enabled-stages DOCUMENT_PARSING,TEXT_CHUNKING,GRAPH_EXTRACTION

# Emit metrics as CloudWatch EMF (auto-extracted by CloudWatch Logs)
run-ingestion --source-directory ./documents --config-path config.yaml --metrics-sink cloudwatch
```

**Resume vs. force-rebuild.** Without `--force-rebuild`, completed stages are
cached and skipped on a re-run:

- `--pipeline-id <id>` resumes that run, at the first failed or incomplete
  stage (with `--enabled-stages`, the first such stage in that set); a stage
  switched on since the pipeline last ran (gleaning or claim extraction, say)
  counts as incomplete. `--resume-from-stage <stage>` re-runs from a chosen
  stage onward. A run without `--pipeline-id` gets a new generated id.
- Before resuming, the CLI checks that every completed stage's output is
  still cached under the current inputs. When it is not (the corpus or the
  config changed), it lists the stages and asks before going on, and the
  resume recomputes them; without a terminal, pass `--yes`.
- With incremental indexing, an automatic resume whose extraction settings
  differ from those its delta was computed with (a stage switched on, another
  model or prompt) restarts at `document_loading`, so the delta and the
  fingerprints recorded on the committed documents follow the new settings
  (`document_parsing` stays cached).
- `--force-rebuild` ignores the cached outputs, recomputes every stage and
  writes the new outputs to the cache, so a later run resumes from them.

**S3 sync** (`--s3-sync --s3-bucket-name <bucket>`) keeps the stage cache in
`s3://<bucket>/<prefix>/...`, so a fresh process (a new Fargate task, say) can
resume without recomputing finished stages:

- The run uploads the files that changed after every stage, so a task that is
  stopped or dies keeps every stage it finished. On SIGTERM (ECS `StopTask`,
  a Step Functions timeout) `run-ingestion` stops the running stage, uploads
  its checkpoint and exits 143; a re-run with the same pipeline id starts at
  that stage.
- A failed upload between stages is logged and retried by the next one. A
  failed or partial initial download or final upload fails the run with
  `CacheSyncError` and a non-zero exit, so the run that lost its checkpoint is the one
  marked failed. An empty remote cache on a fresh pipeline id is not an error.
- S3-synced checkpoints do not expire by `cache.ttl_seconds`; the bucket's
  lifecycle rule expires them (30 days in the CDK deployment). To avoid
  re-embedding unchanged text across runs, also set
  `indexing.opensearch.persist_embedding_cache: true`.

**LLM response cache** (`cache.llm_responses`, on by default): the responses
of the ingestion model calls (extraction, gleaning, claims, translation,
description summaries, community reports, LLM chunking, output fixing) are
kept in the run's cache directory under `llm-cache/`, and synced with the
stage cache. A re-run with the same pipeline id answers a prompt it already
sent from there, so a stage that was stopped, timed out or crashed repays
only the calls it had not made. The key is the rendered prompt plus the model
and its generation settings (model id, max tokens, temperature, reasoning,
guardrail), so a changed prompt, model or setting asks the model again.
Things to know:

- Each cached response is served at most once per run, and only responses of
  earlier runs are served: a retry within the run (after a response failed to
  parse) calls the model again, and the new response replaces the old one. A
  response cut off at the output-token limit is not kept.
- With a temperature above 0 a re-run replays the earlier sample instead of
  drawing a new one; `--force-rebuild` serves nothing from the cache (it still
  stores the new responses).
- A new pipeline id starts with an empty cache. Query and evaluation calls are
  never cached.
- Cost: the cache grows by about one response per model call (a few KB each);
  segments of `cache.llm_responses.flush_every` responses are written (and
  uploaded with `--s3-sync`) as they fill and at every stage checkpoint, so a
  task killed without warning loses at most one segment. The cache is read
  into memory when the run starts.
- A stop also ends the retries of the model calls still running, so the task
  exits once the calls in flight return.

### Index suffix

Every OpenSearch index and Neptune label a corpus uses is named
`<prefix>-<suffix>`, or `<prefix>-<suffix>-<additional_suffix>` when
`indexing.additional_suffix` is set. This guide calls `<suffix>` the **index
suffix**. The same value has a different name on each side:

- ingestion writes it from `processing.document_parsing.index_value`
  (`run-ingestion` has no flag for it);
- queries select it with `--suffix` on `run-rag`/`run-eval`, or
  `RAGInput.suffix` in Python.

Both default to `default`. Use one index suffix per tenant or corpus version,
with one config file each; `indexing.additional_suffix` must be the same on
both sides. Incremental indexing and deletion are scoped to the index suffix
(§5).

The suffix separates names, not access: whoever can query can name any
suffix. A multi-tenant service must derive it from the authenticated caller
on the server, never take it from the request; infrastructure-enforced
options are in [Operator Runbook: Query access and tenant isolation](./operations.md#query-access-and-tenant-isolation).

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

When `source_language == target_language` and `additional_target_languages` is
empty, the translation stage is skipped at no cost, even with `enabled: true`.

- **Analyzers.** Language-aware OpenSearch analyzers are configured under
  `indexing.opensearch.language_analyzers` (e.g. `ko: nori`); unlisted
  languages fall back to `default_analyzer`. The original chunk `text` is
  analyzed for `source_language` and the translated text for
  `target_language`, so set `source_language` to the corpus language even
  when no translation runs: a Korean corpus with `source_language: en` gets
  the `english` analyzer, which keeps Korean words with their particles
  ("홍길동으로부터") as single tokens. Changing `source_language` changes the
  mapping, so re-index the text units.
- **Unicode.** Parsed text and queries are NFC-normalized, so decomposed
  Hangul (NFD, as written by macOS file systems and many PDF text layers)
  matches its composed form.

The analyzer must exist on the cluster: an index whose mapping names one it
lacks is not created, and the indexing stage fails with `the cluster has no
analyzer '<name>'`. `nori` comes from the `analysis-nori` plugin, which the
[local OpenSearch container](#local-stores-in-containers) installs but an
Amazon OpenSearch Service domain has only once its optional Nori package is
associated. The [CDK deployment](../iac/README.md) therefore maps `ko` to the
built-in `cjk` analyzer (overlapping character bigrams, close to what the
embedded engine does for `nori`) through
`UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`, which wins over the
image's config file.

For morphological Korean analysis on the domain, associate the Nori package
(console **Packages**, or `aws opensearch associate-package`; a blue/green
deployment), deploy with `-c opensearch_korean_analyzer=nori`, and re-run once
with `indexing.reset: true` to rebuild the Korean indices.

A text unit whose translation fails keeps its original text and is extracted
in the source language. A failed call for a whole target language fails the
stage unless `processing.ignore_errors` is `true`. Either way the stage reports
the failed units (`failed_units`), and an incremental run records their
documents as failed so the next run translates them again (§5).

---

## 4. Querying (`run-rag`)

`run-rag --query "..."` answers one question, `--interactive` holds a
conversation. Pick a strategy with `--search-strategy` (default `auto`, an LLM
router); every answer returns the sources the answer model actually saw.
[Design §6](./design.md#6-retrieval-two-methodologies) shows how each
strategy retrieves.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--query`, `-q` | — | Single query — provide this or `--interactive` (exactly one is required) |
| `--interactive`, `-i` | off | Interactive chat (auto-enables memory; always `rag` mode) |
| `--mode` | `rag` | `rag` (full generation) or `search` (retrieval only, single `--query` only) |
| `--conversation-id` | — | Name the conversation; its memory lasts while the process runs (see [conversation memory](#interactive-mode--conversation-memory)) |
| `--use-memory` | off | Enable conversation memory, kept in the process (auto in interactive) |
| `--suffix` | — (`default`) | [Index suffix](#index-suffix) to query; must match the corpus's `processing.document_parsing.index_value` |
| `--enable-thinking` | off | Enable model step-by-step reasoning; a no-op for models that always reason (Claude 4.7+ / 5.x, OpenAI GPT), whose depth `aws.bedrock.default_effort` / `fast_effort` set |
| `--search-strategy` | `auto` | `auto`, `drift`, `global`, `local`, `simple`, `mix`, `hybrid`, `naive` |
| `--search-type` | `hybrid` | `hybrid`, `lexical`, `vector`. The lexical side searches the first 150 words of the question (a Hangul, Han or Kana character counts as a word) and tolerates typos only in questions of up to 10 words; the vector side embeds the whole question |
| `--top-k` | `10` | Max search results |
| `--retrieval-multiplier` | `1` | Increase retrieval depth |
| `--disable-query-processing` | off | Skip translation + entity extraction |
| `--filters` | — | `key:value` attribute filters (space-separated) |
| `--output-format` | `text` | `text`, or `json` for one JSON result on stdout (see [Exit codes and JSON output](#cli-exit-codes-and-json-output)) |
| `--verbose`, `-v` | off | Show query-processing info, sources, metrics |
| `--config-path` | — | Path to `config.yaml` |
| `--allow-unknown-keys` | off | Warn on unknown config keys (and `UKR__` variables) and go on, instead of failing (see [How configuration is loaded](#how-configuration-is-loaded)) |
| `--backend`, `--llm-provider`, ... | — | Provider selection; see [Provider flags](#provider-flags) |

### Search strategies — when to use which

**Choosing the methodology.** GraphRAG strategies excel at *summarization and
thematic synthesis* over a corpus (`global` maps over every community report of
one level of the hierarchy).
LightRAG strategies are faster and lean on *dual-level keyword* retrieval — good
for keyword-driven lookups and as a low-cost baseline. Both run through the same
hybrid scorer (BM25 lexical + vector semantic + graph traversal + RRF + Bedrock
rerank); only the retrieval algorithm differs.

**GraphRAG (community-summary):**

| Strategy | Use when | How it works |
|---|---|---|
| `simple` | Fast factual lookups; straightforward questions | Direct OpenSearch vector + keyword retrieval, no graph traversal. Fastest. Includes the claims index when claim extraction is on. |
| `local` | Detailed questions about specific entities/concepts | Extracts query entities → Neptune graph traversal for neighbors/relationships → combined with vector/keyword hits. Injects claims (covariates) when enabled. |
| `global` | Broad, thematic, "what are the main themes" questions | MS GraphRAG global search: maps every community report of one cut of the hierarchy (`search.global_search.community_depth`, default 2 from the root) into scored key points on the fast model, in token-sized batches, and answers from the ranked points. Optional dynamic selection walks the hierarchy from the roots instead. Cost grows with the cut: one map call per `map_context_tokens` of reports. Best for high-level synthesis. |
| `drift` | Multi-part questions that need the broad picture and specific facts several steps away from it | MS GraphRAG DRIFT search: the community reports most similar to a hypothetical answer (HyDE) draft intermediate answers and follow-up queries; each follow-up runs a local search, is answered on the fast model and raises further follow-ups, for `n_depth` rounds; the answer model reduces the ranked intermediate answers and their evidence. The most LLM calls per query of any strategy (about `primer_folds + k_followups * n_depth`). |
| `auto` | You don't know / general use (the default) | An LLM router (`models.roles.strategy_selection`) picks the best strategy from the query among `search.auto_routable_strategies` (default local, mix, global, drift). |

**LightRAG (dual-level keyword):**

| Strategy | Use when | How it works |
|---|---|---|
| `mix` | General LightRAG use; balances graph + chunks | Low-level keywords → entity index, high-level keywords → relationship index, one-hop incident-relationship / endpoint-entity expansion (Neptune multi-hop expansion is opt-in: `search.lightrag_search.enable_graph_expansion`), **plus** naive vector chunk retrieval blended in. |
| `hybrid` | Keyword-driven graph questions | Same as `mix` but without the extra naive chunk blend. |
| `naive` | Fast baseline / comparison eval | Chunk retrieval only, no graph; returns `search.lightrag_search.chunk_stream_top_k` chunks (default 20). Runs `--search-type` like every strategy, so the default `hybrid` adds BM25 to the vector query; `--search-type vector` reproduces upstream LightRAG's pure-vector baseline. |

> For `mix`/`hybrid`, ensure the relationships vector index was built
> (`indexing.opensearch.relationships_index_prefix`, built automatically during
> ingestion) — that is what powers high-level keyword retrieval. Short queries
> that yield no keywords fall back to using the raw query as a low-level keyword
> (gated by `search.lightrag_search.raw_query_fallback_max_len`).

### Examples

```bash
# Single query (auto strategy, hybrid search)
run-rag --query "What are the main themes in the documents?" --config-path config.yaml

# Pick a strategy + search type
run-rag --query "How does entity X relate to Y?" \
  --search-strategy local --search-type hybrid --config-path config.yaml

# LightRAG mode
run-rag --query "Your question" --search-strategy mix --config-path config.yaml

# Retrieval only (no answer generation), JSON output
run-rag --query "..." --mode search --output-format json --config-path config.yaml

# Verbose: show extracted entities, top sources, and metrics
run-rag --query "..." --verbose --config-path config.yaml

# Attribute filters
run-rag --query "..." --filters attr_category:research type:person --config-path config.yaml
```

### Rerank status

Every query reports what reranking did in `metadata.rerank` of the
`SearchResult` and of the RAG output (and so in `--output-format json`):
`status`, `reason` and `stages`, one entry per rerank step with its own
`status`, `scope` (`all`, or the result types it reranked, e.g. `text` for
`local`, `drift`, `mix` and `hybrid`) and `reason`. `--verbose` prints it as one
`Rerank:` line in the execution metrics.

| Status | Meaning |
|---|---|
| `applied` | The rerank model reordered the candidates. |
| `disabled` | Reranking is off: `search.reranking.enabled: false` or `providers.rerank.kind: none`. |
| `unavailable` | Reranking is on but the rerank model could not be built (initialization failed, or the cross-encoder's `[local]` extra is missing); an initialization failure is logged at ERROR. |
| `skipped` | Nothing to rerank: no candidates (including a strategy that returned early), none of a rerank-eligible type, or no query text. |
| `failed` | The rerank call raised (a denied `bedrock:Rerank`, a region without the model, an unreachable endpoint) or returned nothing usable; the fused order was kept. |

A `failed` or `unavailable` reason names what failed without its message: the
AWS error code for a Bedrock error (`AccessDeniedException`,
`ValidationException`), otherwise the exception type.

Every built-in strategy reranks once per query, so `stages` has one entry. A
caller that fuses more than once into the same trace gets one stage per fusion,
and the top-level `status` is the worst stage's, in the order `failed` >
`unavailable` > `applied` > `skipped` > `disabled`. Each result's
`metadata.reranked` flag still marks the results the model reordered. Building a Bedrock reranker makes no AWS call, so a permission
or region problem shows up as `failed` at the first query, not as
`unavailable`. Set `search.reranking.required: true` to fail such queries with
`RerankModelError` instead of answering from the fused order. The status is also
the `kg_rag.rerank_status` attribute of the `kg_rag.fusion` and
`kg_rag.strategy` spans.

### Filters

`--filters key:value ...` (or `RAGInput.filters`) narrows what retrieval ranks.
Filters compile to `term`/`terms`/`range` clauses on OpenSearch and `has`
steps on Neptune. Each store accepts the fields its indexer writes:

| Store | Filterable fields |
|---|---|
| Text units | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.<path>` |
| Entities | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `attr_<key>`, `attributes.<path>` |
| Relationships | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids` |
| Claims | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `description`, `source_text`, `text_unit_ids` |
| Community reports | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `level`, `depth`, `max_cut_depth`, `parent_community_id`, `child_community_ids`, `attr_<key>`, `attributes.<path>` |
| Neptune entity vertices | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `attr_<key>` (where present) |
| Neptune community vertices | `id`, `name`, `level`, `parent`, `size`, `period`, `children` |

On OpenSearch, `attr_<key>` is a document attribute: entry `<key>` of a
document's `filters` metadata is indexed as `attr_<key>`. On Neptune entity
vertices, `attr_<key>` holds the entity's own extracted attributes (for example
`attr_role`). An entity or relationship extracted from several documents
carries every one of their values for a `filters` key (as a list), so it
matches a filter on any of them. Entity and claim `type` and claim `status`
are stored lowercase and `term` matches exactly, so filter on the lowercase
value (`type:person`, not `type:PERSON`). Use exact-match fields (keyword or
numeric) for precise filtering; a `term` filter on an analyzed text field such
as `description` matches single lowercase tokens. A range filter takes
`{"gte": ..., "lte": ...}` through the Python API.

Each filter applies only to the stores that declare its field, so
`type:person` narrows entities, claims, and Neptune entities and leaves text
units unfiltered. The two stores treat `attr_<key>` differently:

- OpenSearch applies `attr_<key>` and `attributes.<path>` strictly on text
  units, entities, and community reports. A document without the attribute is
  excluded, so community reports, which usually lack document attributes, drop
  out of an attribute-filtered query. This is deliberate (fail-closed): an
  attribute filter never returns content it cannot vouch for. A vector or
  hybrid search skips an index none of whose documents match the attribute
  (OpenSearch 2.13 rejects a kNN filter on a field the index does not map).
- Neptune applies `attr_<key>` on entity vertices where present: a vertex
  passes when the property matches or when it has no such property. A
  document-attribute filter such as `attr_category` therefore leaves graph
  expansion intact, while an entity-attribute filter such as `attr_role:buyer`
  removes entities whose role differs. Every other key is strict on both
  stores.

**Filters are not an access-control boundary.** A store that does not
declare a key returns its content unfiltered (relationships and claims have no
`attr_<key>` or `document_ids`, Neptune keeps entity vertices without the
property), and graph expansion and community reports can bring in content
from documents the filter would exclude. Separate tenants or users with
different permissions by [index suffix](#index-suffix), chosen server-side;
see [Operator Runbook: Query access and tenant isolation](./operations.md#query-access-and-tenant-isolation).

A filter key that no store the selected strategy reads declares (for example
`category` or `entity_type`) raises `InvalidFilterError`, whose message lists
the filterable keys. The schema is defined in
`unified_kg_rag/adapters/storage/filter_schema.py`.

### Interactive mode & conversation memory

```bash
run-rag --interactive --config-path config.yaml
# or name the session (its id appears in logs and JSON output):
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

Interactive mode auto-enables memory. In-session commands:

- `help` — list commands
- `new` — start a fresh conversation (new ID)
- `set-filter key:value` — add/update a filter
- `clear-filters` — remove all filters
- `show-config` — show the active configuration
- `quit` / `exit` — end

Conversation memory is held in the process (`MemoryManager`; nothing is
written to disk or a store):

- It carries over between the turns of one interactive session, or between
  calls of one `GraphRAGChain` that reuse a `conversation_id` with
  `use_memory=True` and the same `suffix` and `user_id`. A conversation
  belongs to its index suffix and user, so another tenant's or user's turns
  never reach it.
- Each chain has its own memory; to share it between chains, pass them one
  `MemoryManager(config)` as `memory_manager=`.
- Separate `run-rag --query` invocations do not share it: each starts with no
  history, so `--use-memory` and `--conversation-id` have no effect on a
  single query (`run-rag` logs a WARNING when they are combined with
  `--query`). Use `--interactive` for multi-turn from the CLI.
- The limits are in the `memory` config section.

Before retrieval, a follow-up is rewritten from the history as a standalone
question (one `models.roles.query_rewrite` call per turn that has history;
prompt override key `query_rewrite`), so "what about its suppliers?" is routed,
searched and answered as a question that names the entity from the earlier
turn. The rewrite reads the last `memory.max_messages_per_conversation`
messages, each earlier answer cut to its first 1,000 characters. It runs
whenever `use_memory` finds history, also with `enable_query_processing`
off: that flag covers translation and entity/keyword extraction, while
without the rewrite a follow-up would retrieve nothing useful. Memory adds no
other LLM call per turn.

---

## 5. Incremental indexing

Incremental (delta) indexing re-indexes only documents that are **new or
changed** since the last run, and merges them into the live graph — instead of
rebuilding everything.

### Enable it

```yaml
aws:
  dynamodb:
    enabled: true
    table_name: "unified-kg-rag-on-aws-doc-status-v2"
    create_table_if_missing: true
```

With this on, each `run-ingestion` diffs the corpus against the DynamoDB
document-status registry by **content hash**. In local mode the registry is a
SQLite file instead (`providers.doc_status.kind: sqlite`, switched on by
`providers.doc_status.enabled: true`; see [Local mode](#local-mode-no-aws)).

### Workflows

- **Add a document:** drop the new file into the source directory and re-run
  `run-ingestion`. Only the new file is parsed/extracted/indexed; its entities
  and relationships merge into the existing graph (idempotent `upsert_*`).
- **Modify a document:** edit the file and re-run. The content hash changes, so
  the document is treated as changed: its old artifacts are removed and the new
  version is re-indexed. Only its changed chunks are sent to the model: the
  extraction store (`indexing.opensearch.extraction_records_index_prefix`)
  serves every chunk whose text and extraction settings it already holds a
  record for, in this or any other document (`--force-rebuild` re-extracts
  them all and refreshes the store).
- **Delete a document:** remove it from the source directory and re-run. Its
  **exclusive** artifacts (entities, relationships and claims stated only in
  that document's chunks: the registry lists each document's chunks, and the
  stores are asked what cites them) are deleted; artifacts shared with
  surviving documents are kept, without its chunks. A community or report
  goes once none of the chunks it cites is left.
- **Change the extraction settings** (a tuned `custom_prompts` set,
  `entity_types`, an extraction, gleaning or chunking model, chunk sizes,
  translation, resolution): the next run re-extracts every unchanged document
  extracted with other settings and logs a WARNING with their count and the
  settings that changed (each record stores a fingerprint of its settings).
  - The registry keeps the settings themselves too, compared field by field,
    so a setting a newer release adds is no change while it holds its default.
  - Edits to the built-in prompt templates (code) are not seen: reset after
    editing one.
  - A re-extracted document's chunks still take their stored records when the
    settings that shape a chunk's record (models, extraction, gleaning and
    claim settings, the target language, the prompt templates) are unchanged,
    e.g. after a resolution or chunking change.
- **Change the embedding model or its dimension**
  (`models.embedding.model_id` / `dimension`): an
  incremental run embeds only its new and changed documents, so it would
  mix two models' vectors in one index. Each OpenSearch index records the
  model and dimension it was built with, and the indexing stage stops,
  before any write, when they differ from the run's. Re-run once with
  `indexing.reset: true` to re-embed the corpus. An index that records no
  embedding settings is refused the same way: reset once.
  Queries are checked the same way: `run-rag` and `run-eval` with another
  embedding model or dimension than the index was built with fail with a
  `ConfigurationError` instead of answering from empty search results.
- **Change the corpus or translation languages**
  (`processing.translation` `source_language`, `target_language`,
  `additional_target_languages`, or `indexing.opensearch.language_analyzers`):
  the text fields of a live OpenSearch index keep the analyzers it was built
  with, and a newly translated language would be mapped as a keyword. The
  indexing stage stops, before any write, when the live text fields do not
  match the run's languages: re-run once with `indexing.reset: true`. Dropping
  a language does not stop it.
- **Index a suffix with the registry off** (the `backend: local` preset and
  flag-only runs leave it off): such a run replaces the indices without
  recording their documents, so the registry no longer describes them. Each
  index records whether the registry tracks it, and a later incremental run
  on that suffix stops before any write: re-run once with
  `indexing.reset: true` and the registry on, or keep the registry off for
  that suffix.
- **Rebuild with `indexing.reset: true`**: the run indexes the whole corpus
  without taking the suffix offline. Nothing is cleared first: the vector
  indices are written into new indices whose aliases are swapped as each is
  built, and the graph is rewritten in place, every vertex and edge the
  rebuild writes tagged with it, after which what it did not write is
  dropped. Queries keep reading the previous content meanwhile (the graph
  mixes previous and rebuilt items until the drop). The registry keeps its
  records until the rebuild recorded the corpus; then the namespace's other
  records go. A failed reset drops nothing, so re-running it is safe.

### Deletion scope

A run's **scope** is its index suffix (`document_parsing.index_value`, plus
`indexing.additional_suffix`) and its corpus source
(`document_parsing.source_scope`, by default the resolved source directory;
the container entrypoint sets it to the S3 URI it syncs from). A document's
registry key is its scope and its path relative to the source directory, and a
run only diffs, and treats as deleted, the documents recorded under its own
scope. So:

- a run for another tenant (another `index_value`) never deletes this tenant's
  documents, even when both corpora are staged in the same local directory;
- running a subfolder as its own source directory never deletes the rest of
  the corpus (its files register as separate documents, so do not index the
  same files from two roots into one suffix);
- two corpora indexed into one suffix from different sources are separate
  documents even where both hold the same relative path (`contract.txt` in
  each); each run only updates or deletes its own;
- a file that fails to parse or load is reported as `failed` and keeps its
  indexed content until a run reads it again.

Moving a corpus changes its scope when the scope is the default (the source
directory), and the records of the old scope stay in the registry: no run
deletes another scope's documents, and the run cannot tell a moved corpus from
a second corpus another host indexes into the same registry (a directory that
is missing on this host may exist on that one). The old records keep their
content indexed, including that of files removed or renamed during the move.
To move a corpus, either:

- set `document_parsing.source_scope` to a stable name before moving it, so
  the scope does not change; or
- run once from the new location with
  `--retire-source-scope <old source directory>` (or
  `indexing.retire_source_scopes`). That run treats every record of the old
  scope in its index suffix as deleted: their exclusive text units, entities
  and relationships are removed, content shared with surviving documents is
  kept, and the records are deleted once the removal succeeded. The documents
  at the new location are indexed as new in the same run. Give the scope as
  the registry stores it: the absolute directory or the former
  `source_scope` value, `s3://` URIs included. Trailing and repeated slashes
  of a local path are ignored on both sides, so a former `source_scope` of
  `/mnt/corpus/` is retired by `/mnt/corpus` too. A run
  that names its own scope fails before reading the registry, and a scope
  without records is reported with a WARNING.

A local run with the default scope that has new documents logs one WARNING
naming the other local source directories with records in its index suffix
(from the registry records the diff already reads), pointing to these two
options. It deletes nothing. Set `source_scope` per corpus to silence it when
the directories are separate corpora.

The registry is partitioned by index namespace (the `<namespace>#` prefix of
each document key; the DynamoDB partition key), so a run reads only its own
namespace's records. A table of the earlier `doc_id`-only key schema is
rejected at first use: switch to a new table and run once with
`indexing.reset: true`, see
[Operator Runbook: Incremental runs](./operations.md#incremental-runs).

### Failed documents

A document is recorded `FAILED` when one of its text units failed in
translation, extraction, gleaning or claim extraction, when a base-level
community covering it got no report, when an artifact stage crashed in a
`--continue-on-error` run, or when writing one of its artifacts failed. The
next run treats it as changed, even if the file did not change, and processes
it again, until `indexing.max_document_failures` (default `3`) consecutive
failures of the same content; editing the file or raising the limit retries it
after that. The details and what to do are in
[Operator Runbook: Failed documents](./operations.md#failed-documents).

### Quarantined documents

A document whose indexing kills the run (out of memory, a task timeout) would
stop every later run the same way. After `indexing.max_document_attempts`
(default `3`) runs in a row that started its content and never finished, the
next run removes what the attempts wrote, records it `QUARANTINED`, logs a
WARNING naming the file and indexes the rest; it is skipped until its file or
the extraction settings change. See
[Operator Runbook: Quarantined documents](./operations.md#quarantined-documents).

### Concurrent runs

With the registry, a run holds its index namespace's write lease from its
first stage to its end; a second run on the same namespace stops with
`NamespaceLeaseError` before it reads or writes anything. See
[Operator Runbook: Concurrent runs](./operations.md#concurrent-runs).

### Document size limit

The DynamoDB registry stores each document's text-unit (chunk) ids in one
item, and an item is limited to 400 KB, roughly 10,000 ids. An edited
document's in-progress record holds its old and new chunk ids, so the limit is
about half that for a document you edit. A document over it fails the
indexing stage with an error naming the file, before anything is written;
split it into smaller files.
The SQLite registry of local mode has no such limit.

### Cross-run merge

By default (`indexing.cross_run_merge: true`) a delta run reads the
entities and relationships it touches back from the graph and merges the delta
into them (descriptions, `text_unit_ids`, weights) before the upsert, so an
entity shared with unchanged documents keeps their descriptions and chunk
lineage (which `mix` follows). Merged descriptions over the summarization
budget are re-summarized. `false` overwrites those fields with the delta's
values, and the entity loses its lineage to the unchanged documents' chunks; a
graph adapter without read-back degrades to that.

- **Claims** merge the same way. A claim id is keyed on the entities its
  subject and object resolve to, its type and its dates, not on the document
  or the spelling, so a claim several documents state ("Acme Inc" in one,
  "Acme" in another) is one claim that keeps every document's description,
  source text and text units, read back from the OpenSearch claims index.
- **Changed and deleted documents.** Entities, relationships and claims a
  document shares with others lose its chunks from `text_unit_ids` (the
  weight follows), and their name, type, description, rank and claim text are
  derived again from the chunks left, so what the document contributed goes
  with it.
- **Ids follow the current name.** When another spelling of an entity wins
  (the document that held the winning one is gone, or new documents outvote
  it), the entity takes that name's id, as a full build would give it, and its
  edges and claims follow; stored communities and reports keep the old id
  until a full rebuild.
- **Matching** finds the stored entities a delta name can merge into by their
  compact name key; with `processing.fuzzy_entity_resolution: true` every
  stored name is read instead.
- Extraction-store records that only removed chunks named are deleted after
  each successful run.

---

## 6. Evaluation (`run-eval`)

`run-eval --data-path <dataset.json>` answers every question of a dataset and
scores the answers and their sources. Three evaluators need no LLM and cost
nothing; two use an LLM judge. Several strategies (or `--repeats`) turn the run
into a paired comparison.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--data-path` | **required** | JSON file of questions + ground truths: a local path or an `s3://bucket/key` URI |
| `--output-directory` | `evaluation.outputs_directory` | Where to save results: a local directory or an `s3://bucket/prefix/` URI |
| `--suffix` | — (`default`) | [Index suffix](#index-suffix) to evaluate against |
| `--enable-thinking` | off | Model reasoning, as for `run-rag` (a no-op for models that always reason) |
| `--search-strategy` | `auto` | Strategy used to answer each question; a comma-separated list (`local,mix,global`) runs a [paired comparison](#comparing-strategies) against the first |
| `--repeats` | `1` | Runs per strategy; a comparison averages each query over them and reports the run-to-run spread |
| `--pairwise-judge` | off | With 2+ strategies, also judge each one's answers against the first's with an LLM (see [Comparing strategies](#comparing-strategies)) |
| `--search-type` | `hybrid` | Search method |
| `--top-k` | `10` | Max results |
| `--retrieval-multiplier` | `1` | Retrieval depth |
| `--max-failure-rate` | `1.0` | Exit non-zero when the fraction of queries whose answer generation failed, or the fraction of a metric's attempted values that failed (`metric_outcomes`; skipped values do not count), exceeds this (0.0-1.0). A run where every query, or every attempt of a metric, failed always exits non-zero |
| `--verbose`, `-v` | off | Debug logging |
| `--output-format` | `text` | `text`, or `json` for one JSON result on stdout (see [Exit codes and JSON output](#cli-exit-codes-and-json-output)) |
| `--config-path` | — | Path to `config.yaml` |
| `--allow-unknown-keys` | off | Warn on unknown config keys (and `UKR__` variables) and go on, instead of failing (see [How configuration is loaded](#how-configuration-is-loaded)) |
| `--backend`, `--llm-provider`, ... | — | Provider selection; see [Provider flags](#provider-flags) |

### Evaluators

`evaluation.enabled_evaluators` selects them; all five are on by default. The
three deterministic ones cost nothing and skip a query that lacks their
dataset fields; for a judge-free run set
`enabled_evaluators: [answer_match, retrieval, graph_aware]`.

| Evaluator | LLM | Needs in the dataset | Main metrics |
|---|---|---|---|
| `answer_match` | no | `answer` (+ `answer_aliases`) | `answer_contains` (headline), `exact_match`, `token_f1` |
| `retrieval` | no | `reference_sources` | `hit_at_k`, `recall_at_k`, `mrr` (k = `evaluation.retrieval_k`, default 5) |
| `graph_aware` | no | `expected_entities` / `expected_relationships` | entity and relationship coverage (recall), in the answer and in the retrieved sources |
| `langchain` | judge | `answer` | `correctness`, `partial_correctness` |
| `ragas` | judge | `answer` | `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, `faithfulness` |

An enabled evaluator that cannot be built (no Bedrock access for the judge,
say) or rejects its configuration stops the run; with
`processing.ignore_errors: true` it is dropped instead and listed in
`run_manifest.dropped_evaluators`. Scoring starts only after every answer is
generated, so an evaluator error there never discards the run: the metric is
recorded as failed for that query (as is a judge reply without a usable
score, never 0), the results are saved, and `--max-failure-rate` decides the
exit code.

#### `ragas` contexts

- `context_precision` makes one judge call per context, so it scores at most
  `evaluation.ragas_max_contexts` (default 20; `null` = no cap) top-ranked
  sources per query, applied before the `max_context_tokens` budget: it is
  `context_precision@N`. Without the cap, strategies that report 100+ sources
  (LightRAG `mix`, say) hit `ragas_timeout`.
- `faithfulness` and `context_recall` see every source within
  `max_context_tokens`, so a claim backed by a lower-ranked source is not
  marked unsupported. By default (`evaluation.max_context_tokens: null`) the
  judges get the answer model's budget (`search.token_manager.max_context_tokens`),
  so they grade the answer against the context it was given; a smaller value
  logs a warning.
- Each report records what the judge saw (`judge_contexts` /
  `judge_context_tokens`, `context_precision_contexts` /
  `context_precision_context_tokens`). None of this changes what the answer
  model saw.

#### `graph_aware`

Of the expected graph artifacts, how many appear in the generated answer
(coverage = recall), with the phrase matcher of `answer_contains` (whole-word
match, Korean particles tolerated, substring match for single-word CJK text).
A relationship given as `{"source": "A", "target": "B"}` or `"A -> B"` counts
when the answer mentions both endpoints; any other string must appear as a
phrase. Precision and F1 are not emitted: every entity of a free-text answer
cannot be enumerated reliably, so they would only re-label recall.

Answer-text coverage rises with answer length, so the evaluator also reports
`retrieved_entity_coverage` / `retrieved_relationship_coverage` (the same
expectations matched against the entity names and relationship endpoints of
the sources the answer model was given, a pair matching one relationship in
either direction; skipped when the context had no source of that type, as with
`global`) and `answer_length` (characters). Compare strategies on the
retrieved coverage, and read answer-text coverage next to the length.
Skipped, not 0, is deliberate: a strategy that reads no entity sources by
design (`global`) has no retrieved coverage to measure. The cost is that a
strategy that finds no entity source for a query is not scored on it either,
so in a comparison check the metric's `pairs` and `unpaired`, and
`metric_outcomes`' skipped count in a single run.

#### `retrieval`

Did the sources the answer model saw include the gold documents?

- **Attribution.** Text-unit sources name their file directly; entity,
  relationship and community-report sources are attributed to the files of
  the text units they cite (`text_unit_ids`, looked up in one batch per
  suffix), the file holding most of those text units first.
- **k counts files, not sources.** The sources are flattened into one list of
  distinct files in rank order. A community report is attributed to every
  file of its community, so counting it as one source let five random reports
  score hit@5 0.98 on a 50-file corpus where five random text units score
  0.19.
- **Per source type.** `by_source_type` (aggregated in
  `grouped_statistics.source_type`) has the same metrics over each section
  type's sources alone; its `text` row scores only the text units the answer
  model read verbatim, the fairest cross-strategy comparison.
  `attributable_fraction` (attributable / reported sources) shows how much of
  the context the rank metrics cover.
- **Matching.** NFKC-normalized and case-insensitive on the file name (so
  decomposed NFD Hangul file names and full-width letters match); a
  reference's directories are ignored. A reference without a file extension
  (`.` + 1-5 letters or digits with at least one letter, not a version such
  as `.v2`) also matches a source's stem; one with an extension needs the same
  extension: `docs/Terms.pdf`, `terms.pdf` and `terms` all match the source
  `Terms.pdf`, `terms.docx` does not, and `notes.v1` does not match
  `notes.v2`. A `/` separates directories only in a path-like value (a file
  extension, a URI scheme, or a leading `/`, `./`, `~/`), so titles such as
  `AC/DC` are compared whole.
- **Skips.** The metrics are skipped when a query has no `reference_sources`,
  or has sources but none is attributable; a query that retrieved no sources
  at all scores 0 (a miss).

#### `answer_match`

Scores against `answer` and the optional `answer_aliases` (a string or a
list, top level or under `metadata`, which wins), taking the maximum over
them.

- **`answer_contains`** is the headline: 1.0 when the gold answer or an alias
  appears in the generated answer as a whole-word phrase. Long-form RAG
  answers rarely equal a short gold span, so it tracks correctness far better
  than exact match.
- **Korean and CJK.** Korean particles are tolerated after a word (`서울 특별시`
  matches `서울 특별시는`, `AWS` matches `AWS는`). A bare number may
  carry a Korean unit (`2024` matches `2024년에`) but not a magnitude numeral
  (`3` does not match `3억`). Spacing next to CJK letters is ignored
  (`3억 원` = `3억원`). A single-word CJK gold is matched as a substring, but
  not inside a longer number or Latin word (`2년` does not match `12년`).
- **Yes/no golds** (`yes`/`no`/`nope`; Korean `예`/`네`/`아니요`/`아니오`/
  `아닙니다`; Chinese `是`/`对`/`否`/`不是`/`没有`; Japanese `はい`/`いいえ`) are
  scored by the answer's polarity, as in HotpotQA, since almost any long
  answer contains the word "no" (or `예` in `예외`). The answer is the first
  whole yes/no word that answers the question, skipping Markdown headings, a
  word next to `/` (the options in `Yes/No: No`), `no` and `네` unless
  punctuation or a line end follows them (`there is no clause`, `네 가지`),
  `예` followed by `:` or `)` (`(예: …)`, "e.g.") and `no` followed by a
  number (`No. 5`). So `네, 맞습니다` answers yes but `네 맞습니다` is not
  read as an answer.
- **`exact_match` and `token_f1`** follow the SQuAD v1.1 normalization
  (NFKC, lowercase, punctuation and English articles removed, whitespace
  collapsed), except that Unicode punctuation is removed too and punctuation
  that changes meaning is kept: a leading minus (`-5` is not `5`), a decimal
  point or comma between digits (`1.5` is not `15`; `1,000` = `1000`), a
  percent sign after a number, and a trailing `+`/`#` (`C++`, `C#`). A dash
  between two words or numbers is a space (`1990-1995` is two numbers), and a
  gold that is only an article (`A`) is kept as is. Scores
  are therefore close to, but not comparable with, leaderboard numbers; set
  `evaluation.answer_normalization: official_squad` for the official script.
  Token F1 splits on whitespace, so for Chinese and Japanese it degrades to
  exact match, and Korean particles make both metrics under-count.

### Eval data format

A JSON array of objects. Only `question` is required; everything else is
optional. `expected_entities` / `expected_relationships` are required *only* for
the `graph_aware` evaluator.

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

Per-item `metadata` (e.g. `search_strategy`) overrides the CLI defaults for that
question. Mark a question the corpus cannot answer with
`"metadata": {"answerable": false}`: it is not graded by any evaluator, only on
whether the chain abstained (see `abstention_statistics` below). `id` may also be given as `query_id`. The file is validated before any
query runs: an empty dataset, a non-array file, a missing `question`, a
duplicate id, a wrong field type, or a `metadata` value the RAG chain rejects
(e.g. an unknown `search_strategy`) stops the run with the item index and id.

### Examples

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json --output-directory ./results --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

Either path can be an `s3://` URI, so `run-eval` also runs as a one-off task on
the CDK stack ([After deploy](../iac/README.md#5-evaluate-from-inside-the-vpc)).
An `s3://` dataset is downloaded to a temporary file before the run. With an
`s3://` outputs directory the result files are written to a temporary
directory and uploaded under the prefix when the run ends, with the
`aws.s3.encryption` SSE settings the cache uploads use. If any upload fails,
`run-eval` exits non-zero and keeps the local files (their path is printed).
S3 calls use `aws.profile_name` and `aws.region_name`. The principal needs
`s3:GetObject` on the dataset and `s3:PutObject` on the outputs prefix.

A dataset that cannot be read (missing file, invalid JSON) or fails
validation (an empty array, a missing `question`, a duplicate id, a field of
the wrong type such as a non-list `expected_entities` or an `answer_aliases`
that is not a string or a list of strings, metadata the RAG chain rejects)
stops `run-eval` before any query with `Error: invalid evaluation dataset:`,
the item index and query id, and exit code 2.

Results are written to the outputs directory as
`evaluation_{results,reports,summary}_<timestamp>.json`; when every query
asked for the same strategy (or, with none asked for, used the same one) the
name becomes `..._<strategy>_<timestamp>.json`, so an `auto` run is
`..._auto_...` whatever it routed to.
`<timestamp>` is `YYYYMMDD_HHMMSS_<8 hex characters>`: the random part keeps
runs started in the same second (parallel strategies, CI matrices) from
overwriting each other's files, so a script finds the latest run by sorting
the names or by file time, not by building the name.
The summary holds, per metric, mean/median/stdev/min/max/count
(`metric_statistics`) and scored/failed/skipped counts (`metric_outcomes`), plus:

- `grouped_statistics` — the same statistics split by `search_strategy` (the
  strategy actually used, which varies per query under `auto`), `category` and
  `difficulty`, plus `source_type`: the retrieval metrics per source section
  type (`text`, `entity`, `relationship`, `community`, ...).
- `abstention_statistics` — how often the chain returned its fixed
  no-context reply ("I could not find relevant information…") instead of an
  answer: `abstained`, `answered`, `abstention_rate`, the same `per_strategy`,
  and for `answerable: false` items an `unanswerable` block of the same shape.
  Only the fixed reply counts, and the chain returns it only when retrieval
  finds nothing: a model that declines in its own words is not detected, so on
  a populated index the `unanswerable` rate is a lower bound on correct
  declines, not an accuracy. On answerable items an abstention is graded like
  any answer (normally a miss); each result carries `abstained`.
- `rerank_status_counts` — queries per rerank status, e.g.
  `{"applied": 48, "failed": 2}`; each result carries `rerank_status`
  (`applied`, `disabled`, `unavailable`, `skipped`, `failed`, or `null` when
  answer generation failed first; see [Rerank status](#rerank-status)).
- `run_manifest` — what was measured, so two runs can be compared:
  - the CLI arguments (a `*_url` argument such as `--model-base-url` without
    its userinfo, query and fragment), the package version, `library_versions`
    (ragas, langchain*), the git commit (`git_sha`, and `git_dirty` when
    tracked files differ from it; both `null` outside a checkout), the dataset
    (path, file sha256, query count and a hash of the parsed file, without the
    metadata the command line adds to every query), a UTC timestamp and
    `dropped_evaluators`;
  - the model ids (answer generation, evaluation judge and embedding) and
    `judge_is_answer_model`: LLM judges favour their own outputs, so a judge
    that is the answer model (the default: both use the `default` tier)
    inflates the judge scores; set `models.roles.evaluation` to another model
    when comparing systems;
  - `retrieval_config_sha256`, a hash of every setting that can change an
    answer or a score (the `models` roles resolved to their ids and the
    ingestion settings included: an index built otherwise answers otherwise).
    The settings the stage-cache classification (`CONFIG_KEYING`) marks as
    changing no answer are left out: regions, endpoints, credentials, store
    and output paths, table names, the guardrail id and version, timeouts,
    retry and concurrency knobs, cache, logging and telemetry. So the same
    evaluation hashes the same in every environment, and a new setting counts
    until it is classified as changing no answer;
  - `index`, per queried suffix, the ingestion run that last wrote its vector
    indices, as recorded in the index mappings' `_meta`: `pipeline_id`,
    `mode` (`full`, also for an incremental run that found the registry
    empty, or `incremental`), `built_at`, `documents` (an incremental run
    counts only its new and changed ones), `extraction_fingerprint`,
    `extraction_settings`, `package_version` and, on an incremental record,
    `full_build` (the full build it was applied on top of, with its own
    `pipeline_id`, `built_at` and `documents`) (best-effort: a role without
    the mapping-update permission logs a warning; `null` for a backend
    without index metadata).

  `EvaluationManager.evaluate_dataset` builds it, so library callers get it
  too (pass `dataset_path=` / `cli_args=` to record those).

Per-query reports list each evaluator's `metrics` (there is no overall
score: an average of unrelated metrics has no meaning). Each result also records `retrieved_source_ids`: per reported source, in rank
order, the file names it is attributed to (`[]` when unattributable), and
`retrieved_source_types`, `retrieved_entities` and `retrieved_relationships`
(the section type of each source and the graph elements the model was given).

### Comparing strategies

`run-eval --search-strategy local,mix,global --repeats 3` answers and scores
the same queries with each strategy, `--repeats` times each (the listed
strategy overrides a per-item `search_strategy`), saves every run's files as
usual and adds `evaluation_comparison_<timestamp>.json`:

- `strategies.<s>.metrics.<evaluator>/<metric>` — the mean over queries of
  each query's mean over the repeats, the number of queries, and
  `repeat_stdev` (the spread of the per-run means: the answer model's
  run-to-run noise). `routed_strategies` counts the strategy each answer
  actually used, so `auto` shows how it routed.
- `deltas_vs_baseline.<s>.<metric>` — paired against the first strategy over
  the queries both scored: `mean_delta`, a 95% paired bootstrap confidence
  interval (`ci_low`/`ci_high`; 2,000 resamples of the queries, seed 0),
  `wins`/`ties`/`losses`, the per-query deltas and `unpaired` (queries only
  one side scored, such as a metric skipped for lack of an entity source). A
  query whose answer generation failed scores 0 on every metric another run
  scored it on (`answer_length` excepted), so a strategy cannot look better by
  failing on the queries it would lose; `strategies.<s>.failed_answers` /
  `failed_queries` count the failures, and `warnings` (printed too) names
  strategies that failed different numbers of answers. A difference is
  supported by the query sample when its interval excludes 0. In a simulation with
  binary metrics, a 10-point difference was detected in 15% of runs with 50
  queries and 59% with 200 (the interval held the true value in 95-98% of
  runs), so compare on a few hundred queries.
- `pairwise_vs_baseline.<s>` (with `--pairwise-judge`) — the GraphRAG /
  LightRAG head-to-head LLM judge on comprehensiveness, diversity,
  empowerment and directness, using `models.roles.evaluation`
  (prompt: `custom_prompts.pairwise_comparison_*`). Each pair is judged in
  both orders and a criterion counts as a win only when both orders agree
  (the judge favours the answer it reads first); `candidate_win_rate` counts
  a tie as half. A pair where one answer failed is not judged: the other
  side wins every criterion (`candidate_errors`, `baseline_errors`); a pair
  where both failed counts in `both_errors` only. `failed` counts the pairs
  the judge could not decide, and `--max-failure-rate` applies to them as to
  a metric (every judge call failing is a breach). It judges the first
  repeat's answers, one call per order, query and strategy, under one
  `kg_rag.evaluate` span (`kg_rag.evaluator` = `pairwise_judge`).
- `run_manifest` of the first run.

The console prints the means, the deltas with their intervals, `auto`'s
routing and the pairwise win rates. The failure budget applies to every run.

### Benchmarks (MuSiQue)

`prepare-benchmark musique` turns
[MuSiQue](https://github.com/StonyBrookNLP/musique) (answerable, multi-hop,
CC BY 4.0; Trivedi et al., TACL 2022) into a corpus and a `run-eval` dataset.
Nothing of it is in this repository: by default it downloads the answerable
dev split (2,417 questions) from a pinned revision of a public mirror and
checks its SHA-256; `--data-path` reads a local `musique_ans_v1.0_*.jsonl` from
the official download instead. It writes `corpus/` (one text file per
distinct paragraph: its title, a blank line, its text; each question brings
2-4 supporting paragraphs and up to 18 distractors) and `eval_data.json`
(`answer`, `answer_aliases`, the supporting paragraphs' file names as
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

`answer_contains` and `token_f1` are the answer scores to read (long-form RAG
answers rarely equal a short gold span exactly); `retrieval` scores whether
the supporting paragraphs reached the context, per `category` in
`grouped_statistics`. Ingestion cost grows with the paragraph count; each
`--limit` question adds about 19 paragraphs. Cite the paper when you publish
results.

---

## 7. Visualization (`run-visualization`)

This is a **standalone** renderer that draws from an already-exported
visualization-data JSON. It does **not** re-run ingestion or touch AWS.

The community-detection stage of `run-ingestion` writes
`visualization_data.json`, the `--data-path` input of `run-visualization`, on
every run whose graph covers the whole corpus: a full build, a reset, or an
incremental run in which every document is new or changed. An incremental run
that skips unchanged documents holds only its delta, so it neither exports nor
renders (it logs an INFO line), and an earlier file in a fixed
`outputs_directory` is left as it was; run one full build
(`indexing.reset: true`) to visualize the whole graph after incremental runs.

- **Where.** `graph.visualization.outputs_directory`, by default
  `<cache directory>/<pipeline_id>/visualization/` (`--cache-directory`), so
  with S3 cache sync the file is uploaded with the cache (the sync copies
  `.json` files only; re-render the HTML locally with `run-visualization`).
- **What.** The graph nodes and edges (each node with its level-0
  `community_id`, which the renderers color by) and the community hierarchy;
  vector attributes are left out to keep the file small. By default
  (`graph.visualization.export_data: true`, `enabled: false`) writing it is
  one pass over the graph with no embedding or model call, and
  `run-visualization` computes a seeded spring layout and the configured
  centrality when it renders. With `graph.visualization.enabled: true` the
  stage also renders, and the file carries the embedding-based layout and
  centrality (one embedding call per node and a layout reduction per ingest).
- **Size.** The interactive graph keeps only the top `interactive.max_nodes`
  nodes by degree (default 2000) so large graphs still render in a browser;
  the layout is computed for those nodes only.

Layout and error behaviour:

- `embedding_method: "none"` uses a spring layout and never constructs an
  embedding client.
- `embedding_method: "description"` embeds each node's `name: description` with
  the embedding provider (`providers.embedding`, Bedrock by default) and reduces it with `layout_method`. If embedding fails, the
  visualization step fails when `processing.ignore_errors` is `false` (ingestion
  itself continues and logs the failure). With `ignore_errors: true` it logs an
  ERROR, falls back to a topology-only spring layout, and records
  `"layout_degraded": true` in `visualization_data.json`. A failed
  dimensionality reduction falls back the same way (seeded spring layout,
  `layout_degraded: true`) whatever `ignore_errors` says.
- Node embeddings use the embedding model and dimension of the index
  (`models.embedding`).
- Edge width/opacity in the interactive graph is scaled relative to the graph's
  own weight range (log-scaled, then min-max normalised), so 1-10 strength
  scores and merged counts are both distinguishable.
- Only files a renderer actually wrote are reported (an earlier run's file of
  the same name is removed first). If nothing is rendered (e.g. an empty graph),
  `run-visualization` logs an error and exits with status `1`.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--data-path` | **required** | Exported visualization-data JSON |
| `--output-directory` | `visualization_outputs` | Where to write rendered files |
| `--renderers` | all registered | Renderers to run: `interactive`, `static` |
| `--output-format` | `text` | `text`, or `json` for one JSON result on stdout (see [Exit codes and JSON output](#cli-exit-codes-and-json-output)) |
| `--config-path` | — | Path to `config.yaml` |
| `--allow-unknown-keys` | off | Warn on unknown config keys (and `UKR__` variables) and go on, instead of failing (see [How configuration is loaded](#how-configuration-is-loaded)) |
| `--backend`, `--llm-provider`, ... | — | Accepted like on the other CLIs (no model or store is called); see [Provider flags](#provider-flags) |

The two registered renderers are **`interactive`** (pyvis) and **`static`**
(Bokeh). Their settings live under `graph.visualization` (`interactive.*`,
`static.*`, plus `embedding_method`/`layout_method`).

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

---

## 8. Prompt tuning (`run-prompt-tuning`)

Samples documents from a directory, profiles the corpus (domain / language /
persona / entity types) with the configured language model
(`models.roles.graph_extraction`), and writes a domain-adapted
`custom_prompts` YAML fragment for you to review and merge into `config.yaml`.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--source-directory` | **required** | Directory of documents (`.txt`/`.md`/`.markdown` plus every format `run-ingestion` parses, e.g. `.pdf`) |
| `--output-path` | `tuned_prompts.yaml` | Output YAML path |
| `--max-docs` | `20` | Max documents to sample |
| `--output-format` | `text` | `text`, or `json` for one JSON result on stdout (see [Exit codes and JSON output](#cli-exit-codes-and-json-output)) |
| `--config-path` | — | Path to `config.yaml` |
| `--allow-unknown-keys` | off | Warn on unknown config keys (and `UKR__` variables) and go on, instead of failing (see [How configuration is loaded](#how-configuration-is-loaded)) |
| `--backend`, `--llm-provider`, ... | — | Provider selection; see [Provider flags](#provider-flags) |

```bash
run-prompt-tuning --source-directory ./source --output-path tuned_prompts.yaml --config-path config.yaml
```

The output YAML contains a `custom_prompts` block and a `profile` with the
detected domain and entity types. **Review it**, then copy the prompts you want
into your `config.yaml` under `custom_prompts:`.

- The tuned `graph_extraction_system` and `community_report_system` are a
  domain-adapted preamble followed by the built-in rules and output format,
  verbatim; few-shot examples, when generated, come last.
- The entity categories still come from
  `processing.graph_extraction.entity_types` (the `{entity_types}` variable),
  so copy the profile's `entity_types` there to hold the model to them.
- Plain-text files are read as-is; other formats (PDF, CSV, JSON, custom
  `ParserFactory.register_loader` formats) go through the ingestion loaders,
  and files that fail to parse are skipped with a warning.
- If the profiling model returns no JSON profile, the command exits non-zero
  and writes nothing (a default profile would look tuned).

---

## 9. Domain adaptation

Two complementary levers turn a generic pipeline into a domain-specialized one
(medical, legal, finance, etc.):

### A. `entity_types` (cheapest, highest-impact)

Override the entity categories injected into the extraction prompt — no prompt
rewrite needed:

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

Override any prompt's `*_system` / `*_human` text (defaults are `null` = use the
built-in prompt). Variables in `{braces}` are filled by the framework — keep
them. Common overrides:

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

Available override keys (each `_system` + `_human`): `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `query_rewrite`,
`entity_extraction`, `drift_hyde`, `drift_primer` and `drift_action` (DRIFT
query expansion, primer and follow-up answer),
`strategy_selection`, `keywords_extraction` (LightRAG dual-level),
`global_map` (global-search map-reduce), `pairwise_comparison` (the
`run-eval --pairwise-judge` judge; keep its JSON verdict format), plus the prompt-tuning
`corpus_profile` prompt.

Overrides are checked when the config loads, and a mistake fails the load
with the offending key:

- A `{name}` the prompt is not called with is an error (it would raise
  `KeyError` on every call). Write a literal brace, for example in a JSON
  sample, as `{{` and `}}`: `{{"name": "Vendor"}}`.
- Each prompt's data variables must appear in its system or human template:
  `{input_text}` for `graph_extraction` and `claim_extraction`, `{query}` and
  `{context}` for `answer_generation`, and so on. The other variables (limits
  such as `{max_entities_per_chunk}`, `{entity_types}`) may be left out.
- `run-prompt-tuning` output is already escaped.

A `*_system` override replaces the whole built-in system prompt, including its
output instructions. For `graph_extraction` and `community_report` that is the
XML format the parser reads (and, for extraction, the verbatim `<source_text>`
rule the hallucination guard checks), so a replacement must keep it. Start from
`run-prompt-tuning` output, which puts a domain preamble in front of the
built-in rules and format, or copy the format section from
`unified_kg_rag/domain/prompts/graph_extraction.py`. When an override of these
two prompts no longer mentions a tag the parser reads (`<entities>`,
`<relationships>`, `<source_text>`; `<community_name>`, `<summary>`, `<rating>`,
`<rating_explanation>`, `<findings>`), loading logs a warning naming the missing
tags.

**Recommended flow:** run `run-prompt-tuning` to generate a starting point →
review → merge the useful prompts + tune `entity_types` by hand → re-ingest.

---

## 10. Operations & troubleshooting

The [IAM permissions](#iam-permissions) are in §1, and the messages the CLIs
report, what they mean and the fix, for AWS and local mode, are in
[Operator Runbook: Common errors](./operations.md#common-errors).

### CLI exit codes and JSON output

Every CLI (`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`) exits with:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The run failed: a failed stage or query, a breached `--max-failure-rate`, a failed upload, nothing rendered |
| `2` | Usage or configuration error, before any work: an unknown or invalid flag, a missing input file or directory, an invalid configuration file or environment override, an unset store endpoint |
| `130` | Interrupted with Ctrl-C (SIGINT) |
| `143` | Stopped by SIGTERM (ECS `StopTask`, a Step Functions timeout); `run-ingestion` checkpoints its finished stages first, and `run-eval` with an `s3://` output directory uploads the results of the runs that finished (or keeps them locally and logs the path when the upload fails) |

Flags name files one way across the CLIs: `--source-directory` (a corpus
directory), `--data-path` (an input data file), `--output-directory` /
`--output-path` (what the command writes), `--cache-directory` and
`--config-path`. Flags must be spelled out in full: a prefix such as
`--output` is rejected rather than matched to a longer flag.

`--output-format json` prints one JSON document with the result to stdout
and moves the banner, progress and logs to stderr, so the output pipes into
`jq`: the answer or search results (`run-rag`), the pipeline id, status and
per-stage results (`run-ingestion`; not with `--metrics-sink cloudwatch`,
which also writes to stdout), each run's summary, the comparison and the
failure-budget breaches (`run-eval`), the written files (`run-visualization`),
the output path and detected profile (`run-prompt-tuning`), and the counts
written (`prepare-benchmark`). A run stopped by a signal before it printed its
document prints `{"status": "interrupted", "exit_code": 143}` (`130` for
Ctrl-C) instead, so stdout is always JSON.

### Large / multilingual / heterogeneous corpora

- **Large corpora:** raise `processing.max_concurrency` (LLM stages are
  I/O-bound) as far as the Bedrock quota allows. Keep
  `indexing.neptune.index_concurrency` at `1` unless you measured a gain:
  concurrent write batches can conflict on shared vertices and retry. Enable `indexing.opensearch.persist_embedding_cache` + `--s3-sync`
  so re-runs don't recompute.
- **Multilingual:** set `processing.translation.source_language` /
  `target_language` (+ `additional_target_languages`) and add language analyzers
  under `indexing.opensearch.language_analyzers`. The translation stage no-ops
  for single-language corpora.
- **Heterogeneous domains:** tune `entity_types` to the union of your domains,
  or keep separate indices per domain or tenant: ingest each with its own
  `processing.document_parsing.index_value` and query it with the same value
  as `--suffix` (`indexing.additional_suffix` adds a second segment and must
  match on both sides).
- **Incremental:** enable DynamoDB so large corpora only pay for the changed
  delta on subsequent runs.

### Cost notes

LLM calls dominate cost. The biggest drivers: `graph_extraction` (one+ call per
chunk), `gleaning` (up to `max_rounds` calls per chunk), `community_detection` report
generation, `claim_extraction` (one call per text unit — off by default), and
answer generation per query. Levers: use cheaper models for mechanical stages
(chunking / translation / map-reduce / description summarization already default
to Haiku-class models), cap `gleaning.max_rounds`, leave `claim_extraction`
off unless needed, enable embedding/stage caching, and use incremental indexing
to avoid full re-ingests.


---

## 11. Using the library from Python

The CLIs are thin wrappers over two public classes. Import them, and
everything else you use, from the package root: the names `unified_kg_rag`
exports (`unified_kg_rag.__all__`) are the public API, and other module paths
are internal and may change between releases. Build the `Config` with
`get_config`, which applies the same file and environment-variable layers as
the CLIs (§2).

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

# Ingest. PipelineConfig carries the run options the run-ingestion flags set
# (pipeline_id, cache directory, S3 sync, enabled stages, force_rebuild).
# run() ingests the directory it is given (the constructor's source_directory
# defaults to processing.document_parsing.source_directory and is replaced by
# it): parsing, loading and the incremental registry scope all use it.
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

- `pipeline.run` returns a `PipelineContext`. A failed stage sets its `status`
  to `FAILED` rather than raising, so check it as above (the CLI exits non-zero
  in that case).
- `RAGInput` takes the `run-rag` options as fields: `search_strategy`,
  `search_type`, `top_k`, `suffix` ([index suffix](#index-suffix)), `filters`,
  `conversation_id` and `use_memory`, `target_language`. A plain dict with
  these keys also works.
- `chain.search(...)` (`await chain.asearch(...)`) runs the same retrieval
  without generating an answer and returns a `SearchOutput`: the ranked
  `search_results`, the `processed_query` and `metadata`.
- In async code, use `await chain.ainvoke(...)` and `await chain.aclose()`.
  An async host should also size the event loop's executor once at startup:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`
  (from `unified_kg_rag`).
- Always close what you build. Both classes open connections on first use;
  without `close()`/`aclose()` they stay open until garbage collection.
- Importing the package configures no logging handler; call
  `unified_kg_rag.setup_logging(config)` to get the CLI's logging.
  Without it the package's records reach your handlers with the plain
  message, and structured fields (including values you bind with
  `structlog.contextvars`) become record attributes; a field named like a
  `LogRecord` attribute (`name`, `process`, `msg`, ...) gets a trailing `_`.

Extensions import from the package root too: the ports (`LLMFactoryPort`,
`GraphIndexer`, `DocStatusPort`, ...), `Providers`, `register_provider`,
`BaseSearchStrategy` with `register_strategy`, `ParserFactory` and
`register_renderer`. Both constructors accept injected backends through a
`providers=` bundle
(`Providers(config, llm_factory=..., doc_status=..., vector_indexer_factory=...,
graph_indexer_factory=..., retriever_builders=...)`). Without a bundle,
`DataIngestionPipeline` also takes `doc_status`, `vector_indexer` and
`graph_indexer`, and `GraphRAGChain` takes `model_factory` and
`retriever_builders`; passing one of these together with `providers=` raises
`ValueError`. `GraphRAGChain(memory_manager=...)` combines with either. See
[Design Doc §15](./design.md#15-extension-guide).

---

## 12. Limitations

- **Extraction can hallucinate.** Graph extraction is an LLM step and can emit
  entities or relationships that the source text does not support, which then
  reach retrieval. `processing.graph_extraction.entity_grounding.enabled`
  drops (or down-weights) items whose quoted `source_text` is absent from the
  chunk; it is off by default.
- **Runs are not reproducible.** Extraction, gleaning, community reports, the
  `auto` router and answers are LLM outputs, so two ingestions of the same
  corpus build different graphs and the same query can get different answers.
  Leiden clustering is deterministic for a given graph, and everything else
  is too: given the same model responses, ingestion and queries produce the
  same ids, graph, prompts and answers in any process (whatever its
  `PYTHONHASHSEED`). Compare configurations over more than one ingestion and
  query run.
- **Reranking is regional.** The rerank models are not offered in every
  Bedrock region (see [Required models](#required-models)). Without reranking,
  results are ranked by fusion only; `metadata.rerank` shows it per query
  ([Rerank status](#rerank-status)).
- **Three model providers ship.** Amazon Bedrock (the default), Ollama and
  OpenAI-compatible servers for language models and embeddings, and Bedrock or
  a sentence-transformers cross-encoder for reranking
  ([Local mode](#local-mode-no-aws)). Other providers need a builder
  registered with `register_provider`
  ([Design Doc §15](./design.md#15-extension-guide)).
- **Two store pairings ship.** Neptune (or a TinkerPop Gremlin Server for
  local development) with OpenSearch, and the embedded SQLite graph and
  vector stores for local mode (single host, about 100,000 documents per
  index). Other stores need custom adapters.
- **No serving layer.** The package provides CLIs and a Python library; it has
  no HTTP API, end-user authentication or rate limiting.
- **Filters are not access control** (§4). Separate tenants by index suffix.
- **A lighter ingestion limits the strategies.** With
  `graph.community_detection.enabled: false`, `global` and `drift` have no
  community reports to use; with
  `indexing.opensearch.build_relationship_vector_index: false`, `mix` and
  `hybrid` lose high-level keyword retrieval and `local` its relationship
  section.
- **Incremental indexing approximations** (§5). Delta runs append
  communities instead of re-clustering the whole graph, and stored
  communities and reports keep an entity's old id after it took its current
  name's; only a full rebuild (`indexing.reset: true`) refreshes both.
  Descriptions, names, ids, types, ranks and claims follow the current
  corpus. With the DynamoDB registry one document's chunk ids must fit one
  400 KB item (§5, *Document size limit*).
- **Concurrent runs need the registry.** With it, a second run on one index
  namespace stops with `NamespaceLeaseError` (§5, *Concurrent runs*).
  Without it nothing locks: do not run two ingestions at once against the
  same index suffix (a full reindex's alias swap and old-index cleanup can
  delete the other run's index). The direct `IncrementalIndexer` API takes no
  lease. See [Operator Runbook: Concurrent runs](./operations.md#concurrent-runs).
- **Quarantine does not isolate the culprit** (§5). Every document of a delta
  whose indexing keeps dying is quarantined with the one that kills it.
- **Markdown and HTML** need the optional `unstructured` extra (Python 3.11+),
  which the container image leaves out by default.
- **Changing the embedding model** (or its dimension) requires a reindex
  (`indexing.reset: true`); an incremental run refuses to write into
  indices of another model (see [Workflows](#workflows)).

---

*See also: [README.md](../README.md) for the overview and quickstart, and
[docs/design.md](./design.md) for architecture and internals.*
