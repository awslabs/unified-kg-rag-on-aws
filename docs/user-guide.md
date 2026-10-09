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

Everything below is grounded in the actual CLI flags and config keys in the
codebase. The five console entry points (defined as `pyproject` scripts) are:

| Script | Module | Purpose |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | Build / update the knowledge graph |
| `run-rag` | `application.cli.run_rag_chain` | Query the graph |
| `run-eval` | `application.cli.run_evaluation` | Evaluate retrieval + generation |
| `run-visualization` | `application.cli.run_visualization` | Render an exported graph (no ingestion) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | Generate domain-adapted prompts |

---

## 1. Prerequisites & Installation

### Runtime

- **Python 3.10 – 3.12**
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager; `pip` works too)

### AWS services

| Service | Required? | Used for |
|---|---|---|
| **Amazon Bedrock** | Yes | All LLM calls (chunking, extraction, gleaning, community reports, answer generation), embeddings, and reranking. Enable model access for the model IDs you configure. |
| **Amazon Neptune** | Yes | The knowledge graph (entities, relationships, communities) and multi-hop traversal at query time. |
| **Amazon OpenSearch** | Yes | Vector + BM25 lexical indices (text units, entities, community reports, relationships, claims). |
| **Amazon S3** | Yes | Pipeline cache sync; optional embedding-cache persistence; document storage. |
| **Amazon DynamoDB** | Only for incremental indexing | Document-status registry that diffs the corpus by content hash. |

The framework connects to services that **already exist** — it never creates
them. You have two ways to get there:

- **Bring your own services.** If Neptune, OpenSearch, S3 (and optionally
  DynamoDB) are already running, just record their endpoints in `config.yaml`
  (§2.1 below) and skip ahead. Make sure Bedrock model access is enabled for the
  model IDs you configure.
- **Provision everything with the bundled CDK app.** The repo ships an optional,
  Well-Architected AWS CDK app in [`iac/`](../iac/README.md) that stands up the
  whole stack in one command — networking (VPC + endpoints), the Neptune cluster,
  the OpenSearch domain, the DynamoDB doc-status table, the S3 cache bucket, an
  ECS Fargate data plane, a Step Functions ingestion pipeline, and CloudWatch
  observability, plus an optional region-pinned Bedrock Guardrail:

  ```bash
  cd iac
  python -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt

  cdk synth              # preview — no AWS changes, no cost
  cdk bootstrap          # once per account/region
  cdk deploy --all       # creates billable resources (Neptune + OpenSearch are hourly)
  ```

  When the deploy finishes, copy the Neptune / OpenSearch / S3 endpoints from the
  CloudFormation outputs into your `config.yaml` (the endpoints are bare
  hostnames). Neptune and OpenSearch are VPC-only, so run the CLIs inside the
  VPC, for example as a task of the deployed task definition; the stack's own
  task already receives the endpoints as env vars. `cdk destroy --all` tears the
  `dev` profile back down. See [`iac/README.md`](../iac/README.md) for every
  stack, all `-c key=value` knobs (VPC reuse, instance sizing, CMK, deletion
  protection, cdk-nag), and the production-hardening checklist.

### Install

```bash
git clone <repository-url>
cd unified-kg-rag-on-aws

# uv (recommended)
uv sync

# or pip
pip install -e .
```

Optional extra: parsing **Markdown (.md)** and **HTML (.html)** requires the
`unstructured` package. Without it, only `.pdf`, `.txt`, `.csv`, `.json` are
parsed (the parser raises a clear error naming the missing package for
`.md`/`.html`). On Python 3.11 or 3.12, install the patched parser with
`uv sync --extra unstructured` or `pip install -e '.[unstructured]'`.
The extra requires `unstructured>=0.24.0`, which fixes URL-partitioning SSRF
and no longer depends on NLTK. On Python 3.10 the extra does not install a parser;
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
   OPENSEARCH_USERNAME=your_opensearch_username
   OPENSEARCH_PASSWORD=your_opensearch_password
   ```

   The `.env` file is loaded automatically by the CLIs (`run-ingestion`,
   `run-rag`). When `use_iam: true`, no `.env` is needed.

### Local stores (development)

For development you can replace Neptune and OpenSearch with local containers.
Models still come from Bedrock, so you need AWS credentials with Bedrock
access. S3 is needed only for `--s3-sync`, DynamoDB only for incremental
indexing.

```bash
docker compose -f docker/compose.local.yaml up -d --wait
uv run run-ingestion --config-path docker/config.local.yaml --source-directory ./docs-in
uv run run-rag --config-path docker/config.local.yaml --query "..."
docker compose -f docker/compose.local.yaml down -v
```

[`docker/compose.local.yaml`](../docker/compose.local.yaml) runs a TinkerPop
Gremlin Server (in-memory TinkerGraph) and a single-node OpenSearch 2.13 with
the security plugin disabled and the `analysis-nori` plugin installed (the
Korean mappings use `nori`; Amazon OpenSearch Service ships it built in).
[`docker/config.local.yaml`](../docker/config.local.yaml) points the framework
at them with `aws.neptune.use_ssl: false` and `use_iam: false` (plain `ws://`,
no SigV4) and `aws.opensearch.allow_anonymous: true` with `use_ssl: false`
(plain `http://`, no auth).

To check the stores without Bedrock, run the smoke test, which uses a hashing
embedding provider: `LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py`.

Known differences from Neptune: TinkerGraph keeps the graph in memory only,
and a list property written when a vertex is created keeps duplicate values
(Neptune stores them once). Nothing in the compose file is hardened; it binds
to `127.0.0.1` only.

---

## 2. Configuration

Create your config from the template and point every CLI at it with
`--config-path config.yaml`:

```bash
cp config-template.yaml config.yaml
```

**`config-template.yaml` is the reference for every option.** It lists each key
with its default and a comment on what it does. This section covers how
configuration is loaded, the knobs you will most often tune, and the
environment variables that override the file. For anything not listed here,
read the template.

### How configuration is loaded

Values are resolved in three layers, each overriding the one before:

1. **Built-in defaults**: the Pydantic models in
   `unified_kg_rag/domain/models/config.py`. Without `--config-path`, a CLI runs
   on these defaults plus environment overrides.
2. **Your YAML file**: every key you set replaces its default; every key you
   omit keeps it. A `config.yaml` can therefore hold only the keys you change.
   A dict-valued key (for example `logging.library_levels` or
   `graph.visualization.interactive`) replaces the whole default dict rather
   than merging into it.
3. **Environment variables** (§2.10): applied last, on top of both.

YAML values are validated when the file loads: a wrong type or an unsupported
value stops the CLI with `Configuration validation error: ...`. An unknown key
(a typo, or a key removed in a newer release) does not stop the run. It is
logged at WARNING as `Unknown config key '<path>' is ignored` and dropped, so
check the log after editing the file or upgrading.

Renamed keys are still accepted: each is applied to its replacement with a
WARNING `Config key '<old>' is deprecated; applied as '<new>: <value>'`, and is
ignored if the replacement is also set.

| Former key | Replacement |
|---|---|
| `search.llm_retry` | `aws.bedrock.transient_retry` |
| `aws.bedrock.effort` | `aws.bedrock.default_effort` (same value) |
| `processing.max_retries` | `processing.max_attempts` (same value) |
| `indexing.neptune.max_retries` | `indexing.neptune.max_attempts`, plus one: the former key counted retries after the first try |
| `evaluation.ragas_max_retries` | `evaluation.ragas_max_attempts` (same value) |

Every `max_attempts` key counts total attempts, including the first; `1`
disables the retry.

The tables below give each key's built-in default. `config-template.yaml` uses
the same values.

### 2.1 `aws` — service endpoints & credentials

| Key | Default | What it does / when to change |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | Region of Neptune, OpenSearch, S3, and DynamoDB, and of Bedrock unless `aws.bedrock.region_name` is set. `AWS_REGION` overrides it (see §2.10). The default offers every default model, including both rerank models, which some regions (e.g. `ap-northeast-2`) do not. |
| `aws.profile_name` | `null` | Named AWS profile; `null` uses the default credential chain. |
| `aws.bedrock.region_name` | `null` | Region for Bedrock model, embedding, and rerank calls, and where the guardrail must exist. `null` uses `aws.region_name`. Set it only when your models are enabled in another region; in a private VPC whose only Bedrock route is a VPC endpoint, leave it `null` or set the VPC's region. |
| `aws.bedrock.enable_global_profile` | `true` | Resolve cross-region (global) inference profiles. Keep it on: Claude 4.7+ and GPT models are invocable only through a profile. |
| `aws.bedrock.default_model_id` | `"anthropic.claude-sonnet-5-5"` | Model for every default-tier role (see Model selection notes). |
| `aws.bedrock.fast_model_id` | `"anthropic.claude-haiku-5-5"` | Model for every fast-tier role. |
| `aws.bedrock.default_max_output_tokens` | `16384` | `max_tokens` per request, clamped to the model maximum. Raise it if answers are cut off (`stopReason: max_tokens`); `null` sends the model maximum. |
| `aws.bedrock.default_effort` | `"high"` | Reasoning depth for calls on `default_model_id` (adaptive-thinking Claude and GPT models): `low`, `medium`, `high`, `xhigh`, `max`. Lower it to cut cost and latency. |
| `aws.bedrock.fast_effort` | `"low"` | Reasoning depth for calls on `fast_model_id` when it differs from `default_model_id`. The shipped Claude Haiku 5.5 thinks adaptively, so this sets its reasoning depth; no effect on a fast model that does not reason. |
| `aws.bedrock.enable_1m_context` | `false` | Opt into the 1M window on models where it is a beta (premium billing). Claude 5 has a native 1M window. |
| `aws.bedrock.model_overrides` | `{}` | Capability records for a language model the package does not know (see Model selection notes). Embedding and rerank models are a closed list. |
| `aws.bedrock.guardrail.identifier` | `null` | Bedrock guardrail ID or ARN; setting it enables the guardrail. |
| `aws.bedrock.guardrail.apply_to` | `"query"` | `query` guards only the user-facing query path; `all` guards every call (see the guardrail note below). |
| `aws.bedrock.guardrail.trace` | `false` | Emit the guardrail trace. Needed to detect interventions on the InvokeModel path. |
| `aws.bedrock.transient_retry.max_attempts` | `5` | Attempts per call for transient Bedrock errors that botocore does not retry (such as HTTP 424), on embeddings and query-time calls. `1` disables the retry. |
| `aws.neptune.endpoint` | `null` | **Required.** Neptune cluster endpoint. |
| `aws.neptune.use_iam` | `true` | SigV4-sign Neptune requests. |
| `aws.neptune.use_ssl` | `true` | Connect over `wss://`. Neptune requires it; set `false` (with `use_iam: false`) only for a local Gremlin Server (§1 Local stores). |
| `aws.neptune.pool_size` | `4` | Gremlin connection pool size. The client raises it to `indexing.neptune.index_concurrency` when that is larger. |
| `aws.opensearch.endpoint` | `null` | **Required.** OpenSearch domain endpoint. |
| `aws.opensearch.use_iam` | `false` | `false` reads `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` from the environment (§1 Authentication). |
| `aws.opensearch.allow_anonymous` | `false` | Connect with no auth, for a local OpenSearch with the security plugin disabled (§1 Local stores). Cannot be combined with `use_iam` or username/password. |
| `aws.opensearch.sigv4_service_name` | `"es"` | `es` for a managed domain, `aoss` for OpenSearch Serverless. A wrong value often shows up as zero search hits. |
| `aws.s3.bucket_name` | `null` | Bucket for cache sync and embedding-cache persistence. |
| `aws.s3.encryption.encryption_type` | `"BUCKET_DEFAULT"` | `BUCKET_DEFAULT` lets the bucket's default encryption apply; `AES256` or `aws:kms` (with `kms_key_id`) force a per-object header. |
| `aws.dynamodb.enabled` | `false` | Turn on the doc-status registry for incremental indexing (§5). |
| `aws.dynamodb.table_name` | `"unified-kg-rag-on-aws-doc-status"` | Doc-status table name. |
| `aws.dynamodb.create_table_if_missing` | `true` | Create the table on first use. Set `false` when the table is managed by IaC. |

> **Guardrail scope and placement.** The guardrail must exist in
> `aws.bedrock.region_name`, the region LLM calls go to. With the default
> `apply_to: "query"` it is attached to answer generation, query refinement,
> query-time entity/keyword extraction, and global/DRIFT map-reduce. Ingestion
> models, the prompt tuner, and evaluation judges run unguarded, because a PII
> guardrail that anonymizes `NAME` rewrites extracted entity names to a
> placeholder such as `{NAME}` (distinct people merge into one node), and a
> `PROMPT_ATTACK` filter can block instruction-like corpus text. Use
> `apply_to: "all"` only with a policy that is safe for extraction. On the
> query path `NAME` anonymization also hurts: answers show `{NAME}` instead of
> the people they are about, and entity-seeded search loses its seeds. The
> baseline guardrail that `iac/` creates therefore anonymizes email, phone and
> card numbers but not `NAME`. Each
> intervention is logged at WARNING (`Bedrock guardrail '<id>' intervened on a
> <purpose> model call ...`) with a running count. On the InvokeModel path
> (`ChatBedrock`, used for non-cross-region model ids) an intervention is
> detected only with `trace: true`; the guardrail is enforced either way.
>
> Upgrading: earlier releases guarded every call; set `apply_to: "all"` to keep
> that. Custom code that builds chains with `setup_chain` or calls `get_model`
> for non-query work should pass `model_purpose=ModelPurpose.INGESTION` (or
> `EVALUATION`). Unmarked calls default to `QUERY`: they stay guarded and get
> the query-time transient-error retry.

> **S3 cache encryption.** With the CDK stack's `use_cmk=true`, the bucket
> default is the customer-managed KMS key, so `BUCKET_DEFAULT` uses it. Releases
> before this default sent `AES256`, which silently bypassed a bucket's CMK. When
> you reuse a bucket whose default is SSE-KMS, the writer needs
> `kms:GenerateDataKey` and `kms:Decrypt` on that key.

#### Model selection notes

Every LLM role belongs to one of two tiers. `aws.bedrock.default_model_id`
serves the reasoning-heavy roles and `aws.bedrock.fast_model_id` the light
ones, so switching the whole pipeline to another model is one line. A role's
own `*_model_id` key still wins over its tier:

| Tier | Roles (`*_model_id` keys) |
| --- | --- |
| `default` | `fixing.fixing_model_id`, `processing.graph_extraction.extraction_model_id`, `processing.gleaning.graph_refinement_model_id`, `processing.claim_extraction.extraction_model_id`, `graph.community_detection.report_generation.report_generation_model_id`, `search.{entity_extraction,context_building,answer_generation}_model_id`, `evaluation.evaluation_model_id` |
| `fast` | `processing.chunking.chunking_model_id`, `processing.translation.translation_model_id`, `processing.graph_extraction.description_summarization.summary_model_id`, `search.{translation,strategy_selection}_model_id`, `search.global_search.{community_relevance,map_reduce,map}_model_id`, `search.drift_search.{query_refinement,keyword_expansion,convergence_assessment,primer}_model_id` |

```yaml
aws:
  bedrock:
    default_model_id: "openai.gpt-6-sol"    # every default-tier role
search:
  answer_generation_model_id: "anthropic.claude-opus-5-5"  # one role pinned
```

Model-id keys accept any Bedrock model id, or an inference-profile id such as
`us.anthropic.claude-sonnet-5-5` (used as-is). The models below have a curated
capability record (as do the older Claude 3.x/4.x ids). Any other id still
works: `anthropic.claude-*` ids get the request shape of their generation,
`openai.gpt-*` ids the GPT shape, and other providers a conservative Converse
request (no reasoning or sampling parameters, 32K window, 4K output), each with
one WARNING. Describe or correct a model with `aws.bedrock.model_overrides`
(keys are capability-record fields such as `context_window_size` and
`max_output_tokens`; an unknown key fails fast):

```yaml
aws:
  bedrock:
    model_overrides:
      "amazon.nova-pro-v1:0":
        context_window_size: 300000
        max_output_tokens: 10000
```

Embedding and rerank model ids work differently: they are a closed list, and
`model_overrides` does not apply to them. `embedding_model_id` accepts
`amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1`,
`cohere.embed-v4:0`, `cohere.embed-english-v3` or
`cohere.embed-multilingual-v3`; `rerank_model_id` accepts
`cohere.rerank-v3-5:0` or `amazon.rerank-v1:0`. Any other id fails config
validation. The embedding dimension is written into the OpenSearch vector
mappings and checked against the model's record, so an unknown model has no
safe default; `indexing.opensearch.embedding_dimension` picks among the
dimensions a listed model supports (Titan Embed V2: 256, 512 or 1024; unset
uses the largest). Adding a model is a code change: a member in
`EmbeddingModelId`/`RerankModelId` (`domain/models/config.py`) and a record in
`adapters/aws/bedrock_models.py`.

**Output cap.** Bedrock reserves input + `max_tokens` against the
tokens-per-minute quota when a request starts, so asking for the model maximum
(128K on Claude 5.x) throttles concurrent ingestion long before real usage
does. That is why `default_max_output_tokens` is 16384. Thinking tokens count
toward it, and prompts with long outputs declare a higher floor that wins
(graph and claim extraction, gleaning, community reports and their output
fixer 32768; document translation 65536).

**Prompt caching.** On Claude models that support explicit prompt caching, the
end of each system prompt is marked as a cache checkpoint: a `cachePoint` block
on the Converse API (every inference profile) and `cache_control` on
InvokeModel. A system prompt shorter than the model's minimum checkpoint size
(512 tokens on Claude Sonnet/Opus 5.5 and Opus 5, 1024 on most others, 4096 on
Claude Haiku 4.5 and Opus 4.5-4.7) gets no marker, because Bedrock would accept
it but cache nothing. Cache reads show up as `cache_read` in the response's
`usage_metadata.input_token_details`, and cached input tokens do not count
against the tokens-per-minute quota.

| Model id | Provider | Context / max output | Reasoning control |
| --- | --- | --- | --- |
| `anthropic.claude-sonnet-5-5` (default) | Anthropic | 1M / 128K | adaptive, always on; `effort` low–max |
| `anthropic.claude-opus-5-5` | Anthropic | 1M / 128K | adaptive, always on; `effort` low–max |
| `anthropic.claude-haiku-5-5` | Anthropic | 1M / 128K | adaptive, on by default; `effort` low–max |
| `anthropic.claude-sonnet-5`, `anthropic.claude-opus-5` | Anthropic | 1M / 128K | adaptive, always on; `effort` |
| `anthropic.claude-opus-4-8`, `anthropic.claude-opus-4-7` | Anthropic | 1M / 128K | adaptive, always on; `effort` low–max |
| `anthropic.claude-opus-4-6-v1` | Anthropic | 1M / 128K | opt-in (`--enable-thinking`), adaptive; `effort` low/medium/high/max |
| `anthropic.claude-sonnet-4-6` | Anthropic | 1M / 64K | opt-in, adaptive; `effort` low/medium/high/max |
| `openai.gpt-6.1-sol` | OpenAI | 1M / 131K | `reasoning.effort` low–max, always on |
| `openai.gpt-6-astra`, `openai.gpt-6-sol`, `openai.gpt-6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, always on |
| `openai.gpt-5.6-sol`, `openai.gpt-5.6-terra`, `openai.gpt-5.6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, always on |
| `openai.gpt-5.5`, `openai.gpt-5.4` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, always on |

All of these are inference-profile-only. Only OpenAI's proprietary GPT models
are offered; the open-weight `gpt-oss` models are not.

Three things differ for Claude 4.7-and-later models:

- **Inference profiles are mandatory.** They ship without `ON_DEMAND`
  throughput, so the bare model id is not invocable — a cross-region profile
  must resolve. Keep `enable_global_profile: true` and grant
  `bedrock:ListInferenceProfiles`; the adapter fails fast with the remedy if no
  profile resolves. Note that in `ap-northeast-2` only `global.` profiles exist
  for Claude 5 (no `apac.`), so disabling the global profile leaves no path.
- **`effort` replaces the thinking token budget.** `thinking_budget_tokens` is
  ignored for these models (the old `budget_tokens` request shape is rejected
  with a 400); set `bedrock.default_effort` / `bedrock.fast_effort` instead.
  A call uses `fast_effort` when its model is `fast_model_id` (and that differs
  from `default_model_id`), otherwise `default_effort`; the shipped fast model,
  Claude Haiku 5.5, thinks adaptively, so `fast_effort` (default `low`) sets
  how much it reasons on fast-tier calls. Claude Sonnet 5.5 always thinks, so `--enable-thinking` is a
  no-op for it — depth is `effort` only. A level the model does not accept
  (e.g. `xhigh` on Opus or Sonnet 4.6) fails fast.
- **Sampling parameters are dropped.** `temperature`/`top_k` are not accepted
  and are omitted from requests automatically; steer behaviour by prompting.

OpenAI GPT models differ from Claude in these ways:

- They always go through the Converse API on a `us.`/`global.` inference
  profile (no `apac.`/`eu.` geo profiles; keep `enable_global_profile: true`
  outside the US). The tier's effort (`bedrock.default_effort` /
  `fast_effort`) is sent as
  `reasoning: {effort: ...}` (the flat `reasoning_effort` field is rejected).
  GPT-5.6 and GPT-6.x answered a trivial prompt in roughly 10-25 s even at
  `effort: low`, so size timeouts and concurrency accordingly.
- No Anthropic-only fields are sent (`thinking`, `output_config`,
  `anthropic_beta`, the `\n\nHuman:` stop sequence), and sampling parameters
  are omitted.
- Explicit prompt-cache markers are not sent: Converse supports only implicit
  caching for these models. Bedrock CountTokens does not support them, so the
  retrieval context budget uses the local token estimate.

Claude Fable 5 / 5.1 are not offered: they need a non-default account
data-retention mode (Data Retention API only), and accounts on the default mode
get `data retention mode 'default' is not available for this model` on every
call.

### 2.2 `fixing` — auto-repair malformed model output

| Key | Default | What it does / when to change |
|---|---|---|
| `fixing.enabled` | `true` | When a structured stage gets malformed JSON from the model, ask a model to repair it instead of failing the run. Leave it on. |

### 2.3 `processing` — concurrency, chunking, translation, extraction

LLM stages are Bedrock-I/O-bound, so concurrency can far exceed the CPU count.

| Key | Default | What it does / when to change |
|---|---|---|
| `processing.max_concurrency` | `20` | Concurrent LLM calls within a batch. Lower it if Bedrock throttles; raise it if quota allows. |
| `processing.chunk_concurrency` | `4` | Mini-batch chunks run at once. The Bedrock connection pool is sized to `max_concurrency` × `chunk_concurrency`. |
| `processing.max_attempts` | `5` | Attempts for an ingestion LLM item called on its own after its batch call fails. Transient Bedrock errors, call timeouts and unparseable output are retried; other errors fail fast. `1` disables the retry. |
| `processing.io_workers` | `64` | Threads for blocking query-path I/O (Bedrock calls, Neptune traversals, reranking) in the CLIs and the chain's sync methods. Python's default caps it at `min(32, CPUs + 4)`, six on a 2-vCPU task. An async host calls `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)` (from `unified_kg_rag.shared.utils`) at startup. |
| `processing.ignore_errors` | `false` | Skip items whose LLM step fails instead of failing the run. |
| `processing.deduplicate` | `false` | Drop duplicate documents before extraction. |
| `processing.resolution_method` | `"minhash"` | Entity resolution: `minhash` or `sequence_matcher`. |
| `processing.similarity_threshold` | `0.6` | Fuzzy-match threshold for entity resolution. Raise it if distinct entities merge. |
| `processing.document_parsing.source_directory` | `"source"` | Fallback for library callers. `run-ingestion` requires `--source-directory` (or `GRAPHRAG_SOURCE_DIRECTORY`). |
| `processing.document_parsing.target_directory` | `null` | Export each parsed document as `<stem>.json` for inspection (same as `--target-directory`). Must not be the source directory. |
| `processing.document_parsing.source_scope` | `null` | Corpus identity for incremental deletion: a run only deletes registry documents of its own index suffix and source scope. `null` = the resolved source directory; the container entrypoint sets it to the S3 URI (`GRAPHRAG_SOURCE_SCOPE`). See §5. |
| `processing.chunking.chunker_type` | `"intelligent"` | `intelligent` lets an LLM pick semantic boundaries; `simple` splits by size. |
| `processing.chunking.min_chunk_size` | `1000` | Minimum chunk size in characters; shorter pieces merge into a neighbour. |
| `processing.chunking.max_chunk_size` | `8000` | Maximum chunk size in characters. Must fit the embedding and rerank input limits. |
| `processing.chunking.chunk_overlap` | `500` | Overlap between chunks in characters. |
| `processing.chunking.fallback_chunk_size` | `4800` | Target size for the size-based splitter (`simple`, or when intelligent chunking fails). |
| `processing.translation.enabled` | `true` | Run the translation stage. It is a no-op (zero LLM cost) when the source and target languages match and no additional target is set. |
| `processing.translation.source_language` | `"en"` | Predominant corpus language; used only for the no-op check. |
| `processing.translation.target_language` | `"en"` | Language the corpus is translated into (see §3 Multilingual ingestion). |
| `processing.translation.additional_target_languages` | `null` | Extra target languages to translate into. |
| `processing.graph_extraction.entity_types` | 7 generic types | `"LABEL: description"` items injected into the extraction prompt. The most effective domain-adaptation knob (§9). An empty list lets the model choose. |
| `processing.graph_extraction.max_entities_per_chunk` | `50` | Cap on entities per chunk (relationships: `max_relationships_per_chunk`, also `50`). |
| `processing.graph_extraction.entity_confidence_threshold` | `0.0` | Drop entities below this confidence; `0.0` keeps all. |
| `processing.graph_extraction.description_summarization.enabled` | `true` | Re-summarize merged descriptions longer than `force_summary_threshold_tokens` (`600`) with an LLM. |
| `processing.graph_extraction.entity_grounding.enabled` | `false` | Hallucination guard: drop (or, with `action: "penalize"`, down-weight) entities and relationships whose verbatim `source_text` span is absent from the chunk. Also gates gleaning additions. |
| `processing.gleaning.enabled` | `true` | Extra extraction passes that catch missed entities and relationships. |
| `processing.gleaning.max_rounds` | `3` | Gleaning rounds; later rounds re-glean only units that gained items. `1` matches MS GraphRAG's default. |
| `processing.claim_extraction.enabled` | `false` | Extract claims (one extra LLM call per text unit). When on, `local` search injects matching claims and `simple` search sweeps the claims index. |

### 2.4 `graph` — analysis, community detection, visualization

| Key | Default | What it does / when to change |
|---|---|---|
| `graph.community_detection.enabled` | `true` | Leiden clustering plus community-report generation. Required by GraphRAG `global`/`drift`. Set `false` for a lighter LightRAG-only ingestion. |
| `graph.community_detection.auto_resolution` | `false` | `true` sweeps `auto_resolution_candidates` at every level and keeps the most modular resolution; otherwise `resolution` (`1.0`) is used. |
| `graph.community_detection.auto_resolution_max_nodes` | `10000` | Above this node count the sweep is skipped and `resolution` is used. |
| `graph.community_detection.max_levels` | `5` | Maximum community hierarchy depth. |
| `graph.community_detection.min_community_size` | `3` | Smaller communities merge into neighbours. |
| `graph.community_detection.report_generation.max_report_context_tokens` | `4000` | Token budget for the entity/relationship context in one report prompt. |
| `graph.community_detection.report_generation.content_length` | `"medium"` | Report length: `short`, `medium`, `long`. |
| `graph.analysis.centrality.calculate_betweenness` | `true` | Betweenness centrality. On graphs larger than `betweenness_auto_sample_threshold` (`2000`) nodes it is sampled instead of computed exactly. |
| `graph.visualization.enabled` | `true` | Export visualization data during ingestion. |
| `graph.visualization.outputs_directory` | `null` | Unset writes to `<cache dir>/<pipeline_id>/visualization`, so the data syncs to S3 with the cache. |
| `graph.visualization.layout_method` | `"umap"` | `umap`, `tsne`, or `pca`. |
| `graph.visualization.interactive.max_nodes` | `2000` | Top-N nodes by degree kept in `interactive_graph.html`; `0` or `null` disables the cap. Setting `interactive` replaces its default dict, so also set `physics_enabled: false` to keep physics off. |

### 2.5 `indexing` — OpenSearch & Neptune write side

| Key | Default | What it does / when to change |
|---|---|---|
| `indexing.reset` | `false` | Clear existing indexed data before indexing. |
| `indexing.additional_suffix` | `null` | Appended after the run suffix in every OpenSearch index name and Neptune label (`<prefix>-<suffix>-<additional_suffix>`, where `<suffix>` is `--suffix` or `default`). Use it for versioned or multi-tenant separation. |
| `indexing.cross_run_merge` | `true` | On delta runs, union the delta with existing graph state instead of overwriting, so entities shared with unchanged documents keep their lineage (§5). `false` overwrites. |
| `indexing.cross_run_fuzzy_merge` | `false` | Extend `cross_run_merge` with fuzzy entity-name matching. |
| `indexing.max_failure_rate` | `0.2` | Per-index-type write failure rate above which the indexing stage fails. `1.0` disables the partial-failure gate. |
| `indexing.opensearch.embedding_model_id` | `"amazon.titan-embed-text-v2:0"` | Embedding model, one of a closed list (see Model selection notes). Changing it requires a reindex. |
| `indexing.opensearch.build_relationship_vector_index` | `true` | Relationship vector index for LightRAG `mix`/`hybrid`. Set `false` for a GraphRAG-only deployment. |
| `indexing.opensearch.persist_embedding_cache` | `false` | Persist the embedding cache to S3 so unchanged text is not re-embedded across runs. Requires `aws.s3.bucket_name`. |
| `indexing.opensearch.language_analyzers` | `{en: english, ko: nori}` | Text analyzer per language code; unlisted languages use `default_analyzer` (`standard`). |
| `indexing.opensearch.vector_search.engine` | `"lucene"` | HNSW engine. `lucene` supports `cosinesimil` up to 1024 dimensions; see the template for `faiss`. |
| `indexing.opensearch.index_settings.refresh_interval` | `"1s"` | Raise it (or `"-1"`) for faster bulk loads, then reset for live querying. |
| `indexing.neptune.batch_size` | `100` | Items per Neptune write batch. |
| `indexing.neptune.index_concurrency` | `1` | Concurrent write batches. The Gremlin connection pool grows to match if `aws.neptune.pool_size` is smaller. |
| `indexing.neptune.max_attempts` | `4` | Attempts per Neptune write, including the first. Failures are retried with jittered exponential backoff from `retry_delay_seconds` (`2`), except errors a retry cannot fix (malformed query, access denied, bad parameter), which fail on the first attempt. `1` disables the retry. |
| `indexing.neptune.max_hops` | `3` | Neighbour-expansion depth at retrieval time. |
| `indexing.neptune.property_max_length` | `4000` | Character cap per Neptune property value. Keep it above the longest description that is not re-summarized (summarization triggers above 600 tokens, ~2,400 characters). Takes effect on re-ingestion. |
| `indexing.neptune.entity_importance_source` | `"rank"` | Entity importance in graph-expansion relevance: `rank` (indexed entity rank), `degree` (edge count at query time) or `none` (neutral 0.5, the old behaviour). |
| `indexing.neptune.traversal_fetch_multiplier` | `3` | Graph expansion fetches this many times the result width, ranks, then cuts. `1` = cut in traversal order (old behaviour). |

### 2.6 `search` — retrieval, fusion, reranking, per-strategy knobs

| Key | Default | What it does / when to change |
|---|---|---|
| `search.auto_routable_strategies` | `["local", "mix", "global", "drift"]` | Strategies the `auto` router may pick. Any strategy can still be chosen explicitly. |
| `search.hybrid.lexical_weight` | `0.5` | Lexical weight in the OpenSearch hybrid pipeline (vector: `vector_weight`, also `0.5`). |
| `search.fusion.method` | `"rrf"` | `rrf` (reciprocal rank fusion) or `weighted`. |
| `search.fusion.rrf_k` | `60` | RRF constant `k`. |
| `search.fusion.fusion_weights` | `1.0` per bucket | Per-source-bucket weights; they scale each bucket's contribution under both `rrf` and `weighted`. |
| `search.fusion.diversity_lambda` | `0.5` | MMR trade-off: `1.0` = pure relevance, `0.0` = maximum diversity. |
| `search.reranking.enabled` | `true` | Rerank fused results with `rerank_model_id` (`cohere.rerank-v3-5:0`). |
| `search.reranking.top_k` | `100` | Candidates sent to the reranker. |
| `search.lightrag_search.kg_stream_top_k` | `40` | Width of the LightRAG entity/relationship vector queries (a floor against the request's `top_k`). |
| `search.lightrag_search.chunk_stream_top_k` | `20` | Width of the LightRAG chunk stream. |
| `search.lightrag_search.enable_graph_expansion` | `false` | `mix`/`hybrid`: also expand matches through Neptune. |
| `search.global_search.max_communities` | `10` | Community reports considered by `global` search. |
| `search.global_search.map_batch_size` | `5` | Reports per map-step LLM call; lower it for long reports. |
| `search.global_search.max_map_reduce_tokens` | `8000` | Token budget for ranked key points fed to the reduce step. |
| `search.global_search.reduce_with_llm` | `false` | `true` = a reduce LLM summarizes the packed key points before the answer model rewrites them (one extra LLM call). `false` passes the points straight to the answer model. |
| `search.global_search.reserve_report_slots` | `true` | Reserve `max_communities` fusion slots for community reports and cap their text units at `text_unit_slots`. `false` = one flat `top_k` cut over reports and chunks (old behaviour). |
| `search.global_search.text_unit_slots` | `null` | Text-unit slots next to the reserved report slots; `null` = the query's `top_k`. |
| `search.local_search.entity_frequency_threshold` | `20` | Drop graph-expanded entities that appear in more text units than this (too generic). |
| `search.local_search.include_bridge_relationships` | `true` | Also fetch relationships incident to the expanded entities, edges between two retrieved entities first (multi-hop bridges). Needs the relationship index. `false` = relationship vector query only. |
| `search.drift_search.max_iterations` | `3` | DRIFT iteration budget. |
| `search.drift_search.enable_primer` | `false` | MS GraphRAG primer → follow-up flow (one extra LLM call up front). |
| `search.drift_search.enable_llm_convergence` | `false` | LLM convergence check after each iteration (one extra call per iteration). |
| `search.token_manager.max_context_tokens` | `30000` | Retrieval context budget for the answer prompt (see the note below). |
| `search.token_manager.context_window_headroom_ratio` | `0.1` | Share of the window held back when the budget is derived (`max_context_tokens: null`). |

Per-section-type shares (`search.token_manager.type_budgets`) and `local` slot
quotas (`search.local_search.type_quota`) are in the template.

> **Context budget.** `30000` matches upstream LightRAG's total context budget
> (MS GraphRAG uses 12000). A budget derived from a 1M-token window (~785K)
> never binds, so per-type budgets would never trim anything. The value is
> always clamped to what `search.answer_generation_model_id` accepts alongside
> its output reservation (`aws.bedrock.default_max_output_tokens`), with a
> warning when that happens. With `null`, the budget is derived from that
> model's window minus the output reservation and the headroom ratio;
> `aws.bedrock.enable_1m_context` widens it on models where 1M is a beta.

### 2.7 `memory`, `cache`, `logging`

| Key | Default | What it does / when to change |
|---|---|---|
| `memory.max_conversations` | `100` | Conversations held in conversation memory (§4 Interactive mode). |
| `memory.max_messages_per_conversation` | `20` | Messages kept per conversation. |
| `memory.max_conversation_age_hours` | `168` | Age after which a conversation can be cleaned up. |
| `cache.ttl_seconds` | `86400` | Cache entry TTL; `null` = never expire. |
| `logging.level` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `logging.log_format` | `"structured"` | `structured` or `plain`. |
| `logging.log_to_file` | `true` | The CLIs also write logs to `log_file_path` (`logs/log.txt`, dated as `log_YYYYMMDD.txt`; a relative path is resolved against the working directory). Importing the package as a library configures no handler or file: the host application's logging setup applies. |
| `logging.library_levels` | `{langchain_aws: WARNING, botocore: WARNING, urllib3: WARNING}` | Per-logger levels for chatty libraries. Setting the key replaces the whole map. |

### 2.8 `evaluation`

| Key | Default | What it does / when to change |
|---|---|---|
| `evaluation.enabled_evaluators` | `[langchain, ragas, answer_match, retrieval, graph_aware]` | The deterministic evaluators (`answer_match`, `retrieval`, `graph_aware`) skip queries that lack their ground-truth fields (§6). Remove the LLM judges to evaluate without judge cost. |
| `evaluation.judge_effort` | `"low"` | Reasoning effort for LLM judges; `null` inherits the judge model's tier effort (`aws.bedrock.default_effort` by default). |
| `evaluation.ragas_timeout` | `300` | Seconds to score one metric on one sample; a timeout yields NaN. |
| `evaluation.ragas_max_contexts` | `20` | Top-ranked contexts per sample scored by RAGAS `context_precision` (so it is "@20"); faithfulness and context_recall see the full token-budgeted context. `null` = no cap. |
| `evaluation.ragas_max_workers` | `8` | Concurrent RAGAS jobs. Lower it if Bedrock throttles the judge. |
| `evaluation.ragas_max_attempts` | `3` | Total attempts per judge call. |
| `evaluation.max_context_tokens` | `8192` | Token cap on the context passed to judges. |
| `evaluation.retrieval_k` | `5` | Cutoff for the `retrieval` evaluator's hit@k / recall@k. |
| `evaluation.outputs_directory` | `"outputs/evaluation"` | Where results are written. |

Metric lists (`langchain_metrics`, `ragas_metrics`) are in the template.

### 2.9 `custom_prompts`

Every prompt has a `*_system` / `*_human` override (default `null` = use the
built-in prompt in `unified_kg_rag/domain/prompts/`). See §9. Override what you
need; leave the rest `null`.

### 2.10 Environment variable overrides

These variables override the config file (and the built-in defaults) when set.
Values are converted to the field's type (`true`/`1`/`yes`/`on` count as true
for booleans) but are not re-validated.

| Variable | Overrides | Notes |
|---|---|---|
| `AWS_PROFILE` | `aws.profile_name` | |
| `AWS_REGION` | `aws.region_name` | Also moves Bedrock calls when `aws.bedrock.region_name` is `null`. `AWS_DEFAULT_REGION` is not read. |
| `BEDROCK_REGION` | `aws.bedrock.region_name` | |
| `BEDROCK_GUARDRAIL_IDENTIFIER` | `aws.bedrock.guardrail.identifier` | |
| `NEPTUNE_ENDPOINT` | `aws.neptune.endpoint` | |
| `OPENSEARCH_ENDPOINT` | `aws.opensearch.endpoint` | |
| `OPENSEARCH_USERNAME` | `aws.opensearch.username` | Basic auth when `aws.opensearch.use_iam: false`. |
| `OPENSEARCH_PASSWORD` | `aws.opensearch.password` | Kept masked in logs. |
| `S3_BUCKET_NAME` | `aws.s3.bucket_name` | |
| `GRAPHRAG_DOC_STATUS_TABLE` | `aws.dynamodb.table_name` | |
| `GRAPHRAG_DOC_STATUS_CREATE_TABLE` | `aws.dynamodb.create_table_if_missing` | |
| `LOG_LEVEL` | `logging.level` | |
| `LOG_FORMAT` | `logging.log_format` | |
| `LOG_TO_FILE` | `logging.log_to_file` | |
| `LOG_FILE_PATH` | `logging.log_file_path` | |

`run-ingestion` also reads `GRAPHRAG_SOURCE_DIRECTORY` and `GRAPHRAG_PIPELINE_ID`
as the defaults of `--source-directory` and `--pipeline-id`; an explicit flag
wins.

> **A stray `AWS_REGION` wins over your file.** SSO credential helpers, CloudShell, and
> shell profiles often export `AWS_REGION`, which silently replaces
> `aws.region_name`, and the run then looks for Neptune and OpenSearch (and,
> with `aws.bedrock.region_name` unset, Bedrock) in the wrong region. Run `env | grep -E '^(AWS_REGION|BEDROCK_REGION)='` before a run,
> and unset or correct what you find.

> **LangSmith tracing uploads content.** When `LANGSMITH_TRACING=true` (or the
> older `LANGCHAIN_TRACING_V2=true`) is set, LangChain sends every traced run to
> LangSmith, including prompts, retrieved document text and model outputs. The
> CLIs log a WARNING at startup when tracing is on but do not turn it off, since
> you may want it. Unset the variable before running on a confidential corpus.
> The warning comes from `setup_logging`, so library code that does not call it
> gets none.

The CLIs also load a `.env` file with python-dotenv. A variable already set in
the environment wins over `.env`. The file is searched for from the package's
location upward, not from the current directory, so in a source checkout put it
at the repository root.

The CDK compute stack injects `AWS_REGION`, `BEDROCK_REGION`,
`NEPTUNE_ENDPOINT`, `OPENSEARCH_ENDPOINT`, `S3_BUCKET_NAME`,
`GRAPHRAG_DOC_STATUS_TABLE`, `GRAPHRAG_DOC_STATUS_CREATE_TABLE=false` (the
table is IaC-managed and the task role cannot create tables), `LOG_FORMAT`, and,
when a guardrail is deployed, `BEDROCK_GUARDRAIL_IDENTIFIER`. Incremental
indexing still needs `aws.dynamodb.enabled: true` in the config file.

---

## 3. Ingestion (`run-ingestion`)

Ingestion turns a directory of documents into a knowledge graph indexed in
OpenSearch + Neptune.

### CLI flags (verified)

| Flag | Default | Meaning |
|---|---|---|
| `--source-directory` | `$GRAPHRAG_SOURCE_DIRECTORY` | Directory of source documents. Required to run; if the flag is omitted it falls back to the `GRAPHRAG_SOURCE_DIRECTORY` environment variable. |
| `--target-directory` | none (no export) | Export parsed documents as JSON here for inspection (must not be the source directory) |
| `--cache-directory` | `cache` | Pipeline cache + intermediate results |
| `--force-rebuild` | off | Ignore all existing cache; rebuild from scratch |
| `--s3-sync` | off | Sync cache to S3 (requires `--s3-bucket-name`) |
| `--s3-bucket-name` | — | S3 bucket for cache sync |
| `--s3-prefix` | `pipeline-runs` | S3 key prefix for cache files |
| `--pipeline-id` | `$GRAPHRAG_PIPELINE_ID` | Existing run to resume/inspect. If the flag is omitted it falls back to the `GRAPHRAG_PIPELINE_ID` environment variable. |
| `--resume-from-stage` | — | Stage to resume from (requires `--pipeline-id`) |
| `--verify-metadata` | off | Verify pipeline metadata integrity (needs `--pipeline-id`); exits non-zero when it is corrupt |
| `--repair-metadata` | off | Attempt metadata repair (needs `--pipeline-id`); exits non-zero when the repair fails |
| `--continue-on-error` | off | Keep going when a stage errors |
| `--enabled-stages` | all | Comma-separated stage list to run |
| `--metrics-sink` | `none` | `none`, or `cloudwatch` (emits CloudWatch EMF — Embedded Metric Format — metrics to stdout as dimensionless series; `pipeline_id` is recorded as a log property, not a dimension, so runs add no new metric series) |
| `--config-path` | — | Path to `config.yaml` |

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
7. **`graph_resolution`** — fuzzy-match and merge duplicate entities/relationships.
8. **`claim_extraction`** — optional; extract factual claims (off by default).
9. **`claim_resolution`** — optional; dedupe extracted claims.
10. **`graph_analysis`** — centrality metrics + graph statistics.
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

**Resume vs. force-rebuild:** without `--force-rebuild`, completed stages are
cached and skipped on re-run. Pass `--pipeline-id` to resume a specific prior
run; with `--resume-from-stage` you re-run from a chosen stage onward, otherwise
the pipeline auto-detects the first failed/incomplete stage. `--force-rebuild`
discards all cache and starts over.

**S3 sync** keeps the stage cache in `s3://<bucket>/<prefix>/...`, so a fresh
process (e.g. a new Fargate task) can resume without recomputing finished
stages. A failed or partial sync (the initial download or the final upload)
fails the run with `CacheSyncError` and `run-ingestion` exits non-zero, so in a
phased Step Functions run the phase that lost its checkpoint is the one marked
failed. An empty remote cache on a fresh pipeline id is not an error. For
embeddings specifically, set
`indexing.opensearch.persist_embedding_cache: true` to avoid re-embedding
unchanged text across runs.

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
empty/null, the translation stage is an `is_noop` skip — an English-only corpus
pays **no** translation LLM cost even with `enabled: true`. Language-aware
OpenSearch analyzers are configured under
`indexing.opensearch.language_analyzers` (e.g. `ko: nori`); unlisted languages
fall back to `default_analyzer`.

---

## 4. Querying (`run-rag`)

### CLI flags (verified)

| Flag | Default | Meaning |
|---|---|---|
| `--query`, `-q` | — | Single query — provide this or `--interactive` (exactly one is required) |
| `--interactive`, `-i` | off | Interactive chat (auto-enables memory) |
| `--mode` | `rag` | `rag` (full generation) or `search` (retrieval only) |
| `--conversation-id` | — | Continue an existing conversation |
| `--use-memory` | off | Enable conversation memory (auto in interactive) |
| `--suffix` | — | Index/label suffix for multi-tenant or versioned indices |
| `--enable-thinking` | off | Enable model step-by-step reasoning |
| `--search-strategy` | `auto` | `auto`, `drift`, `global`, `local`, `simple`, `mix`, `hybrid`, `naive` |
| `--search-type` | `hybrid` | `hybrid`, `lexical`, `vector` |
| `--top-k` | `10` | Max search results |
| `--retrieval-multiplier` | `1` | Increase retrieval depth |
| `--disable-query-processing` | off | Skip translation + entity extraction |
| `--filters` | — | `key:value` attribute filters (space-separated) |
| `--output-format` | `text` | `text` or `json` |
| `--verbose`, `-v` | off | Show query-processing info, sources, metrics |
| `--config-path` | — | Path to `config.yaml` |

### Search strategies — when to use which

**Choosing the methodology.** GraphRAG strategies excel at *summarization and
thematic synthesis* over a corpus (community reports give global coverage).
LightRAG strategies are faster and lean on *dual-level keyword* retrieval — good
for keyword-driven lookups and as a low-cost baseline. Both run through the same
hybrid scorer (BM25 lexical + vector semantic + graph traversal + RRF + Bedrock
rerank); only the retrieval algorithm differs.

**GraphRAG (community-summary):**

| Strategy | Use when | How it works |
|---|---|---|
| `simple` | Fast factual lookups; straightforward questions | Direct OpenSearch vector + keyword retrieval, no graph traversal. Fastest. Includes the claims index when claim extraction is on. |
| `local` | Detailed questions about specific entities/concepts | Extracts query entities → Neptune graph traversal for neighbors/relationships → combined with vector/keyword hits. Injects claims (covariates) when enabled. |
| `global` | Broad, thematic, "what are the main themes" questions | Uses community reports + map-reduce over dynamically selected communities. Best for high-level synthesis. |
| `drift` | Complex, multi-faceted questions needing exploration | Iterative query refinement/expansion with convergence detection across rounds. |
| `auto` | You don't know / general use (the default) | An LLM router (`search.strategy_selection_model_id`) picks the best strategy from the query among `search.auto_routable_strategies` (default local, mix, global, drift). |

**LightRAG (dual-level keyword):**

| Strategy | Use when | How it works |
|---|---|---|
| `mix` | General LightRAG use; balances graph + chunks | Low-level keywords → entity index, high-level keywords → relationship index, one-hop incident-relationship / endpoint-entity expansion (Neptune multi-hop expansion is opt-in: `search.lightrag_search.enable_graph_expansion`), **plus** naive vector chunk retrieval blended in. |
| `hybrid` | Keyword-driven graph questions | Same as `mix` but without the extra naive chunk blend. |
| `naive` | Fast baseline / comparison eval | Pure vector chunk retrieval, no graph. The LightRAG baseline. |

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
run-rag --query "..." --filters attr_category:research type:PERSON --config-path config.yaml
```

Filters compile to `term`/`terms`/`range` clauses on OpenSearch and `has`
steps on Neptune. Each store accepts the fields its indexer writes:

| Store | Filterable fields |
|---|---|
| Text units | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.<path>` |
| Entities | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `attr_<key>`, `attributes.<path>` |
| Relationships | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids` |
| Claims | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `description`, `source_text` |
| Community reports | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `attr_<key>`, `attributes.<path>` |
| Neptune entity vertices | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `attr_<key>` (where present) |
| Neptune community vertices | `id`, `name`, `level`, `parent`, `size`, `period`, `children` |

On OpenSearch, `attr_<key>` is a document attribute: entry `<key>` of a
document's `filters` metadata is indexed as `attr_<key>`. On Neptune entity
vertices, `attr_<key>` holds the entity's own extracted attributes (for example
`attr_role`). Use exact-match fields (keyword or
numeric) for precise filtering; a `term` filter on an analyzed text field such
as `description` matches single lowercase tokens. A range filter takes
`{"gte": ..., "lte": ...}` through the Python API.

Each filter applies only to the stores that declare its field, so
`type:PERSON` narrows entities, claims, and Neptune entities and leaves text
units unfiltered. The two stores treat `attr_<key>` differently:

- OpenSearch applies `attr_<key>` and `attributes.<path>` strictly on text
  units, entities, and community reports. A document without the attribute is
  excluded, so community reports, which usually lack document attributes, drop
  out of an attribute-filtered query. This is deliberate (fail-closed): an
  attribute filter never returns content it cannot vouch for.
- Neptune applies `attr_<key>` on entity vertices where present: a vertex
  passes when the property matches or when it has no such property. A
  document-attribute filter such as `attr_category` therefore leaves graph
  expansion intact, while an entity-attribute filter such as `attr_role:buyer`
  removes entities whose role differs. Every other key is strict on both
  stores.

A filter key that no store the selected strategy reads declares (for example
the earlier `category` or `entity_type`) raises `InvalidFilterError`, whose
message lists the filterable keys. Earlier releases ignored such keys silently
and returned unfiltered results. The schema is defined in
`unified_kg_rag/adapters/storage/filter_schema.py`.

### Interactive mode & conversation memory

```bash
run-rag --interactive --config-path config.yaml
# or continue a named session:
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

Interactive mode auto-enables memory. In-session commands:

- `help` — list commands
- `new` — start a fresh conversation (new ID)
- `set-filter key:value` — add/update a filter
- `clear-filters` — remove all filters
- `show-config` — show the active configuration
- `quit` / `exit` — end

For single-shot multi-turn from the CLI, reuse the same `--conversation-id` with
`--use-memory`. Memory limits are under the `memory` config section.

Memory also tracks entities across turns: after each user message an LLM call
(`search.entity_extraction_model_id`) extracts the entities it mentions, and the
most relevant ones from earlier turns are added to the next query's entity focus.
A follow-up such as "what about its suppliers?" therefore still retrieves around
the entity named in an earlier turn.

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
    table_name: "unified-kg-rag-on-aws-doc-status"
    create_table_if_missing: true
```

With this on, each `run-ingestion` diffs the corpus against the DynamoDB
document-status registry by **content hash**.

### Workflows

- **Add a document:** drop the new file into the source directory and re-run
  `run-ingestion`. Only the new file is parsed/extracted/indexed; its entities
  and relationships merge into the existing graph (idempotent `upsert_*`).
- **Modify a document:** edit the file and re-run. The content hash changes, so
  the document is treated as changed: its old artifacts are removed and the new
  version is re-indexed.
- **Delete a document:** remove it from the source directory and re-run. Its
  **exclusive** artifacts (entities/relationships seen only in that document,
  tracked via per-document lineage in the registry) are deleted; artifacts
  shared with surviving documents are kept.

### Deletion scope

A document's registry key is its index suffix (`document_parsing.index_value`,
plus `indexing.additional_suffix`) and its path relative to the source
directory. A run only treats as deleted the documents recorded under its own
**scope**: the same index suffix and the same corpus source
(`document_parsing.source_scope`, by default the resolved source directory;
the container entrypoint sets it to the S3 URI it syncs from). So:

- a run for another tenant (another `index_value`) never deletes this tenant's
  documents, even when both corpora are staged in the same local directory;
- running a subfolder as its own source directory never deletes the rest of
  the corpus (its files register as separate documents, so do not index the
  same files from two roots into one suffix);
- a file that fails to parse or load is reported as `failed` and keeps its
  indexed content until a run reads it again.

Moving a local corpus to another directory changes its default scope: set
`source_scope` to a stable name first, or rebuild.

### Document size limit

The registry stores each document's artifact ids (text units, entities,
relationships, claims, communities, reports) in one DynamoDB item, and an item
is limited to 400 KB, roughly 10,000 ids. A document that produces more fails
the indexing stage with an error naming the file; split it into smaller files.

### Cross-run merge

By default (`indexing.cross_run_merge: true`) a delta run *unions* the delta
with existing graph state (description / `text_unit_ids` / frequency / weight)
before upsert, so an entity shared with unchanged documents keeps their
descriptions and chunk lineage (which `mix` follows). Each delta run reads the
touched entities and relationships back from the graph first, and merged
descriptions over the summarization budget are re-summarized. Setting it to
`false` overwrites the affected fields with the delta's values instead; the
entity then loses its lineage to the unchanged documents' chunks. Requires a
graph adapter that supports read-back (one without it degrades to overwrite).
When a document changes or is deleted, entities and relationships it shares
with other documents lose its chunks from `text_unit_ids` (frequency and
weight follow), but keep the description it contributed until a full rebuild.

---

## 6. Evaluation (`run-eval`)

### CLI flags (verified)

| Flag | Default | Meaning |
|---|---|---|
| `--eval-data-path` | **required** | JSON file of questions + ground truths |
| `--outputs-directory` | `evaluation.outputs_directory` | Where to save results |
| `--suffix` | — | Index/label suffix |
| `--enable-thinking` | off | Model reasoning |
| `--search-strategy` | `auto` | Strategy used to answer each question |
| `--search-type` | `hybrid` | Search method |
| `--top-k` | `10` | Max results |
| `--retrieval-multiplier` | `1` | Retrieval depth |
| `--max-failure-rate` | `1.0` | Exit non-zero when the fraction of queries whose answer generation failed, or the fraction of a metric's attempted values that failed (`metric_outcomes`; skipped values do not count), exceeds this (0.0-1.0). A run where every query, or every attempt of a metric, failed always exits non-zero |
| `--verbose`, `-v` | off | Debug logging |
| `--config-path` | — | Path to `config.yaml` |

### Evaluators

Selected via `evaluation.enabled_evaluators`. An enabled evaluator that cannot
be built (e.g. no Bedrock access for the judge) or rejects its configuration
stops the run; with `processing.ignore_errors: true` it is dropped instead and
listed in `run_manifest.dropped_evaluators`. The same applies while scoring:
an evaluator error (e.g. a judge call that fails) stops the run unless
`ignore_errors` is `true`, in which case the metric is recorded as failed. A
judge reply without a usable score is always recorded as failed, never as 0.
By default all five are enabled:
the deterministic, LLM-free ones (`answer_match`, `retrieval`, `graph_aware`)
are free and skip a query that lacks their dataset fields; for a judge-free run
set `enabled_evaluators: [answer_match, retrieval, graph_aware]`.

- **`langchain`** — LangChain-based text similarity (`langchain_metrics`:
  `correctness`, `partial_correctness`). Needs `answer` ground truth.
- **`ragas`** — RAGAS metrics (`answer_correctness`, `answer_relevancy`,
  `context_precision`, `context_recall`, `faithfulness`). `context_precision`
  makes one judge call per context, so it scores at most
  `evaluation.ragas_max_contexts` (default 20; `null` = no cap) top-ranked
  sources per query, applied before the `max_context_tokens` budget — it is
  `context_precision@N`. Without the cap, strategies that report 100+ sources
  (e.g. LightRAG `mix`) hit `ragas_timeout`. `faithfulness` and
  `context_recall` see every source within `max_context_tokens`, so a claim
  backed by a lower-ranked source is not marked unsupported. Each report
  records what the judge saw: `judge_contexts` / `judge_context_tokens` and
  `context_precision_contexts` / `context_precision_context_tokens`. None of
  this changes what the answer model saw.
- **`graph_aware`** — deterministic, **LLM-free** entity/relationship
  **coverage = recall**: of the expected graph artifacts, how many appear in the
  generated answer, using the same normalization and phrase matcher as
  `answer_contains` (whole-word match; Korean particles tolerated; substring
  match for single-word CJK text). A relationship given as `{"source": "A", "target": "B"}`
  or `"A -> B"` counts when the answer mentions both endpoints; any other string
  must appear as a phrase. Needs `expected_entities` / `expected_relationships`
  in the dataset. **Precision and F1 are deliberately
  NOT emitted** — enumerating every entity in a free-text answer isn't reliably
  possible, so reporting precision/F1 would only re-label the recall signal.
- **`retrieval`** — deterministic, LLM-free: did the sources the answer model saw
  include the gold documents? `hit_at_k`, `recall_at_k` (k =
  `evaluation.retrieval_k`, default 5) and `mrr` against `reference_sources`.
  Text-unit sources name their file directly; entity, relationship and
  community-report sources are attributed to the files of the text units in
  their lineage (`text_unit_ids`, looked up in the chain's document store in
  one batch per suffix). A community report's lineage is its whole community,
  so `global`/`drift` scores are an upper bound on what the answer model read.
  Only attributable sources are ranked, and `attributable_fraction`
  (attributable / reported sources, also in `grouped_statistics` per strategy)
  shows how much of the context the rank metrics cover. Matching is
  case-insensitive on the full name or, when the name ends in a file extension
  (`.` + 1-5 letters/digits, at least one letter), its stem: `docs/Terms.pdf` =
  `terms.pdf` = `terms`. A `/` is a directory separator only in a path-like
  value (a file extension, a URI scheme, or a leading `/`, `./`, `~/`), so
  titles such as `St. Louis Cardinals` or `AC/DC` are compared whole. The rank
  metrics are skipped when a query has no `reference_sources`, or has sources
  but none is attributable; a query that retrieved no sources at all scores 0
  (a miss).
- **`answer_match`** — deterministic, LLM-free answer scores against `answer`
  and optional `metadata.answer_aliases`, taking the max over them.
  **`answer_contains`** (1.0 when the gold answer or an alias appears in the
  generated answer as a whole-word phrase) is the headline deterministic
  metric: long-form RAG answers rarely equal a short gold span, so it tracks
  correctness far better than exact match. Korean particles are tolerated
  (gold `서울 특별시` matches `서울 특별시는`), and a single-word CJK gold is
  matched as a substring. SQuAD-style `exact_match` and `token_f1` are also
  emitted for comparison with published benchmarks. Text is NFKC-normalized, then normalized as in the official SQuAD
  v1.1 script: lowercase, punctuation deleted (`1,000` = `1000`), English
  articles dropped, whitespace collapsed. Token F1 splits on whitespace, so for
  Chinese/Japanese text it degrades to exact match; Korean particles
  (`서울은`) make both metrics under-count.

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
run-eval --eval-data-path my_eval_data.json --config-path config.yaml

run-eval --eval-data-path my_eval_data.json --outputs-directory ./results --config-path config.yaml

run-eval --eval-data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml
```

Results are written to the outputs directory as
`evaluation_{results,reports,summary}_<timestamp>.json`; when every answered
query used the same strategy the name becomes `..._<strategy>_<timestamp>.json`.
The summary holds, per metric, mean/median/stdev/min/max/count
(`metric_statistics`) and scored/failed/skipped counts (`metric_outcomes`), plus:

- `grouped_statistics` — the same statistics split by `search_strategy` (the
  strategy actually used, which varies per query under `auto`), `category` and
  `difficulty`.
- `abstention_statistics` — how often the chain returned its fixed
  no-context reply ("I could not find relevant information…") instead of an
  answer: `abstained`, `answered`, `abstention_rate`, the same `per_strategy`,
  and for `answerable: false` items an `unanswerable` block (`total`,
  `correct_abstentions`, `accuracy`). On answerable items an abstention is
  graded like any answer (normally a miss); each result carries `abstained`.
- `run_manifest` — CLI arguments, model ids (answer generation, evaluation
  judge/embedding), package version, git commit (`git_sha`, and `git_dirty`
  when tracked files differ from it; both `null` unless the package runs from
  a checkout that tracks it), `config_sha256` of the full resolved config, `library_versions`
  (ragas, langchain*), the dataset (path + file sha256, query count and a hash
  of the parsed content) and a UTC timestamp, so two runs can be compared.
  It also lists `dropped_evaluators`. `EvaluationManager.evaluate_dataset`
  builds it, so library callers get it too (pass `dataset_path=` / `cli_args=`
  to record those).

Per-query reports list each evaluator's `metrics`; their `overall_score` is
kept for JSON compatibility but is always `null` (an average of unrelated
metrics has no meaning). Each result also records `retrieved_source_ids`: per reported source, in rank
order, the file names it is attributed to (`[]` when unattributable).

---

## 7. Visualization (`run-visualization`)

This is a **standalone** renderer that draws from an already-exported
visualization-data JSON. It does **not** re-run ingestion or touch AWS.

When `graph.visualization.enabled` is `true`, the community-detection stage of
`run-ingestion` renders the visualizations and also writes
`visualization_data.json` into `graph.visualization.outputs_directory`. When
that is unset (the default), ingestion writes to
`<cache.local_directory>/<pipeline_id>/visualization/`, so with S3 cache sync
enabled `visualization_data.json` is uploaded with the cache (the sync copies
`.json` files only; re-render the HTML locally with `run-visualization`). That
file is the `--data-path` input. The interactive graph keeps only the top
`interactive.max_nodes` nodes by degree (default 2000) so large graphs still
render in a browser. It holds the graph nodes/edges, the computed `layout`, the
community hierarchy, and centrality; vector attributes (`embedding`,
`*_embedding`) are omitted to keep the file small.

Layout and error behaviour:

- `embedding_method: "none"` uses a spring layout and never constructs a
  Bedrock embedding client.
- `embedding_method: "node2vec"` embeds each node's `name: description` with
  Bedrock and reduces it with `layout_method`. If embedding fails, the
  visualization step fails when `processing.ignore_errors` is `false` (ingestion
  itself continues and logs the failure). With `ignore_errors: true` it logs an
  ERROR, falls back to a topology-only spring layout, and records
  `"layout_degraded": true` in `visualization_data.json`. A failed
  dimensionality reduction falls back the same way (seeded spring layout,
  `layout_degraded: true`) whatever `ignore_errors` says.
- `embeddings.bedrock_model_id` must be one of the supported embedding model
  ids; any other value is rejected when the configuration loads.
- Edge width/opacity in the interactive graph is scaled relative to the graph's
  own weight range (log-scaled, then min-max normalised), so 1-10 strength
  scores and merged counts are both distinguishable.
- Only files a renderer actually wrote are reported (an earlier run's file of
  the same name is removed first). If nothing is rendered (e.g. an empty graph),
  `run-visualization` logs an error and exits with status `1`.

### CLI flags (verified)

| Flag | Default | Meaning |
|---|---|---|
| `--data-path` | **required** | Exported visualization-data JSON |
| `--output-dir` | `visualization_outputs` | Where to write rendered files |
| `--renderers` | all registered | Renderers to run: `interactive`, `static` |
| `--config-path` | — | Path to `config.yaml` |

The two registered renderers are **`interactive`** (pyvis) and **`static`**
(Bokeh). Their settings live under `graph.visualization` (`interactive.*`,
`static.*`, plus `embedding_method`/`layout_method`).

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-dir ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

---

## 8. Prompt tuning (`run-prompt-tuning`)

Samples documents from a directory, profiles the corpus (domain / language /
persona / entity types) via Bedrock, and writes a domain-adapted
`custom_prompts` YAML fragment for you to review and merge into `config.yaml`.

### CLI flags (verified)

| Flag | Default | Meaning |
|---|---|---|
| `--source-directory` | **required** | Directory of documents (`.txt`/`.md`/`.markdown` plus every format `run-ingestion` parses, e.g. `.pdf`); `--source-dir` is accepted as an alias |
| `--output` | `tuned_prompts.yaml` | Output YAML path |
| `--max-docs` | `20` | Max documents to sample |
| `--config-path` | — | Path to `config.yaml` |

```bash
run-prompt-tuning --source-directory ./source --output tuned_prompts.yaml --config-path config.yaml
```

The output YAML contains a `custom_prompts` block (and a `profile` with the
detected domain). **Review it**, then copy the prompts you want into your
`config.yaml` under `custom_prompts:`. Plain-text files are read as-is; other
formats (PDF, CSV, JSON, custom `ParserFactory.register_loader` formats) go
through the same loaders as ingestion. Files that fail to parse are skipped
with a warning. If the profiling model returns no JSON profile, the command
exits non-zero and writes nothing (a default profile would look tuned).

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
  graph_extraction_system: |
    You are a medical knowledge extractor. Extract diseases, symptoms, treatments,
    and medications and their relationships. Prioritize clinical accuracy.
  graph_extraction_human: |
    Extract medical entities and relationships from this clinical text:
    {input_text}
    Extraction Limits:
    - Maximum Entities: {max_entities_per_chunk}
    - Maximum Relationships: {max_relationships_per_chunk}

  community_report_system: |
    You are a legal analyst. Report on case law, regulatory frameworks, and
    legal precedents within each topic cluster.

  entity_extraction_system: |
    You are a financial expert. Extract companies, instruments, markets, and metrics
    from user queries.
```

Available override keys (each `_system` + `_human`): `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `context_building`,
`entity_extraction`, `keyword_expansion`, `query_refinement`,
`drift_primer` (DRIFT primer, when `enable_primer` is set),
`strategy_selection`, `keywords_extraction` (LightRAG dual-level),
`global_map` (global-search map-reduce), plus the prompt-tuning
`corpus_profile` prompt.

**Recommended flow:** run `run-prompt-tuning` to generate a starting point →
review → merge the useful prompts + tune `entity_types` by hand → re-ingest.

---

## 10. Operations & troubleshooting

### IAM permissions

Grant the principal running the CLIs access to: Bedrock (InvokeModel /
InvokeModelWithResponseStream for your model IDs, plus embeddings), Neptune
(connect / SigV4 for `use_iam: true`), OpenSearch (read/write the configured
indices), S3 (the configured bucket), and DynamoDB (when incremental indexing is
on).

> **Bedrock reranking needs its own statement.** The Rerank API
> (`bedrock:Rerank`, and `bedrock:InvokeModel` on the rerank model) is a
> separate action from chat/embedding model invocation. Give it `Resource: "*"`
> (or the appropriate rerank model/inference-profile ARNs) in its **own**
> statement — a model-scoped `InvokeModel` statement alone will not authorize
> reranking, and reranking is enabled by default (`search.reranking.enabled`).
> If you cannot grant it, set `search.reranking.enabled: false`.

### Common errors

- **`--source-directory is required`** — pass it (or set
  `$GRAPHRAG_SOURCE_DIRECTORY`); metadata-only ops (`--verify-metadata` /
  `--repair-metadata`) instead need `--pipeline-id`.
- **`--s3-bucket-name must be specified for S3 sync`** — `--s3-sync` requires
  `--s3-bucket-name`.
- **`--pipeline-id is required for --resume-from-stage`** — resuming needs the
  prior run's pipeline ID.
- **`Invalid stage names provided`** — use the exact stage names from §3 (the
  CLI prints the valid set).
- **`No module named 'unstructured'`** — install the `unstructured` extra to
  parse `.md`/`.html`, or convert those documents to a supported format.
- **OpenSearch auth failures with `use_iam: false`** — ensure `.env` has
  `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD`.
- **LightRAG `mix`/`hybrid` returns nothing** — confirm the relationships index
  was built during ingestion and that keyword extraction produced keywords (very
  short queries fall back to the raw query only under
  `raw_query_fallback_max_len`).
- **Pipeline failed mid-run** — re-run with `--pipeline-id <id>` to resume from
  the failed/incomplete stage; use `--verify-metadata` to check for corruption,
  `--repair-metadata` to attempt a fix, or `--force-rebuild` to start clean.

### Large / multilingual / heterogeneous corpora

- **Large corpora:** raise `processing.max_concurrency` /
  `processing.chunk_concurrency` (LLM stages are I/O-bound). For graph writes,
  raise `indexing.neptune.index_concurrency`; the Gremlin connection pool grows
  with it. Enable `indexing.opensearch.persist_embedding_cache` + `--s3-sync`
  so re-runs and multi-phase jobs don't recompute.
- **Multilingual:** set `processing.translation.source_language` /
  `target_language` (+ `additional_target_languages`) and add language analyzers
  under `indexing.opensearch.language_analyzers`. The translation stage no-ops
  for single-language corpora.
- **Heterogeneous domains:** tune `entity_types` to the union of your domains
  (or run separate indices per domain using `--suffix` /
  `indexing.additional_suffix` for multi-tenant separation).
- **Incremental:** enable DynamoDB so large corpora only pay for the changed
  delta on subsequent runs.

### Cost notes

LLM calls dominate cost. The biggest drivers: `graph_extraction` (one+ call per
chunk), `gleaning` (`max_rounds` extra passes), `community_detection` report
generation, `claim_extraction` (one call per text unit — off by default), and
answer generation per query. Levers: use cheaper models for mechanical stages
(chunking / translation / map-reduce / description summarization already default
to Haiku-class models), cap `gleaning.max_rounds`, leave `claim_extraction`
off unless needed, enable embedding/stage caching, and use incremental indexing
to avoid full re-ingests.

---

*See also: [README.md](../README.md) for the overview and quickstart, and
[docs/design.md](./design.md) for architecture and internals.*
