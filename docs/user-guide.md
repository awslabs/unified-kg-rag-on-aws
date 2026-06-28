# Unified Knowledge Graph RAG on AWS — User Guide

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

### Install

```bash
git clone <repository-url>
cd unified-kg-rag-on-aws

# uv (recommended)
uv sync --extra dev

# or pip
pip install -e .
```

Optional extra: parsing **Markdown (.md)** and **HTML (.html)** requires the
`unstructured` package. Without it, only `.pdf`, `.txt`, `.csv`, `.json` are
parsed (the parser raises a clear error naming the missing package for
`.md`/`.html`). Install it with your packaging tool if you need those formats.

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

---

## 2. Configuration

Create your config from the template and point every CLI at it with
`--config-path config.yaml`:

```bash
cp config-template.yaml config.yaml
```

The config is a nested Pydantic model (`unified_kg_rag/domain/models/config.py`).
`config-template.yaml` carries the full schema and inline notes; the most useful
sections and the knobs you will actually tune are below.

### 2.1 `aws` — service endpoints & credentials

```yaml
aws:
  region_name: "ap-northeast-2"
  profile_name: null              # named AWS profile, or null for default chain

  bedrock:
    region_name: "ap-northeast-2" # Bedrock can live in a different region
    assumed_role_arn: null
    enable_global_profile: true   # use cross-region inference profiles
    guardrail:                    # optional Bedrock Guardrails on every LLM call
      identifier: null            # set a guardrail ID/ARN to enable
      version: "DRAFT"
      trace: false

  neptune:
    endpoint:                     # REQUIRED — Neptune cluster endpoint
    port: 8182
    use_iam: true
    pool_size: 4                  # raise alongside indexing.neptune.index_concurrency

  opensearch:
    endpoint:                     # REQUIRED — OpenSearch domain endpoint
    port: 443
    use_ssl: true
    verify_certs: true
    use_iam: false                # false => username/password from .env

  s3:
    bucket_name:                  # REQUIRED for cache sync / embedding-cache persistence
    encryption:
      encryption_type: "AES256"   # NONE | AES256 | aws:kms
      kms_key_id: null

  dynamodb:                       # incremental indexing registry
    enabled: false                # set true to enable delta indexing
    table_name: "unified-kg-rag-on-aws-doc-status"
    create_table_if_missing: true
    billing_mode: "PAY_PER_REQUEST"
```

> **Guardrail placement note:** when deploying multi-region, the Bedrock
> Guardrail must exist in `bedrock.region_name` (the region LLM calls go to),
> not necessarily `region_name`.

### 2.2 `fixing` — auto-repair malformed model output

```yaml
fixing:
  enabled: true
  fixing_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
```

When an LLM returns malformed JSON for a structured stage, this re-asks a model
to repair it instead of failing. Leave on.

### 2.3 `processing` — concurrency, chunking, translation, extraction

LLM stages are Bedrock-I/O-bound, so concurrency can far exceed the CPU count.

```yaml
processing:
  max_concurrency: 20      # concurrent LLM calls within a batch
  chunk_concurrency: 4     # mini-batch chunks running at once
  batch_size: 10
  max_retries: 3
  ignore_errors: false
  deduplicate: false
  resolution_method: "minhash"      # minhash | sequence_matcher
  similarity_threshold: 0.6         # entity-resolution fuzzy-match threshold

  document_parsing:
    source_directory:               # overridden by --source-directory CLI flag
    target_directory: null
    index_value: null
```

**Chunking** — `intelligent` uses an LLM to pick semantic boundaries; `simple`
splits by size. Most-tuned: `min_chunk_size` / `max_chunk_size`.

```yaml
  chunking:
    chunker_type: "intelligent"     # intelligent | simple
    chunking_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
    content_type: "markdown"
    min_chunk_size: 5000
    max_chunk_size: 50000
    chunk_overlap: 500
    pre_chunk_size: 50000
    pre_chunk_overlap: 500
    fallback_chunk_size: 50000
    max_marker_miss_rate: 0.1
```

**Translation** — runs as a pipeline stage, but is a **no-op** (zero LLM cost)
when `source_language == target_language` and `additional_target_languages` is
empty. See §3 for the multilingual workflow.

```yaml
  translation:
    enabled: true
    translation_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
    source_language: "en"           # predominant source language (no-op skip only)
    target_language: "en"
    additional_target_languages: null
```

**Graph extraction** — the heart of ingestion. `entity_types` is the single
most impactful domain-adaptation knob (see §9). Each item is
`"LABEL: short description"`; an empty list lets the model pick freely.

```yaml
  graph_extraction:
    extraction_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_entities_per_chunk: 50
    max_relationships_per_chunk: 50
    entity_confidence_threshold: 0.0
    entity_types:
      - "PERSON: Names, individuals, roles, titles"
      - "ORGANIZATION: Companies, institutions, departments, groups"
      - "LOCATION: Places, addresses, geographic areas, facilities"
      - "CONCEPT: Ideas, theories, methodologies, frameworks, principles"
      - "OBJECT: Documents, tools, products, systems, technologies"
      - "EVENT: Meetings, projects, activities, processes, incidents"
      - "TEMPORAL: Dates, time periods, schedules, deadlines"
    description_summarization:      # collapse over-long merged descriptions
      enabled: true
      summary_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
      force_summary_threshold_tokens: 600
      max_summary_tokens: 256
    entity_grounding:              # hallucination guard (opt-in, off by default)
      enabled: false              # drop/penalize entities+relationships whose
      action: "drop"              # verbatim source_text span is absent from the
      penalty_factor: 0.5         # source chunk (the model invented them); also
      min_span_tokens: 4          # gates gleaner MISSING_* additions
      min_overlap_ratio: 0.6
```

**Entity grounding** is a provenance guard against extraction hallucination: the
extraction prompt asks the model for a verbatim `source_text` span per entity
and relationship, and when `enabled`, anything whose span is not found in its
source chunk is dropped (or confidence/weight-penalized). It is conservative —
a missing or very short span is treated as grounded — so turning it on never
deletes legitimate artifacts on weak signal. It also gates gleaner-introduced
entities/relationships via their `text_evidence`.

**Gleaning** — iterative extraction passes that catch entities/relationships
missed on the first pass (quality vs. cost trade-off).

```yaml
  gleaning:
    enabled: true
    graph_refinement_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_rounds: 3
    convergence_threshold: 0.8
    quality_threshold: 0.9
    min_improvement_threshold: 0.05
    # ...count-based quality/convergence scaling constants
```

**Claim extraction** — OFF by default; each text unit costs an extra LLM call.
When ON, `local` search injects matching claims (MS GraphRAG covariates) and
`simple` search includes the claims index in its sweep.

```yaml
  claim_extraction:
    enabled: false
    extraction_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_entities_per_prompt: 100
```

### 2.4 `graph` — analysis, community detection, visualization

```yaml
graph:
  analysis:
    centrality:
      calculate_degree: true
      calculate_betweenness: true
      calculate_pagerank: true
      calculate_closeness: false
      calculate_eigenvector: false
      pagerank_alpha: 0.85
    statistics:
      calculate_density: true
      calculate_clustering: true
      calculate_components: true

  community_detection:              # Leiden clustering
    enabled: true                   # set false for a lighter LightRAG-only
                                    # ingestion (skips Leiden + community-report
                                    # LLM calls; GraphRAG global/drift need it)
    resolution: 1.0
    random_state: 42
    max_levels: 5
    min_community_size: 3
    auto_resolution: true
    report_generation:              # LLM-generated community summaries (used by global search)
      enabled: true
      report_generation_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
      max_entities_per_report: 50
      max_report_context_tokens: 4000

  visualization:
    enabled: true
    outputs_directory: "outputs/visualization"
    embedding_method: "node2vec"
    layout_method: "umap"           # umap | tsne | pca
```

### 2.5 `indexing` — OpenSearch & Neptune write side

```yaml
indexing:
  reset: false
  additional_suffix: null           # appended to default index/label suffix
  cross_run_merge: false            # on delta runs, union with existing graph state

  opensearch:
    embedding_model_id: "amazon.titan-embed-text-v2:0"
    embedding_dimension: null
    persist_embedding_cache: false  # cache embeddings to S3 across runs/phases
    text_units_index_prefix: "graphrag-text-units"
    entities_index_prefix: "graphrag-entities"
    community_reports_index_prefix: "graphrag-community-reports"
    relationships_index_prefix: "graphrag-relationships"   # enables LightRAG high-level retrieval
    claims_index_prefix: "graphrag-claims"
    default_analyzer: "standard"
    language_analyzers:             # per-language text analyzer (extend freely)
      en: "english"
      ko: "nori"
    vector_search:
      ef_construction: 128
      m: 24
      ef_search: 100
      space_type: "cosinesimil"
      engine: "faiss"               # faiss is the modern kNN engine (nmslib deprecated)

  neptune:
    batch_size: 100
    index_concurrency: 1            # >1 fans write batches over a thread pool
    max_hops: 3                     # neighbor-expansion depth at retrieval time
    max_results_per_hop: 50
    min_entity_importance: 0.5
```

> The `*_index_prefix` keys are the index-name config names.

### 2.6 `search` — retrieval, fusion, reranking, per-strategy knobs

```yaml
search:
  translation_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
  entity_extraction_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
  strategy_selection_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"   # the `auto` router
  context_building_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
  answer_generation_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"    # the answer LLM

  hybrid:
    lexical_weight: 0.5
    vector_weight: 0.5

  fusion:
    method: "rrf"                   # rrf | weighted
    rrf_k: 60
    diversity_lambda: 0.5           # MMR: 1.0 = pure relevance, 0.0 = max diversity
    fusion_weights: { ... }         # only used when method: weighted

  reranking:
    enabled: true
    rerank_model_id: "cohere.rerank-v3-5:0"
    top_k: 100

  lightrag_search:
    raw_query_fallback_max_len: 50  # short queries fall back to raw query as a keyword

  global_search:
    max_communities: 10
    use_dynamic_selection: true
    enable_map_reduce: true
    map_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
    max_map_reduce_tokens: 8000

  local_search:
    entity_frequency_threshold: 20  # drop overly-generic graph-expanded entities

  drift_search:
    enable_query_refinement: true
    enable_keyword_extraction: true
    max_iterations: 3
    initial_top_k: 5

  token_manager:
    max_context_tokens: 200000
```

### 2.7 `memory`, `cache`, `logging`

```yaml
memory:
  max_conversations: 100
  max_messages_per_conversation: 20
  max_conversation_age_hours: 168

cache:
  ttl_seconds: 86400               # null = never expire
  chunking:
    enabled: true
    max_file_size_mb: 50

logging:
  level: "INFO"
  log_format: "structured"
  log_to_file: true
  log_file_path: "logs/log.txt"
```

### 2.8 `evaluation`

```yaml
evaluation:
  outputs_directory: "outputs/evaluation"
  evaluation_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
  enabled_evaluators:
    - langchain
    - ragas
    # - graph_aware                # opt-in; needs expected_entities/relationships
  langchain_metrics: [correctness, partial_correctness]
  ragas_metrics: [answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness]
  save_detailed_results: true
```

### 2.9 `custom_prompts`

Every prompt has a `*_system` / `*_human` override (default `null` = use the
built-in prompt in `unified_kg_rag/domain/prompts/`). See §9. Override what you
need; leave the rest `null`.

---

## 3. Ingestion (`run-ingestion`)

Ingestion turns a directory of documents into a knowledge graph indexed in
OpenSearch + Neptune.

### CLI flags (verified)

| Flag | Default | Meaning |
|---|---|---|
| `--source-directory` | `$GRAPHRAG_SOURCE_DIRECTORY` | Directory of source documents (required to run) |
| `--target-directory` | source dir | Where parsed documents are written |
| `--cache-directory` | `cache` | Pipeline cache + intermediate results |
| `--force-rebuild` | off | Ignore all existing cache; rebuild from scratch |
| `--s3-sync` | off | Sync cache to S3 (requires `--s3-bucket-name`) |
| `--s3-bucket-name` | — | S3 bucket for cache sync |
| `--s3-prefix` | `pipeline-runs` | S3 key prefix for cache files |
| `--pipeline-id` | `$GRAPHRAG_PIPELINE_ID` | Existing run to resume/inspect |
| `--resume-from-stage` | — | Stage to resume from (requires `--pipeline-id`) |
| `--verify-metadata` | off | Verify pipeline metadata integrity (needs `--pipeline-id`) |
| `--repair-metadata` | off | Attempt metadata repair (needs `--pipeline-id`) |
| `--continue-on-error` | off | Keep going when a stage errors |
| `--enabled-stages` | all | Comma-separated stage list to run |
| `--metrics-sink` | `none` | `none` or `cloudwatch` (EMF to stdout) |
| `--config-path` | — | Path to `config.yaml` |

### The 12 pipeline stages

Run order (`DataIngestionPipeline.STAGE_CLASSES`). Use the stage **names**
(case-insensitive) with `--enabled-stages` / `--resume-from-stage`:

1. **`document_parsing`** — extract text per format (`.pdf`, `.txt`, `.csv`,
   `.json`; `.md`/`.html` with the `unstructured` extra).
2. **`document_loading`** — load parsed documents into the pipeline corpus.
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
stages. For embeddings specifically, set
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
| `--query`, `-q` | — | Single query (mutually required with `--interactive`) |
| `--interactive`, `-i` | off | Interactive chat (auto-enables memory) |
| `--mode` | `rag` | `rag` (full generation) or `search` (retrieval only) |
| `--conversation-id` | — | Continue an existing conversation |
| `--use-memory` | off | Enable conversation memory (auto in interactive) |
| `--suffix` | — | Index/label suffix for multi-tenant or versioned indices |
| `--enable-thinking` | off | Enable model step-by-step reasoning |
| `--search-strategy` | `auto` | `auto` `drift` `global` `local` `simple` `mix` `hybrid` `naive` |
| `--search-type` | `hybrid` | `hybrid` `lexical` `vector` |
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
| `auto` | You don't know / general use (the default) | An LLM router (`search.strategy_selection_model_id`) picks the best strategy from the query. |

**LightRAG (dual-level keyword):**

| Strategy | Use when | How it works |
|---|---|---|
| `mix` | General LightRAG use; balances graph + chunks | Low-level keywords → entity index, high-level keywords → relationship index, Neptune expansion, **plus** naive vector chunk retrieval blended in. |
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
run-rag --query "..." --filters category:research entity_type:person --config-path config.yaml
```

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

### Cross-run merge

By default a delta run overwrites the affected graph fields. Set
`indexing.cross_run_merge: true` to instead *union* the delta with existing
graph state (description / `text_unit_ids` / frequency / weight) before upsert —
useful when an entity's description should accumulate across documents. Requires
a graph adapter that supports read-back. Off by default.

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
| `--verbose`, `-v` | off | Debug logging |
| `--config-path` | — | Path to `config.yaml` |

### Evaluators

Selected via `evaluation.enabled_evaluators`:

- **`langchain`** — LangChain-based text similarity (`langchain_metrics`:
  `correctness`, `partial_correctness`). Needs `answer` ground truth.
- **`ragas`** — RAGAS metrics (`answer_correctness`, `answer_relevancy`,
  `context_precision`, `context_recall`, `faithfulness`).
- **`graph_aware`** — deterministic, **LLM-free** entity/relationship
  **coverage = recall**: of the expected graph artifacts, how many appear in the
  generated answer (case-insensitive substring match). Needs `expected_entities`
  / `expected_relationships` in the dataset. **Precision and F1 are deliberately
  NOT emitted** — enumerating every entity in a free-text answer isn't reliably
  possible, so reporting precision/F1 would only re-label the recall signal.
  (Opt in by uncommenting `graph_aware` in `enabled_evaluators`.)

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
    "expected_relationships": ["AI enables machine learning"],
    "metadata": { "search_strategy": "global" }
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

Per-item `metadata` (e.g. `search_strategy`) overrides the CLI defaults for that
question. `id` may also be given as `query_id`.

### Examples

```bash
run-eval --eval-data-path my_eval_data.json --config-path config.yaml

run-eval --eval-data-path my_eval_data.json --outputs-directory ./results --config-path config.yaml

run-eval --eval-data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml
```

Results (per-query details + a summary with mean/median/stdev/min/max per
metric) are written to the outputs directory.

---

## 7. Visualization (`run-visualization`)

This is a **standalone** renderer that draws from an already-exported
visualization-data JSON (produced during ingestion by
`GraphVisualizationManager.export_visualization_data`). It does **not** re-run
ingestion or touch AWS.

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
| `--source-directory` | **required** | Directory of text documents (`.txt`, `.md`, `.markdown`); `--source-dir` is accepted as an alias |
| `--output` | `tuned_prompts.yaml` | Output YAML path |
| `--max-docs` | `20` | Max documents to sample |
| `--config-path` | — | Path to `config.yaml` |

```bash
run-prompt-tuning --source-directory ./source --output tuned_prompts.yaml --config-path config.yaml
```

The output YAML contains a `custom_prompts` block (and a `profile` with the
detected domain). **Review it**, then copy the prompts you want into your
`config.yaml` under `custom_prompts:`. Note it only reads plain-text formats
(`.txt`/`.md`/`.markdown`) for profiling.

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
  raise `indexing.neptune.index_concurrency` *and* `aws.neptune.pool_size`
  together. Enable `indexing.opensearch.persist_embedding_cache` + `--s3-sync`
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
