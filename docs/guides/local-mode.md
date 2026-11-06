# Local mode

Local mode runs ingestion, querying and evaluation on one host without an AWS
account, for development and evaluation (AWS stays the default). For a first
run (install, the two Ollama models, a sample corpus, a judge-free
evaluation), follow [Getting started locally](../getting-started/local.md).
This page covers what you can change and how local mode differs from AWS
mode.

## Providers

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
  (a relative `sqlite_path` resolves against the config file's directory,
  or the working directory without a config file); the
  `indexing.neptune` and `indexing.opensearch` settings apply to them.
- **Token counting** uses the script-aware estimate
  (`providers.llm.tokenizer: tiktoken` for a BPE encoding).
- **Custom kinds** register a builder with
  `adapters.provider_registry.register_provider`
  ([design §15](../design.md#15-extension-guide)).

## Differences from AWS mode

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
  The products run on numpy's BLAS: `run-eval` and `run-rag` default
  `OPENBLAS_NUM_THREADS` to 1, because concurrent queries each starting a
  thread per core spent about 8 times the CPU for the same scores. Set it
  to 1 in the environment of any other process serving concurrent queries
  (an async server, a worker) before numpy loads.
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
([Troubleshooting: Local mode](../troubleshooting.md#local-mode)).

## Provider flags

`run-ingestion`, `run-rag`, `run-eval`, `run-prompt-tuning` and
`run-visualization` take the provider flags (`--backend`, `--llm-provider`,
`--llm-model`, `--embedding-provider`, `--embedding-model`,
`--embedding-dimension`, `--rerank-provider`, `--model-base-url`). Each
overrides its key of the loaded config, a key the file sets included; an
omitted flag leaves the config as it is (`run-visualization` calls no model
or store and only accepts them). The keys each flag sets are in the
[CLI Reference](../reference/cli.md#provider-flags).

## Local stores in containers

[`docker/compose.local.yaml`](../../docker/compose.local.yaml) runs the stores
as servers: a Gremlin Server (in-memory TinkerGraph), a single-node
OpenSearch 2.13 with `analysis-nori` (security plugin off) and, under the
`models` profile, Ollama. OpenSearch is the engine of Amazon OpenSearch
Service, so use it to check ranking before an AWS deployment
([Multilingual corpora](./multilingual.md#analyzers) covers `nori` on AWS).
[`docker/config.local.yaml`](../../docker/config.local.yaml) selects
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

## Local stores with Bedrock models

Set `providers.llm.kind`, `providers.embedding.kind` and
`providers.rerank.kind` to `bedrock` in a config with `backend: local`
(Bedrock model ids then validate against the catalog) and use AWS credentials
with Bedrock access. For the containers, start only the stores
(`docker compose -f docker/compose.local.yaml up -d --wait`) and edit a copy
of `docker/config.local.yaml`.
