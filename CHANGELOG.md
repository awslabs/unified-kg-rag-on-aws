# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial public release preparation: an AWS-native, open-source reference
framework that unifies the GraphRAG and LightRAG retrieval methodologies on
Amazon Bedrock, Neptune, OpenSearch, and DynamoDB. Entries marked
**Breaking** change configuration, stored index data, or a public interface.

### Added
- Two selectable retrieval methodologies: GraphRAG community-summary
  (`auto`/`drift`/`global`/`local`/`simple`) and LightRAG dual-level keyword
  (`mix`/`hybrid`/`naive`) on one ingestion/indexing/caching/hybrid-search stack.
- Incremental indexing via a DynamoDB document-status registry (content-hash
  diff, idempotent upserts, per-document lineage for deletion).
- Triple-hybrid retrieval (lexical + semantic + graph) with RRF fusion and
  Bedrock reranking; multilingual ingestion and retrieval.
- CLIs: `run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
  `run-prompt-tuning`.
- CDK deployment stack (`iac/`) with Well-Architected security defaults.
- Community reports carry `text_unit_ids`/`document_ids` keyword fields for
  filtering by source document (membership provenance, not citation support;
  existing reports stay empty until reports are regenerated and re-indexed).
- Claude Sonnet/Opus 5.5, Opus 4.6-4.8, Sonnet 4.6 and the OpenAI GPT models on
  Bedrock, with provider-aware request shaping (#117).
- `retrieval` (hit@k, recall@k, MRR against `reference_sources`) and
  `answer_match` (exact match and token F1, with `metadata.answer_aliases`)
  evaluators; both are LLM-free (#121).
- Evaluation summaries carry `grouped_statistics` (strategy, category,
  difficulty) and a `run_manifest` (CLI args, model ids, version, dataset
  sha256) (#121).
- `run-prompt-tuning` parses PDF/CSV/JSON and registered loaders like ingestion
  (#121).
- Extraction, gleaning and claim-extraction failure counts in
  `PipelineMetrics`/EMF and an `ExtractionFailures` alarm (#122).
- Warning per unknown configuration key; `logging.library_levels`; run, stage
  and query ids bound into log records; periodic INFO progress logs (#122).
- `evaluation.ragas_timeout`/`ragas_max_workers`/`ragas_max_retries` and
  `evaluation.judge_effort` for thinking-model judges (#113).
- Per-tier reasoning effort: `aws.bedrock.default_effort` (default `high`) and
  `fast_effort` (default `low`, used for calls on `fast_model_id`); a per-call
  effort such as `evaluation.judge_effort` still wins (#128).
- Local stores for development: `docker/compose.local.yaml` (Gremlin Server
  and OpenSearch with nori) with `docker/config.local.yaml`, enabled by
  `aws.neptune.use_ssl: false` (plain `ws://`) and
  `aws.opensearch.allow_anonymous: true`; models still use Bedrock. A smoke
  test runs the graph and vector adapters against them when `LOCAL_STORES=1`
  (#154).
- Callbacks, tags and metadata in the `config` passed to `GraphRAGChain`
  reach every nested LLM call (strategy routing, query processing, context
  building, and the DRIFT and global search calls). **Breaking** for custom
  strategies: `GraphRAGChain` calls `asearch(query, config=...)`, so an
  override must accept `config=None` (and should pass it to its own LLM
  calls). Every `setup_chain` chain is named
  after its prompt (e.g. `AnswerGenerationPrompt`) in traces (#156).

### Changed
- Model ids are free-form strings; `aws.bedrock.default_model_id` and
  `fast_model_id` set every role of their tier, `aws.bedrock.model_overrides`
  describes unknown models, and Claude 4.5 fails fast without an inference
  profile. Python callers drop `.value` on `*_model_id` (#127).
- **Breaking:** Claude Sonnet 5 defaults move to Sonnet 5.5; ingestion stage
  caches built with the old defaults are not reused (#117).
- **Breaking:** a filter key that no retriever of the strategy declares raises
  `InvalidFilterError`; filter scoping uses one declared schema, and Neptune
  `attr_*` filters apply where the property exists (#123).
- Strategies declare `query_inputs` on `@register_strategy`; source truncation
  is reported only as `metadata["truncated"]` (#123).
- **Breaking (index data):** entity names keep their display form and ids are
  keyed on a symbol-preserving key ("C++" and "C#" no longer collide); fully
  re-index existing graphs (#110).
- Entity resolution merges only names with identical identifier tokens and
  compatible types, grouped order-independently (#109).
- The Bedrock guardrail applies to the query path only
  (`aws.bedrock.guardrail.apply_to: all` restores the old scope) (#104).
- **Breaking (IaC):** `create_guardrail` decides whether the stack owns a
  guardrail; bring-your-own needs `-c create_guardrail=false` (#103).
- S3 cache uploads default to the bucket's own encryption (`BUCKET_DEFAULT`)
  (#105).
- Queries without `index_prefixes` sweep only the indices the pipeline builds
  (#111).
- Query-side entity and keyword extraction run only for strategies that read
  them (#94).
- `BatchProcessor` retries only the failed items of a partial batch and only
  retryable errors (#99, #120).
- CountTokens is negative-cached for models that reject it (#98).
- Bedrock client connection pools are sized to ingestion concurrency (#124).
- RAGAS judges score at most `evaluation.ragas_max_contexts` (20) contexts per
  sample, so context metrics are @N (#126).
- `mix`/`hybrid` retrievals run concurrently; multi-hop Neptune expansion is
  opt-in (`search.lightrag_search.enable_graph_expansion`) (#125).
- DRIFT searches the original query first and fuses with local's per-type
  quota; LLM convergence is opt-in (`drift_search.enable_llm_convergence`,
  threshold 0.8) (#125).
- Global search skips per-report LLM scoring
  (`global_search.use_dynamic_selection: true` restores it) and packs 5
  reports per map call (#125).
- Retrieval context budget defaults to 30,000 tokens
  (`token_manager.max_context_tokens: null` derives it from the model) (#125).
- Chunk defaults fit embedding/rerank limits: `max_chunk_size` 8,000,
  `fallback_chunk_size` 4,800, `min_chunk_size` 1,000 (were 50,000/50,000/5,000)
  (#125).
- Gleaning rounds after the first re-send only units that gained items; a
  single-child parent community reuses the child's report; short pre-chunks
  skip the chunking LLM; query embeddings are LRU-cached (#125).
- AUTO routes among local/mix/global/drift with the fast model
  (`search.auto_routable_strategies`, `strategy_selection_model_id`) (#125).
- Ingestion visualization output defaults under the synced cache directory;
  the interactive renderer keeps the top 2,000 nodes by degree (#122).
- Graph-aware relationship coverage counts `{"source", "target"}` pairs and
  `"A -> B"` strings by endpoint mentions (#121).
- Embedding and query-time LLM calls share `aws.bedrock.transient_retry`
  (5 attempts, 60 s budget) (#120).
- Development tools live only in the `dev` dependency group (`uv sync`); the
  Docker image config is tracked and uv is pinned by digest (#119).
- One `Providers` bundle (`unified_kg_rag.adapters.providers`) carries the
  boto3 session and the LLM, embedding, rerank and token-counter providers;
  `GraphRAGChain`, `DataIngestionPipeline` and `EvaluationManager` build it
  once (or accept `providers=`) and hand it to every component they construct.
- `GraphRAGChain` reuses search-strategy instances per event loop instead of
  building one per query.
- Conversation memory stays process-wide by default and is created from the
  first chain's config; `GraphRAGChain(memory_manager=MemoryManager(config))`
  isolates a chain.
- Incremental runs merge a touched entity or relationship with its graph
  state by default (`indexing.cross_run_merge: true`), so an entity shared
  with unchanged documents keeps their description and lineage (#134).
- **Breaking (index data):** community ids hash the level and sorted members
  (Leiden input is sorted), document ids hash the corpus-relative path and the
  full text, and registry keys hash the index namespace and relative path;
  rebuild existing indexes (`indexing.reset: true`) (#134).
- The answer prompt asks the model to chain facts across sources and answer
  directly before the support; the previous wording can be restored through
  `custom_prompts.answer_generation_system`/`_human` (#132).
- Neptune graph expansion fetches `traversal_fetch_multiplier` (3) times the
  result width and keeps the entities closest to the seeds and most important
  (`indexing.neptune.entity_importance_source`: `rank`, `degree` or `none`);
  importance used to read a property no vertex stored (#132).
- Local search adds the relationships incident to its expanded entities,
  in-network edges first (`search.local_search.include_bridge_relationships`),
  and queries its report, relationship and claim sections concurrently (#132).
- Global search passes the ranked map key points to the answer model without
  a reduce LLM call (`search.global_search.reduce_with_llm: true` restores it)
  and reserves fusion slots for community reports
  (`reserve_report_slots`, `text_unit_slots`) (#132).
- Fusion's MMR filter is quadratic instead of cubic and is skipped when the
  cut keeps every candidate; results are unchanged (#132).
- The CLIs and `GraphRAGChain`'s sync methods size the event loop's default
  executor to `processing.io_workers` (64). LangChain runs Bedrock `ainvoke`
  calls there, and Python's default of `min(32, CPUs + 4)` threads let only
  six run at once on a 2-vCPU task. Async hosts call
  `unified_kg_rag.shared.utils.configure_event_loop` (#142).
- The domain `Document` is a plain Pydantic model instead of a LangChain
  `Document` subclass, so `domain/` imports no LangChain (now enforced by a
  test). LangChain loader output is converted in
  `shared.utils.document_converter`; `DirectoryLoader` is no longer a LangChain
  `BaseLoader`. Field names are unchanged, and JSON written by earlier versions
  still loads (its `id`/`type` keys are ignored) (#149).
- The Bedrock model capability catalog (`LanguageModelInfo`,
  `EmbeddingModelInfo`, `RerankModelInfo`, `get_language_model_info`,
  `effective_max_output_tokens`) moved from `adapters.aws.bedrock` to
  `adapters.aws.bedrock_models`; `bedrock.py` keeps the factories, the
  guardrail handler and the cross-region helper (#150).
- The user guide and config field descriptions state that embedding and rerank
  model ids, unlike language-model ids, are a closed list that
  `aws.bedrock.model_overrides` does not cover: the embedding dimension is
  fixed into the OpenSearch vector mappings (#150).
- **Breaking** default: `aws.bedrock.region_name` defaults to `null` and then
  follows `aws.region_name` (an `AWS_REGION` override moves it too), and the
  `aws.region_name` default is `us-west-2` instead of `ap-northeast-2`, which
  offers no Bedrock rerank model. Set `aws.bedrock.region_name` to keep Bedrock
  in another region (#148).
- Every retry knob is named `max_attempts` and counts total attempts,
  including the first: `processing.max_attempts`,
  `indexing.neptune.max_attempts` and `evaluation.ragas_max_attempts` join
  `aws.bedrock.transient_retry.max_attempts`. The former Neptune key counted
  retries after the first try, so `max_retries: N` is read as
  `max_attempts: N+1` and the default stays 4 attempts. **Breaking** for
  library callers: `BatchProcessor(max_retries=...)` is now
  `BatchProcessor(max_attempts=...)`, and the unread
  `PipelineConfig.max_retries` is removed (#151).
- LLM JSON responses are parsed by one helper, `parse_llm_json`; its
  `strict=True` mode raises on unparseable text for LightRAG keyword
  extraction. LangChain judge scores and `partial_correctness` reasoning now
  also accept JSON wrapped in prose (#153).

### Deprecated
- `search.llm_retry`; use `aws.bedrock.transient_retry` (#120).
- `aws.bedrock.effort`; use `aws.bedrock.default_effort` (#128).
- `SearchQuery.metadata["lightrag_mode"]`; use `search_strategy` (#123).
- `processing.max_retries`, `indexing.neptune.max_retries` and
  `evaluation.ragas_max_retries`; use the `max_attempts` keys. A renamed key
  (including `search.llm_retry`) now logs a deprecation WARNING instead of
  being reported as an unknown key that is ignored (#151).

### Removed
- `indexing.neptune.min_entity_importance`: it thresholded an `importance`
  property no vertex stores; the key is now ignored with an unknown-key
  warning (#155).
- The unused `RetrieverType` enum and the latency-optimized inference path
  (`supports_performance_optimization` in model capability records), which no
  caller enabled (#147).
- `OpenSearchClient.aget_mapping` and the live-mapping/Neptune-probe filter
  scoping (#123).
- The `dev` and `docs` extras and the unused `asyncio-throttle` and `rapidfuzz`
  dependencies (#119).
- `S3EncryptionType.NONE` ("NONE" still validates as `BUCKET_DEFAULT`) and
  test-only helpers on the guardrail handler and token counter (#120).

### Fixed
- `NeptuneRetriever` seeding by name or query text (no `id` filter) always
  returned nothing: it required and sorted by an `importance` property that
  entity vertices never store. It now orders entities by `rank` and
  communities by `size`, with no threshold. Strategies seed by id, so their
  results are unchanged (#155).
- Prompt caching works on Converse: system prompts end with a native
  `cachePoint` once they reach the model's minimum cache size (#127).
- Requests cap output at `aws.bedrock.default_max_output_tokens` (16,384, with
  higher floors for long-output prompts) instead of reserving the model
  maximum against the TPM quota (#127).
- System-prompt placeholders are substituted when prompt caching is on (#93).
- `custom_prompts` overrides reach community-report, memory entity extraction
  and DRIFT refinement/expansion/primer chains (#120).
- Ingestion LLM stages fail permanent Bedrock errors on the first attempt, and
  XML that no recovery step parses reaches the output-fixing model (#120).
- Embedding and query-time LLM calls retry transient Bedrock errors such as
  HTTP 424 `ModelErrorException` (#115, #116).
- Global, DRIFT and history-aware paths no longer answer from empty or
  rejected evidence; primer and map-reduce summaries are not reported as
  sources (#96).
- `RAGOutput.sources` lists only what the answer model saw, and OpenSearch hits
  no longer carry embedding vectors (#97).
- MMR normalizes relevance before the diversity penalty (#100).
- Local and LightRAG honour caller filters; local and DRIFT fall back to the
  raw query when no entities are extracted (#101).
- Fatal errors (for example Neptune AccessDenied) surface from every retrieval
  path instead of reading as no results (#102).
- `stream`/`astream` stream answer tokens instead of raising `TypeError` (#107).
- RRF fusion rewards cross-store agreement on the same artifact and counts each
  artifact once per bucket.
- Stage-result cache keys fingerprint the config, model and prompts that
  determine each stage, so changed inputs no longer resume from stale output.
- The gleaner applies `ENTITY_CORRECTION`/`RELATIONSHIP_CORRECTION` issues and
  renames edge endpoints to match.
- Repeated XML tags in LLM output no longer drop the entity or relationship
  (#124).
- Parsed `<stem>.json` files are no longer written into the source directory;
  the export happens only to an explicit `target_directory` (#114, #120).
- The config loader maps `GRAPHRAG_DOC_STATUS_TABLE` and
  `GRAPHRAG_DOC_STATUS_CREATE_TABLE` from the IaC stack (#105).
- A failed S3 cache sync fails the run (`CacheSyncError`) (#106).
- `run-visualization` input is exported, edge widths scale with weight, and an
  embedding failure no longer yields a random layout (#108).
- The config file's `logging` section is applied and every record is rendered
  as JSON in structured mode; `run-rag --output-format json` keeps stdout
  pipeable; tqdm is disabled without a TTY (#122).
- `run-eval` validates the dataset at load and exits non-zero on all-failed
  runs or above `--max-failure-rate`; RAGAS judges the context text once (#121).
- Errored answers and unmeasurable metrics are excluded from evaluation means
  (#95).
- RAGAS no longer forces temperature on Claude 5 judges (#112).
- `config-template.yaml` validates as shipped (`source_directory` defaults to
  `source`) and documents the betweenness sampling and auto-resolution knobs;
  the user guides list common knobs in tables checked against `Config()`
  instead of copied YAML (#129).
- `GraphRAGChain.invoke`/`batch`/`stream` run on one chain-owned event loop,
  so `invoke` works inside a running loop and `batch` no longer leaks a
  retriever set per item.
- Reranking passes `top_n` per call instead of mutating the shared rerank
  model, and fusion + rerank run off the event-loop thread.
- Neptune opens and closes its connection off the event-loop thread, and a
  discarded `AsyncOpenSearch` client is closed.
- Conversation memory reuses the query step's entity extraction instead of a
  second LLM call per user turn.
- Extraction drops relationships whose endpoint entity was dropped instead of
  recreating it, reads an integer confidence of 1 as the scale minimum, and
  treats single letters of any script as designators in entity resolution
  (#130).
- Items without a description get a surrogate embedding text instead of being
  skipped from the vector indexes (#130).
- Communities are detected on the entity-only subgraph, so claim nodes no
  longer join communities or become singleton reports (#130).
- `.json` sources parse without the optional `jq` package, exclude patterns
  match relative to the source root, and repeated claim tags no longer drop
  the claim (#130).
- Incremental deletion is scoped to the run's namespace and corpus
  (`processing.document_parsing.source_scope`, set from
  `GRAPHRAG_SOURCE_SCOPE`); files that fail to parse are reported as `failed`
  instead of deleted, and delta runs no longer overwrite the corpus's
  communities and reports (#134).
- A changed corpus is a stage-cache miss (keys carry a corpus manifest), a
  resume after gleaning keeps the gleaned relationships, and a failed removal
  of stale artifacts blocks the commit (#134).
- The Neptune relationship pre-drop is scoped to the run's entity label, so
  one index suffix no longer deletes another's edges (#134).
- `processing.max_retries` now sets the ingestion LLM retry count; the
  stages passed it but the batch processor kept its own default. The default
  is 5, the value previously in effect (#140).
- The community auto-resolution sweep scores complete partitions. Leiden
  leaves isolated nodes out and modularity rejects a partial partition, so
  every candidate failed silently and `resolution` was always used.
  `graph.community_detection.auto_resolution` now defaults to `false`, the
  behaviour actually in effect until now: with the sweep working, global
  search lost 3 of 20 answers on the E2E corpus. Cached community-detection
  output is recomputed once (#141).
- A conversation turn (question and answer) is appended to memory as one
  unit, so concurrent turns on one conversation no longer interleave (#143).
- `astream`/`stream` bind a `query_id` (and `conversation_id`) to the
  retrieval logs like `ainvoke` (#144).
- `run-rag` no longer translates every query for a same-language corpus: it
  passed the configured target language explicitly, which disabled the
  chain's no-op translation skip (#145).
- Neptune writes fail fast on errors a retry cannot fix (malformed query,
  access denied, bad parameter); they used to be retried with backoff before
  they surfaced. Throttling, concurrent modification and connection loss are
  still retried (#151).
- The Gremlin connection pool is sized to at least
  `indexing.neptune.index_concurrency`. It used `aws.neptune.pool_size` alone,
  so a higher write concurrency queued batches on too few connections (#151).
- The batch processor marks an item that failed every attempt with
  `BATCH_ITEM_FAILED` instead of `{}`, so stages no longer confuse a failure
  with an empty LLM result. Graph extraction output without a well-formed
  `entities` section counts toward `total_extraction_failures` instead of
  reading as an empty success, and an output whose empty `relationships`
  section was dropped by the XML parser keeps its entities (#152).
- Log lines from worker threads (batch processor chunks and call timeouts,
  indexing, embedding, Neptune batches, file loading, resolution) keep the
  bound `pipeline_id`/`stage`/`query_id`: the thread pools are
  `ContextThreadPoolExecutor`s, which run each task in a copy of the
  submitter's `contextvars`. `pipeline_id` is now bound for the whole
  `run-ingestion` run, including the S3 cache sync and the failure report
  (#156).
- The LangChain `partial_correctness` judge is scored only from the JSON
  `score` its prompt asks for: LangChain's CORRECT/INCORRECT word heuristic
  (which scored `"... is correct"}` as 1.0) and the free-text regex fallbacks
  (`"Score: 8/10"` → 1.0) are gone. A judge reply without a valid score in
  [0, 1] (either metric) is recorded under `failed_metrics` instead of as 0.0
  or a clamped value (#PR).
- `run-eval` no longer exits 0 when scoring fails: with
  `processing.ignore_errors: false` (the default) an evaluator error stops the
  run instead of being recorded per query, and `--max-failure-rate` now also
  applies to each metric's failed share of attempted values, so a metric that
  failed on every query exits non-zero (#PR).
- `run-ingestion --verify-metadata` exits non-zero when the metadata is
  corrupt, and `--repair-metadata` when the repair fails or raises (#PR).
- `run-rag` reports `success: false` and exits non-zero when the chain returns
  its `ignore_errors` error fallback (`metadata.error`) instead of an answer
  (#PR).
- Importing the package no longer configures logging: it used to replace the
  host's root handlers, set the root level to INFO and open a `FileHandler` in
  `logs/` next to the installed package (failing on a read-only
  `site-packages`). Only the CLIs call `setup_logging`, and a relative
  `logging.log_file_path` is now resolved against the working directory
  (#PR).
- Visualization: `graph.visualization.embeddings.bedrock_model_id` given as a
  YAML string is validated as an embedding model id at config load (it used to
  fail with `'str' object has no attribute 'value'`, which ingestion logged
  and skipped, so no visualization was written); an edgeless graph no longer
  gets NaN node sizes in the interactive view; a failed dimensionality
  reduction falls back to a seeded spring layout and sets `layout_degraded`
  instead of returning an unseeded random layout (#PR).
- The evaluation `run_manifest` records `git_sha` only when the package runs
  from a checkout that tracks it (an install inside another repository used to
  report that repository's HEAD) and adds `git_dirty` (#PR).
- `run-eval` output files carry a random suffix after the timestamp, so runs
  started in the same second no longer overwrite each other, and a numeric
  `answer` of `0` is kept as a ground truth instead of being treated as
  missing (#PR).
- `run-prompt-tuning` fails instead of emitting a default profile when the
  profiling model returns no JSON profile, splits an `entity_types` string
  (`"PERSON, ORGANIZATION"`) into types instead of characters, and builds the
  example `GraphExtractor` from the tuner's providers (#PR).

### Security
- The CLIs log a WARNING at startup when `LANGSMITH_TRACING` or
  `LANGCHAIN_TRACING_V2` enables LangSmith tracing, which uploads prompts,
  retrieved context and model outputs. Tracing is not turned off (#156).
- Require patched `unstructured>=0.24.0` for optional Markdown/HTML parsing on
  Python 3.11+ (GHSA-4mvj-m6j5-pmf7), which also drops NLTK and its model
  artifact path traversal (GHSA-8mgp-746c-j5xp); Python 3.10 keeps the core
  formats without this extra.
- Constrain the transitive `langchain-openai` to `>=1.1.14`
  (GHSA-r7w7-9xr2-qq2r).
