# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial public release preparation: an AWS-native, open-source reference
framework that unifies the GraphRAG and LightRAG retrieval methodologies on
Amazon Bedrock, Neptune, OpenSearch, and DynamoDB, with an optional local mode
that runs without an AWS account. This section summarizes the changes since
the first public snapshot by user impact; the linked pull requests hold the
details.

### Highlights
- Two selectable retrieval methodologies on one ingestion, indexing, caching
  and hybrid-search stack: GraphRAG community summaries
  (`auto`/`drift`/`global`/`local`/`simple`) and LightRAG dual-level keywords
  (`mix`/`hybrid`/`naive`), with triple-hybrid (lexical + semantic + graph)
  retrieval, RRF fusion, reranking and multilingual ingestion.
- Local mode: `backend: local` (or `--backend local`) runs ingestion, queries
  and evaluation on Ollama, a sentence-transformers cross-encoder and
  embedded SQLite graph and vector stores and doc-status registry, so Ollama
  is the only service to run; Gremlin Server and OpenSearch in containers
  remain an option (#209, #271-#273).
- Optional OpenTelemetry tracing of ingestion and queries, exported over OTLP
  or to Langfuse or Arize Phoenix (#207, #208).
- Incremental indexing that survives interruption: a write-ahead registry
  record, deletion scoped to the run's namespace and corpus, per-namespace
  `indexing.reset`, explicit retirement of moved source scopes and a retry
  limit for documents that keep failing (#175, #186-#199).
- Conversational follow-ups are rewritten into standalone questions before
  retrieval (#212).
- Claude Sonnet 5.5 is the default model and Claude Haiku 5.5 the fast tier
  (#117, #162, #170).
- Three LLM-free evaluators (`graph_aware`, `retrieval`, `answer_match`) next
  to the LangChain and RAGAS judges (#121).
- A CDK stack (`iac/`) with Well-Architected security defaults, alarms and a
  Step Functions ingestion pipeline (#157, #211).

### Breaking changes
Configuration. Removed or renamed keys are reported as unknown keys and
ignored, so check the start-up warnings after upgrading.
- `providers.llm.max_retries` / `providers.embedding.max_retries` (retries
  after the first try) are now `max_attempts` (total attempts, default 3)
  (#216). Every retry knob counts total attempts (#151).
- The renamed-key shim is gone (#216). Replace `search.llm_retry` with
  `aws.bedrock.transient_retry`, `aws.bedrock.effort` with
  `aws.bedrock.default_effort`, and `processing.max_retries`,
  `indexing.neptune.max_retries` and `evaluation.ragas_max_retries` with the
  matching `max_attempts` key (a former `max_retries: N` is
  `max_attempts: N+1`).
- The `"NONE"` alias of `aws.s3.encryption.encryption_type` is removed; use
  `BUCKET_DEFAULT` (#216).
- `graph.visualization.embedding_method: "node2vec"` is now `"description"`
  (the layout always embedded each node's description) (#210).
- The post-retrieval context-building call is removed with
  `search.context_building_model_id` and `custom_prompts.context_building_*`;
  the query rewrite takes over its use of the history (#212).
- Removed keys that no longer did anything: `indexing.cross_run_fuzzy_merge`
  (the incremental merge follows `processing.fuzzy_entity_resolution`),
  `graph.community_detection.report_generation.include_statistics` /
  `include_key_entities` (#214), the score-based gleaning thresholds under
  `processing.gleaning` (#171), `search.global_search.graph_timeout_seconds`
  (#161), `indexing.neptune.min_entity_importance` (#155) and the
  `SearchQuery.metadata["lightrag_mode"]` alias (#212).
- `SearchQuery.label_prefixes` is removed; no retriever read it (#260).
- `processing.chunking.content_type` is removed with the markdown/HTML
  header pre-splitting it selected: no parser builds a markdown or HTML body,
  so every document was always chunked as plain text. Stage caches and
  extraction fingerprints keep their keys (#289).
- `processing.chunk_concurrency` and `processing.batch_size` are removed
  with the chunking they drove: an LLM stage keeps
  `processing.max_concurrency` calls in flight and starts the next item as
  soon as one finishes. The default (20) now is the peak; it burst to 40
  before but averaged about 18 in flight, so set a higher value to go
  faster. RAGAS judges run `evaluation.ragas_max_workers` jobs without
  batching (#296).
- A filter key that no retriever of the strategy declares raises
  `InvalidFilterError` (#123).
- `indexing.opensearch.terms_batch_size`, `max_total_clauses` and
  `reserved_clauses` are removed with the terms-filter batching they drove.
  A list filter is one `terms` query whatever its length; one with more
  values than the index's `max_terms_count` (65,536 by default) raises
  `InvalidFilterError` (#258).
- IaC: `create_guardrail` decides whether the stack owns a guardrail; bring
  your own with `-c create_guardrail=false` (#103).
- `run-ingestion --repair-metadata` is removed. It loaded the metadata it
  was meant to repair, so on a corrupt file it failed like
  `--verify-metadata` and never repaired anything; use `--verify-metadata`
  and, for a corrupt run, `--force-rebuild` or a new `--pipeline-id` (#248).

Defaults. Set the old value explicitly to keep the previous behaviour.
- `graph.visualization.enabled` defaults to `false`; set it to `true` or
  render on demand with `run-visualization` from the `visualization_data.json`
  that ingestion still writes (`graph.visualization.export_data`, default
  `true`: graph and communities, no model call; `run-visualization` computes
  a spring layout and centrality) (#210, #219).
- `backend: local` sets `search.token_manager.max_context_tokens` to 22000
  when the LLM is not Bedrock and sets no `context_window_size`: what the
  default local model's 32,768-token window accepts beside its output
  reservation (30000 was clamped, with a WARNING, on every query) (#224, #226).
- `backend: local` uses the embedded SQLite graph and vector stores, not the
  Gremlin Server and OpenSearch containers; to keep the containers, or for a
  local index built before, see *Upgrading from pre-release builds* (#273).
- Entity resolution merges only names equal after normalization, legal-form
  removal and dense-script spacing. Similarity matching is opt-in
  (`processing.fuzzy_entity_resolution: true`), and
  `processing.similarity_threshold` defaults to 0.85 (was 0.6) (#214).
- A memory query with conversation history is rewritten as a standalone
  question (`search.query_rewrite_model_id`, fast tier) and every step reads
  the rewrite (#212).
- `aws.region_name` defaults to `us-west-2`, and `aws.bedrock.region_name`
  defaults to `null` (follow `aws.region_name`); set it to keep Bedrock in
  another region (#148).
- Default models: Claude Sonnet 5.5 (`default_model_id`, #117) and Claude
  Haiku 5.5 (`fast_model_id`; set `anthropic.claude-haiku-4-5-20251001-v1:0`
  to keep Haiku 4.5) (#170).
- The Bedrock guardrail applies to the query path only
  (`aws.bedrock.guardrail.apply_to: all` restores it) (#104).
- `graph.community_detection.auto_resolution` defaults to `false` (#141).
- Entity and relationship `rank` are computed from node degree (#214), and
  relationship weight sums the extracted strengths per text unit (#189).

Index data. Rebuild existing indexes once with `indexing.reset: true` (or a
full re-index of the affected index type).
- Entity names keep their display form and ids use a symbol-preserving key
  (#110); community, document and registry ids hash new inputs (#134).
- The chunk `text` field uses the source-language analyzer, and
  `additional_target_languages` translations are indexed in
  `translated_text_<language>` fields (#177).
- NFC normalization gives documents that contained decomposed (NFD) text a
  new id; the next incremental run re-indexes them once (#177).
- Claim ids are keyed on the entities the subject and object resolve to
  (with the type and dates), not on the names as extracted. Only indexes
  built with `processing.claim_extraction.enabled: true` are affected: until
  they are rebuilt, a claim a new document restates is stored beside the old
  one instead of merging into it. Run once with `indexing.reset: true` (and
  `--force-rebuild` or a new `--pipeline-id` when resuming a pipeline id
  whose `claim_resolution` stage output is cached) (#290).
- Most stage caches miss once after upgrading (their cache keys changed).
  A run resumed with an old `--pipeline-id` can still reuse stage outputs
  written by an earlier build; use `--force-rebuild` or a new
  `--pipeline-id` to apply the fixes to them (#167, #170, #171, #201, #214).

Library API (Python callers and custom backends).
- `EvaluationRunner(args, rag_chain, config)` takes the run's config (the
  one the chain was built from) instead of loading it again from
  `args.config_path` (#292).
- Custom search strategies: `BaseSearchStrategy._fuse_and_rerank` takes a
  required `rerank_trace=` (create it with `self._new_rerank_trace()` and
  pass it to `SearchResult.metadata["rerank"]`); a call without it raises
  `TypeError` instead of reporting the rerank as `skipped` (#284).
- Import paths (#216 unless noted): `HierarchicalCommunity` →
  `domain.models`; `LanguageModelInfo`, `EmbeddingModelInfo`,
  `RerankModelInfo`, `effective_max_output_tokens` →
  `ports.model_factory`; `adapters.aws.chain_factory` →
  `adapters.llm.chain_factory`; `estimate_token_count` →
  `adapters.llm.token_estimate`; backoff helpers → `adapters.llm.retry`
  (take `is_transient=`); `merge_descriptions` →
  `domain.ingestion.descriptions`; evaluator bases →
  `adapters.evaluators.base` (no package re-exports); LangChain-coupled and
  console helpers → `shared.utils.langchain` / `document_converter` /
  `display` (#164); `visualization/embeddings/node2vec.py` →
  `node_embedder.py` (#210). `adapters.aws` no longer re-exports its
  adapters, and `ports.CachePort` is removed (use
  `shared.cache_manager.CacheManager`).
- One injection path: `doc_status=`, `vector_indexer=`, `graph_indexer=`,
  `model_factory=` and `retriever_builders=` cannot be combined with
  `providers=` (set them on the `Providers` bundle); `IndexingStage` takes
  its indexers from the bundle (#216).
- The AWS session is created lazily. Only `Providers`, `build_providers`,
  `DataIngestionPipeline` and `GraphRAGChain` take `boto_session`; the
  ingestion stages, search strategies, evaluators, `HybridScorer` (now
  `HybridScorer(config, rerank_factory=None, *, providers=None)`),
  `TokenManager`, `PromptTuner` and the visualization manager no longer do
  (read `providers.boto_session`), and `Providers.resolve` takes no session.
  `GraphRAGChain` no longer accepts unknown keyword arguments, so a
  misspelled one raises `TypeError`. `OpenSearchIndexer`,
  `OpenSearchRetriever` and `BedrockNodeEmbedder` require
  `embedding_factory=` (#216, #221).
- Custom model factories: `get_model_info` must return a
  `LanguageModelInfo`; opt into the chain-level retry with
  `is_transient_error(exc)` and `transient_retry` (#216).
- Custom strategies: `asearch(query, config=None)` must accept `config`
  (#156). Subclasses that overrode `GraphRAGChain._build_graph_retriever` /
  `_build_document_retriever` inject `retriever_builders` instead (#212).
- Custom doc-status stores implement every `DocStatusPort` method
  (`get_many`, `put_many`, `record_fits`, lineage overflow) (#214), and
  `diff` takes `extraction_fingerprint` and stores
  `DocStatusRecord.extraction_fingerprint`; a store whose `diff` does not
  take it is diffed by content only, with a WARNING (#224).
- Retrieval errors are classified by type only; the message markers are gone
  (#261). A custom retriever or backend whose error must fail the query
  rather than degrade it to no results raises a `ConnectionError` or
  `ConfigurationError` subclass, such as the typed store errors in
  `shared/exceptions.py` (`StoreConnectionError`, `StoreTimeoutError`,
  `StoreConfigurationError`, `StoreAccessDeniedError`,
  `StoreRequestRejectedError`). An `AWSServiceError("access denied ...")` or
  `RuntimeError("AccessDeniedException ...")` now degrades the section.
- Custom indexers: full-build `index_*` methods take a keyword-only
  `suffixes`; `get_stats` / `get_entity_count` are gone (#169, #214);
  `IndexingManager.index_delta` no longer merges with the stored graph
  (`merge_with_existing_graph`), and `read_entities` /
  `read_relationships` take the index `suffix` (#163).
- Ingestion and storage call shapes:
  - `DirectoryLoader` takes only `source_directory`. `config`, `recursive`,
    `supported_extensions`, `deduplicate`, `similarity_threshold`,
    `minhash_permutations`, `n_grams_size`, `compute_dir_hash` and
    `parse_files` are removed (#299), and `exclude_patterns`, `max_workers`
    and `DEFAULT_EXCLUDE_PATTERNS` with them, so a call such as
    `DirectoryLoader(path, Config())` raises `TypeError` at construction
    instead of failing later in discovery (#PR).
  - `GraphIndexer` / `VectorIndexer` `initialize()` / `clear()` and
    `IndexingManager.initialize()` / `clear_all_data()` return `None` and
    raise the store's typed error instead of returning `False`; a caller that
    tested the returned value now always sees `None` (#294).
  - `BatchProcessor` rejects unknown settings with a `ValidationError`:
    `batch_size=` and `chunk_concurrency=` were silently ignored after #296
    (#PR).
  - `compute_hash(algorithm=)` raises `TypeError`; it always hashes with
    SHA-256 (#300).
- Removed APIs: `IncrementalIndexer.plan` / `prune_changed` /
  `remove_deleted` (use `detect_delta` and `remove_changed_and_deleted`)
  (#214); the LangChain-memory wrapper, `MemoryManager.get_langchain_memory`
  / `add_message` (use `load_history` / `add_turn`) (#212); conversation
  entity tracking: `MemoryManager.load_context` (now `load_history`),
  `add_turn(entities=)`, `MemoryManager(providers=)` and the history's entity
  extractor (#218); the
  `Document` page-element model and `Document.detected_language` (#216);
  `OptimizedContext.quality_score` and other unused parameters and
  test-only methods (#169); `DocStatus.PARSING` / `PROCESSING`,
  `DocStatusRecord.content_summary` / `content_length` and
  `application.cli.run_evaluation.exceeds_failure_budget` (#216);
  `domain.retrieval.strategy_registry.registered_strategies` (#212);
  `NeptuneClient.delete_vertices_in_batches` (the indexer's `clear` deletes
  a label in retried batches itself) (#263); `NeptuneClient.connection`
  (use `NeptuneClient.g`) (#266); and `BaseParser.parse_files`,
  `ParsingStats` and `GraphAnalyzer.centrality_cache`, which nothing used
  (#267).
- `BaseSearchStrategy.search` (a sync wrapper with no caller; use
  `asearch`) and the unused `timeout` parameter of
  `BaseSearchStrategy._safe_aretrieve` are removed (#249).
- `NeptuneRetriever` expands only from seed entity ids passed as an `id`
  filter, which is how every strategy calls it. The name-matching seed
  lookup (used without an `id` filter; it lowercased query terms for a
  case-sensitive match) and the community-seeded traversal (no strategy
  requested it) are removed, with `NeptuneRetriever.DEFAULT_MAX_HOPS`
  (#249).
- `MetricsMixin` and `get_metrics()` on retrievers, search strategies,
  `HybridScorer` and `TokenManager` are removed. Nothing read the counters,
  concurrent queries overwrote each other's values, and the query spans
  already carry the same counts and timings (#253).
- `BatchProcessor(max_retries=)` is `max_attempts=` (#151);
  `execute_with_fallback` no longer takes `batch_func` (sync) or
  `run_config` (#174, #216); `create_robust_xml_output_parser` requires
  `output_tags` (#174).
- `unified_kg_rag.visualization` exports only `GraphVisualizationManager`;
  import the renderers from `unified_kg_rag.adapters.renderers` and the
  embedder and reducer from `unified_kg_rag.visualization.embeddings` (#246).
- `LocalSearchStrategy` no longer takes an `entity_focus_multiplier`
  argument; nothing passed it, so it was always 2, now the class constant
  `ENTITY_CANDIDATES_PER_FOCUS` (#247).
- `Document` keeps its text once, in `page_content`: `Document.content`,
  `pages`, `total_pages` and `file_type` and the `DocumentContent` and
  `Page` models are removed. They copied the text three times (a parsed
  document's parsing and loading cache entries and parsed-output export were
  three times its text) and nothing read the page fields. Exports and caches
  written before still load; their `content.text` is read as the text (#289).

### Upgrading from pre-release builds
- Doc-status registries written before source scopes existed are migrated in
  place: the first run of a scope re-keys each legacy record whose file is
  still in the corpus, without re-extracting it (#188). A record of a file
  removed before the upgrade is never adopted or deleted, so its artifacts
  stay: run once with `indexing.reset: true` to clear them (deleting the
  registry items alone leaves the artifacts) (#190, #197-#199).
- Records the direct API wrote without a scope under a caller's own `doc_id`
  survive every reset; remove them through the doc-status store
  (`list_all()`, then `delete_lineage_overflow` and `delete`) (#200).
- Hidden files and files under a hidden or `__pycache__` subdirectory of the
  corpus (any path component below the root starting with `.`, such as
  `docs/.drafts/`) are no longer parsed (#213). The first incremental run
  after upgrading classifies such files indexed by an earlier build as
  deleted and removes their artifacts and registry records. Rename them
  before upgrading to keep them (they are then indexed under their new
  paths).
- Records written before records carried an extraction fingerprint are
  compared by content only, so upgrading re-extracts nothing; run once with
  `indexing.reset: true` to bring them under the current settings. The
  SQLite registry gains the column on first open (#224).
- Roles with a custom IAM policy need `dynamodb:BatchGetItem` and
  `dynamodb:BatchWriteItem` on the registry table (#191, #194).
- A suffix indexed with the doc-status registry off by a release before
  #233 cannot be told apart from a tracked one (its indices carry no
  marker), so an incremental run takes it as tracked: run such a suffix once
  with `indexing.reset: true` (registry on) after upgrading (#243).
- A `backend: local` config of your own that relied on the preset for the
  stores now gets the embedded SQLite stores, empty, while its registry still
  lists the documents indexed into the containers, so its next incremental
  run stops before writing anything. Add `providers.graph.kind: gremlin` and
  `providers.vector.kind: opensearch` to keep the containers, or run once
  with `indexing.reset: true` to rebuild in the embedded stores (#273).

### Added
- With `observability.capture_content`, the root `kg_rag.query` span records
  the redacted query as `input.value` and, outside `stream` / `astream`, the
  redacted answer as `output.value`, so the Langfuse and Phoenix trace lists
  show each query trace's input and output instead of leaving them empty
  (#292).
- `providers:` config section selecting each role's backend (`llm`,
  `embedding`, `rerank`, `graph`, `vector`, `doc_status`) through a provider
  registry (`@register_provider`); a `[local]` extra; `SQLiteDocStatusStore`;
  an Ollama service in `docker/compose.local.yaml`; and
  `providers.doc_status.enabled` (#154, #209).
- Embedded stores, no server needed: `providers.vector.kind: sqlite`, a
  vector and lexical store in one SQLite file (`providers.vector.sqlite_path`)
  that answers the OpenSearch queries the unchanged indexer and retriever
  build (exact kNN, Lucene BM25, CJK bigram analyzers, neural-search hybrid
  fusion) for about 100,000 documents per index; and
  `providers.graph.kind: sqlite` (`SQLiteGraphIndexer`,
  `SQLiteGraphRetriever`), Neptune's write semantics and graph expansion in
  `providers.graph.sqlite_path`. Single host, one writer at a time, local
  disks only. `snowballstemmer` joins the `[local]` extra for the `english`
  analyzer (#271, #272).
- `observability:` section (off by default, `otel` extra): spans for runs,
  stages, batches and each query step, GenAI model-call spans with token
  usage, log/trace correlation, `session.id` / `user.id`
  (`RAGInput.user_id`), opt-in content capture with `redact_patterns`
  applied at capture and again at export, and `export_timeout_seconds`
  (#207, #208, #211, #219).
- Incremental-indexing controls: `indexing.max_document_failures` (#175),
  `indexing.retire_source_scopes` / `--retire-source-scope` (#192) and
  `processing.document_parsing.source_scope` (#134).
- Model catalog: Claude Sonnet/Opus 5.5, Opus 4.6-4.8, Sonnet 4.6, Haiku 5.5
  and OpenAI GPT models on Bedrock; free-form model ids with
  `aws.bedrock.model_overrides` (#117, #127, #162).
- Reasoning effort per tier (`aws.bedrock.default_effort`, `fast_effort`)
  and for ingestion (`ingestion_effort`) and judges
  (`evaluation.judge_effort`) (#113, #128, #179).
- Evaluation: `retrieval` and `answer_match` evaluators, grouped statistics
  and a run manifest, RAGAS timeout and concurrency knobs, and `s3://`
  dataset and output paths for `run-eval` (#113, #121, #206).
- `run-prompt-tuning` reads every format ingestion parses (#121), and each
  `run-*` `--help` ends with an example and a user-guide link.
- Operability: per-key warnings for unknown config keys, bound run/stage/
  query ids in logs, extraction-failure metrics and alarms, endpoint checks
  at start-up, and runbook sections on backup and restore and OpenSearch
  replicas (#122, #178, #211).
- CDK context keys `opensearch_master_instance`, `flow_log_retention_days`
  (#157), `sfn_timeout_hours`, `opensearch_volume_gib` and
  `guardrail_version` (#211).
- CDK context key `enable_cloudwatch_tracing` (off by default): an ADOT
  collector sidecar sends the app's traces to CloudWatch (X-Ray OTLP
  endpoint, Transaction Search), with X-Ray write actions on the task role,
  an ECR pull-through cache for the collector image and, in private mode, an
  `xray` VPC endpoint; `GRAPHRAG_OBSERVABILITY_ENABLED` /
  `GRAPHRAG_OBSERVABILITY_BACKEND` overrides; and a runbook recipe with
  Bedrock model invocation logging as the account-level backstop (#269).
- Per-query rerank status: `metadata.rerank` on every search result and RAG
  output (`applied`, `disabled`, `unavailable`, `skipped` or `failed`, per
  rerank stage plus a worst-case summary), a `Rerank:` line in `run-rag
  --verbose`, `rerank_status` on each evaluation result and
  `rerank_status_counts` in the summary, and `kg_rag.rerank_status` on the
  fusion and strategy spans. A rerank model that could not be built is now
  logged at ERROR instead of WARNING. Opt-in `search.reranking.required`
  fails the query with `RerankModelError` instead of answering from the
  fused order. Reported by @vibhupb (#83) (#284).

### Changed
- `BatchProcessor` runs one pool of `max_concurrency` items instead of
  chunks of `batch_size` items in parallel, each waiting for its slowest
  item: a slow item holds only its own slot and is retried in it, and the
  calls in flight never exceed `max_concurrency` (the defaults burst to 40,
  averaged about 18, and `max_concurrency` above `batch_size` changed
  nothing). The async path runs each item as its own call the same way, so
  evaluation answers no longer wait chunk by chunk. A run's trace has no
  `kg_rag.batch.chunk` spans. Each Bedrock client's connection pool is
  sized to the larger of `processing.max_concurrency` and
  `processing.io_workers` (64 by default; it was 80) (#296).
- `run-eval` generates up to `processing.max_concurrency` (20 by default)
  answers at once instead of sequential chunks of 10 queries. Lower
  `processing.max_concurrency` for an evaluation run that hits Bedrock or
  store throttling (#301).
- `BatchProcessor.aexecute_with_fallback` takes no `batch_func`: each item
  is one awaited `sequential_func` call (`Runnable.ainvoke`), as on the sync
  path. A one-input `abatch(..., return_exceptions=True)` first call was the
  same `ainvoke` with the error returned in place; a caller still passing
  `batch_func=` gets a `TypeError` (#301).
- CI installs uv 0.12.23 in every workflow instead of whatever release
  `setup-uv` resolves on the day, so a uv release cannot change lock
  handling under an unchanged commit (#297).
- A plain `uv run pytest` no longer selects the real-AWS (`aws`-marked)
  tests: `-m "not aws"` is in pytest's `addopts`. Those tests keep the real
  environment, so with store endpoints exported a bare run used to write to
  the real stores. Pass `-m aws` to run them; any command-line `-m` replaces
  the default (#297).
- CDK alarms: `OpenSearchJvmPressure` follows AWS's recommended OpenSearch
  alarm (maximum at least 95% in 3 consecutive 1-minute periods; it was 85%
  in one 5-minute period), and `NeptuneHighCpu` fires only after 90% CPU in
  3 consecutive 5-minute periods, so normal bulk-indexing peaks no longer
  page (#298).
- Ingestion logs one start and one completion line per stage (with its input
  and output counts); the per-stage `=====` STARTED/COMPLETED banners are
  gone, and the per-stage counts stay in the stage metrics. `DirectoryLoader`
  only reads pre-parsed `Document` JSON for the loading stage: its unused
  `config`, `supported_extensions`, `parse_files`, `recursive`,
  `compute_dir_hash`, `deduplicate`, `similarity_threshold`,
  `minhash_permutations` and `n_grams_size` arguments are removed, and the
  user guide now states the fixed near-duplicate rule of
  `processing.deduplicate` (MinHash Jaccard >= 0.9 over character 3-grams)
  (#299).
- Context packing ranks sections by their retrieval score alone; the fixed
  per-type priority weights (text 1.3 ... general 0.8) are gone. The
  per-type shares (`search.token_manager.type_budgets`) already decided the
  mix, so selection is unchanged except when the window is too small for any
  type's share: the one section seated then is the highest-scoring one (a
  tie goes to the earlier result) rather than a lower-scored text section
  (#287).
- `TokenManager` counts many texts concurrently only when its token counter
  reports `counts_via_api` (the Bedrock counter turns it off once a model
  rejects CountTokens); a custom counter without the attribute keeps the
  answer model's `supports_count_tokens` (#287).
- `visualization_data.json` edges no longer repeat every relationship they
  merge under `attributes.relationships` (about half of the edge bytes); no
  renderer read it, and the edge keeps its own type, weight and description
  (#292).
- `run-eval` `abstention_statistics.unanswerable` has the shape of the
  overall block (`abstained`, `answered`, `abstention_rate`); the
  `correct_abstentions` and `accuracy` keys are gone. Only the fixed
  no-context reply counts as an abstention, and the chain returns it only
  when retrieval finds nothing, so on a populated index `accuracy` stayed
  near 0 even when the model declined correctly; the rate is now documented
  as a lower bound on correct declines (#292).
- The vector indexer no longer keeps every vector it embedded in memory for
  the rest of the run: memory grew with the corpus and the copy was not
  reused (identical texts of one embedding call are still embedded once).
  `indexing.opensearch.persist_embedding_cache` and its S3 cache are
  unchanged (#294).
- `indexing.reset` no longer waits a fixed second after clearing the
  OpenSearch indices: the deletes are acknowledged before they return and
  the rebuild writes new timestamped indices (#294).
- `backend: local` selects the embedded SQLite graph and vector stores
  (`.unified-kg-rag/graph.sqlite3` and `vector.sqlite3` by default), so
  `--backend local` with a native Ollama needs no container.
  `docker/config.local.yaml` selects `gremlin` and `opensearch` explicitly
  and keeps the compose flow as it was. `run-visualization` takes the
  provider flags of the other CLIs (#273).
- A custom vector indexer's `read_claims` no longer has to tag each claim
  with the index it was read from; `IndexingManager` tags the claims it
  writes back after a deletion (#259).
- One policy decides which retrieval failures fail a query
  (`adapters/store_errors.py`). The OpenSearch and Neptune clients raise
  typed errors: `StoreTimeoutError` (a `StoreConnectionError`) for a request
  without an answer, `StoreAccessDeniedError` for HTTP 401/403 or a Neptune
  `AccessDeniedException`, `StoreRequestRejectedError` for an HTTP 400, and
  `StoreConfigurationError` (their base; also an `AWSServiceError` and a
  `ConfigurationError`) for a missing endpoint, credentials or auth method
  (#261).
- `search.fusion.fusion_weights` defaults to `{}`: every bucket weighs 1.0,
  as the 15 listed 1.0 entries did, and weighted fusion no longer warns
  about a bucket without a weight on every query (#258).
- The `graph_analysis` stage no longer computes degree, betweenness and
  PageRank centrality, which no later stage read (the visualization
  computes its own) and which took minutes on a graph of tens of thousands
  of entities; its `centrality_stats` metrics are gone. It also no longer
  caches a copy of graph resolution's entities and relationships (#248).
- CDK: the OpenSearch domain publishes only its application logs. The slow
  search and index logs stayed empty (the app sets no slow-log thresholds),
  so their two log groups are removed; the domain is updated in place
  (#251).
- CDK: in `network_mode=private` (the default) the created VPC no longer has
  public subnets, an internet gateway or `0.0.0.0/0` routes, which the
  isolated data plane never used. The public range stays reserved, so the
  `app` subnets keep their CIDRs and an update of a deployed stack removes
  only those resources (#251).
- Retrieval: AUTO routes among `local`/`mix`/`global`/`drift` with the fast
  model; `mix`/`hybrid` retrievals, DRIFT steps and keyword extraction run
  concurrently; global search answers from ranked map points without a
  reduce call and reranks once; Neptune expansion is opt-in for LightRAG;
  the context budget defaults to 30,000 tokens (#125, #132, #212).
- Ingestion: smaller chunk defaults that fit embedding and rerank limits
  (#125); descriptions are written in the target language (#179); gleaning
  stops per unit once it adds nothing new (#171); output-token floors follow
  the configured limits (#179).
- Conversation memory is process-wide by default;
  `memory.max_conversation_age_hours` is enforced (#176, #212).
- A doc-status registry that cannot be read fails the loading stage
  (`DocStatusRegistryError`) instead of silently rebuilding (#186). A
  relative `providers.doc_status.sqlite_path` resolves against the config
  file, and an incremental run whose registry lists documents the graph no
  longer holds fails and asks for `indexing.reset: true` (#214).
- Querying a suffix nothing was ingested under raises `IndexNotFoundError`
  (#178).
- `config-template.yaml` shows the `providers` section, the Neptune and
  OpenSearch connection keys and the embedding and rerank model ids as
  comments, so a copy of it switches with `backend: local` alone.
- Performance: shared MinHash permutations, float32 embedding cache,
  reused query chains, batched registry reads and writes, and in-process
  claim resolution and gleaning preparation (#176, #191, #194, #213).
- The CloudWatch EMF sink publishes dimensionless aggregate series only
  (#158). Development tools live in the `dev` dependency group (#119).
- `graspologic` is no longer a dependency: Leiden runs on
  `graspologic-native` directly (#225), so the install drops graspologic and
  the packages only it pulled in (gensim, matplotlib, seaborn, statsmodels,
  POT, hyppo and others), and the license check no longer needs an LGPL
  exception for gensim. `botocore` and `langsmith`, imported directly but
  previously only transitive, are now declared (#226).
- Dependencies: pydantic 2.14, langchain-core 1.6.10, langchain-aws 1.8.1,
  langchain-openai 1.7, langsmith 0.14, datasets 5.1, tiktoken 0.14,
  pypdf 6.20, tenacity 9.2 and boto3 1.43.112 in `uv.lock`; aws-cdk-lib and
  aws-cdk.aws-neptune-alpha 2.273 in the CDK lock. typeguard stays at 2.13.3
  and aws-cdk-cloud-assembly-schema below 55, the versions jsii and
  aws-cdk-lib pin (#241, #251).
- `iac/` is a uv project of its own (`iac/pyproject.toml`, `iac/uv.lock`,
  outside the root lock): install it with `cd iac && uv sync`.
  `iac/requirements.txt` and `iac/requirements-lock.txt` are gone. Dependabot
  updates the CDK lock through `uv lock`, so a transitive bump the pinned
  aws-cdk-lib cannot admit no longer opens a pull request that fails to
  install (#251).
- `docker/Dockerfile` takes the base image from a `BASE_IMAGE` build
  argument whose default is the same pinned public.ecr.aws ref, so a plain
  `docker build` is unchanged. When public.ecr.aws rate-limits the CI
  `docker` job (429), the job pulls the same digest from Docker Hub instead
  of retrying a quota that does not recover within minutes (#257).
- Local mode: `docker/compose.local.yaml`'s Gremlin Server indexes the
  `id` property of vertices and edges at start-up. Every graph write looks
  its vertex or edge up by `id` first, which scanned the whole in-memory
  graph: a full build of 10,000 entities and their communities takes about
  20 s instead of 60 s, and 5,000 relationship writes about 5 s instead of
  28 s. Restart the stack (`down` and `up`) to pick it up (#265).
- Graph expansion with `entity_importance_source: none` scores every node
  at the neutral importance 0.5 without reading an `importance` vertex
  property: entity vertices never store one, so the result is unchanged
  (#300).

### Fixed
- `processing.max_attempts` (and every `BatchProcessor`'s `max_attempts`)
  counts the first call, as documented: an item that keeps failing with a
  retryable error is called `max_attempts` times. The first call used to
  come on top of `max_attempts` retries, so default ingestion made 6 calls
  per failing item (now at most 5), and `max_attempts: 1` still retried
  once: the global-search map step and `run-eval` answer generation, which
  leave retries to the query chain, ran each failed call twice (#301).
- IaC: the `NeptuneHighCpu` and `NeptuneLowFreeableMemory` alarms watch
  the writer instance (`Role=WRITER`), and the memory alarm uses the
  window's minimum. They averaged the whole cluster, so with
  `neptune_instances=2` (the non-dev default) an idle reader halved a
  saturated writer's CPU and hid a writer low on memory.
  `NeptuneGremlinErrors` stays cluster-wide. The IaC README states each
  alarm's window (#PR).
- Configuration errors raise `ConfigurationError` (still a `ValueError`)
  instead of a bare `ValueError`: an unknown embedding model or unsupported
  `indexing.opensearch.embedding_dimension`, a missing S3 `bucket_name`, a
  retriever role with no provider and an invalid index suffix. The write
  path checks the suffix with the same `validate_safe_name` as the query
  path, so a suffix with a trailing newline (`"tenant\n"`) is rejected
  rather than reaching index names and graph labels. An embedding model
  that reports no dimension no longer silently gets a 1024-dimension index:
  it uses `indexing.opensearch.embedding_dimension`, and without one the run
  fails with a message naming that key (#297).
- `indexing.neptune.retry_delay_seconds: 0` is accepted with the default
  `max_attempts` and retries at once, as its description says; a validator
  rejected it whenever `max_attempts` was above 1 (#297).
- `processing.ignore_errors` is documented for what it does: it decides
  whether a whole step that fails (a batch call, a translation language, an
  evaluator, global map/reduce, the DRIFT primer, the auto router) raises or
  degrades. Items whose own LLM call fails are always skipped and reported;
  the old "skip items whose LLM step fails" wording described that as the
  flag's job (#297).
- A NaN, infinite or negative relationship strength from the model no longer
  kills ingestion at community detection. Leiden panics on a NaN weight with
  an exception no error handler catches (and rejects infinite ones), and a
  resumed run crashed again on the same graph. Strengths are now parsed as
  finite numbers clamped to `>= 0` (NaN and infinity become the default 1.0),
  and community detection replaces such weights in a graph read back from an
  older store or cache (#288).
- Community detection keeps its coarser levels. The small-community merge
  also ran on each level's cluster graph, whose nodes are whole communities,
  so with the default `min_community_size: 3` a level of two-community groups
  was folded away and the hierarchy often stopped at level 0. The merge now
  applies to entity communities only. Multi-level corpora get more (level 1+)
  communities and reports. Cache keys are unchanged, so a resumed run reuses
  the old communities: re-run with `--force-rebuild` (and `indexing.reset:
  true` where incremental runs keep the stored communities) to restore the
  lost levels (#288).
- The small-community merge moves a community into the neighbor it shares
  the most edge weight with, not the one it has the most edges to (#288).
- Intelligent chunking: an error while reading the model's chunk
  boundaries is logged at WARNING and the pre-chunk is split by size; it was
  logged at DEBUG and the pre-chunk silently kept whole (#289).
- Claims stated with different spellings of an entity ("Acme Inc" and
  "Acme") are one claim in incremental runs too. The stored id hashed the
  names as extracted, so an incremental run stored a second claim beside the
  first, and in a full build the id depended on which spelling came first.
  Claim merges (extraction, resolution and the cross-run merge) also keep
  the document `filters` of every instance in the claim's attributes, as
  entity and relationship merges do; claims are not filtered at query time,
  so retrieval does not change (#290).
- Gleaning corrections (`ENTITY_CORRECTION`, `RELATIONSHIP_CORRECTION`)
  resolve their target among the entities and relationships citing the unit
  being refined, the ones its prompt lists, instead of the whole round's
  graph: a unit could rename, retype or reverse an item only other units
  cite, which an incremental run without that unit cannot reproduce (#291).
- `entity_confidence_threshold` is applied to each chunk's own extraction,
  right after the grounding guard (as gleaning applies it), instead of to the
  entity merged across the batch: the best score over the chunks extracted
  together decided, so a full build and an incremental run kept different
  entities. A chunk that scores an entity below the threshold no longer
  cites it, and the chunk's relationships naming it are dropped; another
  chunk's low score no longer removes a stub. Only non-default thresholds
  are affected (#291).
- `graph.visualization.interactive` set without `physics_enabled` (for
  example only `max_nodes`) keeps physics off, the documented default; the
  renderer fell back to physics on whenever the key was missing (#292).
- The graph-expansion step's `kg_rag.retrieve` span label and failure log
  say `graph expansion` instead of `Neptune graph expansion`, which was wrong
  with the embedded SQLite or Gremlin Server graph store (#292).
- `run-prompt-tuning` profiles the corpus with
  `processing.graph_extraction.extraction_model_id`, the ingestion model the
  tuned prompt is for, instead of the query-time
  `search.entity_extraction_model_id` (#292).
- `run-prompt-tuning`: the corpus-grounded few-shot examples show entity and
  relationship types UPPERCASE, as the extraction rules they follow ask;
  entity types came out lower-cased (`<type>organization</type>`) (#292).
- Query translation: a refusal written with a typographic apostrophe
  ("I’m sorry, …") is caught like its ASCII form and the original query is
  searched; the follow-up rewrite guard and the translation guard now share
  one normalisation (#292).
- Embedded search engine: the analyzers no longer split an ASCII word at
  dense-script runs (it has none), with the same tokens. Analyzing English
  text, which the first lexical query of an index does for every document,
  takes about a third less time with the `english` analyzer and half with
  `standard` (#293).
- Embedded search engine: indexing and updating no longer pass each
  document's vectors through a JSON round trip (a vector given as a list is
  stored as given; other values are still converted to JSON types).
  Indexing 4,000 text units with 1024-dimension embeddings went from about
  5.7 s to 1.3 s (#293).
- Embedded search engine: an unfiltered kNN query scores the stored vectors
  in place instead of copying them for every query, and sorts only the
  documents that score at least the k-th best (ties still go by id, so the
  hits are the same). Over 50k vectors of 1024 dimensions the kNN step went
  from about 190 ms to 19 ms (#293).
- Embedded graph store: a vertex's edges are looked up through
  `(label, source_id, target_id)` and `(label, target_id, source_id)`
  indexes, and the neighbour and incident-edge reads query each endpoint
  separately, instead of walking every edge of the label. A graph expansion
  over 150k edges took about 10 s and now takes milliseconds. A database an
  earlier release wrote gets the new indexes, and loses the two it replaces,
  the first time it is opened (#293).
- The SQLite doc-status registry waits `search.store_timeout_seconds` for
  another process's write lock, as the embedded graph and vector stores do
  (it waited sqlite3's 5 s default), says the database stayed locked in its
  `DocStatusRegistryError`, and closes its connection when opening or
  upgrading the file fails. The three embedded stores now share one SQLite
  connection, transaction and error helper (`adapters/local/sqlite_util.py`),
  so all three recognise a lock by SQLite's error code (#293).
- Graph stores (Neptune, Gremlin Server, embedded SQLite): writing an
  entity or community vertex again replaces its stored properties, so a
  value that became empty (an entity's `type`, a community's `parent` once
  it is a root) or an attribute the item no longer has is dropped instead
  of keeping the previous run's value. The incremental merge reads every
  stored field back before it writes, so it keeps what it should (#294).
- Indexing stage: a store error while preparing the stores or clearing them
  for `indexing.reset` (an access-denied, unreachable or misconfigured
  OpenSearch, Neptune or SQLite store) now fails the stage with the store's
  typed error and reason, logged with its traceback, instead of a bare
  `Failed to initialize indexing pipeline` / `Failed to clear existing data`.
  The `initialize()` and `clear()` methods of the `GraphIndexer` and
  `VectorIndexer` ports return `None` and raise on failure; a custom backend
  that reported failure by returning `False` must raise instead (#294).
- The incident-edge expansion of `local` (bridge relationships) and LightRAG
  `mix`/`hybrid` kept arbitrary edges of a hub entity: its text-less fetch of
  the edges touching the retrieved entities returned up to
  `max_query_size` (100) per endpoint in index order, and the `(rank,
  weight)` order applied only after that cut, so a hub's best-connected edges
  could be dropped. A search without query text on an index with `rank` now
  asks OpenSearch to sort by `rank`, then `weight` (descending); the embedded
  engine supports that `sort` (#295).
- `simple` search and DRIFT's per-iteration document retrieval sweep several
  indexes and cut the merged hits to `top_k x retrieval_multiplier` by raw
  score, but scores of different indexes do not compare (hybrid normalizes
  each index's best hit to about 1.0, a vector search sums the entity and
  community indexes' two or three kNN scores, BM25 depends on each index), so
  one index could crowd out the rest before fusion and reranking. The sweep
  now takes each index's hits rank by rank. DRIFT's within-type ordering
  likewise no longer sorts graph entities (distance bands) together with
  entity-index hits (#295).
- Hybrid search on OpenSearch 2.13 (the version the CDK stack deploys) was
  observed to drop the `size`-th hit of a lexical sub-query that matches more
  than `size` documents and to min-max normalize over a lower match in its
  place, which skewed the fused scores of the index. The retriever now asks a
  hybrid search for one hit more than it returns (on both engines) (#295).
- LightRAG `hybrid` with no extracted keywords and a question too long for
  the raw-query fallback returned nothing without a word; it now logs a
  warning, as upstream does (#295).
- The `indexing.opensearch.build_relationship_vector_index` description said
  only LightRAG `mix`/`hybrid` use the relationship vector index; `local`
  (relationship section, bridge relationships), `simple` and `drift` query it
  too (#295).
- Operations guide: the CDK alarm section lists all eleven alarms, adding
  the seven store-health alarms (OpenSearch, DynamoDB, Neptune) with what
  each one means and what to do; it said "four alarms". An IaC test now
  fails when an alarm or dashboard widget names an EMF metric that is not a
  `PipelineMetrics` field, so renaming a field cannot silently blank an
  alarm (#298).
- `iac/README.md` cost drivers list CloudWatch Container Insights, which the
  ECS cluster always has on (billed custom metrics and performance log
  events), and why it stays on: it is the only per-task CPU and memory
  source for sizing the Fargate task (#298).
- CDK: a zone-aware OpenSearch domain that would span 2 AZs with an odd
  `opensearch_count` (for example `3` with `max_azs=2`) fails at synth with a
  message naming both keys; OpenSearch rejected it only at deploy time
  ("You must choose an even number of data nodes for a two Availability Zone
  deployment") (#298).
- CDK: `-c max_azs=1` is rejected when the context is read ("max_azs must be
  at least 2"); a one-AZ VPC never synthesized, because the Neptune cluster
  needs subnets in two AZs, and failed later with a less direct error (#298).
- CDK task role: `bedrock:CountTokens` is granted with `InvokeModel` on the
  same model ARNs, so models that accept CountTokens get exact token counts
  on the deployed stack instead of silently falling back to estimates after
  an `AccessDenied` (#298).
- `indexing.reset` with the doc-status registry enabled fails the indexing
  stage with `DocStatusRegistryError` when the registry cannot be cleared,
  instead of logging a warning and indexing on: the stale records would make
  the next incremental run skip the re-ingested documents as unchanged
  (#299).
- CloudWatch tracing (`enable_cloudwatch_tracing`): the task role keeps
  only `xray:PutTraceSegments`, the one action the X-Ray OTLP traces
  endpoint authorizes (a role with only `xray:PutSpans` gets a 403);
  `xray:PutSpans` and `xray:PutSpansForIndexing` are dropped. The collector
  exporter is renamed `otlphttp/traces` to `otlp_http/traces`, so ADOT
  v0.50.0 no longer logs the deprecated-alias warning. Verified on real
  AWS (#286).
- Tracing runbook: a `run-eval` run is one `kg_rag.batch` trace with its
  queries as children (not one trace per query); turning on Transaction
  Search creates the `aws/spans` and `/aws/application-signals/data` log
  groups and a 1% indexing rule, and turning it off removes neither; and
  `kg_rag.rerank_status` (per fusion step and per query) is documented next
  to the Rerank call's `kg_rag.rerank.status` (#286).
- Langfuse: the missing-host error now names the full OTLP endpoint
  (`<host>/api/public/otel`) for `OTEL_EXPORTER_OTLP_ENDPOINT`, since the host
  root alone returns 404; the runbook covers Langfuse v4 (its default
  `events_only` mode serves spans from `/api/public/v2/observations`), limits
  the ingestion-version header warning to Langfuse Cloud and says sessions
  need `conversation_id`. Verified against self-hosted Langfuse v4 and v3 (#285).
- A CDK deployment indexes Korean text: an Amazon OpenSearch Service domain
  has no `nori` analyzer unless its optional Nori package is associated, so
  with the default `language_analyzers` (`ko: nori`) a Korean corpus or target
  language failed every Korean index with `mapper_parsing_exception: analyzer
  [nori] has not been configured in mappings`. The compute stack now injects
  `OPENSEARCH_KOREAN_ANALYZER` (new context key `opensearch_korean_analyzer`,
  default `cjk`, OpenSearch's built-in bigram analyzer), which overrides
  `language_analyzers.ko`; set it to `nori` once the package is associated.
  Index creation that names an analyzer the cluster lacks now fails with an
  error naming the analyzer and the fix. The docs no longer claim nori is
  built into Amazon OpenSearch Service (#282).
- `aws`-marked tests (`pytest -m aws`) keep the real environment. The
  suite's environment isolation cleared `NEPTUNE_ENDPOINT`,
  `OPENSEARCH_ENDPOINT` and the ECS container-credential variables and set
  fake keys for them too, so the real-store conformance tests always skipped
  and would have called AWS with the fake keys (#283).
- Configurations that differ only in their OpenSearch index or Neptune
  label prefixes no longer share a doc-status registry namespace in one
  table (the CDK stack injects one fixed table name). The other store set's
  records counted as owners of the same content-derived ids, so a changed
  document's old text units and entities stayed searchable with no owner,
  and a run over a corpus the other set had indexed from the same directory
  found nothing to index. The namespace now carries a tag of the prefixes
  that differ from their defaults; default prefixes keep their namespace.
  A configuration with custom prefixes re-indexes its corpus once and leaves
  its old records in the table (see operations.md, Concurrent runs) (#283).
- The embedded search engine's `english` analyzer drops only Lucene's 33
  English stop words, as OpenSearch's `english` analyzer does. It used
  scikit-learn's 318, which include content words such as "amount", "bill",
  "interest", "system", "name" and "detail", so a query like "interest bill
  amount" found nothing lexically on an English corpus (#277).
- The first lexical query after a write to the embedded search engine is
  about 7x faster (63 s to 8 s at 50,000 documents of 120 words; 27x for
  the `english` analyzer alone): its analyzers split CJK runs with one
  regular expression instead of testing each character twice, and stem each
  distinct word once (#277).
- A filtered kNN query on the embedded search engine that keeps more than
  a tenth of the vectors scores them in place instead of copying the kept
  rows out first, and no query walks every vector id to find the kept ones:
  over 50,000 vectors of 1,024 dimensions (4 BLAS threads) 79 ms to 21 ms
  with 90% kept, 52 ms to 14 ms with 50%, 2 ms to 0.2 ms with 50 ids. A
  narrower filter, or an `l2` field, still scores a copy of the kept rows
  (#302).
- A `term`/`terms` filter on a keyword (or unmapped) field of the embedded
  search engine looks the values up in a per-field map built on the
  field's first filter after a write, instead of reading the field of
  every document in Python on every query: a 50-id filter over 50,000
  documents went from 35 ms to under 0.1 ms, and fetching the edges of 10
  entities from 150,000 relationships from 140 ms to 17 ms (#302).
- A kNN or hybrid query on the embedded search engine no longer returns no
  results when another process writes while it loads the index. The load
  read the generation, the documents and each vector field in separate
  statements, so a commit in between left a vector without a document; the
  `KeyError` that followed was read as a non-fatal failure and the query
  answered from nothing. The load now reads one snapshot, a vector without
  a document (a file changed outside the engine) raises
  `StoreConfigurationError`, and an index another process deleted leaves
  this process's cache (#277).
- The embedded search engine's analyzers keep a word whole across in-word
  punctuation as OpenSearch's standard tokenizer does ("don't", "u.s.a",
  "1,000.50", "2024.01.15"). They split there, so a query for an amount
  such as "1,000,000" matched every document containing "000" (#277).
- An incremental run fails before writing when the vector store lost the
  documents the doc-status registry lists (local mode: `vector.sqlite3`
  deleted or moved, or `providers.vector.sqlite_path` changed). Only the
  graph was probed, so the run called those documents unchanged and left a
  vector store holding just the new and changed ones, with no error. The
  probe reads a few of their text units (`VectorIndexer.read_text_unit_ids`;
  a backend that cannot answer skips it) (#278).
- Graph expansion on the embedded SQLite graph store reads one snapshot.
  Its neighbour and vertex reads ran as separate statements, so an
  ingestion deleting a vertex the walk had reached in between (an
  incremental run while queries are served) made the whole expansion come
  back empty (#278).
- Graph expansion with `entity_importance_source: degree` on the embedded
  SQLite graph store no longer fails on a SQLite build limited to 999 bound
  parameters once a query reaches more than 332 entities: the degree query
  bound its batch three times (#278).
- The embedded search engine's analyzers split words as OpenSearch's
  standard tokenizer does in every script. A word kept no combining mark
  (Python's `\w` has none), so Devanagari, Bengali, Tamil, Telugu, Thai,
  Myanmar and Khmer words, Arabic and Hebrew with vowel marks and
  decomposed Latin ("café" as "e" + U+0301) fell apart into single letters,
  and a query such as "अनुबंध" matched unrelated documents. Words now keep
  their marks and format characters; a Thai, Lao, Myanmar or Khmer run stays
  whole apart from other scripts (OpenSearch does not segment it either);
  in-word punctuation of other scripts and widths joins ("a：b", "١٬٠٠٠",
  "צה"ל"); and symbols ("x²", "½") are no part of a word. Emoji are still
  not indexed (OpenSearch keeps them) (#280).
- The embedded search engine's `english` analyzer stems as OpenSearch's
  (Lucene's `PorterStemmer`): it leaves words of one or two letters ("us",
  "os"; a lone "s" became an empty token), maps "-logy" to "-log" and
  "-bly" to "-ble" before the later steps ("apology" -> "apolog",
  "contractibly" -> "contract") and undoubles any final consonant after
  "-ed"/"-ing" ("trekking" -> "trek"). It now agrees with OpenSearch on all
  457,417 dictionary words and suffixed forms checked. The possessive "'s" comes off a token's end
  rather than out of the text ("pant-'s" keeps its "s") (#280).
- The embedded search engine's analyzers no longer index a lone mark,
  format character or connector after a Han, Hangul or Kana letter
  ("한국어" + U+200C gave a token U+200C; "日本" + soft hyphen + "語" lost
  the "本語" bigram), keep "١٢٣٫٤٥" and "ג׳ורג׳" whole, lowercase "İ" to "i"
  and "Σ" to "σ" as Lucene does (an "İstanbul" query missed "istanbul"), and
  end a word at in-word punctuation followed by a mark but no letter
  ("क:" + U+032A). Indexed text is analyzed when an index loads, so stored
  stores pick this up without re-indexing (#281).
- A search on the embedded search engine reads one snapshot of the file.
  It resolved the index, checked the result window, read the search
  pipeline and loaded each index in separate reads, so an index another
  process swapped out and deleted in between (a blue/green re-index while
  queries are served) failed the query with a bare `TypeError` or answered
  from nothing. It now searches the index it resolved, whole (#280).
- An in-memory embedded search engine (`":memory:"`) no longer answers
  from another, garbage-collected one's indices. Its cached indices were
  keyed by `id(engine)`, which Python reuses after collection; they are now
  keyed by a random id and freed when the engine closes or is collected
  (#280).
- A global-search map step whose every batch is over the model's context
  (langchain-openai's `OpenAIContextOverflowError`) answers from the
  unrated reports again. That error is an HTTP 400 too, so it was read as a
  refused request and failed the query; a level with at most
  `map_batch_size` reports is one batch (#268).
- A global-search map or reduce call that fails under
  `processing.ignore_errors` is recorded: the result metadata (and the
  `RAGOutput` metadata) carries `degraded_error` with the error's type name
  and the `kg_rag.strategy` span carries it as `error.type`. The degraded
  answer had no marker (#268).
- A Neptune write batch that fails but whose items the per-item fallback
  then indexes logs its final attempt as a WARNING, not an ERROR; an item
  that fails on its own still logs an ERROR (#268).
- A Neptune connection lost under concurrent queries no longer leaks
  threads. A query took the traversal source before waiting for a pool
  slot, so after another query reset the connection it sent on the closed
  one and blocked in gremlinpython's `pool.get()` for good; the timeout
  freed the slot but not the thread, and after a few outages the loop's
  default executor had no thread left: every strategy hung, and `aclose()`
  and the process exit with them. The traversal is now built in its worker
  from the current source, closing a connection wakes the requests waiting
  on its pool (they fail as a lost connection), and traversals run on
  threads the retriever owns (#266).
- A query embedding call that still fails after the client's retries (a
  5xx from a crashed Ollama runner or dead vLLM engine, a persistent Bedrock
  `InternalServerException`) fails the query with `EmbeddingModelError`.
  It was read as transient, so every vector section, and the lexical half
  of hybrid with it, came back empty and the query answered "could not find
  relevant information" with no error (#266).
- A global-search map batch over the context of a vLLM or llama.cpp model is
  left unrated, as on Bedrock and OpenAI: those servers report it as a plain
  HTTP 400, which failed the whole query (#266).
- A global-search map step whose every call gets an HTTP 400 (a parameter
  the server does not support, a model id it does not know) fails the query
  with that error. It left every batch unrated and answered from the raw
  reports; with `providers.llm.fast_model_id` set the answer call runs on
  another model, so nothing failed (#267).
- A query degraded under `processing.ignore_errors` records `error.type` on
  its `kg_rag.query` span, as a query that raised does; the span had the
  ERROR status only (#267).
- `run-eval --outputs-directory` is recorded as the run's
  `configuration.outputs_directory` in the evaluation summary, which named
  the config default instead (#267).
- A query whose Neptune write was in flight when another query dropped the
  connection fails as a lost connection instead of waiting forever.
  gremlinpython's `close()` shuts its executor down, so the read the write
  would have started was never scheduled; the retriever's worker thread was
  lost for good (after `pool_size` such races every graph expansion hung)
  and the process exit hung joining it. The indexer's requests, which have
  no timeout, are freed the same way (#267).
- Queries that waited on a failed Neptune connection attempt each raise a
  copy of its error; re-raising the one shared error grew its traceback by
  every waiter's frames and kept them in memory (#264).
- The text-units section is no longer empty for a corpus translated into
  five or more languages: its lexical search covers one field per
  language, and the 200-term budget made more clauses than OpenSearch
  allows (500 `too_many_nested_clauses`). The budget is now spread over the
  fields searched, and a query over the clause limit fails the query
  instead of degrading silently (#264).
- A short OpenSearch throttling burst no longer fails a query: opensearch-py
  retried a 429/502/503/504 back to back, so a burst of a few milliseconds
  used up `search.store_max_attempts`. Query requests now back off between
  attempts (0.25-0.5 s, then 0.5-1 s, doubling up to 2-4 s) and still fail
  once the attempts run out (#264).
- A query no longer hangs when gremlinpython's connection pool stays empty
  after a lost Neptune connection: a graph traversal that does not finish
  within twice `search.store_timeout_seconds` fails the query with
  `StoreTimeoutError` and the connection is replaced (#264).
- A local model server that refuses or cannot serve a query (an
  OpenAI-compatible or Ollama 401/403/404/400, a refused port, a timeout
  after the client's retries) fails the query instead of every strategy
  answering "could not find" while the log blamed OpenSearch (#264).
- A global-search map batch longer than the model's context is left unrated
  and the other batches still answer; Bedrock reports it as a
  `ValidationException`, which failed the whole query. Bedrock's
  `ModelTimeoutException` (HTTP 408) is retried as transient (#264).
- A full build no longer fails when an OpenSearch index create or delete
  loses its response (a reset connection, a 504 while the new index's shards
  were allocated). opensearch-py resends the request, and the resend of a
  create that had already taken effect was rejected with
  `resource_already_exists_exception`, failing the indexing stage; a resent
  delete got `index_not_found_exception`, so `indexing.reset` reported the
  clear as failed. Both now count as done (#265).
- `answer_match` reads Korean `아닙니다`, Chinese `没有` and English `Nope`
  as a "no" answer; such an answer had no polarity and scored 0.0 against a
  `no` gold. "No" joined to the next word by a non-breaking hyphen
  (`No‑cap clause`) is no longer read as the answer, as the ASCII hyphen
  already was not (#265).
- `langchain-aws` is required at 1.8.1 or later (was 1.7.0), the first
  release that reports a prompt longer than the model's context as
  `BedrockContextOverflowError`. On an older install Bedrock's
  `ValidationException` reached the retrievers unwrapped, so such a prompt
  still failed the whole query (#265).
- Ingestion reconnects after a Neptune restart or failover. Its write retries
  reused the pooled websockets the server had closed, which gremlinpython
  never reopens, so every write failed every attempt until the process
  exited; each attempt now rebuilds its traversal over a new connection
  (#261).
- A Neptune write retried after its response was lost no longer writes its
  data twice. Since the retry goes over a new connection (#261), a batch that
  was applied but whose answer was lost was applied again: a full build
  duplicated entity and community vertices (and the relationship edges of
  the duplicated entities), and any run duplicated `MemberOf` edges, while
  the indexing stage reported 0 failed. Every vertex write now creates or
  matches the vertex by id, and a `MemberOf` edge is added only for an
  entity not yet a member (#263).
- A sub-second Neptune connection loss no longer fails an ingestion run in
  its removals, merges or full-build clear. The indexer's reads (the merge
  and removal read-backs, the incident-edge lookup), `delete_by_id`, the
  label clear and the drops before relationship and membership writes now
  reconnect and retry like writes (`indexing.neptune.max_attempts`). Before,
  they had one attempt on the pooled connection, so in a long-lived process
  every later incremental run failed at its first read (#263).
- An OpenSearch bulk write survives a 429/502/503/504 burst longer than the
  client's own retries, which resend at once (six attempts within about
  150 ms). A bulk request still rejected so now resends its items with the
  bulk-item backoff, so a burst of a second or more no longer fails a
  write slice; when the last resend is rejected too, the error raised is a
  typed one (`StoreConnectionError`, `StoreRequestRejectedError`) with the
  status and the server's reason instead of "Streaming bulk operation
  failed." (#263).
- A query whose embedding call cannot succeed fails instead of answering
  "I could not find relevant information": an invalid or inaccessible
  Bedrock model id, a model without on-demand throughput in the region, or
  missing credentials. A Bedrock endpoint that times out after the model
  wrapper's retries fails the query too, as a store that does not answer
  does (#261).
- An OpenSearch cluster still answering 429/502/503/504 after the query
  client's `search.store_max_attempts` attempts (overloaded or restarting)
  fails the query with `StoreConnectionError`; every strategy answered from
  no context. An HTTP 500 for one search still degrades only that search
  (#261).
- Concurrent queries of an unreachable Neptune wait for one connection
  attempt instead of each making its own in turn: 16 queries took 16 times
  `search.store_timeout_seconds` and held as many executor threads (#261).
- Local search keeps the seed entities the entities index found, and the
  text units they cite, when graph expansion degrades; it answered from
  neither (#261).
- An error that names a field such as `author` no longer fails the query:
  retrieval errors are classified by type, not by matching words such as
  "auth" or "endpoint" in their message (#261).
- `run-visualization` rejects an unknown `--renderers` name while parsing
  its arguments (exit code 2), instead of after loading the data and
  computing centrality (#255).
- Used as a library without `setup_logging`, the package no longer raises
  `KeyError` on every log call when the host binds a field named like a
  `LogRecord` attribute (`name`, `process`, `msg`, ...) with
  `structlog.contextvars`; the field is passed with a trailing `_` (#255).
- An embedding-based visualization layout embeds only the nodes the
  interactive page draws (the top `graph.visualization.interactive.max_nodes`
  by degree), as the spring layout already did. Every node was embedded with
  Bedrock, though only those positions are used (#255).
- A three-node graph's visualization layout uses PCA instead of UMAP, whose
  spectral initialization fails with that few points; the layout came out
  `DEGRADED` (#255).
- The `answer_match` evaluator no longer reads emphasis or a quote after
  `no` or the numeral `네` as an answer: in `There is no **cap** on fees.
  Yes, ...`, `no "cap"`, ``no `cap` `` and `네 **가지**` the word starts a
  noun phrase, so a later `Yes` answers; since #259 the `**` or quote read
  as the answering punctuation. Other punctuation before a word still reads
  as an answer (`no (explicit) cap`, `no $500 cap`) (#262).
- The `answer_match` evaluator no longer reads `(예: …)` ("for example") as
  a yes, or the options in `Yes/No: No` and `(네/아니요): 아니요` as the
  answer. `no`, `네` and `예`, which also mean "no X", "four" and "e.g.",
  answer only when punctuation (other than `/`, or `:`/`)` after `예`) or
  the line end follows them, so `No: it does not.`, `**No**: ...` and
  `No (see Clause 5).` still answer; every other polarity word answers
  wherever it stands, and a word next to `/` is skipped (#255, #259).
- Description re-summarization sizes one call's input to the summary
  model's context window (half of what it leaves beside
  `max_summary_tokens`, 1,000 to 12,000 tokens). A fixed 12,000-token input
  overflowed a small local model's window; the default models keep the
  12,000-token budget (#256).
- An automatic resume of an incremental run (same `--pipeline-id`) whose
  extraction settings changed, such as a stage switched on since the last
  run or another extraction model, restarts at `document_loading`. It kept
  the previous run's delta and extraction fingerprint, so the changed stages
  ran only on that delta's documents and the next run with a new pipeline id
  re-extracted the whole corpus (#256).
- Rerunning a completed incremental pipeline id with the doc-status
  registry since turned off restarts at `document_loading` and builds the
  index in full. It restored the previous run's delta, so indexing wrote
  `PENDING`/`PROCESSED` records to the registry the user had disabled
  (#259).
- Deleting a document under a non-default index suffix
  (`processing.document_parsing.index_value`) strips its text units from the
  claims it shared with other documents in that suffix's claims index. The
  claims read back carried no suffix, so the update went to the default
  suffix's index: the claim kept the deleted text units, and a claim with the
  same id there was overwritten (#256).
- The answer context stays within `max_tokens` (and the context budget).
  Each section's `[source_id]` prefix and each section type's header were
  not counted, so an explicit 600-token budget rendered up to 690 tokens.
  Sections are now counted as rendered, and the headers of the types present
  come off the budget (#258).
- A long question no longer fails its lexical search. Lucene allows 1,024
  clauses per query and a `multi_match` makes one per term per field, so
  about 1,100 words (or 1,100 Han characters) failed with a misleading
  "Check the query filters" `ConfigurationError` and about 600 words
  degraded to no lexical hits. The lexical side now searches only the
  first terms of a long question (150 since #260, which counts a word or a
  Hangul, Han or Kana character as one term); the vector side still embeds
  the whole question (#258).
- A question of more than about 10 words no longer fails its lexical search
  on a large index. Typo tolerance (`fuzziness: AUTO`) expands every term
  into up to 50 index terms per field, so on an index of 100,000 distinct
  terms a 15-word question hit Lucene's clause limit: `naive` answered "could
  not find" and `simple`/`mix` lost their text units. Typos are now tolerated
  in questions of up to 10 words, with at most 10 expansions per term; a
  longer question is matched exactly. The lexical side counts a word, or a
  Hangul, Han or Kana character, as one term: the question gets the first
  150 and DRIFT's optional keywords the rest of 200. A single Korean name no
  longer cuts a whole English question to 150 characters (#260).
- A long list filter other than `id` no longer raises the number of hits.
  `filters={"type": [...3,000 values]}` asked every index for 3,000 hits and
  every kNN field for k=3,000; only the `id` list of a fetch by id sets the
  size now (#260).
- `indexing.opensearch.index_settings` is read in either spelling:
  `index.max_terms_count` and `index.max_result_window` fell back to the
  defaults (65,536 and 10,000) (#260).
- A query whose filters list many ids sends one search per index. Long list
  filters were split into batches and every combination of the batches
  searched in turn: two 3,000-id filters took 400 sequential requests, three
  1,000-id filters 1,000, and the hybrid pipeline normalized each batch's
  scores on its own, so hits of different batches were ranked against
  each other on incomparable scores (#258).
- A query whose OpenSearch search is denied (HTTP 401/403, such as a
  fine-grained access control role without `indices:data/read/search`)
  fails with `ConfigurationError`. The 403 body names no permission keyword,
  so the query answered "could not find relevant information" and reported
  success. A Neptune that refuses the websocket (HTTP 401/403) fails the same
  way (#258).
- Queries no longer hang on a store that stops answering. The query-side
  OpenSearch client waits `search.store_timeout_seconds` (30) per request in
  `search.store_max_attempts` (3) attempts instead of the indexing client's
  180 s in 6, and Neptune connects, reads and writes within the same
  timeout; it had none, so a query (and the process exit) blocked forever.
  A request still unanswered fails the query with `StoreConnectionError`; it
  answered from no context and reported success (#258).
- A Neptune connection the server closed (a restart or failover) is
  dropped and the next query reconnects; every later query failed with
  `Connection was already closed.` until the process restarted. Every pooled
  websocket is now signed when it opens, so with IAM auth one first used more
  than five minutes after the connection was built is no longer refused.
  Every failed connection probe read as a connection failure; a throttled or
  failed probe request is now transient like any other, a timed-out one is
  reported as such, and only a refused or lost connection is a
  `StoreConnectionError` (#258).
- `ragas` evaluation counts each retrieved context once per batch. With
  `context_precision` selected, the count-capped selection walked the
  contexts again, which repeated every CountTokens call once a batch had more
  than 1,024 contexts (#246).
- A failed UMAP reduction of the visualization layout now falls back to
  the seeded spring layout and records `"layout_degraded": true`, as the user
  guide documents; it silently returned a PCA layout marked as not degraded
  (#246).
- `run-visualization` and ingestion-time rendering compute the spring
  layout only for the nodes the interactive page draws (the top
  `graph.visualization.interactive.max_nodes` by degree). A large graph was
  laid out whole first: about 3.5 minutes of an 8,000-node render went to
  nodes that were then dropped. Graphs within the cap get the same layout as
  before (#246).
- `retrieval` matches a source by its file-name stem only when the
  reference has no extension: `Report-A.pdf` no longer matches a source
  `Report-A.docx`, and a version suffix such as `.v2` is part of the name, so
  `notes.v1` does not match `notes.v2`. `docs/Report-A.pdf`, `report-a.pdf`
  and `report-a` still match `Report-A.pdf` (#246).
- `answer_match` scores a Korean, Chinese or Japanese yes/no gold (`예`,
  `네`, `아니요`, `아니오`, `是`, `对`, `否`, `不是`, `はい`, `いいえ`) by the
  answer's polarity, as it already did for `yes`/`no`. Such a gold was
  matched as a substring, so `예` scored 1.0 against `아니요, ... 예외 ...`
  (#246).
- DRIFT's query-refinement summary labels community reports `Community:`
  again. It matched the index name against `community_reports`, which the
  real `graphrag-community-reports` prefix never contains, so every report
  was labelled `Item:` (#247).
- `search.fusion.method: weighted` no longer logs three "no configured
  weight" warnings per query. The default `fusion_weights` now name the
  buckets the strategies emit (local's `relationships`,
  `bridge_relationships`, `community_reports` and `claims`, and LightRAG's
  endpoint, incident and linked-chunk streams) and drop
  `opensearch_candidate_community_reports`, which nothing emits. Every
  weight stays 1.0, so ranking is unchanged (#247).
- LightRAG `naive` returns `search.lightrag_search.chunk_stream_top_k`
  chunks (default 20), as upstream `naive_query` returns `chunk_top_k`; it
  returned `top_k` (10). The docs now say that `naive` runs the request's
  `search_type` (BM25 + vector under the default `hybrid`) rather than pure
  vector retrieval (#247).
- A Japanese or Chinese section truncated to fit its context budget keeps
  most of its room instead of collapsing to a few characters. Truncation
  backed off to the last ASCII space, which in such text can be the first
  word; it now ends on whitespace or CJK punctuation near the cut, or at the
  cut itself (#247).
- A small per-query context budget (`RAGInput.max_tokens` of 513 or less)
  no longer yields an empty context and an abstention. The 512-token prompt
  buffer is now taken from the configured or derived
  `search.token_manager.max_context_tokens` only, not from an explicit
  per-query budget (#247).
- With MMR diversity on (`search.fusion.diversity_lambda` < 1, the default),
  `local` and `drift` queries keep every reserved text-chunk slot. The
  diversity cut ran once across all result types before the per-type quota,
  so over-quota entities took slots the quota then dropped and text kept 16
  of its 20; each type is now diversified within its own quota (#247).
- `run-ingestion` prints why it exits 1 (`Error: Pipeline failed at
  stage(s): ...`, as the operations guide documents), and a run with a
  failed stage no longer logs "Pipeline execution completed successfully"
  but a WARNING (#248).
- The ingestion summary's "Cache Hit Rate" and the `cache_hit_rate` metric
  report the run's stage-cache hits; they read 0% on every run (#248).
- A failed parent-level community report no longer records every document
  under it `FAILED`, which made the next incremental run re-extract most of
  the corpus for one report. Only a base-level report failure marks its
  documents; parent failures are logged and counted in `reports_failed`
  (#248).
- Changing `indexing.additional_suffix` or `document_parsing.source_scope`
  for a reused `--pipeline-id` recomputes the documents (from parsing and
  loading respectively) instead of restoring the old ones, which carried
  the previous registry namespace or scope into the run's delta. The stage
  cache keys of a configuration that sets neither are unchanged (#248).
- Switching gleaning or claim extraction on for an existing `--pipeline-id`
  runs the stage: the resume starts at the first enabled stage that has not
  completed in the pipeline. It started after the last recorded stage, so
  the new stage never ran (#248).
- An incremental run whose new and changed documents name no entity (an
  edit that removed the last one) commits them and applies the run's
  deletions. It failed graph extraction ("produced 0 outputs"), so the
  deletions were never applied and every later run failed the same way. A
  full build, or a run in which a text unit failed, still fails (#248).
- OpenSearch terms-filter batching: a `terms_batch_size` below the clause
  budget's share is honoured instead of being raised (50 became 150), and a
  query with two or more over-long list filters searches every combination
  of their batches; batch i of one filter was paired only with batch i of
  the other, so matches across batches were missed (#249).
- A search OpenSearch rejects with HTTP 400 (a filter value of another type
  than its field, a `hybrid_search_pipeline_name` that does not exist) fails
  with a `ConfigurationError` naming the server's reason. It degraded like a
  timeout, so the query was answered from no context, or from the indices
  the filter did not apply to (#249).
- An OpenSearch or Neptune that cannot be reached (connection refused, DNS
  or TLS failure) fails the query with `StoreConnectionError` instead of
  reading as "no results": every strategy abstained and `run-eval` reported
  a 100% success rate against a store that was down. Since #261 a store
  timeout, or throttling or a 502/503/504 that outlasts the client's
  retries, fails the query too; a single HTTP 500 still degrades the one
  sub-search. OpenSearch errors now log the server's reason instead of
  `info=None` (#249).
- Chunking and translation no longer hide bugs in their own code: six
  catch-alls around pure chunking steps (logged at DEBUG) returned the input
  or an empty list, and translation's input preparation returned `[]` on an
  error, which later failed in an unrelated `zip`. Claim extraction raises
  on a result count that does not match its text units instead of warning
  and attributing claims by position (#250).
- `entity_grounding.action: penalize` applies to gleaning additions too: an
  entity or relationship gleaning proposed with evidence not found in the
  chunk was dropped as under `drop`. It is now kept at its confidence (or
  weight) times `penalty_factor`, on extraction's rules (a relationship only
  when the chunk lists both endpoints; an entity still subject to
  `entity_confidence_threshold`). A gleaning stage output cached by an
  earlier version is reused until gleaning runs again (#250).
- An entity or relationship merged from text units of documents with
  different `filters` metadata carries every document's value (a list for a
  key with several), in extraction, gleaning, graph resolution and the
  incremental merge. It kept the filters of whichever instance the merge
  happened to keep, so an `attr_<key>` filter on another of its documents
  missed it depending on input order. Takes effect on the next
  (re-)extraction of the affected documents. The user guide's filter example
  used `type:PERSON`, which matches nothing because types are stored
  lowercase; it now shows `type:person` (#250).
- Description re-summarization no longer sends a whole merged description
  to the model in one call however long it is (a popular entity reached
  ~100k input tokens): a description over 12,000 tokens is summarized
  map-reduce in budget-sized groups of its lines, as LightRAG does. A
  description within the budget is summarized exactly as before (#250).
- Graph resolution renames a relationship's `source_name`/`target_name` to
  the entity its endpoint merged into: an edge extracted as "Acme" kept that
  name after "Acme" merged into "Acme Inc", and the stale name reached
  OpenSearch, Neptune and the LightRAG context. A fuzzy-matching worker
  failure (for example a broken process pool) now fails the stage unless
  `processing.ignore_errors` is on; it fell back to exact matching with a
  warning, leaving duplicate entities in a run that reported success (#250).
- Community detection gives the same communities at every level in every
  process: `graspologic_native` returns its labels in a per-process hash
  order that set the next level's cluster-graph numbering and the
  small-community merge order, so level-1+ communities, their ids and their
  reports could change between two runs on the same graph. A Leiden failure
  now fails the stage with a `GraphError` naming the graph size instead of
  being logged and silently truncating the hierarchy (or leaving a
  misleading "Communities must be detected" error), and
  `graph.community_detection.random_state: 0`, which Leiden rejects, fails
  config validation. The `resolution` description had its direction
  inverted: higher gives more, smaller communities (#250).
- Importing `unified_kg_rag.application.cli.run_rag_chain` no longer puts
  `unified_kg_rag/application` at the front of `sys.path`, where its
  subpackages could shadow top-level modules of the same name (#251).
- On an incremental run, a new relationship whose endpoint merged into a
  stored entity under another name (for example `Acme Inc` into `ACME, Inc.`)
  now carries the stored entity's name in `source_name`/`target_name`, as a
  full build does; it kept the extracted name (#252).
- The incremental cross-run merge no longer writes a description cut at
  `indexing.neptune.property_max_length` to OpenSearch. It read stored
  entities and relationships back from Neptune, which caps property values,
  so a long description lost its tail in OpenSearch on the next merge; a
  capped description is now read from OpenSearch, which stores it in full
  (#252).
- A failed Neptune delete batch is counted as one failure per id in it, with
  the ids, instead of a single failure with none, so the failure rate and the
  logs of a document removal reflect what was not deleted (#252).
- A Neptune relationship write that is retried after a lost response no
  longer adds a second parallel edge with the same id: the edge write looks
  the edge up first (`fold().coalesce(unfold(), addE(...))`) (#252).
- An incremental run that only deletes documents (nothing new or changed to
  extract) now checks the live OpenSearch indices of its configured
  `document_parsing.index_value` before writing; it checked the `default`
  namespace's indices instead (#252).
- Incremental runs merge a claim that another document already stored instead
  of overwriting it. A claim's id does not depend on its document, so adding a
  document that states the same claim replaced its description, source text
  and text units with the new document's only, and deleting that document
  left the claim citing it. Claims now merge as in a full build (read back
  from the OpenSearch claims index, which stores their `text_unit_ids`), and a
  removed document's text units are stripped from claims shared with
  surviving documents (#252).
- OpenSearch indexing embeds and bulk-writes each artifact type 400 items at
  a time instead of embedding the whole type before one bulk call. Memory no
  longer grows with the corpus (about 75 KB per entity before, several
  gigabytes at 50,000 entities with up to six types indexed concurrently), and
  the index is refreshed once per type as before (#252).
- A changed `search.hybrid.lexical_weight` or `vector_weight` now takes
  effect: `run-ingestion` writes the OpenSearch hybrid search pipeline on
  every run. Before, it was created only when absent, so the weights of the
  first run on a domain stayed in force (#252).
- With `indexing.opensearch.persist_embedding_cache`, an embedding-cache
  object that exists but cannot be read (throttling, access denied, a network
  error) is no longer treated as empty: the run embeds without it and does not
  overwrite it. Before, its first flush replaced the object with only this
  run's vectors, dropping every other model's and dimension's entries, and the
  next runs re-embedded the corpus (#252).
- `run-eval --eval-data-path` naming a file that does not exist exits 2 with
  `Error: invalid evaluation dataset: ...`, as the user guide documents and
  as an unreadable or invalid dataset does; it exited 1 (#230).
- A configuration file with a value of the wrong type, invalid YAML or a
  missing `--config-path` file no longer makes `run-ingestion`, `run-rag` or
  `run-eval` print "An unexpected error occurred" with a traceback. All five
  CLIs treat it as a usage error: one `<cli>: error: ...` line and exit code
  2 (`run-prompt-tuning` and `run-visualization` exited 1). A validation
  error now fits on one line, naming each invalid key without echoing its
  value (#230).
- A `local` query against a missing index (for example a wrong `--suffix`)
  no longer also logs "_GatheringFuture exception was never retrieved" with
  a second traceback: the community-report, relationship and claim sections
  that fail alongside the entity chain have their error consumed (#230).
- A `GraphRAGChain` reused across event loops (one `asyncio.run` per query)
  closes the retrievers of each finished loop again: their synchronous close
  ran inside the next query's running loop, where the Neptune (Gremlin)
  client's close fails with "Cannot run the event loop while another loop is
  running", so every loop left a websocket and its threads open (15 queries
  grew a process from 5 to over 40 threads). The close now runs on a worker
  thread (#230).
- Switching `processing.claim_extraction` on, or changing its model, limits
  or claim prompts while it is on, now re-extracts an incrementally indexed
  corpus: the extraction fingerprint stopped at graph resolution, so such a
  run reported 0 changed documents and indexed no claims, without a warning.
  With claim extraction off the fingerprint is unchanged, so default
  configurations are not re-extracted on upgrade; a deployment already
  running with claim extraction on re-extracts its corpus once (#230).
- Attaching, rotating or toggling a Bedrock guardrail under the default
  `apply_to: query` no longer re-keys the stage caches or re-extracts the
  whole corpus on the next incremental run: ingestion runs unguarded then, so
  the guardrail counts toward the cache keys and the extraction fingerprint
  only with `apply_to: all` (and its `trace` flag never does). Likewise the
  settings of a block switched off with `enabled: false` (translation,
  fixing, gleaning, claim extraction, description summarization, entity
  grounding, community reports) no longer move them. Default configurations
  keep their keys; a configuration with a switched-off block whose other
  fields differ from the defaults re-keys once, and so does one with a
  query-only guardrail identifier set (as the CDK stack injects through
  `BEDROCK_GUARDRAIL_IDENTIFIER`): with incremental indexing its first run
  after the upgrade re-extracts the corpus once (#226).
- With translation switched off, changing `processing.translation`'s
  `target_language`, `source_language` or `additional_target_languages`
  re-keys the stage caches and the extraction fingerprint again, so the
  graph is re-extracted instead of kept stale: extraction, gleaning, claim
  and description prompts still write in the target language and the
  OpenSearch indexer picks its analyzers and translated-text fields from
  the languages whatever `enabled` says. Likewise
  `report_generation.max_entities_per_report` still keys community
  detection with reports off, since it caps the entity names each community
  lists. Only the settings a switched-off block alone reads (such as the
  translation model) stay out of the keys (#228).
- Answer matching normalizes text with many dashes in linear time: each dash
  re-scanned the text before it, so a 70 KB answer with 4,000 dashes took
  seconds (#226).
- Answer matching no longer glues a lone capital letter to the word before
  it, which made "Yes I agree" normalize to `yesi agree` and "Vitamin A"
  differ from "vitamin a": a spaced lone letter is again a word or an
  article in either case. A dash next to a lone letter still joins it in
  either case ("Type-A" and "type-a" are both `typea`), while "Type A" is
  read like "Type" plus the article "a" (#228).
- `backend: local` sets its 22,000-token retrieval context budget only for a
  non-Bedrock LLM left at the default 32,768-token window. With
  `providers.llm.kind: bedrock` or an explicit `context_window_size` the
  regular default (30,000, clamped to the model's window) applies instead of
  capping a larger model at the small local one's budget (#226).
- A follow-up rewrite that opens with an apology or refusal and asks the user
  to clarify ("I'm sorry, could you clarify which contract?") is discarded
  like any other refusal; the question mark no longer lets it replace the
  user's query (#226).
- A query against an index suffix nothing was ingested under fails with
  `No indices found for suffix '<suffix>'` again, as documented, instead of
  logging an ERROR and answering "could not find" with exit code 0: the async
  OpenSearch client reports the 404 body as raw JSON text, which the
  `index_not_found_exception` check did not read (#227).
- `DataIngestionPipeline.run(source_directory)` ingests the directory it is
  given. The parsing and loading stages kept the constructor's directory (or
  the configured default), so a run over another directory read the old
  corpus and diffed the incremental registry under the old source scope (#227).
- Local mode: one chain serves queries from successive event loops (a second
  `asyncio.run(chain.ainvoke(...))`, or `stream` and then `astream`). The
  Ollama chat and embedding models kept one async HTTP client, bound to the
  first loop, and failed later calls with "Event loop is closed"; they now
  get one client per loop (#227).
- Local mode: the Ollama chat and embedding models no longer keep a client
  and its open connection for every event loop they were used on, which grew
  by one client and two file descriptors per `asyncio.run`. Their async
  clients open no keep-alive connections, so each connection closes on its
  own loop, and the clients of closed loops are dropped (#229).
- Local mode: an OpenAI-compatible chat or embedding model sends one request
  per call from successive event loops. langchain-openai shared one async
  HTTP client across loops, so the first call on each new loop hit a pooled
  connection of the closed previous loop ("Event loop is closed") and was
  sent again: a paid endpoint billed it twice, and with `max_attempts: 1` it
  failed with `APIConnectionError`. These models now get one async client per
  loop without keep-alive connections, as the Ollama models do (#231).
- An incremental run after a change of the embedding model or dimension
  stops in the indexing stage, before any write, with a `ConfigurationError`
  that asks for `indexing.reset: true`. It re-embedded only the new and
  changed documents, so the live indices mixed the old and new models'
  vectors (or rejected every write on the dimension). Each OpenSearch index
  now records its embedding model and dimension in its mapping's `_meta`;
  indices built before are checked by their dimension only (#231).
- A query against indices built with another embedding model or dimension
  fails with a `ConfigurationError` naming both models (`run-rag` exits 1,
  `run-eval` counts the query as failed). It reported success with an empty
  answer: OpenSearch's "Query vector has invalid dimension" rejection was
  treated as a transient error and degraded to 0 results, and a model of the
  same dimension ranked by unrelated vectors. The retriever checks the
  model and dimension each index records once per index, and treats that
  rejection as the same error (#232).
- With `indexing.opensearch.embedding_dimension` set below the embedding
  model's largest (for example 512 for Titan Embed V2), queries now embed
  with that dimension, as ingestion does. They embedded with the model's
  default, so every vector and hybrid search of such an index came back
  empty (#232).
- A `GraphRAGChain` reused across event loops no longer stalls a query while
  it closes the retrievers of the previous loop: the query waited up to 30
  seconds for a slow (for example Neptune websocket) close. The close runs on
  a background thread, and `close()`/`aclose()` of the chain wait for it
  (#232).
- An invalid environment override is a usage error naming the variable
  (`<cli>: error: Invalid environment variable LOG_LEVEL ...`, exit code 2)
  instead of being blamed on the configuration file or accepted: overrides
  are now checked against the field's constraints, so `LOG_LEVEL=garbage`,
  `LOG_FORMAT=xml` or an empty `AWS_REGION` is rejected (`LOG_LEVEL` still
  accepts any case), and `OPENSEARCH_NUMBER_OF_REPLICAS=abc` no longer ends
  with a traceback. A configuration file that is not UTF-8 or cannot be read
  (permission denied) is the same one-line usage error (#232).
- An incremental run after a change of the corpus or translation languages
  (or `language_analyzers`) stops before any write with a
  `ConfigurationError` that asks for `indexing.reset: true`. It re-extracted
  the documents into the live indices, whose text fields keep the analyzers
  they were built with: a newly translated language was mapped as a keyword
  (`translated_text_ja`), a new corpus language kept the old analyzer. The
  check compares the live mapping's text fields with the run's (#233).
- An incremental run on a suffix last indexed with the doc-status registry
  off (the `backend: local` preset and flag-only runs default to off) stops
  before any write with a `ConfigurationError` that asks for
  `indexing.reset: true`. The registry-off run replaced the indices without
  recording their documents, and the next incremental run trusted the stale
  registry ("1 new, 10 unchanged"), so documents went missing. Each index
  now records in its `_meta` whether the registry tracks it. An index built
  before it recorded its settings is not checked for languages or the
  registry, and is given its settings when its dimension and text fields
  match, so later runs check it in full (#233).
- A run with `indexing.reset: true` that reuses the pipeline id of an
  incremental run (a fixed `--pipeline-id`) rebuilds the stores from the
  whole corpus. It resumed at the indexing stage from the previous run's
  cached delta, so the reset cleared the stores and the registry and indexed
  only the delta's documents; the unchanged ones were missing until the next
  run. `indexing.reset` is now an input of the document-loading stage cache
  key (only when set, so existing caches keep their keys), not of the
  extraction fingerprint (#233).
- An OpenSearch index built before indices recorded their embedding
  settings (#231) is given only its vector dimension, not the run's
  embedding model: another model of the same dimension may have built it,
  and recording the run's model would then refuse the right configuration
  at ingestion and at query time. Its model stays unchecked, as before.
  Recording the dimension is best-effort: a role without
  `indices:admin/mapping/put` gets a WARNING instead of an incremental run
  that stops before its write-ahead. Such an index whose text fields do not
  match the run's languages is still written to, now with a WARNING naming
  the fields and the `indexing.reset: true` remedy (#243).
- A principal that may search but not read index mappings (an OpenSearch
  403) can query again: the first vector or hybrid query's embedding check
  read the mappings and failed every query on the denial. The check is now
  skipped with one WARNING; the search's own errors still fail the query.
  The invalid-dimension reason of a rejected query is cut at the end of the
  reason instead of running on through the rest of the error body (#243).
- Closing a `GraphRAGChain` (`close()`, `aclose()` or its GC finalizer) waits
  only for its own background retriever closes. It waited up to 30 seconds
  for those of every chain in the process, so a dropped chain collected on
  an event-loop thread could block that loop (#243).
- IaC: a zone-aware OpenSearch domain spans as many AZs as it has data
  nodes, up to 3 and up to the VPC's AZs, so `opensearch_count=3` with
  `max_azs=3` uses three zones instead of two; on an existing domain the
  first deploy is a blue/green deployment (see the IaC README).
  `bedrock_model_arns` that
  list only foundation models keep the inference-profile grants, so models
  offered only through a profile stay callable (the profile call is still
  authorized against the listed models). The README documents both, and
  its cross-region guardrail example now sets `network_mode=public`, which
  a `bedrock_region` other than the deploy region requires (#227).
- `run-prompt-tuning` no longer puts up to 200 characters of model output in
  the ERROR it logs when the corpus profile is not JSON: the message carries
  the output's length and digest, and the output itself is logged at DEBUG,
  as the logging policy (user guide §2.7) states (#227).
- `run-eval` reports an unreadable or invalid dataset as
  `Error: invalid evaluation dataset: ...` with exit code 2 instead of "An
  unexpected error occurred" and a traceback, and rejects an `answer_aliases`
  that is not a string or a list of strings at load instead of failing every
  query. The random part of the result file names is documented (#227).
- The `run-ingestion` performance summary of a resumed run counts the stages
  restored from the stage cache under "Stages from Cache" instead of
  reporting 0 and every completed stage as executed (#227).
- `run-ingestion` with a `--target-directory` (or
  `document_parsing.target_directory`) equal to the source directory exits 1
  with a one-line error instead of "An unexpected error occurred" and a
  traceback (#227).
- Queries on Ollama retry a transient error (HTTP 408/429/5xx, dropped
  connection, timeout) instead of failing on the first one: the Ollama client
  never retries, so query chains and Ollama embedding calls now get the
  framework's retry, `providers.llm.max_attempts` /
  `providers.embedding.max_attempts` attempts. OpenAI-compatible servers keep
  the OpenAI client's own retries, with no second layer (#225).
- A local ingestion run no longer imports the AWS SDK at community detection:
  Leiden now calls `graspologic-native` directly, as `graspologic` does,
  instead of importing `graspologic`, whose package import loads boto3
  (through gensim and smart_open) and takes about 10 s. Partitions are
  unchanged (#225).
- Re-running `run-ingestion` with an existing `--pipeline-id` resumes again:
  the integrity check built its cache keys before the corpus fingerprint was
  known, so it reported every stage missing and prompted, which crashed
  without a terminal. Prompts now fail with a clear message without a
  terminal unless `--yes` is passed (#223).
- Retrying a failed Step Functions execution with the same `pipeline_id`
  resumes at the failed stage: an earlier attempt's failed stage result is no
  longer merged into the retry (it failed every phase), and the phases no
  longer pass `--resume-from-stage`, which re-ran each window's LLM stages;
  each phase skips the stages of its window that completed (#223).
- `--force-rebuild` writes the recomputed stage outputs to the cache instead
  of leaving the previous ones in place (#223).
- With `--continue-on-error`, documents whose translation, extraction,
  resolution, analysis or community stage crashed are recorded `FAILED`, not
  `PROCESSED` with a partial graph that the next run skipped as unchanged;
  when parsing, loading or chunking crashed, indexing does not run, where a
  parsing crash used to remove every stored document as deleted (#223).
- `--resume-from-stage` is case-insensitive as documented, and an unknown
  stage name is a usage error listing the valid stages instead of a silent
  no-op; a missing `--config-path` file is a usage error too (#223).
- Generated pipeline ids carry a random suffix, so runs started in the same
  second no longer share a cache directory; `--verify-metadata` reports an
  unknown pipeline id as not found; each phase of a phased run emits only the
  metrics of the stages it ran, so dashboard sums no longer over-count (#223).
- `run-rag` starts again: it passed `enable_thinking` to `GraphRAGChain`, which no
  longer accepts unknown keywords (the flag reaches `RAGInput`) (#222).
- Answer matching keeps a dash next to a lone letter ("Type-A" no longer
  matches "Type-B"); the follow-up rewrite's refusal guard no longer discards
  rewritten questions that start with "I can't…" or "Sorry, …" and catches
  curly-apostrophe refusals; `DataIngestionPipeline`, the chunkers and the
  LangChain/RAGAS evaluators reject a `boto_session` they would have ignored (#222).
- Incremental indexing follows the extraction settings, not only the
  content: each doc-status record stores a fingerprint of the settings it was
  extracted with (parsing through graph resolution: models, prompts such as a
  tuned `custom_prompts` set, `entity_types`, chunking, translation, gleaning,
  resolution), and a run with other settings re-extracts the unchanged
  documents (one WARNING gives their count). Before, unchanged documents kept
  their old extraction even with `--force-rebuild` (#224).
- `run-rag` documents conversation memory as what it is, in-process: a single
  `--query` with `--use-memory` or `--conversation-id` starts with no history
  and now logs a WARNING; use `--interactive` (or one library process) for
  multi-turn. `--interactive --mode search` is rejected instead of silently
  generating answers (#224).
- Importing the package no longer loads a `.env` file into `os.environ`.
  The CLIs load only `./.env` in the working directory, never one found in a
  parent directory (a `~/.env` with `AWS_PROFILE` could switch accounts), and
  never override a variable already set (#224).
- AUTO routing falls back to `local` only when it is routable; with
  `search.auto_routable_strategies` excluding it, a failed or unparseable
  routing call uses the first routable strategy (#224).
- `--backend local` over a config file of another backend resolves the
  relative SQLite registry path against the config file, not the working
  directory (#224).
- An `s3://` source scope with and without a trailing slash is one corpus: a
  run whose spelling the registry does not hold, but the other one, keys its
  documents by the stored spelling instead of re-indexing the corpus beside
  the stored copy, and `--retire-source-scope` reaches either spelling (#224).
- IaC: `-c bedrock_model_arns=arn:...,arn:...` is accepted as a
  comma-separated list; a misspelled boolean (`use_cmk=ture`), a zero or
  out-of-range count, timeout or retention, and a CPU/memory pair Fargate
  does not offer fail synth with an error naming the key, instead of reading
  as `false` or failing at deploy (#224).
- An incremental run that skips unchanged documents no longer replaces
  `visualization_data.json` with a graph of its delta only: it skips the
  export and the render and logs how to visualize the whole graph (a full
  build, then `run-visualization`). The default export gives every node its
  level-0 `community_id`, so `run-visualization` colors communities instead
  of drawing every node grey (#224).
- Incremental indexing: shared artifacts are stripped instead of deleted or
  left citing removed text units; interrupted runs are repaired on the next
  run; resets and deletions no longer touch other index namespaces or
  OpenSearch indices; partial bulk failures mark their documents `FAILED`;
  full and incremental builds produce the same descriptions, weights and
  ranks (#159, #163, #185-#200, #214).
- Graph quality: gleaning no longer revives rejected entities or overwrites
  descriptions; claims no longer resolve onto a different entity; CJK names
  with a different head character no longer merge; communities no longer
  duplicate singletons across levels (#201-#203, #213, #214).
- Claims: when no extracted claim resolves to an entity, or claim resolution
  does not run, no claim is indexed; the unresolved claims (no subject or
  object id) no longer reach the graph, the claims index or the document
  lineage (#220).
- An entity named only inside a relationship is created under the
  relationship's index suffix and filters, so the edge write no longer fails
  (and the document is no longer retried forever) when ingesting with
  `index_value` (#220).
- Gleaning: a re-proposal of a name or edge type a correction replaced
  merges into the corrected item instead of producing a second entity or
  edge with the same id, and graph resolution merges items that share an id,
  so Neptune no longer gets two vertices or parallel edges for one id (#220).
- Incremental deduplication prefers only documents indexed successfully: a
  near-copy of a document that failed indexing is kept instead of it, so the
  content is indexed even when the failed copy used up its retries (#220).
- Clearing a Neptune label stops after a bounded number of delete requests
  and raises instead of looping when the requests keep reporting full
  batches (#220).
- Local mode: the ingestion stages retry the local model servers' transient
  failures (connect and read timeouts, dropped connections, HTTP 408/429/5xx
  from the OpenAI-compatible and Ollama clients) instead of failing the item
  on the first one; they take the classifier from the LLM factory
  (`is_transient_error`), and a factory without one gets these network and
  server errors retried (#221).
- Local mode imports no boto3 or botocore: the CLIs, the ingestion
  pipeline, the query chain and the local stores import the S3 sync and
  transfer helpers, SigV4 signing and the AWS session only when a run uses
  them (#221).
- Two full rebuilds within one second no longer reuse the live OpenSearch
  index: the replacement index was created under the live index's name, so
  it was written into the live index and kept the previous build's
  documents (and an emptied type was not emptied). Index names carry
  milliseconds, a taken name is moved forward, and a full rebuild refuses to
  write into an index that already exists; indices named by earlier builds
  are still reaped (#221).
- The check that the graph still holds the registry's documents probes
  indexed documents that list entities, reading past entity-less ones, so a
  corpus starting with entity-less pages no longer lets a wiped graph pass as
  intact (#221, #225).
- `--backend aws` over a `backend: local` config file no longer keeps the
  local preset's providers and connection keys under the `aws` label: values
  equal to the local preset's return to the AWS defaults (#221).
- Parsing: the LLM XML parser keeps `&`, `<`, repeated siblings and
  sections with a missing closing tag; truncated responses fail instead of
  parsing as partial successes (#165, #174, #182, #213).
- Retrieval: graph expansion returns every seed and its neighbourhood;
  Korean BM25 recall, CJK fuzziness and NFC normalization; fatal errors
  surface instead of reading as no results (#102, #161, #175, #177, #212).
- Retrieval: graph expansion ranks by distance from the seeds first, so
  high-degree neighbours no longer push the query's own seed entities out of
  the cut; the seeds are always kept and the neighbourhood gets its own
  `top_k * retrieval_multiplier` slots (#218).
- Local search: the entity frequency filter no longer drops the query's own
  matched entities, so a hub-heavy graph cannot leave the context with no
  entities; only expanded entities above the threshold are dropped (#218).
- Conversation memory: earlier turns' entities no longer leak into the entity
  focus of a rewritten follow-up, and a turn no longer pays an
  entity-extraction LLM call (for DRIFT and global search too); the rewrite
  resolves references instead and reads a bounded history (earlier answers
  cut to 1,000 characters) (#218).
- An empty or refusing follow-up rewrite no longer replaces the user's
  question; the query is used as is (#218).
- `GraphRAGChain(providers=..., boto_session=...)` raises instead of silently
  ignoring the session (#218).
- Global search: when dynamic selection keeps no community, the fallback
  reports are capped at `max_communities` and get their text units and the
  rerank, instead of the raw retrieved list (#218).
- A shared query-embedding call that fails after all its waiters were
  cancelled no longer logs "Task exception was never retrieved" (#218).
- Evaluation: `answer_match` no longer counts a different number or name as
  a match; judge failures are recorded per query instead of discarding the
  run; non-zero exit codes on failed scoring (#158, #210).
- `answer_match` and `graph_aware`: a bare number matches it followed by a
  Korean unit (`2024` in `2024년에`) but still not a magnitude (`3` is not
  `3억`); a minus sign and a percent sign stay part of a number (`-5` is not
  `5`, `3.5%` is not `3.5 million`); a dash between words is a space
  (`1990-1995` is two numbers, `5-year` = `5 year`); a yes/no gold reads the
  first yes/no that answers the question, past Markdown headings,
  abbreviations and "there is no ..." (#219).
- RAGAS: a failed run marks only its own metrics failed. A failure in the
  run for rows without a reference no longer drops their skipped reference
  metrics, and a failed `context_precision` run (or the run beside it) keeps
  the scores of the other (#219).
- CLIs and config: an empty config loads the defaults; `custom_prompts`
  overrides are validated at load; `run-rag` exits cleanly on usage errors;
  importing the package no longer configures logging (#158, #178).
- IaC: alarms deliver notifications, the corpus no longer expires with the
  cache prefixes, OpenSearch indices get replicas on multi-node domains,
  timed-out or aborted pipeline runs alarm, and `cdk destroy` leaves no
  fixed-name resources (#157, #166, #211).
- IaC: the pipeline-failure notification renders the whole error, so an
  error without a `Cause` no longer fails the SNS publish; the Neptune
  Gremlin-errors alarm needs errors in 3 consecutive minutes, so retried
  write conflicts during parallel writes do not page (#219).
- IaC: the pipeline-failure notification retries its SNS publish on a
  throttled or transient error, which ended the run with no message. The
  `PipelineFailures` alarm on failed executions stays: a phase's error
  handler cannot catch `States.DataLimitExceeded` or `States.Runtime` (such
  as a start input without the fields a phase reads), so a failed phase
  still notifies twice, once with the execution id and error and once from
  the alarm without them (#254).
- Used as a library (no `setup_logging`), the package's log records reach
  the host application's handlers as the plain message, with the structured
  fields as record attributes; a host `logging.basicConfig` printed every
  record as a dict repr (#245).
- A streamed model call the caller stops reading early (`break`, `aclose`)
  ends its span normally; it was exported as an `ERROR` span with
  `error.type: GeneratorExit` (#245).
- Reconfiguring tracing with different settings no longer warns that "a
  global OpenTelemetry tracer provider is already set" when that provider
  is the one the package set itself; the user guide documents that the
  global binding happens once per process (#245).
- A LangChain `ainvoke` on a package retriever runs `aretrieve` on the
  caller's event loop; it ran it on a new loop in a worker thread, away
  from the loop the stores' async clients are bound to. The global and
  simple search spans and failure logs say "Document retrieval" instead of
  "OpenSearch retrieval", which local mode does not use (#300).

### Removed
- `Entity.frequency`: it counted an entity's text units in every merge path
  but no store wrote it and nothing read it (ranking uses `rank`, the local
  search frequency filter counts `text_unit_ids`). Stage caches that carry
  the key still load; it is ignored (#291).
- Internal helpers with no remaining use: `BaseEvaluator` (subclass
  `BaseGraphRAGEvaluator`, which now supplies the default
  `aevaluate_single` and `_initialize_evaluator`), the `algorithm`
  parameter of `compute_hash` (always SHA-256) and
  `unified_kg_rag.shared.get_cache_manager` (import
  `shared.cache_manager.CacheManager`). Prompt classes are plain classes
  instead of frozen dataclasses; they were never instantiated (#300).

### Security
- Prompts delimit corpus and retrieved text as data, and inputs cannot close
  their delimiter tags (#179, #182).
- INFO-level logs no longer carry query text, entity names or model output;
  `run-rag` strips terminal control characters from output; the CLIs warn
  when LangSmith tracing is on (#156, #173).
- Logged tracebacks no longer print each frame's local variables. With
  `log_format: plain` (and `structured` on a terminal) the console and
  `logs/log_*.txt` showed them whenever `rich` was installed, so a failed
  extraction or query wrote prompts, chunk text and model output into the
  log. Console output piped to a file or another program no longer carries
  ANSI colour codes either (#245).
- Gleaning and graph extraction no longer log relationship endpoint names
  at INFO, and gleaning no longer logs a preview of unparseable model output
  at WARNING. Gleaning is on by default, so every ingestion wrote corpus
  entity names (people, companies) to the console and `logs/`. INFO records
  now name the relationship and chunk by short id; entity resolution's
  fuzzy-match failure names the entity by `len=… sha=…` digest. A unit test
  fails when a new INFO+ log call passes names, descriptions, text or
  model-output previews (#255).
- The evaluation summary's `run_manifest.cli_args` records `--model-base-url`
  without its userinfo, query and fragment, which may hold an API key (#255).
- Captured prompt and completion text is redacted as raw text on every
  tracing backend: with `otlp`, `langfuse` or `phoenix` it was redacted only
  at export, JSON-encoded, where a `\b` or `\s` pattern next to a newline
  missed the match (#219).
- `pipeline_id` must be a single path segment (#173).
- The persisted embedding cache honours `aws.s3.encryption` (#173).
- Metadata filters are documented as relevance filters, not an
  access-control boundary (#173).
- Interactive graph tooltips are escaped (#180).
- Least privilege: the task role drops unused Neptune, S3 delete and
  `bedrock:GetInferenceProfile` actions, and private-mode endpoint policies
  admit only the account's principals (#204, #211).
- Dependency floors: `unstructured>=0.24.0` (GHSA-4mvj-m6j5-pmf7,
  GHSA-8mgp-746c-j5xp) and `langchain-openai>=1.1.14`
  (GHSA-r7w7-9xr2-qq2r).
