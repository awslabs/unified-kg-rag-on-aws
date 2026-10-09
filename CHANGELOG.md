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
- `indexing.max_document_failures` (default `3`): an incremental run stops
  retrying a document recorded FAILED that many consecutive runs with
  unchanged content. It stays FAILED and is skipped as unchanged with a
  WARNING until the file changes or the limit is raised; before, a document
  that failed deterministically was pruned and re-extracted on every run. The
  count is stored as `failure_count` on the doc-status record; records
  written before it existed read as `0` (#175).
- `DataIngestionPipeline` accepts `doc_status`, `vector_indexer` and
  `graph_indexer`, threaded to the loading and indexing stages, so the whole
  pipeline runs on custom or in-memory backends through its public
  constructor as `docs/design.md` §15 described. An injected `doc_status`
  turns incremental indexing on without `aws.dynamodb.enabled` (#178).
- `aws.bedrock.ingestion_effort` sets the reasoning effort of default-tier
  ingestion calls (graph extraction, gleaning, claim extraction, community
  reports; the output fixer only when `fixing.fixing_model_id` is a
  default-tier model) separately from query-time
  `default_effort`, so ingestion cost can be lowered on its own. `null` (the
  default) inherits `default_effort`, so behaviour and stage cache keys are
  unchanged unless it is set; fast-tier calls keep `fast_effort` (#179).
- Claude Haiku 5.5 (`anthropic.claude-haiku-5-5`) in the model catalog: 1M
  context, 128K output, adaptive thinking with `effort` low–max, 512-token
  cache minimum, served through inference profiles only (#162).
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
- MinHash entity resolution shares one set of seeded permutations instead of
  regenerating them for every name and query, hashes each name's shingles in
  one batch, and reuses the candidates' signatures when they are queried;
  signatures and matches are unchanged (pinned by a property test). Grouping
  5,000 names: index 0.7 s instead of 3.4 s, queries 0.3 s instead of
  3.5 s (#176).
- The indexer's embedding cache holds vectors as float32 arrays instead of
  lists of Python floats, and keeps one copy (the S3 tier when
  `persist_embedding_cache` is on, else the in-process tier) instead of two:
  5,000 1024-dimension vectors take 43 MB instead of 203 MB. A flush skips
  re-reading the S3 object while its ETag is unchanged since this process
  read or wrote it. The S3 object format is unchanged, so existing caches
  load as before; cached vectors now carry float32 precision, which is what
  the knn field stores (#176).
- `GraphRAGChain` builds each query-step LLM chain (router, entity/keyword
  extraction, translation, context building, answer) once per prompt, model
  and thinking flag and reuses it; it was rebuilt per query, creating two
  boto clients and fresh connection pools each time (~6 ms and new TLS
  connections per step, now a dictionary lookup) (#176).
- The script-aware token estimate, the only counter for models without
  CountTokens (the default Claude 5.5 models), uses one compiled regex instead
  of a per-character Python loop, with identical results; with no CountTokens
  API, `count_tokens_many` counts inline instead of starting an 8-thread pool
  per call. Budgeting 300 sections of 1.5 KB takes 24 ms instead of 476 ms,
  and 32 concurrent queries 0.8 s instead of 16 s (#176).
- `processing.document_parsing.index_value` is documented in the user guide
  §2.3 table and `config-template.yaml` as the ingestion-side suffix that
  `run-rag`/`run-eval --suffix` must match; the multi-tenant guidance no
  longer suggests a `--suffix` flag for `run-ingestion`, which has none (#178).
- `iac/README.md` lists the prerequisites: Python 3.10+, Node.js 20+ and
  AWS CDK CLI 2.1143.0 or later, AWS credentials and Docker (#178).
- The README quickstart runs the CLIs with `uv run`, which a `uv sync`
  install needs unless the virtual environment is activated (#178).
- The `docs/design.md` §15 "run without AWS" example passes a
  `rerank_factory` (reranking is on by default and called Bedrock) and notes
  that the retrievers and indexers must be injected too (#178).
- `CONTRIBUTING.md` and `docs/design.md` §15 describe the real extension
  mechanisms: a new strategy also needs a `SearchStrategy` enum member,
  backends are passed to constructors (there is no backend registry), and
  `run-visualization` only sees renderers imported by
  `adapters/renderers/__init__.py` (#178).
- `GraphRAGChain`, `RAGInput` and `RAGOutput` document the library contract:
  `close()`/`aclose()` when done, which event loop the sync and async
  methods run on, `RAGOutput` in RAG mode versus a dict in SEARCH mode, and
  that `stream`/`astream` yield answer text only (use `ainvoke` for the
  sources) (#178).
- Graph extraction and gleaning write entity and relationship descriptions in
  `processing.translation.target_language` (the language queries are
  translated to, equal to the source language when translation is a no-op).
  The extraction prompt had no output-language instruction, so descriptions
  followed whatever language the model chose, while description
  summarization already wrote in the target language. Entity names and
  verbatim evidence spans stay in the source text's form. Both prompts take a
  new `target_language` variable; existing `custom_prompts` overrides without
  it keep working (#179).
- The fast tier (`aws.bedrock.fast_model_id`) defaults to Claude Haiku 5.5
  (`anthropic.claude-haiku-5-5`) instead of Haiku 4.5. In a real-AWS A/B (79
  documents, 20 questions, 5 strategies, 2 ingests per arm) it matched Haiku
  4.5 on accuracy (151 vs 150 answers containing the gold) and ingestion time,
  at roughly 1/10 the per-token price; its tokenizer counts ~30% more tokens
  for the same text. It thinks adaptively, so `aws.bedrock.fast_effort`
  (default `low`) now takes effect on fast-tier calls. Cached stage outputs
  from chunking on miss once after upgrading, since their default-config cache
  keys include the model id. Set `fast_model_id` back to
  `anthropic.claude-haiku-4-5-20251001-v1:0` to keep the previous model (#170).
- Long-output prompts derive their `max_tokens` floor from the configured
  limits instead of fixed values, so ingestion reserves less of the
  tokens-per-minute quota: the largest answer the limits allow plus 8192
  tokens of reasoning headroom. At the defaults, graph extraction, gleaning
  and their output fixer drop from 32768 to 23192 (100 records x 150
  tokens), document translation from 65536 to 18992 (8000 characters x 1.35
  tokens), claim extraction from 32768 to 29792, and community reports from
  32768 to the 16384 default cap. `setup_chain` takes an optional
  `min_output_tokens` that overrides the prompt's static floor, and
  `BasePrompt.output_floor(config)` returns the derived one (#179).
- Gleaning stops per text unit on what the model returned, not on scores: each
  unit gets up to `max_rounds` refinement calls and is re-sent only while its
  previous answer added an entity or relationship the graph did not have (an
  empty answer, a re-proposed known item, or an ungrounded addition ends that
  unit). The corpus-wide early stop on the model's self-reported
  completeness/accuracy and a convergence score is gone, so units that keep
  gaining now run the full `max_rounds`. The refinement prompt still asks the
  model to score completeness and accuracy before listing issues: the scores
  are no longer read, but dropping that self-check cut round-1 additions by
  about half (+44 vs +89 entities) and final entities by about 5% in an E2E
  A/B. `gleaning_improvement_rate` in the pipeline metrics is
  now the gleaned entities plus relationships per extracted one (it was the
  relative change of the self-reported quality score), and the gleaning stage
  reports `refinement_calls`. Cached gleaning and later stage outputs miss
  once after upgrading, since their default-config cache keys change (#171).
- The output fixer (`fixing.fixing_model_id`) is on the fast tier
  (`aws.bedrock.fast_model_id`, Claude Haiku 5.5 at `fast_effort` `low`)
  instead of the default tier at `high` effort: it only re-emits a
  completion as well-formed XML. `fixing` is an input of every stage cache
  key, so every cached stage output misses once after upgrading. Set
  `fixing.fixing_model_id` to `anthropic.claude-sonnet-5-5` to keep the
  previous model (#179).
- The AUTO router prompt describes exactly `search.auto_routable_strategies`,
  built from the configured list in its order. It used to describe SIMPLE
  (not routable by default) first and steer "simple factual" and "direct
  lookup" queries to it, and left MIX out of two of its three decision axes;
  the hardcoded axes are gone, each strategy's own description carries the
  guidance. `StrategySelectionPrompt` takes a new `strategy_descriptions`
  variable, and `auto` in `auto_routable_strategies` is now rejected (#179).
- **Breaking:** `unified_kg_rag.shared.utils` no longer re-exports the
  LangChain-coupled and console helpers, so importing a `domain` module no
  longer loads LangChain, LangSmith, lxml, tenacity or tqdm. Import
  `BatchProcessor`, `BATCH_ITEM_FAILED` and `RobustXMLOutputParser` from
  `shared.utils.langchain`, `convert_langchain_to_document` from
  `shared.utils.document_converter`, and `console`/`display_*` from
  `shared.utils.display`. The domain purity test now also checks transitive
  imports in a clean interpreter (#164).
- The claim extraction prompt defines one claim taxonomy: the unused "Claim
  Categories" list (FACTUAL_ASSERTION, ENTITY_PROPERTY, ...) is removed, so
  only the CLAIM TYPES that `<claim_type>` takes remain (#179).
- CI runs the local-store adapter smoke test (`tests/integration/test_local_stores.py`)
  against the `docker/compose.local.yaml` Gremlin Server and OpenSearch (#168).
- Tests time out after 120 s each (`pytest-timeout` in the dev group), so a
  hung test fails fast instead of running until the CI job times out (#168).
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
- The CloudWatch EMF sink publishes only the dimensionless aggregate series
  (what the alarms and dashboard query) and records `pipeline_id` as a log
  property: the per-run `pipeline_id` dimension created about 20 new custom
  metric series per ingestion run. Pass `CloudWatchEMFSink(dimension_keys=...)`
  for stable dimensions. EMF lines now go to stdout, as the help text and docs
  already said (they went to stderr) (#158).
- `aws.bedrock.effort` is migrated to `aws.bedrock.default_effort` with a
  deprecation WARNING like the other renamed keys, instead of living on as a
  config field; stage cache keys are unchanged. **Breaking** for library
  callers: the `BedrockConfig.effort` attribute is removed (#169).
- Effort levels are declared once, as `EffortLevel` (with `EFFORT_LEVELS`
  derived from it); `evaluation.judge_effort` and the model catalog use it,
  and `BedrockLanguageModelFactory.VALID_EFFORTS` is removed (#169).
- `PipelineStageType` members are declared in pipeline order, and the cache
  keys and run metadata derive their stage order from it instead of keeping
  their own copies; iterating the enum now yields the stages in run order
  (#169).
- Indexers and retrievers derive OpenSearch alias/index names and Neptune
  vertex labels from one helper (`shared/utils/store_names.py`) instead of two
  copies of the naming rule and a repeated `prefix.capitalize()` at every
  Neptune call site; the names are unchanged (#169).

### Deprecated
- `search.llm_retry`; use `aws.bedrock.transient_retry` (#120).
- `aws.bedrock.effort`; use `aws.bedrock.default_effort` (#128).
- `SearchQuery.metadata["lightrag_mode"]`; use `search_strategy` (#123).
- `processing.max_retries`, `indexing.neptune.max_retries` and
  `evaluation.ragas_max_retries`; use the `max_attempts` keys. A renamed key
  (including `search.llm_retry`) now logs a deprecation WARNING instead of
  being reported as an unknown key that is ignored (#151).

### Removed
- Gleaning config keys `convergence_threshold`, `quality_threshold`,
  `min_improvement_threshold`, `quality_completeness_weight`,
  `initial_quality_entity_scale`, `initial_quality_relationship_scale` and
  `convergence_change_scale` under `processing.gleaning`, with the validator
  that kept `convergence_threshold` below `quality_threshold`. They only tuned
  the removed score-based stop rule and are now ignored with an unknown-key
  warning; `GleaningStats` drops `initial_quality_score`,
  `final_quality_score`, `convergence_achieved`, the per-round averages and
  the per-round `quality_improvement`/`convergence_score` (#171).
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
- Global search's Neptune community expansion, with
  `search.global_search.graph_timeout_seconds` and the
  `opensearch_expanded_community_reports` fusion bucket. Its limit equalled the
  candidate count and the candidates were emitted first, so it returned only
  the candidates: a Neptune and an OpenSearch round trip that re-fetched the
  same reports and counted each twice in fusion. Global search now needs only
  the document retriever; an old config key is ignored (#161).
- **Breaking** for custom indexers: the unreachable `BaseIndexer.get_stats` and
  `GraphIndexer`/`VectorIndexer.get_entity_count` port methods, their Neptune
  and OpenSearch implementations, and the client helpers that served only them
  (`NeptuneClient.get_graph_stats`, `NeptuneClient.submit`,
  `OpenSearchClient.get_index_stats`, `OpenSearchClient.count`). Nothing in the
  pipeline called them (#169).
- Unused parameters and fields: `BaseSearchStrategy(optimization_threshold_factor=,
  default_max_tokens=)`, `convert_langchain_to_document(n_chars=)`,
  `PipelineConfig.batch_size` (the CLI set it, nothing read it),
  `BedrockLanguageModelFactory.DEFAULT_EFFORT`, `StaticRenderer.color_palette`
  and `evaluation.base.SKIP_REASON_ANSWER_FAILED`. **Breaking** for callers
  that pass the removed keyword arguments (#169).
- `OptimizedContext.quality_score` and `TokenManager._calculate_quality_score`:
  the score was computed for every query and never read. **Breaking** for code
  that reads or constructs `OptimizedContext` with `quality_score` (#169).
- Production methods only tests called: `IncrementalIndexer.documents_to_process`
  (use `domain.ingestion.delta_detector.filter_documents_to_process`),
  `BaseGraphRAGRetriever.retrieve` (use `aretrieve`, or LangChain `invoke`),
  `OpenSearchIndexer.embedding_cache_hit_rate` and the explicit `fuzzy_matcher`
  argument of `graph_resolver.find_all_matches_for_entity_task`. **Breaking**
  for code that called them (#169).
- Model fields nothing populated or read: `TextUnit.covariate_ids`,
  `Community.covariate_ids`, `Covariate.covariate_type`/`subject_type`,
  `ConversationContext.current_topics`/`user_intent`, and
  `DocumentElement.coordinates`/`base64_encoding`. Cached stage outputs and
  exported data that still carry them load unchanged (the keys are ignored)
  (#169).

### Fixed
- The interactive graph HTML (`graph.html`, the community hierarchy) escapes
  node tooltips when pyvis renders them as HTML. Once any node title contained
  `href`, pyvis replaced the plain-text tooltip with a popup that sets
  `innerHTML` to the title, so markup in an attribute such as an LLM-written
  description (`<img src=x onerror=...>`) ran as script when the page was
  opened. Plain-text tooltips are unchanged (#180).
- The centrality comparison plot gives repeated node names a ` (2)`, ` (3)`
  suffix on its x axis. Two claims with the same subject, type and object
  (or two same-named entities) made Bokeh reject the axis with
  `DUPLICATE_FACTORS` and draw their bars on one factor (#180).
- `run-eval` sends each item's `question` to the RAG chain even when its
  `metadata` has a `query` key. The metadata key overrode the question at
  answer time, while `load_data` validated the item with the question
  winning (#180).
- Merging an undersized chunk into its neighbour no longer duplicates the
  splitter's `chunk_overlap` or fuses words at the seam. Adjacent chunks were
  concatenated as strings, so a single-paragraph document of ~5,100
  characters got its last ~500 characters twice and tokens like
  `w0799w0717`; the intelligent chunker's LLM-boundary pieces were likewise
  joined without their line break. Merged chunks are now the source text's
  span (overlap dropped and a separator used if a chunk cannot be located)
  (#180).
- `run-prompt-tuning` keeps the built-in output format. The tuned
  `graph_extraction_system` and `community_report_system` replaced the whole
  system prompt with a persona paragraph, dropping the XML schema the parsers
  read and the extraction rules (verbatim `<source_text>` grounding, no
  invented entities); with no generated examples there was no format at all.
  Each is now a domain-adapted preamble followed by the prompt's built-in
  rules and format verbatim, then any few-shot examples, which now carry a
  verbatim `<source_text>` span (records no sentence supports are left out).
  For `run-prompt-tuning` users: re-run it to regenerate prompts written by an
  earlier version. The tuned extraction prompt now holds the model to the
  configured `processing.graph_extraction.entity_types` (`{entity_types}`);
  the profile's entity types are listed as guidance, so copy them into
  `entity_types` to make them the strict categories. Loading a config now
  logs a warning (not an error) naming the output tags a `graph_extraction`
  or `community_report` override no longer mentions (#181).
- A damaged stage cache entry is a miss instead of a partial hit. A chunked
  entry skipped a missing or unreadable chunk with a log line and returned the
  rest as the stage output, and the resume check only tested that the chunk
  files existed. Chunk count and the recorded per-chunk hashes (and the
  single-file content hash) are now verified by both the resume check and the
  load, so a missing, truncated or rewritten file recomputes the stage with a
  WARNING. Chunk files, single-file entries and the index are written to a
  temp file and renamed into place, the index last (#181).
- A stage cache directory that is moved or restored to another path keeps
  its hits. `CacheEntry.local_path` held the absolute path at write time, so
  every entry became a miss after a move. It is now stored relative to the
  pipeline cache directory; entries written before hold an absolute path and
  are re-rooted by stage and file name under the current directory.
  `CacheEntry.exists_locally` is replaced by `resolve_local_path(dir)` (#181).
- `run-prompt-tuning` doubles the braces in every corpus- or model-derived
  field (persona, domain, language, entity types, few-shot examples) before
  writing the `custom_prompts` templates. A sample containing `{"retries": 3}`
  made graph extraction fail to format, and `{input_text}` in a sample was
  substituted with the chunk being extracted (#173).
- LightRAG keyword extraction keeps numeric keywords (`[2024, "Acme"]`) as
  text; it rejected the whole payload, which failed the query under the
  default `processing.ignore_errors: false`. Null, boolean and nested items
  are still rejected (#176).
- Claude Haiku 5.5 (the default fast tier) streams. langchain-aws 1.8.0
  streams only model ids on its allowlist, which lacks Haiku 5.5, so every
  model construction fell back to non-streaming Converse and logged a
  warning. A `supports_streaming` capability (also a `model_overrides` key)
  now sets it per model (#176).
- A Neptune connection whose probe query fails is closed before the error is
  raised, instead of leaving its websocket and thread pool open (#176).
- `memory.max_conversation_age_hours` is enforced: a conversation idle for
  longer is dropped on the next memory lookup (a returning one starts over).
  The TTL was stored and never read, so conversations lived until the
  `max_conversations` limit evicted them (#176).
- The CountTokens `bedrock-runtime` client sizes its connection pool like the
  model clients instead of botocore's default 10, which concurrent queries'
  batched counts (8 at a time each) overran (#176).
- Global search runs `processing.max_concurrency` map calls at once, as
  documented; each map call was its own BatchProcessor chunk, so the default
  chunk concurrency of 4 capped it (40 map calls of 0.5 s: 5.0 s, now
  1.0 s at the default 20) (#176).
- Neptune traversals beyond `aws.neptune.pool_size` wait on the event loop
  instead of in gremlinpython's blocking pool checkout, where each held a
  default-executor thread and the Bedrock calls LangChain runs on that
  executor queued behind them (64 concurrent traversals at pool size 4 delayed
  a 0.2 s executor call to 3.0 s; now 0.2 s). The default pool size stays 4,
  the query-thread count of the default `db.r6g.large` (#176).
- A `GraphRAGChain` dropped without `close()`/`aclose()` now releases its
  cached retrievers when it is garbage-collected; its finalizer only stopped
  the sync-API loop, so the retrievers' sockets outlived the chain (20 dropped
  chains against the local stores: +40 sockets and "Unclosed client
  session" warnings after `gc.collect()`, now +0). Closing the chain
  explicitly is still required for timely release (#176).
- `run-eval` closes its chain on exit, whether the run succeeds or fails, so
  the retrievers' Neptune/OpenSearch sockets are released instead of the
  process ending with "Unclosed client session / connector" warnings (#176).
- **Breaking (index data):** the chunk `text` field is analyzed with the
  source-language analyzer (`language_analyzers[source_language]`, else
  `default_analyzer`) instead of `standard`. An untranslated Korean corpus
  (`source_language: ko`, `target_language: ko`) had almost no BM25 recall:
  `standard` keeps "홍길동으로부터" as one token, so "홍길동" never matched.
  Re-index existing text-unit indices to pick up the new mapping (#177).
- Lexical search no longer applies `fuzziness: AUTO` to a query that contains
  Hangul, Han or Kana. One such character is a whole syllable or morpheme, so
  the one edit AUTO allows on a 3-5 character term matched a different word
  ("김철민" found "김철수"). Latin-script queries keep typo tolerance (#177).
- Entity resolution no longer fuzzy-merges Han/Hangul/Kana names whose final
  character (the head noun) differs ("가나다연구원" / "가나다연구소",
  "김철수" / "김철민"), and does merge spacing and legal-form variants
  ("가나다 상사" / "가나다상사", "(주)가나다" / "가나다", "Acme Inc" /
  "Acme"); the "주" of "(주)" no longer counts as an identifier token. Entity
  ids are unchanged (#177).
- `answer_contains` and `graph_aware` phrase matching: a Korean particle may
  follow any gold word, so `2024`, `AWS` and `Acme Corp` match `2024년에`,
  `AWS는` and `Acme Corp입니다`; spacing next to CJK letters is ignored
  (`3억 원` = `3억원`); and a CJK substring no longer matches inside a longer
  number (`2년` in `12년`, `二年` in `十二年`) (#177).
- The `retrieval` evaluator NFKC-normalizes and casefolds source names before
  matching, so a decomposed (NFD) Hangul file name from macOS or a PDF tool
  and a full-width `ＡＢＣ` match their reference sources (#177).
- Parsed document text and the `RAGInput.query` are NFC-normalized, so
  decomposed (NFD) Hangul from macOS file systems or PDF text layers matches
  composed queries, analyzer dictionaries and entity names. NFC text is
  unchanged and keeps its ids; a document that contained NFD text gets a new
  content hash and document id, so the next incremental run re-indexes it
  once (#177).
- Entity grounding (opt-in) judges Chinese and Japanese spans: their length
  counts each Han/Hangul/Kana character as a token, where the whitespace
  count made every span "too short" and so always grounded. Dense-script
  spans fall back to character-bigram overlap, so a Korean paraphrase that
  only changes particles is kept, and punctuation no longer separates
  "2년이다." from "2년이다" (#177).
- **Breaking (index data):** translations into
  `processing.translation.additional_target_languages` were produced (one LLM
  call per chunk and language) but never indexed. Each is now indexed in its
  own `translated_text_<language>` field with that language's analyzer and
  included in lexical text-unit search. Re-index text units so existing
  indices get the analyzed fields (#177).
- The size-based splitter also splits after CJK sentence terminators
  (`。．｡！？；`) before falling back to spaces and characters, so Chinese and
  Japanese text is no longer cut mid-sentence. Text without these characters
  is chunked exactly as before (#177).
- Querying a suffix nothing was ingested under (before the first ingestion,
  or a typo in `--suffix`) fails with `IndexNotFoundError`: "No indices found
  for suffix '<suffix>' ... Did you run ingestion with
  processing.document_parsing.index_value '<suffix>'?". It used to return a
  successful answer over 0 results while logging `index_not_found` warnings.
  A single missing optional index of an ingested corpus (no claims, no
  community reports) is still skipped (#178).
- `custom_prompts` overrides are checked when the config loads. An unknown
  `{variable}` (often a literal JSON brace, which raised `KeyError` on every
  call) or a missing data variable such as `{input_text}` (which ran the
  prompt on no document text) now fails the load with the offending key;
  literal braces are written `{{`/`}}`. `run-prompt-tuning` escapes braces
  in the overrides it writes (#178).
- Ingestion warns once per skipped file extension with the remedy (for
  `.md`/`.html`, the `uv sync --extra unstructured` install command) instead
  of skipping unparseable files silently, and an empty first stage reports
  "No supported source files found in <dir>" rather than telling the user to
  check a previous stage that does not exist (#178).
- `run-ingestion` (when the `indexing` stage is enabled), `run-rag` and
  `run-eval` check the Neptune and OpenSearch endpoints the run needs at
  start-up and exit with an error naming the config key and environment
  variable (`aws.neptune.endpoint`/`NEPTUNE_ENDPOINT`,
  `aws.opensearch.endpoint`/`OPENSEARCH_ENDPOINT`). A missing endpoint used
  to surface only at the indexing stage, after every paid LLM stage, or at
  query time (#178).
- `run-rag --help`, invalid arguments and a missing `--query`/`--interactive`
  no longer print an asyncio "Task exception was never retrieved ...
  SystemExit" traceback: arguments are parsed before the event loop starts,
  and the missing-mode case is a usage error (exit code 2) (#178).
- Missing, expired or invalid AWS credentials now fail model resolution with
  "No valid AWS credentials for Amazon Bedrock ..." instead of falling back to
  the bare model id, which for inference-profile-only models surfaced as a
  misleading "Enable aws.bedrock.enable_global_profile" error. A missing
  `bedrock:ListInferenceProfiles` grant still falls back (#178).
- An empty (or comment-only) `config.yaml` loads the defaults instead of
  raising a raw `TypeError`; a file whose top level is not a mapping is a
  clear `ValueError` (#178).
- Token budgets for Claude 4.7-and-later models (Opus 4.7/4.8/5/5.5, Sonnet
  5/5.5, Haiku 5.5) scale the local token estimate by 1.3: CountTokens
  rejects them, and their tokenizer counts roughly 1x-1.35x the tokens of
  the older Claude tokenizer the ~4-characters-per-token estimate matches, so
  the retrieval context budget and the RAGAS context cap were under-counted
  by up to ~30%. The factor is a new capability-record field,
  `token_estimate_multiplier` (default 1.0, overridable through
  `aws.bedrock.model_overrides`), applied by the default token counter; API
  counts are never scaled (#179).
- The LLM XML parser no longer tries LangChain's `XMLOutputParser.parse`
  first. Without `defusedxml` (not a dependency) that call raised
  `ImportError` on every response, so the strict and the two re-escaping
  attempts never ran; with `defusedxml` present in the environment, a
  well-formed single-root response (claims, gleaning refinement plan) parsed
  into lists of one-key dicts that the extractors cannot read. Every response
  now goes through the lxml recovery path, which produced all results before,
  so parsed output does not change (#165).
- The LLM XML parser keeps a bare `&` and a `<` that does not start a tag
  as text. lxml recovery ran on the unescaped response and cut them, with
  the characters after them, from single-root responses (claims, gleaning
  refinement plans): `A&B Corp` parsed as `A Corp` and `R&D budget < 5M` as
  `R budget  5M`. The escaping attempts removed in #165 never reached such a
  response, since lxml recovery came first and returned the cut text (#174).
- The LLM XML parser keeps every top-level element. lxml recovery kept only
  the first one, so a response of repeated siblings (`<line_number>4</...>`
  `<line_number>7</...>`, from a model continuing the chunking prompt's
  trailing open `<chunk_boundaries>`) parsed as `{"line_number": "4"}` and
  passed the section check, losing every later boundary. The response is now
  parsed under one synthetic root, so repeated siblings become a list and
  multi-section answers decode `&amp;` and `&lt;` like single-root ones.
  The chunking prompt no longer ends with `<?xml ...?>` and an open
  `<chunk_boundaries>`: it already asks the model to emit the whole document
  (#174).
- A model response that stopped at its output-token limit (`stopReason`
  or `stop_reason` `max_tokens`) fails with `LLMOutputTruncatedError`
  instead of being parsed. The XML parser recovered the sections before the
  cut, so an extraction cut inside `<relationships>` counted as a success
  with its relationships missing. The chain now logs a WARNING with the
  prompt, model purpose and model id, and ingestion counts the item as
  failed; the error is not retried, since the same input hits the same
  limit. Streamed output is passed through and only logged (#174).
- The output-fixing LLM is told the top-level elements its prompt asks for
  (e.g. `<entities>`, `<relationships>`) instead of "Here are the output
  tags: None" with an example that nests the tags, and an empty or
  whitespace-only answer (e.g. thinking only) fails the parse without
  calling it, since it could only invent the structure; the batch retry
  re-asks the original model. `create_robust_xml_output_parser` takes a
  required keyword `output_tags` (**Breaking** for direct callers) (#174).
- `BatchProcessor.execute_with_fallback` runs every item as its own call
  under its own `call_timeout_seconds`. The timeout wrapped a chunk's whole
  `batch()` call, so one slow item discarded the chunk's finished results
  and re-ran every item (10 items with one slow: 20 model calls instead of
  11), while the abandoned calls kept running and billing. Now only the
  timed-out or failed item is retried, with the same `max_attempts`.
  **Breaking** for direct callers: the synchronous method no longer takes
  `batch_func` (the async one still does) (#174).
- The Bedrock client's socket read timeout is 330 s, above
  `BatchProcessor`'s 300 s call timeout. Both were 300 s, so a long
  ingestion call raced them: when the socket timeout won, botocore silently
  re-sent the call, which the call timeout then abandoned. Now the call
  timeout ends such a call and the item is retried under the batch policy
  (backoff, `max_attempts`, logged). An ingestion generation still has to
  finish within the 300 s call timeout (#174).
- Claude requests no longer send the `"\n\nHuman:"` stop sequence. It is a
  marker of the legacy text-completion format; Converse and the InvokeModel
  Messages body are turn-structured, so it only ended translation,
  extraction or answers early, without an error, on text that contains it
  (chat transcripts, quoted dialogue) (#174).
- `cdk destroy` with `removal_destroy=true` (dev default) deletes the
  OpenSearch domain's app, slow-index and slow-search log groups. The domain
  created them with CDK's default `Retain`, so every dev teardown left three
  log groups behind (#166).
- Changing a `graph.analysis` setting (centrality or statistics) no longer
  invalidates the `graph_analysis`, `community_detection` and `indexing`
  stage caches. These settings only shape the stage's centrality and
  statistics, which no later stage reads, yet they were part of every later
  stage's cache key, so a tuning change regenerated every community report.
  The `graph_analysis` and `community_detection` keys change once on upgrade,
  so a resumed pipeline recomputes those stages one time (#167).
- The cross-run merge matches a delta entity or relationship to a stored one
  by id before the name or (source, target, type) key. A gleaning correction
  renames, retypes or reverses an item in place and keeps the id derived from
  its old form, so a later document naming the old form produced a second
  item with the same id: the stored entity's text units and description were
  overwritten, and Neptune got two edges with one id (#175).
- `NeptuneIndexer.read_entities` and `read_relationships` raise
  `AWSServiceError` when the read fails instead of returning `[]`. Removing a
  deleted or changed document's chunks from shared artifacts then counted a
  transient read error as "nothing to update" and dropped the document's
  registry row, leaving the shared artifacts citing removed chunks with no
  lineage to retry; it now counts as a failed removal and keeps the row. A
  failed read during the cross-run merge fails the indexing stage before any
  write instead of overwriting the stored lineage (#175).
- The translation stage honours `processing.ignore_errors`: a failed call for
  a whole target language now fails the stage when it is `false` (the
  translator swallowed every error, so the graph silently mixed languages).
  Text units left untranslated, in any target language, are reported in the
  stage's `failed_units` and through the per-stage failed text units, so an
  incremental run records their documents FAILED and retries them instead of
  recording them PROCESSED (#175).
- Neptune expansion from seed communities (`NeptuneRetriever` queried with
  the community label, e.g. through its LangChain retriever interface) gets
  the entity expansion's per-seed budget from #161. A `limit()` inside
  `repeat()` counted every traverser of the traversal and a final
  `limit(top_k * retrieval_multiplier)` followed, so the seeds filled the
  result: against Gremlin Server, 10 seed communities with 3 members each
  returned only the 10 seeds at `top_k` 10 and the neighbourhoods of 5 of
  them at `top_k` 100. Each seed now gets its members and their
  neighbourhood within its own share of the fetch width, ranked and cut like
  the entity expansion; the same probe returns all 10 seeds, 30 members and
  neighbours of all 10 (#175).
- The opt-in real-AWS ingest-then-search smoke test
  (`GRAPHRAG_TEST_RUN_INGEST=1`) could not run: it built
  `DataIngestionPipeline` without the required `pipeline_config`, called
  `run()` without the source directory and never awaited the async
  `create_rag_chain`. It now does all three, checks the pipeline did not
  fail, and closes the chain (#175).
- `indexing.reset` with the doc-status registry enabled rebuilds from the
  whole corpus and records every document again. The loading stage used to
  diff against the registry first, so the reset cleared the stores but
  indexed only new and changed documents, and the unchanged ones were lost
  (#163).
- Cross-run merge reads relationships back from Neptune with their type (the
  edge label), endpoint names, attributes and `text_unit_ids`, scoped to the
  delta's index suffix. The type and endpoint names were dropped and the
  JSON-encoded `text_unit_ids` came back as one string, so every delta run
  added a second edge with the same id next to the stored one, nested the
  lineage one JSON level deeper and reset the weight. Serialized JSON
  properties are no longer cut at `indexing.neptune.property_max_length`,
  which left long lists unparseable (#163).
- Cross-run merge reads entities back with their attributes,
  `community_ids`, rank and confidence, from the delta's index suffix only,
  and merges each suffix's delta separately. The read was not scoped by
  label, so another tenant's vertex with the same id could be merged in, and
  it dropped the attributes, so the merged entity lost its `index` suffix and
  was written into the default suffix's graph and indices. The merge now
  unions attributes (the delta wins on a shared key, as in the full build)
  (#163).
- Re-applying a delta no longer appends its entity and relationship
  descriptions again: the merge compares description lines instead of the
  whole stored multi-line description (#163).
- The registry lineage records the ids the cross-run merge kept. With
  `indexing.cross_run_fuzzy_merge`, a delta entity folded into a stored one
  was recorded under its own id, so deleting the document never removed the
  stored entity. A delta edge whose endpoint was remapped is now merged with
  the stored edge between the same entities instead of being added next to
  it. The merge moved from `IndexingManager.index_delta` (now a plain upsert)
  to `IncrementalIndexer.commit` through the new
  `IndexingManager.merge_with_existing_graph` (#163).
- Deleting or changing a document strips its text units from the entities
  and relationships it shared with surviving documents and recomputes their
  frequency and weight, in Neptune and OpenSearch. They kept citing the
  removed chunks. Their descriptions still keep the removed document's text
  until a full rebuild, since removing it needs an LLM re-summary (#163).
- An untyped relationship merges with its stored edge: Neptune stores it
  under the `RELATED_TO` label, so it read back with that type and the merge
  key treated it as a different relationship (#163).
- **Breaking** for direct callers: `IndexingManager.index_delta` no longer
  merges with the stored graph (call `merge_with_existing_graph` first, or go
  through `IncrementalIndexer.commit`), and `read_entities`/`read_relationships`
  take the index `suffix` (#163).
- A doc-status record over the DynamoDB 400 KB item limit (a document with
  roughly 10,000+ artifact ids) fails with an error naming the file before the
  registry write, instead of a bare `ValidationException`; the limit is
  documented in the user guide (#159).
- A failed removal of a deleted document's artifacts now fails the indexing
  stage (after the delta is committed) instead of only logging a warning, so
  the run status and the `IndexingFailures` alarm report it; the registry
  rows are still kept for a retry (#159).
- Incremental runs no longer record a document as PROCESSED when graph
  extraction, gleaning or claim extraction failed on any of its text units.
  It is recorded FAILED with the lineage of what was written, and the
  registry diff treats a FAILED record as changed, so the next run prunes and
  re-extracts it instead of leaving the hole in place (#159).
- A Neptune relationship whose source entity vertex is missing is counted as
  a failed write instead of a success: the add-edge traversal returns the
  edge id and an empty result is recorded as an error (#159).
- A graph-extraction answer of only empty sections
  (`<entities></entities><relationships></relationships>`) parses as a valid
  zero-entity result instead of failing, being retried, and going to the
  output-fixing model (#159).
- Graph expansion (local search, LightRAG `enable_graph_expansion`) returns
  the seed entities and a neighbourhood for every seed. The per-hop limit
  inside `repeat()` counted the whole traversal, so the first seeds used it up
  and the seeds themselves were never emitted (10 seeds with 6 neighbours
  each: 1 seed and 2 neighbourhoods came back). The fetch width is now split
  across the seeds, `max_results_per_hop` caps neighbours per node and hop,
  and a seed scores proximity 1.0 (#161).
- Global search's map key points (or reduce summary) reach the answer context
  whole: they are seated before the per-type split instead of sharing the
  `general` share, which cut 7,000 tokens of key points to about 4,200 while
  most of the window stayed unused. Points are still sized by
  `max_map_reduce_tokens` (#161).
- DRIFT fusion orders each section type by its native score. All results
  share one RRF bucket, so the rank within a type was arrival order and a
  later iteration's best item ranked below an earlier iteration's worst. The
  positions of the types in the bucket are unchanged (#161).
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
  or a clamped value (#158).
- `run-eval` no longer exits 0 when scoring fails: with
  `processing.ignore_errors: false` (the default) an evaluator error stops the
  run instead of being recorded per query, and `--max-failure-rate` now also
  applies to each metric's failed share of attempted values, so a metric that
  failed on every query exits non-zero (#158).
- `run-ingestion --verify-metadata` exits non-zero when the metadata is
  corrupt, and `--repair-metadata` when the repair fails or raises (#158).
- `run-rag` reports `success: false` and exits non-zero when the chain returns
  its `ignore_errors` error fallback (`metadata.error`) instead of an answer
  (#158).
- Importing the package no longer configures logging: it used to replace the
  host's root handlers, set the root level to INFO and open a `FileHandler` in
  `logs/` next to the installed package (failing on a read-only
  `site-packages`). Only the CLIs call `setup_logging`, and a relative
  `logging.log_file_path` is now resolved against the working directory
  (#158).
- Visualization: `graph.visualization.embeddings.bedrock_model_id` given as a
  YAML string is validated as an embedding model id at config load (it used to
  fail with `'str' object has no attribute 'value'`, which ingestion logged
  and skipped, so no visualization was written); an edgeless graph no longer
  gets NaN node sizes in the interactive view; a failed dimensionality
  reduction falls back to a seeded spring layout and sets `layout_degraded`
  instead of returning an unseeded random layout (#158).
- The evaluation `run_manifest` records `git_sha` only when the package runs
  from a checkout that tracks it (an install inside another repository used to
  report that repository's HEAD) and adds `git_dirty` (#158).
- `run-eval` output files carry a random suffix after the timestamp, so runs
  started in the same second no longer overwrite each other, and a numeric
  `answer` of `0` is kept as a ground truth instead of being treated as
  missing (#158).
- `run-prompt-tuning` fails instead of emitting a default profile when the
  profiling model returns no JSON profile, splits an `entity_types` string
  (`"PERSON, ORGANIZATION"`) into types instead of characters, and builds the
  example `GraphExtractor` from the tuner's providers (#158).
- `GraphRAGChain` keeps one set of cached retrievers and strategies per event
  loop and releases a loop's set only once that loop is closed (or on chain
  close). A query on a second loop (e.g. `ainvoke()` on the caller's loop while
  `invoke()` runs on the chain's loop thread) used to evict and close the first
  loop's clients under its in-flight queries. The caches are guarded by a lock
  (#160).
- The process-wide `MemoryManager` guards its conversation state with a
  threading lock instead of an `asyncio.Lock`, which bound to the first event
  loop that contended for it (`RuntimeError` on another loop) and did not
  exclude other threads, so turns appended from two threads could interleave
  or hang (#160).
- Context budgeting counts section tokens concurrently (up to 8 at a time,
  each distinct text once) instead of one Bedrock CountTokens call after
  another, which made token counting dominate mix/hybrid query latency. Global
  search packs its ranked key points the same way, off the event loop. Counts
  stay exact (#160).
- With `ignore_errors: false`, a fatal error (no model access, bad credentials,
  unreachable endpoint) in a global-search map call or a DRIFT query
  refinement/keyword expansion fails the query. Global search turned every such
  map call into an unrated batch and answered from the raw reports, and DRIFT
  dropped the error without a log line; non-fatal failures still degrade, now
  with a warning (#160).
- LightRAG keyword extraction checks the shape of the model's JSON: each
  keyword level must be a list of strings. A bare string was split into
  one-character keywords, `null` raised a bare `TypeError`, and a JSON array
  passed as "no keywords" even with `ignore_errors: false`; these now raise
  `LanguageModelError` (or degrade to no keywords when errors are ignored)
  (#160).

### Security
- The user guide and design doc state that metadata filters are relevance
  filters, not an access-control boundary: stores that do not declare a key
  (relationships, claims, Neptune community vertices) return unfiltered
  content. Isolate tenants with separate `suffix` namespaces (#173).
- `pipeline_id` (`--pipeline-id`, `GRAPHRAG_PIPELINE_ID`, which the Step
  Functions task sets, `PipelineConfig` and `DataIngestionPipeline.run`) must
  be a single path segment: letters, digits, `.`, `_` and `-`, starting with a
  letter or digit and without `..`. The id names the local cache directory
  and the S3 prefix, so `../x` or `a/b` reached paths outside the cache
  directory (#173).
- The persisted embedding cache (`persist_embedding_cache`) uploads with the
  `aws.s3.encryption` settings the stage-cache sync already used. It sent no
  SSE header, so `AES256` or `aws:kms` with a specific key were ignored for
  that object (#173).
- `run-rag` prints answers, errors and the verbose query/source panels as
  plain text with terminal control characters (other than newline and tab)
  removed. Model output was rendered as rich markup, so `[link=...]` became a
  terminal hyperlink, and escape sequences in it reached the terminal (#173).
  terminal hyperlink, and escape sequences in it reached the terminal (#179).
- Ingestion prompts (graph extraction, gleaning, claim extraction, description
  summarization, community reports) wrap corpus text and corpus-derived inputs
  in named tags (`<input_text>`, `<current_entities>`, `<entity_data>`, ...)
  and say once that tagged content is data whose instructions are not to be
  followed. A corpus `## heading` no longer reads as a prompt section. Template
  variable names are unchanged, so `custom_prompts` overrides keep working;
  built-in prompt text is not part of the stage cache keys (#179).
- Query-time prompts do the same for retrieved content: answer generation
  (`<context>`), context building, community relevance, global map/reduce
  (`<community_reports>`, `<summaries>`) and the DRIFT primer and query
  refinement prompts (#179).
- The CLIs log a WARNING at startup when `LANGSMITH_TRACING` or
  `LANGCHAIN_TRACING_V2` enables LangSmith tracing, which uploads prompts,
  retrieved context and model outputs. Tracing is not turned off (#156).
- Logs at INFO and above no longer carry query text, rewritten (DRIFT) queries,
  corpus entity names or raw model output; they log lengths, counts, ids and a
  short hash instead, and the text moves to DEBUG. The XML parser's exception
  message no longer embeds model output, and the per-query "no entity focus"
  WARNING is now a DEBUG record (#173).
- Require patched `unstructured>=0.24.0` for optional Markdown/HTML parsing on
  Python 3.11+ (GHSA-4mvj-m6j5-pmf7), which also drops NLTK and its model
  artifact path traversal (GHSA-8mgp-746c-j5xp); Python 3.10 keeps the core
  formats without this extra.
- Constrain the transitive `langchain-openai` to `>=1.1.14`
  (GHSA-r7w7-9xr2-qq2r).
