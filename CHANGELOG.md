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
Configuration. An unknown key fails the load (see below); a moved key fails
naming the key that replaces it.
- One switch turns incremental indexing on: `providers.doc_status.enabled`
  (default `false`; `kind` picks DynamoDB or SQLite). `aws.dynamodb.enabled`
  is a moved key: a config that sets it fails naming the replacement, so move
  `aws.dynamodb.enabled: true` to `providers.doc_status.enabled: true`
  (`UKR__PROVIDERS__DOC_STATUS__ENABLED`). The container image config does.
  The document-loading stage cache keys change, so a resumed pipeline id
  re-runs from document loading once (#344).
- Unknown config keys, in the file or as `UKR__` environment variables, are
  an error instead of a warning: a misspelled key (a guardrail identifier
  under `aws.bedrock.guardrail.identifer`) left its setting at the default
  without a word. Every CLI takes `--allow-unknown-keys` to warn and go on;
  `get_config` and `ConfigLoader` take `allow_unknown_keys=True` (#335).
- Environment overrides use one prefix and name the key they set:
  `UKR__<SECTION>__<KEY>` sets any config key (double underscores between
  path segments, any case; JSON for list and mapping settings). The former
  names are no longer read: `NEPTUNE_ENDPOINT` is
  `UKR__AWS__NEPTUNE__ENDPOINT`, `OPENSEARCH_ENDPOINT` / `_USERNAME` /
  `_PASSWORD` are `UKR__AWS__OPENSEARCH__ENDPOINT` / `__USERNAME` /
  `__PASSWORD`, `BEDROCK_REGION` is `UKR__AWS__BEDROCK__REGION_NAME`,
  `BEDROCK_GUARDRAIL_IDENTIFIER` / `_VERSION` are
  `UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER` / `__VERSION`,
  `S3_BUCKET_NAME` is `UKR__AWS__S3__BUCKET_NAME`, `LOG_LEVEL` /
  `LOG_FORMAT` / `LOG_TO_FILE` / `LOG_FILE_PATH` are `UKR__LOGGING__LEVEL` /
  `__LOG_FORMAT` / `__LOG_TO_FILE` / `__LOG_FILE_PATH`,
  `OPENSEARCH_NUMBER_OF_REPLICAS` is
  `UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS`,
  `OPENSEARCH_KOREAN_ANALYZER` is
  `UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`,
  `GRAPHRAG_DOC_STATUS_TABLE` / `_CREATE_TABLE` are
  `UKR__AWS__DYNAMODB__TABLE_NAME` / `__CREATE_TABLE_IF_MISSING`,
  `GRAPHRAG_SOURCE_SCOPE` is `UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE`,
  `GRAPHRAG_OBSERVABILITY_ENABLED` / `_BACKEND` are
  `UKR__OBSERVABILITY__ENABLED` / `__BACKEND`, and the run-ingestion
  defaults `GRAPHRAG_SOURCE_DIRECTORY` / `GRAPHRAG_PIPELINE_ID` /
  `GRAPHRAG_METRICS_ENVIRONMENT` are `UKR_SOURCE_DIRECTORY` /
  `UKR_PIPELINE_ID` / `UKR_METRICS_ENVIRONMENT`. The CDK stacks and the
  container entrypoint set the new names; redeploy the stacks with the image
  (#335).
- `AWS_REGION` and `AWS_PROFILE` no longer override a region or profile the
  config file sets: they fill `aws.region_name` / `aws.profile_name` only
  when the file leaves them unset, as an explicit setting wins over the
  environment in the AWS SDKs (a shell exporting `AWS_REGION` used to move a
  file's region silently). Use `UKR__AWS__REGION_NAME` to override the file
  (#335).
- `graph.analysis.centrality` moved to `graph.visualization.centrality`: only
  the visualization computes centrality (#335).
- Switching `--backend` (or any provider flag) rebuilds the config from what
  the file, the environment and the caller set, so a value equal to the old
  preset's that you set yourself is kept, and a change a library caller made
  to the `Config` object survives the flags (`Config.as_input`). The removed
  `strip_backend_preset` dropped both (#335).
- Every model setting moved into one top-level `models` section: the two
  tier models (`models.default`, `models.fast`), per-role overrides
  (`models.roles.<role>`, a model id or a tier name), the one embedding model
  with its dimension (`models.embedding`; indices, queries, RAGAS and the
  visualization layout all use it) and the rerank model (`models.rerank`).
  The provider of each family stays `providers.<llm|embedding|rerank>.kind`.
  A config that sets a removed key fails validation naming the key that
  replaces it (unlike other removed keys, which are only warned about); the
  CLI flags `--llm-model`, `--embedding-model` and `--embedding-dimension`
  set the new keys. A setting of a provider the config does not select
  (`providers.llm.temperature` under Bedrock, `aws.bedrock.default_effort`
  with a local model, ...) is ignored with a WARNING. With `backend: local`,
  naming another embedding model than the preset's needs its dimension too
  (the preset's 1024 was silently kept). Library callers read the models
  through `Config.model_id(ModelRole.X)` (`ModelRole` is exported from
  `unified_kg_rag`), `tier_model_id`, `model_tier`,
  `embedding_model_id`, `embedding_dimension` and `rerank_model_id` (#334).

  | Removed key | Replacement |
  |---|---|
  | `aws.bedrock.default_model_id` | `models.default` |
  | `aws.bedrock.fast_model_id` | `models.fast` |
  | `providers.llm.default_model_id` | `models.default` |
  | `providers.llm.fast_model_id` | `models.fast` |
  | `fixing.fixing_model_id` | `models.roles.fixing` |
  | `processing.chunking.chunking_model_id` | `models.roles.chunking` |
  | `processing.translation.translation_model_id` | `models.roles.translation` |
  | `processing.graph_extraction.extraction_model_id` | `models.roles.graph_extraction` |
  | `processing.graph_extraction.description_summarization.summary_model_id` | `models.roles.description_summary` |
  | `processing.gleaning.graph_refinement_model_id` | `models.roles.gleaning` |
  | `processing.claim_extraction.extraction_model_id` | `models.roles.claim_extraction` |
  | `graph.community_detection.report_generation.report_generation_model_id` | `models.roles.community_report` |
  | `search.translation_model_id` | `models.roles.query_translation` |
  | `search.entity_extraction_model_id` | `models.roles.entity_extraction` |
  | `search.strategy_selection_model_id` | `models.roles.strategy_selection` |
  | `search.query_rewrite_model_id` | `models.roles.query_rewrite` |
  | `search.answer_generation_model_id` | `models.roles.answer_generation` |
  | `search.global_search.community_relevance_model_id` | `models.roles.community_relevance` |
  | `search.global_search.map_model_id` | `models.roles.global_map` |
  | `search.global_search.map_reduce_model_id` | `models.roles.global_reduce` |
  | `search.drift_search.primer_model_id` | `models.roles.drift_primer` |
  | `search.drift_search.action_model_id` | `models.roles.drift_action` |
  | `evaluation.evaluation_model_id` | `models.roles.evaluation` |
  | `providers.embedding.model_id` | `models.embedding.model_id` |
  | `providers.embedding.dimension` | `models.embedding.dimension` |
  | `indexing.opensearch.embedding_model_id` | `models.embedding.model_id` |
  | `indexing.opensearch.embedding_dimension` | `models.embedding.dimension` |
  | `evaluation.embedding_model_id` | `models.embedding.model_id` |
  | `graph.visualization.embeddings` | `models.embedding.model_id` |
  | `providers.rerank.model_id` | `models.rerank` |
  | `search.reranking.rerank_model_id` | `models.rerank` |
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
  soon as one finishes. The default (now 40, 20 under `backend: local`) is the peak; it
  burst to 40 before but averaged about 18 in flight. RAGAS judges run `evaluation.ragas_max_workers` jobs without
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
- `indexing.opensearch.hybrid_search_pipeline_name` is removed: hybrid
  search no longer uses an OpenSearch search pipeline, so ingestion writes
  none and `search.hybrid` weights apply from the next query, not the next
  ingestion run. A pipeline an earlier run wrote is left on the domain
  unused; delete it with `DELETE _search/pipeline/<name>` if you like
  (#306).
- `run-ingestion --repair-metadata` is removed. It loaded the metadata it
  was meant to repair, so on a corrupt file it failed like
  `--verify-metadata` and never repaired anything; use `--verify-metadata`
  and, for a corrupt run, `--force-rebuild` or a new `--pipeline-id` (#248).
- `search.drift_search` follows MS GraphRAG DRIFT search (see *Changed*):
  `enable_query_refinement`, `enable_keyword_extraction`,
  `enable_llm_convergence`, `max_iterations`, `initial_top_k`,
  `summary_length`, `summary_char_limit`, `n_entities`,
  `convergence_threshold`, `improvement_threshold`, `enable_primer`,
  `primer_follow_ups` and the `query_refinement_model_id`,
  `keyword_expansion_model_id` and `convergence_assessment_model_id` models
  are removed with the iterative query-rewriting loop, and so are
  `custom_prompts.query_refinement_*` and `keyword_expansion_*`. Use
  `k_followups` (default 20), `primer_folds` (5), `n_depth` (3),
  `local_context_tokens` and `max_answer_tokens`, with the
  `models.roles.drift_primer` and `drift_action` models (both fast tier). `custom_prompts.drift_primer_*`
  overrides must emit a 0-100 integer `score`; `drift_hyde_*` and
  `drift_action_*` are new (#323).
- `search.global_search` follows MS GraphRAG global search (see *Changed*):
  `max_communities`, `map_batch_size`, `reserve_report_slots` and
  `max_text_units` are removed. Use `community_depth` (which cut of the
  community hierarchy is mapped, default 2 from the root),
  `max_reports_per_level` (default 100) and `map_context_tokens` (report
  tokens per map call, default 12000) instead. `text_unit_slots` is an
  integer, default `0` (no text units; `null` is rejected), and
  `relevance_threshold` defaults to 0.2 (was 0.5). Global search needs
  community reports indexed with their place in the hierarchy: re-run
  ingestion once with `indexing.reset: true`; until then global search
  raises `ConfigurationError` instead of answering from nothing (#318).

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
  question (`models.roles.query_rewrite`, fast tier) and every step reads
  the rewrite (#212).
- `aws.region_name` defaults to `us-west-2`, and `aws.bedrock.region_name`
  defaults to `null` (follow `aws.region_name`); set it to keep Bedrock in
  another region (#148).
- Default models: Claude Sonnet 5.5 (`models.default`, #117) and Claude
  Haiku 5.5 (`models.fast`; set `anthropic.claude-haiku-4-5-20251001-v1:0`
  to keep Haiku 4.5) (#170).
- The Bedrock guardrail applies to the query path only
  (`aws.bedrock.guardrail.apply_to: all` restores it) (#104).
- The Bedrock guardrail screens the raw user query and the final answer with
  `ApplyGuardrail` (`aws.bedrock.guardrail.check_input` / `check_output`,
  both on) instead of being attached to every query-path model call. Attached,
  it evaluated whole prompts: on Converse the retrieved corpus and developer
  instructions were judged as user input (false prompt-attack blocks, cost
  growing with context × calls), and on InvokeModel the prompt-attack filter
  did nothing without input tags. A blocked query now gets the blocked
  message without retrieval (`RAGOutput.metadata.blocked`), a masked one
  continues with the masked text, interventions are listed in
  `metadata.guardrail`, and a guardrail error fails the query. With
  `check_output`, `stream` / `astream` deliver the answer as one chunk after
  the check. `apply_to: all` now adds the guardrail to ingestion and
  evaluation calls only. Upgrade: a role outside the CDK stack needs
  `bedrock:ApplyGuardrail` on the guardrail ARN; caches and indexes keep
  their keys (#319).
- `graph.community_detection.auto_resolution` defaults to `false` (#141).
- The `retrieval` evaluator's k counts distinct files, not sources: the
  reported sources are flattened into their files in rank order (a graph
  source's lineage files most-supported first). A community report counted as
  one source but matched every file of its community, so five random reports
  scored hit@5 0.98 on a 50-file corpus against 0.19 for five random text
  units; now both score 0.19. `hit_at_k`, `recall_at_k` and `mrr` drop for
  `global`, `drift` and `mix` (and for any source naming several files);
  scores from earlier runs are not comparable (#315).
- `run-eval --search-strategy` parses to a list (`args.search_strategy`
  is `["auto"]` by default); with several strategies or `--repeats` the
  listed strategy overrides a dataset item's own `search_strategy` (#324).
- The run manifest's `config_sha256` (the full resolved config, so it
  differed between environments) is replaced by `retrieval_config_sha256`
  over the settings that shape answers and scores only; compare new runs by
  the new key (#321).
- `evaluation.max_context_tokens` defaults to `null`: the RAGAS judges get
  the answer budget (`search.token_manager.max_context_tokens`, 30000)
  instead of 8192 tokens, so `faithfulness` and `context_recall` judge an
  answer against the context it was given. Judge input tokens rise up to
  about 3.7x; set `8192` to keep the old cap (it now logs a warning) (#315).
- Entity and relationship `rank` are computed from node degree (#214), and
  relationship weight sums the extracted strengths per text unit (#189).

Index data. Rebuild existing indexes once with `indexing.reset: true` (or a
full re-index of the affected index type).
- The relationship and claim indexes carry their documents' filters
  (`attr_<key>`, `attributes.filters`), so OpenSearch applies an attribute
  filter to them itself. An index built before that returns no
  relationship or claim under an attribute filter (fail-closed) until it is
  rebuilt with `indexing.reset: true`; queries without attribute filters
  are unaffected (#360).
- The doc-status registry has a new format and table. A record lists only
  its document's text units (the per-artifact id lists and the lineage
  overflow items are gone), and records are keyed by index namespace: the
  DynamoDB table's partition key is `namespace` and its sort key `doc_id`,
  so a run reads its namespace with one Query instead of scanning the
  table, and the SQLite registry uses a new `documents` table. The default
  table names are `unified-kg-rag-on-aws-doc-status-v2` (library) and
  `graphrag-doc-status-v2` / `<env>-graphrag-doc-status-v2` (CDK), and a
  table of the earlier key schema is rejected at first use. Text-unit ids
  now derive from the registry key (source scope included) and entity
  vertices store their compact name key (`name_key`). To upgrade: deploy
  the CDK stack (it creates the new table; the old one is retained outside
  `dev`) or point `aws.dynamodb.table_name` at a new table, run once with
  `indexing.reset: true`, then delete the old table (#327).
- Incremental runs derive the content of what they touch from the text units'
  extraction records, which text units now name
  (`attributes.extraction_records`), and resolved entities record the name
  keys they were resolved from (`attributes.entity_keys`). Items indexed
  before keep their stored content (a WARNING counts them) until
  `indexing.reset: true` rebuilds the index (#325).
- Graph extraction and gleaning write one record per text unit (its own
  entities, relationships and rejected names) and graph resolution merges
  the records: `PipelineContext.unit_extractions` replaces `entities`,
  `relationships` and `rejected_entities`, `GraphExtractor.extract_units`
  replaces `extract_from_text_units`, `GraphGleaner.glean` replaces
  `glean_graph`, and `RejectedEntity` and `GleaningIndex` are removed. A
  gleaning correction now changes only what the correcting text unit says,
  and a renamed entity takes the id of its new name (it kept the old
  name's), so a graph indexed by an earlier build differs from what a
  rebuild produces: run once with `indexing.reset: true`. Stage caches of
  graph extraction and gleaning written by an earlier build are recomputed
  (#322).
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

Command line. The CLIs share one convention (user guide §10, *CLI exit
codes and JSON output*); renamed flags have no alias, so update scripts:
- Renamed flags: `run-ingestion --target-directory` → `--output-directory`;
  `run-eval --eval-data-path` → `--data-path` and `--outputs-directory` →
  `--output-directory`; `run-visualization --output-dir` →
  `--output-directory`; `run-prompt-tuning --output` → `--output-path` (the
  `--source-dir` alias is gone; use `--source-directory`);
  `prepare-benchmark --output-dir` → `--output-directory` and `--source` →
  `--data-path`. The run manifest's `cli_args` follow the new names
  (`data_path`, `output_directory`) (#333).
- Flags must be spelled out: an abbreviation such as `--output-dir` or
  `--source-dir` is rejected (exit 2) instead of matching a longer flag
  (#333).
- Exit codes: `2` for every usage or configuration error before any work,
  `1` for a failed run, `130` for Ctrl-C and `143` for SIGTERM. `run-ingestion`
  exited 1 for a missing or invalid `--source-directory`, an
  `--output-directory` equal to the source, `--s3-sync` without a bucket,
  `--resume-from-stage` or `--verify-metadata` without `--pipeline-id` and
  invalid `--enabled-stages`; every CLI exited 1 for an unset store endpoint
  and `run-eval` for a malformed `s3://` URI. `run-visualization` and
  `prepare-benchmark` exit 2 for a missing input file, `run-prompt-tuning`
  for a missing source directory. The CLIs other than `run-ingestion` now
  exit 143 on SIGTERM; as PID 1 in a container they ignored it until killed
  (#333).
- Every CLI takes `--output-format json` (one JSON result on stdout, the rest
  on stderr); `run-ingestion` rejects it with `--metrics-sink cloudwatch`
  (#333).

Deployment (CDK stack).
- The ingestion state machine runs one Fargate task (state `Ingest`, over
  every stage) instead of four phases (`PrepPhase`, `GraphBuildPhase`,
  `AnalysisPhase`, `IndexPhase`). Scripts that read `.States.PrepPhase` (to
  find the cluster, subnets and security groups) read `.States.Ingest`; the
  task's result is under `$.task` (was `$.last_phase`). A pipeline id
  checkpointed by the phased version resumes from the same S3 checkpoints:
  start a new execution with the same input (#328).
- The pipeline EMF metrics carry an `Environment` dimension (the stack's
  `env_name`, injected as `UKR_METRICS_ENVIRONMENT`), and the stack's
  alarms and dashboard read only that series. An image built before this
  change emits the series without dimensions, which the new alarms do not
  read: deploy the image and the stack together (#330).

Library API (Python callers and custom backends).
- A custom doc-status registry keeps a scope's extraction settings
  (`DocStatusPort.get_extraction_settings` / `put_extraction_settings`,
  never listed as a document). `cache_keys.extraction_settings` returns an
  `ExtractionSettings` record; its `fingerprint` is the former
  `extraction_fingerprint` value (#332).
- A custom doc-status registry implements the namespace write lease
  (`DocStatusPort.try_acquire_lease`, `lease_holder`, `release_lease`) and
  the new `DocStatusRecord` fields (`attempt_count`, `attempted_hash`) and
  status (`DocStatus.QUARANTINED`, unchanged while its content is).
  `detect_delta` takes a `max_attempts`. `IndexingManager.clear_all_data` is
  removed (a reset no longer clears the stores first), and a custom graph
  backend can rebuild in place through `GraphIndexer.begin_rebuild` /
  `end_rebuild` (the default still clears each label first) (#331).
- A custom doc-status registry also stores the `DocStatusRecord` fields
  `attempt_batch` and `crash_batch`. `detect_delta` may return
  `DocumentDelta.deferred` documents (neither processed nor deleted), and
  `IncrementalIndexer.write_ahead` no longer takes fingerprints: a run opens
  its attempts with `incremental.open_attempts`. Its `diff` should list the
  unchanged `QUARANTINED` records in `DocumentDelta.quarantined_unchanged`,
  or a raised `indexing.max_document_attempts` does not retry them (#346).
- Conversation memory is per chain: a `GraphRAGChain` builds its own
  `MemoryManager` instead of sharing one process-wide manager, and a
  conversation is keyed by index suffix, `user_id` and `conversation_id`
  (`ConversationKey`), not by `conversation_id` alone. Every default chain
  in a process shared one memory, so a tenant or user who reused another's
  conversation id got that history rewritten into their query. To keep
  history across chains built per request, pass them one manager
  (`GraphRAGChain(memory_manager=shared)`). `get_memory_manager` is removed,
  and `MemoryManager.load_history` / `add_turn` / `get_or_create_memory`
  take a `ConversationKey` (#333).
- The public API is what `unified_kg_rag` exports (`unified_kg_rag.__all__`):
  `Config`, `get_config`, `DataIngestionPipeline`, `IndexingManager`,
  `GraphRAGChain` with `RAGInput` / `RAGOutput` / `SearchOutput`,
  `EvaluationManager`, and the extension surface (the ports, `Providers`,
  `register_provider`, `BaseSearchStrategy` / `register_strategy`,
  `ParserFactory`, `register_renderer`). Import from the package root;
  other module paths are internal and may move without a Breaking entry.
  Retrieval-only queries are `GraphRAGChain.search()` / `asearch()`, which
  return a typed `SearchOutput` (`search_results`, `processed_query`,
  `metadata`) instead of a dict; `ChainMode` and `GraphRAGChain(mode=)` are
  removed, so `invoke`/`ainvoke` always answer (they returned a dict in
  `SEARCH` mode) and `stream` no longer yields nothing in `SEARCH` mode.
  Replace `GraphRAGChain(cfg, mode=ChainMode.SEARCH).ainvoke(q)` with
  `GraphRAGChain(cfg).asearch(q)` and read attributes
  (`out.search_results.results`) instead of keys (#333).
- Incremental removal no longer plans from registry lineage.
  `DocStatusRecord` and `DocumentLineage` keep only `text_unit_ids`
  (`DocumentLineage` rejects the removed id fields),
  `build_document_lineage(documents, text_units, suffix)` takes no artifact
  lists, `IncrementalIndexer.record` takes the artifacts' owners
  (`artifact_owners`), and `DocStatusPort` drops `add_lineage_overflow`,
  `get_lineage_overflow` and `delete_lineage_overflow` (and
  `DocumentDelta.orphan_overflow`) for `list_namespace(namespace)`. The
  legacy-key adoption is removed with `legacy_doc_id`,
  `adopt_legacy_records` and the `legacy_doc_ids` argument of
  `detect_delta`. A custom graph backend implements
  `GraphIndexer.find_ids_citing` (and optionally
  `find_entity_ids_by_name_key`), a custom vector backend
  `VectorIndexer.find_documents` and `delete_extraction_records`; without
  them a removal cannot find what a removed document stated (a WARNING
  says so) (#327).
- Adapter modules are grouped by role: `adapters/aws/` and `adapters/local/`
  hold technology bindings (clients, embedded stores, model factories,
  caches); `storage/`, `retrievers/` and the new `doc_status/` hold the port
  implementations. `adapters.aws.dynamodb` → `adapters.doc_status.dynamodb`;
  `adapters.local.sqlite_doc_status` → `adapters.doc_status.sqlite`;
  `SQLiteGraphIndexer`: `adapters.local.sqlite_graph` →
  `adapters.storage.sqlite_graph_indexer` (`SQLiteGraphStore` stays in
  `adapters.local.sqlite_graph`). The old paths are not kept (#312).
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
    instead of failing later in discovery (#303).
  - `GraphIndexer` / `VectorIndexer` `initialize()` / `clear()` and
    `IndexingManager.initialize()` / `clear_all_data()` return `None` and
    raise the store's typed error instead of returning `False`; a caller that
    tested the returned value now always sees `None` (#294).
  - `BatchProcessor` rejects unknown settings with a `ValidationError`:
    `batch_size=` and `chunk_concurrency=` were silently ignored after #296
    (#303).
  - `compute_hash(algorithm=)` raises `TypeError`; it always hashes with
    SHA-256 (#300).
- `SearchEngineClient.put_search_pipeline` and
  `OpenSearchClient.put_search_pipeline` are removed, and
  `OpenSearchIndexer.initialize` writes nothing: a custom search engine
  need not implement search pipelines or the `hybrid` query (#306).
- `SQLiteSearchEngine` no longer answers a `hybrid` query or takes
  `search(search_pipeline=)`: both now raise `StoreRequestRejectedError`,
  as for any unsupported request. Nothing in the package sent them after
  #306; a database file an earlier version wrote keeps its unused
  `pipelines` table harmlessly (#309).
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
- `BatchProcessor.aexecute_with_fallback` takes no `batch_func` either:
  each item is one awaited `sequential_func` call (`Runnable.ainvoke`), as
  on the sync path. A one-input `abatch(..., return_exceptions=True)` first
  call was the same `ainvoke` with the error returned in place; a caller
  still passing `batch_func=` gets a `TypeError` (#301).
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
Nothing an earlier build stored is read: rebuild every index once with
`indexing.reset: true` (registry on) in the new registry table (see
Breaking changes), and run pipelines that resume a cached
`--pipeline-id` with `--force-rebuild` or a new id (#327, #331). The
compatibility code for earlier builds is removed (#331): an index without
recorded embedding settings (`_meta`) is refused instead of given its
dimension, a stage-cache entry holding an absolute path is a miss, a
parsed-document JSON's `content.text` is no longer read as its text (write
`page_content`), the SQLite graph no longer drops the edge indexes of an
earlier schema, and a custom registry's `diff` must take
`extraction_fingerprint`. Every stage cache key and extraction fingerprint
changed once with the full-config keys (#332).
- Hidden files and files under a hidden or `__pycache__` subdirectory of the
  corpus (any path component below the root starting with `.`, such as
  `docs/.drafts/`) are not parsed (#213).
- Roles with a custom IAM policy need `dynamodb:BatchGetItem`,
  `dynamodb:BatchWriteItem` (#191, #194), `dynamodb:Query` and
  `dynamodb:DescribeTable` (#327) on the registry table, and
  `dynamodb:UpdateTimeToLive` where the library creates the table (#331).
- A `backend: local` config of your own that relied on the preset for the
  stores gets the embedded SQLite stores (#273): add
  `providers.graph.kind: gremlin` and `providers.vector.kind: opensearch` to
  keep the containers.

### Added
- `docs/cli.md`, a CLI reference generated from the argparse parsers of
  every CLI (`scripts/gen_cli_reference.py`; a unit test fails when it is
  stale). The flag help strings now carry what the user guide's flag tables
  said, such as the `--pipeline-id` character rules and the
  `--continue-on-error` FAILED records (#351).
- The community drift metrics of an incremental run report `added_roots`,
  the root communities it appended: each joins every cut of the hierarchy,
  so global search maps that many more reports until a rebuild (a
  synthetic 12-topic corpus grew from 12 to 60 reports over four delta
  runs). The cut stays bounded by `max_reports_per_level` and the fall-back
  to a shallower cut (#347).
- `docs/configuration.md`, a configuration reference generated from the
  config models: every key with its type, constraints, default and
  description, and the moved keys. `scripts/gen_config_reference.py`
  writes it, and a unit test fails when it is stale, so it cannot drift
  from the code. The user guide's configuration section keeps how the
  configuration is loaded, the settings most runs change and the topics
  that need prose, and links to it; the field descriptions were rewritten
  for users (#337).
- The doc-status registry keeps each corpus's extraction settings field by
  field next to the fingerprint on its documents. An incremental run whose
  settings changed names them in its WARNING, and a setting a newer release
  adds is no change while it holds its default, so upgrading does not by
  itself re-extract an incrementally indexed corpus. A resumed run names the
  changed settings the same way, and the index build record (`run-eval`
  manifests) carries them as `extraction_settings` (#332).
- One writing run per index namespace: with the doc-status registry, a
  pipeline run holds its namespace's write lease from its first stage to its
  end (a conditional write in DynamoDB, a transaction in SQLite), renewed by
  a heartbeat every sixth of `indexing.lease_seconds` (default 300). A second
  run stops with `NamespaceLeaseError` naming the holder before it reads or
  writes anything; a run that dies stops renewing, and its lease expires
  (DynamoDB drops it by TTL: the CDK table and a table the library creates
  enable TTL on `lease_expires_at`). A run that lost its lease stops before
  its next registry or store write (#331).
- Documents whose indexing keeps dying are quarantined: the write-ahead
  record counts the consecutive runs that started a document's content and
  never finished it, and once `indexing.max_document_attempts` (default 3)
  is reached the next run removes what the attempts wrote, records the
  document `QUARANTINED` and skips it until its content or the extraction
  settings change. The `total_documents_quarantined` metric and the CDK
  `DocumentsQuarantined` alarm report it (#331).
- Paired strategy comparison: `run-eval --search-strategy local,mix,global
  --repeats N` answers the same queries with each strategy and writes
  `evaluation_comparison_*.json` with per-query deltas against the first
  strategy, seeded paired bootstrap 95% CIs, win/tie/loss counts, the
  run-to-run spread and the strategies `auto` routed to;
  `--pairwise-judge` adds the GraphRAG/LightRAG head-to-head LLM judge
  (comprehensiveness, diversity, empowerment, directness; both orders,
  order-stable verdicts only; `custom_prompts.pairwise_comparison_*`) (#324).
- `prepare-benchmark musique` turns MuSiQue (answerable, CC BY 4.0) into a
  corpus and a `run-eval` dataset: the dev split is downloaded at run time
  from a pinned mirror revision and checked by SHA-256 (or read from a local
  official file with `--source`); no benchmark data is in the repository.
  `evaluation.answer_normalization: official_squad` scores EM/F1 with the
  official SQuAD script, for numbers comparable with published results
  (#324).
- `ingestion_ephemeral_storage_gib` (CDK context, 21-200, default 40) sizes
  the ingestion task's disk, which holds the synced corpus, the parsed
  documents and the stage cache (#328).
- CDK alarms `BedrockThrottling` (at least 100 throttled Bedrock calls in
  each of 3 consecutive 5-minute periods, summed over the account and
  region's models) and `IngestionTaskMemory` (the ingestion task at 90% of
  its memory for 3 minutes, from Container Insights), and dashboard widgets
  for stage durations, the task's memory and CPU, and Bedrock invocations,
  throttles and latency by model (#330).
- Index build record: ingestion records in the vector index mappings'
  `_meta` which run last wrote each suffix (`pipeline_id`, `mode`,
  `built_at`, `documents`, `extraction_fingerprint`, `package_version`), and
  the `run-eval` manifest copies it under `index` per queried suffix, so a
  score can be traced to the index it was measured on (#321).
- An LLM response cache for ingestion (`cache.llm_responses`, on by
  default): the stages' model responses are kept with the run's stage cache
  (and synced to S3 with `--s3-sync`, a segment at a time as they fill), so a
  re-run with the same pipeline id after a stop, timeout or crash repays only
  the model calls the interrupted stage had not made. In a synthetic run
  stopped halfway through graph extraction, the resume made 20 extraction
  calls instead of 42. Each cached response is served at most once per run,
  truncated responses are not kept, and `--force-rebuild` serves none. A
  re-run stage with unchanged prompts (`--resume-from-stage`) now answers from
  the cache; with a temperature above 0 it replays the earlier sample (#326).
- Evaluation: `grouped_statistics.source_type` reports the retrieval metrics
  per source section type (its `text` row scores only the text units the
  answer model read); `graph_aware` adds `retrieved_entity_coverage` /
  `retrieved_relationship_coverage` over the entity and relationship sources
  the answer model was given, and `answer_length`, because answer-text
  coverage rises with answer length; results record
  `retrieved_source_types`, `retrieved_entities` and
  `retrieved_relationships`; the run manifest records
  `models.judge_is_answer_model` (LLM judges favour their own outputs) (#315).
- IaC: `-c neptune_serverless_ncu=MIN-MAX` (e.g. `1-16`) runs the Neptune
  instances as Neptune Serverless, scaling each between the NCU bounds, for
  environments that sit idle most of the day; unset keeps the provisioned
  `neptune_instance`. The app is unchanged (same endpoints, IAM auth and
  Gremlin API) (#319).
- Incremental runs report community drift: entities record their level-0
  communities, and a delta run's indexing stage counts the entities an
  earlier clustering already placed and the earlier communities whose
  reports went stale. At `indexing.community_drift_threshold` (default 0.5)
  of the run's clustered entities it logs a WARNING recommending
  `indexing.reset: true` and emits `community_rebuild_recommended`
  (with `community_drift_overlap_share`). Entities indexed before record no
  community; re-index once with `indexing.reset: true` to start counting
  from a full build (#320).
- The extraction store: each text unit's extraction record (entities,
  relationships, claims) is kept in the vector store's extraction-records
  index (`indexing.opensearch.extraction_records_index_prefix`), keyed by a
  hash of the unit's text and a fingerprint of the settings that shape it
  (models, extraction, gleaning and claim settings, target language, the
  prompt templates in effect, built-in ones included). A chunk with a stored
  record is not sent to the model again: an edited document re-extracts only
  its changed chunks, and a full rebuild, a reset or another document with
  the same chunk reuses the records; `--force-rebuild` re-extracts every
  chunk and rewrites them. The index is only added to (#322).
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
  build (exact kNN, Lucene BM25, CJK bigram analyzers) for about 100,000
  documents per index; and
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
- `indexing.lease_seconds` defaults to 600 s (was 300 s), three times the
  stores' 180 s request timeout; a shorter lease logs a WARNING at the start
  of the run. A run that dies now blocks its namespace for up to 10 minutes
  (#355).
- The ingestion EMF metrics add `total_records_not_stored` and
  `underived_items`, and the CDK dashboard plots them with
  `total_documents_deferred` and `community_rebuild_recommended` (#355).
- The user guide links the CLI reference instead of repeating flag tables,
  the observability trace shape (every span and attribute, the vendor
  attribute names, the known limits) moved to the runbook, and the user
  guide, runbook and design document were condensed (#351).
- `iac/README.md` opens with a quick start, splits the 35-row `-c` option
  table by concern, turns the guardrail notes into a section, and now holds
  the whole CloudWatch tracing recipe (Transaction Search, the collector
  pull, where traces appear, cost, the collector configuration without the
  CDK stacks, Bedrock invocation logging), which the runbook links to. The
  README is shorter and names `providers.doc_status.enabled` as the
  incremental switch (#349).
  incremental switch (#351).
- Global search's map prompt heads each report with its ordinal in the batch
  (`--- Report (id: 1) ---`) instead of its community UUID, and maps the
  points' `report_ids` back to the reports: short ids cost fewer prompt
  tokens and a model copies them back reliably. A custom `GlobalMapPrompt`
  keeps working: the ids come in the `{reports}` variable, as before (#354).
- `run-ingestion --s3-sync` takes the bucket from `aws.s3.bucket_name`
  (`UKR__AWS__S3__BUCKET_NAME`, which the CDK task sets) when
  `--s3-bucket-name` is not given, so the bucket is configured once (#344).
- `graph.visualization.layout`, `interactive` and `static` are typed
  sections instead of free-form maps: a misspelled renderer setting is an
  unknown key (an error by default) rather than silently ignored, values are
  validated, and setting one key no longer drops the others' defaults (#344).
- The DynamoDB, SQLite and in-memory doc-status registries classify a
  corpus with one rule, `DocumentDelta.classify` (with `settings_changed`
  shared with the delta detector), instead of a copy each (#344).
- `auto` no longer routes to `drift` by default: `search.auto_routable_strategies`
  is `local`, `mix`, `global`. DRIFT costs up to 27 LLM calls a query
  (local: 2), a price one misrouted question should not pay; add `drift` to
  the list to route to it. With one routable strategy left, `auto` uses it
  without a router call. The router guide sends questions that chain facts
  from a named entity ("where was the founder of X born") to `local`
  instead of `mix`: on the synthetic multi-hop harness `local` answers them
  (exact match 0.925) and `mix` does not (0.40) (#347).
- A connected component of fewer than
  `graph.community_detection.min_community_size` entities (an isolated
  entity, a pair linked only to each other) gets no community or report.
  Each became a root community of its own: a report LLM call restating one
  entity, and a report in every cut of the hierarchy, which pushed global
  search's cut past its cap on medium corpora (on a synthetic graph the
  depth-2 cut went from 301 to 150 reports at 10,000 entities and from 738
  to 96 at 50,000, with 8-10% fewer reports to generate; the 1.3-1.6% of
  the entities in such components are left out of communities). Leiden
  still partitions the whole graph, so every other community stays as it
  was. Their entities stay in local search. `min_community_size: 1` clusters every
  component. Existing indexes keep their one-entity reports until a
  rebuild (`indexing.reset: true`) (#347).
- A global-search cut larger than `search.global_search.max_reports_per_level`
  gives way to the deepest shallower cut that fits, which covers the same
  entities in fewer, broader reports (`global_cut_depth` in the search
  metadata); only when even the root cut does not fit are the reports most
  similar to the query mapped (`global_level_capped`). A cut of exactly the
  cap is no longer taken for a truncated one when
  `indexing.opensearch.max_query_size` is larger than the cap (#347).
- DRIFT costs less per query: the follow-up local searches no longer rerank
  (the evidence of all of them is reranked once, at the final fusion: one
  rerank call a query instead of up to 61), the defaults are
  `search.drift_search.k_followups: 10` and `n_depth: 2` (at most 27 LLM
  calls a query; MS GraphRAG's 20 and 3 made up to 67). On the synthetic
  quality harness DRIFT keeps exact match 1.000 with 4.4 to 4.5 actions a
  query instead of 7.3 to 7.9 (16.9 instead of 46.8 when every answer
  raises five follow-ups) (#347).
- Query-time fan-out has its own knob, `search.max_concurrency` (default 10,
  per query): the global map calls, the dynamic-selection ratings and the
  DRIFT primer calls and follow-up searches no longer use the ingestion
  setting `processing.max_concurrency` (default 40) (#347).
- The design doc is about half as long: each section opens with a summary,
  the ingestion and incremental sections are split into short subsections,
  the merge rules and the retrieval error policy are tables, the
  configuration internals (loading, model selection, stage cache keys) moved
  to the configuration section, and implementation details the code and the
  configuration reference already state were dropped (#341).
- The user guide opens each section with what it does and the command to
  run, keeps one owner per topic (failed, quarantined and concurrent runs
  link to the runbook; IAM permissions moved to installation), gives
  filters a section of their own, presents the evaluators as a table with
  a short section each, and splits the densest paragraphs into lists. It
  also no longer advises raising `indexing.neptune.index_concurrency` for
  large corpora, which the key's description warns against (#342).
- The operator runbook is organized by task: a task index at the top,
  routine operations, what to do when a run fails, the alarms in a section
  of their own, observability, and the common errors grouped by area
  instead of one 43-row table (#339).
- The user guide, the runbook and the design doc describe the current
  release only; what a pre-release build left behind (registry tables keyed
  by `doc_id` alone, indices without recorded settings) is covered here
  under Upgrading from pre-release builds, and the entries above name the
  current model keys (#338).
- Incremental indexing is cheaper on large corpora and hub entities:
  derivation matches stored entities to the resolved ones through a key
  index instead of a scan (3,000 records: 147 s to 13 s), a first build with
  the registry on skips it (nothing is stored), statements of one item are
  merged once instead of pairwise (8,000 statements of a hub: 8.8 s to
  3 ms), a description is summarized once per text (graph resolution's
  summaries are reused at indexing: a first build makes half the summary
  calls), a stored summary that only grew is summarized from the stored
  summary and the new description, and a stored edge whose rank alone
  changed is updated in place instead of embedded again (a hub's new
  neighbour: 175 to 43 embedded texts) (#348).
- Derivation orders a hub's text units in linear time: the set of its
  stored units was rebuilt for every unit (40,000 units: 98 s to 18 ms)
  (#352).
- The indexing stage computes the extraction-record fingerprint once per
  record kind instead of once per text unit (about 1.5 ms each; 10,000
  units: 32 s to 0.16 s for the record keys). The keys are unchanged
  (#352).
- The incremental lookups read only the fields they use: the text units'
  `attributes` (their extraction-record keys), the `text_unit_ids` of the
  claims and reports citing removed text units, and no field of a text unit
  probed for its id; they fetched whole documents, a text unit's text
  included. The embedded search engine accepts `_source.includes` (#348).
- The indexing stage reports `underived_items`, the stored items an
  incremental run kept as they were because a text unit they cite names no
  stored extraction record (#348).
- `indexing.reset` no longer takes the suffix offline. The stores are not
  cleared first: the vector indices are rebuilt into new indices and swap
  their aliases, as before, and the graph (Neptune, the embedded SQLite
  graph) is rewritten in place, every vertex and edge the rebuild writes
  tagged with the build, after which what it did not write is dropped.
  Queries keep reading the previous content until it is replaced. The
  registry keeps its records until the rebuild recorded the corpus, so a
  failed reset leaves the previous graph and records in place for the retry
  (#331).
- Stage cache keys cover the whole configuration: `cache_keys.CONFIG_KEYING`
  classifies every field as keyed from the first stage whose output it
  shapes or unkeyed with a reason, a test fails on a field left
  unclassified, and an unclassified field keys every stage. The local model
  knobs `providers.llm.temperature`, `max_output_tokens` and
  `context_window_size` (and the registry, retry and retirement settings
  that decide an incremental delta) now key the stages they shape, so
  changing them no longer resumes from output the old values produced. The
  package version keys every stage, so a release that changes a built-in
  prompt template no longer resumes from the previous one's output. Settings
  of the inactive model provider no longer key anything (#332).
- An incremental run finds what a changed or deleted document contributed
  in the stores, by the text units every artifact cites: the graph's
  entities and communities citing them, their edges, and the vector
  store's entities, relationships, claims and community reports. An item
  citing only removed text units is deleted, one citing others too is
  stripped and derived again, and a community or report goes once none of
  its text units is stored. The work no longer grows with the registry: a
  run reads its namespace's records and the cross-run merge looks the
  stored entities a delta name can join up by compact name key instead of
  reading every stored name (with `processing.fuzzy_entity_resolution` off,
  the default) (#327).
- An entity, relationship or claim takes the id a full build gives it when
  derivation changes its name: an entity whose most-supported spelling has
  another id moves to that id with its edges and claims (claims stating
  the same fact merge), and the old ids are deleted once the new ones are
  written. A stored item that a full build merges with it (an edge of
  another spelling between the same entities and of the same type, or the
  copy an interrupted re-key left under the new id) is merged with it,
  never overwritten. Stored
  communities and reports keep listing the old entity id until a rebuild
  (#327).
- Extraction records only removed text units named are deleted after the
  run's delta is written, so the extraction store no longer grows with
  every edit and deletion (#327).
- The CDK ingestion runs as one task per execution. The four phases ran the
  same task definition, image, size and timeout, so they bought nothing a
  single task lacks now that stages are checkpointed to S3 as they finish
  and answered model calls are cached; they cost four container starts,
  four corpus downloads and hashes and four checkpoint downloads, and a
  corpus or image changed mid-execution reached only the later phases. The
  state machine runs the task definition revision deployed with it rather
  than the family's latest active revision (#328).
- Ingestion metrics are emitted as each stage finishes instead of once at
  the end, so a run that is stopped or times out still reports the stages it
  completed; the run's own metrics (duration, cache hit rate) still come at
  the end. A count several stages produce (entities after extraction and
  gleaning) is emitted after the last of them the run has enabled (#330).
- `graph.community_detection.max_cluster_size` also caps a parent's child
  communities: a coarser-level community with more children is partitioned
  again on the cluster graph, so a parent report's roll-up covers every
  child summary. On the synthetic 50,000-entity graph the largest parent goes
  from 172 children to 20 (median 98 to 8), with 8,426 communities instead
  of 7,580 and 5 levels instead of 4. Re-index with `indexing.reset: true`
  to rebuild an existing hierarchy (#320).
- The records of a build are merged in text-unit id order, so the display
  spelling and type ties a merge keeps do not depend on which documents
  were extracted together. Removing a document's chunks from shared items
  re-embeds the ones whose text changed; the others still update lineage,
  weight and now rank in place (#325).
- Level-0 communities are bounded by the new
  `graph.community_detection.max_cluster_size` (default 20): a larger Leiden
  community is partitioned again on its own subgraph until it fits, and the
  coarser levels regroup the pieces. One modularity pass grew leaves with the
  corpus (680 entities at 50,000 on a synthetic graph, where a report sees
  at most 50), so most entities never reached a leaf report. On that graph
  the hierarchy goes from 797 to 7,580 communities (more report calls) and
  from 3 to 4 levels; every leaf has at most 20 entities. The community
  stage cache re-keys by itself; re-index once with `indexing.reset: true`
  to replace an existing index's communities (incremental runs cluster only
  their delta). Set a large `max_cluster_size` to keep the previous
  partition (#314).
- Hybrid search ranks the same whatever the number of hits asked for: the
  lexical and kNN searches of an index each fetch at least 50 hits, are
  min-max normalized over those and then cut to the requested size. They
  were normalized over only the returned hits, so the top 4 of a 4-hit and
  a 40-hit query shared 75% of their hits; in the reviewers' simulation
  precision@4 at 4 hits rises from 0.51 to 0.71. Small hybrid queries fetch
  more hits per search (#317).
- The answer context no longer leaves budget unused while sections are cut:
  a share a type cannot fill (all of its candidates fit) goes to the types
  that still have candidates, by share, and LightRAG `mix`/`hybrid` give it
  to chunks first, as upstream LightRAG does. Per-type shares were hard
  caps, which left 27-34% of the 30,000-token window empty in typical
  `local` and `mix` queries; those now use over 99% of it (#317).
- Fusion reranks before it diversifies: fuse, rerank, MMR diversity
  filtering, then the per-type quota (or `top_k`) cut. MMR used to cut each
  type to its quota on fused scores first, so 16-50% of the reranker's top
  20 chunks never reached it; now about 97% are kept, and MMR trades
  relevance against redundancy on the reranker's scores. The reranker
  receives the best `search.reranking.top_k` fused candidates of the types
  it reranks, as that setting already documented (#317).
- DRIFT search is MS GraphRAG's DRIFT instead of iterative query
  rewriting: the community reports most similar to a hypothetical answer
  (HyDE) prime intermediate answers and follow-up queries in folds; for
  `n_depth` rounds the best-placed follow-ups each run a local search and
  are answered, with new follow-ups, on the fast model; the answer model
  reduces the ranked intermediate answers and the local evidence behind
  them. A query costs about `1 + primer_folds + k_followups * n_depth` LLM
  calls (was 5). On a synthetic corpus of seven-fact chains on the embedded
  stores (12 questions, fake model), the answer context held 6.75 of 7 facts
  per question (was 1.17; 2.92 with the old primer; local search 3.08), at
  15.3 LLM calls per question (#323).
- Global search maps every community report of one cut of the community
  hierarchy, as MS GraphRAG does, instead of the `top_k` reports most
  similar to the query: the communities at `community_depth` from the root
  plus the leaves above it, so every entity is covered once. Reports are
  packed into map calls by tokens and mapped on the fast model
  `processing.max_concurrency` at a time; similarity only cuts a level
  larger than `max_reports_per_level` (flagged `global_level_capped`).
  `use_dynamic_selection` is MS GraphRAG's dynamic community selection
  (rate the roots, descend into the relevant ones' children) instead of
  re-scoring retrieved reports. The context is the ranked key points and
  the reports behind them; text units only with `text_unit_slots`. Each
  indexed report records its `level`, `depth`, parent and children. On a
  synthetic 60-topic corpus on the embedded stores (27 reports of about
  2,700 tokens), the answer context named 59 of 60 topics (was 25) from all
  21 reports of the cut (was 10 retrieved reports), in 6 map calls (was 2):
  map cost now grows with the cut, not with `top_k` (#318).
- Gleaning refines each text unit's own record: its prompt lists the
  unit's own spelling of each entity and its own strengths, an answer's
  issues name entities as that prompt did (a rename applies after the
  answer's other issues), and a unit is re-gleaned while its answers add
  items its record lacked, whatever other units hold. A unit whose
  extraction failed is not gleaned (its document is retried anyway). The
  claim prompt lists the unit's own entities instead of the resolved ones,
  so a unit's claims depend on the unit alone (#322).
- OpenSearch indexing keeps 16 embedding requests in flight (was 8), so it
  embeds and bulk-writes 800 items of a type at a time (was 400). In the
  real-AWS E2E 1,600 texts took 9.4 s instead of 18.8 s, with no Bedrock
  throttling (#310).
- `processing.max_concurrency` defaults to 40 (was 20). In the real-AWS
  E2E ingestion took 27% less time at 40 with no Bedrock throttling, and
  under a heavier mixed load the 0.9% of calls throttled all succeeded on
  retry. The setting also bounds the queries `run-eval` runs at once and
  the global-search map calls in flight, and sizes the OpenSearch query
  pool (now 120 connections). Bedrock quotas differ per account, model and
  Region: lower it if the logs show `ThrottlingException` retries. The
  `backend: local` preset keeps 20, since a local model server queues the
  calls it cannot run at once. Stage caches are unaffected; the setting is
  not part of their keys (#310).
- The data-plane image installs uv 0.12.23, the version CI uses (#303).
- The OpenSearch query client keeps up to the larger of
  `processing.io_workers` and three times `processing.max_concurrency`
  connections (120 by default, between 10 and 200) instead of
  opensearch-py's 10: a query sends up to three searches at once, and a
  request waiting for a connection used up its `search.store_timeout_seconds`
  under `run-eval`. A query with an attribute filter (`attr_<key>`,
  `attributes.<path>`) runs its probe alongside the lexical and kNN
  searches instead of before them, so it waits for one round trip instead
  of two; when no document has the attribute the searches are cancelled
  (#309).
- The embedded search engine (`providers.vector.kind: sqlite`) selects the
  `size` best matches instead of sorting every match (about 13 ms instead
  of 120 ms for the top 10 of 100,000); the hits and their order, ties
  included, are unchanged (#309).
- `BaseIndexer.initialize` is a no-op by default, like `close`: a custom
  `VectorIndexer` or `GraphIndexer` with nothing to prepare need not
  override it (#309).
- Tracing: `run-eval`'s judge model calls (`langchain`, `ragas`) nest under
  one `kg_rag.evaluate` span per evaluator instead of each starting a root
  trace of its own (#309).
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
- Ctrl-C (or a `SystemExit`) during indexing no longer lets the worker
  threads it leaves running write after the run released its lease, when
  another run may hold it: a released lease fails their next write, and
  Ctrl-C sets the stop fence like SIGTERM, so they stop before the release.
  The stop fence exempts the thread that runs the pipeline instead of the
  main thread (#358).
- A SIGTERM no longer waits for the model calls in flight (up to the call
  timeout, past the task's 120 s stop timeout) or, in entity resolution and
  MinHash deduplication, for every queued process-pool task (#358).
- A run that lost its write lease closes its processing attempts when no
  other run holds the lease (it retakes it for the close) instead of
  counting as a death of every document it processed; while another run
  holds it the attempts stay open, as before (#358).
- A lost lease is no longer swallowed by `OpenSearchClient.delete_indices`
  (reported as an `AWSServiceError` after the remaining batches were sent)
  or by the best-effort cleanup of unused extraction records (#358).
- The operations runbook says that with `capture_content` the answer
  model's span keeps the answer as generated even when the guardrail blocks
  it, and that a custom `ContentGuardPort` emits no `kg_rag.guardrail` span
  (#359).
- With `--output-format json`, a CLI that fails before printing its result
  (`run-prompt-tuning`, `run-visualization`) prints
  `{"status": "failed", "exit_code": 1}` (or 2) instead of nothing.
  `prepare-benchmark` exits 130 on Ctrl-C without a traceback, and a data
  file that is not MuSiQue JSONL (an invalid line or a malformed item) is a
  usage error (exit 2) instead of a `KeyError` (#359).
- `prepare-benchmark musique` keeps file names distinct under the retrieval
  evaluator's matching (case, NFKC, `_` for a space), so two paragraphs
  titled `A_B` and `A B` no longer count as the same reference (#359).
- The strategy comparison's guardrail warning no longer says every metric
  of a blocked answer is scored on the blocked message: an answer blocked on
  output has no retrieval or context score, so those queries are unpaired
  (#359).
- `run-ingestion --retire-source-scope` also tries any value that is not a
  URI as a path resolved like `--source-directory`, so a bare relative path
  (`old-corpus`) retires the old directory's records (#359).
- RAGAS no longer judges a guardrail-blocked answer on its (dropped)
  contexts: `context_precision`, `context_recall` and `faithfulness` are
  skipped as `guardrail_blocked` (they scored near 0, and faithfulness came
  back NaN and counted as failed toward `--max-failure-rate`); the answer
  metrics still run (#359).
- Merging document filters is linear: unioning the filters of a report's
  40,000 text units took 11 s (a membership test on a list), now 0.04 s;
  the value order is unchanged (#360).
- A global map phase whose every call timed out fails the query (unless
  `processing.ignore_errors`) and records `TimeoutError`: a call
  BatchProcessor abandons at `call_timeout_seconds` raised no error the map
  step saw, so it answered from the unjudged reports after the timeout with
  a parse-error label. Dynamic community selection tolerates one failed
  rating as the map phase tolerates one failed batch (the report is kept and
  recorded); a step none of whose ratings succeeded fails the query (#360).
- An entity-attribute filter (`attributes.<path>` other than
  `attributes.filters.<key>`) no longer cuts graph expansion to the seeds:
  only document filters are checked against the text units, which never
  carry an entity's own attributes (#360).
- An attribute filter no longer costs a post-filter fetch per relationship
  or claim sub-search, nor fails the query when a hub entity cites more text
  units than one `terms` clause holds (`max_terms_count`). On a 3000-document
  hub corpus DRIFT's store responses under a filter fell from 42 to 29 MB
  (1.8 to 1.0 s) and local's from 2.3 to 1.7 MB (#360).
- A run that lost its namespace's write lease during a full build no longer
  swaps the OpenSearch alias onto the partial index and deletes the complete
  one: the vector indexer re-raises `NamespaceLeaseError` instead of
  recording it as failed items, and checks the lease before each alias swap
  and stale-index deletion (#355).
- A SIGTERM during indexing stops the run within seconds instead of after
  the whole indexing phase (often past the task's 120 s stop timeout, so the
  task was killed, counted as a death and lost its checkpoint): the store
  writes of the worker threads stop at their next request, and the thread
  pools no longer wait for their running tasks once the block raised (#355).
- Every attempt of a Neptune or OpenSearch write request checks the lease,
  the transport's retries included (up to 6 x 180 s before) (#355).
- A renewal that answers after the lease expired (a frozen process, a slow
  registry) loses the lease instead of re-taking it: another run may have
  written and released the namespace meanwhile (#355).
- A run checks its lease at each stage start and before each checkpoint
  upload, so one that lost it in a long stage that does not write stops
  there, and a stopped run that lost it uploads no checkpoint (#355).
- `indexing.reset` skips documents recorded QUARANTINED for their current
  content and extraction settings and keeps their records; it used to
  process them and die on them every time (#355).
- The warning for documents that produced no text units says the run counts
  as a death of each (with the count) rather than only that they stay
  PENDING (#355).
- Docs match the code again: a document is `FAILED` in five cases (an
  extraction record that could not be stored, `records_not_stored`, is the
  fifth), `global` maps one cut and falls back to a shallower one over
  `max_reports_per_level` (`kg_rag.global.select` carries `kg_rag.cut`), the
  response-cache key no longer names the guardrail, removed config keys warn
  and are ignored while moved ones fail, deletions remove vector copies
  before graph copies, and the incremental switch is
  `providers.doc_status.enabled` (#350).
  `providers.doc_status.enabled` (#351).
- `run-prompt-tuning --max-docs` below 1 is a usage error (exit 2) instead
  of silently sampling one document (#353).
- `run-visualization` exits 2 (usage error) on a data file that is not
  valid exported visualization JSON, as `run-eval` does on an invalid
  dataset, instead of 1 (#353).
- `run-ingestion --output-format json` names the `pipeline_id` in the
  interrupted status document, so a stopped run can be resumed with
  `--pipeline-id` without reading the logs (#353).
- `run-rag --interactive --output-format json` is rejected as a usage error
  (exit 2) instead of running a chat that printed nothing on stdout (#353).
- `run-ingestion --retire-source-scope` resolves a local path the way the
  source directory is resolved (`~`, `..`, symlinks), so a path through a
  symlink or a relative path retires the records stored under the resolved
  directory instead of matching none (#353).
- Ctrl-C is handled alike by every CLI: `run-prompt-tuning` and
  `run-visualization` now print the `{"status": "interrupted"}` JSON
  document with `--output-format json`, and `run-rag` exits 130 without a
  traceback when interrupted while waiting on a model or store (#353).
- `prepare-benchmark musique`: a paragraph whose title starts with `.` is no
  longer written as a hidden file the parser skips (#353).
- The CLI exit-code table no longer says an index built with the registry
  off exits 2: `run-ingestion` reports a configuration conflict found while
  running as a failed run (exit 1) (#353).
- `run-eval --pairwise-judge`: a judge reply with a missing or malformed
  verdict for a criterion now fails the pair (counted in `failed`, held to
  `--max-failure-rate`) instead of counting as a tie (#353).
- Evaluation: an answer the guardrail blocked (which drops its sources) no
  longer scores as a retrieval miss; its `retrieval` metrics and retrieved
  coverage are skipped as `guardrail_blocked`. The strategy comparison
  reports each strategy's `guardrail_blocked_counts` and warns when they
  differ, and `run-eval --output-format json` includes them per run (#353).
- DRIFT's primer calls run at most `search.max_concurrency` at a time, as
  the design and configuration reference state: the folds bypassed the
  semaphore the follow-up actions share, so all `primer_folds` calls went
  out at once (#354).
- Global search's cut probe works at the defaults: the cap is now at most
  `indexing.opensearch.max_query_size` - 1 (99 by default), so the browse
  that asks for one report more than the cap fits one request and a cut of
  exactly the cap counts as complete. With `max_reports_per_level` equal to
  `max_query_size` (both 100 by default) the probe could ask for no more
  than the cap, so a cut of exactly the cap was taken for a larger one (#354).
- An incremental run clusters a small delta component that touches a stored
  entity: a new entity whose only relationship is to an existing one was a
  pair in the delta graph and got no community or report since components
  under `min_community_size` are left out, where a full build clusters it with
  its stored neighbour. The community stage reads the stored entities of the
  small components' members (one graph read by id) on incremental runs (#354).
- Global search no longer answers silently from unjudged reports when its map
  step fails: an unrated batch (a failed or unparseable map call) is recorded
  in the search metadata (`degraded_error`), and when every map call failed,
  other than on a prompt too long for the model, the query fails with the
  first error unless `processing.ignore_errors` is on, as DRIFT does (#354).
- iac/README.md no longer says a second concurrent ingestion stops before it
  reads anything: it syncs the corpus and its checkpoint first, then stops
  with `NamespaceLeaseError` before it reads the stores or writes (#345).
- IaC: on Neptune Serverless (`neptune_serverless_ncu`) the writer's
  low-freeable-memory alarm fires only while the writer is at its maximum
  capacity (at its minimum, about 2 GiB, low free memory is what makes it
  scale up, so the fixed 512 MiB floor alarmed on an idle cluster), and a
  new `NeptuneServerlessAtMaxCapacity` alarm fires when the writer stays at
  the maximum for 15 minutes (#345).
- IaC: the guardrail stack publishes the guardrail it creates as a version
  (`AWS::Bedrock::GuardrailVersion`, `GuardrailVersion` output) for the
  tasks to pin with `-c guardrail_version=<version>`; synth warns when
  `guardrail_identifier` is set without one, since the tasks then apply the
  mutable `DRAFT`. A change to the baseline policy publishes a new version
  and keeps the old one until compute is redeployed with the new number
  (iac/README.md) (#345).
- IaC: a misspelled CDK context key (`-c neptune_instnace=...`) fails the
  synth naming the closest setting, as an unknown app config key does,
  instead of leaving its setting silently at the default; CDK's own keys
  (feature flags, `aws:cdk:*`, cached lookups) and `_doc` notes are not
  checked (#345).
- An S3 connection error while uploading an LLM response-cache segment no
  longer fails the model call that filled it: the cache logs the failure and
  hands the segment over again at its next write (the stage checkpoint upload
  sends it too), and a `BotoCoreError` (for example
  `EndpointConnectionError`) in any S3 cache transfer now counts as a
  per-file failure (`CacheSyncError`) instead of escaping raw (#345).
- The extraction store no longer grows without bound: a full build or
  `indexing.reset` whose text units were all written keeps only the
  extraction records its text units name, so records of earlier extraction
  settings and of deleted or edited documents' chunks (their extracted
  content included) are dropped instead of kept forever; incremental runs
  already deleted the records only removed text units named. The operator
  runbook states the retention. A switch of extraction settings and back
  re-extracts once (#345).
- Evaluation counts guardrail blocks: each result records whether the
  guardrail blocked its query or answer (`guardrail_blocked`), and the
  summary counts them (`guardrail_blocked_counts`, also printed by
  `run-eval`), since a blocked item is scored on the blocked message, and
  `retrieval_config_sha256` covers whether a guardrail is on (its id stays
  out), so a guarded and an unguarded run no longer hash the same (#345).
- The last outputs that followed the interpreter's hash seed
  (`PYTHONHASHSEED`, a new salt per process) no longer do: the community
  members in `visualization_data.json` are listed sorted rather than in set
  order, a small community tied to two neighbours joins the lower-numbered
  one even when a float sum in set order would split the tie by one ulp,
  and the entity and chunk ids a strategy collects from its hits keep their
  rank order. A test runs a full build, an incremental run and every search
  strategy in two processes with different hash seeds and requires
  identical stores, prompts, exports and answers (#343).
- `GraphIndexer.delete_by_id` and `find_incident_relationship_ids` without a
  suffix act on the default suffix, as every other indexer method does;
  they reached every suffix, so a library caller that left the suffix out
  deleted another tenant's vertex with the same content-derived id (#344).
- The warning that a `custom_prompts` override drops the output tags its
  parser reads is logged once per override, not once per validation (each
  config layer and provider-flag rebuild repeated it) (#344).
- Hybrid min-max fusion raises every normalized score to at least 0.001, not
  only an exact 0: a hit just above its sub-query's minimum ranked below the
  minimum hit itself, as in the OpenSearch plugin (#344).
- The user guide names the five prompts that have no `custom_prompts`
  override (chunking, document and query translation, global search's
  community rating and reduce step) instead of saying every prompt has one;
  two docstrings named the removed `aws.bedrock_region` and `action_model_id`
  keys (#344).
- Every CLI exits 2 on a `ConfigurationError` met while running (an index
  built with the registry off, a parsed-JSON directory equal to the corpus);
  only `run-eval` did, the others exited 1. Invalid provider flags are
  reported with the loader's one-line summary, not pydantic's raw text (#344).
- In local mode the stage cache keys, the extraction settings and the
  extraction-record keys count `aws.bedrock.default_max_output_tokens`: a
  local model is asked for that cap too, but the keys dropped it with the
  other Bedrock settings, so a changed cap reused output made under the old
  one. Local-mode stage caches and extraction records made before miss once;
  the recorded extraction settings compare unchanged, so nothing is
  re-extracted incrementally on that account (#344).
- Global search's dynamic community selection follows the children of a
  relevant report down to the leaves, as MS GraphRAG does:
  `search.global_search.community_depth` bounds only the whole-depth
  ratings of a selection that found nothing relevant. The walk stopped at
  `community_depth`, so a relevant report's relevant children were never
  rated (#347).
- Global search credits a key point only to the reports it comes from: the
  map prompt now asks for the `report_ids` behind each point (MS GraphRAG's
  `[Data: Reports (ids)]`), and only those reports become context and
  sources. Every report of a map batch was credited before, so a point drawn
  from one report put up to a dozen unrelated ones in the answer context and
  `sources`; on a synthetic 40-topic corpus a single-topic question reported
  4 sources (1 relevant) and now reports the 1. A point without usable ids
  (a custom `global_map` prompt) still credits its batch (#347).
- One failed DRIFT action (a throttled or timed-out model call) no longer
  fails the whole query: its answer is left out, the local evidence it
  found is kept and `degraded_error` names the failure. A query fails only
  on a fatal error, or when every action (or every primer call) failed and
  `processing.ignore_errors` is off; a round whose actions all failed ends
  the rounds. When the query does fail, the actions still in flight are
  cancelled instead of running their LLM calls to completion (#347).
- A paired strategy comparison no longer drops the queries a strategy
  failed to answer: a strategy that crashed on the queries it would lose
  looked as good as one that answered them. A failed answer scores 0 on
  every metric another run scored the query on (`answer_length` excepted),
  each strategy reports `failed_answers` / `failed_queries`, every delta
  `unpaired`, and `warnings` names strategies with different failure
  counts. The pairwise judge gives a pair with one failed answer to the
  other side (`candidate_errors`, `baseline_errors`, `both_errors`), its
  judge failures count against `--max-failure-rate` (all failing exits 1),
  and its judge calls nest under one `kg_rag.evaluate` span. The user guide
  says why retrieved entity coverage is skipped rather than 0 when a context
  has no entity source, and what that costs a comparison (#340).
- A `run-eval` with an `s3://` output directory that is stopped (SIGTERM,
  Ctrl-C) or crashes part-way uploads the results of the runs that
  finished, or keeps them locally and logs where; they were deleted
  unuploaded (#340).
- `run_manifest.retrieval_config_sha256` covers every setting that can
  change an answer or a score: the hand-kept section list missed
  `graph.community_detection.enabled`, `processing.claim_extraction.enabled`,
  `processing.ignore_errors` and the Neptune traversal settings, so runs
  that answered differently hashed the same. The hash now takes every
  setting except those the stage-cache classification marks as changing no
  answer (`cache_keys.answer_settings`), so its value differs from earlier
  manifests (#340).
- `run_manifest.dataset.content_sha256` hashes the dataset file's content:
  it included the metadata `run-eval` adds to each query, so the runs of
  one comparison hashed the same dataset differently (#340).
- The index build record of an incremental run that found the registry
  empty says `full`, and an incremental record keeps the full build it was
  applied on top of (`full_build`), so the manifest no longer reports the
  last delta's document count as the index's (#340).
- An `auto` run's result files are named `..._auto_...`, not after the one
  strategy it happened to route every query to (#340).
- `--output-format json` prints `{"status": "interrupted", "exit_code": ...}`
  when a signal stops a CLI before it printed its document (#340).
- `prepare-benchmark` with a non-empty `corpus/` exits 2 (a usage error)
  before downloading the benchmark, instead of exiting 1 after (#340).
- The trace-shape docs list the `global` and `drift` spans (#340).
- A run resumed with the same pipeline id after the extraction stages wrote
  its text units without the keys of their extraction records: the stages
  set the keys on the units after the stage cache had saved them, so the
  resumed run's units named none and no later incremental run could derive
  what they state. The keys are now computed from the run's records when
  the units are indexed (#348).
- A text unit whose extraction record could not be written to the
  extraction store was still indexed naming the record, and incremental
  runs kept the items it states as they were. A record that is not stored
  now fails its unit (`records_not_stored` in the graph resolution and
  claim extraction metrics): its document is recorded `FAILED` and
  extracted again (#348).
- A graph entity write that failed while a run re-keyed an entity no longer
  loses an edge an unchanged document states: the edges were written after
  their endpoints' failed writes, and a graph store dropped an edge's stored
  copy before finding its new endpoint missing. Edges of an entity whose
  write failed are held back (counted failed), and Neptune, Gremlin Server
  and the embedded graph keep the stored copy of an edge whose endpoint is
  missing (#348).
- A run killed while it deleted items between the two stores no longer
  leaves vector copies no later run finds: the vector copies are deleted
  first, the graph copies (through which a retry finds them) last (#348).
- Incremental runs converge to a full build where an earlier run had merged
  what a full build keeps apart: a stored entity holding two spellings that
  now resolve apart is split, with its edges and claims following the part
  that states them; the edge between them that the merge had dropped as a
  self-loop is restored, and a stored edge whose endpoints now resolve to
  one entity is deleted. The stored edges of a re-keyed entity are derived
  with it, so a run retried after dying between the stores no longer keeps
  a re-pointed copy next to the derived edge. Edges take the per-unit
  strengths of the full build's resolution, a claim's object name follows
  its derived entity as its subject's does, and the claim extraction stage
  merges a claim's statements in text-unit id order, as derivation does.
  The randomized convergence check of the review (25 sequences each over
  three seeds, with per-chunk entity types) now finds no divergence, where
  it found 3 to 7 per 25 (#348).
- Incremental runs over two spellings of one name ("Initech Corp",
  "Initech") converge to a full build in the cases the cycle-40 review
  found: a stored entity split into its spellings keeps its own under its
  id (the other part was dropped as "held by another item"); a derived
  entity is merged with a stored spelling of its name that cites none of
  the run's text units (a deletion left a duplicate); a claim a full build
  keeps apart from one that had merged it is derived as a claim of its own,
  and a claim's own derived id wins over another spelling's, so running a
  finished pipeline id again no longer deletes an unchanged document's
  claim; and the stored edges of the entity another one merged into are
  derived too (a duplicate edge of double weight was left). An entity
  merged into a stored one of another spelling extends the stored summary
  instead of summarizing every line its text units state. The extended
  convergence fuzzer finds no divergence in 400 sequences without injected
  crashes, where it found 7, nor in 80 on Gremlin Server and OpenSearch,
  where it found 3 (#352).
- A run killed after it split two spellings of one name ("Initech Corp",
  "Initech") in the graph but before it wrote their edges and claims no
  longer leaves the other spelling's statements where the split took them
  from once resumed: the graph entities were the only record of the split,
  so the retry found nothing to split and kept an edge of double weight and
  a claim of the entity about itself. A derived entity and another spelling
  of its name that share a text unit now have the other spelling's edges
  derived with them and its claims follow (#357).
- Removing a document no longer leaves its text unnoticed in community
  reports: a kept report that cites a removed text unit is counted in the
  `community_drift` metrics (`outdated_reports`, deletion-only runs
  included) and recommends a rebuild (`community_rebuild_recommended`).
  Reports are not regenerated on incremental runs (#348).
- The graph-expansion degree (`indexing.neptune.entity_importance_source:
  degree`) counts an entity's relationships only, not its community
  memberships, as the setting describes (#348).
- The docs no longer contradict the code: an unknown config key fails the
  load (not a warning), the four causes of a `FAILED` document are listed
  alike in the user guide, the runbook and the design doc, a second
  ingestion on the CDK stack stops with `NamespaceLeaseError`, `naive` is
  BM25 + vector, and `config-template.yaml` lists
  `evaluation.answer_normalization` and no longer mentions the removed
  `effort` key (#336).
- Runs that die processing documents no longer quarantine documents that
  did nothing wrong. Only a run that died counts: each run opens an attempt
  on the documents it processes right after the diff and closes it however
  it ends in the process, so a run stopped by SIGTERM (a task timeout,
  StopTask) and a full rebuild (`indexing.reset`) count nothing; three
  rebuilds that timed out used to make the next incremental run quarantine
  and remove the whole corpus. The attempt opens before the first stage, so
  a document that kills chunking, extraction or gleaning is caught too (it
  used to block every run). A document is quarantined only once it died
  processed alone; when runs died with several documents, the next runs
  process half of the suspects and defer the other new and changed
  documents (`total_documents_deferred`) until the culprit is alone, about
  `log2(n) + 1` runs for one culprit among `n`. A resumed run whose delta
  the registry no longer matches (another run re-indexed or deleted one of
  its documents) or whose runs died often enough to isolate a culprit
  resumes from document_loading instead (#346).
- A run that failed with an error (a store outage) no longer counts toward
  quarantine either: three such runs used to make the first healthy run
  quarantine and remove the documents. A changed document whose runs died
  before its indexing started keeps its indexed version while quarantined,
  and raising `indexing.max_document_attempts` above a quarantined
  document's deaths retries it, as the setting promised (`DocStatusPort.diff`
  reports the unchanged quarantined records in
  `DocumentDelta.quarantined_unchanged`) (#346).
- A reset whose graph writes partly failed (within
  `indexing.max_failure_rate`) kept the previous graph beside the new one,
  yet recorded the corpus and cleared the replaced documents' records, so
  their vertices and edges were never removed. The skipped drop is now a
  failed write (`neptune_sweep`): the rebuild fails and records nothing, and
  the next run removes what the previous graph still holds (#346).
- A run that lost its namespace's write lease could keep writing: the lease
  was checked once per step, not before the commit's store writes, each
  indexing phase or the rebuild's sweep (which could drop what the new
  holder had written), and a renewal that answered late made the run trust
  the lease past the store's expiry (the lease requests used the registry
  client's 10 attempts of up to 60 s). The renewal time is now taken when
  the renewal is sent, the run stops writing once less than a third of the
  term is left, the heartbeat renews every sixth of
  `indexing.lease_seconds`, and the lease is a write fence
  (`shared.write_fence`) that every indexing phase, the sweep, every Neptune
  batch and drop and every OpenSearch bulk request checks. The lease
  requests use short timeouts and 3 attempts, the end of a run no longer
  waits for a hung renewal, a stopped run uploads its checkpoint before it
  releases the lease, and a run downloads its checkpoint only once it holds
  the lease (#346).
- Two corpora written to one index namespace that hold the same file with
  the same text no longer share text units, so removing one corpus's copy
  keeps the other's (#327).
- A dev deployment's ingestion runs no longer trip a prod deployment's
  alarms in the same account and region: the EMF metrics were published
  without dimensions under one namespace, and every deployment's
  `IndexingFailures` and `ExtractionFailures` alarms read the same series
  (#330).
- `auto` no longer routes queries to `global` or `drift` when
  `graph.community_detection.enabled` is false: without community reports
  they answered from nothing. The routable set is now
  `search.auto_routable_strategies` without the strategies the pipeline
  cannot serve (`local` when none is left), and the CLI endpoint preflight
  for `auto` checks the same set (#317).
- LightRAG `mix`/`hybrid` incident-edge expansion and `local` search's
  bridge relationships keep every typed edge between two entities: edges
  were deduplicated by their endpoint pair, as upstream LightRAG does for
  its one-edge-per-pair graph, so of "Vendor supplies Buyer" and "Vendor
  invoices Buyer" only the first reached the context. They are now
  deduplicated by relationship id (#317).
- Incremental runs no longer keep what a changed or deleted document
  contributed to the items it shared with other documents: their
  description, name, type, confidence, document filters and rank, and a
  shared claim's description, source text and status, are derived from the
  records of the text units they cite now, by the full build's merge and
  resolution, so they equal a full build of the current corpus. A name
  follows the most-supported spelling, the stored edges of an entity whose degree or
  name changed get their new rank and endpoint name, and a description is
  summarized again only when the text it is derived from changed. The
  property tests and the crash-recovery fixture compare descriptions,
  names, types, ranks and claims with a full build (#325).
- A stopped or interrupted ingestion no longer keeps retrying the model
  calls that were running: their batch makes no further attempt and a
  backoff wait ends at once, so the task exits once the calls in flight
  return instead of being killed after the stop timeout (#326).
- A SIGTERM no longer hangs `run-ingestion --s3-sync` when it arrives while
  a stage is handing work to a thread pool: the handler uploaded the
  checkpoint itself, and the upload waited on a lock the interrupted code
  held. The handler now only stops the stage; the checkpoint is uploaded as
  the run unwinds (#326).
- The CLI provider flags no longer drop environment overrides. On every CLI
  (`run-ingestion`, `run-rag`, `run-eval`, `run-prompt-tuning`,
  `run-visualization`), any provider flag (`--backend`, `--llm-provider`,
  `--rerank-provider` ...) rebuilt the config from the keys the file set, so
  an override under a section the file left out was lost: with the image's
  `docker/config.yaml`, the deployed task's environment and
  `--rerank-provider none`, `BEDROCK_GUARDRAIL_IDENTIFIER` (the query
  guardrail was silently off), `BEDROCK_REGION`, `NEPTUNE_ENDPOINT`,
  `S3_BUCKET_NAME`, `GRAPHRAG_SOURCE_SCOPE`, `OPENSEARCH_NUMBER_OF_REPLICAS`,
  `OPENSEARCH_KOREAN_ANALYZER` and the `GRAPHRAG_OBSERVABILITY_*` settings
  fell back to their defaults; without a config file, `--backend aws` also
  lost `OPENSEARCH_ENDPOINT` and `LOG_LEVEL`. The environment is now a layer
  of the input, merged over the file before the config is validated, so the
  order is defaults < `backend` preset < file < environment < provider
  flags. An environment value is now also checked with the rest of the
  config: `OPENSEARCH_USERNAME` keeps the local preset's anonymous access
  off, as the same key in the file does (#313).
- `run-ingestion --s3-sync` no longer loses the stages a stopped task had
  finished. Checkpoints reached S3 only after the last stage, and the
  container ignored SIGTERM (Python runs as PID 1 there), so a task ended by
  StopTask or a Step Functions timeout was killed after 30 s with nothing
  uploaded and the next execution repeated every stage of the phase. The run
  now uploads the changed checkpoint files after every stage (stage data
  first, the run metadata last); on SIGTERM it uploads, stops the running
  stage and exits 143, and a re-run with the same pipeline id starts at that
  stage. The app container's stop timeout is 120 s (was 30 s) (#316).
- An S3-synced run resumed more than a day later no longer recomputes every
  stage: checkpoints carried the 24 h `cache.ttl_seconds` expiry while the
  bucket keeps them for 30 days. With `--s3-sync` checkpoints have no TTL,
  including ones an earlier run stamped with one; the bucket lifecycle
  expires them (#316).
- A corrected relationship and the uncorrected one other text units state
  each keep their own description and document filters; both kept the
  merged description, and the uncorrected one every unit's filters (#322).
- Community detection no longer logs the WARNING "Graph has no weighted
  edges" when the hierarchy ends normally at a level whose communities no
  edge joins; it logs that at INFO and stops, with the same communities. An
  entity graph without weighted edges still warns (#310).
- A Neptune that accepts the connection but never answers no longer blocks
  ingestion forever: each ingestion request (a write, a read-back, a drop)
  now fails after 180 s, above Neptune's own default query timeout of
  120 s, and is retried like other transient write errors. The indexer's
  Gremlin client had no timeout (#310).
- Neptune graph expansion fetches at most its per-seed budget
  (`top_k * retrieval_multiplier * indexing.neptune.traversal_fetch_multiplier`
  split over the seeds) per seed, nearest hop first. Neptune applied the
  budget to each traverser of each hop, so one seed on a tree of fan-out 8
  with a budget of 5 fetched 5 + 25 entities; with
  `traversal_fetch_multiplier: 1` (no cut after ranking) all of them reached
  the context. Gremlin Server and the SQLite graph already capped the
  seed's total (#310).
- A filtered vector or hybrid query no longer logs an ERROR with a traceback
  for the kNN search it discards once the attribute probe finds no matching
  document (the expected 400 "Rewrite first" on OpenSearch 2.13). The async
  OpenSearch client now logs its transport errors at DEBUG and names the
  index in the error; the retriever logs each error it keeps once (#310).
- A batch LLM call that raised a `TimeoutError` of its own (a socket read
  timeout, say) is no longer reported as exceeding
  `BatchProcessor.call_timeout_seconds`: its own error propagates, so the log
  names the real cause (#309).
- An attribute filter (`attr_<key>`, `attributes.<path>`) no longer fails
  vector and hybrid searches on OpenSearch 2.13 with `ConfigurationError`
  ("Rewrite first") when an index never mapped the attribute, such as the
  community reports of `simple`, `global`, `drift` and `local` with
  `--filters attr_category:x`. Such an index is now skipped (no document
  can match), as documented: the reports drop out of the answer instead of
  failing it (#306).
- kNN search on the default lucene engine explores
  `indexing.opensearch.vector_search.ef_search` candidates (default 100):
  lucene ignores the `index.knn.algo_param.ef_search` index setting, so each
  query searched only as many candidates as it returned hits (10 to 40) and
  missed about 1 in 10 of its true nearest neighbours. The retriever sends
  `k = max(hits, ef_search)` on lucene and still returns the hits asked
  for; faiss and nmslib keep reading the index setting (#306).
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
  alarm's window (#303).
- OpenSearch hybrid search fuses its lexical and kNN hits in the retriever
  (min-max normalization of each over its own hits, weighted by
  `search.hybrid`), as two searches per index. OpenSearch 2.13's `hybrid`
  query normalized the lexical side over a document outside its returned
  hits (the last-indexed match), which skewed every fused score and the
  order of the top hits; the one-spare-hit workaround fixed only part of
  it. The embedded engine and OpenSearch now return the same hybrid
  ranking (#306).
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
  the `graph_extraction` model, the ingestion model the tuned prompt is for,
  instead of the query-time `entity_extraction` model (#292).
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
  It resolved the index, checked the result window and loaded each index
  in separate reads, so an index another
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
  reports; with a separate fast-tier model (`models.fast`) the answer call runs on
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
- A search OpenSearch rejects with HTTP 400 (such as a filter value of
  another type than its field) fails
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
- OpenSearch indexing embeds and bulk-writes each artifact type 800 items at
  a time instead of embedding the whole type before one bulk call. Memory no
  longer grows with the corpus (about 75 KB per entity before, several
  gigabytes at 50,000 entities with up to six types indexed concurrently), and
  the index is refreshed once per type as before (#252).
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
- A document whose relationships all have strength 0 (or negative, which
  parsing clamps to 0) no longer kills ingestion at community detection.
  Leiden panics, with an exception no error handler catches, on a graph
  whose edges all weigh 0, and an incremental run clusters only the new
  documents' subgraph, so one such document crashed the run and every
  resume. Such a graph now gets one community per entity, as a graph
  without edges does (#304).
- An incremental run no longer stores a relationship twice when a gleaning
  correction reversed or retyped it in one document and a later document
  states the corrected form. The corrected edge kept the id of its original
  (source, target, type), so the incremental read-back (by the delta's
  ids) missed it and the two became parallel edges with split weight and
  two vector documents, where a full build has one. A correction now gives
  the edge the id of its corrected key. Edges stored by an earlier version
  keep their id until a full rebuild (#304).
- A gleaning correction that reverses or retypes a relationship changes only
  the correcting chunk's statement of it. Extraction merges every chunk's
  statement of an edge before gleaning, so one chunk's correction rewrote
  what other documents said: a full build stored one corrected edge citing
  both documents where an incremental run (indexing them in separate runs)
  stored two edges, and deleting the correcting document left the other
  document's statement in the corrected direction. The correcting chunk's
  share (its text unit and strength) now moves to a corrected copy and the
  other chunks keep the uncorrected edge, so both paths store the same
  edges and weights. Both edges keep the merged description, and the
  uncorrected one the union of the document filters (#308).
- One relationship with an absurd strength (such as `1e300`, or a denormal
  like `1e-320`) no longer kills ingestion and every resume at community
  detection: Leiden panics on weights near `1e154` and on denormals.
  Parsed strengths, and edge weights read back from stores before
  clustering, are now clamped to `[1e-9, 1e9]`, with smaller positive values
  read as 0. A panic in Leiden now fails the stage with a `GraphError`
  instead of escaping the error handling (#308).
- Graph resolution and the cross-run merge merge attributes with one
  function (`base_processor.merge_attributes`), so a full build and an
  incremental build give an item the same document `filters`: the
  incremental merge kept a delta's multi-valued filter unsorted (or a
  one-element list instead of a scalar) when the stored item had no
  filters, and the vector store's `attr_*` fields followed. An item without
  attributes now has none (`None`) on both paths; resolution gave it `{}`
  (#304).
- An `indexing.reset` whose doc-status registry cannot be cleared raises the
  registry adapter's own error: a configuration error (for example a missing
  table name) is no longer reported as a `DocStatusRegistryError` asking to
  re-run once the registry is reachable (#305).
- A `BaseGraphRAGEvaluator` subclass that implements neither
  `evaluate_single` nor `aevaluate_single` fails with `TypeError` when the
  class is defined, instead of failing every query at evaluation time (#305).

### Removed
- The `aws.bedrock.guardrail.intervened` attribute on model (`chat <model>`)
  spans, and the detector behind it: no model call carries a guardrail since
  the guardrail screens the query and answer through `ApplyGuardrail`, so it
  was never set. The `kg_rag.guardrail` span keeps the attribute (#353).
- Unused code: `shared.config.reload_config` and `ConfigLoader.reload_config`
  (build a new `ConfigLoader` instead), the `BedrockModelId` type alias,
  `ClaimExtractor.extract_from_text_units` (`extract_units` is the stage's
  path), `BaseProcessor._merge_description`, the `match_none` clause of the
  embedded search engine (no query sends it) and an unreachable
  unknown-fusion-method error. A stage-cache entry without content hashes
  (written by pre-release builds) is now a miss instead of being read
  unverified (#344).
- `aws.bedrock.guardrail.apply_to` and the attached-guardrail path behind
  `apply_to: all` (`GuardrailInterventionHandler`): the guardrail screens
  the user query and the answer with `ApplyGuardrail` only, and ingestion
  and evaluation models are never guarded (an anonymizing or prompt-attack
  policy silently emptied extractions). A config that sets the key loads with
  a WARNING and the key ignored. Stage caches built under `apply_to: all`
  miss once (#344).
- `indexing.cross_run_merge`: an incremental run always merges its delta
  into the stored graph. `false` overwrote an entity shared with unchanged
  documents with the delta's text units, so removing the changed document
  later deleted an entity the unchanged ones still stated. A config file or
  `UKR__INDEXING__CROSS_RUN_MERGE` that still sets it loads with a WARNING
  and the key ignored; nothing to re-index (#344).
- `S3CacheManager.DEFAULT_S3_PREFIX` (`"cache"`), which nothing used: the
  pipeline always passes its `s3_prefix` (default `pipeline-runs`), and
  `S3CacheManager` now requires `prefix=`, so a direct caller can no longer
  write checkpoints under a prefix the CDK bucket lifecycle does not
  expire (#345).
- `MemberOf` edges from entities to communities: no query read them, a
  re-keyed entity lost them, and they counted toward its degree. A community
  is its vertex and an entity names its communities (`community_ids`).
  Graphs built before keep their edges until `indexing.reset: true`; the
  embedded graph's `memberships` table is no longer written (#348).
- The native-score blend of RRF fusion: RRF scores are plain
  `weight / (rrf_k + rank)` sums again. The blend claimed to restore order
  within a stream, which RRF already keeps; scaled to the rank-1 step, it
  outweighed the smaller steps deeper in a list and inverted 20 positions
  of a 60-item DRIFT bucket that mixes graph-distance and index scores
  (#317).
- The machinery that applied gleaning corrections to the merged graph:
  `GleaningIndex.split_relationship` and its rename index, the gleaning
  round merge, and the id-sharing group join and endpoint-citation pass of
  graph resolution (the merge of the records gives one item per id and
  cites endpoints) (#322).
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
- ragas no longer sends usage telemetry from `run-eval`: the evaluators
  package sets `RAGAS_DO_NOT_TRACK=true` unless it is already set (set it
  to `false` to opt back in) (#PR).
- Document attribute filters (`attr_<key>`, `attributes.filters.<key>`) now
  apply to relationships, claims and graph expansion: local, mix, hybrid,
  DRIFT and simple returned relationships from excluded documents, even for
  a filter no document matches. The relationship and claim indexes carry
  their documents' filters and OpenSearch applies the filter (a document's
  removal narrows them); graph expansion keeps an expanded entity other
  than the seeds only when one of its text units passes the filter, checked
  by an ids-only fetch in batches one request holds (#354, #PR).
- A community report's document filters (`attr_<key>`, `attributes.filters.*`)
  are the union of its own text units' document filters. Every report copied
  the attributes of the graph's first entity, so an attribute filter kept or
  dropped reports by that entity's documents (8 of 13 reports carried the
  wrong category on a synthetic corpus), and each report carried that
  entity's own attributes. A report without document filters has none and an
  attribute filter excludes it. Re-index (`indexing.reset: true`) to rewrite
  stored reports (#356).
- A guardrail-blocked answer no longer returns the evidence behind it:
  `sources` and `search_results.results` are empty (they carried the
  retrieved text the refused answer was built from). Masking covers the
  answer only, as the user guide now states. The query span records the
  query only once screened (the masked text, nothing for a blocked query)
  rather than the raw query before screening, and a masked query no longer
  rewrites the caller's `RAGInput` (#345).
- IaC: the query role can no longer write documents through a path that
  ends in `_search`. Its `POST` grant on `<domain>/*/_search` also matched
  `<index>/_doc/_search`, `_create/_search` and `_update/_search` (an IAM `*`
  spans `/`), which OpenSearch serves as a document create or update with the
  id `_search`; an explicit deny now covers any deeper `_search` path and the
  document and bulk APIs. Its S3 listing is limited to the `eval/` prefix
  (it listed the whole cache bucket) (#345).
- Prompts delimit corpus and retrieved text as data, and inputs cannot close
  their delimiter tags (#179, #182).
- IaC: queries get their own task definition and role
  (`QueryTaskDefinitionArn` / `QueryTaskRoleArn` outputs of
  `GraphRagCompute`). The role reads only: Neptune `connect` and
  `ReadDataViaQuery`, OpenSearch `GET`/`HEAD` and `POST` on `_search` paths,
  Bedrock, and the cache bucket's `eval/` prefix; no DynamoDB and no store
  writes or deletes. `run-rag` / `run-eval` previously ran with the ingestion
  role (`neptune-db:DeleteDataViaQuery`, `es:ESHttp*`, doc-status read/write).
  Upgrade: run queries with `QueryTaskDefinitionArn` (iac/README.md, After
  deploy) and keep `run-eval` datasets and results under `s3://<cache
  bucket>/eval/`. The operator runbook explains that the index suffix is a
  naming separation a serving layer must bind from the authenticated caller,
  and how to isolate tenants in the infrastructure (#319).
- The CDK task role's `bedrock:ApplyGuardrail` is granted on the one
  guardrail passed as `guardrail_identifier` (in `bedrock_region`) instead of
  every guardrail in the account and region, and not at all without one
  (#319).
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
- The `unstructured` extra requires `unstructured>=0.24.1`: 0.24.0 kept a
  parsed document's markup unescaped in the HTML it generates, a stored XSS
  (GHSA-v5mq-3xhg-98m9, CVE-2026-104250) (#307).
- The containers of `docker/compose.local.yaml` drop all Linux capabilities
  and run with `no-new-privileges` (#307).
- The data-plane image declares `HEALTHCHECK NONE`: its tasks are batch
  jobs tracked by exit code (#307).
- Retry backoff jitter draws from `random.SystemRandom`, not the seedable
  module PRNG (CWE-338 scanner finding; jitter is not a security
  boundary) (#307).
