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
  and evaluation on Ollama, a sentence-transformers cross-encoder, Gremlin
  Server, a local OpenSearch and a SQLite doc-status registry (#209).
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
- A filter key that no retriever of the strategy declares raises
  `InvalidFilterError` (#123).
- IaC: `create_guardrail` decides whether the stack owns a guardrail; bring
  your own with `-c create_guardrail=false` (#103).

Defaults. Set the old value explicitly to keep the previous behaviour.
- `graph.visualization.enabled` defaults to `false`; set it to `true` or
  render on demand with `run-visualization` from the `visualization_data.json`
  that ingestion still writes (`graph.visualization.export_data`, default
  `true`: graph and communities, no model call; `run-visualization` computes
  a spring layout and centrality) (#210, #PR).
- `backend: local` sets `search.token_manager.max_context_tokens` to 22000,
  what the default local model's 32,768-token window accepts beside its
  output reservation (30000 was clamped, with a WARNING, on every query) (#224).
- Entity resolution merges only names equal after normalization, legal-form
  removal and dense-script spacing. Similarity matching is opt-in
  (`processing.fuzzy_entity_resolution: true`), and
  `processing.similarity_threshold` defaults to 0.85 (was 0.6) (#214).
- `processing.chunking.content_type` defaults to `text`; `markdown`/`html`
  split at headers only for documents with that body (#213).
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
- Most stage caches miss once after upgrading (their cache keys changed).
  A run resumed with an old `--pipeline-id` can still reuse stage outputs
  written by an earlier build; use `--force-rebuild` or a new
  `--pipeline-id` to apply the fixes to them (#167, #170, #171, #201, #214).

Library API (Python callers and custom backends).
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
  `embedding_factory=` (#216, #PR).
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
- Custom indexers: full-build `index_*` methods take a keyword-only
  `suffixes`; `get_stats` / `get_entity_count` are gone (#169, #214);
  `IndexingManager.index_delta` no longer merges with the stored graph
  (`merge_with_existing_graph`), and `read_entities` /
  `read_relationships` take the index `suffix` (#163).
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
  `application.cli.run_evaluation.exceeds_failure_budget` (#216); and
  `domain.retrieval.strategy_registry.registered_strategies` (#212).
- `BatchProcessor(max_retries=)` is `max_attempts=` (#151);
  `execute_with_fallback` no longer takes `batch_func` (sync) or
  `run_config` (#174, #216); `create_robust_xml_output_parser` requires
  `output_tags` (#174).

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

### Added
- `providers:` config section selecting each role's backend (`llm`,
  `embedding`, `rerank`, `graph`, `vector`, `doc_status`) through a provider
  registry (`@register_provider`); a `[local]` extra; `SQLiteDocStatusStore`;
  an Ollama service in `docker/compose.local.yaml`; and
  `providers.doc_status.enabled` (#154, #209).
- `observability:` section (off by default, `otel` extra): spans for runs,
  stages, batches and each query step, GenAI model-call spans with token
  usage, log/trace correlation, `session.id` / `user.id`
  (`RAGInput.user_id`), opt-in content capture with `redact_patterns`
  applied at capture and again at export, and `export_timeout_seconds`
  (#207, #208, #211, #PR).
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

### Changed
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

### Fixed
- A configuration file with a value of the wrong type, invalid YAML or a
  missing `--config-path` file no longer makes `run-ingestion`, `run-rag` or
  `run-eval` print "An unexpected error occurred" with a traceback. All five
  CLIs treat it as a usage error: one `<cli>: error: ...` line and exit code
  2 (`run-prompt-tuning` and `run-visualization` exited 1). A validation
  error now fits on one line, naming each invalid key without echoing its
  value (#PR).
- A `local` query against a missing index (for example a wrong `--suffix`)
  no longer also logs "_GatheringFuture exception was never retrieved" with
  a second traceback: the community-report, relationship and claim sections
  that fail alongside the entity chain have their error consumed (#PR).
- A `GraphRAGChain` reused across event loops (one `asyncio.run` per query)
  closes the retrievers of each finished loop again: their synchronous close
  ran inside the next query's running loop, where the Neptune (Gremlin)
  client's close fails with "Cannot run the event loop while another loop is
  running", so every loop left a websocket and its threads open (15 queries
  grew a process from 5 to over 40 threads). The close now runs on a worker
  thread (#PR).
- Switching `processing.claim_extraction` on, or changing its model, limits
  or claim prompts while it is on, now re-extracts an incrementally indexed
  corpus: the extraction fingerprint stopped at graph resolution, so such a
  run reported 0 changed documents and indexed no claims, without a warning.
  With claim extraction off the fingerprint is unchanged, so default
  configurations are not re-extracted on upgrade (#PR).
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
- The check that the graph still holds the registry's documents reads past
  any number of entity-less records, still in batches, instead of stopping
  after 1,000: a corpus starting with more than 1,000 entity-less pages no
  longer lets a wiped graph pass as intact (#225).
- Re-running `run-ingestion` with an existing `--pipeline-id` resumes again:
  the integrity check built its cache keys before the corpus fingerprint was
  known, so it reported every stage missing and prompted, which crashed
  without a terminal. Prompts now fail with a clear message without a
  terminal unless `--yes` is passed, including `--repair-metadata` (#223).
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
  indexed documents that list entities, reading past entity-less ones (up to
  1,000 records), so a corpus starting with 20 entity-less pages no longer
  lets a wiped graph pass as intact (#221).
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

### Security
- Prompts delimit corpus and retrieved text as data, and inputs cannot close
  their delimiter tags (#179, #182).
- INFO-level logs no longer carry query text, entity names or model output;
  `run-rag` strips terminal control characters from output; the CLIs warn
  when LangSmith tracing is on (#156, #173).
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
