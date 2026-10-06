# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Require patched `unstructured>=0.24.0` for optional Markdown/HTML parsing on
  Python 3.11+ (GHSA-4mvj-m6j5-pmf7). The updated dependency removes NLTK and
  its unpatched model-artifact path traversal (GHSA-8mgp-746c-j5xp) from the
  lockfile. Python 3.10 retains core formats but no longer installs this extra.
- Constrain the transitive `langchain-openai` dependency to `>=1.1.14` for its
  image-token-counting SSRF fix (GHSA-r7w7-9xr2-qq2r).

Initial public release preparation. This is an AWS-native, open-source reference
framework that unifies the GraphRAG and LightRAG retrieval methodologies on
Amazon Bedrock, Neptune, OpenSearch, and DynamoDB.

### Added
- Two selectable retrieval methodologies: GraphRAG community-summary
  (`auto`/`drift`/`global`/`local`/`simple`) and LightRAG dual-level keyword
  (`mix`/`hybrid`/`naive`), sharing one ingestion/indexing/caching/hybrid-search
  stack.
- Incremental indexing via a DynamoDB document-status registry (content-hash
  diff, idempotent upserts, per-document lineage for deletion).
- Triple-hybrid retrieval (lexical + semantic + graph) with RRF fusion and
  Bedrock reranking; multilingual ingestion and retrieval.
- CLIs: `run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
  `run-prompt-tuning`.
- CDK deployment stack (`iac/`) with Well-Architected security defaults.
- Community reports now carry `text_unit_ids` and `document_ids`, indexed as
  keyword fields, so reports can be filtered by source document. These are the
  candidate sources associated with the community's members (community
  membership provenance): they may include material `max_entities_per_report`
  or the token budget kept out of the report prompt, and they do not establish
  sentence-level citation support. The schema alone does not backfill existing
  reports: those index both fields as empty lists until report generation and
  indexing are re-run (for example `run-ingestion --force-rebuild`).

### Changed
- Language-model ids are plain strings: every `*_model_id` key accepts any
  Bedrock model id or inference-profile id, so a new model can be used without
  a release. `aws.bedrock.default_model_id` and `aws.bedrock.fast_model_id`
  set the model for all roles of their tier at once; a role's own
  `*_model_id` still wins, so existing configs load unchanged. Ids without a
  curated capability record get provider-family defaults (`anthropic.claude-*`
  by generation, `openai.gpt-*`) or conservative Converse defaults, with one
  warning, and `aws.bedrock.model_overrides` describes or corrects a model.
  Python callers that read `config.<section>.<role>_model_id.value` should
  drop `.value`; `LanguageModelId` members remain valid inputs. Claude Haiku,
  Sonnet and Opus 4.5 now fail fast when no inference profile resolves, like
  the other profile-only models.

Retrieval and indexing defaults changed for latency, cost, and upstream parity.
Each previous behaviour stays available through the setting in parentheses.
- `mix`/`hybrid` run independent retrievals concurrently; the multi-hop Neptune
  expansion is opt-in (`search.lightrag_search.enable_graph_expansion: true`).
- DRIFT searches the original query first, fuses with the local per-type quota,
  and its LLM convergence check is opt-in
  (`search.drift_search.enable_llm_convergence: true`; threshold now 0.8,
  `convergence_threshold: 0.1` for the old value).
- Global search skips per-report LLM relevance scoring
  (`search.global_search.use_dynamic_selection: true`) and packs 5 reports per
  map call (`map_batch_size: 2`).
- The retrieval context budget defaults to 30,000 tokens
  (`search.token_manager.max_context_tokens: null` derives it from the answer
  model's window).
- Chunk sizes fit the embedding/rerank input: `max_chunk_size` 8,000,
  `fallback_chunk_size` 4,800, `min_chunk_size` 1,000 characters (old: 50,000 /
  50,000 / 5,000).
- Gleaning rounds after the first re-send only the text units that gained
  items in the previous round (the default stays 3 rounds).
- AUTO routes among local/mix/global/drift with Haiku
  (`search.auto_routable_strategies: [simple, local, global, drift]`,
  `search.strategy_selection_model_id: anthropic.claude-sonnet-5-5`).
- Incremental runs union a touched entity or relationship with its existing
  graph state by default (`indexing.cross_run_merge: true`, previously
  `false`). The overwrite default replaced an entity shared with unchanged
  documents by its delta-only description and `text_unit_ids`, cutting the
  lineage to those documents' chunks that `mix` follows. Each delta run now
  reads the touched entities and relationships back from Neptune before the
  upsert (`cross_run_merge: false` restores overwriting).

### Fixed
- An incremental run no longer overwrites the corpus's communities and
  community reports. Community ids were positional (`L0_C0`, ...) and report
  ids hash them, so a delta run, which clusters only the delta subgraph,
  upserted its `L0_C0` over the full corpus's. Community ids are now a hash of
  the level and the sorted member entity ids (the positional label remains as
  `short_id`/`name`), so delta communities and reports are appended. Nodes and
  edges are also sorted before Leiden, which otherwise partitions the same
  graph differently depending on insertion order. **Re-index required:**
  community and report ids change; an existing index keeps its positional
  communities next to new ones until it is rebuilt.
- Incremental deletion is scoped to the run's tenant and corpus. `deleted`
  was every registry document missing from the run's input across the whole
  table, and the registry key hashed only the local path, so a per-file parse
  failure, a run over a subfolder, or another tenant's run (all corpora are
  staged in `/tmp/graphrag-source` on the container) deleted indexed content.
  The registry key now hashes the index namespace (`index_value` +
  `indexing.additional_suffix`) and the path relative to the source directory;
  a run only deletes records of its own scope (namespace + corpus source,
  new `processing.document_parsing.source_scope`, which the container
  entrypoint sets to the S3 URI via `GRAPHRAG_SOURCE_SCOPE`); and files that
  fail to parse or load are reported as `failed` instead of deleted.
  `DocStatusPort.diff` takes an optional `scope`. **Re-index required:**
  registry keys change and records written before this release carry no
  scope, so they are never deleted automatically; rebuild with
  `indexing.reset: true` (which also clears the registry).
- Two files with the same name in different folders and the same first 100
  characters no longer share a `document_id` (and therefore text-unit ids).
  The id is now derived from the path relative to the corpus root plus a hash
  of the full text (`shared/utils/document_identity.py`), so it also no longer
  depends on where the corpus is checked out or synced. **Re-index required:**
  document and text-unit ids change, so an existing index keeps its old text
  units next to the new ones until it is rebuilt (`indexing.reset: true` or a
  fresh index suffix).
- A changed corpus is now a stage-cache miss. Stage cache keys fingerprinted
  only the source path, so with a fixed `pipeline_id` an edited, added or
  removed file resumed from the previous corpus's documents and incremental
  delta. Every key now also carries a corpus manifest fingerprint (relative
  path, size and SHA-256 of each ingestible file; modification times are
  ignored so an `aws s3 sync` handoff between phases still resumes), plus
  `aws.bedrock.default_max_output_tokens` and `aws.bedrock.model_overrides`.
  Existing stage caches are recomputed once after upgrading.
- Resuming a pipeline after the gleaning stage keeps the gleaned
  relationships. The save and restore paths used two separately maintained
  stage-output mappings, and the restore one loaded only gleaning's entities.
  Both now derive from one definition (`shared.pipeline_manager.STAGE_OUTPUTS`),
  which also restores the parsing stage's documents (a resume from
  `document_loading` previously started with none) and graph analysis's
  relationships.
- An incremental run now fails before committing when removing the stale
  artifacts of changed documents fails. The commit replaced those documents'
  registry lineage, so the artifacts that failed to delete were no longer
  referenced anywhere and stayed in the stores permanently; the registry now
  keeps the old lineage and the next run retries the removal.
- Indexing relationships for one index suffix no longer deletes another
  suffix's edges. The Neptune idempotency pre-drop matched edges by id alone,
  and relationship ids do not depend on the suffix; it is now scoped to edges
  whose source vertex carries the run's entity label, like `delete_by_id`.
- Prompt caching now takes effect on the Converse API, which every Claude
  4.5+/5.x inference profile uses. The system prompt carried an Anthropic
  `cache_control` key that langchain-aws drops when it builds a Converse
  request, so nothing was ever cached there; Converse requests now end the
  system prompt with a native `cachePoint` block, and InvokeModel keeps
  `cache_control`. A marker is only added when the system prompt reaches the
  model's minimum checkpoint size (`min_cache_tokens` in the capability record).
- LLM requests no longer ask for the model's maximum output by default.
  Bedrock reserves input + `max_tokens` against the tokens-per-minute quota at
  request start, so sending 128K on Claude 5.x throttled concurrent calls far
  below real usage. `aws.bedrock.default_max_output_tokens` (16384) now caps
  each request; long-output prompts (extraction, gleaning, claims, community
  reports, document translation) keep a higher floor, and the derived
  retrieval context budget reserves the capped output instead of the model
  maximum. Set the key to `null` to restore the previous behaviour.
- `custom_prompts` overrides for community reports
  (`community_report_system`/`_human`), conversation-memory entity extraction
  (`entity_extraction_*`) and the DRIFT query refinement, keyword expansion and
  primer steps (`query_refinement_*`, `keyword_expansion_*`, `drift_primer_*`)
  now take effect. Those chains were built without the overrides, so the
  built-in prompts were always used.
- Ingestion no longer writes parsed `<stem>.json` files into the source
  directory, where the loading stage misread raw files as JSON and a re-run
  ingested the previous output as new documents. The loading stage reuses the
  parsed documents directly, and the JSON export now happens only when
  `processing.document_parsing.target_directory` (or `--target-directory`) is
  set to a directory other than the source directory.
- Bedrock embedding calls now retry transient model errors
  (`ModelErrorException`, `ModelNotReadyException`, and service-side 5xx or
  throttling that outlast botocore's own retries) with bounded exponential
  backoff and jitter. botocore's retry modes do not treat HTTP 424
  `ModelErrorException` as retryable, so a brief model-side fault previously
  dropped the affected text units, entities, or relationships from the vector
  indexes. Indexing and query-time embedding both use the retry. Embeddings
  that still fail are reported in a single WARNING summary instead of one ERROR
  per item.
- Query-time LLM calls (strategy routing, query entity and keyword extraction,
  translation, context building, answer generation, the global-search
  relevance, map and reduce steps, the DRIFT steps, and conversation-memory
  entity extraction) now retry transient Bedrock errors such as HTTP 424
  `ModelErrorException`. A single transient model fault previously failed the
  whole query, or silently degraded it under `ignore_errors`. Non-transient
  errors still fail fast. `setup_chain` derives the retry from
  `model_purpose`, so ingestion chains keep their existing `BatchProcessor`
  retry without a second layer.
- Ingestion LLM stages no longer retry permanent Bedrock errors such as
  `AccessDeniedException`, `ValidationException` or `ResourceNotFoundException`
  with backoff; those fail the item on the first attempt. Transient Bedrock
  errors, call timeouts and unparseable model output are still retried with
  backoff.
- With `fixing.enabled` (the default), XML output that no recovery step can
  parse now reaches the output-fixing model, which previously never ran.
- Embedding and query-time LLM calls share one transient-error retry policy,
  `aws.bedrock.transient_retry` (5 attempts, 60s budget per call by default).
  The `search.llm_retry` key used earlier in this release cycle is still
  accepted and maps to it.
- RRF fusion now accumulates a cross-store match. The fusion key was derived
  from a hash of the rendered content, and the graph and vector stores render
  the same artifact differently, so an entity present in both produced two keys
  and graph/vector rank agreement was never rewarded. An artifact now counts
  once per fusion bucket, at its best rank there: DRIFT concatenates every
  iteration into one bucket and dedupes by rendered content, so an entity
  reached over two paths arrived twice and its repetition outranked rank 1.
- Ingestion stage-result cache keys now carry a fingerprint of the inputs that
  determine a stage's output (its config subtree, model id, and prompt
  overrides, cumulative over the upstream stages it consumes). A changed prompt,
  model, or processing rule previously resumed from stale stage output unless
  the run used `--force-rebuild`, a new `pipeline_id`, or hit a TTL expiry.
  The resume point follows the cache, not the recorded status: an auto resume
  starts at the first completed stage whose output the cache no longer holds
  under the current inputs and recomputes it and every stage downstream; an
  explicit `--resume-from-stage` whose prerequisite is missing fails before any
  stage runs, naming the stage to resume from. Caches written before this
  change (keys without a fingerprint) carry no record of their inputs and are
  treated as a miss: the run logs the legacy key it found and recomputes from
  that stage once. A completed stage that produced an empty output now caches
  it, so "produced nothing" is distinguishable from "not cached".
- The gleaner now applies the `ENTITY_CORRECTION` and `RELATIONSHIP_CORRECTION`
  issues `GraphRefinementPrompt` asks for (wrong entity name or type, wrong
  relationship type or direction). Issue dispatch branched only on the two
  `MISSING_*` types, so every correction the model returned was discarded, and
  gleaning could only ever add to the graph, never fix it. The prompt now also
  specifies the `<details>` shape for both correction types, which it previously
  requested without defining. After a round's corrections and duplicate merge,
  every relationship's `source_name` / `target_name` is rewritten from the final
  entity id-to-name mapping, so a renamed or merged entity is named the same on
  its edges (which the indexers read) as on the node.
