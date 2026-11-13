# Design: AWS service integration

Part of the [design reference](../design.md), which lists every section. Code cites
the section numbers: they never change.

## 8. AWS Service Integration

| Service | Module | Purpose |
|---|---|---|
| **Bedrock** | `adapters/aws/bedrock.py`, `bedrock_models.py` (capability catalog) | LLM, embedding and rerank; inference-profile resolution, provider-aware request shaping (adaptive thinking and `effort` on Claude 4.6+, `reasoning.effort` on OpenAI GPT), prompt caching where supported; any model id works through curated rows, provider-family defaults and `aws.bedrock.model_overrides` |
| **Neptune** | `adapters/aws/neptune.py` | Gremlin over `wss://` with SigV4 (`ws://` for a local Gremlin Server); batch upsert and delete, `indexing.neptune.index_concurrency` concurrent batches; every indexer request is retried over a new connection and safe to repeat (vertices and edges are created or matched by id); an edge whose endpoint vertex is missing fails and keeps its stored copy; communities are vertices only (no `MemberOf` edges: an entity names its communities in `community_ids`) |
| **OpenSearch** | `adapters/aws/opensearch.py` | kNN (HNSW, `lucene` engine by default, which supports `cosinesimil` on every OpenSearch 2.x version; §8.5) and BM25, sync and async SigV4 clients, alias management, bulk upsert and delete, per-language analyzers |
| **S3** | `adapters/aws/s3_cache.py` | Stage-cache sync (the bucket's default encryption unless `aws.s3.encryption` forces one) |
| **DynamoDB** | `adapters/doc_status/dynamodb.py` | The doc-status registry |

The AWS session is the bundle's (`Providers(boto_session=...)`, so a fake or
moto session can be injected), created from `aws.profile_name` when an AWS
client first asks for it.

### 8.1 Retrieval Error Visibility

Authentication, configuration and store failures are never disguised as "no
results". One policy, `is_fatal_retrieval_error()` (`adapters/store_errors.py`),
decides **by exception type alone**: fatal errors fail the query, the rest
degrade only the section that raised them. Every sub-retrieval goes through
`BaseSearchStrategy._safe_aretrieve`, so a section handler cannot swallow a
fatal error.

| Failure | Raised as | Outcome |
|---|---|---|
| Store unreachable, connection closed, or 429/502/503/504 after the client's retries | `StoreConnectionError` | fatal |
| No answer within `search.store_timeout_seconds` | `StoreTimeoutError` | fatal |
| HTTP 401/403, Neptune `AccessDeniedException` | `StoreAccessDeniedError` | fatal |
| HTTP 400 (a filter value of another type, a vector of another dimension) | `StoreRequestRejectedError` with the server's reason | fatal |
| Bedrock error that is not transient (bad model id, no access, no credentials) | botocore error | fatal |
| Query embedding failed, whatever the cause | `EmbeddingModelError` | fatal (every vector section needs it) |
| Local model server 4xx other than 408/429 | `openai` / `httpx` error | fatal |
| A model server that cannot be reached (refused, no route, an unresolved name, no connection in time), after its client's retries (a query's chain does not retry it) | `ModelUnreachableError` naming the provider kind, `base_url` and how to start it, of the server the connection was made to when the transport says (`adapters/llm/reachability.py`; ingestion's embedding probe, the query) | fatal, also under `processing.ignore_errors` (never absorbed by the router, rewrite or query processing fallbacks; the error answer) |
| A prompt longer than the model's context | `ContextOverflowError` (or a plain 400 from vLLM / llama.cpp) | not fatal: that global-search map batch is left unrated, unless every batch fails |
| One failed request (an HTTP 500 for one search) | any other exception | degrades that section |
| The whole query past `search.query_timeout_seconds`, or a stream's first chunk past `search.model_timeout_seconds` | `QueryTimeoutError`, raised by `GraphRAGChain` once its tasks (and, through `gather_or_cancel`, every section's) are cancelled | fatal (the error answer under `processing.ignore_errors`) |

A failure that degrades a section is recorded in the query's
`DegradationLedger` (`domain/retrieval/degradation.py`; the per-query scope,
`adapters/retrieval/base.py` `query_scope`): `_safe_aretrieve` and
`_fetch_by_ids` record each absorbed read (`step: retrieval`, or
`findings_lineage`), the strategies their failed model calls (global's
rating and map, DRIFT's expansion, primer and actions, iterative's
controller) and the chain its `ignore_errors` fallbacks (`routing`,
`follow_up_rewrite`, `query_processing` and its translation, entity and
keyword calls, `context_optimization`, also with reason `context_budget` when
no retrieved section fits the context budget). Each search reports its own entries
(`SearchResult.metadata["degraded"]`, set by `BaseSearchStrategy` for every
strategy), the answer every entry of the query (`RAGOutput.metadata["degraded"]`,
each `step`, `reason`, `error_type`), and `degraded_error` stays the error
type of the first. A findings-lineage read of a strategy whose answers do not
cite (`search.citations`; global and DRIFT by default) only makes its reported
sources less precise: it is recorded apart (`metadata.citations_unresolved`,
its entries in the same shape), as is a failed read of the documents of the
answer's own graph citations (`citation_lineage`). The ledger is the one
channel: query processing records its fallbacks there (no field of
`ProcessedQuery` repeats them), and a caller driving the composed runnable
without `ainvoke` gets a ledger opened at its entry. `AnswerView`
(`QueryOutcome.of_output`, over `DegradationReport.of`) reads the three keys
back, with an absorbed error's class, type and message, for the console (a
"partial answer" notice), the MCP tools (`degraded`, `degraded_error`,
`citations_unresolved` after the answer) and `run-rag`, none of which reads
the metadata keys itself.
`run-eval` counts the answers with any entry (`degraded_queries`).

A query is checked before its first model call: its namespace is resolved
once (`domain/retrieval/store_names.py` `resolve_namespace`: the suffix, else
the config's `index_value`, else `default`, the rule of every entry point)
and must hold a text-units index (`BaseGraphRAGRetriever.arequire_namespace`,
once per namespace and chain), else `IndexNotFoundError`.

Neptune specifics: a lost connection is dropped so the next query or retry
opens a new one; a query traversal runs on retriever-owned threads, bounded by
twice `search.store_timeout_seconds`; closing the client fails requests still
waiting for a pooled connection or a late read; the indexer's read timeout is
a fixed 180 s, above Neptune's own 120 s query timeout.

### 8.2 Client Lifecycle / Resource Release

Retrievers open a Neptune WebSocket and thread pool and OpenSearch HTTP
pools, which leak until garbage collection unless closed, so every layer
exposes a best-effort `close()` / `aclose()` that never raises:

- `OpenSearchClient` and `NeptuneClient` close their pools (OpenSearch also
  when the event loop changes, awaiting the old client on its own loop).
  A closed Neptune retriever opens no connection or worker pool again: a
  traversal still running at the close, or started after it, fails with
  `StoreConnectionError`.
- `GraphRAGChain.close()` / `aclose()` close what the chain built, then
  stop the chain's loop (cancelling the sync calls still in flight), even
  when a retriever close or the wait for one fails.
  `IndexingManager.close()` closes both its indexers, injected ones too.
  Indexers passed to `DataIngestionPipeline(vector_indexer=...,
  graph_indexer=...)` are borrowed: the pipeline never closes them, their
  owner does. A `Providers` indexer factory returns a new instance per call,
  which its caller closes. Whoever builds a chain must close it; a finalizer
  is only a safety net. The CLIs close in `finally`.
- `GraphRAGChain` caches retrievers and strategies per event loop; its sync
  entry points run on one chain-owned loop thread, so repeated sync calls reuse
  one set of clients. Blocking calls run in `asyncio.to_thread`, on a default
  executor sized to `processing.io_workers` (which bounds concurrent Bedrock
  calls; `configure_event_loop` for async hosts).
- Concurrent queries in one process share one interpreter lock: their
  Python work (fusion, MMR, the embedded engine's scoring) runs one at a
  time, so a query's latency under N concurrent ones grows with N times its
  CPU. The hot paths stay vectorized or compiled (the dense-script test is
  one regex; the embedded engine lists only what an id lookup or an
  exclusion names), and the query CLIs start with one BLAS thread
  (`launch.one_blas_thread`, `OPENBLAS_NUM_THREADS=1` unless set): with a
  thread per core per kNN product, 40 concurrent local queries spent about
  8 times the CPU for the same scores.

### 8.3 Multilingual Processing

- **Analyzers.** `indexing.opensearch.language_analyzers` maps a language to an
  OpenSearch analyzer. Each supported language has a default: the built-in
  language analyzers (`german`, `english`, `spanish`, `french`, `italian`,
  `portuguese`, `russian`), `cjk` (character bigrams) for `ja` and `zh`, and
  `nori` for `ko`. An Amazon OpenSearch
  Service domain has `nori` only once its Nori package is associated, so the
  CDK deployment maps `ko` to the built-in `cjk` unless
  `opensearch_nori_package` associates the package (a custom resource) and
  maps `ko` to `nori`; an analyzer the cluster lacks fails index creation
  with an error naming it.
- **Identity keys** keep the letters of all scripts (§3).
- **Translation** is skipped when the source and target languages match and
  no other target is set (`TranslationConfig.is_noop`). A query is
  translated only when ingestion translated: not when the translation stage
  is disabled or a no-op, unless the caller asks for a language
  (`RAGInput.target_language`). Its entities and keywords are extracted in
  the language the index was built in: `source_language` when the stage is
  disabled, `target_language` otherwise, also when the query requests
  another language. The requested language is the translation's only: the
  text units carry one `translated_text_<language>` field per additional
  target, which their lexical search reads, but the entity, relationship,
  report and claim indexes hold the index language alone, so an entity or
  keyword in the requested language would match neither their BM25 nor
  their `compact_name_key` seeds. The answer prompt writes in the query's
  language.
- **One index language.** `cache_keys.index_language(config)` is the one
  rule: `target_language` when translation runs, else `source_language`.
  Extraction, gleaning, description summaries, community reports (the
  report prompt names it) and the summary tree write in it, the text units
  are embedded in it, and the graph indexes' text fields are analyzed for it
  (`OpenSearchIndexer.language`). The chain records the language a
  strategy's own prompts write in as `SearchQuery.metadata["query_language"]`:
  the index language, whatever was requested, since their output either
  searches the indexes (DRIFT's and iterative's follow-up queries) or is
  context for the answer model, which answers in the query's language
  (DRIFT's intermediate answers, global's map points). DRIFT's primer and
  actions, global's map points and iterative's controller read it through
  `BaseSearchStrategy.query_language`, falling back to the index language;
  DRIFT's HyDE writes in the index language too.
  The extraction-record and answer fingerprints key the index language
  under their earlier `target_language` field, so a translated index keeps
  its records. Where the translation settings do not say it (translation
  off, `source_language` other than `target_language`: extracted in the
  target language before), the extraction settings and the stage caches
  from translation on key it too (`cache_keys._language_input`), so such a
  namespace re-extracts instead of mixing languages; every other config
  keeps its keys. The claim prompt reads the unit's text in the index
  language and writes each `<description>` in it, keeping entity names and
  `<source_text>` as written, as extraction does.
- **Language-aware parsing.** Query entities split at `,`, `，`, `、`,
  `;`, `；` and line breaks (`domain/retrieval/query_entities.py`). Global
  search's relevance rating skips a scale written in Korean, Japanese or
  Chinese as in English; claim dates read `2024年3月15日`, `2024년 3월`,
  `2024. 3.`, Japanese era years (`令和6年`) and CJK quarters, the month
  names of German, Spanish, French, Italian, Portuguese and Russian
  (`15. März 2024`, `15 de marzo de 2024`), and corpus-language
  placeholders (`不明`, `미상`, `해당없음`) as unknown; an object-less claim's
  object is `NONE` whatever placeholder the model wrote.
  The follow-up rewrite and the query translation answer JSON, and a model
  that cannot give one answers a fixed ASCII sentinel (`[CANNOT_REWRITE]`,
  `[CANNOT_TRANSLATE]`), so no refusal phrase list (it knew four languages)
  is needed.
- **Encoding.** A text, CSV, HTML or JSON file is first read as UTF-8,
  under any locale (`file_loaders.DEFAULT_ENCODING`; `open()`'s default,
  the locale's encoding, read a UTF-8 file as another text on a host whose
  locale is not UTF-8). Only when that read fails with a decode error is the
  file read again in `processing.document_parsing.encoding`, else the
  encoding `charset-normalizer` detects (none detected: the file fails, it
  is not text). A file that decodes as UTF-8 is kept as read, even when it
  was written in another encoding.

### 8.5 OpenSearch engine version

The CDK domain runs OpenSearch 2.19 (`OPENSEARCH_VERSION`,
`iac/stacks/storage_stack.py`), the last 2.x line, and the local stack
(`docker/compose.local.yaml`) runs the same minor version. The application
itself is version-agnostic within 2.x: it works on 2.13 clusters as before.

- **Why 2.19, not 3.x.** OpenSearch Service upgrades any 2.x domain to 2.19
  in one step, and reaches 3.x only from 2.19. 2.19 has the longer support
  window (extended support to 2030-11-07, against 2028-11-07 for 2.13), and
  it is the version the opt-in vector storage keys were measured on
  ([On-disk vector recall](../concepts/defaults-and-evidence.md#on-disk-vector-recall)).
  3.x is a major version with breaking changes (index-level
  `index.knn.algo_param.ef_construction`, `m` and `space_type` settings
  fail its upgrade check; the app sets `ef_search` only, which is not on
  that list) and is left for a later, separately rehearsed step.
- **Upgrade in place, never replaced.** The domain carries the
  `EnableVersionUpgrade` update policy (`enable_version_upgrade=True`), so a
  changed `EngineVersion` is an in-place blue/green upgrade: same endpoint,
  ARN and indices. Without the policy CloudFormation replaces the domain (a
  new, empty one; the old one retained and billed outside dev). The
  service takes its own snapshot before upgrading and restores it if the
  upgrade fails; a completed upgrade cannot be rolled back (the way back is
  a manual snapshot restored on a new 2.13 domain).
- **Nori.** Optional plugins are version-specific and the service upgrades
  an associated one with the domain (an upgrade fails when the target
  version lacks the plugin). `Custom::OpenSearchNoriPackage` depends on the
  domain, so it updates after the upgrade: its new `EngineVersion` resolves
  the 2.19 `analysis-nori` package (a new physical id), finds it already
  active and completes; the update's cleanup then dissociates the 2.13
  package only if it is still listed. Indices that use the `nori` analyzer
  keep working, since the analyzer name is unchanged.
- **What 2.19 enables, and what stays off.** Defaults are unchanged, so an
  upgraded domain needs no reindex and a 2.13 cluster keeps working:
  - The opt-in `vector_mode` / `compression_level` keys (2.17+) and faiss
    with `cosinesimil` (2.19) now work on the CDK domain. They stay opt-in:
    they change the mapping (a reset rebuild), and their memory saving pays
    off only at millions of vectors while costing recall (the measurement
    above). The default stays `lucene` HNSW with `cosinesimil`.
  - No version detection is added: the run's index-creation probe already
    asks the live cluster whether it takes the configured mapping, before
    the first model call, and names the version a refused key needs.
  - The server-side `hybrid` query with a normalization or
    score-ranker (RRF) search pipeline is not adopted. The retriever fuses a
    lexical and a kNN search itself (`OpenSearchRetriever`, §7), which the
    embedded engine reproduces exactly (`tests/integration/test_store_parity.py`)
    and which the multi-bucket `HybridScorer` RRF builds on; a pipeline
    would split the ranking path by backend for no measured gain.
  - The 2.18 default-engine change (nmslib to faiss) does not apply: the
    mapping always names its engine.
- **Failure modes.** An upgrade blocked by its pre-checks (red indices,
  over 90% disk, JVM pressure over 75%, a running snapshot) or a failed
  upgrade (the service restores its pre-upgrade snapshot) fails the stack
  update with the domain on 2.13. A
  slow upgrade (15 minutes to hours) holds the stack update; ingestion and
  queries keep running against the domain with higher latency. A
  dissociation that fails in the cleanup leaves the stack updated and the
  extra package associated ([operations runbook](../operations.md#upgrading-opensearch)).
  A completed upgrade cannot be undone, so a rollback of the same stack
  update cannot return the domain to 2.13: the runbook upgrades with a
  stack update that changes nothing else. A stack deployed before the
  update policy existed (0.2.3) gets the policy in an earlier deploy that
  keeps 2.13: CloudFormation documents the policy only for an update of
  `EngineVersion` under it, not for one that adds it at the same time,
  where a replacement (a new, empty domain) cannot be ruled out.
- **Security and cost.** No IAM, network or encryption change. The same
  instance types (Graviton r6g/m6g) run 2.19. The upgrade's blue/green
  deployment doubles the nodes while it runs, but with an unchanged
  instance type the service bills only the larger cluster, so the upgrade
  itself costs nothing extra.
  On 2.17+ the default 8 GiB dedicated master type is sized for up to 30
  data nodes and 15,000 shards (10 and 10,000 before).

---
