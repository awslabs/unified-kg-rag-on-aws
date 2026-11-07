# Troubleshooting

Answers to the questions users ask most, then every error message the CLIs
and the library print, with what it means and what to do. Messages are in
English, as the code prints them; search this page for a phrase of yours.
Procedures (recovering a run, alarms, upgrades) are in the
[Operator Runbook](./operations.md); exit codes and the JSON result in
[CLI exit codes and JSON output](./reference/exit-codes-and-json.md).

## Frequently asked questions

### Which source formats are read?

| Extension | Core install | With an extra |
|---|---|---|
| `.pdf`, `.txt`, `.csv`, `.json` | Read | — |
| `.md`, `.markdown` | Read as written (the Markdown syntax stays in the text) | `unstructured`: rendered to its text by unstructured's partitioner, in place of the core reader |
| `.html`, `.htm` | The text, by the standard library's parser: one paragraph per block element, list items as `- item`, table cells joined by ` \| `, no script or style | `unstructured`: its partitioner, in place of the core reader |
| `.docx` | Skipped, naming the extra | `docx` (python-docx): the body's paragraphs and tables in order; headers, footers and comments are not read |

When the `unstructured` extra is installed it always wins for Markdown and
HTML. The two readers write different text, so installing or removing it
changes those documents' content hashes: the next incremental run indexes
them again as changed documents. A file of a type no loader reads logs
`Skipping N file(s) in '<dir>' of a type no loader reads (...)`, naming the
extra that reads it, and is not indexed; an incremental run keeps what it
indexed before and does not treat it as deleted. Install an extra with
`uv sync --extra docx` (or `unstructured`), or build the image with
`--build-arg UV_EXTRAS="--extra docx"`
([`iac/README.md`: Build and push the app image](../iac/README.md#2-build-and-push-the-app-image)).

### Why does a run fail with "No supported source files found"?

The stage `document_parsing` found no file a loader reads in the source
directory: the directory is wrong or empty (a volume not mounted, an S3 sync
that copied nothing, a wrong prefix), or every file is of a type no loader
reads (see the question above). The message lists the supported extensions,
and the warnings before it name the skipped files. Check the directory and
its contents, then re-run. A run whose corpus is seed graphs alone does not
fail here ([Seed graphs: Lifecycle](./guides/seed-graphs.md#lifecycle)).

### Why do Bedrock calls hang in a private VPC?

Bedrock is called in another region than the VPC endpoint's. A private VPC
(the CDK default `network_mode: private`) has no NAT: its only route to
Bedrock is its own region's VPC endpoint, so a call to another region gets
no answer. Every Bedrock call goes to `aws.bedrock.region_name`, or
`aws.region_name` when that is `null`. Keep `aws.bedrock.region_name` unset
or set it to the VPC's region; in the CDK stacks leave `bedrock_region`
unset in `private` mode
([`iac/README.md`: Configuration](../iac/README.md#configuration-cdkjson-context---c-keyvalue)).

### Why is reranking missing from my results?

Rerank models are offered in some regions only (not in `ap-northeast-2`, for
example), and `bedrock:Rerank` needs an IAM statement of its own. A query
whose rerank call fails keeps the fused order and reports
`metadata.rerank.status: failed` with the AWS error code as the reason;
`unavailable` means the rerank model could not be built
([Querying: Rerank status](./guides/query.md#rerank-status)). The codes
read against their causes:

- `AccessDeniedException` ("You are not authorized to invoke the Rerank
  operation"): the region does not offer the rerank model, or no statement
  grants `bedrock:Rerank`.
- `ValidationException` ("... is not authorized to perform:
  bedrock:InvokeModel on resource: ...foundation-model/cohere.rerank-v3-5:0"):
  the Rerank API also checks `bedrock:InvokeModel` on the reranking model,
  and the role's `InvokeModel` grant does not list it (a scoped
  `bedrock_model_arns` without the rerank model's ARN).

Point
`aws.bedrock.region_name` at a region with the model, grant `bedrock:Rerank`
on `Resource: "*"` and `bedrock:InvokeModel` on the rerank model
([IAM permissions](./getting-started/aws.md#iam-permissions)), or set
`providers.rerank.kind: none`
([Required models](./getting-started/aws.md#required-models)).

### Why did my run use another region after an SSO login?

`AWS_REGION` fills `aws.region_name` when the configuration file leaves it
unset, and SSO helpers and CloudShell export it. The stores' region then
follows the shell, and so do the Bedrock calls while
`aws.bedrock.region_name` is `null`. A region set in the file is not moved
by the variable, and `AWS_DEFAULT_REGION` is not read. Set `aws.region_name`
(and `aws.bedrock.region_name` if Bedrock runs elsewhere) in the file
([Configuration: Environment variable overrides](./guides/configure.md#environment-variable-overrides)).

### Why does index creation fail with "the cluster has no analyzer 'nori'"?

The library maps Korean to the `nori` analyzer by default
(`indexing.opensearch.language_analyzers`, `ko: nori`), and an
Amazon OpenSearch Service domain has `nori` only once the `analysis-nori`
package is associated with it. The CDK stacks map `ko` to `cjk` for this
reason. Set `UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO=cjk` (one
entry; in a config file the key replaces the whole map, so list the other
languages' analyzers too), or associate the package
([Operator Runbook: Korean with Nori](./operations.md#korean-with-nori)).
Analyzers of a live index cannot change: a switch needs one run with
`indexing.reset: true`
([Multilingual corpora: Analyzers](./guides/multilingual.md#analyzers)).

### Why does model creation fail with "only available through a cross-region inference profile"?

Claude Sonnet 5.5 and Haiku 5.5 have no on-demand throughput: they are
invoked through a cross-region inference profile, which the model factory
resolves by listing the account's profiles. Keep
`aws.bedrock.enable_global_profile: true`, grant
`bedrock:ListInferenceProfiles`, or choose a region where a profile for the
model exists
([Required models](./getting-started/aws.md#required-models),
[Model Catalog](./reference/models.md)).

### What does `NamespaceLeaseError` mean?

With the doc-status registry on, one run writes an index namespace at a
time: it holds the namespace's write lease from its first stage to its end.
`Index namespace '...' is being written by <host>:<pid>:<id>` means another
run holds it: wait for that run and start again. A dead run's lease expires
within `indexing.lease_seconds` (default 600 s) of its last renewal, and
nothing needs cleaning up. `This run lost the write lease ...` means this
run's renewals failed or another run took the lease over: re-run once the
other run ended, and check that the registry is reachable
([Operator Runbook: Concurrent runs](./operations.md#concurrent-runs)).

### Why does a deployed ingestion stop with "needs confirmation, but stdin is not a terminal"?

The run asked a question no one could answer, before it wrote anything
(exit 1). Usually it is `This run deletes N of the M documents ...`: more
than `indexing.max_deletion_share` (default 0.5) of the scope's indexed
documents are missing from the corpus, which a partial S3 sync, an expired
prefix or a volume not mounted look like. Check the corpus first. If the
files are really gone, re-run once with `UKR_CONFIRM_DELETIONS=1` as an
environment override of the task (`--confirm-deletions` on a command line)
([Incremental indexing: Mass deletions](./guides/incremental.md#mass-deletions);
the override:
[Operator Runbook: Run a one-off ingestion task](./operations.md#run-a-one-off-ingestion-task)).
A reset that would drop another scope's seed graph documents asks too, and
takes `UKR_CONFIRM_RESET=1` (`--confirm-reset`). `--yes` answers every
question.

### Why do queries fail with `QueryTimeoutError` under load?

A query may take `search.query_timeout_seconds` (default 180 s) in all.
Under load the usual cause is the Bedrock tokens-per-minute quota: throttled
calls are retried inside the deadline, so a quota too small for the load
shows up as timeouts rather than as slower answers. Lower
`processing.max_concurrency` (for `run-eval`) or `search.max_concurrency`,
lower `models.default_max_output_tokens`, or ask for a higher quota;
the `BedrockThrottling` alarm shows whether this is the cause. Raise the
deadline only for queries that are slow on their own (`drift`, long answers)
([Operator Runbook: Query deadlines and model timeouts](./operations.md#query-deadlines-and-model-timeouts)).

### Why does an incremental run exit 2 with "holds embeddings of ..." or "was built for other ... settings"?

The live indices were built with another embedding model or dimension
("holds embeddings of ..."), other translation languages or other
`language_analyzers` ("was built for other language settings") than the
run's configuration, and the run refuses to mix them, before its first model call.
Run once with `indexing.reset: true` (it re-embeds or re-translates the
whole corpus once), or restore the previous settings
([Incremental indexing: Workflows](./guides/incremental.md#workflows)).

### How do I find out why a document is `FAILED` or `QUARANTINED`?

`FAILED` documents failed in a stage and are retried by the next run;
`QUARANTINED` ones were set aside after runs repeatedly died while
processing them, and are skipped until their file changes or the limit is
raised. The runbook shows how to list them and their cause
([Failed documents](./operations.md#failed-documents),
[Quarantined documents](./operations.md#quarantined-documents)); the rules:
[Incremental indexing](./guides/incremental.md#failed-documents).

## Common errors

Messages are abridged; the log line names the key, file or stage. A store,
model or credential error fails a query rather than answering from no context.

### Configuration and start-up

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Missing endpoint configuration for ...` / `Configuration validation error: ...` | An endpoint the run needs is unset, or a value has the wrong type | Set or fix the named key |
| `ConfigurationError: Unknown config keys: '<path>' ...` (exit 2) | A typo, or a key dropped without a replacement (a renamed key fails with `settings moved; replace <old> -> <new>`; a removed one, for example `indexing.cross_run_merge` or `aws.bedrock.guardrail.apply_to`, only logs `Config key ... is removed and ignored`; the configuration reference lists the [moved](./reference/configuration.md#moved-keys) and [removed](./reference/configuration.md#removed-keys) keys) | Fix or delete the key; `--allow-unknown-keys` loads anyway with a WARNING per key |
| Usage errors (exit 2): `--source-directory is required for pipeline execution`, `--s3-sync needs --s3-bucket-name or aws.s3.bucket_name`, `--pipeline-id is required for --resume-from-stage`, `--enabled-stages: invalid stage names ...`, `argument --resume-from-stage: Unknown pipeline stage ...` | A required flag is missing, or a stage name is unknown | Pass the flag, or a stage name from the message ([CLI Reference](./reference/cli.md#run-ingestion)) |
| `... needs confirmation, but stdin is not a terminal` | Completed stages' outputs are no longer cached under the current inputs | `--yes` (recomputes them), or a new `--pipeline-id` |
| `Stage '<stage>', upstream of this run's enabled stages, has not completed` | An earlier stage failed or lost its output | Run the earlier stages first |
| `Stage '<stage>' is in the run's enabled stages, but no selected methodology needs it (indexing.methodologies: ...)` (`run-ingestion` exit 2, before any model call) | `--enabled-stages` lists a stage that builds only what an unselected methodology reads (`summary_tree` without `raptor`, `community_detection` without `graphrag`) | Add the methodology the message names (`--methodologies`, or `indexing.methodologies`), or leave the stage out |
| `observability.enabled is true but OpenTelemetry is not installed` | The `otel` extra is missing | `uv sync --extra otel`, or turn tracing off |
| `Skipping N file(s) in '<dir>' of a type no loader reads (2 '.docx', ...)` / `No supported source files found in '<dir>'` | The `docx` extra is not installed (`.docx`), or no loader is registered for the type. The files are not indexed; an incremental run keeps what they indexed before and lists them as `failed` (`total_files_skipped`) | Install the extra the message names (rebuild the image with it), or convert the files. Until then the skipped files' indexed content stays as it was, edits to them included |
| `Not installed: 'rdflib' (for processing.graph_extraction.schema_source.kind 'owl')` (exit 2) | The `ontology` extra is missing | `uv sync --extra ontology`, or build the image with it ([Ontology schema in the image](./operations.md#ontology-schema-in-the-image)) |
| `Ontology file '...' is OWL/XML ...` / `... names a remote JSON-LD context ...` / `... more than max_entity_types ...` (exit 2) | A format the `owl` source does not read, or a selection over a cap | Follow the message: convert the file, inline the context, or narrow the selection ([Ontology mapping](./reference/ontology-mapping.md#not-followed-not-read)) |
| `Ontology file '...' holds ... (MAX_XML_TEXT_CHUNKS)` / `... (MAX_XML_LITERAL_ELEMENTS)` / `... has a line of N bytes ... (MAX_LINE_BYTES)` / `... limit on input amplification factor ...` (exit 2) | A shape the RDF/XML or N-Triples parser takes quadratic time over (refused before parsing) | Convert the file to Turtle (`robot convert --input onto.rdf --output onto.ttl`) |
| `Ontology IRIs with whitespace: ...` / `Ontology rdf:List is cyclic ...` (exit 2) | An IRI the RDF/XML or JSON-LD parser let through, or a malformed list | Fix the named IRI (`%20` for a space), or end the list with `rdf:nil` |
| `SeedGraphError: seed graph '<name>': '<file>' line N: ...` (`duplicate node id ...`, `property ... is not a string, number, boolean or null`, `not valid JSON`, `"kind" is ...`) (exit 2) | A seed record the import refuses | Fix the named line ([Seed graph formats](./reference/seed-graph-formats.md)) |
| `seed graph '<name>': edges name unknown nodes ...` / `... the edge from ... has no type ...` (exit 2) | An edge to a node the seed does not state, or with neither a `type` nor a `predicate_iri` a relationship type declares | Add the node, set `options.dangling: skip`, or type the edge |
| `seed graph '<name>': N names are shared by distinct nodes (ids ...); set on_name_collision to merge or qualify` (exit 2) | Two nodes of the seed are one entity by name | Rename them, or set `on_name_collision` |
| `seed graph '<name>': ... more nodes than max_nodes (N)` (or edges, `max_edges`) / `... exceed max_property_keys ...` / `... more than max_bytes ...` / `shard N would hold M text units, over 2000 ...` (exit 2) | The seed is over a cap (nothing is truncated) | Raise the cap, or for a shard `shards` or `max_items_per_unit` |
| `Unknown processing.seed_graphs[].kind '<kind>'; registered: csv, jsonl, rdf` / `Not installed: 'rdflib' (for processing.seed_graphs '<name>' kind 'rdf')` (exit 2) | A kind no source registers (a custom source starts from Python after importing its module), or the `ontology` extra missing | Fix the kind, or install the extra ([Seed graphs in the image or S3](./operations.md#seed-graphs-in-the-image-or-s3)) |
| `Access to seed source '<path>' denied (...)` / `Seed source '<path>' not found (no such S3 object)` (exit 2) | The ingestion task role cannot read the seed | Grant `s3:GetObject` (and `s3:ListBucket`), or move the seed ([Seed graphs in the image or S3](./operations.md#seed-graphs-in-the-image-or-s3)) |

### Credentials and models

| Message (abridged) | Meaning | Action |
|---|---|---|
| `No valid AWS credentials for Amazon Bedrock in region ...` | Missing or expired credentials | Refresh them; check `AWS_PROFILE` / `aws.profile_name` |
| `Model '<id>' is only available through a cross-region inference profile, but none resolved ...` | No on-demand throughput and no profile found | Keep `aws.bedrock.enable_global_profile: true`, grant `bedrock:ListInferenceProfiles`, or use another region |
| `Reranking failed: ...` (per query, `metadata.rerank.status: failed`) / `Reranking initialization failed; queries keep the fused order ...` (`unavailable`) | The Rerank call was denied or the model is not in the region (`AccessDeniedException`), `bedrock:InvokeModel` is not granted on the rerank model (`ValidationException`), or the rerank model could not be built | Grant `bedrock:Rerank` on `Resource: "*"` and `bedrock:InvokeModel` on the rerank model, use a region with the model, check `models.rerank` / `providers.rerank`, or set `providers.rerank.kind: none` |
| `RerankModelError: ...` | As above, with `search.reranking.required: true` | Fix the cause, or `required: false` to answer from the fused order |
| `ClientError: ... (ValidationException) ... InvokeModel ...` / `(ResourceNotFoundException)` / `NoCredentialsError` during a query | The query embedding call cannot succeed | Check `models.embedding.model_id`, `aws.bedrock.region_name`, model access and credentials |
| `EmbeddingModelError: Query embedding failed: ...` | The embedding call failed after retries (a crashed Ollama runner or vLLM engine, a persistent `InternalServerException`) | Check the embedding server and its logs; retry |
| Bedrock calls hang in a private VPC | Bedrock is called in another region than the VPC endpoint's | Keep `aws.bedrock.region_name` unset or the VPC's region |

### Stores: connection, permissions, requests

| Message (abridged) | Meaning | Action |
|---|---|---|
| `StoreConnectionError: ... Cannot connect to host ...` / `Neptune is not reachable ...` / `... status=429 ... (still failing after the client's retries)` | Refused or unresolved, closed by Neptune (restart, failover; reopened next time), or OpenSearch 429/502/503/504 for all `retry.query_store_max_attempts` attempts. An unreachable vector store fails `run-ingestion` at its store check, before any model call | Check endpoint, port, that the store runs, security groups and VPC endpoints |
| `OpenSearch did not answer a request on <index> within N s in M attempt(s)` / `Neptune did not answer within N s` | No answer within `search.store_timeout_seconds` (ingestion: Neptune fixed 180 s, item retried) | Check the store's health and load, or raise the timeout |
| `OpenSearch denied the search on <index>: ... status=403 ...` (or `401`) | Fine-grained access control: the role lacks `indices:data/read/search`; or bad credentials | Grant it (and `indices:admin/mappings/get`, `indices:monitor/settings/get`), or fix the credentials |
| `Neptune refused the connection (HTTP 403)` | No `neptune-db:connect` on the cluster, or signed for another region | Grant it (and the needed `neptune-db:*` data actions); set `aws.region_name` to the cluster's region |
| OpenSearch authentication fails with `use_iam: false` | No basic-auth credentials | Set `UKR__AWS__OPENSEARCH__USERNAME` / `__PASSWORD` (for example in `.env`) |
| `OpenSearch rejected the search on <index>: ...` | 400 `query_shard_exception`: a filter value of the wrong type. 500 `too_many_clauses` / `too_many_nested_clauses`: `indices.query.bool.max_clause_count` below 1,024 | Fix the value, or keep the limit at 1,024 or more |
| `Cannot create the OpenSearch index '<index>': the cluster has no analyzer '<name>'` (`mapper_parsing_exception`) | An analyzer in `indexing.opensearch.language_analyzers` is missing, typically `ko: nori` without the Nori package | Use one the cluster has (`ko: cjk`, the CDK default), or associate the package ([Korean with Nori](./operations.md#korean-with-nori)); re-run |
| `The search engine refused to create an index: ...` (exit 2, before any model call) | Before its first model call, `run-ingestion` creates and deletes a probe index per index the run will create (all of them on a full build or reset, the missing ones on an incremental run), named `<extraction-records alias>-settings-probe-<12 hex digits>`; the store refused one (the error names the index it stood in for, `(a probe of '<alias>')`): an `indexing.opensearch.index_settings` key, the `vector_search` method, `vector_mode` / `compression_level` (`unknown parameter [mode]`: OpenSearch before 2.17) or an analyzer, or (`maximum shards open`) no room for the new indices' shards | Fix or remove the named setting, or delete unneeded indices or raise `cluster.max_shards_per_node`; re-run. A probe a stopped run left (an unaliased index of that name) is deleted by the namespace's next run |

### Ingestion runs and the doc-status registry

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Pipeline failed at stage(s): ... First error (<stage>): ...` | A stage failed; exit 1. The message names the first failed stage's error, and a `Next step:` line the pipeline id to resume with | Fix the cause the error names and resume with the same `--pipeline-id` (`--force-rebuild` starts clean) |
| `The embedding model '<id>' returns vectors of dimension <n>, but the indices are mapped with <m> (models.embedding.dimension)` (`run-ingestion` exit 2, before any extraction) | The model (often a local server's) returns another dimension than the configured one; the run embeds one probe text before the first model stage to find out | Set `models.embedding.dimension` (`--embedding-dimension`) to `<n>`, or use a model that returns `<m>`; an index already built with another dimension needs a new namespace (`--suffix`) or `--reset` |
| `Pipeline resume failed: ...` (`PipelineResumeError`, exit 1, before any model call) | The pipeline id cannot be resumed as asked: a stage this run must rerun is not enabled (the message names it), an explicit `--resume-from-stage` past a stage whose cached output is gone, or its saved state could not be read | Enable the named stage or resume from it (`--resume-from-stage <stage>`); for unreadable state, a new `--pipeline-id` or `--force-rebuild` |
| `CacheSyncError` | The S3 stage-cache download or upload failed | Check bucket permissions and `aws.s3.encryption`; resume |
| `DocStatusRegistryError` | The registry table is missing, not accessible, or throttled | Create it or fix `aws.dynamodb.table_name`, grant the permissions the message names; resume |
| `The reset of index namespace '...' drops the seed graph documents of source scope(s) ...` (or `rebuilds it from this run's seed graphs alone ...`, or, with the registry off, `No source document reached this run, ... rebuild the index '...' from the seed graphs alone, replacing what it holds`) `; continue?` with `... needs confirmation, but stdin is not a terminal` or `No one can confirm it (run-ingestion --confirm-reset does)` (exit 1, before any write) | A reset would drop another scope's seed graph documents, or, from seeds alone, another scope's documents; or, with the registry off, seeds alone would replace an index that holds items; and no one answered yes | Re-run with `--confirm-reset` or `UKR_CONFIRM_RESET=1` ([Seed graphs in the image or S3](./operations.md#seed-graphs-in-the-image-or-s3)), or pass `confirm=` to `DataIngestionPipeline` (a no raises `ConfirmationDeclinedError`) |
| `This run deletes N of the M documents the doc-status registry holds for source scope '...' (P%, over indexing.max_deletion_share S) ...; continue?` with `... needs confirmation, but stdin is not a terminal` or `No one can confirm it (run-ingestion --confirm-deletions does)` (exit 1, at `document_loading`, before any write) | More than `indexing.max_deletion_share` of the scope's indexed documents are missing from the corpus | Check the corpus (sync, prefix, mount) and re-run. If the files are really gone, re-run with `--confirm-deletions` (or `UKR_CONFIRM_DELETIONS=1`, or `--yes`) ([Incremental indexing: Mass deletions](./guides/incremental.md#mass-deletions)) |
| `No source document reached this run, only its seed graphs, but the doc-status registry holds N documents of source scope '...'` (exit 1, at `document_loading`, before any write) | The corpus is empty (a volume not mounted, an S3 sync that copied nothing) while seeds are configured and the scope has documents | Put the corpus in place and re-run. To delete the scope's documents, retire it from a run of another scope (`--retire-source-scope`); for seeds of their own, use a seed-only scope ([Seed graphs: Lifecycle](./guides/seed-graphs.md#lifecycle)) |
| `Incremental indexing is enabled but no document delta was computed` | Indexing ran without a registry diff (loading failed under `continue_on_error`) | Re-run from `document_loading` once the registry is reachable |
| `NamespaceLeaseError: Index namespace '...' is being written by ...` | Another run holds the write lease | Wait; a dead run's lease expires within `indexing.lease_seconds` ([Concurrent runs](./operations.md#concurrent-runs)) |
| `NamespaceLeaseError: This run lost the write lease ...` | Renewals failed until the lease expired, or another run took it | Re-run after the other run; check the registry is reachable |
| `ConfigurationError: Doc-status registry: DynamoDB table '...' has the key schema ... of an earlier release` | A table keyed by `doc_id` alone | Point `aws.dynamodb.table_name` at a new table (or the stack's), and run once with `indexing.reset: true` ([Incremental indexing: Enable it](./guides/incremental.md#enable-it)) |
| `ConfigurationError: The OpenSearch index '...' records no embedding settings ...` | The index has no recorded embedding settings (`_meta`), which every index this version builds carries | Run once with `indexing.reset: true` |
| `ConfigurationError: The OpenSearch index '...' maps 'attributes.seed_properties' field by field ...` (exit 2, before any model call) | The live entities index maps seed graph properties (`attributes.seed_properties`), which this version stores unmapped | Run once with `indexing.reset: true` |
| `ConfigurationError: The OpenSearch index '...' stores the vectors of '...' as ...` (exit 2, before any model call) | `indexing.opensearch.vector_mode` / `compression_level` changed since the live index was built | Run once with `indexing.reset: true`, or restore the previous values ([On-disk vector storage](./operations.md#on-disk-vector-storage)) |
| `ConfigurationError: The OpenSearch index '...' holds embeddings of ...` / `... was built for other language settings ...` (exit 2, before any model call) | An incremental run's embedding model or dimension, translation languages or `language_analyzers` differ from those the live index was built with (also a run with nothing to extract: no change, or only deletions) | Run once with `indexing.reset: true` (it re-embeds or re-translates the whole corpus once), or restore the previous settings |
| `ConfigurationError: The OpenSearch index '...' was built by a run with the doc-status registry off ...` (exit 2, before any model call) | The live index was built by a run with `providers.doc_status.enabled: false`, so the registry this run turned on does not list the documents it holds (also on `backend: local`) | Run once with `indexing.reset: true` and the registry on, which rebuilds the stores and fills the registry; or keep the registry off for this suffix |
| `The doc-status registry lists indexed documents of this namespace whose entities the graph store does not hold` (or `whose text units the vector store does not hold`) | A store lost what the registry records (a restarted Gremlin Server, `down -v`, a restored graph, deleted indices, a moved `vector.sqlite3`) | Re-run with `indexing.reset: true`, or delete the namespace's records; locally delete the three store files together or restore the moved one |

### Queries

| Message (abridged) | Meaning | Action |
|---|---|---|
| `No indices found for suffix '<suffix>' ...` | Nothing was ingested under that suffix | Ingest, or query the corpus's suffix |
| `QueryTimeoutError: The query did not finish within N s (search.query_timeout_seconds)` / `The answer stream sent no first chunk within N s (search.model_timeout_seconds)` | The query, or a stream's first chunk, ran past its deadline: a stalled or throttled model call, a slow store, or a long strategy | [Query deadlines and model timeouts](./operations.md#query-deadlines-and-model-timeouts) |
| `Read timeout on endpoint URL ...` (`ReadTimeoutError`) on a query | A query-time Bedrock request got no answer within `search.model_timeout_seconds`, on every attempt | Check Bedrock's health and the VPC endpoint; raise `search.model_timeout_seconds` if answers are legitimately long |
| `InvalidFilterError: Unknown filter key(s) ...` / `... has no value (null)` / `... not a scalar` / `... lists N values ...` | No store the strategy reads declares the filter key; a filter value is null (`None` from Python; alone, in a list or as a range bound: OpenSearch rejects such a query); a list holds a list or mapping; or a list filter holds more values than one `terms` query takes (`index.max_terms_count`, default 65,536) | Use a key the message lists; drop the null or give the value to match; list strings, numbers or booleans; narrow the list (or raise `max_terms_count` in `indexing.opensearch.index_settings` and reindex) |
| `MethodologyNotIndexedError: Search strategy '<strategy>' reads the <methodology> methodology, which the index namespace '<suffix>' does not hold ...` (`run-rag` and `run-eval` exit 2) | The namespace was last written without that methodology in `indexing.methodologies` (or with a switch that turns off what it needs, such as `graph.community_detection.enabled: false`; for `global` and `drift` also `graph.community_detection.report_generation.enabled: false`, which the message names) | Use a strategy the message lists, or add the methodology and re-run ingestion ([methodology profiles](./guides/ingest.md#methodology-profiles)). For `tree`, `run-ingestion --methodologies <yours>,raptor --enabled-stages summary_tree` builds only the tree; without `raptor` selected the stage is refused ([summary tree](./guides/ingest.md#a-raptor-summary-tree-opt-in)) |
| `Query keyword extraction (model role entity_extraction, '<model>') returned no usable keyword JSON: ...` (`run-rag` exit 1) | The model answering `mix`, `hybrid` (and `iterative` over them) keyword extraction wrote no JSON object with `high_level_keywords` / `low_level_keywords` (a small local model, a cut-off answer) | Set `models.roles.entity_extraction` to a model that follows the JSON instructions; or `processing.ignore_errors: true` to search without keywords; or `--disable-query-processing` |
| LightRAG `mix` / `hybrid` returns nothing | No relationship index, or no keywords extracted | Check the relationships index exists; short queries fall back to the raw query only under `raw_query_fallback_max_len` |

### Local mode

| Message (abridged) | Meaning | Action |
|---|---|---|
| `... stayed locked ...` (`database is locked`) | Another process held the SQLite write lock longer than `search.store_timeout_seconds` | One ingestion per store file at a time, or raise the timeout; use a local disk |
| `... processed N inputs but produced 0 outputs`, with `Connection refused` in the log | Ollama or the OpenAI-compatible server is not reachable at `providers.llm.base_url` / `providers.embedding.base_url` | Start it (`ollama serve`, or `docker compose -f docker/compose.local.yaml --profile models up -d --wait`), check `base_url` (`--model-base-url`), pull the models |
| `AuthenticationError` / `NotFoundError` (401 / 404), `APIConnectionError` / `APITimeoutError`, `ResponseError: model ... not found` | Key refused, model not served, or server unreachable | Check `base_url` and `api_key` under `providers.llm` / `providers.embedding` and the model names (`models.default`, `models.embedding.model_id`); pull the models |
