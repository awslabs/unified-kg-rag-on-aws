# Using the library from Python

The CLIs wrap two public classes, `DataIngestionPipeline` and
`GraphRAGChain`. Import them, and everything else, from the package root:
the names in `unified_kg_rag.__all__` are the public API, other module paths
are internal and may change. `get_config` applies the same file and
environment layers as the CLIs
([Configuration](./configure.md#how-configuration-is-loaded)).

## Ingest and query

```python
from unified_kg_rag import (
    DataIngestionPipeline,
    GraphRAGChain,
    PipelineConfig,
    PipelineStageStatus,
    RAGInput,
    get_config,
)

config = get_config("config.yaml")

# Ingest. PipelineConfig carries the run-ingestion run options (pipeline_id,
# cache directory, S3 sync, enabled stages, force_rebuild). run() ingests the
# directory it is given: parsing, loading and the registry scope all use it.
pipeline = DataIngestionPipeline(config, PipelineConfig(pipeline_id="run-001"))
try:
    context = pipeline.run("./source")
    if context.status == PipelineStageStatus.FAILED:
        raise SystemExit("ingestion failed")
finally:
    pipeline.close()  # releases the indexers' Neptune/OpenSearch connections

# Query. One chain serves many queries, including concurrent ones.
chain = GraphRAGChain(config)
try:
    output = chain.invoke(RAGInput(query="How are Alice and Acme related?", search_strategy="mix"))
    print(output.answer)
    for source in output.sources:  # only what the answer model saw
        print(source)
finally:
    chain.close()
```

- `RAGInput` takes the `run-rag` options as fields (`search_strategy`,
  `search_type`, `top_k`, `suffix`
  ([index suffix](../concepts/scopes-and-namespaces.md#index-suffix)),
  `filters`, `conversation_id`, `use_memory`, `history`, `target_language`);
  a plain dict works too. What each option does: [Querying](./query.md);
  `history` is the conversation of a caller that keeps it itself
  ([History from the caller](./query.md#history-from-the-caller)).
- `output.citations` lists the numbered context sections the answer's `[n]`
  markers refer to, with their documents and pages, and whether the answer
  cites each; it is empty for a strategy that does not cite (`global` and
  `drift` by default, [Citations](./query.md#citations)).
- `chain.search(...)` (`await chain.asearch(...)`) retrieves without
  answering and returns a `SearchOutput` (`search_results`,
  `processed_query`, `metadata`).

## Documents from memory

With the doc-status registry on (`providers.doc_status.enabled: true`), a
pipeline adds, updates and deletes documents given from memory through the
same incremental path as `run`: each call is one run (namespace lease,
stage cache, commit), and it names its changes, so the source scope's other
documents are kept (a run over a directory deletes what the directory
lacks).

```python
from unified_kg_rag import DataIngestionPipeline, InputDocument, PipelineConfig

pipeline = DataIngestionPipeline(config, PipelineConfig())
try:
    pipeline.insert_documents([
        InputDocument(id="contracts/2024-07.txt", text="...", filters={"category": "contract"}),
        {"id": "ticket-1832", "text": "...", "metadata": {"author": "support"}},
    ])
    pipeline.insert_documents([{"id": "ticket-1832", "text": "edited ..."}])  # re-extracted
    pipeline.delete_documents(["contracts/2024-07.txt"])
finally:
    pipeline.close()
```

- An `id` is the document's key in its source scope, as a relative path
  is in a corpus directory (no `..`, no leading `/`); sources cite the
  document by `file_name`, else the id's last segment. `filters` are its
  document attributes (queries filter on `attr_<key>`).
- The documents belong to the source scope `"api"` unless `source_scope=`
  names another; a directory corpus's documents are deleted by their
  relative paths with `source_scope=` its scope (its `source_scope`, or
  its absolute directory).
- Both return the run's `PipelineContext`; `incremental_delta` lists what
  was new, changed, deleted and `kept`. `delete_documents` raises
  `DocumentNotFoundError`, before any write, for an id the registry does
  not hold in the scope. Without the registry, or with `indexing.reset`,
  both raise `ConfigurationError`: a full build of the given documents
  would replace the rest of the namespace.
- A methodology added to `indexing.methodologies` that builds what the
  namespace does not hold (`graphrag`'s communities and reports on a
  `lightrag` namespace) is refused too, before any model call: adding it
  rebuilds the namespace from every document, and a call holds only its
  own. Keep the methodologies the namespace holds for these calls, or add
  it with one `run` over a directory holding every document of the
  namespace as a file named by its id, with `source_scope` the documents'
  scope (`"api"`): it reuses the stored extractions, but deletes a
  document it does not read (an id without a file extension the parser
  reads). A namespace of its own (`index_value`) extracts them again.

## Inspect and export a namespace

`describe_namespace(config, suffix)` returns what an index namespace holds
(methodologies, servable strategies, indices with their item counts and
embedding model, the build record, the registry's records by status), and
`export_graph(config, path, suffix=..., format="jsonl" | "graphml")`
streams its graph with the provenance of every item to a file; both read
the stores only. `run-describe` prints them
([Inspect and export](./inspect.md)).

## Results and errors

- A failed stage sets the returned `PipelineContext`'s `status` to `FAILED`
  rather than raising, so check it as above.
- The errors to tell apart are exported from the package root:
  `ConfigurationError` (a setting to fix; `StoreRequestRejectedError` among
  them), `InvalidFilterError`, `IndexNotFoundError`, `StoreConnectionError`,
  `QueryTimeoutError`, `NamespaceLeaseError`, `DocumentNotFoundError`,
  `PipelineExecutionError` and
  `PipelineInterrupted` (a `KeyboardInterrupt`); all but the last are
  `GraphRAGException`s. `DataIngestionPipeline(...)` raises `SeedGraphError`
  (a `ConfigurationError`) for a seed graph it cannot import.
- `DataIngestionPipeline.run` raises `ConfigurationError`,
  `StoreConnectionError`, `NamespaceLeaseError` and
  `ConfirmationDeclinedError` as they are, and wraps any other crash outside
  a stage in `PipelineExecutionError`. A model client's own transport error
  (`httpx.ConnectError` from a local model server, say) is raised as is.
- A run asks before some writes, through two callables
  (`callable(question) -> bool`) given to `DataIngestionPipeline(...)`:
  `confirm=` answers a reset that would drop another source scope's seed
  graph documents (or, from a corpus of seed graphs alone, its documents)
  and, with the registry off, a rebuild from seed graphs alone of an index
  that holds items ([Seed graphs: Lifecycle](./seed-graphs.md#lifecycle));
  `confirm_deletions=` answers an incremental run that deletes more than
  `indexing.max_deletion_share` of its source scope's documents
  ([Incremental indexing: Mass deletions](./incremental.md#mass-deletions)).
  Without the callable, or on a no, `run` raises `ConfirmationDeclinedError`
  before it writes.

What a message means and what to do about it:
[Troubleshooting](../troubleshooting.md).

## Async code

Use `await chain.ainvoke(...)` and `await chain.aclose()`, and size the
loop's executor once at startup:
`configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`.
A process serving concurrent queries on the embedded stores should also run
with `OPENBLAS_NUM_THREADS=1` in its environment, as `run-eval` and `run-rag`
do ([local mode](./local-mode.md#differences-from-aws-mode)).

## Closing

Always `close()` / `aclose()` what you build: both classes open connections
on first use and keep them until garbage collection otherwise. A sync call
still running on another thread when the chain closes raises
`StoreConnectionError`; a later call reconnects.

## Logging

Importing the package configures no logging handler;
`unified_kg_rag.setup_logging(config)` gives the CLI's logging. Without it,
structured fields (including `structlog.contextvars` bindings) reach your
handlers as record attributes, a trailing `_` added to one named like a
`LogRecord` attribute (`name`, `msg`, ...).

`setup_logging` changes process-wide state, so call it only where the
package owns logging: it replaces the structlog configuration, sets the root
logger's level to `logging.level` and the `logging.library_levels` loggers'
levels, and adds a stdout handler (plus the dated file with
`logging.log_to_file`) to the root logger. Handlers already on the root
logger stay: they receive every record too, so the console shows each line
twice when one of them writes there, and for the package's records they get
the plain message and the exception, not the structured fields. A repeated
call replaces only the handlers it added. What the logs contain:
[Configuration](./configure.md#what-the-logs-contain).

## Extensions and injected backends

Extensions import from the package root too: the ports (`LLMFactoryPort`,
`GraphIndexer`, `DocStatusPort`, ...), `Providers`, `register_provider`,
`BaseSearchStrategy` with `register_strategy`, `ParserFactory` and
`register_renderer` (a new strategy also needs a `SearchStrategy` member,
so it is an in-tree change). Both constructors take injected backends, either as a
`providers=Providers(config, ...)` bundle or as individual arguments, not
both: `doc_status=`, `vector_indexer=` and `graph_indexer=` on the pipeline,
`model_factory=` and `retriever_builders=` on the chain (which also takes
`memory_manager=` and `guard=`); see
[Design Doc §15](./extend.md#15-extension-guide). An indexer passed as
`vector_indexer=` / `graph_indexer=` stays yours to close: the pipeline never
closes it.
