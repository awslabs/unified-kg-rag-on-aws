# Telemetry reference

The spans the package emits when tracing is on (how to turn it on and where to
send the traces: [Observability](../guides/observability.md)), the audit
record it writes per query, and the model usage it counts with tracing on or
off.

## Trace shape

Every span of a query or an ingestion run carries `kg_rag.namespace`, the
namespace it reads or writes (the tenant boundary), so a trace backend
filters by it.

| Span | Parent | Attributes |
|---|---|---|
| `kg_rag.ingestion.run` | root | `kg_rag.pipeline_id`, `kg_rag.mode` (`full`, `incremental`) and an incremental run's `kg_rag.delta.new` / `.changed` / `.deleted` counts; status `ERROR` when the run failed |
| `kg_rag.ingestion.stage` | run | One per stage: input/output counts, status, the stage's numeric metrics as `kg_rag.stage.<metric>`, `error.type` and the exception when it crashed |
| `kg_rag.batch` | stage, or root for `run-eval` answer generation | Item and failure counts; in `run-eval` the run's `kg_rag.eval.strategy` and `kg_rag.eval.repeat` (1-based), as on every span of the run |
| `kg_rag.mcp.tool` | root | One per MCP tool call (`run-mcp-server`): `mcp.tool.name`, `kg_rag.records`, `kg_rag.cancelled` (its client left or the server stopped: no error status); with `capture_content` the redacted arguments as `input.value`. A `kg_search` or `kg_answer` call's `kg_rag.query` span nests under it |
| `kg_rag.console.ask` | root | One per console question: `kg_rag.search_strategy`, `kg_rag.query_id` (the id its `kg_rag.query` span's log lines and result carry); the query span nests under it |
| `kg_rag.query` | root, the `run-eval` batch, an MCP tool call, or a console question | `kg_rag.query_id`, `session.id` and `gen_ai.conversation.id` (the conversation id), `user.id`, `kg_rag.mode` (`rag`, or `search` for a retrieval-only search), `kg_rag.top_k`, `kg_rag.use_memory`, `kg_rag.requested_strategy` and `kg_rag.strategy` (the one that ran; of a stream stopped early too, once routed), `kg_rag.blocked`, `kg_rag.degraded` (answered without a step), `kg_rag.query_limited` (its limits cut it short: the `query_limited` metric), `kg_rag.error_class` (of an error answer under `processing.ignore_errors`), `kg_rag.effort` (`RAGInput.effort`) and `kg_rag.limits.max_model_calls` / `kg_rag.limits.max_input_tokens` (the query's limits in force), when set; once it ends, `kg_rag.error` (an error answer), `kg_rag.abstained`, an answer's `kg_rag.sources` (how many the answer model was given) and `kg_rag.cited_source_ids` (the ids of the sections it cites, up to 50), and `kg_rag.cost_usd` when `models.prices` prices every model it called; with `capture_content` the redacted query as `input.value` and the answer as `output.value` (a streamed one too). Its `kg_rag.query_id` and trace id are the result's `metadata.query_id` and `metadata.trace_id`, and its log lines carry `query_id` and `namespace` |
| `kg_rag.strategy` | query | `kg_rag.rerank_status` for the whole query |
| `kg_rag.retrieve` | strategy | `kg_rag.hits` and the hits per result type, and the first 50 results' ids and scores in rank order (`kg_rag.source_ids`, `kg_rag.scores`: parallel lists) |
| `kg_rag.fusion` | strategy | `kg_rag.streams`, in/out counts (`kg_rag.results_in`, `kg_rag.results_out`), `kg_rag.rerank_enabled`, `kg_rag.reranked`, the step's `kg_rag.rerank_status`, and the fused (and reranked) order's first 50 ids and scores (`kg_rag.source_ids`, `kg_rag.scores`) |
| `kg_rag.rerank` | fusion | `kg_rag.rerank.status` (`ok`, `empty`, `error`), document counts |
| `kg_rag.context_build` | query (`drift`: the round's actions span) | Tokens used vs the token budget |
| `kg_rag.global.select`, `kg_rag.global.map` | strategy (`global`) | Select, once per query: `kg_rag.depth` (configured), `kg_rag.cut` (the cut mapped after falling back to a shallower one), `kg_rag.reports`, `kg_rag.dynamic`. Map: `kg_rag.reports`, `kg_rag.map_calls`, `kg_rag.report_tokens` |
| `kg_rag.iterative.round` | strategy (`iterative`) | One per round: round 0 runs the base strategy (`kg_rag.round`, `kg_rag.base`); a follow-up round has `kg_rag.round`, `kg_rag.queries` (its rewritten searches), `kg_rag.bridge_entities`, `kg_rag.base_rerun` and its streams' hits (`kg_rag.hits.<stream>`) |
| `kg_rag.graph_query.plan`, `.execute`, `.fallback` | strategy (`structured`) | Plan: `kg_rag.graph_query.status` (`planned` or `fallback`), `.reason` (a fixed code), `.repairs`. Execute: `.status`, `.reason`, `.operation`, `.rows`, `.truncated`, `.backend`. Fallback, when the query falls back: `kg_rag.strategy` (the strategy it runs) |
| `kg_rag.drift.primer`, `kg_rag.drift.actions` | strategy (`drift`) | Primer: `kg_rag.reports`, `kg_rag.answers`. One actions span per round (`kg_rag.depth`, `kg_rag.actions`), with the round's local searches under it |
| `kg_rag.guardrail` | query | One per screened query or answer: `aws.bedrock.guardrail.id`, `.source` (`input`, `output`), `.action` (`none`, `masked`, `blocked`), `.intervened`, `.policies`; with the contextual grounding check, `aws.bedrock.guardrail.grounding.checked`, `.grounding.failed`, `.grounding.score`, `aws.bedrock.guardrail.relevance.score` |
| `kg_rag.evaluate` | root | One per evaluator: `kg_rag.evaluator`, `kg_rag.items`, `kg_rag.eval.strategy`, `kg_rag.eval.repeat`; the judge's model spans under it |
| `chat <model>`, `embeddings <model>` | the calling span | OpenTelemetry GenAI conventions, kind `CLIENT` (as Bedrock rerank spans): `gen_ai.request.model`, `.max_tokens`, `.temperature`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.usage.cache_read.input_tokens` / `cache_creation.input_tokens` when Bedrock reports prompt-cache use, `gen_ai.usage.reasoning.output_tokens` when the response reports reasoning tokens (LangChain's `output_token_details.reasoning`, a part of the output tokens), finish reasons, `error.type`, `kg_rag.model.purpose` (what the model was built for: `query`, `ingestion`, `evaluation`). No cost: the query span's `kg_rag.cost_usd` prices the whole query |

Every span's resource carries `service.name` (`observability.service_name`,
unless `OTEL_SERVICE_NAME` sets it) and `service.version` (the package
version, unless `OTEL_RESOURCE_ATTRIBUTES` sets `service.version`), so a
trace backend tells one release's traces from another's.

With `capture_content`, the answer model's `chat <model>` span keeps the
answer as generated, even when the guardrail then blocks or masks it, and
the `kg_rag.guardrail` span comes from the Bedrock guardrail adapter only (a
custom `ContentGuardPort` emits none).

`kg_rag.rerank.status` is the outcome of one Rerank call (`error`: the query
fell back to the fused order; `empty`: nothing usable came back), while
`kg_rag.rerank_status` is what reranking did, set even without a call:
`applied`, `disabled`, `unavailable` (no rerank model), `skipped` (nothing to
rerank) or `failed`; on the strategy span it is the worst step, as in
[`metadata.rerank.status`](../guides/query.md#rerank-status).

On export, spans keep their `gen_ai.*` and `kg_rag.*` attributes and gain the
vendor's names: Langfuse `langfuse.observation.type` (`generation` for model
calls, `embedding`, `retriever` for retrieval and rerank, `span` otherwise)
and, with `capture_content`, `langfuse.observation.input` / `output`; Phoenix
`openinference.span.kind`, `llm.model_name`, `llm.provider`,
`llm.token_count.*` and, with `capture_content`, `input.value` /
`output.value`.

Known limits:

- The async batch path records one span per batch, not per chunk.
- A model call cancelled during `ainvoke` (its task cancelled, or the call cut
  off by `asyncio.wait_for` / `asyncio.timeout`, as RAGAS's `ragas_timeout`
  does) gets no LangChain callback; its model span is ended as `ERROR`
  (`error.type: CancelledError`) when the task that started it ends. A stream
  the caller stops reading early ends its span normally.
- With `otlp`, `langfuse` or `phoenix` the first backend the process installs
  also becomes OpenTelemetry's global tracer provider, which can be set only
  once: after a later configuration with other settings, the package's spans
  use the new exporter, but spans other libraries emit through the global API
  are dropped.

## Audit record

Every query writes one audit record, tracing on or off: an INFO record of the
`unified_kg_rag.audit` logger whose event is `query_audit` and whose fields
are its own keys (with `logging.log_format: structured`, one JSON object per
line, next to the record's `timestamp`, `level` and `logger`). It is written
where every query ends in `GraphRAGChain`, so `run-rag`, `run-eval`, the MCP
server, the console and an embedded chain write the same record: an answer
and a search (`kg_search`, `run-rag --mode search`), one that failed, timed
out or was refused, a stream its reader closed early, and a query its caller cancelled (an MCP
client that disconnected, a task cancelled by its host: `outcome`
`stopped`, its usage up to there; `failed` with `error_class` `timeout`
when its deadline had passed, as at the MCP server's backstop). An MCP
`kg_answer` that timed out waiting for a call slot, before the chain ran
it, gets its record from the server.

It holds ids, counts and digests, never the text of the question, the answer
or a source: the question and the answer are their length and hex SHA-256
(`question`, `answer`), so a stored answer can be checked against its
record. A digest is no anonymization: a short or common question can be
found by hashing guesses, so protect the logs as you would the questions.
`user_id` is recorded as given (`RAGInput.user_id`, or the
[trusted identity header](../guides/mcp.md#http-and-remote-deployment)): pass
an opaque id.

| Field | Meaning |
|---|---|
| `record_version` | `1`; raised when a field is renamed, removed or changes meaning (an added optional field keeps it) |
| `query_id`, `trace_id` | The ids of the query's log lines and trace (`metadata.query_id`, `metadata.trace_id`) |
| `finished_at`, `latency_ms` | When it ended (UTC) and how long it took, its admission included |
| `mode` | `rag` (an answer) or `search` |
| `namespace`, `user_id`, `conversation_id` | Its index namespace and `RAGInput.user_id` / `conversation_id` |
| `requested_strategy`, `strategy` | The strategy asked for and the one that ran (`null` when none did: a blocked or failed query, a stream stopped before its route) |
| `outcome` | `answered`, `abstained`, `blocked`, `error` (the error answer of `processing.ignore_errors`), `failed` (raised), `refused` (a caller's error, `ErrorClass.is_caller_error`) or `stopped` (its reader stopped) |
| `error_class`, `degraded`, `query_limited` | A failure's class, whether it answered without a step, whether its limits cut it short (as the query metrics) |
| `package_version`, `config_fingerprint` | The package that answered, and the hex SHA-256 of every setting that can change an answer (`run-eval`'s manifest `retrieval_config_sha256`: one value for one configuration) |
| `build` | The index build it read, from its namespace's build record as its admission read it (cached for up to 300 s): `pipeline_id`, `mode`, `built_at`, `package_version`, `extraction_fingerprint`, and an incremental build's `full_build_pipeline_id`; `null` without a record or for a refused input |
| `question`, `answer` | `{length, sha256}` of the question asked and the answer returned (`null`: none) |
| `evidence` | `sources` (how many the answer model was given, or a search returned) and their `source_ids`, then the context sections the answer cites (`[n]`) as `cited_source_ids`, `cited_text_unit_ids` and `cited_document_ids`; at most 50 ids each, `truncated` when one was cut; empty cited lists for a strategy whose citations are off (`search.citations.strategies`) |
| `models`, `usage` | The model ids it sent requests to, and its usage (the `metadata["usage"]` counts below) |
| `cost_usd`, `unpriced_models` | Its USD cost by `models.prices`, `null` with a model that has no price there, named in `unpriced_models` |

Nothing is built while the logger is off for INFO: a CLI writes the records
at the default `logging.level` (`INFO`), `logging.library_levels:
{unified_kg_rag.audit: WARNING}` turns them off, and an embedding
application enables them with `logging.getLogger("unified_kg_rag.audit")`
at INFO and a handler of its own (to keep them apart from the other logs,
add the handler there and set `propagate = False`). On the CDK stacks the
records go to the task's CloudWatch Logs group with the other logs, under
its retention period; to keep them longer, subscribe a filter on
`"query_audit"` to a destination of its own.

## Usage counters

Model usage is counted with tracing on or off: every chat model the factories
build records each call (the tokens its response reports) for the query or
ingestion stage that made it. `RAGOutput.metadata["usage"]` (and
`SearchOutput.metadata["usage"]`) holds the query's `llm_calls`,
`input_tokens`, `output_tokens`, `cache_hits`, `estimated_calls`,
`cache_read_input_tokens`, `cache_write_input_tokens`, `request_retries`,
`sent_requests`, `throttled_requests`, `failed_calls`, `rerank_calls` and `rerank_search_units` (the reranker's
requests and their billed search units, one per 100 documents a request
ranks, a document of more than 500 estimated tokens with the query counting
as several, as Amazon Bedrock bills them; not model calls), `embedding_calls`,
`embedding_input_tokens` and `embedding_estimated_calls` (the embedding
model's requests that returned, every provider's, one per request the adapter
sends, and their input tokens: Titan's own `inputTextTokenCount`, else the
script-aware estimate, those calls counted in `embedding_estimated_calls`; not
model calls, not in `input_tokens`, priced by `embedding_per_million_tokens`
in `models.prices`), and the same counts
`by_role` (the model's purpose: `query`, `ingestion`, `evaluation`;
`embedding` and `rerank` for the embedding model's and reranker's requests) and
`by_model` (the model id the factory resolved). `run-ingestion
--output-format json` reports `token_usage` (the stages this run executed,
with `stages.<stage>`), `--metrics-sink cloudwatch` emits it per stage
([runbook](../operations.md#alarms)), and `run-eval` records it per answer
(`EvaluationResult.usage`) and per run (`summary.usage`). A response the
ingestion response cache served, a chunk-boundary or community report
answer the extraction store served, a description summary a full build took
from its stored copy, or a query translation or entity or keyword extraction
the chain's extraction cache served (`search.query_extraction_cache_size`), counts as
a cache hit, not a call; a call that
failed before its response started is not in `llm_calls` but in
`failed_calls`; retries are calls: each attempt sent counts. A query's
limits (`search.query_max_model_calls`, `search.query_max_input_tokens`)
count a retried call once instead, at its latest attempt, so a transient
fault under a binding limit is retried rather than refused; an ingestion
run's budget (`processing.max_run_model_calls`, `processing.max_run_usd`)
bounds what was sent and counts every attempt.
`input_tokens` includes the prompt-cache reads and writes, which are also
counted apart (`cache_read_input_tokens`, `cache_write_input_tokens`, from
Bedrock's `cacheReadInputTokens` / `cacheWriteInputTokens`) since they are
billed at other rates. `request_retries` counts the requests botocore resent
inside one call (its `RetryAttempts`): the call counts once in `llm_calls`,
but a request resent after a read timeout or a server error is a Bedrock
invocation of its own (a resent throttled one is not). `sent_requests`
counts every request sent to Bedrock and `throttled_requests` those Bedrock
throttled, each attempt once (a botocore hook of the model clients: the
language, embedding and rerank models'; not the CountTokens client, whose
free calls would dilute the share, nor the guardrail's; a throttling error
inside a response stream is not counted). A query embedding concurrent
queries share (the same text, embedded once for all of them) counts once,
in the earliest of them still waiting for it: the query that sent it, then,
once that one stopped waiting (it timed out, its reader left), the next;
the others reuse it as they would a cached one. So Bedrock's
CloudWatch `Invocations` can exceed `llm_calls` by up to `request_retries +
failed_calls`, plus any calls made outside a usage scope (the `run-eval`
warm-up queries), plus every embedding call: the usage counts language
model calls only (an embedding span has its text and character counts, no
tokens or price). A call that has not answered when the
usage is read (an `auto` query's unread extraction, still running in the
background), was cancelled in flight (by `search.query_timeout_seconds`,
say) or failed once its response had started streaming (a stream cut
mid-answer, which the query chain retries; also an Ollama `invoke`, whose
client streams inside the call) was sent and is billed, so it
counts in `llm_calls` and in
`estimated_calls`, at an estimate of its input tokens; its output tokens
are 0, but for a failed stream's, estimated from the text it streamed.
One that answers later is settled in `run-eval`'s run usage. A call that
answers without reporting usage (a local server that returns none, say)
stays estimated the same way, rather than reading as free, with its output
tokens estimated from its text and its resent requests counted.
`stream` / `astream` yield text only; the `QueryResult` of `aquery_events`
carries the usage as `ainvoke`'s output does.

A query's USD is `metadata["cost_usd"]` (`RAGOutput` and `SearchOutput`),
its usage priced by `models.prices` as every USD figure is
(`domain/pricing.py`): `null` when a model it called has no price there,
named in `metadata["unpriced_models"]`. The query span carries it as
`kg_rag.cost_usd`, the audit record as `cost_usd`, and `--metrics-sink
cloudwatch` as `query_cost_usd`, beside `query_input_tokens` and
`query_output_tokens` ([runbook](../operations.md#alarms)).
