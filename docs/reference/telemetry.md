# Telemetry reference

The spans the package emits when tracing is on (how to turn it on and where to
send the traces: [Observability](../guides/observability.md)), and the model
usage it counts with tracing on or off.

## Trace shape

Every span of a query or an ingestion run carries `kg_rag.namespace`, the
index suffix it reads or writes (the tenant boundary), so a trace backend
filters by it.

| Span | Parent | Attributes |
|---|---|---|
| `kg_rag.ingestion.run` | root | `kg_rag.pipeline_id`, `kg_rag.mode` (`full`, `incremental`) and an incremental run's `kg_rag.delta.new` / `.changed` / `.deleted` counts; status `ERROR` when the run failed |
| `kg_rag.ingestion.stage` | run | One per stage: input/output counts, status, the stage's numeric metrics as `kg_rag.stage.<metric>`, `error.type` and the exception when it crashed |
| `kg_rag.batch` | stage, or root for `run-eval` answer generation | Item and failure counts; in `run-eval` the run's `kg_rag.eval.strategy` and `kg_rag.eval.repeat` (1-based), as on every span of the run |
| `kg_rag.mcp.tool` | root | One per MCP tool call (`run-mcp-server`): `mcp.tool.name`, `kg_rag.records`; with `capture_content` the redacted arguments as `input.value`. A `kg_search` or `kg_answer` call's `kg_rag.query` span nests under it |
| `kg_rag.console.ask` | root | One per console question: `kg_rag.search_strategy`, `kg_rag.query_id` (the id its `kg_rag.query` span's log lines and result carry); the query span nests under it |
| `kg_rag.query` | root, the `run-eval` batch, an MCP tool call, or a console question | `session.id`, `user.id`, `kg_rag.blocked`, `kg_rag.degraded` (answered without a step), `kg_rag.error_class` (of an error answer under `processing.ignore_errors`); with `capture_content` the redacted query as `input.value` and (outside `stream` / `astream`) the answer as `output.value`. Its id and trace id are the result's `metadata.query_id` and `metadata.trace_id`, and its log lines carry `query_id` and `namespace` |
| `kg_rag.strategy` | query | `kg_rag.rerank_status` for the whole query |
| `kg_rag.retrieve` | strategy | Hits per result type, up to 50 source ids |
| `kg_rag.fusion` | strategy | In/out counts, the step's `kg_rag.rerank_status` |
| `kg_rag.rerank` | fusion | `kg_rag.rerank.status` (`ok`, `empty`, `error`), document counts |
| `kg_rag.context_build` | query (`drift`: the round's actions span) | Tokens used vs the token budget |
| `kg_rag.global.select`, `kg_rag.global.map` | strategy (`global`) | Select, once per query: `kg_rag.depth` (configured), `kg_rag.cut` (the cut mapped after falling back to a shallower one), `kg_rag.reports`, `kg_rag.dynamic`. Map: `kg_rag.reports`, `kg_rag.map_calls`, `kg_rag.report_tokens` |
| `kg_rag.drift.primer`, `kg_rag.drift.actions` | strategy (`drift`) | Primer: `kg_rag.reports`, `kg_rag.answers`. One actions span per round (`kg_rag.depth`, `kg_rag.actions`), with the round's local searches under it |
| `kg_rag.guardrail` | query | One per screened query or answer: `aws.bedrock.guardrail.id`, `.source` (`input`, `output`), `.action` (`none`, `masked`, `blocked`), `.intervened`, `.policies`; with the contextual grounding check, `aws.bedrock.guardrail.grounding.checked`, `.grounding.failed`, `.grounding.score`, `aws.bedrock.guardrail.relevance.score` |
| `kg_rag.evaluate` | root | One per evaluator: `kg_rag.evaluator`, `kg_rag.items`, `kg_rag.eval.strategy`, `kg_rag.eval.repeat`; the judge's model spans under it |
| `chat <model>`, `embeddings <model>` | the calling span | OpenTelemetry GenAI conventions, kind `CLIENT` (as Bedrock rerank spans): `gen_ai.request.model`, `.max_tokens`, `.temperature`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.usage.cache_read.input_tokens` / `cache_creation.input_tokens` when Bedrock reports prompt-cache use, finish reasons, `error.type`. No cost |

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

- A streamed answer (`stream` / `astream`) is its own trace; its model span
  takes `session.id` / `user.id` from the answer step's run metadata, so it
  still groups with its session.
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
as several, as Amazon Bedrock bills them; not model calls), and the same counts
`by_role` (the model's purpose: `query`, `ingestion`, `evaluation`;
`embedding` and `rerank` for the embedding model's and reranker's requests) and
`by_model` (the model id the factory resolved). `run-ingestion
--output-format json` reports `token_usage` (the stages this run executed,
with `stages.<stage>`), `--metrics-sink cloudwatch` emits it per stage
([runbook](../operations.md#alarms)), and `run-eval` records it per answer
(`EvaluationResult.usage`) and per run (`summary.usage`). A response the
ingestion response cache served, a chunk-boundary or community report
answer the extraction store served, a description summary a full build took
from its stored copy, or a query entity or keyword extraction the chain's
extraction cache served (`search.query_extraction_cache_size`), counts as
a cache hit, not a call; a failed
call is not in `llm_calls` but in `failed_calls`; retries are calls.
`input_tokens` includes the prompt-cache reads and writes, which are also
counted apart (`cache_read_input_tokens`, `cache_write_input_tokens`, from
Bedrock's `cacheReadInputTokens` / `cacheWriteInputTokens`) since they are
billed at other rates. `request_retries` counts the requests botocore resent
inside one call (its `RetryAttempts`): the call counts once in `llm_calls`,
but a request resent after a read timeout or a server error is a Bedrock
invocation of its own (a resent throttled one is not). `sent_requests`
counts every request sent to Bedrock and `throttled_requests` those Bedrock
throttled, each attempt once (a botocore hook of every Bedrock runtime
client, the language, embedding and rerank models'; a throttling error
inside a response stream is not counted). So Bedrock's
CloudWatch `Invocations` can exceed `llm_calls` by up to `request_retries +
failed_calls`, plus any calls made outside a usage scope (the `run-eval`
warm-up queries), plus every embedding call: the usage counts language
model calls only (an embedding span has its text and character counts, no
tokens or price). A call that has not answered when the
usage is read (an `auto` query's unread extraction, still running in the
background) or was cancelled in flight (by `search.query_timeout_seconds`,
say) was sent and is billed, so it counts in `llm_calls` and in
`estimated_calls`, at an estimate of its input tokens and no output tokens;
one that answers later is settled in `run-eval`'s run usage. A call that
answers without reporting usage (a local server that returns none, say)
stays estimated the same way, rather than reading as free, with its output
tokens estimated from its text and its resent requests counted.
`stream` / `astream` yield text only and report no usage.
