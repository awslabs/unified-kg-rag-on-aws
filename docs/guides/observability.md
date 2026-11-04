# Observability

Tracing is off by default. With `observability.enabled: true`, every ingestion
run and every query becomes an OpenTelemetry trace that any OTLP-capable
backend (Jaeger, Grafana Tempo, an OpenTelemetry Collector, Langfuse, Phoenix)
can show. It needs the `otel` extra (`uv sync --extra otel`); enabling tracing
without it fails at start-up with a `ConfigurationError` naming the extra, and
with tracing off nothing OpenTelemetry-related is imported. Every span and
attribute, the vendor attribute names, the known limits and the model-usage
counters are in the [Telemetry reference](../reference/telemetry.md).

## Settings

- `observability.backend`: `none` (the process's global tracer provider,
  which a host application or `opentelemetry-instrument` sets up), `otlp` (an
  OTLP exporter the package installs), or `langfuse` / `phoenix` (that
  exporter pointed at the vendor). The other keys (`capture_content`,
  `redact_patterns`, `sample_rate`, `service_name`, `export_timeout_seconds`)
  are in the [reference](../reference/configuration.md#observability).
- The exporter is configured by the standard OpenTelemetry environment
  variables, never by the config file, so a collector key never lands in it:
  `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`),
  `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TIMEOUT` and
  `OTEL_EXPORTER_OTLP_PROTOCOL` (`http/protobuf`, or `grpc` for port 4317).
- Every span of a query carries `session.id` (`RAGInput.conversation_id`)
  and, when set, `user.id` (`RAGInput.user_id`), which Langfuse and Phoenix
  group traces by; pass an opaque user id, not an e-mail address. Log records
  written inside a span carry its `trace_id` and `span_id`.
- Spans carry no query text, passages or model output unless
  `capture_content: true`.

## Jaeger

To send traces to a local Jaeger, which accepts OTLP on port 4318:

```bash
docker run --rm -d --name jaeger -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one:latest
uv sync --extra otel
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

```yaml
observability:
  enabled: true
  backend: "otlp"
```

Run a query and open <http://localhost:16686>: the service `unified-kg-rag`
shows one `kg_rag.query` trace per query and one `kg_rag.ingestion.run` trace
per ingestion run. For a collector or a vendor endpoint, point
`OTEL_EXPORTER_OTLP_ENDPOINT` at it, pass credentials in
`OTEL_EXPORTER_OTLP_HEADERS` from Secrets Manager or task secrets, and set
`OTEL_EXPORTER_OTLP_PROTOCOL=grpc` for a gRPC endpoint (port 4317).

To check it works: the start-up log has
`Observability enabled (backend: 'otlp', ...)`; log records written inside a
span carry its `trace_id` and `span_id`; a query trace has `kg_rag.retrieve`,
`kg_rag.fusion`, `kg_rag.context_build` and `chat <model>` children, whose
`gen_ai.usage.input_tokens` is the run's token usage.

## Langfuse

Langfuse and Phoenix ingest OTLP, so they need only the `otel` extra, no
vendor SDK; their settings come from the environment only, and a standard
`OTEL_EXPORTER_OTLP_*` variable wins over the vendor one. A local Langfuse
runs from its Docker Compose file (Langfuse v4; v3 works the same way):

```bash
git clone --depth 1 https://github.com/langfuse/langfuse.git
cd langfuse && docker compose up -d     # UI on http://localhost:3000
```

Create a project and an API key pair in the UI, then:

```bash
uv sync --extra otel
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...   # deployments: from Secrets Manager
export LANGFUSE_SECRET_KEY=sk-lf-...   # as task secrets, never in a file
```

```yaml
observability:
  enabled: true
  backend: "langfuse"
```

- `LANGFUSE_HOST` is required (there is no default host); for Langfuse Cloud
  use the region's URL, for example `https://cloud.langfuse.com`. The
  exporter posts to `<host>/api/public/otel/v1/traces` over HTTP only (gRPC
  fails at start-up); with `OTEL_EXPORTER_OTLP_ENDPOINT` instead, give
  `<host>/api/public/otel` (the host root returns 404).
- Both keys are required and sent as a Basic `Authorization` header with
  `x-langfuse-ingestion-version: 4`. If you set `OTEL_EXPORTER_OTLP_HEADERS`
  yourself, include that version header, or Langfuse Cloud can take minutes
  to show traces.
- Traces group in Sessions by `RAGInput.conversation_id` and in Users by
  `RAGInput.user_id`. Langfuse v4's default `events_only` mode disables
  `/api/public/traces`, `/observations` and `/sessions`: read spans through
  `/api/public/v2/observations` or the UI.

## Arize Phoenix

```bash
docker run --rm -d --name phoenix -p 6006:6006 -p 4317:4317 \
  arizephoenix/phoenix:latest            # UI on http://localhost:6006
uv sync --extra otel
```

```yaml
observability:
  enabled: true
  backend: "phoenix"
```

With nothing else set, spans go to `http://localhost:6006/v1/traces` (`:4317`
with `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`) and land in the project named by
`service_name` (`PHOENIX_PROJECT_NAME` wins). For a remote Phoenix set
`PHOENIX_COLLECTOR_ENDPOINT` to the server root and, when it needs one,
`PHOENIX_API_KEY` (sent as a Bearer `authorization` header). Spans carry
OpenInference kinds (`LLM`, `EMBEDDING`, `RETRIEVER`, `RERANKER`, `CHAIN`), and
the Sessions view groups a conversation's queries.

## CloudWatch through the ADOT collector sidecar

CloudWatch accepts OTLP traces signed with SigV4, so a collector next to the
app signs and forwards them. On the CDK stacks, `-c enable_cloudwatch_tracing=true`
adds an AWS Distro for OpenTelemetry (ADOT) collector sidecar and points the
app at it. The whole recipe (turning on Transaction Search, the image, the
first collector pull, where traces appear, cost, the collector configuration
without the CDK stacks, and Bedrock model invocation logging as an
account-level backstop) is in
[iac/README.md: CloudWatch tracing](../../iac/README.md#cloudwatch-tracing-optional).

## Export limits, private VPCs and privacy

- **Export never blocks a run.** Spans leave from a background thread; each
  request is bounded by `observability.export_timeout_seconds` (10 s;
  `OTEL_EXPORTER_OTLP_TIMEOUT` wins), a full queue drops spans, and failures
  are logged at `ERROR`. Against an unreachable collector the flush at exit
  takes about one timeout per remaining batch of 512 spans, at most about
  30 s, then drops the rest.
- **Private VPCs.** The endpoint must be reachable from the tasks: a
  collector sidecar on `localhost`, a Langfuse or Phoenix inside the VPC, or
  the X-Ray interface endpoint for the collector; hosted Langfuse or Phoenix
  needs internet egress, which `private` mode does not have. Build the image
  with `--build-arg UV_EXTRAS="--extra otel"`.
- **Credentials** (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`,
  `PHOENIX_API_KEY`, `OTEL_EXPORTER_OTLP_HEADERS`) come from the environment
  only: inject them as ECS task secrets. On the CDK stacks, store them as
  JSON keys of one Secrets Manager secret and deploy with
  `-c observability_secret_arn=<complete secret ARN>` and, for keys other
  than the two Langfuse ones,
  `-c observability_secret_keys=OTEL_EXPORTER_OTLP_HEADERS,...` (in
  `private` mode the stacks then add a Secrets Manager interface endpoint,
  which a reused VPC must provide, or the tasks fail to start; it reaches
  the deploy region only, so synth refuses a secret of another region); the
  non-secret settings (`LANGFUSE_HOST`, `UKR__OBSERVABILITY__*`) go in the
  image config or the run's container overrides. The start-up line
  `Observability exporter for backend '...'` shows the endpoint and header
  names, never values.

Spans carry counts, ids, model names and token usage only.
`capture_content: true` adds prompt and completion text to model spans, and
the query and answer to the `kg_rag.query` span (after `redact_patterns`):
enable it only where the trace backend may hold corpus data. With `otlp`,
`langfuse` or `phoenix`, `redact_patterns` also applies at export to every
string attribute, event attribute (an exception message can quote a query)
and status message, `session.id` and `user.id` included; with `none` the
host's pipeline exports spans as recorded, so redact in its collector.
`RAGInput.user_id` is exported as given: pass an opaque id, not an e-mail
address. With LangSmith tracing on (`LANGSMITH_TRACING=true`), LangChain
uploads content to LangSmith independently of these settings.
