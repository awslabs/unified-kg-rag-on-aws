# Design: agent access, visualization, console and prompts

Part of the [design reference](../design.md), which lists every section. Code cites
the section numbers: they never change.

### 8.4 Agent access: the read-only MCP server

`run-mcp-server` (the `mcp` extra) serves one index namespace to agents as
[Model Context Protocol](https://modelcontextprotocol.io) tools, so an agent
(an IDE assistant, a Strands or LangGraph agent, an agent on Amazon Bedrock
AgentCore) can search, ask and walk the graph without a client library. It
is opt-in: nothing changes for a deployment that does not start it. The
user guide is [MCP server](../guides/mcp.md).

**Tools.** Five, all read-only (annotated `readOnlyHint`, `idempotentHint`,
not `destructiveHint`, not `openWorldHint`). No tool ingests, deletes or
writes memory; no tool takes an index name, an endpoint or a file path.

| Tool | Runs | Model calls | Returns |
|---|---|---|---|
| `kg_search` | `GraphRAGChain.asearch` | those of the strategy's query processing (the `auto` router, query entities or keywords), the query embedding, the rerank | ranked supporting-content records |
| `kg_answer` | `GraphRAGChain.ainvoke` (with the caller's `history`, if given) | the above plus the follow-up rewrite (with a `history`) and the answer | the answer first, `abstained`/`abstention_reason`, what it was given without (`degraded`, `degraded_error`, `citations_unresolved`), then the records it cites (`records`: `cited`, the first `mcp.uncited_records` of an answer that cites none; `all`; `none`), `usage`, `blocked`, the strategy run |
| `kg_get_entity` | the entities index by `id`, `iri` or exact `name.keyword`; without an exact match, a lexical search of `name` | none | up to `limit` entities, how they matched |
| `kg_neighbors` | the graph walk from one entity (`max_hops`, the configured per-hop budget) and its incident relationships | none | the neighbours (hop distance, path) and the relationships with their text units and documents |
| `kg_describe_schema` | the namespace's build record and held artifacts, the resolved extraction schema | none (a schema source may read its file or S3 object once) | held methodologies, the strategies they serve, the build, entity and relationship types and patterns, the filter keys |

**Supporting-content record.** Every search hit, answer source, entity and
relationship comes back in one chunk-oriented shape, mapped from the one
view of an answer the console and `run-rag` read too
(`application/presentation/answer_view.py` over
`domain/retrieval/source_fields.py`: id lists, documents, file name, title,
entities, seed provenance, the `[n]` marker), so a caller can map it
onto the "supporting content" records other agent and ontology tooling
exchanges without per-kind code: `kind` (the section type: `text`,
`entity`, `relationship`, `community`, `claim`, `summary`), `chunk_id`,
`text` (cut to `mcp.max_text_chars`, `text_truncated` set), `score`,
`source_id`, `document_id` and `document_name` (a text unit's file name,
never its path), `documents` (the file names a graph item's text units
come from, resolved in one fetch by id), `text_unit_ids`, `entities` (each
`id`, `name`, `type`, `iri`: the item itself for an entity, the endpoints
for a relationship, and for a chunk the entities of the same response that
cite it), `provenance` (a seed graph's `kind`, `seed_graphs` and `iri`, as
in `RAGOutput.sources`) and `synthesized` (model-written, such as global
search's key points). The chain never reports a model-written section as a
source, so `kg_answer` adds one record for each citation no source is (a
`key_point` or `intermediate_answer`, before the sources), with its
screened text and the documents its citation resolved under the caller's
filters. A `query_result` record (the structured strategy's table, §6.6)
keeps its table up to 6000 characters, as the console shows it, names its
plan description and lists the entities it returned, by id and name. A text record's
`document_id` is read where the text unit stores it
(`attributes.document_id`), as `RAGOutput.sources` reads it. A
`kg_answer` record also carries its `citation` (the `[n]` marker of §7.1)
and `cited` (whether the answer uses it). `kg_answer` takes the caller's
conversation as an optional `history` (at most
`memory.max_messages_per_conversation` turns of up to
`mcp.max_response_chars` characters), passed as `RAGInput.history` (§7.2):
the server stays stateless.

**Layer placement.** The server is a driving adapter in the entry-point
layer, next to the other CLIs (§2.1): `application/cli/mcp_tools.py` holds
the tool logic and result models and imports no MCP SDK (its tests,
`tests/unit/test_mcp_tools.py`, run on a core install too), `application/cli/mcp_server.py` binds it to the SDK's
`MCPServer` (imported lazily; a missing SDK is a `ConfigurationError`
naming the extra), and `application/cli/run_mcp_server.py` is the console
script. The tools call only the public chain (`asearch` and `ainvoke`,
which check the strategy and the filter keys as every query does, and
`typed_filter_values` / `check_filter_values` first) and
`application/retrieval/graph_lookup.py`, which reads the stores through the
chain's own retrievers (as `methodology_check` and `filter_validation` do)
with text-less fetches by id and the graph walk: the same read paths the
strategies use, so every backend that serves queries (Neptune, OpenSearch,
the embedded SQLite stores) serves the tools. `tests/unit/test_layering.py`
is unchanged and its allowlist stays empty. `build_mcp_server` and
`build_mcp_app` (the stateless HTTP application with its `Host`/`Origin`
checks set) are exported from the package root for a host that mounts the
server in its own ASGI application. Neither changes the chain it is given:
with `mcp.strategies` set, the chain must be built with
`allowed_strategies` within it, and another is refused.

**Transport.** stdio by default: the agent starts the server as a child
process and owns its lifetime, there is no listening socket, and stdout
carries only protocol messages (the banner and the logs go to stderr).
Streamable HTTP (`--transport streamable-http`) is stateless (no session,
every call stands alone, so only POST is served: a GET, which would hold
an event stream open, and a DELETE get 405) and binds `127.0.0.1:8000` by
default. On any loopback address (all of `127.0.0.0/8`, `::1`,
`localhost`) the server passes the SDK explicit `Host` and `Origin`
allowlists naming that address and the loopback names, so a web page cannot
reach it through DNS rebinding: a foreign `Host` gets 421 and a foreign
`Origin` 403. (The SDK turns its own check on only for exactly `127.0.0.1`,
`localhost` and `::1`, and for an unset host, which also refused every
request a proxy forwarded with its own `Host`.) Binding any other address
needs `--allow-remote`, and there the headers are not checked: the proxy in
front owns them (`build_mcp_app(allowed_hosts=...)` checks them for a host
application that wants it). uvicorn's logs, access lines included, go to
the logging setup on stderr. The server has **no authentication**: it does not
check who calls it, and the documentation claims nothing it cannot enforce.
A remote deployment puts it behind something that authenticates every
request and only forwards authenticated ones (AgentCore Runtime or Gateway
inbound authorization, an authenticating reverse proxy, a private network
whose every client may read the namespace) and keeps the port unreachable
otherwise.

**Scope.** The tenant scope is fixed when the server starts: the namespace
is `--suffix`, else `processing.document_parsing.index_value` (as for
`run-rag`), and `indexing.additional_suffix` comes from the config. A tool
call can name another namespace only if `mcp.allowed_namespaces` lists it
(the tools then take a `namespace` argument; without the list they have
none). Caller `filters` are typed and checked as a query's are
(`typed_filter_values`, then the chain's key and value checks against the
indexes the strategy reads), and only narrow a search. They are not an
isolation boundary: an entity or relationship merges the text of every
document that states it, so a document filter does not hide what another
document said about the same entity. Data a caller must not read belongs in
another namespace. `mcp.strategies` limits which strategies the tools run
(cost control); a strategy the namespace does not hold is refused with the
chain's `MethodologyNotIndexedError` message, which names what it holds.
The list binds the chain (`GraphRAGChain(allowed_strategies=...)`) and is
enforced where the chain resolves the strategy (`_resolve_strategy`), not
after the answer: a strategy outside it is refused (`InvalidArgumentError`),
and `auto`'s candidates (the routing fallback of
`strategy_registry.routable_strategies` included) are cut to the list
before the router is asked, so neither the router nor the fallback after
a failed routing can pick another. When the namespace serves none of the
listed strategies, `auto` is refused before any model call
(`MethodologyNotIndexedError`, naming the allowed list and what `auto`
could have routed to).

**Limits.** Every argument is bounded in the published input schema and
checked again in the tool: `top_k` and `limit` by `mcp.max_top_k`,
`max_hops` by `mcp.max_hops`, neighbours by `mcp.max_neighbors`, the query
by `mcp.max_query_chars`. Each record's text is cut to `mcp.max_text_chars`
and a response past `mcp.max_response_chars` drops its lowest-ranked
records (`truncated`, `omitted`); the MCP message carries the result
twice (structured content and its JSON text), so it is up to about 2-4
times that on the wire. Each call runs under one deadline
(`search.query_timeout_seconds`) that covers the wait for one of the
`mcp.max_concurrent_calls` slots and the work, the chain's own deadline
inside it included. Work that does not watch the deadline (the graph tools'
store calls) ends at a backstop `_BACKSTOP_GRACE_SECONDS` (1 s) past it. A
`kg_answer` that times out in the slot queue or at the backstop has no
query metric of the chain's own, so the tool records one
(`GraphRAGChain.record_unrun_query`); the call's `query_id`, bound when it
starts, is the one the chain's query uses and the one its error names. The tools are stateless: no conversation memory (`use_memory` is
off), so no caller reaches another caller's history.

**Usage, telemetry and redaction.** `kg_search` and `kg_answer` return the
chain's `metadata.usage` (model calls, input and output tokens, by role and
model); the graph tools make no model call and return zero usage. Each call
opens a `kg_rag.mcp.tool` span (tool name, outcome, record count) under
which the chain's `kg_rag.query` span nests. Its arguments are recorded as
`input.value` only with `observability.capture_content`, after
`observability.redact_patterns`, the policy every other span follows;
INFO logs carry a digest of the query (`text_digest`), never its text. A
caller-correctable error (an invalid argument or filter, a strategy the
namespace does not hold, the deadline) is returned as a tool error with its
message, also under `processing.ignore_errors`, whose error answer names
the error's type and class (`metadata.error_type`, `metadata.error_class`);
any other failure returns a generic message and is logged on the server
(`log_error`: an outage with its text; an unexpected failure by its type,
code and HTTP status, a digest and where it happened at ERROR, its text and
traceback at DEBUG), so a store or model endpoint or a stack trace never
reaches the caller. Every
surface reads one classification of a failure (`shared/exceptions.py`
`classify` → `ErrorClass`) and maps the class itself: the CLIs to an exit
code (`cli/conventions.py` `EXIT_CODES`), the MCP server to the
caller-correctable classes above, the console to the error kind its Ask
page names. Which error text a surface shows is one rule,
`shown_error_text(error, audience=...)`: an MCP `caller` sees the text of
the caller-correctable classes and the deadline; the console's `operator`,
who configured the stores and models, sees every class's text but an
unexpected `failure`'s (a provider's or a library's, which may quote a
prompt or an ARN), unless the message is the package's own
(`has_own_text`: a stage failure named by its dominant cause, a question
the run asks). The console applies it to its API errors, the Ask error
event, an absorbed error answer's `error_message` in the result event and
the history, and a strategy's "not available" reason; the CLIs to their
output and JSON status' `error` (`shown_text`, so the console's job pages
show what the run printed), ingestion to each stage's recorded
`error_message` (in the run's metadata and its S3 checkpoint) and
`run-eval` to a failed answer's `error_message`. Each Ask event carries the
`query_id` (and `trace_id`) that find its log lines; every surface logs an
error but a refusal and the deadline, by `log_error`.

**Untrusted text.** Chunk text, descriptions and answers are corpus or model
text, and the server instructions and every tool description say to treat
them as data, not instructions. The tools return them in structured fields,
never as a prompt. The server offers no write path, so text planted in the
corpus cannot make it change anything; it may still steer an agent that
holds other tools (a shell, a mailer), which is the agent's to guard.
`kg_search` and `kg_answer` pass through the configured guardrail as any
query does (the query screened, the answer screened); the graph tools make
no model call and return stored text as it is, so the guardrail does not
screen them.

**Multilingual, hybrid, provenance, methodologies.** Tool descriptions are
English; arguments may be in any supported language. `kg_search` and
`kg_answer` take the chain's query-language path (the query translated when
the index was, the answer in the question's language) and its hybrid
retrieval (`search_type` hybrid by default: BM25 and kNN fused, then
reranked). `kg_get_entity` matches names exactly on `name.keyword`, then
lexically on `name` with the index's analyzer (`nori`, `cjk` and the
language analyzers), a hit that is the name as the strategies link one
(`linking.exact_links`) reported as a `name` match, deliberately without
an embedding: a name lookup
needs no vector, and costs no model call. Every record keeps the
provenance the chain reports. `kg_describe_schema` lists the methodologies
the namespace holds, the strategies they serve and those `auto` routes
among, the fields `run-describe` reports (one `NamespaceSummary` over one
`NamespaceHoldings` rule, which the chain's refusal and the console read
too); `kg_neighbors` reads the
relationships with their descriptions and text units from the relationships
index when the namespace holds the relationship vectors (lightrag,
hipporag), else the edges from the graph (ids and weights only); a
namespace without a graph (raptor only) answers both graph tools with an
error naming what it holds.

**Failure modes.**

| Situation | Outcome |
|---|---|
| The `mcp` extra is missing | `run-mcp-server` exits 2 naming the extra (`build_mcp_server` raises `ConfigurationError`) |
| A store endpoint is unset, an extra a provider needs is missing | exit 2 before the server starts (the CLI preflight) |
| A non-loopback `--host` without `--allow-remote` | exit 2 |
| An argument out of bounds, an unknown namespace or strategy, a bad filter | tool error naming the argument |
| A strategy the namespace does not hold | tool error with the chain's message (what it holds, how to add it) |
| The deadline passes | tool error `QueryTimeoutError`; the chain cancels its calls |
| A store or model failure | tool error with a generic message; logged by `log_error` (an outage with its text, an unexpected failure as its type, a digest and where it happened, its text and traceback at DEBUG) |
| The client closes stdin (stdio) | the server exits 0; SIGTERM exits 143, Ctrl-C 130 |

**Cost.** `kg_search` costs what `run-rag --mode search` costs for the same
strategy (with `auto`, the router call; the query embedding; one rerank
request), `kg_answer` what `run-rag` costs (plus
the answer call, usually the largest), and the graph tools one to four
store requests and no model call. An agent may call a tool many times per
task, so `mcp.tools` can drop `kg_answer` (the agent then writes the answer
from `kg_search`'s records with its own model) and `mcp.strategies` can
leave out the costlier strategies (`drift`, `global` with dynamic
selection, `iterative`).

**Deployment (follow-up).** This round adds no infrastructure. The query
image can run `run-mcp-server --transport streamable-http --host 0.0.0.0
--allow-remote` as an AgentCore Runtime MCP server (the container listens
on port 8000 at `/mcp`, stateless) with Runtime inbound authorization, or
behind AgentCore Gateway; the IAM policy is the query task role's (read
access to the stores, the models the strategies call). Both need a
real-AWS verification before they are documented as supported.

## 10. Visualization & Analytics

Ingestion renders nothing: at the end of a full build it writes the index
namespace's graph and community hierarchy, read from the stores, as JSON
(`graph.visualization.export_data`,
`application/storage/visualization_export.py` over
`domain/ingestion/visualization_data.py`); an incremental run writes none,
and `run-visualization` (and the console's Graph page when opened) builds it
with the same `write_visualization_data` when no export names the build
the stores hold (`current_visualization_data`), so the file is the whole
namespace. `run-visualization` renders that
file without ingestion through the renderers of `adapters/renderers/`
(`BaseRenderer`, `@register_renderer`, `RenderContext`, `resolve_renderers`,
`render_resolved`):
`InteractiveRenderer` (pyvis network and community hierarchy) and
`StaticRenderer` (degree, centrality and community-size charts as HTML with
inline SVG, written without a plotting package and with every name
HTML-escaped), computing a seeded spring layout and the configured
centrality first. The interactive renderer is the `viz` extra: the adapters
import the renderers on use, so the registry, `BaseRenderer` and `static`
work on a core install, and a missing extra fails the run
(`register_renderer(..., requires=...)`, `check_renderer_dependencies`)
before anything renders.
Using it: [Visualization](../guides/visualize.md).

### 10.1 The console (`run-console`, opt-in)

A local web app over the same entry points the CLIs use: ask questions,
ingest a corpus, run and compare evaluations, explore the exported graph,
see what queries cost, and edit the configuration. It is a prototype for
one person on one host, not a multi-user service. Using it:
[Console](../guides/console.md).

**Placement.** The server is an entry point, in
`application/cli/console/` (the CLIs' layer, so `test_layering.py` holds it
with an empty allowlist), started by `run-console`
(`application/cli/run_console.py`). It imports the application layer
(`GraphRAGChain`, `get_config`, the ingestion pipeline's stage table),
evaluation's comparison helpers (`answer_cost_usd`), domain models and
the shared kernel, and of the adapters only the two public registries the
facade exports from (`strategy_registry`, `ParserFactory`; pinned by
`test_run_console.py`). Its
packages, FastAPI, Uvicorn and python-multipart, are the `console` extra;
without them `run-console` exits 2 naming the extra. The web client is a
Vite + React + TypeScript app in `console/web/` (source and npm lockfile
committed, the build output not); `run-console` serves `console/web/dist`
when it is built and otherwise says how to build it.

**Data flow.**

| Page | Server path | What runs |
|---|---|---|
| Ask | `POST /api/ask` (server-sent events) | One `GraphRAGChain` per loaded config, in the server's event loop. `ainvoke(on_answer_chunk=...)` hands the answer model's text over as it streams; the final `RAGOutput` follows, with its sources, `metadata.routing` (how `auto` chose), rerank status, usage and the query's trace id |
| Ingest | `POST /api/jobs/ingest` | `run-ingestion` in a child process, with the console's cache directory and a pipeline id per corpus |
| Evaluate | `POST /api/jobs/evaluate`, `GET /api/evaluations/*` | `run-eval --output-format json` in a child process, writing into its own directory under the evaluation outputs; a saved run is read back as written, its per-question scores (`evaluations/questions`) through the evaluation's own saved-results reader and `per_query_means` |
| Graph | `GET /api/graphs/data`, `GET /api/graphs/entity` | Reads a `visualization_data.json` the ingestion exported, once per version of the file (path, modification time, size); degrees, communities, source kinds and the layout of the 600 best-connected entities (`spring_layout`) are computed on read. The page gets those entities and the edges between them in full and every other entity as `{id, name, degree}`; `graphs/entity` returns one entity with every edge that touches it |
| Dashboard | `GET /api/dashboard` | Reads the console's query log (SQLite under its cache directory) and the evaluation result files |
| Settings | `GET/POST /api/config/*` | Validates against `Config`, diffs against the file, writes the YAML file |

**Jobs in child processes, not threads.** The ingestion pipeline stops
through a signal in the thread that runs it (`request_stop` raises where
that thread is, and defers during a checkpoint write); a thread of the
server cannot be stopped that way. A child process running the unchanged
CLI gets the stop the CLIs are built for: Stop sends SIGTERM, the
pipeline checkpoints its finished stages and exits 143 (rerun resumes), and
`run-eval` keeps the runs it finished; a process that outlives the CLIs'
100 s stop grace is killed. The child also isolates the server from a
crash or a memory peak of a run, and the result is exactly what the CLI
would produce. Progress is read, not pushed: an ingestion's stages from
its `pipeline_metadata.json` (written after every stage) against the stage
plan the console computes from the config (`DataIngestionPipeline.STAGE_CLASSES`,
`CONFIG_ENABLED`, `selected_stages`; pinned to a built pipeline's stages by
a test), an evaluation's from the run summaries in its directory against
strategies x repeats. Logs go to a file per job; the page shows its tail
and each failed stage's error.

**Concurrency.** One ingestion or evaluation job per index namespace at a
time; a second is refused with the running job's id, also when both
requests arrive at once (the namespace is claimed before the first await).
A pipeline id is per corpus folder: a readable slug of its name and a hash
of its resolved path and namespace. A question the run asks before it
writes (a setting of a resumed id changed or its cache is incomplete, a reset, a mass deletion) is
answered only by its own flag (`run-ingestion --confirm-integrity`,
`--confirm-reset`, `--confirm-deletions`); unanswered, the job names it and
the page offers to start again with that flag. A job a console process left
running when it ended is marked lost at the next start. Across processes the
pipeline's namespace lease still decides (a run started by another
`run-ingestion` makes the console's run fail with `NamespaceLeaseError`'s
message). Queries run while a job does; after an ingestion finishes the
chain is rebuilt, so the next answer reads the new index.

**Streaming.** `GraphRAGChain.ainvoke` takes an optional
`on_answer_chunk` callback: the answer step then streams the answer model
and passes each text chunk on; the returned `RAGOutput` is unchanged and
stays authoritative (the client replaces the streamed text with its
`answer`). It streams nothing when the output guardrail screens answers
(nothing may be shown before screening) or when the answer is read for
the `[NO_ANSWER]` token (`search.abstention`), so a decline is never shown
half-written. `astream` is unchanged; it yields no sources. A streamed
answer is the unstreamed one: its text is the chunks joined, so
`citations` and `unknown_citation_markers` come out the same
(`test_answer_chunks.py`). Each `chunk` event carries the markers that
chunk closes, read by `CitationMarkerScanner` (a marker split across chunks
counts once it closes); the `result` event gives each source the marker
it was shown under and each citation the rank of its source, so the
page links an answer's `[n]` to that source in the trace.

**Follow-ups.** The page keeps the conversation and sends it with a
follow-up as `RAGInput.history` (each answered question and its final
answer; failed, stopped and blocked turns left out), not a conversation
id in the chain's memory: a settings save or an ingestion rebuilds the
chain, which would drop an in-process memory, and a config with memory
off still continues the conversation. The chain bounds and screens the
history as it does for any caller.

**Settings.** The page edits the YAML file the console was started with
(`--config-path`), field by field (each with its `Field(description=...)`
from the JSON schema of `Config`) or as text. A change is validated by
loading it as `Config` would be (unknown keys included), shown as a diff
of the file, and written only on Save, refused when the file changed on
disk since it was read. Secrets are never sent to the browser or written:
`SecretStr` values (`providers.*.api_key`, `aws.opensearch.password`) are
masked in every response, and a file or an edit that sets one is refused
with the environment variable to use instead
(`UKR__PROVIDERS__LLM__API_KEY`). The observability panel shows which
`LANGFUSE_*` / `OTEL_*` variables are set, never their values.

**Security.** It binds 127.0.0.1 by default; another address needs
`--allow-remote`, and the guide says to put the console behind an
authenticating reverse proxy then: it has no accounts. Every request needs
a per-launch random access token (the printed address carries it once; the
server moves it into an HttpOnly, SameSite=Strict cookie; scripts send it
as `X-Console-Token`), so another local user or a page in another tab
cannot drive the console. A Settings change to where a provider sends its
requests (a `base_url`, a store endpoint or port) is saved only when
confirmed: those requests carry the credentials the environment holds. Every path the client names (a corpus, a
dataset, an upload, a graph file, a saved run) is resolved, symlinks
included, and must stay inside the directories the console was started
with: reads under `--source-directory` (default: the working directory)
and the configured outputs, writes only to the uploads directory, the
cache directory, the evaluation outputs and the config file. Uploads keep
their relative paths, sanitized segment by segment. Requests with a
foreign `Origin` or `Host` are refused (DNS rebinding), and no response
carries a CORS header. The client makes no request to another host: the
fonts (OFL) are bundled by the build, so it works offline and in a private
VPC.

**Cost.** The console adds no model call: a question costs what the same
`run-rag` query costs, and a job what its CLI run costs. The dashboard
prices recorded usage with `evaluation.prices` (`answer_cost_usd`); without
prices it shows tokens only, as `run-eval` does.

**Failure modes.** A missing extra exits 2 before binding. A config that
does not validate starts the console with the Settings page only, showing
the errors. A chain that cannot be built (a store unreachable) makes Ask
report the error per question; jobs still start (their own run reports it).
A server stopped by a signal sends SIGTERM to its running jobs and waits
for them up to the CLIs' stop grace (110 s). A streamed answer cut at the
output-token limit fails as the unstreamed one does
(`STRICT_STREAM_TRUNCATION` makes the truncation guard raise after the last
chunk).

**Identity.** The client is a knowledge-graph workbench where every claim
has a trail; its brief, tokens and references are in
[`console/web/DESIGN.md`](../../console/web/DESIGN.md): a developer tool, with
a dark, GitHub-like palette and a matching light one, a sticky top bar,
cards, a chat-style answer view and a force-directed canvas graph. One
signature element: the *provenance trace*, an indigo rail from an answer
through each source it read (ranked), the passage and file it came from, to
the entity or relationship in the graph, where the traced node wears a
dashed ring and its edge carries particles; matched words are in a
highlighter. Text is set in Pretendard (Latin and Hangul; Pretendard JP for
a Japanese page), ids and figures in JetBrains Mono, both OFL and
self-hosted (Pretendard as its vendor ships it, unmodified), per locale
through `:lang()`. A light and a dark theme are two full
token sets (the dark one its own, not inverted), following the system
setting unless the top bar's toggle picks one; a same-origin script sets
the theme before the first paint, as the CSP allows no inline script. The
graph canvas (react-force-graph-2d) loads with the Graph page only. UI copy is English
with nine more locales (German, Spanish, French, Italian, Japanese, Korean,
Portuguese, Russian and Chinese), English being
the source and the fallback.

---

## 11. Prompts and Prompt Tuning

- **Prompts** (`domain/prompts/`) are `BasePrompt` classes whose templates and
  variables are class attributes, version-controlled as code. All but five
  can be overridden from config (`custom_prompts`); the
  [Domain adaptation guide](../guides/domain-adaptation.md#prompts-without-an-override)
  lists the five.
- **Prompt tuning** (`application/prompts/tuner.py`, after MS GraphRAG
  `prompt_tune`) profiles a corpus sample (domain, language, persona, entity
  types) and writes a domain-adapted `custom_prompts` fragment to review and
  apply; nothing is applied at run time. A tuned prompt replaces only the
  persona preamble and keeps the built-in output rules (the XML schema, the
  verbatim `source_text` grounding), so the parser and grounding guard keep
  working. The rules carry the `{entity_types}` and `{relationship_types}`
  variables, rendered from config at call time
  (`GraphExtractionPrompt.config_variables`). A variable that stands for
  rule text the template would otherwise spell out declares it
  (`inlined_slots`): the extraction-record fingerprint hashes the template
  with it filled in, so moving rule text into a variable re-keys no record.
- **Schema drafts** (`run-prompt-tuning --schema-out`, `SchemaDraft` in
  `application/prompts/tuner.py`): the profile's entity types (the
  configured specs when the config declares them), the configured
  relationship types, then those the grounded examples state most often,
  counted by `schema_key` (at most `MAX_DRAFT_RELATIONSHIP_TYPES` more).
  Configured patterns are kept, and a type that has them gets no other;
  any other type gets patterns only when every (source type, target type)
  the examples give it between two draft entity types appears
  `MIN_PATTERN_SUPPORT` times, so `align` never counts the draft's own
  examples off pattern. A draft over the loaded schema therefore drops
  nothing it declares; the CLI still refuses to write over that file or,
  without `--force`, any existing one. The file is a
  `schema_source: {kind: file}` document validated as a schema before it is
  written, and names `align`, never `strict`: a draft from a few chunks
  would drop what they did not show. A pure function of the profile and the
  records, so a fixed model gives the same file.

---
