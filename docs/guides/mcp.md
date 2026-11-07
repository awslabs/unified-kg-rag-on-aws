# MCP server

> 🇰🇷 한국어: [mcp.ko.md](./mcp.ko.md)

`run-mcp-server` serves one ingested index to agents as read-only
[Model Context Protocol](https://modelcontextprotocol.io) (MCP) tools. An
agent that speaks MCP (an IDE assistant, a Strands, LangGraph or other
framework agent, an agent hosted on Amazon Bedrock AgentCore) can then
search the index, ask it questions and walk its graph without this
package's Python API. The server writes nothing: there is no ingestion,
deletion or memory tool. Its design (layering, security reasoning, failure
modes) is in [the design doc](../design.md#84-agent-access-the-read-only-mcp-server);
every flag is in the [CLI reference](../reference/cli.md#run-mcp-server).

## Install and start

The server needs the `mcp` extra (the official MCP Python SDK):

```bash
uv sync --extra mcp                      # from a clone
pip install "unified-kg-rag-on-aws[mcp]" # from a package index
```

Without it, `run-mcp-server` exits 2 naming the extra. The server reads the
same configuration as `run-rag`: the stores, the models and the index
namespace (`--suffix`, else `processing.document_parsing.index_value`, as in
[Querying](./query.md)). Start it from the agent's MCP configuration, which
launches it as a child process and talks over its stdin and stdout (the
`stdio` transport, the default). Most MCP clients take a block like this
one:

```json
{
  "mcpServers": {
    "unified-kg-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/unified-kg-rag-on-aws",
               "run-mcp-server", "--config-path", "/path/to/config.yaml"]
    }
  }
}
```

The server logs to stderr; its stdout carries only protocol messages. It
exits 0 when the client closes the connection, 143 on SIGTERM and 130 on
Ctrl-C.

## Tools

| Tool | What it does | Model calls |
|---|---|---|
| `kg_search` | Retrieval only: the ranked supporting content for a query (any language), with `strategy`, `search_type`, `top_k` and `filters` as in `run-rag --mode search` | The strategy's query processing, the query embedding, the rerank |
| `kg_answer` | The full RAG answer, the sources the answer model saw, `usage` and the abstention fields (`abstained`, `abstention_reason`) | The above and the answer |
| `kg_get_entity` | Entities by `id`, `iri` or `name` (exact first; without an exact name, the closest names, `match: search`) | None |
| `kg_neighbors` | The entities within `max_hops` of one entity and the relationships among them | None |
| `kg_describe_schema` | The methodologies and strategies the namespace holds, its last build, the extraction schema's types and patterns, the filter keys and the server's limits | None |

Every tool is annotated read-only and idempotent. Call
`kg_describe_schema` first: it says which strategies the namespace can
run (a strategy it does not hold is refused, as in `run-rag`) and which
filter keys a search may give.

## Supporting content

`kg_search` and `kg_answer` return `records`, and the graph tools return
their entities and relationships in the same shape, so one mapping turns
any of them into the chunk-oriented "supporting content" records that agent
and ontology tooling exchange:

| Field | Holds |
|---|---|
| `rank`, `kind`, `score` | Position (1 is best), the item's kind (`text`, `entity`, `relationship`, `community`, `claim`, `summary`) and its relevance score |
| `chunk_id`, `text`, `text_truncated` | The chunk (text unit) id of a text record; the text, cut to `mcp.max_text_chars` |
| `source_id`, `name`, `type` | The stored item's id; an entity's or report's name, a relationship as `A -> B`; the entity or relationship type |
| `document_id`, `document_name`, `documents` | The source document: its id where stored, its file name (never its path); for a graph item, the file names of the chunks it was extracted from |
| `text_unit_ids` | The chunks a graph item was extracted from (the first 20) |
| `entities` | References (`id`, `name`, `type`, `iri`): the entity itself, a relationship's or claim's endpoints, or the entities of the same response that cite this chunk |
| `provenance`, `synthesized` | Seed-graph provenance (`kind`, `seed_graphs`, `iri`); `true` for model-written text such as global search's key points |
| `attributes` | Kind-specific scalars: `rank`, `weight`, `in_schema`, `type_iri`, `predicate_iri`, `hops` |

A response past `mcp.max_response_chars` drops its lowest-ranked records
and says so (`truncated`, `omitted`).

## Scope and limits

The namespace is fixed when the server starts, on the server's side, as
[a tenant's suffix must be](../concepts/scopes-and-namespaces.md#not-an-access-control-boundary).
A tool call can read another one only if `mcp.allowed_namespaces` lists it
(the tools then take a `namespace` argument; without the list they have
none), so list only namespaces every caller may read. Filters only narrow
a search and are not an isolation boundary: an entity's description merges
every document that mentions it. Serve a tenant whose data others must not
read from its own server and namespace, with the store roles of the
[operator runbook](../operations.md#query-access-and-tenant-isolation).

| Key | Default | Bounds |
|---|---|---|
| `mcp.tools` | all five | The tools listed; leave `kg_answer` out to make no answer-model call |
| `mcp.allowed_namespaces` | `[]` | Further namespaces a call may name |
| `mcp.strategies` | every strategy | The strategies `kg_search` and `kg_answer` may run; with `auto` in the list, `auto` routes only among the listed ones, and is refused before any model call when the namespace serves none of them |
| `mcp.max_top_k` | `20` | `top_k` of a search, `limit` of `kg_get_entity` |
| `mcp.max_hops`, `mcp.max_neighbors` | `2`, `50` | A `kg_neighbors` walk |
| `mcp.max_query_chars` | `2000` | A query or entity name |
| `mcp.max_text_chars`, `mcp.max_response_chars` | `2000`, `60000` | A record's text, a response's records |
| `mcp.max_concurrent_calls` | `4` | Calls run at once; others wait within the deadline |

Every call runs under `search.query_timeout_seconds` (180 s by default) and
fails with a timeout message past it. The tools keep no conversation
memory. All keys: [Configuration reference](../reference/configuration.md#mcp).

## HTTP and remote deployment

`--transport streamable-http` serves a stateless endpoint at
`http://127.0.0.1:8000/mcp`. On a loopback address the server checks the
`Host` and `Origin` headers, so a web page cannot reach it through DNS
rebinding. **The server authenticates no caller.** Binding another address
needs `--allow-remote`:

```bash
run-mcp-server --config-path config.yaml --transport streamable-http \
  --host 0.0.0.0 --port 8000 --allow-remote
```

Run it so only authenticated requests reach it: behind Amazon Bedrock
AgentCore Gateway or AgentCore Runtime with inbound authorization, behind a
reverse proxy that authenticates every request, or on a private network
whose every client may read the namespace. The query task's IAM role (read
access to the stores and the models the strategies call) is the policy it
needs. Hosting on AgentCore or the CDK stack's query task is not yet
verified on AWS.

A host application can mount the server itself:

```python
from unified_kg_rag import GraphRAGChain, build_mcp_server, get_config

chain = GraphRAGChain(get_config("config.yaml"))
server = build_mcp_server(chain, namespace="default")
app = server.streamable_http_app(stateless_http=True, json_response=True)
# Serve `app` with any ASGI server behind authentication; close the chain
# (await chain.aclose()) when the application stops.
```

## Cost, telemetry and untrusted text

- **Cost.** `kg_search` costs what `run-rag --mode search` costs,
  `kg_answer` what `run-rag` costs, and the graph tools no model call. An
  agent may call tools many times per task: restrict `mcp.tools` and
  `mcp.strategies` to bound it. `kg_search` and `kg_answer` return `usage`
  (model calls and tokens, by role and model).
- **Telemetry.** With [observability](./observability.md) on, each call is
  a `kg_rag.mcp.tool` span; its arguments are recorded only with
  `observability.capture_content`, after `redact_patterns`. Logs carry a
  digest of the query, not its text. A store or model failure reaches the
  caller as a generic message; the server log has the details.
- **Untrusted text.** Chunk text, descriptions and answers come from the
  corpus or a model. The tool descriptions tell the agent to treat them as
  data, and the tools return them in structured fields. Text planted in
  the corpus can at most lead an agent to more read-only calls on the
  same namespace. `kg_search` and `kg_answer` pass through the configured
  [guardrail](./guardrail.md); the graph tools make no model call and
  return stored text as it is.

## Troubleshooting

| Message | Fix |
|---|---|
| `The MCP server needs the mcp extra` | `uv sync --extra mcp` |
| `Missing endpoint configuration for the MCP server` | Set the store endpoints, as for `run-rag` ([Configuration](./configure.md)) |
| `is not a loopback address ... needs --allow-remote` | Bind `127.0.0.1`, or add `--allow-remote` behind an authenticating proxy |
| `Strategy '...' is not offered` / `does not hold` | Call `kg_describe_schema` and pick a listed strategy |
| `holds no knowledge graph` | The namespace was built with `raptor` only; the graph tools need graphrag, lightrag or hipporag |
| `The call failed inside the server` | Read the server's stderr log |
