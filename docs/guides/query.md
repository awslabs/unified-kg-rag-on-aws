# Querying

> 🇰🇷 한국어판: [query.ko.md](./query.ko.md)

`run-rag --query "..."` answers one question, `--interactive` holds a
conversation. Pick a strategy with `--search-strategy` (default `auto`, an LLM
router); every answer returns the sources the answer model actually saw.
Which strategy to pick, and how answers are shaped:
[Retrieval](../concepts/retrieval.md). Which indexes a query reads:
[Index suffix](../concepts/scopes-and-namespaces.md#index-suffix).

Every flag with its default: [CLI Reference](../reference/cli.md#run-rag).

## Examples

```bash
# Single query (auto strategy, hybrid search)
run-rag --query "What are the main themes in the documents?" --config-path config.yaml

# Pick a strategy and search type
run-rag --query "How does entity X relate to Y?" \
  --search-strategy local --search-type hybrid --config-path config.yaml

# Retrieval only, JSON output
run-rag --query "..." --mode search --output-format json --config-path config.yaml

# A graph query for counts, rankings and paths (opt-in)
run-rag --query "How many vendors supply buyers in Norland?" \
  --search-strategy structured --config-path config.yaml

# HippoRAG 2 PageRank (opt-in; auto never routes to it)
run-rag --query "Where was the founder of the company that acquired X born?" \
  --search-strategy ppr --config-path config.yaml

# Verbose (entities, every source with its score and text, metrics) with
# attribute filters
run-rag --query "..." --verbose --filters attr_category:research type:person \
  --config-path config.yaml
```

What the text output shows:

- **An answer** is followed by one line naming its sources: the files its
  text units came from and how many graph elements of each type the model
  read (`Sources (13): files: acme.txt, globex.txt; 7 relationships, 4
  entities, 1 community report`). `--verbose` lists each source instead,
  with its type, file or name, score and the start of its text, then the
  query processing and the metrics; in `--interactive` mode too.
- **`--mode search`** prints the hits: type, file (a text unit, with its
  chunk when the file has several) or name (an entity, a community report;
  a relationship's two ends), score and the start of the text, the first
  20 unless `--verbose`.
- **A partial answer** (one given without a step that failed, such as a
  throttled map call of global search) says so on a line after it:
  `Partial answer: given without steps that failed: global_map (1/3 reports
  unrated)`; the steps the query's limits skipped are a line of their own
  (`Skipped to stay within the query's limits: ...`), and an answer they
  left no context declines with `Declined: the query's limits ...`, naming
  the settings to raise (`search.query_max_model_calls`,
  `search.query_max_input_tokens` or their `UKR__` variables); a
  citation lineage read that failed adds `Some citations name no document`.
  `--verbose` also lists the model-written sections the answer may cite
  (global search's key points, DRIFT's intermediate answers).
- **`--output-format json`** prints the result as one document; an answer's
  `search_results` there leaves out `results`, which repeat its `sources`,
  unless `--verbose` (they also hold what the context budget cut). It also
  holds what the console and the MCP server report: `degraded`,
  `degraded_error`, `citations_unresolved`, `error_class` and, for an
  answer, `written` (the model-written sections).

## Filters

`--filters key:value ...` (or `RAGInput.filters`) narrows what retrieval ranks;
a pair without a key, a `:` or a value, a key given twice and a key no index the
strategy reads declares are usage errors (exit 2). A list (any of its values)
or a range goes in `--filters-json`, one JSON object as `RAGInput.filters`
takes it (`'{"start_date": {"gte": "2022-07-01"}, "category": ["a", "b"]}'`),
checked the same way; with `--filters`, each key in one of the two.

- A filter applies only to the stores that declare its field
  (`type:person` leaves text units unfiltered). A key no store the selected
  strategy reads declares (e.g. `category`, or `type` with `naive` or
  `global`) raises `InvalidFilterError`, listing the filterable keys. `auto`
  routes only among the strategies it may route to that apply every key.
- `attr_<key>` matches entry `<key>` of a document's `filters` metadata.
  OpenSearch applies it strictly (fail-closed): an item without the
  attribute is excluded. An entity, relationship, claim or community report
  from several documents matches when one of its documents does.
- Entity and claim `type` and claim `status` are stored lowercase, and a
  filter value of theirs matches its lowercase form too: `type:PERSON`
  matches `person`.
  Boolean and numeric fields take their type's values
  (`in_schema:false`, `rank:2`).

Every filterable field per store, value typing, dates, which strategy
applies which key and how graph expansion is filtered:
[Filters reference](../reference/filters.md). Filters are not an access
control: [Not an access-control boundary](../concepts/scopes-and-namespaces.md#not-an-access-control-boundary).

## Personalized PageRank (`ppr`)

`--search-strategy ppr` answers with HippoRAG 2's retrieval: the question's
relationship and passage hits seed a PageRank over the graph around them,
which ranks the passages (no query entity extraction, one LLM call). It
needs the relationship vector index and the graph, not community reports.
Its settings are `search.ppr_search`, and its fusion buckets
(`ppr_passages`, `ppr_seed_text_units`, `ppr_facts`) take
`search.fusion.fusion_weights`. It is the cheap, fast option: answers
within the noise of `local` and `mix` on multi-hop questions at a fraction
of their cost ([evidence](../concepts/defaults-and-evidence.md#the-ppr-strategy)).
`auto` never picks it until a measurement shows where it wins: select it
explicitly. `local` orders its text units the same way by default
(`search.local_search.text_unit_ranking: ppr`; `support` orders them by
how many expanded entities cite each); DRIFT's follow-up local searches keep `support`
(`search.drift_search.text_unit_ranking`), unmeasured with `ppr`. See [Retrieval](../concepts/retrieval.md) and
[Design §6.4](../design/retrieval-hipporag-raptor.md#64-hipporag-2-personalized-pagerank-ppr-local-ranking).

## Rerank status

Every query reports what reranking did in `metadata.rerank` of the
`SearchResult` and the RAG output (and `--output-format json`): `status`,
`reason`, and `stages`, one entry per rerank step with its `status`, `scope`
(`all`, or the reranked result types, e.g. `text` for `local`, `drift`, `mix`
and `hybrid`) and `reason`.

| Status | Meaning |
|---|---|
| `applied` | The rerank model reordered the candidates. |
| `disabled` | `providers.rerank.kind: none`. |
| `unavailable` | The rerank model could not be built (initialization failed, logged at ERROR, or the cross-encoder's `[cross-encoder]` extra is missing). |
| `skipped` | Nothing to rerank: no candidates, none of a rerank-eligible type, or no query text. |
| `failed` | The call raised (denied `bedrock:Rerank`, a region without the model, an unreachable endpoint) or returned nothing usable; the fused order was kept. |

- A `failed` or `unavailable` reason is the AWS error code or the exception
  type, without the message.
- Built-in strategies rerank once per query. A caller that fuses several
  times into one trace gets one stage per fusion; the top-level `status` is
  the worst, in the order `failed` > `unavailable` > `applied` > `skipped` >
  `disabled`. `metadata.reranked` marks each reordered result.
- A Bedrock permission or region problem shows as `failed` at the first
  query, not `unavailable`. `search.reranking.required: true` fails such
  queries with `RerankModelError` instead of answering from the fused order.

## Query deadline

A query (`run-rag`, `invoke`/`ainvoke`, `search`/`asearch`, a stream) may take
`search.query_timeout_seconds` (default 180) in all. Past it the query fails
with `QueryTimeoutError` (exported from `unified_kg_rag`; also a builtin
`TimeoutError`) and its model and store calls in flight are cancelled:

- `run-rag` prints the message and exits `1`; `run-eval` counts the query as
  failed (`error_message` set, not scored) and goes on with the others.
- With `processing.ignore_errors` the query returns the error answer instead,
  with the message in `search_results.metadata.error`.
- A streamed answer must also send its first chunk within
  `search.model_timeout_seconds` of its model call; once chunks flow, the
  overall deadline applies. Any chunk counts: a reasoning model's streamed
  reasoning (dropped from the answer) ends that wait. Until answer text
  arrives, a transient error (a Bedrock stream's throttling event or a
  dropped connection included) is retried as for an unstreamed query, on
  every provider.
- `null` turns both deadlines off.

The query's model calls (answer generation and the other query-time LLM
calls, query embeddings, Bedrock rerank) use their own clients:
`search.model_timeout_seconds` (default 90) for each request's answer and
`search.model_max_attempts` (default 2) botocore attempts, so a stalled
Bedrock call gives up after about 3 minutes. Ingestion and evaluation-judge
calls keep the 330 s read timeout and 6 attempts. On an OpenAI-compatible or
Ollama server the query calls' request timeout is
`providers.llm.timeout_seconds` / `providers.embedding.timeout_seconds`,
capped at `search.model_timeout_seconds`; `backend: local` presets 300 s
model and 900 s query timeouts, since a local server queues requests (a
`null` there is replaced by the preset: set a large number instead).

A non-streamed answer arrives in one piece, so `search.model_timeout_seconds`
also caps how long the model may generate it. Raise it (and the query
timeout) for long answers or high reasoning effort. The defaults come from
measured query times on real AWS
([evidence](../concepts/defaults-and-evidence.md#query-deadlines-and-model-timeouts)).
Tuning is in the
[Operator Runbook](../operations.md#query-deadlines-and-model-timeouts).

### Model-call and input-token limits

The deadline is one of a query's limits (`QueryLimits`). Two more are off by
default: `search.query_max_model_calls` (every model call of the query: the
router, the query processing, the strategy's own calls and the answer) and
`search.query_max_input_tokens` (the input tokens of those calls, each
counted at its start at the local estimate of its prompt). They count the
calls the query plans: a retry of a call that failed transiently (a stream
cut before its answer text) counts as that call, reusing the room its
failed attempt held, while the usage (`metadata.usage`) counts every attempt
sent. So one query can send up to its backend's query retry attempts
(`retry.model.max_attempts` on Bedrock, `retry.model_server.max_attempts` on
Ollama and OpenAI-compatible servers) times the calls and input tokens the
limits allow, and each attempt's client may resend it on its own (Bedrock up
to `search.model_max_attempts` requests; an OpenAI-compatible client up to
`retry.model_server.max_attempts`, before the response starts). An ingestion
run's budgets (`processing.max_run_*`) count every attempt. A call past one is refused before its request is sent:

- `global` rates and maps, `drift` expands, primes and follows up, and
  `iterative` judges only what fits beside the answer's call. What did not
  fit is left out and listed in `metadata.degraded` (reason `query_limit`):
  global maps its most relevant report batches (under
  `use_dynamic_selection` the reports left unrated are kept, as a report
  whose rating failed, and the descent stops there), DRIFT its best
  follow-ups (an action whose share holds no section of its evidence is
  left out too), iterative stops its rounds (`stop_reason: query_limit`). Each planned
  call is counted as the meter will count it: its prompt, the question it
  carries and what it reads.
- `auto` routes only to strategies whose fewest calls (`fewest_model_calls`,
  the query extractions they read, the translation the question needs, and
  the answer) fit after its router call
  (`metadata.routing.over_limits` names those it left out); when none does,
  to those whose calls fit without the translation and extractions, which
  it then skips as optional calls; and under
  either limit starts no speculative extraction.
- The answer's context is cut to the input tokens left after the answer
  prompt's own templates and instructions and the question, less
  `search.token_manager.context_window_headroom_ratio` of them, and the
  context as built (its section headers and the blank lines between
  sections included) is packed to that room at the meter's estimate, never
  past it, whether the model's token counter counts more or less than that
  estimate, and never left empty while a section fits it; the fan-out strategies keep that prompt's tokens for the answer
  too. A context cut this way, its sections left out or cut short, is listed in
  `metadata.degraded` (step `context_optimization`, reason
  `query_limit: N of M sections cut`).
- One call is kept for the answer, and under an input-token limit its
  input too (its prompt, the question and the least context): an optional
  call that would take either is not sent, and its step is answered without
  (degraded, reason `query_limit`). The optional calls are the query
  translation and extractions, the follow-up rewrite (the
  follow-up is then searched as written), the structured strategy's planner
  (its fallback answers) and an output repair (`models.roles.fixing`; the
  reply that did not parse counts as unrepaired). A query that names `drift`
  keeps the call of its one action too (without it DRIFT finds nothing),
  and under an input-token limit that action's least input. A
  search (`asearch`, `kg_search`, `run-rag --mode search`) makes no answer,
  so keeps nothing for one; an optional call past a limit is then skipped
  the same way.
- Under an input-token limit the fan-out keeps for the answer what it will
  read (the answer prompt, DRIFT's answers so far, the least context),
  not the configured ceilings, and DRIFT's actions share what is left,
  each action's built context kept to its share (an action the meter still
  refuses is left out as one that did not fit, its follow-up's evidence
  kept); `auto` makes no router call the limit cannot take, routing to the
  fallback among the strategies that fit the call limit.
- A query that names a strategy whose answer call does not fit (no section
  of its context fits the room the limit leaves) abstains: it returns the
  fixed decline with `abstention_reason: query_limit`, no answer call made.
  Only `auto` fails up front: when no strategy it may route to fits the
  call limit, it raises `UsageLimitExceededError` (exported from
  `unified_kg_rag`; error class `limit_exceeded`, exit code `1`; a caller's
  error, not counted in the `query_failed` metric) before any call, or
  returns the error answer under `processing.ignore_errors`.
- A query's `effort` (below) raises its answer's `max_tokens` for that
  level's reasoning; the answer's context then takes only what the model's
  window leaves beside that output.

A caller lowers any of the three for one query, never raises them:

```python
from unified_kg_rag import QueryLimits, RAGInput

chain.invoke(RAGInput(query="...", limits=QueryLimits(max_model_calls=6, deadline_seconds=60)))
```

The MCP tools `kg_search` and `kg_answer` take the same `limits` argument
([MCP server](./mcp.md)).

### Reasoning effort

`RAGInput.effort` (`low`, `medium`, `high`, `xhigh`, `max`) sets how hard
one query's answer model reasons, in place of its tier's configured
`aws.bedrock.default_effort` / `fast_effort`
([Models](../reference/models.md)). It turns reasoning on for a model that
opts in, and leaves room for that level's reasoning (its
`EFFORT_THINKING_TOKENS`, `low` 2,048 up to `max` 32,768) on every model
family: the answer's max_tokens is raised above it, past
`models.default_max_output_tokens` and any output ceiling
(`search.output_ceilings`) when needed, never past the model's maximum. Only the answer call takes it; the router,
the query processing and the strategies' own calls keep their tier's. An
older Claude model (4.5 and before), which takes a thinking budget, not a
level, thinks on that level's budget (`low` 2,048 tokens up to `max` 32,768).
An effort the answer model cannot apply is refused before any model call
(`InvalidArgumentError`, error class `invalid_argument`): a level it does not
take, or any level on a model that does not reason (a local server's, or a
Bedrock model without thinking), which would otherwise drop it. Unset,
nothing changes. `run-rag --effort` and `run-eval --effort` set it for every
query of the run (in `run-eval` a dataset item's `metadata.effort` wins, and
every effort the runs ask for, an item's included, is checked for each run's
answer model, an `--arm`'s and the closed-book arm's too, before any run
answers, in either `--run-order`, so one an answer model refuses stops the
run with exit `2` before any answer; the closed-book arm answers at the
item's effort as the strategies do); the
query span (`kg_rag.effort`) and the query metrics record (`effort`) name it.
The MCP tools and the console's Ask page do not take it: unlike a limit, an
effort can raise a query's cost and latency, so an MCP caller's would need a
server ceiling first.
`RAGInput.enable_thinking` is deprecated (a `DeprecationWarning`): it turns
reasoning on at the tier's effort.

```python
chain.invoke(RAGInput(query="...", effort="low"))
```

## Streaming

`GraphRAGChain.aquery_events` (sync: `query_events`) runs the query `ainvoke` runs (its deadline,
limits, guardrail and memory included) and yields typed events, told apart
by `kind`:

| Event | `kind` | When |
|---|---|---|
| `QueryStart` | `start` | The query was admitted: its `query_id`, conversation, requested strategy, namespace and trace id. |
| `QueryRouted` | `routed` | Its strategy is resolved; `routing` says how `auto` chose. |
| `QuerySources` | `sources` | The sources the answer model is given, before the first chunk. |
| `AnswerChunk` | `chunk` | Answer text as the model writes it. |
| `QueryResult` | `result` | The `RAGOutput` `ainvoke` returns. |

```python
from unified_kg_rag import AnswerChunk, QueryResult, RAGInput

async for event in chain.aquery_events(RAGInput(query="...")):
    if isinstance(event, AnswerChunk):
        print(event.text, end="", flush=True)
    elif isinstance(event, QueryResult):
        output = event.output  # sources, citations, usage, metadata
```

- The chunks are a preview; `QueryResult` is authoritative. Under the output
  guardrail the answer model still streams (its first chunk bounded), but
  nothing is shown until the guardrail screened the whole answer: then its
  sources and the screened answer as one chunk, before the turn is written
  to memory, a decline's or an empty context's fixed reply included (without
  the guardrail a fixed reply comes only in the result). The
  `[NO_ANSWER]` token (`search.abstention`) is held back and stripped.
- A failure raises after the events sent before it; with
  `processing.ignore_errors` the result carries the error answer.
- The query waits for the reader to take each event: a reader that stops
  and closes the iterator (`aclose()`, or `break` inside
  `async with aclosing(...)`) ends it there; nothing is written to memory and
  the query metrics count no failure.
- Its query metrics, the `query_id` and namespace of its log lines and
  errors, and its trace are `ainvoke`'s.
- `astream` / `stream` yield the answer text alone: each chunk, or, when
  none was shown, the final answer once. `ainvoke(on_answer_chunk=...)` is
  deprecated (a `DeprecationWarning`): read the `AnswerChunk` events.
  LangChain's own `astream_events` (callback events) is left as LangChain
  defines it.

## Citations

Answers cite the documents they rest on (`search.citations`): those of every
strategy but `global` and `drift`, by default. The answer context numbers its
evidence `[1]`, `[2]`, ... (passages, entities, relationships, claims,
community reports, summaries), and the answer prompt asks the model to cite
them with those markers, ASCII in every answer language. `global` and `drift`
answer without markers unless you turn them on: on sensemaking questions
their answers without citations were more comprehensive
([evidence](../concepts/defaults-and-evidence.md#citations)). With them on,
global search's key points and DRIFT's intermediate answers are
numbered one by one too: each names the documents of the reports (or the
local context) it was written from, as MS GraphRAG's `[Data: Reports (...)]`
does, even when the evidence budget left those reports out of the context.
A report, in every strategy that reads reports, names the documents of the
entities and relationships its findings cite, not of its whole community (a
report indexed before its findings were kept falls back to the community; in
a GraphRAG-only namespace, which has no relationships index, it names its
entities' documents). They are model output, so they are never
listed as sources. A marker has
at most three digits, so a bracketed year such as `[2020]` is text:

```text
Depot stores the widgets for Vendor [1][3].
창고는 공급사의 부품을 보관합니다 [2].
```

`RAGOutput.citations` lists one record per number: `marker`, `cited` (the
answer uses it), `section_type`, `title`, the main `document_id` /
`file_name` / `file_path` (relative to the corpus root) / `page` and
`page_end`, `chunk_id` (a passage), `text_unit_ids`, every document as
`documents`, and seed-graph `provenance` and `iri`. A graph section names the
documents of up to `search.citations.lineage_units` of its text units that
the query's filters and suffix let through (batched store reads: a document
they exclude is never named, and the next units are read in its place, in
at most four reads). A number the answer uses that the
context did not show is in `metadata.unknown_citation_markers`, never a
citation. The store read runs alongside the answer call. A follow-up's
rewrite reads the earlier answers without their markers.

- `run-rag` prints the cited sources under the answer; `--output-format
  json` includes every citation.
- PDF passages carry their pages; text ingested before this release has no
  pages or relative paths until its documents are ingested again.
- A streamed answer carries the same markers: read them from the chunks with
  `CitationMarkerScanner` (a marker split across chunks is held back until it
  closes). `astream` returns text only; the `QueryResult` of
  `aquery_events`, as `ainvoke`, returns the citations.
- A blocked answer has no citations; a masked one keeps them, flagged from
  the masked text; a declined answer keeps them like its sources.
- Per strategy: `search.citations.strategies` is merged over its default
  `{global: false, drift: false}`, and an entry wins over `enabled` either way.
  `auto` follows the strategy it routes to, and `iterative` follows its base
  (`search.iterative_search.base`) unless it is named itself. A strategy that
  does not cite gets an unnumbered context, no citation instruction and no
  `RAGOutput.citations`, and `run-rag`, `kg_answer` and the console show no
  markers for it.

  ```yaml
  search:
    citations:
      strategies: {global: true, drift: true}  # cite in global and DRIFT too
  ```

- `search.citations.enabled: false` restores the earlier context and prompt
  for every strategy the map does not name. The markers an answer asked to
  cite writes stay in its text; `run-eval`'s `answer_match` scores the words
  without them, and an answer not asked to cite as written.

## Declined answers

When retrieval returns nothing, the chain answers "I could not find relevant
information..." without calling the model, in the question's language (one
fixed sentence per supported language, told by the question's script and
function words; the index language's, or English, when they do not tell),
and sets `metadata.abstained` and `metadata.abstention_reason: no_context`
(`no_reports` for `global` on a namespace that holds no community report: a
corpus too small for communities, or one ingested without them;
`query_limit` when the query's limits left out the steps that would have
found a context).
Tell a decline by `metadata.abstained`, not by the text. A model that declines in its own
words is otherwise an ordinary answer. `search.abstention` (opt-in, off by
default) makes declines explicit:

```yaml
search:
  abstention:
    mode: mark            # off (default) | mark
    strip_marker: true    # remove the token from the returned answer
    on_insufficient: hedge   # hedge (default) | abstain
```

- **`mode: mark`** appends an instruction to the answer prompt (concise,
  comprehensive or your `custom_prompts` override): when the context does not
  support the answer, start with the fixed token `[NO_ANSWER]` and say, in the
  query's language, what is missing. An answer that starts with it gets
  `metadata.abstained: true` and `abstention_reason: marker`, in any answer
  language (full-width `［NO_ANSWER］` and leading Markdown emphasis are
  accepted). The token is removed unless `strip_marker: false`; an answer that
  is only the token becomes the fixed no-context reply. `stream` holds back at
  most the first 32 characters until it knows.
- A declined answer keeps its `sources`: the model saw that context, and an
  evaluator can audit it. Only a guardrail block (or an empty context) leaves
  none. The output guardrail screens the answer without the token.
- **`on_insufficient`** applies only when the strategy flagged its own context
  insufficient (`search_results.metadata.context_sufficient: false`, with an
  optional `insufficiency_reason`); `iterative` is the shipped strategy that sets it.
  `hedge` asks the model to answer what the context supports and say what it
  does not; `abstain` tells it not to answer and to write the decline in the
  query's language, and reports the output abstained
  (`abstention_reason: insufficient_context`). Neither adds a model call.
  Measured with `iterative`, `abstain` declined many answerable questions
  and lost token F1, so `hedge` stays the default
  ([evidence](../concepts/defaults-and-evidence.md#declines-and-abstention)).
- `mark` changes the answer prompt, so answers and EM / F1 change. Measured
  on multi-hop questions, it turned hedged correct answers into declines
  (lower EM) for machine-readable declines, so it stays opt-in
  ([evidence](../concepts/defaults-and-evidence.md#declines-and-abstention));
  compare `mark` runs as their own arm. How declines are scored:
  [Evaluation: unanswerable questions](./evaluate.md#unanswerable-questions-and-declines).

## Follow-up rounds for multi-hop questions

`--search-strategy iterative` runs `search.iterative_search.base` (`mix` by
default, or `local`) and lets a fast model (`models.roles.query_controller`)
judge the context before the answer: when the context does not answer the
question, the base search reruns from the bridge entities the controller
names (the entity the missing hop is about, such as the depot in "where is
the depot of the carrier the vendor hired?") and its follow-up queries search
the passages, for at most `max_rounds` (default 2) rounds. The answer model
answers once, from the fused candidates. No default selects it, but `auto`
may route a question that chains facts through an entity it does not name to
it (`search.auto_routable_strategies`). Over `mix` it answered multi-hop
questions significantly better than `mix` alone, and the median query took
markedly longer
([evidence](../concepts/defaults-and-evidence.md#iterative-follow-up-rounds)).
To keep `auto` off it, list the routable strategies without it:
`search.auto_routable_strategies: [local, mix, global]`.

```yaml
search:
  iterative_search:
    base: local          # or mix (default)
    max_rounds: 2        # follow-up rounds after the first search
    max_next_queries: 2  # follow-up queries per round
    round_top_k: 5       # passages each follow-up query retrieves
models:
  roles:
    query_controller: fast   # the controller's model (default: the fast tier)
```

- **Cost.** At most `max_rounds + 1` controller calls on the fast tier,
  a base search and a rerank per round;
  the answer is generated once. `metadata["iterative"]` reports the rounds,
  why they stopped (`stop_reason`: `sufficient`, `max_rounds`,
  `no_follow_up`, `deadline`, or a degraded step), each verdict, the
  follow-up queries and the rounds' own model usage.
- **Deadline.** A round starts only while `search.query_timeout_seconds`
  leaves `search.model_timeout_seconds` for the answer; a round past that is
  cancelled and the answer reads what the earlier rounds found.
- **Languages.** The follow-up queries and bridge names are written in the
  query's language (the index's when translation is off).
- **Filters** apply in every round. **Sources** are the passages and graph
  items the answer model read, with the same provenance as the base's.
- **Insufficient context.** When the controller still finds the context
  insufficient after the last round, `metadata["context_sufficient"]` is
  `false` (the stop reason in `insufficiency_reason`), and
  `search.abstention.on_insufficient` decides whether the answer hedges or
  declines ([Declined answers](#declined-answers)).

How it works and how it compares with IRCoT and KAG:
[design](../design/retrieval-graphrag-lightrag.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in).

## Summary-tree search (opt-in)

`--search-strategy tree` searches a namespace built with the `raptor`
methodology ([ingestion](./ingest.md#a-raptor-summary-tree-opt-in)): the
passages and every level of the summary tree at once, BM25 + kNN each,
fused and reranked together; each summary kept brings its best passages.
It is opt-in: on multi-hop questions it answered better than `naive` but
well below the graph strategies, in one repeat on one corpus ([evidence](../concepts/defaults-and-evidence.md#raptor-summary-tree)):
`auto` routes to it only on a `raptor`-only namespace, and a namespace
without a tree refuses it before any model call.

```yaml
search:
  tree_search:
    summary_top_k: 10        # summaries searched for and kept
    text_top_k: 20           # passages searched for and kept
    max_nodes_per_level: 3   # summaries of one level at most
    expand_children: 2       # passages each summary brings (0: none)
    graph_signal: auto       # or off: the entity-led buckets
  token_manager:
    type_budgets:
      summary: 0.5           # the summaries' share beside text's 0.5
```

- **Sources.** A summary is reported with `level`, `n_leaves`, `child_ids`
  and `text_unit_ids` (its passages), and its expanded passages with
  `expanded_from` (the summary's id); `run-eval`'s retrieval evaluator
  attributes a summary through its passages (a `summary` row in
  `by_source_type`).
- **Graph signal.** Where the namespace also holds the entities index
  (`graphrag` or `lightrag` selected), the entities a search with the raw
  query finds add the summaries and passages that cite them (no model call;
  `metadata["graph_signal"]` says whether it ran).
- **Filters.** A summary keeps a document filter key only when all its
  documents share the value: under a filter it is kept only when all its
  documents pass, and its passages are fetched under the same filters.
- **Cost.** A `naive` query plus a few searches and a wider rerank; no
  model call before the answer.

How it works and how it differs from RAPTOR's own code:
[design](../design/retrieval-hipporag-raptor.md#65-raptor-summary-tree-raptor-tree-opt-in).

## Structured graph queries (opt-in)

`--search-strategy structured` answers counts, rankings, lists, paths
between two named entities and seed-entity properties from a query the
graph store runs: a model writes the question as a JSON plan (never query
text), code checks it and runs it read-only within the query's suffix, and
the answer reads the result table, the entities it names and their
passages. Without a structured answer the query runs
`search.structured.fallback_strategy` (`local`); `metadata["structured"]`
says what happened. Opt-in and unmeasured: `auto` routes to it only once it
is in `search.auto_routable_strategies`. The guide:
[Structured graph queries](./structured-queries.md).

```bash
run-rag --query "Which vendor supplies the most buyers?" \
  --search-strategy structured --config-path config.yaml
```

## Interactive mode & conversation memory

```bash
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

Interactive mode turns memory on. Commands: `help`, `new` (a fresh
conversation id), `set-filter key:value`, `clear-filters`, `show-config`,
`quit` / `exit`. A command is the whole line (only `set-filter` takes an
argument): `help me find ...` is a question. `set-filter` refuses a key no
index of the search strategy declares, as `--filters` does (with
`--interactive`, before the first question: exit code 2). With `auto` it
refuses a key only when, with the filters already set, no strategy `auto`
may route to applies them all.

Conversation memory lives in the process (`MemoryManager`, limits in the
`memory` section):

- It carries over between the turns of one session, or between calls of one
  `GraphRAGChain` reusing a `conversation_id` with `use_memory=True` and the
  same `suffix` and `user_id` (other tenants' or users' turns never reach
  it). Chains share it only through one `MemoryManager(config)` passed as
  `memory_manager=`.
- Separate `run-rag --query` invocations start with no history, so
  `--conversation-id` does nothing there (`run-rag` logs a WARNING); use
  `--interactive`, which always keeps the session's memory, for multi-turn
  from the CLI.
- A follow-up is rewritten from the history as a standalone question before
  retrieval (one `models.roles.query_rewrite` call per turn with history;
  prompt key `query_rewrite`), so "what about its suppliers?" names the
  entity from the earlier turn. The rewrite reads the last
  `memory.max_messages_per_conversation` messages, each earlier answer cut to
  1,000 characters, and runs even with `enable_query_processing` off (that
  flag covers translation and entity/keyword extraction). Memory adds no
  other LLM call.

### History from the caller

A server that keeps the conversation itself (one request per turn) passes it
as `RAGInput.history` instead of using memory:

```python
chain.invoke(RAGInput(
    query="And its price?",
    history=[
        {"role": "user", "content": "Which part does Vendor supply?"},
        {"role": "assistant", "content": "Vendor supplies the widgets [1]."},
    ],
))
```

The follow-up rewrite reads it as it reads memory: the last
`memory.max_messages_per_conversation` turns, each answer cut to 1,000
characters, and only the most recent turns that fit
`memory.max_history_tokens` (default 4,000, estimated). With the input
guardrail on, the turns are screened like the query: one check for all of
them, then one per turn only when it intervenes; a masked turn is used
masked, a blocked one is dropped (both in `metadata.guardrail`), and the
query goes on. `history` and `use_memory=True` together are refused; a query
with `history` writes nothing to memory.
