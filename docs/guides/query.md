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

# HippoRAG 2 PageRank (opt-in; auto never routes to it)
run-rag --query "Where was the founder of the company that acquired X born?" \
  --search-strategy ppr --config-path config.yaml

# Verbose (entities, top sources, metrics) with attribute filters
run-rag --query "..." --verbose --filters attr_category:research type:person \
  --config-path config.yaml
```

## Filters

`--filters key:value ...` (or `RAGInput.filters`) narrows what retrieval ranks;
a pair without a key, a `:` or a value, a key given twice and a key no index the
strategy reads declares are usage errors (exit 2).

- A filter applies only to the stores that declare its field
  (`type:person` leaves text units unfiltered). A key no store the selected
  strategy reads declares (e.g. `category`, or `type` with `naive` or
  `global`) raises `InvalidFilterError`, listing the filterable keys. `auto`
  routes only among the strategies it may route to that apply every key.
- `attr_<key>` matches entry `<key>` of a document's `filters` metadata.
  OpenSearch applies it strictly (fail-closed): an item without the
  attribute is excluded. An entity, relationship, claim or community report
  from several documents matches when one of its documents does.
- Entity and claim `type` is stored lowercase: filter `type:person`, not
  `type:PERSON`. Boolean and numeric fields take their type's values
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
[Design §6.4](../design.md#64-hipporag-2-personalized-pagerank-ppr-local-ranking).

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
| `unavailable` | The rerank model could not be built (initialization failed, logged at ERROR, or the cross-encoder's `[local]` extra is missing). |
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
  overall deadline applies.
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

## Citations

Every answer cites the documents it rests on (`search.citations`, on by
default). The answer context numbers its evidence `[1]`, `[2]`, ... (passages,
entities, relationships, claims, community reports, summaries; not global
search's key points or DRIFT's intermediate answers, which are model output),
and the answer prompt asks the model to cite them with those markers, ASCII
in every answer language:

```text
Depot stores the widgets for Vendor [1][3].
창고는 공급사의 부품을 보관합니다 [2].
```

`RAGOutput.citations` lists one record per number: `marker`, `cited` (the
answer uses it), `section_type`, `title`, the main `document_id` /
`file_name` / `file_path` (relative to the corpus root) / `page` and
`page_end`, `chunk_id` (a passage), `text_unit_ids`, every document as
`documents`, and seed-graph `provenance` and `iri`. A graph section names the
documents of up to `search.citations.lineage_units` of its text units (one
batched store read per query, under the query's filters and suffix: a
document they exclude is never named). A number the answer uses that the
context did not show is in `metadata.unknown_citation_markers`, never a
citation.

- `run-rag` prints the cited sources under the answer; `--output-format
  json` includes every citation.
- PDF passages carry their pages; text ingested before this release has no
  pages or relative paths until its documents are ingested again.
- A streamed answer carries the same markers: read them from the chunks with
  `CitationMarkerScanner` (a marker split across chunks is held back until it
  closes). `astream` returns text only; `ainvoke` returns the citations.
- A blocked answer has no citations; a masked one keeps them, flagged from
  the masked text; a declined answer keeps them like its sources.
- `search.citations.enabled: false` restores the earlier context and prompt.
  The markers stay in the answer text; `run-eval`'s `answer_match` scores the
  answer as written.

## Declined answers

When retrieval returns nothing, the chain answers "I could not find relevant
information..." without calling the model, in the question's language (one
fixed sentence per supported language, told by the question's script and
function words; the index language's, or English, when they do not tell),
and sets `metadata.abstained` and `metadata.abstention_reason: no_context`.
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
[design](../design.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in).

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
[design](../design.md#65-raptor-summary-tree-raptor-tree-opt-in).

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
  `--use-memory` and `--conversation-id` do nothing there (`run-rag` logs a
  WARNING); use `--interactive` for multi-turn from the CLI.
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
