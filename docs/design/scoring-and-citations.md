# Design: hybrid scoring and citations

Part of the [design reference](../design.md), which lists every section. Code cites
the section numbers: they never change.

## 7. Hybrid Scoring and Token Management

- **HybridScorer** (`adapters/retrieval/hybrid_scorer.py`) fuses the
  per-source results by RRF (`rrf_k`) or weights, reranks, applies MMR
  diversity (`diversity_lambda`) and cuts to the per-type quota or `top_k`.
  Reranking comes before diversity (as upstream LightRAG reranks every chunk
  candidate before truncating), so a cut on fused scores cannot drop what the
  reranker ranks first. Each query records its rerank outcome on its own
  `RerankTrace`, reported as `SearchResult.metadata["rerank"]`; a rerank model
  that cannot be built leaves the fused order (`unavailable`) or raises under
  `search.reranking.required`.
- **TokenManager** (`adapters/retrieval/token_manager.py`) packs the context
  into the answer model's budget. Each section type gets a hard-capped share
  (`search.token_manager.type_budgets`); within it sections go in score order
  and the first that overflows is truncated. Budget a type leaves unused goes
  to the types that still have candidates (first to the strategy's
  `spare_context_budget_first` types, as LightRAG gives chunks what its graph
  sections leave), so shares, never cross-type score comparisons, decide the
  mix. The shares are of `max_context_tokens`, not upstream's numbers: at
  30000, LightRAG mix/hybrid get shares of text 18072, entities 7228 and
  relationships 4698 tokens (upstream: about 16000 / 6000 / 8000); after the
  512-token prompt buffer, the question and the section headers about 17700
  / 7050 / 4600 are packed. Synthesized global
  sections are seated first. `RAGOutput.sources` is
  built from this selection, so it lists only what the answer model saw,
  truncated text included. In the answer context every evidence section
  starts with its citation marker `[n]` (§7.1); with citations off, and in
  the intermediate prompts of DRIFT and iterative search, passages, claims and
  reports render with an `[id]` prefix and entity and relationship lines
  without one, since nothing reads those ids and a UUID cost about a quarter
  of their tokens. An entity line shows the entity's type, from either store.
- **Token counting** (`adapters/aws/token_counter.py`,
  `adapters/local/token_counter.py`). The Bedrock `count_tokens` API is used
  for language models that accept it, with a script-aware estimate otherwise
  (`adapters/llm/token_estimate.py`); CountTokens rejects the default Claude
  5.x models and every embedding and rerank model, so with the defaults every
  count is the estimate. A non-transient failure marks the model unsupported
  for the process. Local models use the estimate or tiktoken.

### 7.1 Document citations

An answer cites the documents it rests on. The parts, on by default for
every strategy but `global` and `drift` (`search.citations.enabled`; off
restores the context and prompts of earlier releases and returns no
citations):

- **Per strategy.** `search.citations.strategies` maps a strategy to on or
  off over `enabled` (an entry wins either way), merged over the default
  `{global: false, drift: false}` like `answer_prompt_style`: on 50
  sensemaking questions the pairwise judge preferred their uncited answers on
  comprehensiveness
  ([evidence](../concepts/defaults-and-evidence.md#citations)).
  `citations.citations_enabled(config, strategy)` resolves it once for the
  chain, the evaluators, the run manifest and the console, for the strategy
  a query ran: `auto`'s route (an `auto` key is refused), and for a wrapper,
  its own entry, else its base's (`iterative` over `mix` cites as `mix`
  does; `wrapped_strategy`), else `enabled`. `structured` is a strategy like
  the others (on). A strategy that does not cite gets an unnumbered context
  (`number_sections=False`), no citation instruction, no
  `RAGOutput.citations` and no `unknown_citation_markers`; the evaluators
  read its answers as written and skip its citation metrics
  (`citations_disabled`). The run manifest records the resolved setting of
  every strategy (`citations.strategies`), so `--baseline-results` and
  `--from-results` compare, and `--rescore` reads, each strategy's answers
  as they were written (a run saved before it followed `enabled`
  everywhere).

- **Numbered context.** `optimize_context(..., number_sections=True)` (the
  chain's answer context only; DRIFT's and iterative's intermediate prompts
  stay unnumbered, so a number they write never reaches the answer as if it
  were one) gives every retrieved section a marker `[1]`, `[2]`, ... in the
  order the context string renders them (by section-type block, then by
  rank). The marker replaces the `[id]` prefix of passages, claims and
  reports and is added to entity and relationship lines (about 3 tokens; a
  section is counted with a `[999]` placeholder so the budget holds, and
  sections past the 999th get none: an answer's marker has at most three
  digits). A
  synthesized section (global search's key points, DRIFT's intermediate
  answers) is model output, not evidence, and never a source; each point or
  answer is a section of its own carrying the lineage of what it was written
  from (`evidence_lineage`: the text units and documents of the reports a
  point credits, of a primer fold's reports or of the sections of an
  action's local context) and flagged `citable`, so it gets a marker and its
  citation resolves that lineage like a report's: MS GraphRAG's reduce
  cites `[Data: Reports (...)]` and DRIFT's `[Data: Sources (...)]`, and the
  documents are named even when `evidence_tokens` left the reports out. A
  report's lineage there is that of the records its findings cite
  (`BaseSearchStrategy._findings_lineages`: `report_cited_records`, each
  finding's relationships before its entities, read by id from the
  entities and, where the namespace holds it, the relationships index, one
  fetch each), not its community's membership, which a corpus-spanning
  community shares with every point drawn from it; a report stored without
  findings, whose findings cite nothing, or none of whose cited records is
  stored any more (or whose read failed: a `findings_lineage` degraded
  step of an answer that cites, `metadata.citations_unresolved` of one
  whose strategy's citations are off) keeps the membership. The records are read in the query's namespace
  (index prefix and suffix) without the caller's filters: they say what
  the report rests on, and the citation applies the filters to their units
  (Resolution below); read under the filters, a record of another document
  came back missing and the point fell back to the membership, the
  caller's own documents. A namespace without the relationships index
  (GraphRAG alone) cites the entities its findings cite: the graph store
  holds the edges but has no read by edge id. The report sections of
  `local`, `global` and DRIFT carry the same lineage
  (`findings_text_unit_ids` / `findings_document_ids` beside the
  membership's `text_unit_ids`), so a report's citation and a point's
  written from it name the same documents. A synthesized section without
  lineage gets no marker. Its text is screened by the output guardrail:
  joined checks of at most 12,000 characters each, one after another
  (`_JOINED_CHECK_CHARS`: ApplyGuardrail allows 25 text units of 1,000
  characters a second by default, and AWS's long-context guidance starts
  from chunks of 12; a longer section is checked alone), then, for a chunk whose check intervened
  (recorded in `metadata.guardrail` with `scope: joined`), one per section
  (`search.max_concurrency` at a time), each masked or dropped as before. A marker has at most three digits,
  read and stripped alike, so a bracketed year (`[2020]`) is text; an
  earlier answer reaches the follow-up rewrite without its markers
  (`history_buffer`). The lineage read runs alongside the answer call
  (`_answer_with_citations_step`; a failed answer cancels it).
- **Prompt.** `CITATION_INSTRUCTIONS` is appended to the system prompt of
  whichever answer prompt runs (concise, comprehensive or a `custom_prompts`
  override), like the abstention text: cite the supporting items after each
  statement with their markers, written as is (ASCII digits in square
  brackets) in every answer language, and only markers the context shows.
  It is the same text for every strategy that cites, `global` and `drift`
  turned on included. An earlier "cite where possible" suffix for those two
  (keep a relevant point with no marker to cite) departed from MS GraphRAG's
  global reduce prompt ("Do not include information where the supporting
  evidence for it is not provided"); against the same answers without it,
  it gained no comprehensiveness and lost `global` directness, so it was
  removed, and with citations off by default their answer prompts are the
  ones before citations.
- **Resolution** (`application/retrieval/citations.py`). `RAGOutput.citations`
  lists one `Citation` per numbered section: its marker, section type,
  source id, title (file, entity, `A -> B`, report title), seed-graph
  provenance (`seed_provenance`) and IRI, the text units it rests on and the
  documents they belong to (`CitedDocument`: document id, file name, file
  path relative to the corpus root, pages). A passage names its own unit,
  document and pages. A graph section (entity, relationship, claim, report,
  summary) resolves through its `text_unit_ids` lineage: up to
  `search.citations.lineage_units` units, those in the context first, are
  read from the text-units index in batched fetches by id (in-context
  units need none) under the query's suffix and the caller's filters that
  index declares, so a document the filters exclude, or another tenant's,
  is never named; units that do not come back are left out, and the next
  ones of the lineage are read in pages (what a citation still lacks,
  doubled each page) until it has the limit or its lineage is read, four
  pages at most (`MAX_LINEAGE_PAGES`: a citation reads at most 15 times
  `lineage_units` of its units, and names no document when none of them
  passes). Its documents
  are ordered by how many of those units each holds; the first is the
  citation's `document_id` / `file_name` / `file_path` / `page`. Seed units
  (`sg-...`) are virtual documents: they are not looked up, and a seed-only
  fact cites its seed graphs and IRI (`provenance.kind: seed_graph`) with no
  document. A failed lookup leaves the graph citations without documents
  (`metadata.citations_unresolved`), never fails the answer.
- **Markers in the answer** (`domain/retrieval/citations.py`). `[1]`,
  `[1, 2]`, `[1][3]`, `[1-3]` and the full-width forms CJK answers use
  (`［1］`, `【1】`) are read; `Citation.cited` marks the ones the answer
  uses. A number the context did not show is never a citation: it is listed
  in `metadata.unknown_citation_markers`. `CitationMarkerScanner` reads the
  same markers from streamed chunks (a marker split across chunks is held
  back until it closes), so a client of `astream` sees the citations as they
  are written; the citations of an answer that was streamed are parsed from
  the whole text, as `ainvoke` does.
- **Edges unchanged.** A blocked answer returns no citations (as no
  sources); a masked one keeps them, their `cited` flags read from the
  screened text. An empty context returns none (no model call); a declined
  answer keeps them, like its sources.
- **Pages.** The document converter records where each page starts in the
  combined text (`page_offsets`, 1-based page numbers) when the loader emits
  one document per page (PDF). The chunker locates each chunk in the text
  (in order, never searching before the previous chunk) and stores
  `page_start` / `page_end` in the text unit's attributes, beside its
  `relative_path`. Chunk boundaries, texts and ids do not change, so no
  stored record is re-keyed; a namespace indexed before shows pages and
  relative paths only for documents indexed again.
- **Deviations from the originals.** LightRAG returns a reference list of
  file paths numbered per document, and MS GraphRAG asks for `[Data:
  Entities (ids); Sources (ids)]` records. A marker per section is cheaper
  in the prompt than either (no id lists) and maps every graph section to
  its documents after the answer, from lineage the stores already hold.
- **Evaluation.** The retrieval evaluator adds `citation_precision` (cited
  citations naming a reference file / cited citations naming a file) and
  `citation_recall` (references named by a cited citation / references)
  against `reference_sources` (`evaluation/citation_scoring.py`).

### 7.2 Caller-supplied conversation history

`RAGInput.history` (a list of `{role: user | assistant, content}` turns,
oldest first) is the conversation for a stateless caller (a web server, the
MCP server) that keeps it itself. It takes the place memory has in the chain:
the follow-up rewrite reads it, and nothing else does. It is bounded before
use: the last `memory.max_messages_per_conversation` turns, each answer cut to
1,000 characters as memory cuts it, then the oldest dropped until the rest fit
`memory.max_history_tokens` (the script-aware estimate). With the input
guardrail on, the bounded turns are screened as the query is (`INPUT`): one
check over all of them, and only when it intervenes one per turn; a masked
turn is replaced and a blocked one dropped (recorded in `metadata.guardrail`),
so a client replaying a refused turn is not refused forever. `history` and
`use_memory` together are refused (`ValueError`); without `history` memory
works as before, and a query with `history` writes nothing to memory.

---
