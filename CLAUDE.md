# Project Guide for Claude Code

AWS-native Knowledge Graph RAG framework. Reimplements the Microsoft GraphRAG
and LightRAG methodologies on Bedrock + Neptune + OpenSearch, with DynamoDB for
incremental-indexing state.

## Architecture (hexagonal / ports & adapters)

The package is laid out in explicit hexagonal layers so the two RAG
methodologies (GraphRAG community-summary, LightRAG dual-level keyword) share
one ingestion/indexing/caching/hybrid-search infrastructure and only differ at
the algorithm layer. Dependencies point **inward**: `application → adapters →
ports → domain`, with `shared` as a cross-cutting kernel any layer may use.

- **`domain/`**: technology-agnostic core — `models/` (Pydantic), pure
  algorithms (`ingestion/` merge/delta/resolve/analyze, `retrieval/`
  strategy registry + mixins), and `prompts/` templates. No boto3/LangChain/
  backend imports (enforced by review; verifiable with grep).
- **`ports/`**: the abstract interfaces the domain depends on — `DocStatusPort`
  (`Protocol`) and the write-side indexer ABCs `BaseIndexer`/`GraphIndexer`/
  `VectorIndexer` (+ `IndexingStats`). `ports/__init__` is the port catalog and
  documents that the retrieval/evaluation abstract bases are *adapter bases*
  (they construct infra in `__init__`) re-exported for discovery.
- **`adapters/`**: concrete technology bindings — `aws/` (Bedrock, Neptune,
  OpenSearch, DynamoDB, S3), `storage/` (indexers), `retrievers/`,
  `search_strategies/`, `retrieval/` (base + hybrid scorer + token/memory
  managers), `ingestion/` (LLM/IO-coupled chunker/extractor/loader/parser/
  translator), `renderers/`, `evaluators/`.
- **`application/`**: orchestration + entry points — `cli/` (pyproject scripts
  resolve to `unified_kg_rag.application.cli.*`), `ingestion/` (pipeline + stages),
  `storage/indexing_manager`, `retrieval/rag_chain`.
- **`shared/`**: cross-cutting kernel — config, logging, exceptions, metrics,
  cache/pipeline managers, `utils/`.
- **`evaluation/` and `visualization/`**: real logic packages (evaluators +
  manager; render loop + `embeddings/`/`exporters/`/`renderers/`). Import
  everything else from its real location (`application.retrieval.rag_chain`,
  `application.storage.indexing_manager`, `application.ingestion.pipeline`,
  `adapters.*`, `domain.*`).
- **Registries over hardcoded dispatch**: search strategies register via
  `@register_strategy` (`domain/retrieval/strategy_registry.py`). Follow this
  pattern — and `ParserFactory.register_loader` /
  `EvaluationManager._resolve_evaluator_class` — instead of `if/elif` dispatch.

See `docs/design.md` §2 for the full layer map and dependency rule.

### Adding things
- **New search strategy**: subclass `BaseSearchStrategy`, decorate with
  `@register_strategy(SearchStrategy.X, required_roles=(...))`, export from
  `adapters/search_strategies/__init__.py`. No edits to `rag_chain` needed.
- **New storage/LLM backend**: implement the relevant port and inject it —
  `IndexingManager(vector_indexer=…, graph_indexer=…)`,
  `GraphRAGChain(model_factory=…, retriever_builders=…)`,
  `*(embedding_factory=…)`. Defaults stay Bedrock/Neptune/OpenSearch; don't
  hardcode a backend into a manager's `__init__`. (See design.md §15 "Custom
  backends".)
- **New parser / file format**: `ParserFactory.register_loader(".ext", Loader)`
  with any LangChain `BaseLoader` subclass — auto-discovered and parseable, no
  factory edit.
- **New evaluator**: subclass `BaseGraphRAGEvaluator`, add a branch in
  `EvaluationManager._resolve_evaluator_class`, add an `EvaluatorType` enum.
- **New config section**: add a Pydantic `BaseModel`, attach it to its parent
  via `Field(default_factory=...)`, document it in `config-template.yaml`.

## Capabilities & CLIs

- **Two retrieval methodologies** (user-selectable via `RAGInput.search_strategy`):
  GraphRAG community-summary (`auto`/`drift`/`global`/`local`/`simple`) and
  LightRAG dual-level keyword (`mix`/`hybrid`/`naive`). Both share the same
  ingestion, indexing, caching, multilingual, and hybrid-scoring infrastructure;
  only the retrieval algorithm differs. LightRAG `mix`/`hybrid` extract high/low
  keywords (`KeywordsExtractionPrompt`) and query a relationship vector index
  (high-level) + entity index (low-level) + Neptune expansion.
- **Incremental indexing**: enable `aws.dynamodb` to diff a corpus by content
  hash and only (re)index new/changed documents, merging into the live graph
  (`IncrementalIndexer`, `ingestion/merge/`, idempotent `upsert_*`/`delete_by_id`
  on both indexers). Deletions remove a document's *exclusive* artifacts via
  per-document lineage in the DynamoDB registry.
- **CLIs** (`pyproject` scripts): `run-ingestion`, `run-rag`, `run-eval`,
  `run-visualization` (render from exported graph data, no ingestion),
  `run-prompt-tuning` (profile a corpus → domain-adapted `custom_prompts` YAML).
- **Evaluation**: `langchain` + `ragas` (text similarity) plus `graph_aware`
  (entity/relationship coverage = recall, from ground-truth
  `expected_entities`/`expected_relationships`; precision/F1 intentionally not
  emitted — see the evaluator docstring). Add an evaluator by subclassing
  `BaseGraphRAGEvaluator` + a branch in `EvaluationManager._resolve_evaluator_class`.
- **Visualization**: renderers register via `@register_renderer`
  (`visualization/renderers/`); the manager and the standalone CLI drive them
  through one registry + `RenderContext`.

## Code standards

- **Types**: modern built-ins (`list`, `dict`, `X | None`); no legacy `typing.List`.
- **Models**: Pydantic at all boundaries (not dataclasses, except frozen config records).
- **Files**: `pathlib`, not `os.path`.
- **Packages**: `uv` (`uv sync --extra dev`), not pip.
- **Logging**: `%`-formatting, not f-strings — `logger.info("did %s", x)`.
- **LLM calls**: LangChain LCEL (`prompt | llm | parser`); prompts live in
  `unified_kg_rag/domain/prompts/*.py` for version control, overridable via
  `custom_prompts` config.
- **Exceptions**: specific custom types from `shared/exceptions.py`; fail fast at
  boundaries; degrade gracefully only where recovery is meaningful.

## No customer data (MANDATORY)

This repo is destined for public/awslabs release. Customer-identifying or
customer-confidential content MUST NEVER appear in application code, tests,
evaluation datasets, fixtures, prompts, comments, docs, diagrams, or commit
messages. This includes: customer/project names and codenames (e.g. internal
engagement names), real contract figures/clauses, document filenames from a
customer corpus, account ids, endpoints, and internal hostnames.

- **Tests/fixtures/eval**: use neutral *synthetic* data (generic entities like
  "Vendor"/"Buyer", made-up amounts). Never paste real customer text.
- **Real-corpus E2E**: keep customer corpora out of the repo entirely — local
  only, under the gitignored `.test-corpus-backup/`, synced to S3 at runtime.
- **Before committing**: grep the diff for customer terms; if any slips into
  history, a history scrub (squash/filter) is required before publish.

## Testing

Layout: `tests/{unit,integration,property,fixtures/fakes}/`. Markers: `unit`,
`integration`, `property`, `aws` (real AWS — skipped in CI), `slow`.

- Tests run **AWS-free by default**. Use the port-based in-memory fakes in
  `tests/fixtures/fakes/` (e.g. `FakeDocStatusStore`) instead of mocking boto3
  ad hoc; use `moto` when an adapter must be exercised against a boto3 surface.
- `pytest-asyncio` is in `asyncio_mode = "auto"` — `async def test_*` just works.
- Property tests (`hypothesis`) cover invariants: hashing determinism, diff
  partition completeness, merge laws, fusion monotonicity.
- Coverage gate ratchets up per milestone toward 80% (currently `--cov-fail-under=78`
  in CI). Run: `uv run pytest -m "not aws" --cov=unified_kg_rag`.

## Quality gate

CI (`.github/workflows/quality.yml`) and `.pre-commit-config.yaml` run ruff,
black, isort, mypy, and pytest+coverage. Install hooks with `pre-commit install`.

## Git & collaboration workflow

This is a **public awslabs repository**. Treat `main` as protected shared
history — the loose "commit straight to main" habits from the private prototype
do **not** carry over. These rules apply to human contributors **and** to Claude
Code working in this repo.

- **Never commit or push directly to `main`.** Do all work on a topic branch
  (`feat/…`, `fix/…`, `docs/…`, `chore/…`) and land it through a Pull Request,
  even for solo work — the PR is the review checkpoint and the audit trail.
- **One logical change per PR/commit.** Keep them small and focused; don't mix a
  refactor, a feature, and a doc fix in one commit.
- **Conventional Commits** for every message: `type(scope): summary` in the
  imperative mood, ≤72-char subject. Types: `feat`, `fix`, `docs`, `refactor`,
  `test`, `chore`, `style`, `perf`, `ci`, `build`. Body explains *why*, not
  *what* the diff already shows.
- **Green before you push.** Run `uv run pytest -m "not aws"` and the pre-commit
  hooks (ruff/black/isort/mypy) locally; never push a commit that fails the CI
  quality gate. Fix formatting *before* committing, not in a follow-up.
- **No force-push to `main`, no history rewrites** on the published branch.
  History rewriting is reserved for a one-off customer-data scrub (see below) and
  requires explicit owner sign-off — it breaks every clone, fork, and open PR.
- **Never merge your own PR without review** when a second maintainer is
  available; for genuinely solo changes, still open the PR, let CI pass, and use
  a squash-merge so `main` keeps one clean commit per change.
- **Ask before side-effecting actions** — pushing, opening/merging PRs, tagging
  releases, and any AWS resource creation or real-AWS tests. Approval for one
  action is not standing approval for the next.
- **Customer-data gate is a hard blocker.** Before every commit, grep the staged
  diff for customer terms (see *No customer data* above). If any customer data
  has already reached history, publication is blocked until a history scrub
  (orphan-squash or `git filter-repo`) removes it — and on GitHub a force-push
  alone does **not** purge dangling commits, so open a GitHub Support request to
  garbage-collect the old SHAs.

## Notes

- This file holds project conventions for Claude Code and contributors.
- `references/graphrag` and `references/LightRAG` are vendored upstream copies
  for porting reference — not part of this package.
- Ask before side-effecting AWS actions (resource creation, real-AWS tests).
