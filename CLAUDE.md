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

- **Root facade (`unified_kg_rag/__init__.py`)**: the public API. It lazily
  re-exports the names users and extenders import (`from unified_kg_rag import
  GraphRAGChain, Providers, ...`); everything else is internal. Adding or
  removing a name means editing `tests/unit/test_public_api.py` (and a
  CHANGELOG "Breaking" entry for a removal or rename).
- **`domain/`**: technology-agnostic core — `models/` (Pydantic), pure
  algorithms (`ingestion/` merge/delta/resolve/analyze, `retrieval/`
  strategy registry, index prefixes, hybrid normalization), and `prompts/` templates. No boto3/LangChain/
  backend imports, direct or transitive (`tests/unit/test_domain_purity.py`
  imports every domain module in a clean interpreter and checks
  `sys.modules`).
- **`ports/`**: the abstract interfaces the domain depends on — `DocStatusPort`,
  `ModelFactoryPort`/`TokenCounterPort` (`Protocol`s) and the
  write-side indexer ABCs `BaseIndexer`/`GraphIndexer`/`VectorIndexer`
  (+ `IndexingStats`). `ports/__init__` is the port catalog. The
  retrieval/evaluation abstract bases are *adapter bases* (they construct infra
  in `__init__`): `ports/__init__` names them but does not export them; import
  them from `adapters.retrieval.base` / `adapters.evaluators.base`.
- **`adapters/`**: concrete technology bindings. Rule: vendor packages hold
  clients/engines/model factories/caches, role packages hold port impls
  (a role package may import a vendor one, not the reverse). Vendor: `aws/`
  (Bedrock, Neptune/OpenSearch clients, S3), `local/` (Ollama/OpenAI-compatible,
  graph store `sqlite_graph.py`, vector engine `search_engine/`). Role:
  `storage/` (indexers, incl. `sqlite_graph_indexer.py`), `retrievers/`,
  `doc_status/` (DynamoDB/SQLite `DocStatusPort` impls), `guardrail/`
  (Bedrock `ContentGuardPort` impl). Also
  `llm/` (provider-neutral chain assembly, retry, token estimate),
  `search_strategies/`, `retrieval/` (base + hybrid scorer + token/memory
  managers), `ingestion/` (LLM/IO-coupled chunker/extractor/loader/parser/
  translator), `renderers/`, `evaluators/` (incl. the evaluator bases).
- **`application/`**: orchestration + entry points — `cli/` (pyproject scripts
  resolve to `unified_kg_rag.application.cli.*`), `ingestion/` (pipeline + stages),
  `storage/indexing_manager`, `retrieval/rag_chain`.
- **`shared/`**: cross-cutting kernel — config, logging, exceptions, metrics,
  cache/pipeline managers, `utils/`. Keep `shared/__init__` and
  `shared/utils/__init__` dependency-light: import LangChain-coupled helpers
  from `shared.utils.langchain` / `shared.utils.document_converter` (and rich
  helpers from `shared.utils.display`), not from the package root.
- **`evaluation/` and `visualization/`**: real logic packages (evaluators +
  manager; render loop + `embeddings/`/`exporters/`). Code inside the
  package imports everything else from its real location (`application.retrieval.rag_chain`,
  `application.storage.indexing_manager`, `application.ingestion.pipeline`,
  `adapters.*`, `domain.*`).
- **Registries over hardcoded dispatch**: search strategies register via
  `@register_strategy` (`domain/retrieval/strategy_registry.py`). Follow this
  pattern — and `ParserFactory.register_loader` / `@register_renderer` —
  instead of `if/elif` dispatch. `EvaluationManager._resolve_evaluator_class`
  is the one explicit type → class map (a new evaluator adds a branch there).

See `docs/design.md` §2 for the full layer map and dependency rule.

### Adding things
Recipes for new strategies, backends, evaluators, renderers, parsers and
config sections live in one place: `docs/design.md` §15 "Extension Guide"
(read it before extending; update it, not this file, when a recipe changes).
The rules to keep in mind:
- Extend through the registries and the one evaluator type map, never new
  `if/elif` dispatch.
- Inject backends through constructors (`Providers`, `vector_indexer=`/
  `graph_indexer=`, `retriever_builders=`, `doc_status=`); never hardcode a
  backend into a manager's `__init__`.
- A config field change means a user-facing `Field(description=...)` and
  `uv run python scripts/gen_config_reference.py` (regenerates
  `docs/configuration.md`; `tests/unit/test_config_reference.py` checks it).

## Capabilities & CLIs

- **Two retrieval methodologies** (user-selectable via `RAGInput.search_strategy`):
  GraphRAG community-summary (`auto`/`drift`/`global`/`local`/`simple`) and
  LightRAG dual-level keyword (`mix`/`hybrid`/`naive`). Both share the same
  ingestion, indexing, caching, multilingual, and hybrid-scoring infrastructure;
  only the retrieval algorithm differs. LightRAG `mix`/`hybrid` extract high/low
  keywords (`KeywordsExtractionPrompt`) and query a relationship vector index
  (high-level) + entity index (low-level) + Neptune expansion.
- **Incremental indexing**: enable the doc-status registry
  (`providers.doc_status.enabled`; DynamoDB, or SQLite in local mode) to diff a corpus by content hash and only
  (re)index new/changed documents, merging into the live graph
  (`IncrementalIndexer`, `ingestion/merge/`, idempotent `upsert_*`/`delete_by_id`
  on both indexers). The registry lists each document's text units; a
  deletion asks the stores what cites them and removes only what no surviving
  text unit still states.
- **CLIs** (`pyproject` scripts): `run-ingestion`, `run-rag`, `run-eval`,
  `run-visualization` (render from exported graph data, no ingestion),
  `run-prompt-tuning` (profile a corpus → domain-adapted `custom_prompts` YAML),
  `prepare-benchmark` (MuSiQue → corpus + `run-eval` dataset, downloaded at
  run time, never committed). `run-eval --search-strategy a,b --repeats N`
  runs a paired comparison (`evaluation/comparison.py`, optional
  `--pairwise-judge`). `docs/cli.md` lists every flag; it is generated from
  the parsers by `scripts/gen_cli_reference.py`, so change a flag's `help=`
  and regenerate (a unit test fails when it is stale).
- **Evaluation**: `langchain` + `ragas` (LLM judges) plus three LLM-free
  evaluators: `graph_aware` (entity/relationship coverage = recall, from
  ground-truth `expected_entities`/`expected_relationships`; precision/F1
  intentionally not emitted — see the evaluator docstring), `retrieval`
  (hit@k/recall@k/MRR against `reference_sources`) and `answer_match` (exact
  match/token F1 against `answer` + `metadata.answer_aliases`).
- **Visualization**: renderers register via `@register_renderer`
  (`adapters/renderers/`); the manager and the standalone CLI drive them
  through one registry + `RenderContext`.

## Code standards

- **Types**: modern built-ins (`list`, `dict`, `X | None`); no legacy `typing.List`.
- **Models**: Pydantic at all boundaries (not dataclasses, except frozen config records).
- **Files**: `pathlib`, not `os.path`.
- **Packages**: `uv` (`uv sync`, which includes the dev group), not pip.
- **Logging**: `%`-formatting, not f-strings — `logger.info("did %s", x)`.
- **LLM calls**: LangChain LCEL (`prompt | llm | parser`); prompts live in
  `unified_kg_rag/domain/prompts/*.py` for version control, overridable via
  `custom_prompts` config.
- **Exceptions**: specific custom types from `shared/exceptions.py`; fail fast at
  boundaries; degrade gracefully only where recovery is meaningful.

## No customer data (MANDATORY)

This is a public repository. Customer-identifying or
customer-confidential content MUST NEVER appear in application code, tests,
evaluation datasets, fixtures, prompts, comments, docs, diagrams, or commit
messages. This includes: customer/project names and codenames (e.g. internal
engagement names), real contract figures/clauses, document filenames from a
customer corpus, account ids, endpoints, and internal hostnames.

- **Tests/fixtures/eval**: use neutral *synthetic* data (generic entities like
  "Vendor"/"Buyer", made-up amounts). Never paste real customer text.
- **Real-corpus tests**: keep non-public corpora out of the repo entirely; the
  gitignored `.test-corpus-backup/` is for local copies only.
- **Before committing**: grep the staged diff for customer terms.

## Testing

Layout: `tests/{unit,integration,property,fixtures/fakes}/`. Markers: `unit`,
`integration`, `property`, `aws` (real AWS — skipped in CI), `slow`.

- Tests run **AWS-free by default**. Use the port-based in-memory fakes in
  `tests/fixtures/fakes/` (e.g. `FakeDocStatusStore`) instead of mocking boto3
  ad hoc; use `moto` when an adapter must be exercised against a boto3 surface.
  `tests/unit/test_fake_port_signatures.py` fails if a fake's method
  signatures drift from the port's.
- Each test times out after 120 s (`pytest-timeout`; override with
  `@pytest.mark.timeout(n)`).
- `tests/integration/test_local_stores.py` needs `LOCAL_STORES=1` and the
  `docker/compose.local.yaml` stores; the CI `local-stores` job runs it.
- `tests/integration/test_embedded_e2e.py` runs `backend: local` end to end on
  the embedded SQLite stores (no containers, no AWS) in the default jobs.
- `pytest-asyncio` is in `asyncio_mode = "auto"` — `async def test_*` just works.
- Property tests (`hypothesis`) cover invariants: merge laws, incremental
  rounds and interrupted runs converging with a full build, fusion
  monotonicity; the `DocStatusPort.diff` partition runs against every registry
  in `tests/integration/test_doc_status_conformance.py`.
- Coverage gate ratchets up with measured coverage (the `--cov-fail-under` floor in
  `quality.yml` sits just under it: currently 95, measured ~96%). Run:
  `uv run --extra otel pytest -m "not aws" --cov=unified_kg_rag`.

## Quality gate

CI (`.github/workflows/quality.yml`) and `.pre-commit-config.yaml` run ruff,
black, isort, mypy, and pytest+coverage. Install hooks with `pre-commit install`.

## Git & collaboration workflow

Treat `main` as protected shared history. These rules apply to human
contributors **and** to Claude Code working in this repo.

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
- **No force-push to `main`, no history rewrites**: they break every clone,
  fork, and open PR.
- **Never merge your own PR without review** when a second maintainer is
  available; for genuinely solo changes, still open the PR, let CI pass, and use
  a squash-merge so `main` keeps one clean commit per change.
- **Ask before side-effecting actions** — pushing, opening/merging PRs, tagging
  releases, and any AWS resource creation or real-AWS tests. Approval for one
  action is not standing approval for the next.
- **Customer-data check is a hard blocker.** Before every commit, grep the
  staged diff for customer terms (see *No customer data* above); never commit or
  push a change that contains any.

## Notes

- This file holds project conventions for Claude Code and contributors.
- `references/` is gitignored: a place for local copies of the upstream
  GraphRAG and LightRAG sources used as porting reference, not part of this
  package.
- Ask before side-effecting AWS actions (resource creation, real-AWS tests).
