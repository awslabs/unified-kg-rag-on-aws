# Project Guide for Claude Code

AWS-native Knowledge Graph RAG framework. Reimplements Microsoft GraphRAG (and,
from M3, LightRAG methodology) on Bedrock + Neptune + OpenSearch, with DynamoDB
for incremental-indexing state.

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
  resolve to `aws_graphrag.application.cli.*`), `ingestion/` (pipeline + stages),
  `storage/indexing_manager`, `retrieval/rag_chain`.
- **`shared/`**: cross-cutting kernel — config, logging, exceptions, metrics,
  cache/pipeline managers, `utils/`.
- **Facades**: `retrieval/`, `storage/`, `ingestion/`, `evaluation/`,
  `visualization/` remain as thin `__init__` re-export shims so public import
  paths stay stable across the layer split.
- **Registries over hardcoded dispatch**: search strategies register via
  `@register_strategy` (`domain/retrieval/strategy_registry.py`). Follow this
  pattern — and the declarative `ParserFactory._loader_configs` /
  `EvaluationManager.EVALUATOR_MAPPING` — instead of `if/elif` dispatch.

See `docs/tech-doc.md` §2 for the full layer map and dependency rule.

### Adding things
- **New search strategy**: subclass `BaseSearchStrategy`, decorate with
  `@register_strategy(SearchStrategy.X, required_roles=(...))`, export from
  `adapters/search_strategies/__init__.py`. No edits to `rag_chain` needed.
- **New storage/LLM backend**: implement the relevant port; register it in the
  corresponding registry. Do not hardcode it into a manager's `__init__`.
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
  `BaseGraphRAGEvaluator` + an `EVALUATOR_MAPPING` entry.
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
  `aws_graphrag/domain/prompts/*.py` for version control, overridable via
  `custom_prompts` config.
- **Exceptions**: specific custom types from `shared/exceptions.py`; fail fast at
  boundaries; degrade gracefully only where recovery is meaningful.

## Testing

Layout: `tests/{unit,integration,property,fixtures/fakes}/`. Markers: `unit`,
`integration`, `property`, `aws` (real AWS — skipped in CI), `slow`.

- Tests run **AWS-free by default**. Use the port-based in-memory fakes in
  `tests/fixtures/fakes/` (e.g. `FakeDocStatusStore`) instead of mocking boto3
  ad hoc; use `moto` when an adapter must be exercised against a boto3 surface.
- `pytest-asyncio` is in `asyncio_mode = "auto"` — `async def test_*` just works.
- Property tests (`hypothesis`) cover invariants: hashing determinism, diff
  partition completeness, merge laws, fusion monotonicity.
- Coverage gate ratchets up per milestone toward 80% (currently `--cov-fail-under=63`
  in CI; 63% actual). Run: `uv run pytest -m "not aws" --cov=aws_graphrag`.

## Quality gate

CI (`.github/workflows/quality.yml`) and `.pre-commit-config.yaml` run ruff,
black, isort, mypy, and pytest+coverage. Install hooks with `pre-commit install`.

## Notes

- This file holds project conventions for Claude Code and contributors.
- `references/graphrag` and `references/LightRAG` are vendored upstream copies
  for porting reference — not part of this package.
- Ask before side-effecting AWS actions (resource creation, real-AWS tests).
