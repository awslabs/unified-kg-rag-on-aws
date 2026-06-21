# Parsebox Contracts — End-to-End Test Plan

A plan to validate `aws-graphrag` end to end on real contract documents from the
docparser corpus (Example Customer EPC/procurement contracts & ITB schedules).
This is a **manual, cost-incurring, real-AWS** exercise — it provisions Neptune/
OpenSearch and calls Bedrock — so it runs only with explicit approval, never in
CI. It complements (does not replace) the AWS-free integration suite.

## 0. Why this corpus

- Source: `/data/projects/example-customer/docparser/data` (2256 PDFs, 414 DOCX).
- Contract-shaped, entity-rich documents (parties, scope, schedules, equipment,
  dates, monetary terms) — exactly the entity/relationship/claim structure the
  graph targets, and a good stress test for the recent fixes (claim literal
  objects, relationship endpoint recovery, community reports).
- Mixed EN/KO content exercises the translation + multilingual path.

## 1. Sample selection (keep it small & cheap first)

Start with a **tiny, curated subset** (5–10 docs) before any full run:

| Tier | Files | Purpose |
|---|---|---|
| Smoke (1 doc) | `…/Example Site/01_ITB/ExampleAgreement.pdf` (11 KB) | fastest ingest→search round trip |
| Small (≈5 docs) | a few ITB `Schedule *` PDFs from one project (Example Site) | cohesive graph; cross-doc shared entities |
| Mixed-lang (+2) | one KO-heavy spec (`함안_5 VOL2…기술규격서`) + one EN | translation + multilingual keywords |

Copy the chosen files into a scratch corpus dir (do **not** point ingestion at
the whole 2256-PDF tree on a first run):

```bash
mkdir -p /tmp/pbx-e2e/docs
cp "<selected pdfs>" /tmp/pbx-e2e/docs/
```

## 2. Prerequisites (provision once, with approval)

- Bedrock model access in the configured region (embeddings + chat + rerank).
- A Neptune cluster endpoint (IAM auth) and an OpenSearch domain endpoint.
- A `config.yaml` from `config-template.yaml` with those endpoints filled in.
  - Set `processing.translation` target language as desired (EN).
  - Leave `indexing.cross_run_merge: false` for run 1 (test incremental later).
  - Optionally set a Bedrock `guardrail.identifier`.
- Cost guardrails: small corpus, `chunking` defaults, watch Bedrock token usage
  (a `--metrics-sink cloudwatch` run surfaces PipelineMetrics).

## 3. Execution steps

### 3.1 Full ingest (run 1)
```bash
run-ingestion --source-directory /tmp/pbx-e2e/docs --config-path config.yaml \
  --metrics-sink cloudwatch
```
Capture: stage timings, entity/relationship/claim counts, and the new stats —
`relationships_filtered_by_confidence`, claim resolution `reduction_rate`,
"Materialized N entities referenced only by relationships",
community/community-report counts.

### 3.2 Retrieval (both methodologies)
```bash
# GraphRAG
run-rag --query "Who are the parties to the land lease agreement and what is the term?" \
  --search-strategy local  --config-path config.yaml
run-rag --query "Summarize the obligations across the ITB schedules." \
  --search-strategy global --config-path config.yaml
# LightRAG
run-rag --query "lease term, rent, termination" \
  --search-strategy mix --config-path config.yaml
```

### 3.3 Incremental (run 2)
- Add 1 new doc, modify 1, delete 1 from `/tmp/pbx-e2e/docs`.
- Re-run with `aws.dynamodb.enabled: true`:
```bash
run-ingestion --source-directory /tmp/pbx-e2e/docs --config-path config.yaml
```

### 3.4 Visualization
```bash
run-visualization --data-path <exported_graph.json> --output-dir /tmp/pbx-e2e/viz
```

## 4. Assertions / success criteria

Functional (the recent fixes specifically):
- [ ] **Claims not over-dropped**: claims with literal objects (dates, amounts,
      "TRUE/FALSE" status) survive resolution with `object_id=None`; subject is
      a real entity. Spot-check `claim resolution` log + indexed claims.
- [ ] **Relationships not over-dropped**: relationship count is materially higher
      than entity-listed-pairs; `_materialize_relationship_endpoints` log shows
      stubs created; no "Entity not found … dropping" pattern for valid edges.
- [ ] **Community reports** generated and retrievable (global search returns
      report-grounded answers).
- [ ] Retrieval returns grounded, citation-bearing answers for the queries above
      (parties, term, obligations) under both methodologies.

Incremental (run 2):
- [ ] Deleted doc's *exclusive* entities/relationships/claims/community-reports
      removed from Neptune + OpenSearch + DynamoDB registry; shared ones kept.
- [ ] Changed doc's stale artifacts pruned then re-upserted; unchanged docs
      skipped (registry diff), not re-embedded (embedding cache).
- [ ] With `cross_run_merge: true`, a shared entity's description/`text_unit_ids`
      accumulate across runs (read-merge-write) rather than overwrite.

Operational:
- [ ] No unhandled exceptions; failed-doc count acceptable; CloudWatch EMF
      metrics emitted; Guardrail (if enabled) does not block legitimate content.

## 5. Automation hook

The opt-in `tests/integration/test_real_aws_e2e.py` (`-m aws`) already scaffolds
connectivity + an ingest→search round trip gated by `GRAPHRAG_TEST_CONFIG` and
`GRAPHRAG_TEST_RUN_INGEST`. To automate this plan, point it at the docparser
scratch corpus:

```bash
GRAPHRAG_TEST_CONFIG=./config.yaml GRAPHRAG_TEST_RUN_INGEST=1 \
  uv run pytest -m aws -v
```
A follow-up can add a docparser-specific `-m aws` test that asserts the section-4
claim/relationship-survival criteria against the small sample.

## 6. Teardown

- `run-ingestion … --force-rebuild` clears indices, or drop the test indices/
  Neptune labels and the DynamoDB doc-status table.
- Remove `/tmp/pbx-e2e`.
- Decommission Neptune/OpenSearch if provisioned only for this test.

> Approval gate: provisioning AWS resources and running ingestion incur cost and
> create resources — get explicit sign-off before steps 2–3 (per the project's
> "ask before side-effecting AWS actions" rule).
