# Legal / OSPO Intake Packet — aws-graphrag → awslabs

> **Purpose.** This file assembles the facts an Amazon employee (the repo owner)
> needs to file the internal open-source **outbound release** request with Legal
> and the Open Source Program Office (OSPO). The submission itself is a
> Midway-gated internal process and must be performed by the owner — this packet
> is the supporting evidence, not the submission.

## 1. What is being released

- **Repo:** `aws-graphrag` — an AWS-native Knowledge Graph RAG framework
  (Bedrock + Neptune + OpenSearch + DynamoDB), reimplementing the Microsoft
  GraphRAG methodology and the LightRAG dual-level-keyword methodology over one
  hexagonal ingestion/indexing/hybrid-search stack.
- **Target org:** `awslabs` (reusable framework/library — not a sample, not an
  AWS Solution).
- **Proposed license:** Apache-2.0 (awslabs default for code projects).

## 2. The blocking question — IP / SOW ownership

**This is the gate.** Source headers previously read *"Developed Content as
defined in the AWS Service Terms and the SOW between the parties"* — i.e. the
code was produced under a customer ProServe Statement of Work.

**Legal must confirm that AWS holds the rights to open-source this Developed
Content** before publication. Resolve this first; everything else is mechanical.

Questions to bring to Legal:
- Does the governing SOW assign ownership of this Developed Content to AWS, or to
  the customer, or jointly?
- If customer-owned or joint, is a release/relicense consent required?
- Are there any customer-confidential materials embedded? (See §4 — the scrub
  removed customer corpora and identifiers; confirm none remain.)

## 3. Dependency license profile (self-scan, pre-OSPO)

Scanned the installed environment (200 distributions; 34 direct runtime deps).
**No strong-copyleft (GPL / AGPL / SSPL) dependencies — no blocker for Apache-2.0
outbound.**

Direct runtime dependencies are all permissive:
- **MIT / MIT-style:** asyncio-throttle, datasketch, graspologic, langchain(+aws/
  community/core/text-splitters), pillow (MIT-CMU), pydantic, PyYAML, RapidFuzz,
  rich
- **BSD (2/3-clause):** bokeh, lxml, nest-asyncio, networkx, numpy, pandas, pypdf,
  python-dotenv, pyvis, scikit-learn, umap-learn
- **Apache-2.0:** aws-assume-role-lib, boto3, datasets, gremlinpython, opensearch-py,
  ragas, tenacity
- **Dual / mixed:** structlog (MIT OR Apache-2.0), llvmlite (BSD-2 AND Apache-2.0
  WITH LLVM-exception), tqdm (MPL-2.0 AND MIT)

**MPL-2.0 transitive deps** (certifi, hypothesis[dev], orjson, pathspec, tqdm):
MPL is *file-level* copyleft and is compatible with distributing an Apache-2.0
project that merely depends on them (their source is not modified or
redistributed here). Standard and widely accepted in AWS OSS.

> The OSPO will run its own authoritative license scan; this self-scan is to
> surface blockers early. Regenerate after any `uv lock --upgrade`.

## 4. Scrub status (done on `publish-prep` branch)

Verified by the scrub inventory (see `PUBLISHING.md`):
- Git history is **clean** — no AWS account IDs, ARNs, real service endpoints, or
  secrets were ever committed; `docker/config.yaml` and `cdk.context.json` (which
  hold real dev-stack identifiers) are git-ignored and untracked.
- 218 SOW source headers → Apache-2.0 SPDX; LICENSE MIT-0 → Apache-2.0;
  customer-specific docs (Example Customer / Example Site corpus plan) deleted;
  internal hostnames/codenames/CI removed. See `PUBLISHING.md` for the full list.

**Still to verify manually before publish (cannot be automated here):**
- [ ] Visually inspect tracked images for embedded internal data:
      `assets/ingestion_pipeline.png`, `assets/retrieval_pipeline.png`,
      `assets/interactive_graph.jpg` (screenshots can leak account IDs/endpoints
      that text scans miss).
- [ ] Local working-tree cleanup on the publishing machine: delete `logs/*.txt`,
      remove `docker/config.yaml` and `cdk.context.json` (all gitignored).

## 5. Process checklist (owner-performed, Midway-gated)

1. [ ] Resolve §2 IP/SOW ownership with **Legal**.
2. [ ] File the **OSPO outbound open-source release** request on the internal
       Open Source portal; target org `awslabs`; attach this packet.
3. [ ] Complete **security review** + **management/leadership approval** per the
       OSPO workflow.
4. [ ] OSPO authoritative **license scan** passes.
5. [ ] **awslabs org admins** create the public repo and grant maintainer access.
6. [ ] Publish from a **fresh squashed commit** of `publish-prep` (post-Phase-D
       local cleanup), then verify CI green and re-grep for internal references.

> Exact internal form names / SLAs are not web-documented — confirm the current
> process on the internal Open Source portal and with your manager / awslabs
> admins. Do not infer the steps from external sources.
