# Publishing to awslabs — Pre-Release Checklist

This repository is being prepared for potential open-source publication to the
[awslabs](https://github.com/awslabs) GitHub org. awslabs is the right home: this
is a reusable framework/library (not a demo → not `aws-samples`; not a turnkey
one-click deployable → not AWS Solutions).

The owner-performed process below follows the **AWS ProServe GenAI Innovation
Center "Open Source Contribution Playbook"**
([w.amazon.com/bin/view/AWS/Teams/Proserve/GenAIID/GenAIIDResourcesProgram/AssetSharing/OpenSource](https://w.amazon.com/bin/view/AWS/Teams/Proserve/GenAIID/GenAIIDResourcesProgram/AssetSharing/OpenSource/)),
which is the sanctioned path for a ProServe builder to release a reusable asset to
`awslabs`. The playbook recommends a dedicated `awslabs` repo (one repo per
project) over `aws-samples`/`aws-solutions-library-samples`, and confirms
**Apache-2.0** (not MIT-0, which it says explicitly NOT to use for `aws`/`awslabs`
orgs).

This file tracks what is **DONE** in this repo (the code/scrub/governance work,
Phase B–D) vs. the **owner-performed, Midway-gated process** that must run before
anything goes public (Phase A). Nothing here authorizes publication.

## Phase A — Owner-performed launch process (Midway-gated, NOT done here)

These steps require the owner's authenticated identity (training, tickets, Launch
Manager, lawyer assignment, org/repo self-service). They cannot be automated here.
The IP review (A5) is where the SOW "Developed Content" ownership question is
formally cleared — i.e. the old "Legal blocker" is a step in this flow, not a
separate gate. The FAST/GASP case study in the playbook shows a ProServe asset
completing exactly this path (IP Release ticket V2071786068).

1. [ ] **Choose license + name.** License = Apache-2.0 (done). The name
       `unified-kg-rag-on-aws` is a creative name → a **Trademark Legal Risk Review** is
       likely required (most take 1–2 days; only registration takes ~7 months).
2. [ ] **Prerequisite training:** GitHub Training + AWS Launch Manager Training
       (~20–30 min) on atoz.amazon.work.
3. [ ] **Link GitHub account** to your Amazon alias via Open Sourcerer Connect
       Account (`console.harmony.a2z.com/open-sourcerer/connect-account`).
4. [ ] **Create the Open Source Release ticket**
       (`t.corp.amazon.com/create/templates/0dc2e94d-5225-4f08-b512-a2cd5b0fdd77`).
5. [ ] **AWS Launch Manager (ALM) launch** (`regions.aws.dev/alm-product/products`):
       create an Open Source launch (Region IAD/us-east-1), then resolve tasks:
       - [ ] **IP Review** (mandatory) — ticket template
         `t.corp.amazon.com/create/templates/533581d8-4a83-40d5-bae0-ac2fbed41102`;
         assign to your IP lawyer via Pathfinder (`lawyer-update.corp.amazon.com`).
         **This clears the SOW Developed-Content ownership question.**
       - [ ] **Trademark Legal Risk Review** (mandatory).
       - [ ] **AppSec / PCSR security review** (mandatory) — initiate via the River
         workflow; a Public Content Security Review (PCSR) is sufficient for an
         `awslabs` source release (binary artifacts would also need an AppSec
         Open Source review — N/A here, no binaries).
       - [ ] **Dependency Review** + **Open Source Blog** (required before public).
       - [ ] **awslabs Repository Approval** ticket
         `t.corp.amazon.com/create/templates/27808ec8-a9ca-4e51-a2f2-b1dd2dbd7e82`
         (needs links to the IP Review, Trademark Review, and ALM launch page).
6. [ ] **Join the `awslabs` org** via Open Sourcerer Self-Invite
       (`console.harmony.a2z.com/open-sourcerer/self-invite`).
7. [ ] **Create the repo** via Open Sourcerer Create Repo
       (`console.harmony.a2z.com/open-sourcerer/create-repo`) — never create it
       manually or on a personal account. License: Apache. Then push the scrubbed
       tree (Phase D: fresh squashed commit).

> POCs/refs: Open Source guidelines `w.amazon.com/bin/view/Open_Source/Open_Sourcing`;
> licensing `w.amazon.com/bin/view/Open_Source/LicensingForGitHubProjects` (note:
> Amazon OSS copyright notices should be **dateless** — our headers already are).

## Phase B — Code scrub (DONE on publish-prep branch)

- [x] Replaced all **218 `*.py` SOW headers** with the standard SPDX header:
      `# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.` /
      `# SPDX-License-Identifier: Apache-2.0`.
- [x] `LICENSE` switched from **MIT-0 → Apache-2.0** (awslabs default; matches headers).
- [x] Deleted internal CI: `.gitlab-ci.yml`, `.gitlab/sast-ruleset.toml`.
- [x] Deleted customer-specific docs: an internal E2E plan (customer ITB
      corpus) and a security-triage doc (internal GitLab triage process).
- [x] `pyproject.toml` Repository URL → `https://github.com/awslabs/unified-kg-rag-on-aws`.
- [x] Genericized internal codename comments (`NaviWikiGraph`/`AnchorNetwork`) in
      `iac/`.
- [x] Scrubbed GitLab/`code.aws.dev` references from `CLAUDE.md`, `CONTRIBUTING.md`,
      `docs/design.md`, `README.ko.md`, `iac/README.md`, and a test comment.
- [x] Verified: **no AWS account IDs, ARNs, real endpoints, or secrets in tracked
      files**; git history is clean (account IDs / endpoints / `docker/config.yaml`
      / `cdk.context.json` were never committed).

## Phase C — Governance files (DONE on publish-prep branch)

- [x] `NOTICE` (Apache-2.0 §4(d) attribution).
- [x] `CODE_OF_CONDUCT.md` (Amazon Open Source Code of Conduct).
- [x] `SECURITY.md` (AWS vulnerability-reporting page; no public issues).
- [x] `THIRD_PARTY_LICENSES` (MIT attribution / provenance for the ported
      microsoft/graphrag + HKUDS/LightRAG methodology; the `references/` copies
      are git-ignored and not redistributed).
- [x] `.github/workflows/quality.yml` + `security.yml` (ported the GitLab quality
      gate + ASH scan to GitHub Actions).

## Phase D — Remaining before publish

- [x] **`references/` vendored trees — no action needed.** They are git-ignored
      (`.gitignore` line `references/`) and never tracked, so they are
      automatically excluded from the published repo (no license-scan surface).
      `THIRD_PARTY_LICENSES` attributes the ported methodology as courtesy.
- [x] **README badges added** (License/CI/coverage; URLs point at the planned
      `awslabs/unified-kg-rag-on-aws` repo — verify the slug when the repo is created).
- [x] **`README.ko.md` decision:** kept (multilingual support is a headline
      feature; the Korean README reinforces it). Revisit if awslabs prefers
      English-only.
- [ ] **Local working-tree cleanup** (untracked, never commit): delete `logs/*.txt`,
      sanitize/remove `docker/config.yaml` (real dev endpoints + account id) and
      `cdk.context.json` (account id in AZ-lookup cache). All are gitignored — do
      this on the machine that performs the publish.
- [ ] **Publish from a fresh squashed commit** (safest internal→public path) once
      Legal clears it, even though history scanned clean.
- [ ] Final OSPO/automated license scan must pass. (Self-scan, 2026-06: 200
      distributions / 34 direct runtime deps — **zero strong-copyleft GPL/AGPL/
      SSPL**, so no Apache-2.0 outbound blocker; direct deps are MIT/BSD/Apache,
      with 5 transitive MPL-2.0 deps that are dependency-only and compatible.
      Re-scan after any `uv lock --upgrade`.)

## Notes

- No CLA/DCO infrastructure is required by current AWS OSS templates — only the
  "we will ask you to confirm the licensing of your contribution" clause, already
  in `CONTRIBUTING.md`.
- `docker/config.yaml` and `cdk.context.json` are `.gitignore`d and were never
  committed; keep them out of the public tree.
