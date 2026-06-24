# Publishing to awslabs — Pre-Release Checklist

This repository is being prepared for potential open-source publication to the
[awslabs](https://github.com/awslabs) GitHub org. awslabs is the right home: this
is a reusable framework/library (not a demo → not `aws-samples`; not a turnkey
one-click deployable → not AWS Solutions).

This file tracks what is **DONE** on the `publish-prep` branch vs. what **REMAINS**
(notably the Legal/IP blocker, which gates everything).

> The work below was done on a branch and is reversible. Nothing here authorizes
> publication — that requires the Legal clearance and OSPO approval in Phase A.

## Phase A — Legal / IP clearance (BLOCKING, internal, NOT done here)

- [ ] **SOW clearance.** Source headers previously read *"Developed Content as
      defined in the AWS Service Terms and the SOW between the parties"* — i.e.
      this was produced under a customer ProServe SOW. **Legal must confirm AWS
      holds the rights to open-source it before publishing.** This gates everything.
- [ ] File the **OSPO outbound open-source release request** (internal Open Source
      portal; Midway-gated). Target org: `awslabs`.
- [ ] Obtain **security review** + **management/leadership approval** per the OSPO
      workflow.
- [ ] Confirm **awslabs org admins** will create the repo and grant maintainer access.

> The exact internal forms/SLAs are not web-documented — check the internal Open
> Source portal and confirm the process with your manager / the awslabs admins.

## Phase B — Code scrub (DONE on publish-prep branch)

- [x] Replaced all **218 `*.py` SOW headers** with the standard SPDX header:
      `# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.` /
      `# SPDX-License-Identifier: Apache-2.0`.
- [x] `LICENSE` switched from **MIT-0 → Apache-2.0** (awslabs default; matches headers).
- [x] Deleted internal CI: `.gitlab-ci.yml`, `.gitlab/sast-ruleset.toml`.
- [x] Deleted customer-specific docs: `docs/docparser-e2e-plan.md` (Example Customer
      Example Customer / Example Site corpus), `docs/security-triage.md` (internal GitLab
      triage process).
- [x] `pyproject.toml` Repository URL → `https://github.com/awslabs/aws-graphrag`.
- [x] Genericized internal codename comments (`NaviWikiGraph`/`AnchorNetwork`) in
      `iac/`.
- [x] Scrubbed GitLab/`code.aws.dev` references from `CLAUDE.md`, `CONTRIBUTING.md`,
      `docs/tech-doc.md`, `README.ko.md`, `iac/README.md`, and a test comment.
- [x] Verified: **no AWS account IDs, ARNs, real endpoints, or secrets in tracked
      files**; git history is clean (account IDs / endpoints / `docker/config.yaml`
      / `cdk.context.json` were never committed).

## Phase C — Governance files (DONE on publish-prep branch)

- [x] `NOTICE` (Apache-2.0 §4(d) attribution).
- [x] `CODE_OF_CONDUCT.md` (Amazon Open Source Code of Conduct).
- [x] `SECURITY.md` (AWS vulnerability-reporting page; no public issues).
- [x] `THIRD_PARTY_LICENSES` (MIT attributions for vendored `references/graphrag`,
      `references/LightRAG`).
- [x] `.github/workflows/quality.yml` + `security.yml` (ported the GitLab quality
      gate + ASH scan to GitHub Actions).

## Phase D — Remaining before publish

- [x] **`references/` vendored trees — no action needed.** They are git-ignored
      (`.gitignore` line `references/`) and never tracked, so they are
      automatically excluded from the published repo (no license-scan surface).
      `THIRD_PARTY_LICENSES` attributes the ported methodology as courtesy.
- [x] **README badges added** (License/CI/coverage; URLs point at the planned
      `awslabs/aws-graphrag` repo — verify the slug when the repo is created).
- [x] **`README.ko.md` decision:** kept (multilingual support is a headline
      feature; the Korean README reinforces it). Revisit if awslabs prefers
      English-only.
- [ ] **Local working-tree cleanup** (untracked, never commit): delete `logs/*.txt`,
      sanitize/remove `docker/config.yaml` (real dev endpoints + account id) and
      `cdk.context.json` (account id in AZ-lookup cache). All are gitignored — do
      this on the machine that performs the publish.
- [ ] **Publish from a fresh squashed commit** (safest internal→public path) once
      Legal clears it, even though history scanned clean.
- [ ] Final OSPO/automated license scan must pass.

## Notes

- No CLA/DCO infrastructure is required by current AWS OSS templates — only the
  "we will ask you to confirm the licensing of your contribution" clause, already
  in `CONTRIBUTING.md`.
- `docker/config.yaml` and `cdk.context.json` are `.gitignore`d and were never
  committed; keep them out of the public tree.
