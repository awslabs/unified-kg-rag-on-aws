# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment

For this project, also include the Python version, the AWS region, the CLI command you ran, the relevant part of your
`config.yaml` with endpoints, account ids and credentials removed, and the error or log lines. Log records at `INFO`
and above do not contain query or corpus text; `DEBUG` logs do, so review them before attaching. The issue templates
in [`.github/ISSUE_TEMPLATE/`](./.github/ISSUE_TEMPLATE/) ask for these fields.


## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Development workflow

### Set up

You need Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/). AWS credentials are needed only for real-AWS runs; the
test suite runs without AWS.

```bash
git clone https://github.com/<your-username>/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
# The package, the dev group and the extras CI's quality job installs (.github/workflows/quality.yml).
uv sync --extra otel --extra eval --extra viz --extra ontology --extra mcp --extra docx --extra console
uv run pre-commit install   # runs ruff, isort, black and mypy on every commit
```

Without uv: create a virtual environment, activate it, and run
`pip install -e '.[otel,eval,viz,ontology,mcp,docx,console]' --group dev` (pip 25.1 or later).

### Branches and pull requests

- Never commit to `main`. Work on a topic branch named after the change type, for example `feat/…`, `fix/…`,
  `docs/…` or `chore/…`, and land it through a pull request.
- Keep one logical change per pull request and per commit. Do not mix a refactor, a feature and a documentation fix.
- Fill in the [pull request template](./.github/PULL_REQUEST_TEMPLATE.md).
- Pull requests are squash-merged after CI passes and a maintainer approves, so `main` keeps one commit per change.
- Add an entry under `## [Unreleased]` in [CHANGELOG.md](./CHANGELOG.md) for any user-visible change. Release history
  belongs in the changelog only, not in `docs/` ([Documentation](#documentation)).
- A **Breaking changes** entry covers what users depend on: the names exported from `unified_kg_rag` (pinned in
  `tests/unit/test_public_api.py`; changing that list is a deliberate API change), configuration keys, CLI flags and
  exit codes, index and registry formats, and the CDK stack's inputs and outputs. Moving or renaming an internal module
  is not a breaking change and needs no entry; neither does a change to code only reached through internal imports.
- Entries compare with the last release. A fix to something not yet released amends that change's entry instead of
  adding a *Fixed* one, which lists only bugs a release had. Add a user's action as the next number of *Upgrading*
  (or to its *From a `main` checkout* list when only unreleased formats or keys need it), and link the docs page
  rather than repeating it.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary` in the imperative mood, with a
subject of 72 characters or fewer. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `style`, `perf`, `ci`,
`build`. The body explains why the change is needed; the diff already shows what changed.

```text
fix(incremental): retry documents whose extraction failed
```

### Moving code and `git blame`

Move or reformat code in a pull request of its own that changes nothing else, and add its commit on `main` to
[`.git-blame-ignore-revs`](./.git-blame-ignore-revs). GitHub's blame view skips the commits listed there; locally,
run `git blame --ignore-revs-file .git-blame-ignore-revs <file>`, or set it once with
`git config blame.ignoreRevsFile .git-blame-ignore-revs`. To follow lines that moved into another file (for example
from the former `config.py` into `unified_kg_rag/domain/models/config/`), add `-C -C`:
`git blame -C -C <file>`.

### Checks to run before pushing

CI ([`.github/workflows/quality.yml`](./.github/workflows/quality.yml)) runs the same checks, so run them locally first:

```bash
uv run pre-commit run --all-files               # ruff, isort, black, mypy and file hygiene
uv run pytest -n logical --cov=unified_kg_rag   # every suite with coverage, as CI's quality job runs it
```

Both run in the environment `uv sync` set up above (`uv run` keeps its extras). `-n logical` runs the suite on every
logical CPU through pytest-xdist, as CI does (`-n auto` counts physical cores only); coverage still combines across
the workers. CI measures coverage with `COVERAGE_CORE=sysmon` (Python
3.12's low-overhead tracer; the suite measures lines, not branches), which reports the same lines as the default
tracer.

CI fails below the coverage threshold set by `--cov-fail-under` in `quality.yml`. It also runs the suite on Python
3.11 (and, as a non-blocking early warning, on 3.13 with the `requires-python` cap lifted), checks the optional
parsers, checks a core install without the `eval` and `viz` extras or the dev group (only the `test` group, the
test runner; tests that need an extra skip there; mark a new one with `tests/fixtures/extras.py`), runs the local-store smoke test, builds and runs the container image, and synthesizes `iac/` with cdk-nag. The
`coverage-combined` job adds the local-store job's coverage (the Neptune and OpenSearch adapters against real stores)
to the quality job's and writes the total to its job summary; it reports only, the floor applies to the quality job's
own report. A weekly [`hypothesis-explore`](./.github/workflows/hypothesis-explore.yml) run draws the property tests
from random seeds with a larger budget (`HYPOTHESIS_PROFILE=explore`); pull requests stay derandomized (`ci`). A weekly
[`fidelity`](./.github/workflows/fidelity.yml) run installs the original HippoRAG 2, RAPTOR and BenchmarkQED at pinned
commits and versions (outside the lock) and runs the parity tests that skip without them, failing if any still skips.
`iac/` is a separate uv project; if you change it, also run `cd iac && uv sync && uv run cdk synth -c enable_cdk_nag=true && uv run pytest` (see [`iac/README.md`](./iac/README.md)).

When a pull request changes `pyproject.toml` or `uv.lock`, the
[`licenses`](./.github/workflows/licenses.yml) workflow also syncs every extra and fails on a copyleft,
source-available or unlabeled license among the runtime dependencies. Reviewed exceptions are listed, each with its
reason, in `.github/scripts/check_licenses.py`; add one only after reviewing the license. The `console-web` job runs
the same check over the runtime npm dependencies the console's web client bundles, from the license fields of
`console/web/package-lock.json`.

The [`security`](./.github/workflows/security.yml) workflow runs the Automated Security Helper (ASH: bandit, semgrep,
detect-secrets, checkov, cdk-nag, npm-audit) and fails on an unsuppressed finding at MEDIUM or above. Its `pip-audit`
job checks every locked runtime dependency (all extras, no dev group) for known vulnerabilities and fails on one
that [`.github/pip-audit-accepted.txt`](./.github/pip-audit-accepted.txt) does not list with the reason it does not
apply and the date it is accepted until (past it, the advisory gates again: re-read the reason before extending it);
remove a line once a fixed release is locked. To run ASH locally
(the same pinned version and dependency date, as CI does; reports land in the gitignored `.ash/ash_output/`):

```bash
uvx --exclude-newer 2026-11-01T00:00:00Z git+https://github.com/awslabs/automated-security-helper.git@e73de85c1db7f443276370c4a6f7501815756780 --source-dir .
```

Fix a true positive. For a false positive, suppress it with its reason: on the statement for a single site
(`# nosec B608 - <reason>`, `# nosemgrep: <rule id> - <reason>` on the line above, `# pragma: allowlist secret`), or
in `.ash/.ash.yaml` for a whole class of files (each entry has a `reason`). ASH matches `.gitignore` against the
absolute path, so in a checkout below a directory the file ignores (for example `.claude/worktrees/`)
detect-secrets, which scans ASH's file list, finds nothing; it also skips every path under a dot-directory, so a
checkout anywhere below one (`~/.claude/...`, `~/.cache/...`) is not scanned for secrets either. Run it from a clone
or a worktree outside such directories (`git worktree add --detach /tmp/ash-scan HEAD`).

### Pins Dependabot does not move

[Dependabot](./.github/dependabot.yml) updates both uv locks, the console's npm lock (`console/web`), the SHA-pinned
workflow actions and the local store images in `docker/compose.local.yaml`. These pins it cannot read; bump them by
hand, together where they must match:

- The uv version, `version: "0.12.23"` under every `astral-sh/setup-uv` step in `.github/workflows/` (Dependabot moves
  the action's SHA, not this input), and the `ghcr.io/astral-sh/uv` image `docker/Dockerfile` copies uv from
  (`COPY --from`, tag and digest). Held at 0.12 for now (`HELD` in `scripts/check_pins.py`: the weekly report lists
  uv 0.13 as held, which opens no issue, and a newer 0.12 patch as behind; delete the entry to lift the hold): 0.13.0 has breaking changes (Python 3.15 as the default, the tar codec, constraint-file parsing) to check against
  the locks, the workflows and the image, and its image digest needs a Docker build to verify.
- The container base image, the `BASE_IMAGE` ARG default in `docker/Dockerfile` (tag and digest; Dependabot does not
  resolve an ARG in `FROM`).
- The apt pins of the runtime stage's `apt-get install` line in `docker/Dockerfile` (`libgomp1=…`, `zlib1g=…`).
  Debian publishes a security fix to `trixie-security` before a `python:3.12-slim` digest ships it, so a fixable
  finding can fail the image scan while the base digest is current. Pin the fixed version on that line (the
  `zlib1g` pin is one), with a comment naming the CVEs, and drop the pin once the base digest ships that version
  (the weekly check below says when). Bump a pin when the repository offers a newer version: the superseded one
  leaves the repository and the build fails.
- The AWS CDK CLI, `npm install -g aws-cdk@…` in the `iac` job of `quality.yml`, and that job's `node-version`.
- The ADOT collector sidecar, `COLLECTOR_VERSION` and `COLLECTOR_DIGEST` in `iac/stacks/compute_stack.py`.
- The Trivy image the `docker` jobs scan with (`aquasec/trivy`, tag and digest, in `quality.yml`).

The weekly [`pin-freshness`](./.github/workflows/pin-freshness.yml) workflow runs `scripts/check_pins.py`, which
compares the base image, uv, apt, ADOT, CDK CLI and Trivy pins with upstream (a base index whose amd64 and arm64
images are the pinned ones is current), and builds and scans the image with
the opt-in extras (a superset of the default image) as the `docker-unstructured` job does. It opens (or updates) one
issue listing the pins that are behind and the fixable findings, and another when the image does not build; a week
that finds everything current (or the image built) closes it. The image scan (Trivy, `--severity HIGH,CRITICAL --ignore-unfixed`) fails a pull request on a fixable HIGH or CRITICAL
finding in either image; bump the pin that clears it, or accept it in [`.trivyignore.yaml`](./.trivyignore.yaml) with
its reason and an expiry.

### Tests

- Tests run without AWS. Use the port-based in-memory fakes in `tests/fixtures/fakes/` (for example
  `FakeDocStatusStore`) rather than ad-hoc boto3 mocks, and `moto` when an adapter must be exercised against a boto3
  surface.
- Markers: `unit`, `integration`, `property`, `aws` (real AWS, excluded in CI), `slow` (over 5 s; CI still runs
  them), `perf` (timing assertions; deselected by default and run only by the weekly `hypothesis-explore` job) and
  `weekly` (heavy fidelity cases; deselected by default, pull requests run a sample and the weekly `fidelity` job all
  of them). `tests/conftest.py` applies `unit`, `integration` or `property` by the directory a test lives in. Each test
  times out after 120 s.
- `tests/integration/test_local_stores.py`, the `container` cases of the store conformance suites and of
  `test_embedded_e2e.py` run against the local containers in `docker/compose.local.yaml` when `LOCAL_STORES=1` is
  set (`LOCAL_STORES_GREMLIN_PORT` / `LOCAL_STORES_OPENSEARCH_PORT` point them at other ports).
- Every test file passes when run on its own: the `test-isolation` CI job runs each unit, integration and property
  test file in a process of its own (`python scripts/check_test_isolation.py tests/unit tests/integration
  tests/property`), so a file that passes only after another bound a lazy import or filled a cache fails there.
- Tests, fixtures and evaluation data use synthetic content only (generic entities such as "Vendor" and "Buyer",
  made-up amounts). Never add real corpus text, customer names, account ids, endpoints or hostnames to code, tests,
  docs or commit messages.

### Single-owner guards

A concept implemented in two places drifts, so each shared one has an owner module, and
`tests/unit/test_single_owner_guards.py` (with `test_namespace_resolution.py`) fails on a copy of it: a
`SourceGuard` names the pattern a copy takes, the owner, and an allowlist of the other places that may hold it
(`tests/fixtures/source_guards.py`). The [Extension guide](./docs/guides/extend.md#15-extension-guide) lists the owners.

- **A failing guard names the copy and the owner to call.** Call the owner. Names a module imports are resolved, so
  an alias (`import json as j`, `from os import replace`) or a relative import (`from .degradation import ...`) is
  still a copy.
- **A place that holds the pattern but is not a copy** (a store's own codec, a dataset file read with `json.loads`)
  gets an allowlist entry: `"path::scope": (count, "reason")`, the scope being the enclosing function or class, or
  the name a module- or class-level assignment binds (`path::_TABLE`). The count is how many matches the place
  holds: one more fails as a copy, and fewer fails too, so the entry shrinks or goes when the place is cleaned up.
  The reason says what the place holds instead of the concept, and must stay true.
- **A new guard** names a finder that matches the owner's own form (a test fails when the owner holds no match, so a
  finder that cannot see the concept cannot guard it), every path it scans must exist, and its pattern is pinned by
  `test_each_pattern_catches_the_forms_copies_took` with each form a copy took.
- **The guards are tripwires for the forms they know, not coverage.** A finder matches the forms copies took and
  the forms its pins and planted files spell; a copy written another way passes it (most forms written outside
  the finders' vocabulary do). Review owns the rest: a reviewer who sees a second implementation of a guarded
  concept asks for the owner whether or not a guard fails. Do not widen a finder one idiom at a time to catch a
  form review found; fix the copy, and widen a finder only where one rule covers a whole family (the decoders of
  G2 and G10 are a denylist of decoding modules, not of the calls one takes).
- **Planted copies are tuning data.** Each file of `tests/fixtures/guard_copies` is one copy (or code near the
  concept that is none), headed by the guard, the module it stands in and what the guard must do: `copy`, `clean`,
  `missed: why` or `flagged: why`; the `tuned_` files were a held-out set until the finders were tuned on them. The
  finders were written against all of them, so they pin behaviour, not measure it: a measurement of what a guard
  misses needs forms written after it, kept out of the repository. Keep a form the finder still misses as
  `missed`: the test fails when the finder starts to see it, so the header moves to `copy`.

### Code conventions

- Modern built-in types (`list`, `dict`, `X | None`), Pydantic models at boundaries, `pathlib` for paths.
- `%`-style logging arguments (`logger.info("did %s", x)`), not f-strings.
- LLM calls as LangChain LCEL chains; prompts live in `unified_kg_rag/domain/prompts/` and can be overridden with
  `custom_prompts`.
- Raise the specific exception types from `unified_kg_rag/shared/exceptions.py`.
- Catch the specific exceptions a call raises. A broad `except Exception` (ruff `BLE`) is for isolating one
  item's failure or degrading on purpose, and says why on the line: `except Exception as e:  # noqa: BLE001 - <reason>`.
  When it logs an unexpected error at ERROR and goes on, it logs the traceback too (`exc_info=True`).
- mypy runs with `disallow_any_generics` and `no_implicit_reexport`: give generics their type arguments
  (`Callable[..., Any]`, `dict[str, Any]`), and import a name from the module that defines it, or list it in that
  package's `__all__`.
- Keep the dependency rule: `domain/` must not import boto3, LangChain or a backend client, directly or transitively
  (`tests/unit/test_domain_purity.py` checks this).

### Extending the framework

New search strategies, storage or model backends, parsers, renderers, evaluators and config sections each have a
recipe in the [Extension guide (§15)](./docs/guides/extend.md#15-extension-guide). Most need a registration rather
than an edit to dispatch code.

### Documentation

[docs/index.md](./docs/index.md) is the documentation hub: it maps every page by reader and by type. Add a new page
to its page map, and link it from the pages around it.

**Where a page goes.** The documentation follows [Diátaxis](https://diataxis.fr/), one folder per type:

| Folder or page | Type | What goes there |
|---|---|---|
| `docs/getting-started/` | Tutorial | Step-by-step first runs (locally, on AWS) that end with a working result |
| `docs/concepts/` | Explanation | How and why it works: the overview and "beyond the papers", retrieval, scopes and namespaces, and the measured evidence |
| `docs/guides/` | How-to | One page per task (configure, ingest, query, incremental indexing, evaluate, ...): the steps and the settings that matter, linking the reference for every option |
| `docs/reference/` | Reference | Complete lists to look up: CLI flags, configuration keys, models, filters, output files, exit codes, formats, telemetry |
| `docs/operations.md` | Runbook | Procedures an operator runs against a deployment, and what to do when a run, an alarm or a query fails. `iac/README.md`, code and error messages link its section headings: keep them |
| `docs/troubleshooting.md` | Reference | Frequent questions and every error message with its meaning and action |
| `docs/design.md`, `docs/design/` | Explanation, for contributors | Architecture, algorithms and design decisions. `docs/design.md` is the index: every section heading with a summary and a link to the `docs/design/` page that holds it (§15 is `docs/guides/extend.md`). A new section adds its heading to both; code cites the section numbers (`design §6.1.1`), so never renumber or rename a section |
| [`iac/README.md`](./iac/README.md) | How-to | The CDK stacks, their options and the after-deploy commands |

- **Current behaviour only.** A page describes what the code does. Do not write "previously", "now", "no longer" or
  "(old behaviour)", and do not cite pull request numbers (`scripts/check_docs.py` fails a `#123` outside the
  CHANGELOG; name the change by feature and date). Release history and upgrade notes go in
  [CHANGELOG.md](./CHANGELOG.md) (*Upgrading* for what a user must do), which the runbook's
  [Upgrading the library](./docs/operations.md#upgrading-the-library) links.
- **Measured evidence** (benchmark numbers, measured costs and latencies, dated prices, how a default was chosen) goes
  in [docs/concepts/defaults-and-evidence.md](./docs/concepts/defaults-and-evidence.md), with the method and how to
  reproduce it. Other pages state the rule and link the evidence.
- **Extension recipes** (new strategies, backends, evaluators, renderers, parsers, config sections) live in
  the [Extension guide (§15)](./docs/guides/extend.md#15-extension-guide). Update the recipe there when the way to
  extend changes.
- **Diagrams.** A change that adds, removes or renames a stage, a strategy or a methodology, or changes how a
  strategy fuses its results, updates the diagrams that show it in the same pull request: the
  `assets/*.drawio.png` files embed their draw.io source, so open the PNG in draw.io, edit it and export it again
  as PNG with "Include a copy of my diagram" checked. `scripts/check_docs.py` fails a PNG without that copy or with a
  broken chunk: the draw.io command-line export (`drawio -x -e`) writes its zTXt chunk with a wrong CRC and no `IEND`,
  so rewrite the chunk CRCs and append `IEND` after such an export.

**Korean pages.** User-facing documentation is in English. The Tier-1 pages, the ones most Korean readers use, have a
Korean twin `<page>.ko.md` next to the English page. The one list of them is `TRANSLATED_PAGES` in
[`scripts/check_docs.py`](./scripts/check_docs.py): `README.md`, `docs/index.md`, the two `getting-started/` pages,
`concepts/overview.md`, `concepts/retrieval.md` and `concepts/scopes-and-namespaces.md`, and `guides/configure.md`,
`ingest.md`, `multilingual.md`, `query.md`, `structured-queries.md`, `incremental.md`, `domain-adaptation.md`,
`evaluate.md`, `mcp.md` and `console.md`. Every other page is English-only, and a Korean page that links one adds "(영문)"
after the link.

1. **Same pull request.** A change to a Tier-1 English page updates its `.ko.md` twin in the same pull request.
2. **Lag note.** If you cannot write the Korean text, say so in the pull request description and add this note at
   the top of the `.ko.md` page: `> 이 번역은 영문판보다 늦을 수 있습니다. 최신 내용은 영문판을 확인하세요.` A
   maintainer then updates the translation and removes the note.
3. **Structure check.** Each Tier-1 pair keeps the same number of headings per level, fenced code blocks and table
   rows, and the same set of link targets (a link to a Tier-1 page maps to its `.ko.md` twin). A mismatch fails CI
   unless the Korean page carries the lag note; that page is reported as lagging instead.
4. **New Korean pages.** A `.ko.md` file that `TRANSLATED_PAGES` does not list fails the check. Add a page to the list
   only with a full translation of the same structure.
5. **Korean style.** Korean headings keep their Korean heading ids, and links between Korean pages use them. Use
   Korean terms with the English term in parentheses at first use, following the glossary in
   [docs/index.ko.md](./docs/index.ko.md#용어집).

**The docs checker.** Run `uv run python scripts/check_docs.py` after editing a document. It checks every relative link,
`#fragment` (heading id), image, `<a href>` and reference-style link definition in the tracked Markdown files (each
target must exist and be tracked by Git), every inline code span that names a repository path, the `docs/*.md` paths
and this repository's `blob/main/docs/...#fragment` URLs cited in `unified_kg_rag/`, `scripts/`, `iac/`, `tests/`,
`.github/`, `docker/`, `config-template.yaml` and `pyproject.toml`, and the Korean rules above (the same inline code
spans and every English number in the Korean page; a Korean page links the Korean twin of a translated page and
marks a link to an English-only page "(영문)" in the same clause). It exits 1 and lists each failure.
`tests/unit/test_docs_links.py` and `tests/unit/test_docs_translations.py` run the same checks in CI as part of the
default test suite, and `tests/unit/test_docs_facts.py` checks the counts and lists the docs state (stages,
strategies, the `auto` routable list, methodologies) against the code: a sentence that states one carries an
explicit `<a id>` the test reads. Keep a heading's
text when other pages link its fragment, or update every link to it in the same pull request.

**Generated docs.** Two reference pages are generated from the code. Edit the source, regenerate, and commit both (a
unit test fails when a page is stale):

- [docs/reference/configuration.md](./docs/reference/configuration.md): every configuration key has a user-facing
  `Field(description=...)`. After changing a config field, run `uv run python scripts/gen_config_reference.py`. Add
  a key to `config-template.yaml`, the short starting file, only when most deployments set it.
- [docs/reference/cli.md](./docs/reference/cli.md): every CLI flag has a user-facing `help=`. After adding or
  changing a flag, run `uv run python scripts/gen_cli_reference.py`.

### Releasing

The version lives in one place, `__version__` in `unified_kg_rag/__init__.py`; `pyproject.toml` reads it from there
(`[tool.hatch.version]`). Between releases `main` carries the next version with a `.devN` suffix (for example
`0.3.0.dev0`), so a build from `main` is never mistaken for a release. Maintainers release as follows:

1. On a `chore/release-X.Y.Z` branch, set `__version__ = "X.Y.Z"` (a pre-release such as `X.Y.Zrc1` works the same
   way).
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new empty `## [Unreleased]` above
   it, and update the link references at the bottom: `[Unreleased]` compares `vX.Y.Z...HEAD`, and `[X.Y.Z]` compares
   the previous tag with `vX.Y.Z` (the first tagged release links to `releases/tag/vX.Y.Z` instead).
3. Optionally, run the `release` workflow by hand (Actions, Run workflow) on the branch: it is a dry run that builds and
   checks the artifacts and publishes nothing. Its notes fall back to `[Unreleased]`, a placeholder line when that
   section is empty (right after a release).
4. Land the pull request, then tag the merge commit on `main` and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z" <merge-commit>
   git push origin vX.Y.Z
   ```

   The tag push runs [.github/workflows/release.yml](./.github/workflows/release.yml). It fails unless the tagged commit
   is on `main`, the tag is `v` plus `__version__` and the changelog has a non-empty section for that version; it builds
   the sdist and wheel, runs `twine check --strict`, checks both carry LICENSE and NOTICE, installs the wheel in a clean
   environment, writes a CycloneDX SBOM of the locked dependencies with each package's licence (read from an environment
   synced with every extra, `.github/scripts/sbom_licenses.py`), attests the build provenance of the distributions and
   the SBOM (Sigstore), and creates the GitHub release with both files, their checksums, the SBOM and the changelog
   section's notes (`scripts/changelog_section.py`): the section's introduction, *Highlights*, *Breaking changes*,
   *Upgrading* and *Security*, then a link to the version's heading in the tagged `CHANGELOG.md` for *Added*, *Changed*
   and *Fixed*. A release body resolves no relative link, so `--link-base` points the section's repo-relative links at
   the tagged tree. A version with an `a`, `b`, `rc` or `.dev` segment is marked as a pre-release.
5. In a follow-up pull request, set `__version__` to the next development version (`X.Y.(Z+1).dev0` or
   `X.(Y+1).0.dev0`).

A failed release run creates no release; fix the cause on `main` and release a new version rather than moving a pushed
tag.

Publishing to PyPI is prepared and off. The `publish-pypi` job of `release.yml` runs after the GitHub release, on a
tag push only, and only when the repository variable `PYPI_PUBLISH` is `true`. It runs in the `pypi` GitHub
environment with `permissions: id-token: write` only, downloads the build job's artifacts, verifies their checksums
and uploads the sdist and wheel with the SHA-pinned `pypa/gh-action-pypi-publish` action through PyPI trusted
publishing (OpenID Connect), so no API token is stored in the repository. A PyPI upload cannot be replaced or
re-used under the same version, so a failed upload is fixed by releasing a new version.

To turn it on, a maintainer does the following once:

1. On PyPI, signed in to the account that will own the project (with two-factor authentication on), open "Your
   projects" > "Publishing" and add a pending publisher (GitHub): PyPI project name `unified-kg-rag-on-aws`, owner
   `awslabs`, repository `unified-kg-rag-on-aws`, workflow `release.yml`, environment `pypi`. No upload is needed:
   the first upload from this workflow creates the project, and the pending publisher becomes its trusted publisher.
   A pending publisher does not reserve the name: if someone else registers it first, the pending publisher is
   invalidated, so publish soon after adding it.
2. In the repository's settings, create the `pypi` environment. Restrict its deployment branches and tags to the tag
   pattern `v*`, and add required reviewers when a second maintainer is available, so every upload waits for an
   approval.
3. In the repository's settings, add the variable `PYPI_PUBLISH` with the value `true` (Settings > Secrets and
   variables > Actions > Variables). Deleting it, or any other value, turns publishing off again.

The next tag push then uploads the release to PyPI.

## Finding contributions to work on

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the
default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any
'help wanted' issues is a great place to start.


## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our
[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a
public GitHub issue. See [SECURITY.md](./SECURITY.md) for the logging policy and tracked dependency advisories.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your
contribution.
