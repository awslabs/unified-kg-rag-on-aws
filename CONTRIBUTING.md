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

You need Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/). AWS credentials are needed only for real-AWS runs; the
test suite runs without AWS.

```bash
git clone https://github.com/<your-username>/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
uv sync                          # creates .venv with the package and the dev dependency group
uv run pre-commit install        # runs ruff, isort, black and mypy on every commit
```

Without uv: create a virtual environment, activate it, and run `pip install -e . --group dev` (pip 25.1 or later).

### Branches and pull requests

- Never commit to `main`. Work on a topic branch named after the change type, for example `feat/…`, `fix/…`,
  `docs/…` or `chore/…`, and land it through a pull request.
- Keep one logical change per pull request and per commit. Do not mix a refactor, a feature and a documentation fix.
- Fill in the [pull request template](./.github/PULL_REQUEST_TEMPLATE.md).
- Pull requests are squash-merged after CI passes and a maintainer approves, so `main` keeps one commit per change.
- Add an entry under `## [Unreleased]` in [CHANGELOG.md](./CHANGELOG.md) for any user-visible change. Release history
  belongs in the changelog only, not in the user guide or design doc.
- A **Breaking changes** entry covers what users depend on: the names exported from `unified_kg_rag` (pinned in
  `tests/unit/test_public_api.py`; changing that list is a deliberate API change), configuration keys, CLI flags and
  exit codes, index and registry formats, and the CDK stack's inputs and outputs. Moving or renaming an internal module
  is not a breaking change and needs no entry; neither does a change to code only reached through internal imports.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary` in the imperative mood, with a
subject of 72 characters or fewer. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `style`, `perf`, `ci`,
`build`. The body explains why the change is needed; the diff already shows what changed.

```text
fix(incremental): retry documents whose extraction failed
```

### Checks to run before pushing

CI ([`.github/workflows/quality.yml`](./.github/workflows/quality.yml)) runs the same checks, so run them locally first:

```bash
uv run pre-commit run --all-files                       # ruff, isort, black, mypy and file hygiene
uv run pytest -m "not aws"                              # unit, property and integration suites
uv run --extra otel pytest -m "not aws" --cov=unified_kg_rag  # with coverage, as CI measures it
```

CI fails below the coverage threshold set by `--cov-fail-under` in `quality.yml`. It also runs the suite on Python
3.10, checks the optional parsers, runs the local-store smoke test, builds and runs the container image, and
synthesizes `iac/` with cdk-nag. `iac/` is a separate uv project; if you change it, also run
`cd iac && uv sync && uv run cdk synth -c enable_cdk_nag=true && uv run pytest` (see [`iac/README.md`](./iac/README.md)).

When a pull request changes `pyproject.toml` or `uv.lock`, the
[`licenses`](./.github/workflows/licenses.yml) workflow also syncs every extra and fails on a copyleft,
source-available or unlabeled license among the runtime dependencies. Reviewed exceptions are listed, each with its
reason, in `.github/scripts/check_licenses.py`; add one only after reviewing the license.

### Tests

- Tests run without AWS. Use the port-based in-memory fakes in `tests/fixtures/fakes/` (for example
  `FakeDocStatusStore`) rather than ad-hoc boto3 mocks, and `moto` when an adapter must be exercised against a boto3
  surface.
- Markers: `unit`, `integration`, `property`, `aws` (real AWS, excluded in CI) and `slow`. Each test times out after
  120 s.
- `tests/integration/test_local_stores.py` runs against the local containers in `docker/compose.local.yaml` when
  `LOCAL_STORES=1` is set.
- Tests, fixtures and evaluation data use synthetic content only (generic entities such as "Vendor" and "Buyer",
  made-up amounts). Never add real corpus text, customer names, account ids, endpoints or hostnames to code, tests,
  docs or commit messages.

### Code conventions

- Modern built-in types (`list`, `dict`, `X | None`), Pydantic models at boundaries, `pathlib` for paths.
- `%`-style logging arguments (`logger.info("did %s", x)`), not f-strings.
- LLM calls as LangChain LCEL chains; prompts live in `unified_kg_rag/domain/prompts/` and can be overridden with
  `custom_prompts`.
- Raise the specific exception types from `unified_kg_rag/shared/exceptions.py`.
- Keep the dependency rule: `domain/` must not import boto3, LangChain or a backend client, directly or transitively
  (`tests/unit/test_domain_purity.py` checks this).

### Extending the framework

New search strategies, storage or model backends, parsers, renderers, evaluators and config sections each have a
recipe in [docs/design.md §15 Extension Guide](./docs/design.md#15-extension-guide). Most need a registration rather
than an edit to dispatch code.

### Documentation

- User-facing documentation is in English. `README.md` and `docs/user-guide.md`, `docs/design.md`,
  `docs/operations.md` and `docs/models.md` have Korean translations (`*.ko.md`); `docs/glossary.ko.md` fixes the
  Korean terms.
- **EN/KO sync rule:** a change to a document that has a `.ko.md` twin updates both files in the same pull request. If
  you cannot write the Korean text, say so in the pull request description and add this note at the top of the
  `.ko.md` file so readers know it lags: `> 이 번역은 영문판보다 늦을 수 있습니다. 최신 내용은 영문판을 확인하세요.`
  A maintainer then updates the translation and removes the note.
- Describe current behavior only. Do not write "previously", "now" or "(old behaviour)" in the guides; put release
  history in [CHANGELOG.md](./CHANGELOG.md).
- Every configuration key has a user-facing `Field(description=...)`. After changing a config field, run
  `uv run python scripts/gen_config_reference.py` to regenerate [docs/configuration.md](./docs/configuration.md)
  (a unit test fails when it is stale), and update `config-template.yaml`, the commented starting file.


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
