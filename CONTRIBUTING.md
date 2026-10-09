# Contributing to Unified Knowledge Graph RAG on AWS

We welcome contributions to the Unified Knowledge Graph RAG on AWS framework! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (the project's package manager)
- AWS CLI configured with appropriate permissions (only for real-AWS runs; the
  test suite is AWS-free)
- Git for version control
- Familiarity with AWS services (Bedrock, Neptune, OpenSearch, S3)

### Development Setup
1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/unified-kg-rag-on-aws.git
   cd unified-kg-rag-on-aws
   ```

2. **Install Dependencies**
   ```bash
   uv sync  # creates .venv with the package and the dev dependency group
   ```
   Without uv: `python -m venv .venv`, activate it, then
   `pip install -e . --group dev` (pip >= 25.1).

3. **Set Up Pre-commit Hooks**
   ```bash
   uv run pre-commit install
   ```

## 📋 Development Guidelines

### Extending the framework (ports, adapters & registries)

The codebase uses a hexagonal (ports & adapters) architecture with registries, so
most extensions need **no edits to existing dispatch code**. See `CLAUDE.md` for
the full guide. In short:

- **New search strategy** — add a member to the `SearchStrategy` enum
  (`domain/models/retrieval.py`; the CLI `--search-strategy` choices follow
  it), subclass `BaseSearchStrategy`, decorate with
  `@register_strategy(SearchStrategy.X, required_roles=(...))`, and export it
  from `adapters/search_strategies/__init__.py`. No edit to `rag_chain` is
  needed; add it to `search.auto_routable_strategies` if `auto` may pick it.
- **New storage / LLM backend** — implement the relevant port from `ports/`
  and pass it to the constructor that uses it (there is no backend registry):
  `Providers(...)` for models, `DataIngestionPipeline(..., doc_status=...,
  vector_indexer=..., graph_indexer=...)` or `IndexingManager(...)` for the
  write side, `GraphRAGChain(retriever_builders=...)` for retrieval. Do not
  hardcode it into a manager's `__init__`. See `docs/design.md` §15.
- **New evaluator** — subclass `BaseGraphRAGEvaluator`, add a branch in
  `EvaluationManager._resolve_evaluator_class`, and add an `EvaluatorType` enum
  value.
- **New visualization renderer** — subclass `BaseRenderer` and decorate
  with `@register_renderer("name")`. Registration happens on import: the
  visualization manager uses any renderer whose module was imported in the
  process, while `run-visualization` only sees the renderers that
  `adapters/renderers/__init__.py` imports, so a renderer in this package must
  be imported there.
- **New config section** — add a Pydantic `BaseModel`, attach via
  `Field(default_factory=...)`, document it in `config-template.yaml`.

Tests run **AWS-free by default** — use the port-based fakes in
`tests/fixtures/fakes/` (or `moto`) rather than ad-hoc boto3 mocks. Markers:
`unit`, `integration`, `property`, `aws` (real AWS, skipped in CI), `slow`.

### Code Style
- Follow **PEP 8** standards
- Use **type hints** for all function parameters and return values
- Write **descriptive variable and function names**
- Keep functions focused and under 30 lines when possible
- Use **Pydantic models** for data structures over dataclasses

### Code Quality Tools
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting and code analysis
- **mypy**: Static type checking

Run quality checks:
```bash
# Format code
uv run black unified_kg_rag tests

# Sort imports
uv run isort unified_kg_rag tests

# Lint code
uv run ruff check unified_kg_rag tests

# Type checking
uv run mypy unified_kg_rag

# Or run every pre-commit hook at once
uv run pre-commit run --all-files
```

### Testing
- Write **unit tests** for new functionality
- Keep coverage at or above the current CI gate (**84%**, ratcheted up with
  measured coverage — see `.github/workflows/quality.yml` `--cov-fail-under`)
- Use **pytest** for testing framework
- Prefer the port-based in-memory fakes in `tests/fixtures/fakes/` (e.g.
  `FakeDocStatusStore`) over ad-hoc boto3 mocking; use `moto` when an adapter
  must be exercised against a real boto3 surface. In practice DynamoDB and S3
  are tested with `moto`, Neptune and OpenSearch with the in-memory fakes
- Tests that need real AWS services carry the `aws` marker and are excluded in CI

Run tests (AWS-free by default):
```bash
# Run all non-AWS tests (unit, integration, property)
uv run pytest -m "not aws"

# Run with coverage
uv run pytest -m "not aws" --cov=unified_kg_rag --cov-report=html

# Run a specific test file
uv run pytest tests/unit/test_chunker_logic.py
```

CI (`.github/workflows/quality.yml`) runs ruff, black, isort, mypy, and pytest
with the coverage gate (one `-m "not aws"` run that includes the unit,
property and integration suites), plus the oldest supported Python (3.10),
the optional-parser checks, and `cdk synth` with cdk-nag for `iac/`. `.github/workflows/security.yml` runs a
non-blocking ASH security scan on pushes to `main`.

## 🔄 Contribution Process

### 1. Issue Creation
- **Search existing issues** before creating new ones
- Use the **issue templates** (bug report, feature request)
- Provide **clear descriptions** and **reproduction steps** for bugs
- Include **use cases** and **expected behavior** for feature requests

### 2. Branch Strategy
- Create feature branches from `main`
- Use descriptive branch names: `feature/add-new-search-strategy` or `fix/memory-leak-in-pipeline`
- Keep branches focused on single features or fixes

### 3. Pull Request Process
1. **Create Pull Request**
   - Fill in the PR template (`.github/PULL_REQUEST_TEMPLATE.md`)
   - Link related issues
   - Provide clear description of changes

2. **Code Review Requirements**
   - All CI checks must pass
   - At least one approving review required
   - No merge conflicts with main branch

3. **Merge Requirements**
   - Squash commits for clean history
   - Update documentation if needed
   - Add entry to CHANGELOG.md

## 📝 Documentation

### Code Documentation
- Use **docstrings** for all public functions and classes
- Follow **Google docstring format**
- Include **parameter types** and **return value descriptions**
- Provide **usage examples** for complex functions

Example:
```python
def extract_entities(text: str, model_id: str) -> list[Entity]:
    """Extract entities from text using specified LLM model.

    Args:
        text: Input text to process
        model_id: Bedrock model identifier for entity extraction

    Returns:
        List of extracted Entity objects with names and types

    Raises:
        ExtractionError: If entity extraction fails

    Example:
        >>> entities = extract_entities("John works at AWS", "claude-3")
        >>> print(entities[0].name)
        "John"
    """
```

### README Updates
- Update README.md for new features
- Add configuration examples
- Include CLI usage examples
- Update API documentation

## 🏗️ Architecture Guidelines

### AWS-Native Principles
- **Prefer managed services** over self-hosted solutions
- **Use IAM roles** instead of access keys when possible
- **Implement proper error handling** for AWS service calls
- **Follow AWS Well-Architected Framework** principles

### Design Patterns
- **Single Responsibility**: Each class/function has one clear purpose
- **Dependency Injection**: Use interfaces for AWS service dependencies
- **Factory Pattern**: For creating AWS service clients
- **Strategy Pattern**: For different search and processing strategies

### Performance Considerations
- **Batch operations** when possible
- **Implement caching** for expensive operations
- **Use async/await** for I/O operations
- **Monitor token usage** for LLM calls

## 🐛 Bug Reports

### Information to Include
- **Environment details** (Python version, OS, AWS region)
- **Configuration** (sanitized config.yaml)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages** and stack traces
- **Log files** (with sensitive data removed)

Open a bug with the **Bug report** template
(`.github/ISSUE_TEMPLATE/bug_report.md`), which asks for these fields.

## 💡 Feature Requests

### Guidelines
- **Describe the use case** clearly
- **Explain the business value**
- **Provide implementation suggestions** if possible
- **Consider AWS-native alternatives**

Open a proposal with the **Feature request** template
(`.github/ISSUE_TEMPLATE/feature_request.md`).

## 🔒 Security

### Security Guidelines
- **Never commit** AWS credentials or sensitive data
- **Use IAM roles** with minimal required permissions
- **Sanitize logs** to remove sensitive information
- **Follow AWS security best practices**

### Reporting Security Issues
- **Do not** create public issues for security vulnerabilities
- **Report** security concerns privately to the maintainers
- **Include** detailed description and reproduction steps
- **Allow** reasonable time for response before disclosure

### Dependency / SAST scan findings
The `security` workflow (`.github/workflows/security.yml`) runs the Automated
Security Helper (ASH) in report-only mode. Assess findings for reachability and
disposition before acting; pin or upgrade dependencies via `uv lock --upgrade`.

## 📦 Release Process

### Version Management
- Follow **Semantic Versioning** (SemVer)
- Update `__version__` in `unified_kg_rag/__init__.py` (`pyproject.toml`
  reads the version from it dynamically)
- Create **release notes** with changes
- Tag releases in Git

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Release notes prepared
- [ ] Git tag created

## 🤝 Community

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community support
- **Pull Requests**: Code contributions and reviews

### Code of Conduct
- Be **respectful** and **inclusive**
- **Help others** learn and contribute
- **Focus on** constructive feedback
- **Follow** GitHub's Community Guidelines

## 📚 Resources

### Learning Resources
- [Microsoft GraphRAG Research Papers](https://arxiv.org/abs/2404.16130)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Amazon Neptune Documentation](https://docs.aws.amazon.com/neptune/)
- [Amazon OpenSearch Documentation](https://docs.aws.amazon.com/opensearch-service/)

### Development Tools
- [AWS CLI](https://aws.amazon.com/cli/)
- [AWS SDK for Python (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [LangChain Documentation](https://python.langchain.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## 🙏 Recognition

Contributors will be recognized in:
- **README acknowledgments** and **release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to Unified Knowledge Graph RAG on AWS! 🚀
