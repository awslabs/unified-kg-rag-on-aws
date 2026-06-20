# Contributing to AWS Native Graph RAG

We welcome contributions to the AWS Native Graph RAG framework! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- AWS CLI configured with appropriate permissions
- Git for version control
- Familiarity with AWS services (Bedrock, Neptune, OpenSearch, S3)

### Development Setup
1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/aws-graphrag.git
   cd aws-graphrag
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set Up Pre-commit Hooks** (if available)
   ```bash
   pre-commit install
   ```

## 📋 Development Guidelines

### Extending the framework (ports, adapters & registries)

The codebase uses a hexagonal (ports & adapters) architecture with registries, so
most extensions need **no edits to existing dispatch code**. See `CLAUDE.md` for
the full guide. In short:

- **New search strategy** — subclass `BaseSearchStrategy`, decorate with
  `@register_strategy(SearchStrategy.X, required_roles=(...))`, export from
  `adapters/search_strategies/__init__.py`.
- **New storage / LLM backend** — implement the relevant port in
  `ports/` and register it; do not hardcode it into a manager's `__init__`.
- **New evaluator** — subclass `BaseGraphRAGEvaluator`, add an
  `EVALUATOR_MAPPING` entry and an `EvaluatorType` enum value.
- **New visualization renderer** — subclass `BaseRenderer`, decorate with
  `@register_renderer("name")`; the manager and `run-visualization` pick it up.
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
black aws_graphrag tests

# Sort imports
isort aws_graphrag tests

# Lint code
ruff check aws_graphrag tests

# Type checking
mypy aws_graphrag
```

### Testing
- Write **unit tests** for new functionality
- Maintain **80%+ code coverage**
- Use **pytest** for testing framework
- Mock AWS services in unit tests using `pytest-mock`

Run tests:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aws_graphrag --cov-report=html

# Run specific test file
pytest tests/test_specific_module.py
```

## 🔄 Contribution Process

### 1. Issue Creation
- **Search existing issues** before creating new ones
- Use **issue templates** when available
- Provide **clear descriptions** and **reproduction steps** for bugs
- Include **use cases** and **expected behavior** for feature requests

### 2. Branch Strategy
- Create feature branches from `main`
- Use descriptive branch names: `feature/add-new-search-strategy` or `fix/memory-leak-in-pipeline`
- Keep branches focused on single features or fixes

### 3. Pull Request Process
1. **Create Pull Request**
   - Use the PR template
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

### Bug Report Template
```markdown
**Environment:**
- Python version: 3.10.x
- OS: macOS/Linux/Windows
- AWS Region: us-east-1

**Configuration:**
```yaml
# Relevant config sections (remove sensitive data)
```

**Steps to Reproduce:**
1. Run command: `run-ingestion --source-directory ./docs`
2. Observe error in logs

**Expected Behavior:**
Documents should be processed successfully

**Actual Behavior:**
Pipeline fails with error: [error message]

**Additional Context:**
- Log files attached
- Occurs with specific document types
```

## 💡 Feature Requests

### Guidelines
- **Describe the use case** clearly
- **Explain the business value**
- **Provide implementation suggestions** if possible
- **Consider AWS-native alternatives**

### Feature Request Template
```markdown
**Use Case:**
As a [user type], I want to [functionality] so that [benefit].

**Current Limitation:**
Currently, the framework cannot [limitation].

**Proposed Solution:**
Implement [solution] using [AWS services/approach].

**Alternative Solutions:**
- Option 1: [alternative]
- Option 2: [alternative]

**Additional Context:**
- Related to issue #123
- Similar to feature in [other project]
```

## 🔒 Security

### Security Guidelines
- **Never commit** AWS credentials or sensitive data
- **Use IAM roles** with minimal required permissions
- **Sanitize logs** to remove sensitive information
- **Follow AWS security best practices**

### Reporting Security Issues
- **Do not** create public issues for security vulnerabilities
- **Report** security concerns through GitHub's private vulnerability reporting feature
- **Include** detailed description and reproduction steps
- **Allow** reasonable time for response before disclosure

## 📦 Release Process

### Version Management
- Follow **Semantic Versioning** (SemVer)
- Update version in `pyproject.toml`
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
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to AWS Native Graph RAG! 🚀
