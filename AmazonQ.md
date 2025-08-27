# Claude Development Principles

## Core Philosophy

### Rapid Prototyping
- **Implementation First**: Validate core functionality with working code before extensive documentation
- **Self-Documenting Code**: Express intent through clear function names and logical structure
- **Iterative Refinement**: Build in small, testable increments with frequent validation
- **Documentation After Stabilization**: Comprehensive documentation follows API stabilization

### Modular Architecture
- **Single Responsibility**: Each module has one clear, well-defined purpose
- **Interface-Based Design**: Abstract interfaces enable swappable implementations
- **Dependency Injection**: Support testing and configuration through DI patterns
- **Clear Module Boundaries**: Well-defined APIs with proper encapsulation

### AWS-Native Design
- **Managed Services Priority**: Minimize operational overhead with AWS services
- **Cloud-Native Patterns**: Leverage event-driven, serverless, and auto-scaling architectures
- **Loose Coupling**: Enable independent development/deployment via APIs, queues, and events
- **Infrastructure as Code**: Ensure reproducible deployments with CloudFormation/CDK

## Coding Principles

### Naming and Types
```python
# ✅ Modern built-in generics
def process_docs(docs: list[dict[str, str]]) -> dict[str, list[str]]:
    pass

# ❌ Avoid legacy typing imports
from typing import Dict, List
def process_docs(docs: List[Dict[str, str]]) -> Dict[str, List[str]]:
    pass
```

- **Intent-Revealing Names**: `document_processor` vs `doc_proc`
- **Consistent Conventions**: Follow PEP 8 (`snake_case` functions/variables, `PascalCase` classes)
- **Mandatory Type Hints**: Apply comprehensive type hints to all function signatures
- **Modern Types**: Prefer built-in `dict`, `list`, `tuple` over `typing` module equivalents

### Function Design
```python
# ✅ Clear signature
def extract_metadata(document: dict[str, str], include_timestamps: bool = True) -> dict[str, str | int]:
    return {}

# ❌ Unclear signature
def process(*args, **kwargs):
    pass
```

- **Small, Focused Functions**: Keep under 20-30 lines with single purpose
- **Explicit Parameters**: Avoid `*args`, `**kwargs` in favor of clear signatures
- **Pure Functions Preferred**: Minimize external state modifications
- **Immutability Patterns**: Use Pydantic models, return new objects vs mutation

### Error Handling
```python
# ✅ Specific custom exceptions
class DocumentProcessingError(Exception):
    def __init__(self, document_id: str, reason: str):
        self.document_id = document_id
        super().__init__(f"Failed to process document {document_id}: {reason}")

# ✅ Resource management
def process_with_aws():
    with boto3.client('s3') as client:
        # Safe client usage
        pass
```

- **Specific Exceptions**: Custom exceptions for clear error categorization
- **Input Validation**: Pydantic validation at module boundaries with fail-fast approach
- **Resource Management**: Context managers for safe resource handling
- **Graceful Degradation**: Meaningful fallback behaviors for partial failures

## Technology Stack

### Core Libraries
- **Data Structures**: Pydantic (instead of dataclasses)
- **File System**: pathlib (instead of os.path)
- **Prompt Management**: Python files (.py) for version control
- **Package Management**: uv (instead of pip)

### Integration Patterns
- **LCEL(LangChain Expression Language) Utilization**: Structure LLM interactions as reusable, modular chains
- **Consistent AWS Clients**: Standardize boto3 session management

