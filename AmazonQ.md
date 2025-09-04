# Development Guidelines

## Code Implementation Rules

### Write Minimal, Working Code
- Generate only essential code to solve the immediate problem
- Avoid extensive comments, docstrings, or documentation during development
- Use clear function names and structure to express intent
- Build incrementally with frequent validation
- **Always seek the simplest, most optimal solution - avoid over-engineering**
- **Start with the most straightforward approach before adding complexity**
- **Prioritize working solutions over perfect architecture**

### Python Code Standards

**Type Hints & Naming:**

```python
# ✅ Use modern built-in types with descriptive names
def extract_document_metadata(documents: list[dict[str, str]],
                            include_timestamps: bool = True) -> dict[str, str | int]:
    return {}

# ❌ Avoid legacy typing and unclear names
from typing import Dict, List
def process(*args, **kwargs) -> Dict[str, List[str]]:
    pass
```

**Function Design:**
- Keep functions under 20-30 lines with single responsibility
- Use explicit parameters instead of `*args`, `**kwargs`
- Prefer pure functions that return new objects vs mutation
- Use Pydantic models for data validation at boundaries
- **Choose the most direct implementation path**

**Logging:**

```python
# ✅ Use % formatting for performance
logger.info("Processing document %s with %d pages", document_id, page_count)

# ❌ Avoid f-strings in logging
logger.info(f"Processing document {document_id} with {page_count} pages")
```

## Architecture Patterns

### Required Libraries
- **Data Models**: Use Pydantic (not dataclasses)
- **File Operations**: Use pathlib (not os.path)
- **Package Management**: Use uv (not pip)
- **Choose tools that solve problems efficiently, not for novelty**

### LangChain Integration
Structure all LLM interactions using LangChain Expression Language (LCEL):

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Analyze: {document}")
chain = prompt | llm | StrOutputParser()
```

- Store prompts in Python files (.py) for version control
- Build modular, reusable chains using LCEL syntax
- Use proper output parsers for structured responses
- **Implement the minimal chain structure that meets requirements**

### AWS Services
- Prioritize managed services to minimize operational overhead
- Use boto3 with consistent session management patterns
- Implement event-driven communication with EventBridge, SQS, SNS
- Design for horizontal scaling with stateless services
- **Select AWS services based on actual needs, not feature completeness**

## Error Handling & Reliability

### Exception Management
- Create specific custom exceptions for different error types
- Implement fail-fast principle - detect errors early
- Use Pydantic validation at module boundaries
- Design graceful degradation for partial functionality
- **Handle only the errors you can meaningfully recover from**

### Testing Approach
- Create essential tests in `tests/` directory structure:

```text
tests/
├── unit/           # Core business logic tests
├── integration/    # AWS service integration tests
└── fixtures/       # Test data and mocks
```

- Focus on unit tests, selective integration tests
- Ensure test isolation and repeatability
- Avoid test proliferation - quality over quantity
- **Test critical paths first, edge cases second**

## Project Structure Guidelines

### Organization Principles
- Group related functionality by feature, not technical layer
- Separate presentation, business logic, and data layers
- Use clear module boundaries with well-defined APIs
- Implement dependency injection for testability
- **Start with simple flat structure, refactor when complexity demands it**

### Performance Considerations
- Use async patterns for I/O-bound operations
- Implement caching strategies and connection pooling
- Design for event-driven architecture to decouple components
- Optimize for horizontal scaling patterns
- **Optimize based on measured bottlenecks, not assumptions**

## Development Philosophy

### Simplicity First
- **Solve the immediate problem with the most straightforward approach**
- **Avoid premature abstractions and complex design patterns**
- **Refactor toward better design only when current approach becomes limiting**
- **Question every dependency and feature - remove what doesn't add clear value**
- **Prefer boring, proven solutions over exciting new approaches**
- **Build incrementally: working solution → optimized solution → scalable solution**
