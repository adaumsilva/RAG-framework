# Contributing to RAG Framework

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adaumsilva/RAG-framework/pulls)
[![Good First Issues](https://img.shields.io/github/issues/adaumsilva/RAG-framework/good%20first%20issue)](https://github.com/adaumsilva/RAG-framework/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
[![GitHub contributors](https://img.shields.io/github/contributors/adaumsilva/RAG-framework)](https://github.com/adaumsilva/RAG-framework/graphs/contributors)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Thank you for taking the time to contribute! Every contribution — bug fixes, new integrations, documentation improvements, or examples — makes this framework better for everyone.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Running Tests](#running-tests)
- [Pull Request Process](#pull-request-process)
- [Issue Labels](#issue-labels)
- [Good First Issues](#good-first-issues)
- [Adding a New Integration](#adding-a-new-integration)

---

## Code of Conduct

By participating in this project you agree to treat everyone with respect and professionalism. We follow the [Contributor Covenant](https://www.contributor-covenant.org/). In short: be kind, be constructive, and be patient.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/RAG-framework.git
   cd RAG-framework
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feat/openai-embedder
   ```

---

## Development Setup

We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

To install optional extras (e.g. while adding a PDF loader):

```bash
pip install -e ".[dev,pdf]"
```

---

## Making Changes

### Project structure

```
ragframework/
├── base.py          ← Abstract base classes — the contracts every component must honour
├── config.py        ← RAGConfig dataclass
├── exceptions.py    ← Custom exception hierarchy
├── document/        ← DocumentLoader and TextChunker implementations
├── embeddings/      ← Embedder implementations
├── retriever/       ← Retriever implementations
├── generator/       ← Generator implementations
├── pipeline/        ← RAGPipeline orchestrator
└── utils/           ← Shared helpers
```

### Key conventions

| Rule | Why |
|------|-----|
| Subclass the right ABC from `ragframework/base.py` | Keeps the pipeline plug-and-play |
| Raise the matching custom exception (`LoaderError`, `EmbedderError`, …) | Makes error handling predictable |
| Keep hard dependencies optional | Users should only install what they need |
| Guard optional imports at the top of the module | Fail fast with a helpful message |
| Write tests in `tests/` mirroring the package structure | CI must stay green |

### Optional dependency pattern

```python
# ragframework/embeddings/openai.py
from __future__ import annotations

try:
    import openai
except ImportError as exc:
    raise ImportError(
        "OpenAI support requires 'ragframework[openai]'. "
        "Install it with: pip install ragframework[openai]"
    ) from exc

from ragframework.base import Embedder

class OpenAIEmbedder(Embedder):
    ...
```

---

## Running Tests

```bash
# Run the full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=ragframework --cov-report=term-missing

# Run only a specific module
pytest tests/test_document/ -v
```

Linting and type checking:

```bash
ruff check ragframework/
mypy ragframework/
```

All checks must pass before a PR can be merged.

---

## Pull Request Process

1. Make sure `pytest`, `ruff`, and `mypy` all pass locally.
2. Add or update tests to cover your changes.
3. Update docstrings and `CHANGELOG.md` under `[Unreleased]`.
4. Keep PRs focused — one feature or bug fix per PR.
5. Fill in the PR template when you open the pull request.
6. A maintainer will review and may request changes. Once approved, it will be merged.

---

## Issue Labels

| Label | Meaning |
|-------|---------|
| `good first issue` | Suitable for first-time contributors — well-scoped and documented |
| `enhancement` | New feature or integration |
| `bug` | Something isn't working correctly |
| `documentation` | Docs, docstrings, or examples |
| `help wanted` | Extra attention needed — maintainers welcome input |
| `question` | Discussion or clarification needed |
| `wontfix` | Out of scope for this project |

---

## Good First Issues

Not sure where to start? Check the curated backlog in [`.github/GOOD_FIRST_ISSUES.md`](.github/GOOD_FIRST_ISSUES.md). Each issue includes:

- A clear description and motivation
- Acceptance criteria
- Which files to create or modify
- Useful resources and references

Browse issues on GitHub: [good first issue](https://github.com/adaumsilva/RAG-framework/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)

---

## Adding a New Integration

Most contributions are new implementations of an existing ABC. Here is a checklist:

- [ ] Create `ragframework/<module>/<name>.py` (e.g. `ragframework/embeddings/openai.py`)
- [ ] Subclass the correct ABC from `ragframework/base.py`
- [ ] Guard the optional import with a helpful error message
- [ ] Export the class from the module's `__init__.py`
- [ ] Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`
- [ ] Write tests in `tests/test_<module>/test_<name>.py`
- [ ] Add a usage snippet to `examples/` or the README
- [ ] Update `CHANGELOG.md`

That's it — open a PR and we'll take it from there!
