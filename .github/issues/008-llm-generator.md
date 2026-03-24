---
title: "OpenAI / Anthropic LLM Generator"
labels: ["good first issue", "enhancement"]
---

## Description

Add `OpenAIGenerator` and/or `AnthropicGenerator` that format retrieved chunks as context and call the respective chat completion API to produce a grounded answer.

## Motivation

`EchoGenerator` is a placeholder that just returns the retrieved text. Real applications need an LLM to synthesise a coherent answer from the retrieved context. This is the final piece needed for a fully functional RAG pipeline.

## Acceptance criteria

- [ ] `OpenAIGenerator` in `ragframework/generator/openai.py` **and/or** `AnthropicGenerator` in `ragframework/generator/anthropic.py`
- [ ] Subclasses `Generator` from `ragframework/base.py`
- [ ] Configurable `model`, `api_key` (env var fallback), `system_prompt`, `max_tokens`
- [ ] Formats context chunks into a clear prompt (chunk number + content)
- [ ] Uses `openai` / `anthropic` SDK (already in optional extras in `pyproject.toml`)
- [ ] Import guarded; raises `GeneratorError` on failure
- [ ] Unit tests using `unittest.mock` — no real API calls in CI
- [ ] Exported from `ragframework/generator/__init__.py`
- [ ] Usage example added to `examples/`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/generator/openai.py` and/or `ragframework/generator/anthropic.py`
- `ragframework/generator/__init__.py` — export
- `tests/test_generator/` — new test files
- `examples/` — add example script

## Resources

- [OpenAI Chat Completions](https://platform.openai.com/docs/guides/text-generation)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- `EchoGenerator` in `ragframework/generator/echo_generator.py` as reference
