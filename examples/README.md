# Examples

| File | Description |
|------|-------------|
| [basic_rag.py](basic_rag.py) | End-to-end pipeline using only built-in components (no API key needed) |

## Running the examples

```bash
# from the repo root
pip install -e ".[dev]"
python examples/basic_rag.py
```

## Adding real LLM / embedder support

The built-in `RandomEmbedder` and `EchoGenerator` are placeholders.
Swap them for real implementations once the community contributes them
(see [Good First Issues](../.github/GOOD_FIRST_ISSUES.md)):

```python
# future usage (once contributed)
from ragframework.embeddings.openai import OpenAIEmbedder
from ragframework.generator.openai import OpenAIGenerator

pipeline = RAGPipeline(
    ...
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    generator=OpenAIGenerator(model="gpt-4o-mini"),
)
```
