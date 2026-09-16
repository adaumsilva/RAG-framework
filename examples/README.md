# Examples

## FAISS retriever

Install the optional dependency with `pip install "ragframework[faiss]"`, then
index chunks whose embeddings have already been populated:

```python
from ragframework.base import Chunk
from ragframework.retriever import FAISSRetriever

retriever = FAISSRetriever()
retriever.add([
    Chunk(id="north", content="North", embedding=[1.0, 0.0]),
    Chunk(id="east", content="East", embedding=[0.0, 1.0]),
])
matches = retriever.retrieve([0.9, 0.1], top_k=1)
```

| File | Description |
|------|-------------|
| [basic_rag.py](basic_rag.py) | End-to-end pipeline using only built-in components (no API key needed) |
| [basic_rag.ipynb](basic_rag.ipynb) | Interactive Jupyter notebook covering the complete RAG pipeline |

## Running the examples

### Python script

From the repository root:

```bash
pip install -e ".[dev]"
python examples/basic_rag.py
```

### Jupyter notebook

Install Jupyter dependencies:

```bash
pip install jupyter ipykernel
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
examples/basic_rag.ipynb
```

Select the Python environment for the RAG Framework and run the notebook
cells from top to bottom.

The notebook demonstrates:

1. Loading a document.
2. Splitting the document into chunks.
3. Generating embeddings.
4. Adding chunks to the retriever.
5. Querying the RAG pipeline.
6. Inspecting retrieved source chunks.

## Adding real LLM / embedder support

The built-in `RandomEmbedder` and `EchoGenerator` are placeholders.
Swap them for real implementations once the community contributes them
(see [Good First Issues](../.github/GOOD_FIRST_ISSUES.md)).

For example, future integrations may look like:

```python
from ragframework.embeddings.openai import OpenAIEmbedder
from ragframework.generator.openai import OpenAIGenerator

pipeline = RAGPipeline(
    ...
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    generator=OpenAIGenerator(model="gpt-4o-mini"),
)
```

These integrations are not included in the current built-in framework.
