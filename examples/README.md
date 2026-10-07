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
| [anthropic_generator.py](anthropic_generator.py) | End-to-end pipeline using `AnthropicGenerator` (requires `ANTHROPIC_API_KEY`) |

## Running the examples

### Python script

From the repository root:

```bash
pip install -e ".[dev]"
python examples/basic_rag.py
```

For the Anthropic example:

```bash
pip install -e ".[dev,anthropic]"
export ANTHROPIC_API_KEY=your-key-here
python examples/anthropic_generator.py
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

## OpenAI embeddings

Install the OpenAI extra, which includes both the OpenAI SDK and the tokenizer
used for request batching, then set `OPENAI_API_KEY`:

```bash
pip install "ragframework[openai]"
export OPENAI_API_KEY=your-key-here
```

Create embeddings synchronously with the default `text-embedding-3-small`
model:

```python
from ragframework.embeddings import OpenAIEmbedder

with OpenAIEmbedder() as embedder:
    vectors = embedder.embed(["First document", "Second document"])
```

The embedder batches by item and token limits, preserves input order, and also
provides `await embedder.aembed(texts)` for asynchronous use. Close the client
with `close()` / `aclose()` or use the corresponding context manager.
