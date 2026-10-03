# Getting Started

RAG Framework is a modular Python framework for building Retrieval-Augmented Generation (RAG) pipelines. It provides components for loading documents, splitting them into chunks, generating embeddings, retrieving relevant chunks, and generating answers.

## Installation

RAG Framework requires Python 3.10 or newer. Install the core package with:

```bash
pip install ragframework
```

For development from a clone, create and activate a virtual environment, then install the development dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
pip install -e ".[dev]"
```

Optional integrations are installed with extras:

```bash
pip install "ragframework[pdf]"          # PDFLoader
pip install "ragframework[huggingface]"  # HuggingFaceEmbedder
pip install "ragframework[faiss]"        # FAISSRetriever
pip install "ragframework[chromadb]"     # ChromaRetriever
```

For a local editable install, use the same extras after `-e`, for example `pip install -e ".[pdf]"`.

## Quickstart

The repository includes a complete example that uses built-in components and does not require an API key. From the repository root, run:

```bash
python examples/basic_rag.py
```

The example loads a text file, chunks it, embeds the chunks, indexes them, and queries the pipeline. Its `RandomEmbedder` and `EchoGenerator` are demonstration components; use a semantic embedder and an LLM-backed generator for production applications.

## Logging

RAG Framework uses Python's standard-library logging and is silent by default. Enable pipeline stage summaries with:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

Use `logging.DEBUG` to include per-batch embedding and retriever indexing details.

## Building a Pipeline

The pipeline coordinates a loader, chunker, embedder, retriever, and generator. This setup matches the current `RAGPipeline` constructor and the imports used by `examples/basic_rag.py`:

```python
from ragframework.config import RAGConfig
from ragframework.document import FixedSizeChunker, TextFileLoader
from ragframework.embeddings import RandomEmbedder
from ragframework.generator import EchoGenerator
from ragframework.pipeline import RAGPipeline
from ragframework.retriever import InMemoryRetriever

pipeline = RAGPipeline(
    loader=TextFileLoader(),
    chunker=FixedSizeChunker(chunk_size=200, chunk_overlap=40),
    embedder=RandomEmbedder(dim=64, seed=42),
    retriever=InMemoryRetriever(),
    generator=EchoGenerator(),
    config=RAGConfig(top_k=3),
)
```

Alternatively, `RAGPipeline.from_config()` builds the chunker from `RAGConfig`'s chunk settings:

```python
pipeline = RAGPipeline.from_config(
    RAGConfig(chunk_size=200, chunk_overlap=40, top_k=3),
    loader=TextFileLoader(),
    embedder=RandomEmbedder(dim=64, seed=42),
    retriever=InMemoryRetriever(),
    generator=EchoGenerator(),
)
```

## Loading Documents

`DocumentLoader.load(source)` returns a list of `Document` objects. The built-in text loader accepts a file path:

```python
documents = TextFileLoader().load("example.txt")
print(documents[0].content)
```

For PDFs, install the `[pdf]` extra and use the public `PDFLoader` export. By default, it returns one document per page; pass `split_pages=False` to return one document for the whole file:

```python
from ragframework.document import PDFLoader

documents = PDFLoader().load("report.pdf")
```

## Chunking

Chunkers turn each `Document` into `Chunk` objects. `FixedSizeChunker` splits into character windows. `RecursiveChunker` tries its configured separators in order to keep chunks together along paragraph, line, sentence, or word boundaries where possible:

```python
from ragframework.document import RecursiveChunker

chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(documents[0])
```

Both chunkers also provide `from_config(config)` class methods.

## Embeddings

An `Embedder` converts a list of texts into vectors. `RandomEmbedder` is useful for examples and tests, but its vectors do not provide meaningful semantic retrieval. For local semantic embeddings, install the `[huggingface]` extra:

```python
from ragframework.embeddings import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
vectors = embedder.embed(["A sample sentence."])
```

## Ingestion

`RAGPipeline.ingest(source)` loads, chunks, embeds, and indexes a source. It returns the number of chunks added:

```python
n_chunks = pipeline.ingest("example.txt")
print(f"Ingested {n_chunks} chunks")
```

Use `pipeline.ingest_many(["first.txt", "second.txt"])` to ingest several sources and get their total chunk count.

## Querying

After ingestion, call `query` with a question. The optional `top_k` keyword overrides the configured result count for that call:

```python
response = pipeline.query("What are the stages of a RAG pipeline?", top_k=3)
print(response.answer)
```

The returned `RAGResponse` includes the original query and the retrieved `source_chunks` used as context.

## Inspecting Retrieved Results

Each retrieved chunk includes its ID, content, and metadata:

```python
for chunk in response.source_chunks:
    print(chunk.id)
    print(chunk.content)
    print(chunk.metadata)
```

## Choosing a Retriever

The `InMemoryRetriever` needs no extra dependency and is convenient for small examples. `FAISSRetriever` and `ChromaRetriever` implement the same retriever interface; install the matching optional extra before constructing either one:

```python
from ragframework.retriever import FAISSRetriever

retriever = FAISSRetriever()
```

```python
from ragframework.retriever import ChromaRetriever

retriever = ChromaRetriever()  # ephemeral collection
# Or persist the collection to disk:
# retriever = ChromaRetriever(persist_directory="./chroma-data")
```

Pass the selected retriever to `RAGPipeline` in place of `InMemoryRetriever`.

## Understanding the Core ABCs

The abstract base classes in `ragframework.base` define the interfaces implemented by pipeline components:

- `DocumentLoader.load(source)` returns documents.
- `TextChunker.chunk(document)` returns chunks.
- `Embedder.embed(texts)` returns one vector per text.
- `Retriever.add(chunks)` indexes embedded chunks, and `Retriever.retrieve(query_embedding, top_k=5)` returns relevant chunks.
- `Generator.generate(query, context)` returns an answer string.

`RAGPipeline` connects these components; `RAGConfig` holds settings such as chunk size, overlap, and default `top_k`.
