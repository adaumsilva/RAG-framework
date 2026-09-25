\# Getting Started



RAG Framework is a modular Python framework for building Retrieval-Augmented Generation (RAG) pipelines.



It provides simple abstractions for loading documents, splitting them into chunks, generating embeddings, retrieving relevant chunks, and generating answers.



\## Installation



\### Requirements



RAG Framework requires Python 3.10 or newer.



For development, create and activate a virtual environment:



```bash

python -m venv .venv

```



On Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install the framework with development dependencies:



```bash

pip install -e ".\[dev]"

```



The core framework only requires NumPy. Optional integrations can be installed for additional functionality.



\### Optional Dependencies



Install PDF support:



```bash

pip install -e ".\[pdf]"

```



Install DOCX support:



```bash

pip install -e ".\[docx]"

```



Install Hugging Face support:



```bash

pip install -e ".\[huggingface]"

```



Install ChromaDB support:



```bash

pip install -e ".\[chromadb]"

```



Install FAISS support:



```bash

pip install -e ".\[faiss]"

```



To install all optional integrations:



```bash

pip install -e ".\[all]"

```



\## Quickstart



The repository includes a built-in example that runs without an API key.



Run:



```bash

python examples/basic\_rag.py

```



The example demonstrates the complete RAG workflow:



1\. Load a document.

2\. Split the document into chunks.

3\. Generate embeddings.

4\. Add chunks to the retriever.

5\. Query the pipeline.

6\. Generate an answer from the retrieved context.

## Logging

RAG Framework uses Python's standard-library logging and is silent by default.
Enable pipeline stage summaries with:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Use `logging.DEBUG` to include per-batch embedding and retriever indexing details.

\## Building a Pipeline



A RAG pipeline is created by providing implementations for the main components:



```python

from ragframework.config import RAGConfig

from ragframework.document.chunkers import FixedSizeChunker

from ragframework.document.loaders import TextFileLoader

from ragframework.embeddings.random\_embedder import RandomEmbedder

from ragframework.generator.echo\_generator import EchoGenerator

from ragframework.pipeline.rag import RAGPipeline

from ragframework.retriever.in\_memory import InMemoryRetriever



pipeline = RAGPipeline(

&#x20;   loader=TextFileLoader(),

&#x20;   chunker=FixedSizeChunker(chunk\_size=200, chunk\_overlap=40),

&#x20;   embedder=RandomEmbedder(dim=64, seed=42),

&#x20;   retriever=InMemoryRetriever(),

&#x20;   generator=EchoGenerator(),

&#x20;   config=RAGConfig(top\_k=3),

)

```



The built-in `RandomEmbedder` and `EchoGenerator` are intended for examples and testing. They do not provide production-quality semantic retrieval or LLM-generated answers.



\## Loading Documents



`DocumentLoader` defines the interface for loading source content.



The built-in text loader can load a text file:



```python

loader = TextFileLoader()

documents = loader.load("example.txt")

```



A loader returns one or more `Document` objects containing the document content and metadata.



\## Chunking



Long documents are divided into smaller pieces before embedding.



The framework provides `FixedSizeChunker`:



```python

chunker = FixedSizeChunker(

&#x20;   chunk\_size=200,

&#x20;   chunk\_overlap=40,

)

```



The `chunk\_size` controls the approximate size of each chunk, while `chunk\_overlap` allows neighboring chunks to share some content.



\## Embeddings



Embeddings convert text into numerical vectors that can be used for similarity-based retrieval.



The built-in example uses:



```python

embedder = RandomEmbedder(dim=64, seed=42)

```



`RandomEmbedder` is useful for demonstrating the pipeline without requiring an external service or API key.



For meaningful semantic retrieval, replace it with a real embedding implementation when an integration is available.



\## Ingestion



Documents are loaded, chunked, embedded, and added to the retriever during ingestion:



```python

n\_chunks = pipeline.ingest("example.txt")

print(f"Ingested {n\_chunks} chunks")

```



The return value is the number of chunks ingested into the pipeline.



\## Querying



After ingestion, query the pipeline:



```python

response = pipeline.query(

&#x20;   "What are the stages of a RAG pipeline?"

)

```



The returned `RAGResponse` contains the generated answer and the chunks retrieved as context.



```python

print(response.answer)

print(len(response.source\_chunks))

```



\## Inspecting Retrieved Results



Retrieved chunks can be inspected through `source\_chunks`:



```python

for chunk in response.source\_chunks:

&#x20;   print(chunk.id)

&#x20;   print(chunk.content)

```



This is useful for understanding which parts of the source documents were provided as context for the answer.



\## Understanding the Core ABCs



RAG Framework uses abstract base classes (ABCs) to keep the pipeline modular.



\### DocumentLoader



Responsible for loading source documents.



```python

DocumentLoader.load(source)

```



\### TextChunker



Responsible for splitting documents into chunks.



```python

TextChunker.chunk(document)

```



\### Embedder



Responsible for converting text into embedding vectors.



```python

Embedder.embed(texts)

```



\### Retriever



Responsible for indexing chunks and retrieving relevant chunks for a query.



```python

Retriever.add(chunks)

Retriever.retrieve(query\_embedding, top\_k)

```



\### Generator



Responsible for generating an answer using the query and retrieved context.



```python

Generator.generate(query, context)

```
