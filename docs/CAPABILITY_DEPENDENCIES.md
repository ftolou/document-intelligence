# Capability dependency extras

The Python package keeps its generic typed contracts lightweight and exposes optional dependencies by capability. Install only the capability needed by the consuming runtime instead of copying dependencies from the full local application requirements.

## Base package

```bash
python -m pip install document-intelligence-pipeline
```

The base package declares `pydantic`, which is used directly by generic application and extraction contracts.

## Semantic retrieval and embeddings

```bash
python -m pip install "document-intelligence-pipeline[semantic]"
```

This extra supports the provider-neutral semantic indexing/retrieval surface and the current HTTP embedding adapters. It includes NumPy because the generic semantic retriever performs vector scoring with NumPy. It intentionally does not install local OCR/GPU packages.

Representative import surface:

```python
from receipt_intelligence.adapters.embeddings.factory import (
    EmbeddingProviderConfig,
    build_embedding_gateway,
)
from receipt_intelligence.rag.item_indexer import ItemEmbeddingIndexer
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever
```

## RAG-SQL orchestration

```bash
python -m pip install "document-intelligence-pipeline[rag-sql]"
```

This extra adds LangGraph for the generic RAG-SQL orchestration adapter and NumPy for the semantic candidate-resolution path reused by query-filter resolution. It remains separate from the HTTP embedding-provider adapters, so consumers that only need RAG-SQL orchestration do not install those adapters' transport dependencies.

## OpenAI one-shot extraction

```bash
python -m pip install "document-intelligence-pipeline[openai-extraction]"
```

This extra supports the optional OpenAI one-shot receipt extraction entry point while keeping the local Paddle/Qwen/Gemma runtime out of lightweight deployments.

Representative import surface:

```python
from receipt_intelligence.extraction.config import ExtractionRequest
from receipt_intelligence.extraction.openai_one_shot import run_openai_one_shot_extraction
from receipt_intelligence.pipeline.integrated_receipt_pipeline import run_receipt_extraction
```

The extra intentionally does not install PaddlePaddle, PaddleOCR, OpenCV, NumPy, LangGraph, or Flask. The local reference application continues to use `requirements/app.txt` for its full runtime stack.

## Development validation

Core CI builds the package and installs each capability extra in a fresh runner before importing its documented surface. This is the compatibility contract for downstream consumers: a capability extra must declare the packages required by that capability instead of relying on consumer-side installation order or undeclared transitive dependencies.
