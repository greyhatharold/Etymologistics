"""
RAG (Retrieval-Augmented Generation) pipeline for etymology data.

This package provides a lightweight, extensible system for storing and
retrieving etymological data, embeddings, and references.
"""

from .rag_pipeline import (
    RAGPipeline,
    EtymologyDocument,
    RetrievalResult,
    StorageBackend,
    ChromaBackend
)

__version__ = "0.1.0"
__all__ = [
    "RAGPipeline",
    "EtymologyDocument",
    "RetrievalResult",
    "StorageBackend",
    "ChromaBackend"
] 