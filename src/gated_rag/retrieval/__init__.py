"""Retrieval package: interface + config-driven factory.

build_retriever(cfg, embedder) is the seam: FAISS default, Vector Search optional. Swapping is a
config edit, not a rewrite.
"""
from __future__ import annotations

from ..config import RetrievalConfig
from ..embedding.base import Embedder
from .base import RetrievedChunk, Retriever


def build_retriever(cfg: RetrievalConfig, embedder: Embedder) -> Retriever:
    """Resolve retrieval.backend -> concrete Retriever.

    Hint: 'faiss' -> FaissRetriever; 'vector_search' -> VectorSearchRetriever (# VERIFY entitlement);
    else raise ValueError. Import impls lazily inside the branch so databricks-vectorsearch stays optional.
    """
    # TODO: dispatch on cfg.backend.
    raise NotImplementedError


__all__ = ["Retriever", "RetrievedChunk", "build_retriever"]
