"""Retrieval package: interface + config-driven factory.

build_retriever(cfg, embedder) is the seam: FAISS default, Vector Search optional. Swapping is a
config edit, not a rewrite.
"""
from __future__ import annotations

from ..config import RetrievalConfig
from ..embedding.base import Embedder
from .base import RetrievedChunk, Retriever


def build_retriever(cfg: RetrievalConfig, embedder: Embedder) -> Retriever:
    """Resolve retrieval.backend -> concrete Retriever. Impls imported lazily per branch."""
    if cfg.backend == "faiss":
        from .faiss_retriever import FaissRetriever

        return FaissRetriever(cfg, embedder)
    if cfg.backend == "vector_search":
        from .vector_search_retriever import VectorSearchRetriever  # VERIFY entitlement

        return VectorSearchRetriever(cfg, embedder)
    raise ValueError(f"unknown retrieval.backend: {cfg.backend!r}")


__all__ = ["Retriever", "RetrievedChunk", "build_retriever"]
