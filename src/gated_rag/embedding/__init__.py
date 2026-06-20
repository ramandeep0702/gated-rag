"""Embedding package: interface + config-driven factory.

build_embedder(cfg) is the only thing callers need — swapping backends is a config edit.
"""
from __future__ import annotations

from ..config import EmbeddingConfig
from .base import Embedder


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Resolve embedding.backend -> concrete Embedder. Impls imported lazily per branch."""
    if cfg.backend == "sentence_transformers":
        from .sentence_transformers_embedder import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(cfg)
    raise ValueError(f"unknown embedding.backend: {cfg.backend!r}")


__all__ = ["Embedder", "build_embedder"]
