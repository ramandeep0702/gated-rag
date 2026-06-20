"""Embedding package: interface + config-driven factory.

build_embedder(cfg) is the only thing callers need — swapping backends is a config edit.
"""
from __future__ import annotations

from ..config import EmbeddingConfig
from .base import Embedder


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Resolve embedding.backend -> concrete Embedder.

    Hint: 'sentence_transformers' -> SentenceTransformersEmbedder(cfg); else raise ValueError.
    Import impls lazily inside the branch to keep optional deps optional.
    """
    # TODO: dispatch on cfg.backend.
    raise NotImplementedError


__all__ = ["Embedder", "build_embedder"]
