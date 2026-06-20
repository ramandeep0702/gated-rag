"""Embedder interface. Implementations are selected by embedding.backend via build_embedder()."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    """Turns text into vectors. Swap the impl in config without touching callers."""

    @abstractmethod
    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Return a (len(texts), dim) float32 array. Honor batch_size and the normalize flag.

        is_query distinguishes search queries from indexed passages: some models (e.g. BGE) need a
        query-side instruction prefix for asymmetric retrieval, so passages and queries must be
        embedded differently. Passage embedding is the default (is_query=False).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality. Must equal embedding.dim in config and the index dimension."""
        raise NotImplementedError
