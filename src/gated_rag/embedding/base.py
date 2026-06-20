"""Embedder interface. Implementations are selected by embedding.backend via build_embedder()."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    """Turns text into vectors. Swap the impl in config without touching callers."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array. Honor batch_size and the normalize flag."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality. Must equal embedding.dim in config and the index dimension."""
        raise NotImplementedError
