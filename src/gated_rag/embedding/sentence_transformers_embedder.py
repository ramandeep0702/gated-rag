"""SentenceTransformers-backed Embedder. Default backend; CPU-friendly, runs on Free Edition."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import EmbeddingConfig
from .base import Embedder


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        # TODO: store cfg; lazily construct SentenceTransformer(cfg.model) on first use.
        # Hint: defer the heavy import (from sentence_transformers import SentenceTransformer)
        #       so importing this module stays cheap in tests.
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        # TODO: encode in batches of cfg.batch_size; if cfg.normalize, L2-normalize; return float32.
        raise NotImplementedError

    @property
    def dim(self) -> int:
        # TODO: return cfg.dim (assert it matches the loaded model's output dimension).
        raise NotImplementedError
