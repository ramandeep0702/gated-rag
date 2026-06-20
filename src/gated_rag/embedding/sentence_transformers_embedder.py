"""SentenceTransformers-backed Embedder. Default backend; CPU-friendly, runs on Free Edition."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import EmbeddingConfig
from .base import Embedder


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self._st = None  # lazily constructed SentenceTransformer (heavy import deferred)

    def _model(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.cfg.model)
        return self._st

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode in batches of cfg.batch_size; L2-normalize if configured; return float32."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.cfg.dim), dtype=np.float32)

        vecs = self._model().encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self.cfg.dim
