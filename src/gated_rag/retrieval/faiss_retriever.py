"""FAISS-backed Retriever. Default backend — local, exact, runs on Free Edition for sure."""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import numpy as np

from ..config import RetrievalConfig
from ..embedding.base import Embedder
from .base import RetrievedChunk, Retriever

_INDEX_FILE = "index.faiss"
_META_FILE = "meta.json"


class FaissRetriever(Retriever):
    def __init__(self, cfg: RetrievalConfig, embedder: Embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self._index = None              # faiss index; built in index()/load()
        self._meta: list[dict] = []     # parallel to vectors: contract_id, chunk_id, text, char_span

    def _new_index(self, dim: int):
        import faiss

        if self.cfg.faiss.metric == "ip":
            return faiss.IndexFlatIP(dim)   # inner product; cosine when vectors are normalized
        return faiss.IndexFlatL2(dim)

    def index(self, gold_chunks: Iterable[dict]) -> None:
        """Build the index from gold chunks and record citation metadata in insertion order."""
        chunks = list(gold_chunks)
        if not chunks:
            raise ValueError("FaissRetriever.index got no chunks")

        mat = np.vstack([np.asarray(c["embedding"], dtype=np.float32) for c in chunks])
        self._index = self._new_index(mat.shape[1])
        self._index.add(mat)
        self._meta = [
            {
                "contract_id": c["contract_id"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "char_span": tuple(c["char_span"]) if c.get("char_span") else None,
            }
            for c in chunks
        ]

    def query(self, text: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        """Embed `text`, search, and return up to top_k chunks WITH citations."""
        if self._index is None:
            raise RuntimeError("index is empty; call index() or load() first")

        k = min(top_k or self.cfg.top_k, len(self._meta))
        vec = self.embedder.embed([text]).astype(np.float32)
        scores, ids = self._index.search(vec, k)

        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:  # faiss pads with -1 when fewer than k results exist
                continue
            m = self._meta[idx]
            out.append(
                RetrievedChunk(
                    contract_id=m["contract_id"],
                    chunk_id=m["chunk_id"],
                    text=m["text"],
                    score=float(score),
                    char_span=tuple(m["char_span"]) if m["char_span"] else None,
                )
            )
        return out

    def persist(self) -> None:
        """Write the FAISS index + a JSON metadata sidecar to cfg.faiss.persist_path."""
        import faiss

        if self._index is None:
            raise RuntimeError("nothing to persist; build the index first")
        path = self.cfg.faiss.persist_path
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, _INDEX_FILE))
        with open(os.path.join(path, _META_FILE), "w", encoding="utf-8") as f:
            json.dump(self._meta, f)

    def load(self) -> None:
        """Load a previously persisted index + metadata sidecar from cfg.faiss.persist_path."""
        import faiss

        path = self.cfg.faiss.persist_path
        self._index = faiss.read_index(os.path.join(path, _INDEX_FILE))
        with open(os.path.join(path, _META_FILE), "r", encoding="utf-8") as f:
            self._meta = json.load(f)
