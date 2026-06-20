"""FAISS-backed Retriever. Default backend — local, exact, runs on Free Edition for sure."""
from __future__ import annotations

from typing import Iterable, Optional

from ..config import RetrievalConfig
from ..embedding.base import Embedder
from .base import RetrievedChunk, Retriever


class FaissRetriever(Retriever):
    def __init__(self, cfg: RetrievalConfig, embedder: Embedder) -> None:
        # TODO: store cfg + embedder; init self._index = None and a parallel list of chunk metadata.
        # Hint: keep id->metadata (contract_id, chunk_id, text, char_span) alongside the vector index,
        #       since FAISS only stores vectors + integer ids.
        raise NotImplementedError

    def index(self, gold_chunks: Iterable[dict]) -> None:
        # TODO: stack embeddings into a float32 matrix; build index per cfg.faiss.index_type/metric
        #       (flat + ip for normalized cosine); add vectors; record metadata in insertion order.
        raise NotImplementedError

    def query(self, text: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        # TODO: k = top_k or cfg.top_k; vec = embedder.embed([text]); index.search(vec, k);
        #       map ids -> metadata -> RetrievedChunk(score=...). Returns chunks WITH citations.
        raise NotImplementedError

    def persist(self) -> None:
        # TODO: faiss.write_index to cfg.faiss.persist_path; dump metadata sidecar (json/parquet).
        raise NotImplementedError

    def load(self) -> None:
        # TODO: faiss.read_index from cfg.faiss.persist_path; load metadata sidecar.
        raise NotImplementedError
