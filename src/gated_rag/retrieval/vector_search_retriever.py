"""Databricks Vector Search Retriever. OPTIONAL backend.

# VERIFY: Vector Search requires an entitlement that may not exist on Databricks Free Edition.
# This impl is only constructed when retrieval.backend == vector_search; the FAISS default path
# must remain fully functional without it.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..config import RetrievalConfig
from ..embedding.base import Embedder
from .base import RetrievedChunk, Retriever


class VectorSearchRetriever(Retriever):
    def __init__(self, cfg: RetrievalConfig, embedder: Embedder) -> None:
        # TODO: store cfg + embedder; create a VectorSearchClient lazily.
        # Hint: from databricks.vectorsearch.client import VectorSearchClient  # VERIFY import path/availability
        raise NotImplementedError

    def index(self, gold_chunks: Iterable[dict]) -> None:
        # TODO: ensure endpoint cfg.vector_search.endpoint_name + index cfg.vector_search.index_name exist;
        #       upsert chunks. If embedding_source == precomputed, push our vectors; if managed, push text.
        raise NotImplementedError

    def query(self, text: str, top_k: Optional[int] = None,
              filter_contract_id: Optional[str] = None) -> list[RetrievedChunk]:
        # TODO: k = top_k or cfg.top_k; if precomputed, embed text then similarity_search by vector,
        #       else query by text; when filter_contract_id is set, pass it as a metadata filter
        #       (filters={"contract_id": filter_contract_id}); map results -> RetrievedChunk WITH citations.
        raise NotImplementedError

    def persist(self) -> None:
        # No-op: the index is managed server-side.
        return None

    def load(self) -> None:
        # No-op: the index is managed server-side (just reference it by name).
        return None
