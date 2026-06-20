"""Retriever interface + citation-carrying result type.

Backend is chosen by retrieval.backend via build_retriever(). Every result carries enough to cite
the source: contract id + clause/char span.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with everything needed to cite it."""
    contract_id: str
    chunk_id: str
    text: str
    score: float
    char_span: Optional[tuple[int, int]] = None   # offsets into the source contract, for citation


class Retriever(ABC):
    """Vector retrieval behind a stable interface.

    Constructed with an Embedder so query() can take raw text. Backends: FAISS (default),
    Databricks Vector Search (optional).
    """

    @abstractmethod
    def index(self, gold_chunks: Iterable[dict]) -> None:
        """Build/populate the index from gold chunks.

        Expected keys per chunk: contract_id, chunk_id, text, embedding (list[float]), char_span.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, text: str, top_k: Optional[int] = None,
              filter_contract_id: Optional[str] = None) -> list[RetrievedChunk]:
        """Embed `text`, search, and return up to top_k chunks WITH citations.

        top_k defaults to config. When filter_contract_id is set, search is scoped to that
        contract's chunks only — the "ask questions about THIS contract" path, and how the eval
        is scored (CUAD questions are contract-agnostic, so global search is ill-posed).
        """
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        """Persist the index if the backend supports it (no-op for managed backends)."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        """Load a previously persisted index."""
        raise NotImplementedError
