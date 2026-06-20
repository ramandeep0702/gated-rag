"""FaissRetriever — global vs per-contract scoped search. Skipped where faiss isn't installed."""
from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # noqa: F841  (skip whole module if backend absent)

from gated_rag.config import FaissConfig, RetrievalConfig
from gated_rag.embedding.base import Embedder
from gated_rag.retrieval.faiss_retriever import FaissRetriever


class FakeEmbedder(Embedder):
    """Maps known query strings to fixed unit vectors; no model download."""

    def __init__(self, table):
        self._table = table

    def embed(self, texts, is_query=False):
        return np.vstack([self._table[t] for t in texts]).astype(np.float32)

    @property
    def dim(self):
        return 2


def _cfg():
    return RetrievalConfig(
        backend="faiss", top_k=3,
        faiss=FaissConfig(index_type="flat", metric="ip", persist_path="./.faiss"),
        vector_search=None,
    )


def _unit(x, y):
    v = np.array([x, y], dtype=np.float32)
    return v / np.linalg.norm(v)


def _gold(cid, chunk, vec, span):
    return {"contract_id": cid, "chunk_id": f"{cid}:{chunk}", "text": chunk,
            "char_span": span, "embedding": vec}


def test_scoped_query_restricts_to_contract():
    a_vec, b_vec = _unit(1, 0), _unit(0, 1)
    gold = [
        _gold("A", "a0", a_vec, (0, 10)),
        _gold("B", "b0", b_vec, (0, 10)),
        _gold("B", "b1", _unit(1, 1), (10, 20)),
    ]
    embedder = FakeEmbedder({"q": a_vec})
    r = FaissRetriever(_cfg(), embedder)
    r.index(gold)

    # Global: best match for q (== a_vec) is A's chunk.
    assert r.query("q")[0].contract_id == "A"

    # Scoped to B: must never return A, even though A is the global best.
    scoped = r.query("q", filter_contract_id="B")
    assert scoped and all(c.contract_id == "B" for c in scoped)


def test_scoped_query_unknown_contract_is_empty():
    embedder = FakeEmbedder({"q": _unit(1, 0)})
    r = FaissRetriever(_cfg(), embedder)
    r.index([_gold("A", "a0", _unit(1, 0), (0, 10))])
    assert r.query("q", filter_contract_id="ZZZ") == []
