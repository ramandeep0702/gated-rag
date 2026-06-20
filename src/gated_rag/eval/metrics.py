"""Retrieval-quality metrics, scored against the CUAD-derived golden set.

A golden item: {question, contract_id, answer_spans: list[(start, end)]}.
A retrieval result for an item: the ordered list[RetrievedChunk] returned by the retriever.
A chunk is "relevant" if it is from the right contract AND its char_span overlaps a golden span.

Each function scores ONE golden item; the caller (pipeline) averages across the golden set.
"""
from __future__ import annotations

from typing import Sequence

from ..retrieval.base import RetrievedChunk


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if half-open intervals [a0,a1) and [b0,b1) intersect."""
    return a[0] < b[1] and b[0] < a[1]


def is_relevant(chunk: RetrievedChunk, golden_contract_id: str,
                golden_spans: Sequence[tuple[int, int]]) -> bool:
    """True if chunk is from the golden contract and overlaps any golden answer span."""
    if chunk.contract_id != golden_contract_id or chunk.char_span is None:
        return False
    span = (int(chunk.char_span[0]), int(chunk.char_span[1]))
    return any(_overlaps(span, (int(g[0]), int(g[1]))) for g in golden_spans)


def recall_at_k(results: Sequence[RetrievedChunk], golden_contract_id: str,
                golden_spans: Sequence[tuple[int, int]], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return 1.0 if any(
        is_relevant(c, golden_contract_id, golden_spans) for c in results[:k]
    ) else 0.0


def reciprocal_rank(results: Sequence[RetrievedChunk], golden_contract_id: str,
                    golden_spans: Sequence[tuple[int, int]]) -> float:
    """1 / rank of the first relevant result (rank starts at 1); 0.0 if none."""
    for i, c in enumerate(results):
        if is_relevant(c, golden_contract_id, golden_spans):
            return 1.0 / (i + 1)
    return 0.0


def groundedness(results: Sequence[RetrievedChunk], golden_contract_id: str,
                 golden_spans: Sequence[tuple[int, int]]) -> float:
    """Does the TOP-1 retrieved chunk actually contain the answer? char-span overlap, no LLM judge.

    # VERIFY: swap to an LLM-judged variant later if you want answer-faithfulness, not just overlap.
    """
    if not results:
        return 0.0
    return 1.0 if is_relevant(results[0], golden_contract_id, golden_spans) else 0.0
