"""Retrieval-quality metrics, scored against the CUAD-derived golden set.

A golden item: {question, contract_id, answer_spans: list[(start, end)]}.
A retrieval result for an item: the ordered list[RetrievedChunk] returned by the retriever.
A chunk is "relevant" if it is from the right contract AND its char_span overlaps a golden span.
"""
from __future__ import annotations

from typing import Sequence

from ..retrieval.base import RetrievedChunk


def is_relevant(chunk: RetrievedChunk, golden_contract_id: str,
                golden_spans: Sequence[tuple[int, int]]) -> bool:
    """True if chunk is from the golden contract and overlaps any golden answer span.

    Hint: same contract_id AND any span-overlap(chunk.char_span, golden_span).
    """
    # TODO: contract match + char-span overlap test.
    raise NotImplementedError


def recall_at_k(results: Sequence[RetrievedChunk], golden_contract_id: str,
                golden_spans: Sequence[tuple[int, int]], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0 (averaged over items by the caller).

    Hint: any(is_relevant(c, ...) for c in results[:k]).
    """
    # TODO: top-k relevance hit.
    raise NotImplementedError


def reciprocal_rank(results: Sequence[RetrievedChunk], golden_contract_id: str,
                    golden_spans: Sequence[tuple[int, int]]) -> float:
    """1 / rank of the first relevant result (rank starts at 1); 0.0 if none.

    Hint: find first relevant index i -> 1/(i+1). MRR is the mean of these over items.
    """
    # TODO: first-relevant reciprocal rank.
    raise NotImplementedError


def groundedness(results: Sequence[RetrievedChunk], golden_contract_id: str,
                 golden_spans: Sequence[tuple[int, int]]) -> float:
    """Does the TOP-1 retrieved chunk actually contain the answer? char-span overlap, no LLM judge.

    Hint: is_relevant(results[0], ...) -> 1.0/0.0; mean over items is the corpus groundedness.
    # VERIFY: swap to an LLM-judged variant later if you want answer-faithfulness, not just overlap.
    """
    # TODO: top-1 overlap test.
    raise NotImplementedError
