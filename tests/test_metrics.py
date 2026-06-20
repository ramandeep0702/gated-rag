"""Retrieval-quality metrics + the threshold gate."""
from __future__ import annotations

import pytest

from gated_rag.config import GateConfig
from gated_rag.eval import metrics as M
from gated_rag.eval.gate import enforce, evaluate_gate
from gated_rag.retrieval.base import RetrievedChunk


def chunk(contract_id, span, score=1.0):
    return RetrievedChunk(
        contract_id=contract_id, chunk_id=f"{contract_id}:{span[0]}",
        text="x", score=score, char_span=span,
    )


GOLDEN_CID = "C1"
GOLDEN_SPANS = [(100, 200)]


def test_is_relevant_requires_contract_and_overlap():
    assert M.is_relevant(chunk("C1", (150, 180)), GOLDEN_CID, GOLDEN_SPANS) is True
    # right span, wrong contract
    assert M.is_relevant(chunk("C2", (150, 180)), GOLDEN_CID, GOLDEN_SPANS) is False
    # right contract, no overlap
    assert M.is_relevant(chunk("C1", (300, 400)), GOLDEN_CID, GOLDEN_SPANS) is False
    # touching boundaries do not overlap (half-open intervals)
    assert M.is_relevant(chunk("C1", (200, 250)), GOLDEN_CID, GOLDEN_SPANS) is False


def test_recall_at_k_honors_cutoff():
    results = [chunk("C2", (0, 10)), chunk("C2", (0, 10)), chunk("C1", (150, 160))]
    assert M.recall_at_k(results, GOLDEN_CID, GOLDEN_SPANS, k=2) == 0.0
    assert M.recall_at_k(results, GOLDEN_CID, GOLDEN_SPANS, k=3) == 1.0


def test_reciprocal_rank_is_one_over_first_hit():
    results = [chunk("C2", (0, 10)), chunk("C1", (150, 160))]
    assert M.reciprocal_rank(results, GOLDEN_CID, GOLDEN_SPANS) == pytest.approx(0.5)
    assert M.reciprocal_rank([], GOLDEN_CID, GOLDEN_SPANS) == 0.0


def test_groundedness_checks_top1_only():
    hit_first = [chunk("C1", (150, 160)), chunk("C2", (0, 10))]
    hit_second = [chunk("C2", (0, 10)), chunk("C1", (150, 160))]
    assert M.groundedness(hit_first, GOLDEN_CID, GOLDEN_SPANS) == 1.0
    assert M.groundedness(hit_second, GOLDEN_CID, GOLDEN_SPANS) == 0.0


def _gate(**thresholds):
    return GateConfig(enabled=True, on_failure="fail", thresholds=thresholds)


def test_evaluate_gate_pass_and_fail():
    measured = {"recall_at_5": 0.9, "mrr": 0.7, "groundedness": 0.8}
    assert evaluate_gate(measured, _gate(recall_at_5=0.8, mrr=0.65)).passed is True

    failing = evaluate_gate(measured, _gate(recall_at_5=0.95))
    assert failing.passed is False
    assert "recall_at_5" in failing.failures


def test_evaluate_gate_flags_missing_metric():
    result = evaluate_gate({"mrr": 0.7}, _gate(groundedness=0.7))
    assert result.passed is False
    assert "groundedness" in result.failures


def test_enforce_raises_on_hard_failure():
    result = evaluate_gate({"recall_at_5": 0.1}, _gate(recall_at_5=0.8))
    with pytest.raises(SystemExit):
        enforce(result, _gate(recall_at_5=0.8))


def test_enforce_warns_but_continues():
    cfg = GateConfig(enabled=True, on_failure="warn", thresholds={"recall_at_5": 0.8})
    result = evaluate_gate({"recall_at_5": 0.1}, cfg)
    enforce(result, cfg)  # must not raise
