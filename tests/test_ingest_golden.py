"""Ingest normalization + golden-set derivation (pure transforms, no network)."""
from __future__ import annotations

from gated_rag.eval.golden import derive_golden
from gated_rag.ingest import aggregate_contracts, to_bronze_record


def raw(title, question, texts, starts):
    return {
        "id": f"{title}__{question}",
        "title": title,
        "context": "CONTEXT " + title,
        "question": question,
        "answers": {"text": texts, "answer_start": starts},
    }


def test_to_bronze_record_builds_spans():
    rec = to_bronze_record(raw("C1", "Governing law?", ["New York"], [10]))
    assert rec["contract_id"] == "C1"
    ann = rec["annotations"][0]
    assert ann["question"] == "Governing law?"
    assert ann["answer_spans"] == [(10, 18)]  # 10 + len("New York")


def test_aggregate_dedupes_and_gathers_annotations():
    recs = [
        to_bronze_record(raw("C1", "Q1", ["a"], [0])),
        to_bronze_record(raw("C1", "Q2", ["bb"], [5])),
        to_bronze_record(raw("C2", "Q1", [], [])),
    ]
    rows = aggregate_contracts(recs)
    assert len(rows) == 2  # C1, C2
    c1 = next(r for r in rows if r["contract_id"] == "C1")
    assert len(c1["annotations"]) == 2


def test_derive_golden_skips_unanswered_clauses():
    recs = [
        to_bronze_record(raw("C1", "Q1", ["a"], [0])),     # answered
        to_bronze_record(raw("C1", "Q2", [], [])),          # clause absent -> dropped
    ]
    golden = derive_golden(aggregate_contracts(recs))
    assert len(golden) == 1
    assert golden[0]["question"] == "Q1"
    assert golden[0]["answer_spans"] == [[0, 1]]
