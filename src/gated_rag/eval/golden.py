"""Derive the golden eval set from CUAD annotations.

A golden item is {question, contract_id, answer_spans}. We keep only annotations that actually have
a span — CUAD includes clause questions with no answer (the clause is absent from that contract),
and those make no sense as retrieval targets. The golden set is the ground truth the eval gate
scores against, so it lives next to the metrics.
"""
from __future__ import annotations

import json
from typing import Any, Iterable


def derive_golden(bronze_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bronze contract rows -> golden items, one per (contract, answered clause question)."""
    golden: list[dict[str, Any]] = []
    for row in bronze_rows:
        cid = row["contract_id"]
        for ann in row.get("annotations", []):
            spans = ann.get("answer_spans") or []
            if not spans:
                continue
            golden.append(
                {
                    "question": ann["question"],
                    "contract_id": cid,
                    "answer_spans": [[int(s), int(e)] for s, e in spans],
                }
            )
    return golden


def write_golden(golden: Iterable[dict[str, Any]], path: str) -> None:
    """Write the golden set as JSON Lines (one item per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in golden:
            f.write(json.dumps(item) + "\n")


def load_golden(path: str) -> list[dict[str, Any]]:
    """Read a JSON Lines golden set."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
