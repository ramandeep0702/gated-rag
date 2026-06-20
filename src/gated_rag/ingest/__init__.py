"""CUAD ingest helpers.

Reusable functions the Databricks notebooks orchestrate (notebooks/01_ingest.py). Keep Spark/UC
I/O in the notebook; keep pure transforms here so they're testable off-cluster.

CUAD (`theatticusproject/cuad-qa`) is a SQuAD-style QA build: one row per (contract, clause-question),
where `context` is the full contract text and `answers` carries the annotated span(s). The same
contract repeats across ~41 clause questions, so `aggregate_contracts` dedupes to one row per
contract with all annotations gathered — Bronze holds contracts, not QA pairs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from ..config import CorpusConfig

SOURCE = "theatticusproject/cuad-qa"


def load_cuad(cfg: CorpusConfig) -> Iterable[dict[str, Any]]:
    """Yield raw CUAD QA records from Hugging Face (respecting cfg.limit).

    Each record: {id, title, context, question, answers:{text:[...], answer_start:[...]}}.
    The heavy `datasets` import is deferred so importing this module stays cheap in tests.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
    for i, rec in enumerate(ds):
        if cfg.limit is not None and i >= cfg.limit:
            break
        yield dict(rec)


def _spans_from_answers(answers: dict[str, Any]) -> list[tuple[int, int]]:
    """CUAD `answers` -> list of (start, end) char spans into the contract `context`."""
    texts = answers.get("text", []) or []
    starts = answers.get("answer_start", []) or []
    spans: list[tuple[int, int]] = []
    for txt, start in zip(texts, starts):
        if txt and start is not None and start >= 0:
            spans.append((int(start), int(start) + len(txt)))
    return spans


def to_bronze_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw CUAD QA record into a Bronze-shaped dict.

    `annotations` holds this row's single clause annotation as a one-element list; aggregation
    across the contract's rows happens in `aggregate_contracts`.
    """
    return {
        "contract_id": raw["title"],
        "title": raw["title"],
        "text": raw["context"],
        "annotations": [
            {
                "question": raw["question"],
                "answer_spans": _spans_from_answers(raw.get("answers", {})),
            }
        ],
        "source": SOURCE,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def aggregate_contracts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe Bronze records by contract_id, gathering all annotations into one row per contract.

    First occurrence wins for the contract text/metadata; annotation lists are concatenated.
    Returns rows in first-seen order for determinism.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = rec["contract_id"]
        if cid not in by_id:
            by_id[cid] = {**rec, "annotations": list(rec["annotations"])}
        else:
            by_id[cid]["annotations"].extend(rec["annotations"])
    return list(by_id.values())
