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


PARQUET_REVISION = "refs/convert/parquet"   # the Hub's auto-converted, script-free copy of CUAD


def load_cuad(cfg: CorpusConfig) -> Iterable[dict[str, Any]]:
    """Yield raw CUAD QA records from Hugging Face.

    Each record: {id, title, context, question, answers:{text:[...], answer_start:[...]}}.
    cfg.limit caps the number of distinct *contracts* (not QA rows), matching the config's intent
    — all clause questions for the first N contracts are yielded.
    """
    seen: set[str] = set()
    for rec in _iter_cuad_records(cfg):
        title = rec["title"]
        if cfg.limit is not None and title not in seen and len(seen) >= cfg.limit:
            break
        seen.add(title)
        yield dict(rec)


def _iter_cuad_records(cfg: CorpusConfig) -> Iterable[dict[str, Any]]:
    """Raw CUAD QA dicts, robust across datasets / huggingface_hub versions.

    Tries datasets.load_dataset first; on any failure — the script-loader removal in datasets>=3,
    or the fsspec/hf_hub glob incompatibility seen on Databricks runtimes — falls back to reading
    the Hub's auto-converted parquet shards directly (stable API, no dataset-module glob).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
        for rec in ds:
            yield dict(rec)
        return
    except Exception:
        pass
    yield from _iter_cuad_parquet(cfg)


def _iter_cuad_parquet(cfg: CorpusConfig) -> Iterable[dict[str, Any]]:
    """Download the parquet shards for the split and yield row dicts via pandas/pyarrow."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(cfg.hf_dataset, repo_type="dataset", revision=PARQUET_REVISION)
    prefix = f"default/{cfg.hf_split}/"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards under {prefix} on {cfg.hf_dataset}@{PARQUET_REVISION}")
    for shard in shards:
        local = hf_hub_download(cfg.hf_dataset, shard, repo_type="dataset", revision=PARQUET_REVISION)
        for rec in pd.read_parquet(local).to_dict(orient="records"):
            yield rec


def _spans_from_answers(answers: Any) -> list[tuple[int, int]]:
    """CUAD `answers` -> list of (start, end) char spans into the contract `context`.

    Tolerates both the datasets shape (dict of python lists) and the pandas/pyarrow shape (dict of
    numpy arrays), so truthiness is never taken on an array.
    """
    if answers is None:
        return []
    texts = answers["text"] if "text" in answers else None
    starts = answers["answer_start"] if "answer_start" in answers else None
    texts = list(texts) if texts is not None else []
    starts = list(starts) if starts is not None else []
    spans: list[tuple[int, int]] = []
    for txt, start in zip(texts, starts):
        if txt and start is not None and int(start) >= 0:
            spans.append((int(start), int(start) + len(str(txt))))
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
