"""CUAD ingest helpers.

Reusable functions the Databricks notebooks orchestrate (notebooks/01_ingest.py). Keep Spark/UC
I/O in the notebook; keep pure transforms here so they're testable off-cluster.
"""
from __future__ import annotations

from typing import Any, Iterable

from ..config import CorpusConfig


def load_cuad(cfg: CorpusConfig) -> Iterable[dict[str, Any]]:
    """Yield raw CUAD records from Hugging Face.

    Hint: datasets.load_dataset(cfg.hf_dataset, split=cfg.hf_split); apply cfg.limit; yield dicts.
    """
    # TODO: load from HF and yield raw records (defer Spark — return plain dicts).
    raise NotImplementedError


def to_bronze_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw CUAD record into the Bronze schema.

    Hint: extract contract_id/title, full contract text, and keep clause annotations as a nested
    field for later golden-set derivation. Return a flat-ish dict ready for a Delta write.
    """
    # TODO: map raw fields -> {contract_id, title, text, annotations, source, ingested_at}.
    raise NotImplementedError
