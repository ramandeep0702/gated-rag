"""Config loader + validation."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from gated_rag.config import load_config, validate

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


def test_load_default_config():
    cfg = load_config(CONFIG_PATH)
    assert cfg.corpus.hf_dataset == "theatticusproject/cuad-qa"
    assert cfg.retrieval.backend == "faiss"
    assert cfg.chunking.strategy in {"token", "structure"}
    assert cfg.embedding.dim == 384
    assert cfg.eval.gate.enabled is True
    assert cfg.raw, "raw YAML should be retained for round-tripping"


def test_validate_rejects_unknown_backend():
    cfg = load_config(CONFIG_PATH)
    bad = dataclasses.replace(cfg.retrieval, backend="pinecone")
    cfg = dataclasses.replace(cfg, retrieval=bad)
    with pytest.raises(ValueError, match="retrieval.backend"):
        validate(cfg)


def test_validate_rejects_overlap_ge_max_tokens():
    cfg = load_config(CONFIG_PATH)
    bad = dataclasses.replace(cfg.chunking, overlap_tokens=cfg.chunking.max_tokens)
    cfg = dataclasses.replace(cfg, chunking=bad)
    with pytest.raises(ValueError, match="overlap_tokens"):
        validate(cfg)


def test_validate_rejects_ip_without_normalize():
    cfg = load_config(CONFIG_PATH)
    bad_emb = dataclasses.replace(cfg.embedding, normalize=False)
    cfg = dataclasses.replace(cfg, embedding=bad_emb)
    with pytest.raises(ValueError, match="normalize"):
        validate(cfg)


def test_validate_rejects_unknown_threshold_metric():
    cfg = load_config(CONFIG_PATH)
    bad_gate = dataclasses.replace(cfg.eval.gate, thresholds={"f1_score": 0.9})
    bad_eval = dataclasses.replace(cfg.eval, gate=bad_gate)
    cfg = dataclasses.replace(cfg, eval=bad_eval)
    with pytest.raises(ValueError, match="unrecognized metric"):
        validate(cfg)
