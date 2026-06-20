"""End-to-end local pipeline: CUAD -> chunk -> embed -> FAISS -> eval gate.

Runs entirely on the local default path (FAISS + SentenceTransformers) with no Databricks
dependency, so the eval gate is reproducible on a laptop or in CI. The Databricks medallion
notebooks orchestrate the same `src/` helpers against Unity Catalog.

Usage:
    python -m gated_rag.pipeline --config configs/default.yaml --limit 25

`--limit` overrides corpus.limit for fast dev loops. Exit code is non-zero when the gate fails.
"""
from __future__ import annotations

import argparse
from statistics import mean
from typing import Any

from .chunking import chunk_contract
from .config import Config, load_config
from .embedding import build_embedder
from .eval import metrics as M
from .eval.gate import evaluate_gate, enforce
from .eval.golden import derive_golden, write_golden
from .ingest import aggregate_contracts, load_cuad, to_bronze_record
from .retrieval import build_retriever
from .retrieval.base import Retriever


def build_corpus(cfg: Config) -> list[dict[str, Any]]:
    """Ingest CUAD and collapse QA rows into one Bronze row per contract."""
    bronze = aggregate_contracts(to_bronze_record(r) for r in load_cuad(cfg.corpus))
    print(f"[ingest] {len(bronze)} contracts")
    return bronze


def build_index(cfg: Config, bronze: list[dict[str, Any]]) -> Retriever:
    """Chunk -> embed -> FAISS index. Returns a ready-to-query retriever."""
    chunks = [
        ch
        for row in bronze
        for ch in chunk_contract(row["contract_id"], row["text"], cfg.chunking)
    ]
    print(f"[chunk]  {len(chunks)} chunks ({cfg.chunking.strategy} strategy)")

    embedder = build_embedder(cfg.embedding)
    vectors = embedder.embed([c.text for c in chunks])
    gold = [
        {
            "contract_id": c.contract_id,
            "chunk_id": c.chunk_id,
            "text": c.text,
            "char_span": c.char_span,
            "embedding": vectors[i],
        }
        for i, c in enumerate(chunks)
    ]

    retriever = build_retriever(cfg.retrieval, embedder)
    retriever.index(gold)
    print(f"[embed]  {len(gold)} vectors indexed in {cfg.retrieval.backend}")
    return retriever


def evaluate(cfg: Config, retriever: Retriever,
             golden: list[dict[str, Any]]) -> dict[str, float]:
    """Run every golden question through the retriever and average the metrics."""
    k_values = cfg.eval.k_values
    max_k = max(k_values)
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    rr: list[float] = []
    gnd: list[float] = []

    for item in golden:
        results = retriever.query(item["question"], top_k=max_k)
        cid, spans = item["contract_id"], item["answer_spans"]
        for k in k_values:
            recalls[k].append(M.recall_at_k(results, cid, spans, k))
        rr.append(M.reciprocal_rank(results, cid, spans))
        gnd.append(M.groundedness(results, cid, spans))

    measured = {f"recall_at_{k}": mean(recalls[k]) for k in k_values}
    measured["mrr"] = mean(rr)
    measured["groundedness"] = mean(gnd)
    return measured


def _report(measured: dict[str, float], cfg: Config) -> None:
    thresholds = cfg.eval.gate.thresholds
    print("\n===== retrieval quality =====")
    for name in sorted(measured):
        thr = thresholds.get(name)
        flag = ""
        if thr is not None:
            flag = "  PASS" if measured[name] >= thr else "  FAIL"
            flag += f" (>= {thr:.2f})"
        print(f"{name:16} {measured[name]:.3f}{flag}")
    print("=============================\n")


def run(config_path: str, limit: int | None = None) -> dict[str, float]:
    """Full pipeline. Raises SystemExit (non-zero) if the gate fails with on_failure='fail'."""
    cfg = load_config(config_path)
    if limit is not None:
        cfg.corpus.limit = limit

    bronze = build_corpus(cfg)
    golden = derive_golden(bronze)
    write_golden(golden, cfg.eval.golden_path)
    print(f"[golden] {len(golden)} questions with annotated spans -> {cfg.eval.golden_path}")

    retriever = build_index(cfg, bronze)
    measured = evaluate(cfg, retriever, golden)
    _report(measured, cfg)

    if cfg.eval.gate.enabled:
        enforce(evaluate_gate(measured, cfg.eval.gate), cfg.eval.gate)
    return measured


def main() -> None:
    p = argparse.ArgumentParser(description="Run the gated-rag eval pipeline locally.")
    p.add_argument("--config", default="configs/default.yaml", help="path to config YAML")
    p.add_argument("--limit", type=int, default=None,
                   help="override corpus.limit (number of contracts) for fast dev runs")
    args = p.parse_args()
    run(args.config, limit=args.limit)


if __name__ == "__main__":
    main()
