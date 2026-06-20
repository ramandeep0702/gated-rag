# gated-rag

Contract-intelligence RAG over [CUAD](https://www.atticusprojectai.org/cuad)
(Contract Understanding Atticus Dataset) with **evaluation gates**.

The point isn't "it answers." The point is that retrieval quality is **measured and
enforced**: a run *fails* (non-zero exit) if recall@k / MRR / groundedness fall below
configured thresholds — the same way a data pipeline fails its data-quality checks.
The CUAD expert clause annotations give a ready-made golden eval set
(question = clause-category prompt, answer = annotated span), so the gate is grounded
in expert labels, not vibes.

> **Status: skeleton.** Logic is intentionally stubbed (`TODO`). This repo defines the
> structure, interfaces, and config surface; the bodies are written incrementally.

## Architecture

- **Delta medallion in Unity Catalog**: Bronze (raw contracts + metadata) → Silver
  (cleaned, chunked) → Gold (embedded chunks).
- **Retrieval behind a swappable interface**, selected by config:
  1. **FAISS** (default) — local, runs on Databricks Free Edition for sure.
  2. **Databricks Vector Search** (optional) — used *if* the workspace supports it. `# VERIFY`
- Retrieval returns chunks **with citations** (contract id + clause span).
- **Eval harness** scores recall@k, MRR, and groundedness against the CUAD-derived
  golden set, with **configurable thresholds that fail the run** if unmet (the gate).

## Layout

```
gated-rag/
├── configs/default.yaml        # the one config: corpus, chunking, embedding, retriever, eval gates
├── src/gated_rag/
│   ├── config.py               # typed schema + loader for default.yaml
│   ├── ingest/                 # CUAD load + Bronze helpers (notebooks orchestrate these)
│   ├── chunking/               # token vs structure-aware chunking (see docs/design.md)
│   ├── embedding/              # Embedder interface + SentenceTransformers impl
│   ├── retrieval/              # Retriever interface + FAISS (default) + Vector Search (optional)
│   └── eval/                   # recall@k / MRR / groundedness + the threshold gate
├── notebooks/                  # Databricks-flavored orchestration (01_ingest.py, ...)
├── eval/golden/                # CUAD-derived golden set (jsonl)
└── docs/design.md              # framework/config boundary + chunking decision
```

## Config

Everything is driven by [`configs/default.yaml`](configs/default.yaml). Swapping the
retriever backend, embedding model, chunk size, or eval thresholds is a config edit,
not a code change.

## Free Edition notes

Anything that may not run on Databricks Free Edition is flagged `# VERIFY` in code/config:
Vector Search entitlement, Unity Catalog catalog naming, serverless, and MLflow tracking.
The FAISS + SentenceTransformers default path is chosen to avoid those dependencies.
