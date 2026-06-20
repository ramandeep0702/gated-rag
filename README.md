# gated-rag

Contract-intelligence RAG over [CUAD](https://www.atticusprojectai.org/cuad)
(Contract Understanding Atticus Dataset) with **evaluation gates**.

The point isn't "it answers." The point is that retrieval quality is **measured and
enforced**: a run *fails* (non-zero exit) if recall@k / MRR / groundedness fall below
configured thresholds — the same way a data pipeline fails its data-quality checks.
The CUAD expert clause annotations give a ready-made golden eval set
(question = clause-category prompt, answer = annotated span), so the gate is grounded
in expert labels, not vibes.

> **Status: core implemented.** The full local path runs end-to-end — ingest → chunk → embed →
> FAISS retrieval → eval gate — and is covered by unit tests + CI. The Databricks Vector Search
> backend remains optional (`# VERIFY` entitlement). See *Quickstart* below.

## Architecture

- **Delta medallion in Unity Catalog**: Bronze (raw contracts + metadata) → Silver
  (cleaned, chunked) → Gold (embedded chunks).
- **Retrieval behind a swappable interface**, selected by config:
  1. **FAISS** (default) — local, runs on Databricks Free Edition for sure.
  2. **Databricks Vector Search** (optional) — used *if* the workspace supports it. `# VERIFY`
- Retrieval returns chunks **with citations** (contract id + clause span), and is **scoped
  per-contract** (`query(..., filter_contract_id=...)`) — the realistic "ask questions about THIS
  contract" path, and far better than global search on CUAD's contract-agnostic questions
  (recall@5 0.50 vs 0.13; see [docs/design.md](docs/design.md) §3.2).
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

## Quickstart (local, no Databricks)

The default path is FAISS + SentenceTransformers, so the whole pipeline and its eval gate run on a
laptop or in CI:

```bash
pip install -e .            # core deps (faiss-cpu, sentence-transformers, datasets, tiktoken, ...)
pip install -e ".[dev]"     # + pytest, ruff

# fast smoke run over a handful of contracts (downloads CUAD + the embedding model on first use)
python -m gated_rag.pipeline --limit 25

# full corpus run (all ~510 contracts)
python -m gated_rag.pipeline
```

The run prints recall@k / MRR / groundedness, then **exits non-zero if any gate threshold is unmet**
— that exit code is the whole point. Unit tests (no network needed) run with `pytest`.

## Config

Everything is driven by [`configs/default.yaml`](configs/default.yaml). Swapping the
retriever backend, embedding model, chunk size, or eval thresholds is a config edit,
not a code change.

## Free Edition notes

Anything that may not run on Databricks Free Edition is flagged `# VERIFY` in code/config:
Vector Search entitlement, Unity Catalog catalog naming, serverless, and MLflow tracking.
The FAISS + SentenceTransformers default path is chosen to avoid those dependencies.

Resolved so far (via [`notebooks/00_verify_env.py`](notebooks/00_verify_env.py) on a real Free
Edition workspace): there is **no `main` catalog** — the writable catalog is **`workspace`**, so
`catalog.catalog` is set accordingly.
