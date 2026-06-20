# gated-rag — Design

Contract-intelligence RAG over CUAD where **retrieval quality is enforced, not just reported**.
A run fails (non-zero exit) when recall@k / MRR / groundedness fall below configured thresholds —
the same discipline as gating a data pipeline on data-quality checks. The CUAD expert clause
annotations supply the golden set, so the gate is grounded in expert labels.

---

## 1. The framework / config boundary

The rule: **config decides *which* and *how much*; code decides *how*.** Anything you'd reasonably
tune between runs lives in `configs/default.yaml`; the mechanism lives in `src/`.

| Concern | Lives in config (`default.yaml`) | Lives in code (`src/gated_rag`) |
|---|---|---|
| Corpus source / split / limit | ✅ `corpus.*` | loader mechanics (`ingest/`) |
| UC namespace | ✅ `catalog.*` | DDL / write logic (notebook + `ingest/`) |
| Chunk strategy + sizes | ✅ `chunking.*` | the two chunkers (`chunking/`) |
| Embedding model / batch / dim | ✅ `embedding.*` | encode logic (`embedding/`) |
| Retriever backend + params | ✅ `retrieval.*` | the two retrievers (`retrieval/`) |
| Metrics, k, **thresholds**, gate action | ✅ `eval.*` | metric math + gate (`eval/`) |
| Seed, MLflow | ✅ `run.*` | logging wiring |

**Why this line and not more abstraction:** the brief is "config-driven where cheap, not
over-abstracted." Backends that genuinely have multiple real implementations (embedder, retriever)
get an interface + factory. Things with exactly one implementation (the gate, the metrics) are just
functions — no plugin system, no registry. `config.py` holds typed dataclasses mirroring the YAML so
the boundary is checked (`validate()`) rather than discovered at runtime via `KeyError`.

**Flow:** `load_config()` → ingest (Bronze) → chunk (Silver) → embed (Gold) →
`build_retriever(cfg, build_embedder(cfg))` → run golden questions → `metrics` →
`evaluate_gate()` → `enforce()`.

---

## 2. The retriever abstraction — why FAISS-default + Vector-Search-optional

`Retriever` is an ABC returning `RetrievedChunk` objects that **carry citations** (`contract_id` +
`char_span`). Two implementations sit behind `build_retriever(cfg, embedder)`, chosen by
`retrieval.backend`:

- **FAISS (default).** Local, exact (`flat` index), zero entitlements, runs on Free Edition for
  sure. This is the path the demo and the eval gate always run on, so the project is *defensible
  without depending on a feature that might not exist on the account*.
- **Databricks Vector Search (optional).** The "production on Databricks" story — managed index,
  scales past in-memory FAISS. Gated behind `# VERIFY`: Free Edition may lack the entitlement, so it
  must never be on the critical path.

**Why an interface rather than just FAISS:** the value of the project is the *eval gate*, and the
gate must be able to score whichever backend is live. A stable `Retriever` contract means swapping
backends is a one-line config change (`retrieval.backend: faiss → vector_search`) and the eval
harness, citations, and notebooks don't change. It also lets the same golden set compare backends
head-to-head — itself a demoable result ("Vector Search vs FAISS at recall@5").

**Why the embedder is injected into the retriever:** `query()` takes raw text, so the retriever owns
embedding the query with the *same* model used for the corpus. Prevents the classic train/serve
embedding mismatch and keeps callers from having to embed before searching.

**Deliberately not abstracted:** no pluggable vector-store registry, no reranker layer (yet), no
generation/LLM-answer layer — retrieval quality is the thing being gated, so the system stops at
retrieval + citations. A generation layer can sit on top later without touching this contract.

---

## 3. Decisions

> Framed for your call. Fill in the **Decision** lines; leave the reasoning trail.

### 3.1 Chunking: token-window vs structure-aware  ← YOUR CALL

CUAD contracts are clause-numbered legal documents; the golden answers are **annotated spans inside
specific clauses**. So chunk boundaries directly affect whether a relevant span survives intact —
which directly affects recall@k and groundedness, the metrics the gate enforces.

**Option A — token windows** (`chunking.strategy: token`)
- *For:* trivial, deterministic, model-agnostic; guaranteed to respect `max_tokens`; overlap
  mitigates boundary splits.
- *Against:* blind to clause structure — a window can straddle two clauses or bisect the exact span
  CUAD annotated, hurting groundedness; chunk text won't align to a citable clause unit.

**Option B — structure-aware** (`chunking.strategy: structure`)
- *For:* boundaries follow clause/section headings, so chunks map to citable units and annotated
  spans tend to stay whole → better groundedness and cleaner citations.
- *Against:* depends on reliable heading detection (CUAD formatting is inconsistent); oversized
  clauses still need a token sub-split (already stubbed); more code to get right.

**Things to weigh**
- Which maximizes **groundedness** (top-1 chunk actually contains the annotated span)? That's the
  metric most sensitive to boundaries.
- Citation quality: structure-aware gives "Clause 9.2" not "chars 14233–14745".
- Cost of getting heading detection wrong vs. the simplicity of token windows.

**Suggested approach:** ship **token** as the baseline to get a number through the gate, then try
**structure** and let the golden-set metrics decide — the harness exists precisely to make this an
empirical call, not an aesthetic one.

**Decision:** _______________________________________________  (rationale: ____________________)

### 3.2 Eval thresholds (gate values)  ← YOUR CALL, after baseline

`eval.gate.thresholds` currently holds placeholders (recall@5 ≥ 0.80, MRR ≥ 0.65, groundedness ≥
0.70). Set real values **after** a baseline run so the gate is honest, not aspirational.

**Decision:** _______________________________________________

### 3.3 Groundedness definition

Current: char-span overlap of the **top-1** retrieved chunk against the annotated CUAD span — no LLM
judge, fully deterministic, free. An LLM-judged faithfulness variant is possible later (`# VERIFY`)
but adds cost/non-determinism and isn't needed to gate *retrieval*.

**Decision (keep span-overlap? add LLM judge later?):** _______________________________________

### 3.4 Open `# VERIFY` items (Free Edition)
- Vector Search entitlement (`retrieval.vector_search.*`).
- UC catalog name (`main` vs `workspace`) + `CREATE SCHEMA` privilege.
- `faiss-cpu` install/import on the cluster.
- MLflow tracking availability (`run.mlflow.enabled`).
- CUAD HF split names (`corpus.hf_split`).
