# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Index + Eval Gate — Gold → metrics → **gate**
# MAGIC
# MAGIC Builds a FAISS index from the Gold vectors, derives the golden set from the Bronze annotations,
# MAGIC scores **per-contract** retrieval (recall@k / MRR / groundedness), logs the result to a Delta
# MAGIC `eval_runs` table, and **fails the task if any threshold is unmet**. This is the gate — the
# MAGIC reason the project exists: retrieval quality is enforced like a data-quality check.
# MAGIC
# MAGIC **Pipeline position:** 01_ingest → 02_chunk_silver → 03_embed_gold → `04_index_and_eval`.

# COMMAND ----------

# MAGIC %pip install -q faiss-cpu sentence-transformers pyyaml numpy

# COMMAND ----------

import os
import sys


def bootstrap():
    """Put the repo's src/ on the path; prefer the bundle-provided repo_root, else walk up from cwd."""
    candidates = []
    try:
        dbutils.widgets.text("repo_root", "")
        rr = dbutils.widgets.get("repo_root").strip()
        if rr:
            candidates.append(rr)
    except Exception:
        pass
    p = os.getcwd()
    for _ in range(6):
        candidates.append(p)
        p = os.path.dirname(p)
    for c in candidates:
        if os.path.isdir(os.path.join(c, "src", "gated_rag")):
            sys.path.insert(0, os.path.join(c, "src"))
            return c
    raise RuntimeError(f"could not locate gated_rag/src — tried {candidates}")


REPO_ROOT = bootstrap()
from gated_rag.config import load_config

cfg = load_config(os.path.join(REPO_ROOT, "configs", "default.yaml"))
cat = cfg.catalog
bronze_fqn = f"{cat.catalog}.{cat.schema}.{cat.bronze_table}"
gold_fqn = f"{cat.catalog}.{cat.schema}.{cat.gold_table}"
runs_fqn = f"{cat.catalog}.{cat.schema}.eval_runs"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the retriever from Gold

# COMMAND ----------

from gated_rag.embedding import build_embedder
from gated_rag.retrieval import build_retriever

gold = spark.table(gold_fqn).collect()
gold_chunks = [
    {
        "contract_id": r["contract_id"],
        "chunk_id": r["chunk_id"],
        "text": r["text"],
        "char_span": (r["char_start"], r["char_end"]),
        "embedding": r["embedding"],
    }
    for r in gold
]

embedder = build_embedder(cfg.embedding)
retriever = build_retriever(cfg.retrieval, embedder)   # faiss by default; vector_search is # VERIFY
retriever.index(gold_chunks)
print(f"indexed {len(gold_chunks)} chunks in {cfg.retrieval.backend}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Derive the golden set from Bronze annotations

# COMMAND ----------

from gated_rag.eval.golden import derive_golden

bronze = spark.table(bronze_fqn).select("contract_id", "annotations").collect()
bronze_rows = [
    {
        "contract_id": r["contract_id"],
        "annotations": [
            {"question": a["question"], "answer_spans": [list(s) for s in a["answer_spans"]]}
            for a in r["annotations"]
        ],
    }
    for r in bronze
]
golden = derive_golden(bronze_rows)
print(f"{len(golden)} golden questions with annotated spans")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score per-contract retrieval
# MAGIC CUAD questions are contract-agnostic, so retrieval is scoped to each question's contract
# MAGIC (`filter_contract_id`) — the realistic "ask questions about THIS contract" path.

# COMMAND ----------

from statistics import mean

from gated_rag.eval import metrics as M

k_values = cfg.eval.k_values
max_k = max(k_values)
recalls = {k: [] for k in k_values}
rr, gnd = [], []

for item in golden:
    cid, spans = item["contract_id"], item["answer_spans"]
    results = retriever.query(item["question"], top_k=max_k, filter_contract_id=cid)
    for k in k_values:
        recalls[k].append(M.recall_at_k(results, cid, spans, k))
    rr.append(M.reciprocal_rank(results, cid, spans))
    gnd.append(M.groundedness(results, cid, spans))

measured = {f"recall_at_{k}": mean(recalls[k]) for k in k_values}
measured["mrr"] = mean(rr)
measured["groundedness"] = mean(gnd)
for name in sorted(measured):
    print(f"{name:16} {measured[name]:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the run to Delta (eval sink)
# MAGIC MLflow is `# VERIFY` on Free Edition, so results land in a Delta `eval_runs` table by default.

# COMMAND ----------

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.functions import current_timestamp, lit

from gated_rag.eval.gate import evaluate_gate

gate = evaluate_gate(measured, cfg.eval.gate)

RUN_SCHEMA = StructType([
    StructField("metrics", MapType(StringType(), DoubleType()), False),
    StructField("thresholds", MapType(StringType(), DoubleType()), False),
    StructField("passed", BooleanType(), False),
    StructField("n_questions", DoubleType(), False),
])
run_df = (
    spark.createDataFrame(
        [{
            "metrics": {k: float(v) for k, v in measured.items()},
            "thresholds": {k: float(v) for k, v in cfg.eval.gate.thresholds.items()},
            "passed": gate.passed,
            "n_questions": float(len(golden)),
        }],
        schema=RUN_SCHEMA,
    )
    .withColumn("run_at", current_timestamp())
    .withColumn("backend", lit(cfg.retrieval.backend))
)
run_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(runs_fqn)
print(f"logged run -> {runs_fqn} (passed={gate.passed})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## THE GATE
# MAGIC Fail the task (and the job) when retrieval quality is below threshold and `on_failure == fail`.

# COMMAND ----------

if not gate.passed:
    breaches = ", ".join(
        f"{n}={m:.3f} < {t:.3f}" for n, (m, t) in gate.failures.items()
    )
    if cfg.eval.gate.on_failure == "fail":
        raise Exception(f"[gate] FAIL — retrieval quality below threshold: {breaches}")
    print(f"[gate] WARN — thresholds unmet (on_failure=warn): {breaches}")
else:
    print("[gate] PASS — all thresholds met")

dbutils.notebook.exit(f"gate_passed:{gate.passed}")
