# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Embed — Silver → Gold
# MAGIC
# MAGIC Embeds every Silver chunk and writes **Gold**: the chunk + its vector. Gold is the retrieval
# MAGIC source — the embeddings the FAISS index (or Databricks Vector Search) is built from. Model and
# MAGIC batch size come from `embedding.*`; the embedder lives in `src/gated_rag/embedding`.
# MAGIC
# MAGIC **Pipeline position:** 01_ingest → 02_chunk_silver → `03_embed_gold` → 04_index_and_eval.

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers pyyaml numpy

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
silver_fqn = f"{cat.catalog}.{cat.schema}.{cat.silver_table}"
gold_fqn = f"{cat.catalog}.{cat.schema}.{cat.gold_table}"
print(f"{silver_fqn} -> {gold_fqn}  (model={cfg.embedding.model}, dim={cfg.embedding.dim})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embed chunk text
# MAGIC Passages are embedded raw (no query prefix). The embedder downloads the model on first use —
# MAGIC needs outbound network from the cluster.

# COMMAND ----------

from gated_rag.embedding import build_embedder

silver = spark.table(silver_fqn).select(
    "contract_id", "chunk_id", "text", "char_start", "char_end", "section"
).collect()

embedder = build_embedder(cfg.embedding)
vectors = embedder.embed([r["text"] for r in silver])   # (N, dim) float32
assert vectors.shape[1] == cfg.embedding.dim, f"dim mismatch: {vectors.shape[1]} != {cfg.embedding.dim}"
print(f"embedded {len(silver)} chunks -> {vectors.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Delta table (chunk + vector)

# COMMAND ----------

from pyspark.sql.types import (
    ArrayType,
    FloatType,
    LongType,
    StringType,
    StructField,
    StructType,
)

GOLD_SCHEMA = StructType([
    StructField("contract_id", StringType(), False),
    StructField("chunk_id", StringType(), False),
    StructField("text", StringType(), False),
    StructField("char_start", LongType(), False),
    StructField("char_end", LongType(), False),
    StructField("section", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), False),
])

gold_rows = [
    {
        "contract_id": r["contract_id"],
        "chunk_id": r["chunk_id"],
        "text": r["text"],
        "char_start": r["char_start"],
        "char_end": r["char_end"],
        "section": r["section"],
        "embedding": [float(x) for x in vectors[i]],
    }
    for i, r in enumerate(silver)
]

df = spark.createDataFrame(gold_rows, schema=GOLD_SCHEMA)
(df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(gold_fqn))
print(f"wrote {df.count()} embedded chunks -> {gold_fqn}")

# COMMAND ----------

display(spark.table(gold_fqn).select("contract_id", "chunk_id", "section").limit(5))
dbutils.notebook.exit(f"gold:{df.count()}")
