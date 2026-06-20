# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Chunk — Bronze → Silver
# MAGIC
# MAGIC Reads Bronze contracts and writes **Silver**: one row per chunk, carrying the `char_span`
# MAGIC offsets needed to cite the chunk and to score it against CUAD's annotated answer spans.
# MAGIC Strategy + sizes come from `chunking.*` in config; the chunkers live in `src/gated_rag/chunking`.
# MAGIC
# MAGIC **Pipeline position:** 01_ingest → `02_chunk_silver` → 03_embed_gold → 04_index_and_eval.

# COMMAND ----------

# MAGIC %pip install -q tiktoken pyyaml

# COMMAND ----------

import os
import sys


def bootstrap():
    p = os.getcwd()
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "src", "gated_rag")):
            sys.path.insert(0, os.path.join(p, "src"))
            return p
        p = os.path.dirname(p)
    raise RuntimeError("could not locate gated_rag/src — run this from the repo Git folder")


REPO_ROOT = bootstrap()
from gated_rag.config import load_config

cfg = load_config(os.path.join(REPO_ROOT, "configs", "default.yaml"))
cat = cfg.catalog
bronze_fqn = f"{cat.catalog}.{cat.schema}.{cat.bronze_table}"
silver_fqn = f"{cat.catalog}.{cat.schema}.{cat.silver_table}"
print(f"{bronze_fqn} -> {silver_fqn}  (strategy={cfg.chunking.strategy}, max_tokens={cfg.chunking.max_tokens})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk each contract
# MAGIC Bronze is small (~510 rows), so we pull text to the driver and chunk in Python — simpler and
# MAGIC more portable than a tiktoken UDF on executors.

# COMMAND ----------

from gated_rag.chunking import chunk_contract

contracts = spark.table(bronze_fqn).select("contract_id", "text").collect()

silver_rows = []
for row in contracts:
    for ch in chunk_contract(row["contract_id"], row["text"], cfg.chunking):
        silver_rows.append({
            "contract_id": ch.contract_id,
            "chunk_id": ch.chunk_id,
            "text": ch.text,
            "char_start": ch.char_span[0],
            "char_end": ch.char_span[1],
            "section": ch.section,
        })

print(f"{len(contracts)} contracts -> {len(silver_rows)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Silver Delta table

# COMMAND ----------

from pyspark.sql.types import LongType, StringType, StructField, StructType

SILVER_SCHEMA = StructType([
    StructField("contract_id", StringType(), False),
    StructField("chunk_id", StringType(), False),
    StructField("text", StringType(), False),
    StructField("char_start", LongType(), False),
    StructField("char_end", LongType(), False),
    StructField("section", StringType(), True),
])

df = spark.createDataFrame(silver_rows, schema=SILVER_SCHEMA)
(df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_fqn))
print(f"wrote {df.count()} chunks -> {silver_fqn}")

# COMMAND ----------

display(spark.table(silver_fqn).limit(5))
dbutils.notebook.exit(f"silver:{df.count()}")
