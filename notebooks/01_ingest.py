# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest — CUAD → Bronze
# MAGIC
# MAGIC Loads CUAD from Hugging Face and writes the **Bronze** Delta table in Unity Catalog: one row
# MAGIC per contract — raw text + metadata + the clause annotations (kept nested for later golden-set
# MAGIC derivation). Thin orchestration over the `gated_rag` library; the pure transforms live in
# MAGIC `src/gated_rag/ingest`.
# MAGIC
# MAGIC **Pipeline position:** `01_ingest` → 02_chunk_silver → 03_embed_gold → 04_index_and_eval.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC Install what the serverless env lacks, then put the repo's `src/` on the path. `%pip` restarts
# MAGIC Python, so the bootstrap import runs in the next cell.

# COMMAND ----------

# MAGIC %pip install -q datasets pyyaml

# COMMAND ----------

import os
import sys


def bootstrap():
    """Locate the repo's src/ (walking up from the notebook) and load the typed config."""
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
catalog_cfg = cfg.catalog
print(f"catalog target: {catalog_cfg.catalog}.{catalog_cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Corpus size
# MAGIC `limit` widget overrides `corpus.limit` for fast first runs (empty = use config = all ~510
# MAGIC contracts). This is the one place corpus size is set; downstream notebooks just read Bronze.

# COMMAND ----------

dbutils.widgets.text("limit", "", "Max contracts (blank = config default)")
_limit = dbutils.widgets.get("limit").strip()
if _limit:
    cfg.corpus.limit = int(_limit)
print(f"corpus.limit = {cfg.corpus.limit}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog target
# MAGIC `workspace` is the writable catalog on Free Edition (resolved in `00_verify_env`; there is no
# MAGIC `main`). Schema is created if missing.

# COMMAND ----------

from gated_rag.config import CatalogConfig


def ensure_namespace(catalog_cfg: CatalogConfig) -> str:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_cfg.catalog}.{catalog_cfg.schema}")
    return f"{catalog_cfg.catalog}.{catalog_cfg.schema}.{catalog_cfg.bronze_table}"


bronze_fqn = ensure_namespace(catalog_cfg)
print(bronze_fqn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load CUAD → normalize → dedupe to one row per contract

# COMMAND ----------

from gated_rag.ingest import aggregate_contracts, load_cuad, to_bronze_record

raw_records = list(load_cuad(cfg.corpus))
bronze_rows = aggregate_contracts(to_bronze_record(r) for r in raw_records)
print(f"loaded {len(raw_records)} QA rows -> {len(bronze_rows)} contracts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Bronze Delta table
# MAGIC Explicit schema because `annotations` is nested (`array<struct<question, answer_spans>>`,
# MAGIC where each span is `[start, end]`).

# COMMAND ----------

from pyspark.sql.types import (
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)

BRONZE_SCHEMA = StructType([
    StructField("contract_id", StringType(), False),
    StructField("title", StringType(), True),
    StructField("text", StringType(), False),
    StructField("annotations", ArrayType(StructType([
        StructField("question", StringType(), False),
        StructField("answer_spans", ArrayType(ArrayType(LongType())), False),
    ])), False),
    StructField("source", StringType(), True),
    StructField("ingested_at", StringType(), True),
])


def to_spark_row(row: dict) -> dict:
    """tuples -> lists so Spark accepts the nested span arrays."""
    return {
        **row,
        "annotations": [
            {"question": a["question"], "answer_spans": [list(s) for s in a["answer_spans"]]}
            for a in row["annotations"]
        ],
    }


df = spark.createDataFrame([to_spark_row(r) for r in bronze_rows], schema=BRONZE_SCHEMA)
(df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_fqn))
print(f"wrote {df.count()} rows -> {bronze_fqn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke check

# COMMAND ----------

display(spark.table(bronze_fqn).select("contract_id", "title", "source").limit(5))
n = spark.table(bronze_fqn).count()
n_annot = spark.sql(
    f"SELECT count(*) AS c FROM {bronze_fqn} WHERE size(annotations) > 0"
).collect()[0]["c"]
assert spark.sql(f"SELECT count(*) AS c FROM {bronze_fqn} WHERE contract_id IS NULL").collect()[0]["c"] == 0
print(f"OK — {n} contracts, {n_annot} with annotations")
dbutils.notebook.exit(f"bronze:{n}")
