# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest — CUAD → Bronze
# MAGIC
# MAGIC Loads the CUAD corpus from Hugging Face and writes a **Bronze** Delta table in Unity Catalog:
# MAGIC raw contract text + metadata + the clause annotations (kept nested for later golden-set
# MAGIC derivation). Silver (chunking) and Gold (embedding) are separate notebooks.
# MAGIC
# MAGIC **Skeleton** — function signatures + TODO bodies. Logic is yours.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC `%pip install` only what the cluster lacks. On Free Edition, `datasets` may not be preinstalled.

# COMMAND ----------

# MAGIC %pip install -q datasets pyyaml
# MAGIC # dbutils.library.restartPython()   # uncomment if pip installed into the notebook env  # VERIFY needed on Free Edition

# COMMAND ----------

from gated_rag.config import load_config, CorpusConfig, CatalogConfig
from gated_rag.ingest import load_cuad, to_bronze_record

# TODO: make sure src/ is importable from the notebook.
# Hint: either `%pip install -e /Workspace/Repos/.../gated-rag` or sys.path.append the repo's src/.

cfg = load_config("../configs/default.yaml")   # TODO: VERIFY path resolution from the notebook cwd
corpus_cfg: CorpusConfig = cfg.corpus
catalog_cfg: CatalogConfig = cfg.catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog target
# MAGIC `# VERIFY` Free Edition catalog naming. Free Edition provisions a workspace catalog — often
# MAGIC `main` (sometimes `workspace`). Confirm with `SHOW CATALOGS` and set `catalog.catalog` in the
# MAGIC YAML accordingly. Schema is created if missing.

# COMMAND ----------

def ensure_namespace(catalog_cfg: CatalogConfig) -> str:
    """Ensure <catalog>.<schema> exists; return the fully-qualified bronze table name.

    Hint: spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}");
          return f"{catalog}.{schema}.{bronze_table}".
    # VERIFY: CREATE SCHEMA may require privileges Free Edition doesn't grant on `main`.
    """
    # TODO: create schema if missing, return FQ bronze table name.
    raise NotImplementedError


bronze_fqn = ensure_namespace(catalog_cfg)
print(bronze_fqn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load CUAD from Hugging Face
# MAGIC `theatticusproject/cuad-qa`. Annotations (clause category → answer span) are the golden-set
# MAGIC source, so keep them — don't flatten them away here.

# COMMAND ----------

def read_cuad(corpus_cfg: CorpusConfig) -> "list[dict]":
    """Pull raw CUAD records (respecting corpus.limit) as plain dicts.

    Hint: delegate to gated_rag.ingest.load_cuad(corpus_cfg); materialize the generator to a list.
    # VERIFY split name (corpus.hf_split) against the dataset card.
    """
    # TODO: call load_cuad and collect.
    raise NotImplementedError


raw_records = read_cuad(corpus_cfg)
print(f"loaded {len(raw_records)} raw records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalize → Bronze rows
# MAGIC One row per contract: `contract_id, title, text, annotations, source, ingested_at`.

# COMMAND ----------

def to_bronze_rows(raw_records: "list[dict]") -> "list[dict]":
    """Map raw CUAD records to Bronze rows.

    Hint: [to_bronze_record(r) for r in raw_records]. CUAD repeats the same contract across many
    QA rows — dedupe by contract_id so Bronze holds one row per contract, annotations aggregated.
    """
    # TODO: normalize + dedupe by contract_id.
    raise NotImplementedError


bronze_rows = to_bronze_rows(raw_records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Bronze Delta table

# COMMAND ----------

def write_bronze(bronze_rows: "list[dict]", bronze_fqn: str) -> None:
    """Write Bronze rows as a managed Delta table in Unity Catalog.

    Hint: spark.createDataFrame(bronze_rows) — define an explicit schema (annotations is nested);
          df.write.format('delta').mode('overwrite').option('overwriteSchema','true').saveAsTable(bronze_fqn).
    # VERIFY: managed table location works on Free Edition (no external location needed).
    """
    # TODO: build DataFrame with explicit schema, write Delta, saveAsTable.
    raise NotImplementedError


write_bronze(bronze_rows, bronze_fqn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke check

# COMMAND ----------

# TODO: display(spark.table(bronze_fqn).limit(5)); assert count == expected (corpus.limit or 510).
# Hint: also assert no null contract_id and that annotations survived the write.
