# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Verify Environment (Databricks Free Edition)
# MAGIC
# MAGIC Resolves the `# VERIFY` items before we build further. Each check prints **PASS / FAIL / INFO**
# MAGIC and appends to a results table. The **last cell prints a summary block** — paste it back to
# MAGIC confirm before we generate the Silver/Gold notebooks and the FAISS retriever.
# MAGIC
# MAGIC This notebook is fully implemented (it's a diagnostic tool, not pipeline logic). Safe to re-run;
# MAGIC it cleans up its probe schema/table.

# COMMAND ----------

# MAGIC %pip install -q datasets
# MAGIC # dbutils.library.restartPython()   # uncomment + re-run from here if imports below fail after install

# COMMAND ----------

# Results accumulator. Each entry: (check, status, detail). status in {PASS, FAIL, INFO}.
RESULTS: "list[tuple[str, str, str]]" = []


def record(check: str, status: str, detail: str = "") -> None:
    RESULTS.append((check, status, detail))
    print(f"[{status:4}] {check}" + (f" — {detail}" if detail else ""))


# Probe namespace used for the writable-catalog test (created then dropped).
PROBE_SCHEMA = "gated_rag_envcheck"
PROBE_TABLE = "probe"
CUAD_DATASET = "theatticusproject/cuad-qa"   # keep in sync with configs/default.yaml -> corpus.hf_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. faiss-cpu installs and imports

# COMMAND ----------

try:
    import faiss  # noqa: F401
    record("faiss import", "PASS", f"version={getattr(faiss, '__version__', 'unknown')}")
except Exception as e:  # not installed on this cluster
    print(f"faiss not present, attempting install... ({type(e).__name__})")
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
        import faiss  # noqa: F401
        record("faiss import", "PASS", f"installed; version={getattr(faiss, '__version__', 'unknown')}")
    except Exception as e2:
        record("faiss import", "FAIL", f"{type(e2).__name__}: {e2}")
        # VERIFY: if this FAILs, faiss-cpu may need a cluster-level library or restartPython() after %pip.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Which UC catalog is writable + can we CREATE SCHEMA?
# MAGIC Tries `main` then `workspace`. For each: CREATE SCHEMA → CREATE TABLE (Delta) → INSERT → SELECT →
# MAGIC clean up. First catalog that survives all steps is our write target.

# COMMAND ----------

def test_catalog_writable(catalog: str) -> "tuple[bool, str]":
    """Return (writable, detail). Best-effort cleanup even on partial failure."""
    fq_schema = f"{catalog}.{PROBE_SCHEMA}"
    fq_table = f"{fq_schema}.{PROBE_TABLE}"
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fq_schema}")
        spark.sql(f"CREATE OR REPLACE TABLE {fq_table} (id INT) USING DELTA")
        spark.sql(f"INSERT INTO {fq_table} VALUES (1)")
        n = spark.sql(f"SELECT COUNT(*) AS n FROM {fq_table}").collect()[0]["n"]
        return (n == 1, f"create+insert+read ok (n={n})")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")
    finally:
        # Cleanup — ignore errors so a half-created probe doesn't linger.
        for stmt in (f"DROP TABLE IF EXISTS {fq_table}", f"DROP SCHEMA IF EXISTS {fq_schema} CASCADE"):
            try:
                spark.sql(stmt)
            except Exception:
                pass


# Show what's even visible first (helps if neither candidate works).
try:
    cats = [r[0] for r in spark.sql("SHOW CATALOGS").collect()]
    record("SHOW CATALOGS", "INFO", ", ".join(cats) or "(none)")
except Exception as e:
    record("SHOW CATALOGS", "FAIL", f"{type(e).__name__}: {e}")

writable_catalog = None
for cand in ("main", "workspace"):
    ok, detail = test_catalog_writable(cand)
    record(f"catalog writable: {cand}", "PASS" if ok else "FAIL", detail)
    if ok and writable_catalog is None:
        writable_catalog = cand

record("=> chosen write catalog", "INFO" if writable_catalog else "FAIL",
       writable_catalog or "none writable — set configs catalog.catalog manually  # VERIFY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. MLflow tracking available?
# MAGIC If unavailable, we log eval results to a Delta `dq_log`-style table instead.

# COMMAND ----------

try:
    import mlflow
    with mlflow.start_run(run_name="env_check") as run:
        mlflow.log_metric("env_check", 1.0)
    record("mlflow tracking", "PASS", f"logged a run (id={run.info.run_id[:8]}...)")
    record("=> eval sink", "INFO", "mlflow")
except Exception as e:
    record("mlflow tracking", "FAIL", f"{type(e).__name__}: {e}")
    record("=> eval sink", "INFO", "delta dq_log table (mlflow unavailable)  # VERIFY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Databricks Vector Search entitlement
# MAGIC Expected **unavailable** on Free Edition. Confirming lets us commit to the FAISS default path.

# COMMAND ----------

vs_available = False
try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    eps = vsc.list_endpoints()  # entitlement / API reachability probe
    vs_available = True
    record("vector search", "INFO", f"AVAILABLE — endpoints reachable ({type(eps).__name__})")
except Exception as e:
    record("vector search", "INFO", f"unavailable (expected on Free Edition): {type(e).__name__}")

record("=> retriever backend", "INFO",
       "vector_search possible" if vs_available else "faiss (committed default)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. CUAD HF split names actually present
# MAGIC Confirms `corpus.hf_split` in config matches reality. Needs outbound network from the cluster.

# COMMAND ----------

try:
    from datasets import get_dataset_split_names
    splits = get_dataset_split_names(CUAD_DATASET)
    record("CUAD splits", "PASS", f"{CUAD_DATASET} -> {splits}")
except Exception as e:
    record("CUAD splits", "FAIL", f"{type(e).__name__}: {e}")
    # VERIFY: failure may be no network egress from Free Edition, or dataset needs a config name.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — paste this block back

# COMMAND ----------

def render_summary(results: "list[tuple[str, str, str]]") -> str:
    width = max((len(c) for c, _, _ in results), default=10)
    lines = ["===== gated-rag env verification ====="]
    for check, status, detail in results:
        lines.append(f"{status:4} | {check.ljust(width)} | {detail}")
    lines.append("======================================")
    return "\n".join(lines)


print(render_summary(RESULTS))
