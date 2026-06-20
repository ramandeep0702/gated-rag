# Databricks medallion pipeline

Four notebooks build the medallion in Unity Catalog and enforce the eval gate. They are thin
orchestration over the `gated_rag` library (same code the local pipeline + tests use), so the logic
is identical on a laptop and on Databricks.

```
00_verify_env       diagnostics — resolves the # VERIFY items (run once)
01_ingest           CUAD            -> workspace.gated_rag.contracts_bronze   (one row per contract)
02_chunk_silver     Bronze          -> workspace.gated_rag.chunks_silver      (one row per chunk + char_span)
03_embed_gold       Silver          -> workspace.gated_rag.chunks_gold        (chunk + embedding vector)
04_index_and_eval   Gold + Bronze   -> metrics -> THE GATE  (+ logs to workspace.gated_rag.eval_runs)
```

The gate fails task `index_and_eval` (and the job) if recall@k / MRR / groundedness fall below the
thresholds in [`configs/default.yaml`](../configs/default.yaml).

## Prerequisites

- A Databricks **Git folder** pointing at this repo (Workspace → Repos → Add → this GitHub URL), OR
  deploy via the Asset Bundle (below). Both make the repo's `src/` importable — each notebook walks
  up from its location to find `src/gated_rag`.
- Run **`00_verify_env`** first and confirm the writable catalog is `workspace` (already set in
  config). Free Edition has no `main`.
- Outbound network from the cluster: `01_ingest` downloads CUAD from Hugging Face, `03/04` download
  the embedding model, and chunking downloads the tiktoken vocab on first use.

## Run option A — Workflows UI (simplest on Free Edition)

1. Add this repo as a Git folder.
2. Workflows → Create Job. Add four **notebook tasks** pointing at `notebooks/01..04`, chaining
   each to depend on the previous one. Use serverless compute.
3. On the `01_ingest` task, set parameter `limit = 25` for a fast first run (blank = full corpus).
4. Run. You get the task-DAG pipeline view; `eval_runs` accumulates one row per run.

## Run option B — one-command deploy (infra-as-code)

The deploy scripts install the Databricks CLI if needed, authenticate from env vars, validate the
bundle, deploy the Job, and (with the run flag) trigger it. Get a token from Databricks →
**Settings → Developer → Access tokens**, and your workspace URL from the browser address bar.

```powershell
# Windows (PowerShell)
$env:DATABRICKS_HOST="https://dbc-xxxxxxxx-xxxx.cloud.databricks.com"
$env:DATABRICKS_TOKEN="dapi..."
./scripts/deploy.ps1 -Run                 # fast first run (limit=25)
./scripts/deploy.ps1 -Run -Limit ""       # full ~510-contract corpus
```

```bash
# bash
export DATABRICKS_HOST=https://dbc-xxxxxxxx-xxxx.cloud.databricks.com
export DATABRICKS_TOKEN=dapi...
./scripts/deploy.sh --run                 # fast first run (limit=25)
./scripts/deploy.sh --run --limit ""      # full corpus
```

Nothing in [`../databricks.yml`](../databricks.yml) needs editing — host/token come from the env
vars. The bundle deploys a Job whose tasks pull notebooks from GitHub `main`, so push before you run.

## After Gold lands — Genie

`workspace.gated_rag.chunks_gold` (and `contracts_bronze`) are governed Unity Catalog tables, so you
can point a **Genie Space** at them for natural-language Q&A over the corpus (text-to-SQL: "how many
contracts have a governing-law clause?", "show chunks from contract X near char 14000"). That is a
separate surface from the vector retrieval the gate scores — both read the same Gold table.
