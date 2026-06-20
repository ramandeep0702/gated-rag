#!/usr/bin/env bash
# Deploy (and optionally run) the gated-rag medallion Job on Databricks via the Asset Bundle.
#
# Auth comes from DATABRICKS_HOST / DATABRICKS_TOKEN env vars (nothing is written to disk).
#   export DATABRICKS_HOST=https://dbc-xxxxxxxx-xxxx.cloud.databricks.com
#   export DATABRICKS_TOKEN=dapi...
#   ./scripts/deploy.sh --run                 # fast first run (limit=25)
#   ./scripts/deploy.sh --run --limit ""      # full ~510-contract corpus
set -euo pipefail

TARGET="dev"; LIMIT="25"; RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1 ;;
    --target) TARGET="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."

die() { echo "ERROR: $*" >&2; exit 1; }

# 1. Ensure the CLI is installed.
if ! command -v databricks >/dev/null 2>&1; then
  echo "Databricks CLI not found — installing..."
  curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh \
    || die "CLI install failed; see https://docs.databricks.com/dev-tools/cli/install.html"
fi
echo "databricks CLI: $(databricks --version)"

# 2. Auth check.
: "${DATABRICKS_HOST:?set DATABRICKS_HOST (your workspace URL)}"
: "${DATABRICKS_TOKEN:?set DATABRICKS_TOKEN (Settings -> Developer -> Access tokens)}"

# 3. validate -> 4. deploy -> 5. (optional) run.
echo "== validate =="; databricks bundle validate -t "$TARGET"
echo "== deploy ==";   databricks bundle deploy   -t "$TARGET"
echo "Deployed job 'gated-rag-medallion' to target '$TARGET'."

if [ "$RUN" -eq 1 ]; then
  echo "== run (limit='$LIMIT') =="
  databricks bundle run gated_rag_medallion -t "$TARGET" --var "limit=$LIMIT"
  echo "Run complete."
else
  echo "Deployed but not run. Re-run with --run, or start it from Workflows in the UI."
fi
