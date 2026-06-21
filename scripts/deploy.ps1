# Deploy (and optionally run) the gated-rag medallion Job on Databricks via the Asset Bundle.
#
# One command: ensure the Databricks CLI, authenticate from -DatabricksHost/-Token (or the
# DATABRICKS_HOST/DATABRICKS_TOKEN env vars), validate the bundle, deploy the 4-task Job, and
# -- with -Run -- trigger it.
#
# Examples:
#   ./scripts/deploy.ps1 -DatabricksHost https://dbc-xxxx.cloud.databricks.com -Token dapi... -Run
#   $env:DATABRICKS_HOST="https://dbc-xxxx.cloud.databricks.com"; $env:DATABRICKS_TOKEN="dapi..."
#   ./scripts/deploy.ps1 -Run -Limit ""        # "" = full corpus

param(
  [string]$DatabricksHost = $env:DATABRICKS_HOST,
  [string]$Token          = $env:DATABRICKS_TOKEN,
  [string]$Target         = "dev",
  [string]$Limit          = "25",
  [switch]$Run
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Die([string]$msg) {
  Write-Host "ERROR: $msg" -ForegroundColor Red
  exit 1
}

# 1. Ensure the Databricks CLI is installed.
if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
  Write-Host "Databricks CLI not found - attempting install via winget..." -ForegroundColor Yellow
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id Databricks.DatabricksCLI -e --accept-source-agreements --accept-package-agreements
  }
  if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
    Die "Could not install the Databricks CLI automatically. Install from https://docs.databricks.com/dev-tools/cli/install.html and re-run."
  }
}
$ver = databricks --version
Write-Host "databricks CLI: $ver" -ForegroundColor Green

# 2. Auth. The CLI + bundle read these env vars; nothing is written to disk.
if (-not $DatabricksHost) {
  Die "No workspace host. Pass -DatabricksHost or set DATABRICKS_HOST (your Free Edition URL, e.g. https://dbc-xxxxxxxx-xxxx.cloud.databricks.com)."
}
if (-not $Token) {
  Die "No token. Pass -Token or set DATABRICKS_TOKEN (Databricks: Settings -> Developer -> Access tokens)."
}
$env:DATABRICKS_HOST  = $DatabricksHost
$env:DATABRICKS_TOKEN = $Token

# 3. validate -> 4. deploy -> 5. (optional) run.
Write-Host "`n== validate ==" -ForegroundColor Cyan
databricks bundle validate -t $Target
if ($LASTEXITCODE -ne 0) { Die "bundle validate failed" }

Write-Host "`n== deploy ==" -ForegroundColor Cyan
databricks bundle deploy -t $Target
if ($LASTEXITCODE -ne 0) { Die "bundle deploy failed" }
Write-Host "Deployed job 'gated-rag-medallion' to target '$Target'." -ForegroundColor Green

if ($Run) {
  Write-Host "`n== run (limit='$Limit') ==" -ForegroundColor Cyan
  databricks bundle run gated_rag_medallion -t $Target --var "limit=$Limit"
  if ($LASTEXITCODE -ne 0) { Die "job run failed (check the failing task in the Workflows UI)" }
  Write-Host "Run complete." -ForegroundColor Green
}
else {
  Write-Host "Deployed but not run. Add -Run to trigger it, or start it from Workflows in the UI." -ForegroundColor Yellow
}
