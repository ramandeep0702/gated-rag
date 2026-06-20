<#
.SYNOPSIS
  Deploy (and optionally run) the gated-rag medallion Job on Databricks via the Asset Bundle.

.DESCRIPTION
  One command does everything: ensures the Databricks CLI is present, authenticates from
  -DatabricksHost/-Token (or the DATABRICKS_HOST/DATABRICKS_TOKEN env vars), validates the bundle,
  deploys the 4-task Job, and -- with -Run -- triggers it.

.EXAMPLE
  ./scripts/deploy.ps1 -DatabricksHost https://dbc-xxxx.cloud.databricks.com -Token dapi... -Run

.EXAMPLE
  # with env vars already set:
  $env:DATABRICKS_HOST="https://dbc-xxxx.cloud.databricks.com"; $env:DATABRICKS_TOKEN="dapi..."
  ./scripts/deploy.ps1 -Run -Limit ""        # "" = full corpus
#>
[CmdletBinding()]
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

function Die($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# 1. Ensure the Databricks CLI is installed.
if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
  Write-Host "Databricks CLI not found — attempting install via winget..." -ForegroundColor Yellow
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id Databricks.DatabricksCLI -e --accept-source-agreements --accept-package-agreements
  }
  if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
    Die "Could not install the Databricks CLI automatically. Install it from https://docs.databricks.com/dev-tools/cli/install.html and re-run."
  }
}
Write-Host "databricks CLI: $((databricks --version) 2>&1)" -ForegroundColor Green

# 2. Auth. The CLI + bundle read these env vars; nothing is written to disk.
if (-not $DatabricksHost) { Die "No workspace host. Pass -DatabricksHost or set DATABRICKS_HOST. Find it in your Free Edition URL (https://dbc-xxxxxxxx-xxxx.cloud.databricks.com)." }
if (-not $Token)          { Die "No token. Pass -Token or set DATABRICKS_TOKEN. Create one in Databricks: Settings -> Developer -> Access tokens." }
$env:DATABRICKS_HOST  = $DatabricksHost
$env:DATABRICKS_TOKEN = $Token

# 3. Validate -> 4. Deploy -> 5. (optional) Run.
Write-Host "`n== validate ==" -ForegroundColor Cyan
databricks bundle validate -t $Target; if ($LASTEXITCODE) { Die "bundle validate failed" }

Write-Host "`n== deploy ==" -ForegroundColor Cyan
databricks bundle deploy -t $Target;   if ($LASTEXITCODE) { Die "bundle deploy failed" }
Write-Host "Deployed job 'gated-rag-medallion' to target '$Target'." -ForegroundColor Green

if ($Run) {
  Write-Host "`n== run (limit='$Limit') ==" -ForegroundColor Cyan
  databricks bundle run gated_rag_medallion -t $Target --var "limit=$Limit"
  if ($LASTEXITCODE) { Die "job run failed (check the task that errored in the Workflows UI)" }
  Write-Host "Run complete." -ForegroundColor Green
} else {
  Write-Host "`nDeployed but not run. Add -Run to trigger it, or start it from Workflows in the UI." -ForegroundColor Yellow
}
