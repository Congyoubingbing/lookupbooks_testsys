param(
  [string]$PythonExe = "",
  [string]$BatchFile = "",
  [string]$AgentConfig = "config/agent.yaml",
  [string]$OutputsRoot = "./outputs",
  [string]$LibraryRoot = "../shared_assets/library",
  [string]$RunsDir = "./runs/batch",
  [string]$SuiteId = "full_lookupbooks_suite",
  [int]$Workers = 10,
  [switch]$SuiteJudge
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
  param([string]$ExplicitPython)
  if ($ExplicitPython -and (Test-Path $ExplicitPython)) { return (Resolve-Path $ExplicitPython).Path }
  if ($env:CONDA_PREFIX) {
    $condaPy = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $condaPy) { return (Resolve-Path $condaPy).Path }
  }
  $cmd = Get-Command python -ErrorAction Stop
  return $cmd.Source
}

function Resolve-BatchFile {
  param([string]$Explicit)
  if ($Explicit) { return $Explicit }
  $testsDir = Join-Path $PSScriptRoot "..\tests"
  $cand = Get-ChildItem -Path $testsDir -File -Filter "*.md" | Where-Object { $_.Name -match "q0" } | Sort-Object Name | Select-Object -First 1
  if (-not $cand) {
    $cand = Get-ChildItem -Path $testsDir -File -Filter "*.md" | Sort-Object Name | Select-Object -First 1
  }
  if (-not $cand) { throw "No markdown batch file found under: $testsDir" }
  return (Join-Path "tests" $cand.Name)
}

$PYTHON = Resolve-Python -ExplicitPython $PythonExe
$ResolvedBatch = Resolve-BatchFile -Explicit $BatchFile
Write-Host "[lookupbooks_testsys] Python=$PYTHON"
Write-Host "[lookupbooks_testsys] BatchFile=$ResolvedBatch"

$argsList = @(
  "-u", "run.py", "batch-ask",
  "--batch-file", $ResolvedBatch,
  "--agent-config", $AgentConfig,
  "--outputs-root", $OutputsRoot,
  "--library-root", $LibraryRoot,
  "--runs-dir", $RunsDir,
  "--workers", "$Workers",
  "--suite", "M1-M8",
  "--suite-id", $SuiteId,
  "--suite-m8-ks", "1,3,5"
)
if ($SuiteJudge) { $argsList += "--suite-judge" }

& $PYTHON @argsList
if ($LASTEXITCODE -ne 0) { throw "lookupbooks_testsys suite failed: $LASTEXITCODE" }
