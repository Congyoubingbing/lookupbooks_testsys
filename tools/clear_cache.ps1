param(
  [string]$Root = "."
)

# Delete per-book cache folders under outputs.
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File .\tools\clear_cache.ps1

$outputs = Join-Path $Root "outputs"
if (-not (Test-Path $outputs)) {
  Write-Host "No outputs/ folder found at: $outputs"
  exit 0
}

Get-ChildItem -Path $outputs -Directory -Recurse -Force |
  Where-Object { $_.Name -eq "_cache" } |
  ForEach-Object {
    Write-Host "Removing cache: $($_.FullName)"
    Remove-Item -Recurse -Force $_.FullName
  }

Write-Host "Done."
