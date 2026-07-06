#!/usr/bin/env bash
set -euo pipefail

# Delete per-book cache folders under outputs.
# Usage (from project root):
#   bash tools/clear_cache.sh

ROOT="${1:-.}"
OUT="$ROOT/outputs"

if [ ! -d "$OUT" ]; then
  echo "No outputs/ folder found at: $OUT"
  exit 0
fi

find "$OUT" -type d -name "_cache" -prune -print -exec rm -rf {} +
echo "Done."
