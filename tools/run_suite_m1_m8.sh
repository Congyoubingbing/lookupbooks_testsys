#!/usr/bin/env bash
set -euo pipefail
PYTHON_EXE="${PYTHON_EXE:-python}"
AGENT_CONFIG="${AGENT_CONFIG:-config/agent.yaml}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-./outputs}"
LIBRARY_ROOT="${LIBRARY_ROOT:-../shared_assets/library}"
RUNS_DIR="${RUNS_DIR:-./runs/batch}"
SUITE_ID="${SUITE_ID:-full_lookupbooks_suite}"
WORKERS="${WORKERS:-5}"
BATCH_FILE="${BATCH_FILE:-}"

if [[ -z "$BATCH_FILE" ]]; then
  BATCH_FILE="$(find ./tests -maxdepth 1 -type f -name '*.md' | grep -i 'q0' | sort | head -n 1 || true)"
  if [[ -z "$BATCH_FILE" ]]; then
    BATCH_FILE="$(find ./tests -maxdepth 1 -type f -name '*.md' | sort | head -n 1 || true)"
  fi
fi
if [[ -z "$BATCH_FILE" ]]; then
  echo "No markdown batch file found under ./tests" >&2
  exit 1
fi

exec "$PYTHON_EXE" -u run.py batch-ask   --batch-file "$BATCH_FILE"   --agent-config "$AGENT_CONFIG"   --outputs-root "$OUTPUTS_ROOT"   --library-root "$LIBRARY_ROOT"   --runs-dir "$RUNS_DIR"   --workers "$WORKERS"   --suite M1-M8   --suite-id "$SUITE_ID"   --suite-m8-ks "1,3,5"
