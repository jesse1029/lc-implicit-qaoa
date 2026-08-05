#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/lcqaoa_followup_20260712"
OUT="$ROOT/results/lightning_c64"
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT/repo"

set +e
"$ROOT/.venv_lightning/bin/python" scripts/run_P0_2_lightning_gpu_adjoint.py \
  --cases-dir cases \
  --output-dir "$OUT" \
  --host-label rtx3090_gpu1 \
  --gpu 1 \
  --repeats 3 \
  >"$OUT/run.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$OUT/exit_code"
exit "$code"
