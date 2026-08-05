#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/lc_implicit_qaoa_peers/venvs/lcqaoa-core/bin/python}"
LOG="$ROOT/results/biomedical_suite.log"

cd "$ROOT"
mkdir -p results
: > "$LOG"

if [[ "${WAIT_FOR_GPU_FREE:-1}" == "1" ]]; then
  threshold="${GPU_FREE_THRESHOLD_MB:-2000}"
  echo "[0/4] Waiting for GPU memory.used < ${threshold} MiB" | tee -a "$LOG"
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    echo "GPU memory.used=${used} MiB util=${util}%" | tee -a "$LOG"
    if [[ "${used:-999999}" =~ ^[0-9]+$ ]] && [[ "$used" -lt "$threshold" ]]; then
      break
    fi
    sleep 60
  done
fi

echo "[1/4] Biomedical feature-selection benchmark" | tee -a "$LOG"
"$PY" scripts/run_biomedical_feature_selection.py \
  --out-runtime results/biomedical_runtime.csv \
  --out-selectors results/biomedical_selectors.csv \
  --markdown results/biomedical_feature_selection.md \
  --include-openml 2>&1 | tee -a "$LOG"

echo "[2/4] Paper figures" | tee -a "$LOG"
"$PY" scripts/make_paper_figures.py 2>&1 | tee -a "$LOG"

echo "[3/4] Paper tables" | tee -a "$LOG"
"$PY" scripts/make_paper_tables.py 2>&1 | tee -a "$LOG"

echo "[4/4] reproducibility summary" | tee -a "$LOG"
"$PY" scripts/make_reproducibility_report.py 2>&1 | tee -a "$LOG"

echo "BIOMEDICAL_SUITE_DONE" | tee -a "$LOG"
