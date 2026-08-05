#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/lc_implicit_qaoa_peers/venvs/lcqaoa-core/bin/python}"

cd "$ROOT"
mkdir -p results

if [[ "${WAIT_FOR_GPU_FREE:-1}" == "1" ]]; then
  threshold="${GPU_FREE_THRESHOLD_MB:-2000}"
  echo "[0/5] Waiting for GPU memory.used < ${threshold} MiB"
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    echo "GPU memory.used=${used} MiB util=${util}%"
    if [[ "${used:-999999}" =~ ^[0-9]+$ ]] && [[ "$used" -lt "$threshold" ]]; then
      break
    fi
    sleep 60
  done
fi

echo "[1/5] Weighted QUBO benchmark"
"$PY" scripts/run_qubo_benchmarks.py \
  --out results/qubo_benchmark.csv \
  --markdown results/qubo_benchmark.md

echo "[2/5] Multi-seed robustness"
"$PY" scripts/run_multiseed_stats.py \
  --out results/multiseed_stats.csv \
  --markdown results/multiseed_stats.md

echo "[3/5] Optimization-loop benchmark"
"$PY" scripts/run_optimization_benchmarks.py \
  --out results/optimization_benchmark.csv \
  --markdown results/optimization_benchmark.md

echo "[4/5] Paper figures"
"$PY" scripts/make_paper_figures.py

echo "[5/6] Paper tables"
"$PY" scripts/make_paper_tables.py

echo "[6/6] reproducibility summary"
"$PY" scripts/make_reproducibility_report.py

echo "AAAI_COMPLETION_SUITE_DONE"
