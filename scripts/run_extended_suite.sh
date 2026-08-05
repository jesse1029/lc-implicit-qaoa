#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/lc_implicit_qaoa_peers/venvs/lcqaoa-core/bin/python}"

cd "$ROOT"
mkdir -p results

if [[ "${WAIT_FOR_GPU_FREE:-1}" == "1" ]]; then
  threshold="${GPU_FREE_THRESHOLD_MB:-2000}"
  echo "[0/6] Waiting for GPU memory.used < ${threshold} MiB"
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

echo "[1/6] Exact adjoint-gradient benchmark"
"$PY" scripts/run_adjoint_gradient_benchmarks.py \
  --out results/adjoint_gradient_benchmark.csv \
  --markdown results/adjoint_gradient_benchmark.md

echo "[2/6] Batching/topology ablation"
"$PY" scripts/run_topology_ablation.py \
  --out results/topology_ablation.csv \
  --markdown results/topology_ablation.md

echo "[3/6] Real-data sparse QUBO case study"
"$PY" scripts/run_real_qubo_case_study.py \
  --out results/real_qubo_case_study.csv \
  --markdown results/real_qubo_case_study.md

echo "[4/7] Biomedical feature-selection benchmark"
"$PY" scripts/run_biomedical_feature_selection.py \
  --out-runtime results/biomedical_runtime.csv \
  --out-selectors results/biomedical_selectors.csv \
  --markdown results/biomedical_feature_selection.md \
  --include-openml

echo "[5/7] Paper figures"
"$PY" scripts/make_paper_figures.py

echo "[6/7] Paper tables"
"$PY" scripts/make_paper_tables.py

echo "[7/7] reproducibility summary"
"$PY" scripts/make_reproducibility_report.py

echo "STRENGTHENED_SUITE_DONE"
