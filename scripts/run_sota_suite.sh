#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/lc_implicit_qaoa_peers/venvs/lcqaoa-core/bin/python}"

cd "$ROOT"
mkdir -p results

echo "[1/4] GPU and environment"
hostname | tee results/sota_suite_host.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee results/sota_suite_gpu.txt || true

echo "[2/4] Extended sparse scaling"
"$PY" scripts/run_sota_sparse_scale.py \
  --out results/sota_sparse_scale.csv \
  --markdown results/sota_sparse_scale.md \
  --include-naive

echo "[3/4] Gradient benchmark"
"$PY" scripts/run_gradient_benchmarks.py \
  --out results/gradient_benchmark.csv \
  --markdown results/gradient_benchmark.md

echo "[4/4] SOTA report"
"$PY" scripts/make_sota_report.py

echo "SOTA_SUITE_DONE"
