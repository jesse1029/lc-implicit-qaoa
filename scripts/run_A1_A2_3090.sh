#!/usr/bin/env bash
set -euo pipefail

cd $HOME/lc_implicit_qaoa_qokit_probe_20260702/lcqaoa_project
export CUDA_VISIBLE_DEVICES=1
PY=$HOME/lc_implicit_qaoa_qokit_probe_20260702/envs/qokit-official-py311/bin/python
OUT=results/benchmark_suite_20260704_3090
mkdir -p "$OUT/logs"

echo "START_A1 $(date -Iseconds)"
"$PY" scripts/run_comparator_regime.py \
  --peer-mode none \
  --seeds 10 \
  --max-batch-states 524288 \
  --out-dir "$OUT/A1_official_comparator_regime" \
  2>&1 | tee "$OUT/logs/A1_official_comparator_regime.log"

echo "START_A2 $(date -Iseconds)"
"$PY" scripts/run_A2_training_quality.py \
  --seeds 10 \
  --inits 5 \
  --max-batch-states 524288 \
  --out-dir "$OUT/A2_training_quality" \
  2>&1 | tee "$OUT/logs/A2_training_quality.log"

echo "DONE $(date -Iseconds)"
