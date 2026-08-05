#!/usr/bin/env bash
set -euo pipefail

cd $HOME/lc_implicit_qaoa_qokit_probe_20260702/lcqaoa_project
export CUDA_VISIBLE_DEVICES=0
PY=$HOME/lc_implicit_qaoa_qokit_probe_20260702/envs/qokit-official-py311/bin/python
OUT=results/benchmark_suite_20260704_3090
mkdir -p "$OUT/logs"

echo "START_A2_NON3REG $(date -Iseconds)"
"$PY" scripts/run_A2_training_quality.py \
  --seeds 10 \
  --inits 5 \
  --max-batch-states 524288 \
  --families weighted_sparse_qubo qubo_modular_sparse er_deg3 scale_free \
  --out-dir "$OUT/A2_training_quality_non3reg" \
  2>&1 | tee "$OUT/logs/A2_training_quality_non3reg.log"
echo "DONE_A2_NON3REG $(date -Iseconds)"
