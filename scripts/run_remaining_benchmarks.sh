#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$HOME/lc_implicit_qaoa_peers/venvs/lcqaoa-core/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT="results/benchmark_suite_20260704"
mkdir -p "$OUT/logs"

echo "[START] $(date -Is) A5 QUBO generality"
"$PY" scripts/run_A5_qubo_generality.py \
  --out-dir "$OUT/A5_qubo_generality" \
  --max-batch-states 524288 \
  2>&1 | tee "$OUT/logs/A5_qubo_generality.log"

echo "[START] $(date -Is) A6 ablation"
"$PY" scripts/run_A6_extended_ablation.py \
  --out-dir "$OUT/A6_ablation" \
  --max-batch-states 524288 \
  2>&1 | tee "$OUT/logs/A6_ablation.log"

echo "[START] $(date -Is) A8 sampling handoff"
"$PY" scripts/run_A8_sampling_handoff.py \
  --seeds 5 \
  --steps 80 \
  --samples 4096 \
  --out-dir "$OUT/A8_sampling_handoff" \
  2>&1 | tee "$OUT/logs/A8_sampling_handoff.log"

echo "[START] $(date -Is) comparator LC/status matrix"
"$PY" scripts/run_comparator_regime.py \
  --peer-mode none \
  --seeds 10 \
  --max-batch-states 524288 \
  --out-dir "$OUT/A1_official_comparator_regime" \
  2>&1 | tee "$OUT/logs/A1_official_comparator_regime.log"

echo "[START] $(date -Is) A2 end-to-end training quality"
"$PY" scripts/run_A2_training_quality.py \
  --seeds 10 \
  --inits 5 \
  --max-batch-states 524288 \
  --out-dir "$OUT/A2_training_quality" \
  2>&1 | tee "$OUT/logs/A2_training_quality.log"

echo "[START] $(date -Is) A9 artifact integrity refresh"
"$PY" scripts/make_A9_artifact_integrity.py \
  --out-dir "$OUT/A9_artifact_integrity" \
  2>&1 | tee "$OUT/logs/A9_artifact_integrity.log"

echo "[START] $(date -Is) A10 theorem blocks refresh"
"$PY" scripts/write_A10_theorem_blocks.py \
  --out-dir "$OUT/A10_method_theorems" \
  2>&1 | tee "$OUT/logs/A10_method_theorems.log"

echo "[DONE] $(date -Is) remaining benchmark suite"
