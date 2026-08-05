#!/usr/bin/env bash
set -u

project="<HOME>/lc_implicit_qaoa_20260630"
python="<HOME>/miniconda3/envs/lcqaoa-exp/bin/python"
out="${1:?output directory required}"
gpu_index="${2:-1}"
cpu_set="${3:-80-95}"

cd "$project" || exit 2
if [[ -e "$out" ]]; then
  echo "Refusing to overwrite existing output: $out" >&2
  exit 3
fi
mkdir -p "$out"

{
  date --iso-8601=seconds
  hostname
  uptime
  nproc
  nvidia-smi
  ps -eo user,pid,psr,pcpu,pmem,etimes,cmd --sort=-pcpu | head -40
} > "$out/pre_run_state.txt" 2>&1

initial_apps="$(
  nvidia-smi -i "$gpu_index" \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null
)"
if [[ -n "$initial_apps" ]]; then
  printf '%s\n' "$initial_apps" > "$out/contaminating_processes.txt"
  echo "GPU $gpu_index is not clean; replication not started." >&2
  echo 90 > "$out/exit_code.txt"
  exit 90
fi

taskset -c "$cpu_set" env CUDA_VISIBLE_DEVICES="$gpu_index" \
  "$python" scripts/run_tensor_plan_reuse_dispatch.py \
  --out-dir "$out/results" \
  --query-count 10 \
  --seeds 5 \
  --timeout 300 \
  --global-max-n 26 \
  --skip-qtensor \
  --gpu-order alternate \
  --case-filter small_global \
  --case-filter near_crossover \
  --case-filter moderate_grid \
  > "$out/run.log" 2>&1 &
experiment_pid=$!
echo "$experiment_pid" > "$out/experiment_pid.txt"

(
  echo "timestamp,index,memory_used_mib,utilization_percent,temperature_c,pstate,sm_clock_mhz,memory_clock_mhz,power_w"
  while kill -0 "$experiment_pid" 2>/dev/null; do
    timestamp="$(date --iso-8601=ns)"
    sample="$(
      nvidia-smi -i "$gpu_index" \
        --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw \
        --format=csv,noheader,nounits 2>/dev/null
    )"
    printf '%s,%s\n' "$timestamp" "$sample"
    sleep 1
  done
) > "$out/gpu_trace.csv" 2>&1 &
monitor_pid=$!

(
  echo "timestamp|compute_processes"
  while kill -0 "$experiment_pid" 2>/dev/null; do
    timestamp="$(date --iso-8601=ns)"
    apps="$(
      nvidia-smi -i "$gpu_index" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | paste -sd ';' -
    )"
    printf '%s|%s\n' "$timestamp" "$apps"
    sleep 1
  done
) > "$out/gpu_process_trace.txt" 2>&1 &
process_monitor_pid=$!

wait "$experiment_pid"
rc=$?
wait "$monitor_pid" 2>/dev/null || true
wait "$process_monitor_pid" 2>/dev/null || true
echo "$rc" > "$out/exit_code.txt"

{
  date --iso-8601=seconds
  nvidia-smi
  nvidia-smi -i "$gpu_index" \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits
} > "$out/post_run_state.txt" 2>&1

exit "$rc"
