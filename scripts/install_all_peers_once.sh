#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot installer for LC-Implicit-QAOA peer baselines.
#
# Intended usage inside tmux on the GPU host:
#   cd $HOME/lc_implicit_qaoa_20260630
#   bash scripts/install_all_peers_once.sh
#
# It asks for sudo once at the beginning, keeps sudo alive while it runs,
# installs every configured peer environment, and writes one log file.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/lc_implicit_qaoa_peers}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/results/install_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/install_all_peers_${STAMP}.log}"
STATUS_FILE="${STATUS_FILE:-$LOG_DIR/install_all_peers.status}"
PROBE_JSON="${PROBE_JSON:-$PROJECT_ROOT/results/peer_probe_latest.json}"
PROBE_MD="${PROBE_MD:-$PROJECT_ROOT/results/peer_probe_latest.md}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_MPS_JULIQAOA="${RUN_MPS_JULIQAOA:-1}"
CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit}"
MPS_JULIQAOA_REPO_URL="${MPS_JULIQAOA_REPO_URL:-https://github.com/lanl/JuliQAOA.jl}"
MPS_JULIQAOA_REV="${MPS_JULIQAOA_REV:-mps}"

export PREFIX CUDA_TOOLKIT_PACKAGE MPS_JULIQAOA_REPO_URL MPS_JULIQAOA_REV
export PATH="/usr/local/cuda/bin:$HOME/.juliaup/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/results"
touch "$LOG_FILE"
ln -sfn "$LOG_FILE" "$LOG_DIR/install_all_peers.latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

mark_status() {
  printf '%s | %s\n' "$(date --iso-8601=seconds)" "$*" | tee "$STATUS_FILE" >/dev/null
}

cleanup() {
  if [[ -n "${SUDO_KEEPALIVE_PID:-}" ]]; then
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi
}

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  mark_status "FAILED line=${line_no} exit=${exit_code} log=$LOG_FILE"
  echo
  echo "FAILED at line ${line_no}, exit ${exit_code}"
  echo "Log: $LOG_FILE"
  exit "$exit_code"
}

trap 'on_error $LINENO' ERR
trap cleanup EXIT

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  mark_status "RUNNING ${name}"
  "$@"
  mark_status "DONE ${name}"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

echo "LC-Implicit-QAOA one-shot peer installer"
echo "Project root : $PROJECT_ROOT"
echo "Prefix       : $PREFIX"
echo "Log          : $LOG_FILE"
echo "Status       : $STATUS_FILE"
echo "Started      : $(date --iso-8601=seconds)"

mark_status "STARTED log=$LOG_FILE"

if ! have_cmd sudo; then
  echo "sudo is required for system packages, but sudo was not found."
  exit 1
fi

echo
echo "Requesting sudo once. The script will keep the credential alive until it exits."
sudo -v
(
  while true; do
    sudo -n true >/dev/null 2>&1 || exit 0
    sleep 45
  done
) &
SUDO_KEEPALIVE_PID=$!

run_step "system packages" bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --system --no-probe

if have_cmd nvcc; then
  echo
  echo "===== cuda toolkit ====="
  echo "nvcc already present: $(command -v nvcc)"
  nvcc --version | head -n 4 || true
  mark_status "DONE cuda toolkit already-present"
else
  run_step "cuda toolkit" bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --cuda-toolkit --no-probe
  export PATH="/usr/local/cuda/bin:$PATH"
fi

run_step "core python peers" bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --core --no-probe
run_step "qtensor env" bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --qtensor --no-probe
run_step "cuaoa build" bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --cuaoa --no-probe
run_step "julia runtime" bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --julia --no-probe
run_step "juliqaoa main" env MPS_JULIQAOA_REPO_URL= bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --juliqaoa --no-probe

if [[ "$RUN_MPS_JULIQAOA" == "1" ]]; then
  run_step "mps-juliqaoa branch" bash -lc '
    set -euo pipefail
    export PATH="$HOME/.juliaup/bin:$PATH"
    mkdir -p "$PREFIX/julia-mps"
    julia --project="$PREFIX/julia-mps" "$PROJECT_ROOT/scripts/install_mps_juliqaoa.jl"
  '
else
  echo "RUN_MPS_JULIQAOA=0; skipped MPS-JuliQAOA branch install."
fi

run_step "optional repos" bash "$PROJECT_ROOT/scripts/install_missing_peers.sh" --optional-repos --no-probe

run_step "peer probe" python3 "$PROJECT_ROOT/scripts/probe_all_peers.py" \
  --prefix "$PREFIX" \
  --out "$PROBE_JSON" \
  --markdown "$PROBE_MD"

if [[ "$RUN_SMOKE" == "1" && -x "$PREFIX/venvs/lcqaoa-core/bin/python" ]]; then
  run_step "lcqaoa smoke test" "$PREFIX/venvs/lcqaoa-core/bin/python" "$PROJECT_ROOT/scripts/run_smoke.py"
fi

mark_status "COMPLETE log=$LOG_FILE probe=$PROBE_MD"

echo
echo "All configured install steps completed."
echo "Log   : $LOG_FILE"
echo "Probe : $PROBE_MD"
echo "Status: $STATUS_FILE"
