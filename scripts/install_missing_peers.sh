#!/usr/bin/env bash
set -euo pipefail

# Install helper for LC-Implicit-QAOA peer baselines on Ubuntu GPU hosts.
#
# Design:
# - Default action is non-system and user-space only.
# - Python dependency conflicts are isolated into separate virtualenvs.
# - System-changing installs require explicit flags.
# - Existing venvs/repos are reused; nothing is deleted.
#
# Examples:
#   bash scripts/install_missing_peers.sh --core
#   bash scripts/install_missing_peers.sh --system --core
#   bash scripts/install_missing_peers.sh --cuda-toolkit --cuaoa
#   bash scripts/install_missing_peers.sh --julia --juliqaoa
#   bash scripts/install_missing_peers.sh --qtensor
#   bash scripts/install_missing_peers.sh --all

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/lc_implicit_qaoa_peers}"
export PATH="/usr/local/cuda/bin:$HOME/.juliaup/bin:$HOME/.juliaup/bin:$HOME/.cargo/bin:$HOME/.cargo/bin:$PATH"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CORE_VENV="${CORE_VENV:-$PREFIX/venvs/lcqaoa-core}"
QTENSOR_VENV="${QTENSOR_VENV:-$PREFIX/venvs/qtensor-py310}"
CUAOA_VENV="${CUAOA_VENV:-$PREFIX/venvs/cuaoa-py312}"
CUAOA_BUILD_PREFIX="${CUAOA_BUILD_PREFIX:-$PREFIX/cuaoa-build-prefix}"
SRC_DIR="${SRC_DIR:-$PREFIX/src}"

QTENSOR_REPO_URL="${QTENSOR_REPO_URL:-https://github.com/DaniloZZZ/QTensor.git}"
CUAOA_REPO_URL="${CUAOA_REPO_URL:-https://github.com/JFLXB/cuaoa.git}"
JULIQAOA_REPO_URL="${JULIQAOA_REPO_URL:-https://github.com/lanl/JuliQAOA.jl}"

# Optional: set these if/when the authors publish installable repositories.
MPS_JULIQAOA_REPO_URL="${MPS_JULIQAOA_REPO_URL:-}"
BMQSIM_REPO_URL="${BMQSIM_REPO_URL:-}"
QUEENV2_REPO_URL="${QUEENV2_REPO_URL:-}"

DO_SYSTEM=0
DO_CUDA_TOOLKIT=0
DO_CORE=0
DO_QTENSOR=0
DO_CUAOA=0
DO_JULIA=0
DO_JULIQAOA=0
DO_OPTIONAL_REPOS=0
DO_PROBE=1
DRY_RUN=0

usage() {
  cat <<'USAGE'
Install helper for LC-Implicit-QAOA peer baselines on Ubuntu GPU hosts.

Examples:
  bash scripts/install_missing_peers.sh --core
  bash scripts/install_missing_peers.sh --system --core
  bash scripts/install_missing_peers.sh --cuda-toolkit --cuaoa
  bash scripts/install_missing_peers.sh --julia --juliqaoa
  bash scripts/install_missing_peers.sh --qtensor
  bash scripts/install_missing_peers.sh --all

Flags:
  --system          sudo apt install base build tools and python venv support
  --cuda-toolkit    sudo install NVIDIA CUDA toolkit via NVIDIA Ubuntu 24.04 repo
  --core            create core Python env: CuPy, CUDA-Q/cudaq, QOKit, qblaze
  --qtensor         create isolated QTensor env; prefers Python 3.10 via uv
  --cuaoa           clone/build CUAOA if nvcc and Rust toolchain are available
  --julia           install Julia with juliaup if julia is absent
  --juliqaoa        install JuliQAOA.jl; optional MPS repo via MPS_JULIQAOA_REPO_URL
  --optional-repos  clone optional BMQSim/QueenV2 repos if URL env vars are set
  --all             run all install groups above
  --no-probe        skip final peer_status probe
  --dry-run         print commands without executing
  -h, --help        show this help

Useful env vars:
  PREFIX=/path/to/env-root
  PYTHON_BIN=python3.12
  QTENSOR_REPO_URL=https://github.com/DaniloZZZ/QTensor.git
  CUAOA_REPO_URL=https://github.com/JFLXB/cuaoa.git
  JULIQAOA_REPO_URL=https://github.com/lanl/JuliQAOA.jl
  MPS_JULIQAOA_REPO_URL=<repo-url-if-known>
  BMQSIM_REPO_URL=<repo-url-if-known>
  QUEENV2_REPO_URL=<repo-url-if-known>
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) DO_SYSTEM=1 ;;
    --cuda-toolkit) DO_CUDA_TOOLKIT=1 ;;
    --core) DO_CORE=1 ;;
    --qtensor) DO_QTENSOR=1 ;;
    --cuaoa) DO_CUAOA=1 ;;
    --julia) DO_JULIA=1 ;;
    --juliqaoa) DO_JULIQAOA=1 ;;
    --optional-repos) DO_OPTIONAL_REPOS=1 ;;
    --all)
      DO_SYSTEM=1
      DO_CUDA_TOOLKIT=1
      DO_CORE=1
      DO_QTENSOR=1
      DO_CUAOA=1
      DO_JULIA=1
      DO_JULIQAOA=1
      DO_OPTIONAL_REPOS=1
      ;;
    --no-probe) DO_PROBE=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [[ $DO_SYSTEM -eq 0 && $DO_CUDA_TOOLKIT -eq 0 && $DO_CORE -eq 0 && \
      $DO_QTENSOR -eq 0 && $DO_CUAOA -eq 0 && $DO_JULIA -eq 0 && \
      $DO_JULIQAOA -eq 0 && $DO_OPTIONAL_REPOS -eq 0 ]]; then
  usage
  exit 0
fi

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

run_shell() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    bash -lc "$*"
  fi
}

ensure_dir() {
  run mkdir -p "$1"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

create_or_reuse_venv() {
  local python_bin="$1"
  local venv_path="$2"
  if [[ ! -x "$venv_path/bin/python" ]]; then
    run "$python_bin" -m venv "$venv_path"
  else
    echo "Reusing venv: $venv_path"
  fi
  run "$venv_path/bin/python" -m pip install --upgrade pip wheel setuptools
}

clone_or_update() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    run git -C "$dest" fetch --all --tags --prune
    run git -C "$dest" pull --ff-only
  else
    run git clone --recursive "$url" "$dest"
  fi
}

install_system_packages() {
  echo "Installing base build tools and Python venv support with sudo apt."
  run sudo apt-get update
  run sudo apt-get install -y \
    ca-certificates curl wget git build-essential cmake ninja-build pkg-config \
    python3-pip python3-venv python3.12-venv unzip liblbfgs-dev
}

install_cuda_toolkit() {
  echo "Installing NVIDIA CUDA toolkit repo for Ubuntu 24.04. This is needed for nvcc/CUAOA."
  local keyring="/tmp/cuda-keyring_1.1-1_all.deb"
  run wget -O "$keyring" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
  run sudo dpkg -i "$keyring"
  run sudo apt-get update
  # cuda-toolkit tracks the latest toolkit in the NVIDIA repo. Override if needed:
  #   CUDA_TOOLKIT_PACKAGE=cuda-toolkit-12-8 bash scripts/install_missing_peers.sh --cuda-toolkit
  run sudo apt-get install -y "${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit}"
  echo "After this, make sure nvcc is on PATH, e.g.: export PATH=/usr/local/cuda/bin:\$PATH"
}

install_core_python_env() {
  ensure_dir "$PREFIX/venvs"
  create_or_reuse_venv "$PYTHON_BIN" "$CORE_VENV"
  local req="$PROJECT_ROOT/requirements.txt"
  if [[ -f "$req" ]]; then
    run "$CORE_VENV/bin/python" -m pip install -r "$req"
  fi
  run "$CORE_VENV/bin/python" -m pip install cupy-cuda12x cudaq qokit qblaze
  echo "Core env ready: source $CORE_VENV/bin/activate"
}

ensure_uv() {
  if have_cmd uv; then
    return
  fi
  echo "Installing uv into user space for Python 3.10 QTensor env management."
  run_shell "curl -LsSf https://astral.sh/uv/install.sh | sh"
  export PATH="$HOME/.local/bin:$PATH"
}

install_qtensor_env() {
  ensure_dir "$PREFIX/venvs"
  ensure_dir "$SRC_DIR"
  local python310=""
  if have_cmd python3.10; then
    python310="$(command -v python3.10)"
    create_or_reuse_venv "$python310" "$QTENSOR_VENV"
  else
    ensure_uv
    if [[ ! -x "$QTENSOR_VENV/bin/python" ]]; then
      run uv venv --python 3.10 "$QTENSOR_VENV"
    else
      echo "Reusing venv: $QTENSOR_VENV"
    fi
    run "$QTENSOR_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
  fi

  local dest="$SRC_DIR/QTensor"
  clone_or_update "$QTENSOR_REPO_URL" "$dest"
  if [[ -d "$dest/qtree" ]]; then
    run "$QTENSOR_VENV/bin/python" -m pip install "$dest/qtree"
  fi
  run "$QTENSOR_VENV/bin/python" -m pip install "$dest"
  echo "QTensor env ready: source $QTENSOR_VENV/bin/activate"
}

ensure_rust() {
  if have_cmd cargo; then
    return
  fi
  echo "Installing Rust toolchain via rustup for CUAOA."
  run_shell "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
  # shellcheck disable=SC1090
  [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
}

patch_cuaoa_sources() {
  local dest="$1"
  local device_info="$dest/cuaoa/internal/src/wrapper/device_info.cpp"
  if [[ -f "$device_info" ]] && grep -q 'cudaGetDeviceProperties_v2' "$device_info"; then
    echo "Patching CUAOA CUDA device-info API for current CUDA headers."
    run sed -i 's/cudaGetDeviceProperties_v2/cudaGetDeviceProperties/g' "$device_info"
  fi
}

install_cuaoa() {
  if ! have_cmd nvcc; then
    cat <<'MSG'
nvcc is missing. Run one of:
  bash scripts/install_missing_peers.sh --cuda-toolkit
  CUDA_TOOLKIT_PACKAGE=cuda-toolkit-12-8 bash scripts/install_missing_peers.sh --cuda-toolkit
Then reopen the shell or export PATH=/usr/local/cuda/bin:$PATH.
MSG
    return 0
  fi
  ensure_rust
  ensure_dir "$SRC_DIR"
  local dest="$SRC_DIR/cuaoa"
  clone_or_update "$CUAOA_REPO_URL" "$dest"
  patch_cuaoa_sources "$dest"
  if [[ ! -x "$CUAOA_VENV/bin/python" ]]; then
    run python3 -m venv "$CUAOA_VENV"
  fi
  run "$CUAOA_VENV/bin/python" -m pip install --upgrade pip wheel setuptools maturin numpy==1.26.4

  local cuq_root=""
  find_cuq_root() {
    local root
    shopt -s nullglob
    for root in \
      "$CUAOA_VENV"/lib/python*/site-packages/cuquantum \
      "$CORE_VENV"/lib/python*/site-packages/cuquantum \
      "$HOME"/.local/lib/python*/site-packages/cuquantum \
      /usr/local/lib/python*/dist-packages/cuquantum \
      /usr/lib/python*/dist-packages/cuquantum; do
      if [[ -f "$root/include/custatevec.h" ]]; then
        printf '%s\n' "$root"
        shopt -u nullglob
        return 0
      fi
    done
    shopt -u nullglob
    return 1
  }
  cuq_root="$(find_cuq_root || true)"
  if [[ -z "$cuq_root" ]]; then
    run "$CUAOA_VENV/bin/python" -m pip install cuquantum-cu12
    if [[ $DRY_RUN -eq 1 ]]; then
      cuq_root="$CUAOA_VENV/lib/python*/site-packages/cuquantum"
    else
      cuq_root="$(find_cuq_root || true)"
    fi
  fi

  local lbfgs_header=""
  local lbfgs_lib=""
  for p in /usr/include/lbfgs.h /usr/local/include/lbfgs.h; do
    [[ -f "$p" ]] && lbfgs_header="$p" && break
  done
  for p in /usr/lib/x86_64-linux-gnu/liblbfgs.so /usr/local/lib/liblbfgs.so; do
    [[ -e "$p" ]] && lbfgs_lib="$p" && break
  done
  if [[ $DRY_RUN -eq 1 && ( -z "$lbfgs_header" || -z "$lbfgs_lib" ) ]]; then
    lbfgs_header="/usr/include/lbfgs.h"
    lbfgs_lib="/usr/lib/x86_64-linux-gnu/liblbfgs.so"
  fi
  if [[ -z "$lbfgs_header" || -z "$lbfgs_lib" ]]; then
    cat <<'MSG'
liblbfgs headers/libs are missing. On Ubuntu run:
  bash scripts/install_missing_peers.sh --system
or manually:
  sudo apt-get install -y liblbfgs-dev
MSG
    return 0
  fi

  local custatevec_header="$cuq_root/include/custatevec.h"
  local custatevec_lib=""
  if [[ $DRY_RUN -eq 1 ]]; then
    custatevec_lib="$cuq_root/lib/libcustatevec.so.1"
  else
    custatevec_lib="$(find "$cuq_root" \( -type f -o -type l \) -name 'libcustatevec.so*' 2>/dev/null | sort | head -n 1 || true)"
  fi
  if [[ ! -f "$custatevec_header" || -z "$custatevec_lib" ]]; then
    cat <<MSG
cuQuantum/cuStateVec header or library was not found under:
  $cuq_root
Try:
  $CUAOA_VENV/bin/python -m pip install --upgrade cuquantum-cu12
MSG
    return 0
  fi

  run mkdir -p "$CUAOA_BUILD_PREFIX/include" "$CUAOA_BUILD_PREFIX/lib"
  run ln -sf "$custatevec_header" "$CUAOA_BUILD_PREFIX/include/custatevec.h"
  run ln -sf "$custatevec_lib" "$CUAOA_BUILD_PREFIX/lib/libcustatevec.so"
  run ln -sf "$custatevec_lib" "$CUAOA_BUILD_PREFIX/lib/$(basename "$custatevec_lib")"
  run ln -sf "$lbfgs_header" "$CUAOA_BUILD_PREFIX/include/lbfgs.h"
  run ln -sf "$lbfgs_lib" "$CUAOA_BUILD_PREFIX/lib/liblbfgs.so"

  if [[ -f "$dest/pyproject.toml" ]]; then
    (cd "$dest" && export CONDA_PREFIX="$CUAOA_BUILD_PREFIX" LD_LIBRARY_PATH="$CUAOA_BUILD_PREFIX/lib:${LD_LIBRARY_PATH:-}" && run "$CUAOA_VENV/bin/python" -m pip install -e .) || \
    (cd "$dest" && export CONDA_PREFIX="$CUAOA_BUILD_PREFIX" LD_LIBRARY_PATH="$CUAOA_BUILD_PREFIX/lib:${LD_LIBRARY_PATH:-}" && run "$CUAOA_VENV/bin/python" -m maturin develop --release --interpreter "$CUAOA_VENV/bin/python")
  elif [[ -f "$dest/setup.py" ]]; then
    export CONDA_PREFIX="$CUAOA_BUILD_PREFIX" LD_LIBRARY_PATH="$CUAOA_BUILD_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    run "$CUAOA_VENV/bin/python" -m pip install -e "$dest"
  else
    echo "CUAOA cloned, but no pyproject.toml/setup.py was found. Inspect: $dest"
  fi
}

install_julia() {
  if have_cmd julia; then
    echo "Julia already installed: $(command -v julia)"
    return
  fi
  echo "Installing Julia via juliaup in user space."
  run_shell "curl -fsSL https://install.julialang.org | sh -s -- --yes"
  export PATH="$HOME/.juliaup/bin:$PATH"
}

install_juliqaoa() {
  if ! have_cmd julia; then
    install_julia
  fi
  run julia -e "using Pkg; Pkg.add(url=\"$JULIQAOA_REPO_URL\"); Pkg.precompile()"
  if [[ -n "$MPS_JULIQAOA_REPO_URL" ]]; then
    run julia -e "using Pkg; Pkg.add(url=\"$MPS_JULIQAOA_REPO_URL\"); Pkg.precompile()"
  else
    echo "MPS_JULIQAOA_REPO_URL is not set; skipped MPS-JuliQAOA package install."
  fi
}

clone_optional_repos() {
  ensure_dir "$SRC_DIR"
  if [[ -n "$BMQSIM_REPO_URL" ]]; then
    clone_or_update "$BMQSIM_REPO_URL" "$SRC_DIR/bmqsim"
  else
    echo "BMQSIM_REPO_URL not set; no public install repo was configured."
  fi
  if [[ -n "$QUEENV2_REPO_URL" ]]; then
    clone_or_update "$QUEENV2_REPO_URL" "$SRC_DIR/queenv2"
  else
    echo "QUEENV2_REPO_URL not set; no public install repo was configured."
  fi
}

probe_status() {
  if [[ -f "$PROJECT_ROOT/scripts/probe_all_peers.py" ]]; then
    run "$PYTHON_BIN" "$PROJECT_ROOT/scripts/probe_all_peers.py" \
      --prefix "$PREFIX" \
      --out "$PROJECT_ROOT/results/peer_probe_after_install.json" \
      --markdown "$PROJECT_ROOT/results/peer_probe_after_install.md"
  elif [[ -f "$PROJECT_ROOT/scripts/peer_status.py" ]]; then
    run "$PYTHON_BIN" "$PROJECT_ROOT/scripts/peer_status.py" \
      --out "$PROJECT_ROOT/results/peer_status_after_install.json" \
      --markdown "$PROJECT_ROOT/results/peer_status_after_install.md"
  else
    echo "No peer probe script found; skipped probe."
  fi
}

echo "Project root: $PROJECT_ROOT"
echo "Install prefix: $PREFIX"

[[ $DO_SYSTEM -eq 1 ]] && install_system_packages
[[ $DO_CUDA_TOOLKIT -eq 1 ]] && install_cuda_toolkit
[[ $DO_CORE -eq 1 ]] && install_core_python_env
[[ $DO_QTENSOR -eq 1 ]] && install_qtensor_env
[[ $DO_CUAOA -eq 1 ]] && install_cuaoa
[[ $DO_JULIA -eq 1 ]] && install_julia
[[ $DO_JULIQAOA -eq 1 ]] && install_juliqaoa
[[ $DO_OPTIONAL_REPOS -eq 1 ]] && clone_optional_repos
[[ $DO_PROBE -eq 1 ]] && probe_status

cat <<EOF

Done.

Useful activation commands:
  source "$CORE_VENV/bin/activate"
  source "$QTENSOR_VENV/bin/activate"

If CUDA toolkit was installed, you may need:
  export PATH=/usr/local/cuda/bin:\$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}

EOF
