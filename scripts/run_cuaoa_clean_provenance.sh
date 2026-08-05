#!/usr/bin/env bash
set -euxo pipefail

PREFIX="${PREFIX:-$HOME/lc_implicit_qaoa_peers}"
COMMIT="${CUAOA_COMMIT:-33a3b2fbb16631c03fb9dff1c43a901ff11d429f}"
SOURCE_REPO="${CUAOA_SOURCE_REPO:-$PREFIX/src/cuaoa}"
CLEAN_REPO="${CUAOA_CLEAN_REPO:-$PREFIX/src/cuaoa-clean-${COMMIT:0:7}}"
VENV="${CUAOA_CLEAN_VENV:-$PREFIX/venvs/cuaoa-clean-py312}"
BUILD_PREFIX="${CUAOA_BUILD_PREFIX:-$PREFIX/cuaoa-build-prefix}"
WHEEL_DIR="${CUAOA_CLEAN_WHEEL_DIR:-$PREFIX/clean-wheels-20260712}"

export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
mkdir -p "$WHEEL_DIR"

if [[ ! -e "$CLEAN_REPO/.git" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$CLEAN_REPO" "$COMMIT"
fi

test "$(git -C "$CLEAN_REPO" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$CLEAN_REPO" status --porcelain)"

python3.12 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade \
  pip wheel setuptools maturin numpy==1.26.4

cd "$CLEAN_REPO"
export CONDA_PREFIX="$BUILD_PREFIX"
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:${LD_LIBRARY_PATH:-}"

{
  git remote -v
  git log -1 --format=fuller
  git status --porcelain
  nvcc --version
  rustc --version
  g++ --version | head -1
  "$VENV/bin/python" --version
  "$VENV/bin/python" -m pip freeze
} > "$WHEEL_DIR/environment.txt"

"$VENV/bin/python" -m maturin build \
  --release --interpreter "$VENV/bin/python" --out "$WHEEL_DIR"

WHEEL="$(find "$WHEEL_DIR" -maxdepth 1 -name 'pycuaoa-0.1.0-*.whl' | sort | head -1)"
test -n "$WHEEL"
sha256sum "$WHEEL" > "$WHEEL_DIR/wheel_sha256.txt"
"$VENV/bin/python" -m pip install --force-reinstall "$WHEEL"

cd /tmp
"$VENV/bin/python" - <<'PY'
import pycuaoa

print(pycuaoa.__file__)
print(pycuaoa.get_devices_info())
PY

echo CLEAN_BUILD_SUCCESS
