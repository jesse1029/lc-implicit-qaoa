#!/usr/bin/env bash
set -euxo pipefail

PREFIX="${PREFIX:-$HOME/lc_implicit_qaoa_peers}"
COMMIT="${CUAOA_COMMIT:-33a3b2fbb16631c03fb9dff1c43a901ff11d429f}"
SOURCE_REPO="${CUAOA_SOURCE_REPO:-$PREFIX/src/cuaoa}"
PATCHED_REPO="${CUAOA_PATCHED_REPO:-$PREFIX/src/cuaoa-device-info-patch-${COMMIT:0:7}}"
VENV="${CUAOA_PATCHED_VENV:-$PREFIX/venvs/cuaoa-device-info-patch-py312}"
BUILD_PREFIX="${CUAOA_BUILD_PREFIX:-$PREFIX/cuaoa-build-prefix}"
WHEEL_DIR="${CUAOA_PATCHED_WHEEL_DIR:-$PREFIX/device-info-patch-wheel-20260712}"

export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
export TMPDIR="$PREFIX/tmp"
mkdir -p "$WHEEL_DIR" "$TMPDIR"

if [[ ! -e "$PATCHED_REPO/.git" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$PATCHED_REPO" "$COMMIT"
fi
test "$(git -C "$PATCHED_REPO" rev-parse HEAD)" = "$COMMIT"

DEVICE_INFO="$PATCHED_REPO/cuaoa/internal/src/wrapper/device_info.cpp"
if grep -q 'cudaGetDeviceProperties_v2' "$DEVICE_INFO"; then
  sed -i 's/cudaGetDeviceProperties_v2/cudaGetDeviceProperties/g' "$DEVICE_INFO"
fi
git -C "$PATCHED_REPO" diff --check
test "$(git -C "$PATCHED_REPO" status --porcelain | wc -l)" -eq 1
git -C "$PATCHED_REPO" diff -- cuaoa/internal/src/wrapper/device_info.cpp \
  > "$WHEEL_DIR/device_info_compatibility.patch"
sha256sum "$WHEEL_DIR/device_info_compatibility.patch" \
  > "$WHEEL_DIR/patch_sha256.txt"

python3.12 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade \
  pip wheel setuptools maturin numpy==1.26.4

cd "$PATCHED_REPO"
export CONDA_PREFIX="$BUILD_PREFIX"
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:${LD_LIBRARY_PATH:-}"
{
  printf 'commit=%s\n' "$COMMIT"
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
  --release --skip-auditwheel --interpreter "$VENV/bin/python" --out "$WHEEL_DIR"
WHEEL="$(find "$WHEEL_DIR" -maxdepth 1 -name 'pycuaoa-0.1.0-*.whl' | sort | head -1)"
test -n "$WHEEL"
sha256sum "$WHEEL" > "$WHEEL_DIR/wheel_sha256.txt"
"$VENV/bin/python" -m pip install --force-reinstall "$WHEEL"
{
  echo '--- post-build installed environment ---'
  sha256sum "$WHEEL"
  "$VENV/bin/python" -m pip freeze
} >> "$WHEEL_DIR/environment.txt"

cd /tmp
export LD_LIBRARY_PATH="$PATCHED_REPO/cuaoa/internal/lib:$BUILD_PREFIX/lib:${LD_LIBRARY_PATH:-}"
"$VENV/bin/python" - <<'PY' > "$WHEEL_DIR/import_smoke.txt"
import hashlib
from pathlib import Path

import numpy as np
import pycuaoa
import pycuaoa._core

core = Path(pycuaoa._core.__file__)
print(f"package={pycuaoa.__file__}")
print(f"core={core}")
print(f"core_sha256={hashlib.sha256(core.read_bytes()).hexdigest()}")
sim = pycuaoa.CUAOA.from_map(2, {(0,): 1.0, (1,): 1.0, (0, 1): -2.0}, depth=2)
handle = pycuaoa.create_handle(2)
try:
    gradients, value = sim.gradients(
        handle,
        betas=np.asarray([0.1, 0.2]),
        gammas=np.asarray([-0.3, -0.4]),
    )
    print(f"value={value}")
    print(f"beta_gradient={np.asarray(gradients.betas).tolist()}")
    print(f"gamma_gradient={np.asarray(gradients.gammas).tolist()}")
finally:
    handle.destroy()
PY

echo PATCHED_BUILD_SUCCESS
