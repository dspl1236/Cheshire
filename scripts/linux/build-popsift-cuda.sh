#!/usr/bin/env bash
# Cheshire Linux: build PopSIFT with the CUDA backend, for the CUDA AliceVision build.
#
# Nothing is ported here. PopSIFT is a CUDA project to begin with - third_party/popsift is a
# clone of alicevision/popsift v0.10.0, and scripts/apply_popsift_patch.py is what adapts it *to*
# HIP. So the CUDA build wants the pristine upstream source, which is the easy direction.
#
# Why this exists: every libpopsift.so on the development machines was a HIP build (they declare
# libamdhip64), so the CUDA AliceVision had ALICEVISION_USE_POPSIFT=OFF and died on
# --describerTypes sift with a bare std::runtime_error and no message (2026-09-21).
#
# Usage: scripts/linux/build-popsift-cuda.sh [arch]
#   arch default 61: both NVIDIA test cards are compute 6.1 (docs/18).
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/third_party/popsift"
ARCH="${1:-61}"
CUDA="${CUDA_PATH:-/usr/local/cuda-12.9}"
DEPS="${AV_DEPS:-/opt/AliceVision_deps}"
BLD="${POPSIFT_BUILD:-$HOME/popsift-cuda-build}"
INST="${POPSIFT_INSTALL:-/opt/popsift-cuda}"
JOBS="${JOBS:-$(nproc)}"

[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA/bin/nvcc - set CUDA_PATH" >&2; exit 1; }

if [ ! -d "$SRC/.git" ]; then
  echo "[popsift] cloning alicevision/popsift v0.10.0 into $SRC"
  git clone --depth 1 --branch v0.10.0 https://github.com/alicevision/popsift.git "$SRC"
else
  # a HIP build may have left the tree patched; CUDA wants it pristine
  git -C "$SRC" checkout -- . 2>/dev/null || true
fi
echo "[popsift] source $(git -C "$SRC" log --oneline -1)"
echo "[popsift] $("$CUDA/bin/nvcc" --version | sed -n 4p), sm_$ARCH"

rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INST" \
  -DCMAKE_PREFIX_PATH="$DEPS;$CUDA" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DPopSift_BUILD_EXAMPLES=OFF \
  -DPopSift_BUILD_DOCS=OFF \
  -DBUILD_SHARED_LIBS=ON

cmake --build "$BLD" -j "$JOBS"
cmake --install "$BLD"

echo "=== installed ==="
L=$(find "$INST" -name "libpopsift.so*" -type f | head -1)
echo "  $L"
# a HIP libpopsift declares libamdhip64; this one must declare libcudart
ldd "$L" | grep -iE "cudart|amdhip" | sed 's/^/    /'
echo "  cmake package: $(find "$INST" -name PopSiftConfig.cmake | head -1)"
echo
echo "Now rebuild AliceVision: scripts/linux/build-alicevision-cuda.sh (picks it up automatically)"
