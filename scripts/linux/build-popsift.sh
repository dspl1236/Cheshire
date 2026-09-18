#!/usr/bin/env bash
# Build PopSIFT (alicevision/popsift v0.10.0) as a HIP library for AMD, for GPU SIFT on Linux.
#
# Usage: scripts/linux/build-popsift.sh [name] [archs]
#   name   names the build and install directories (default: linux)
#   archs  semicolon list of code objects (default: the full RDNA1 to RDNA4 range)
#
# Installs to build/popsift-<name>-install. Point AliceVision's CHESHIRE_POPSIFT_DIR /
# -DPopSift_DIR at <install>/lib/cmake/PopSift.
#
# Unlike Windows, one fat library covers everything: the Linux ROCm runtime enumerates RDNA1 and
# RDNA2 natively, so they do not need the separate HIP 6.2 toolchain that Windows requires.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROCM="${ROCM_PATH:-/opt/rocm}"
NAME="${1:-linux}"
ARCHS="${2:-gfx1010;gfx1012;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201}"
BLD="$ROOT/build/popsift-$NAME"
INST="$ROOT/build/popsift-$NAME-install"

# ERRCHK checks after every kernel launch; off for release builds, on when bringing up a target.
ERRCHK="${CHESHIRE_POPSIFT_ERRCHK:-OFF}"

echo "[popsift] name=$NAME"
echo "[popsift] archs=$ARCHS"
echo "[popsift] rocm=$ROCM errchk=$ERRCHK"
echo "[popsift] install=$INST"

# the generated sift_config.h, the source fixes and the unity translation unit
python3 "$ROOT/scripts/apply_popsift_patch.py"

cmake -G Ninja -S "$ROOT/hip/port/popsift" -B "$BLD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INST" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM" \
  -DCHESHIRE_POPSIFT_ARCH="$ARCHS" \
  -DCHESHIRE_POPSIFT_ERRCHK="$ERRCHK"

cmake --build "$BLD" --target install

echo "[popsift] done: $INST"
ls -l "$INST/lib/"libpopsift* 2>/dev/null || true
