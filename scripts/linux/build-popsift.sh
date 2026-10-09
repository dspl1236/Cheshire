#!/usr/bin/env bash
# Build PopSIFT (alicevision/popsift v0.10.0) as a HIP library for AMD, for GPU SIFT on Linux.
#
# Usage: scripts/linux/build-popsift.sh [name] [archs]
#   name   names the build and install directories (default: linux)
#   archs  semicolon list of code objects (default, 0.4.4: RDNA1 by name - gfx1010 to gfx1013, a family no chip will
#          join - and the generic target of each later family, gfx10-3-generic (RDNA2, the APUs included),
#          gfx11-generic (RDNA3/3.5) and gfx12-generic (RDNA4), so a chip of those families that ships later has GPU
#          SIFT without a rebuild. PopSIFT runs from a generic code object since apply_popsift_patch.py 3f moved the
#          grid filter off rocThrust (docs/16); before that the list named 21 chips. No Vega: wave64, and PopSIFT's
#          kernels have only ever run wave32)
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
ARCHS="${2:-gfx1010;gfx1011;gfx1012;gfx1013;gfx10-3-generic;gfx11-generic;gfx12-generic}"
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
