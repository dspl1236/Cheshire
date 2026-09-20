#!/usr/bin/env bash
# Cheshire Linux: build AliceVision with the CUDA backend, from the same patched tree the HIP build
# uses, against the same superbuild dependencies (scripts/linux/build-deps.sh).
#
# Usage: scripts/linux/build-alicevision-cuda.sh [configure|build|install] [arch list]
#   arch list default 61: both test cards (GTX 1080 Ti, GTX 1050 Ti) are compute 6.1, and CUDA 12.9
#   is the LAST toolkit that supports Pascal - 13 removed Maxwell, Pascal and Volta (docs/18).
#
# Nothing is ported for this. The GPU sources added in v0.2.5-v0.2.16 are CUDA dialect; the HIP
# build reaches them by force-including cheshire/cuda_to_hip.h through a
# $<COMPILE_LANGUAGE:HIP> generator expression, which a CUDA build never receives. And the patch's
# HIP block is gated on "no CUDA toolkit found", so ALICEVISION_USE_CUDA=ON skips it entirely.
#
# The memory bridge IS in this build (since 2026-09-21). bridge.h reaches CUDA through
# cheshire/hip_to_cuda.h, and memory.hpp routes its five device alloc/free sites to it. Disable at
# run time with CHESHIRE_BRIDGE=0, or pick CUDA unified memory instead with
# CHESHIRE_CUDA_MANAGED=1; measurements for all three in docs/18.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AV_DEV="$ROOT/third_party/aliceVision"
STEP="${1:-install}"
ARCHS="${2:-61}"
CUDA="${CUDA_PATH:-/usr/local/cuda-12.9}"
AV_DEPS="${AV_DEPS:-/opt/AliceVision_deps}"
AV_BUILD="${AV_BUILD:-$HOME/av-cuda-build}"
AV_INSTALL="${AV_INSTALL:-/opt/AliceVision_cuda}"
AV_BUNDLE="${AV_BUNDLE:-$AV_INSTALL/bundle}"
JOBS="${JOBS:-$(nproc)}"

# GPU SIFT. PopSIFT is a CUDA project to begin with (third_party/popsift is a clone of
# alicevision/popsift v0.10.0; scripts/apply_popsift_patch.py is what adapts it *to* HIP), so this
# needs a build rather than a port - see scripts/linux/build-popsift-cuda.sh. Built OFF until
# 2026-09-21, which made the bundle die on --describerTypes sift with a bare std::runtime_error.
POPSIFT="${CHESHIRE_POPSIFT:-ON}"
POPSIFT_DIR="${CHESHIRE_POPSIFT_DIR:-/opt/popsift-cuda/lib/cmake/PopSift}"
if [ "$POPSIFT" = "ON" ] && [ ! -f "$POPSIFT_DIR/PopSiftConfig.cmake" ]; then
  echo "[popsift] no PopSiftConfig.cmake under $POPSIFT_DIR - building without GPU SIFT" >&2
  echo "[popsift] build it with scripts/linux/build-popsift-cuda.sh, or set CHESHIRE_POPSIFT=OFF" >&2
  POPSIFT=OFF; POPSIFT_DIR=""
fi
[ "$POPSIFT" = "ON" ] && echo "[popsift] GPU SIFT from $POPSIFT_DIR"

# fixup_bundle resolves every DT_NEEDED by name against these paths, so a library that is not
# listed becomes "cannot resolve item ... READ_ELF given FILE that does not exist" and the
# bundle target fails outright (seen the moment PopSIFT was switched on, 2026-09-21).
# PopSIFT installs outside the deps prefix, so it has to be added explicitly.
BUNDLE_LIB_PATHS="$AV_DEPS/lib;$CUDA/lib64"
if [ "$POPSIFT" = "ON" ]; then
  POPSIFT_LIB="$(cd "$POPSIFT_DIR/../.." 2>/dev/null && pwd)"
  [ -n "$POPSIFT_LIB" ] && BUNDLE_LIB_PATHS="$BUNDLE_LIB_PATHS;$POPSIFT_LIB"
fi

# Beats upstream's FORCEd "all-major" (patch step 5b), which would compile every .cu five times.
export CHESHIRE_CUDA_ARCHS="$ARCHS"

if [ ! -x "$CUDA/bin/nvcc" ]; then
  echo "no nvcc at $CUDA/bin/nvcc - set CUDA_PATH" >&2
  exit 1
fi
echo "[cuda] $("$CUDA/bin/nvcc" --version | sed -n 4p)"
echo "[cuda] architectures: $ARCHS"
echo "[cuda] build=$AV_BUILD install=$AV_INSTALL jobs=$JOBS"

# Same patch the HIP build applies; the HIP-specific parts gate themselves off when CUDA is found.
# Export is skipped for the same reason as the HIP script: CRLF in the submodule working tree would
# rewrite the reviewable patch as a whole-tree diff.
CHESHIRE_SKIP_PATCH_EXPORT=1 python3 "$ROOT/scripts/apply_hip_patch.py"

mkdir -p "$AV_BUILD"
cd "$AV_BUILD"
cmake "$AV_DEV" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$AV_DEPS;$CUDA" \
  -DCMAKE_INSTALL_PREFIX="$AV_INSTALL" \
  -DALICEVISION_BUNDLE_PREFIX="$AV_BUNDLE" \
  "-DALICEVISION_BUNDLE_SEARCH_LIBS_PATHS=$BUNDLE_LIB_PATHS" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA" \
  -DALICEVISION_USE_CUDA=ON -DALICEVISION_USE_HIP=OFF -DALICEVISION_USE_SYCL=OFF \
  -DALICEVISION_USE_POPSIFT="$POPSIFT" ${POPSIFT_DIR:+-DPopSift_DIR=$POPSIFT_DIR} \
  -DALICEVISION_USE_CCTAG=OFF -DALICEVISION_USE_APRILTAG=OFF \
  -DALICEVISION_USE_OPENCV=OFF -DALICEVISION_USE_ONNX=OFF -DALICEVISION_USE_ONNX_GPU=OFF \
  -DALICEVISION_USE_USD=OFF -DALICEVISION_USE_ALEMBIC=ON -DALICEVISION_BUILD_LIDAR=OFF \
  -DALICEVISION_BUILD_TESTS=OFF -DALICEVISION_BUILD_DOC=OFF -DALICEVISION_BUILD_SWIG_BINDING=OFF \
  -DMINIGLOG=ON -DTARGET_ARCHITECTURE=core \
  ${CHESHIRE_CMAKE_EXTRA:-}

echo "=== what the configure decided"
grep -E "ALICEVISION_HAVE_CUDA|ALICEVISION_HAVE_HIP|CMAKE_CUDA_ARCHITECTURES" CMakeCache.txt || true

[ "$STEP" = "configure" ] && exit 0
cmake --build . -j "$JOBS"
[ "$STEP" = "build" ] && exit 0
cmake --install .
echo "=== installed to $AV_INSTALL"
[ "$STEP" = "install" ] && exit 0

cmake --build . --target bundle
echo "=== bundled to $AV_BUNDLE"

# libcuda is the DRIVER library and must come from the machine that has the GPU. The toolkit ships
# a link-only stub in lib64/stubs, and a WSL box carries its own flavour; bundling either gives a
# package that loads and then fails on the first API call. Same class of mistake as the WSL HSA
# runtime the HIP bundle has to swap out, so it gets the same treatment - drop it, let the host
# driver provide it.
if ls "$AV_BUNDLE"/lib/libcuda.so* >/dev/null 2>&1; then
  echo "removing libcuda from the bundle - it belongs to the host driver"
  rm -f "$AV_BUNDLE"/lib/libcuda.so*
fi
echo "=== CUDA runtime in the bundle"
ls "$AV_BUNDLE"/lib/libcudart.so* 2>/dev/null || echo "  WARNING: no libcudart in the bundle"
