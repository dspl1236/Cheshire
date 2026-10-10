#!/usr/bin/env bash
# Cheshire Linux: build AliceVision's third-party dependencies with its own superbuild
# (ALICEVISION_BUILD_DEPENDENCIES=ON), CUDA-free. Tested in WSL2 Ubuntu 24.04 with ROCm 7.2.
# Usage: scripts/linux/build-deps.sh [install-prefix] [build-dir]
#
# The four switches pinned OFF below (SuiteSparse, LAPACK, Xerces-C, OpenMP) are what the noble deps that every Linux
# release through 0.4.4 shipped were configured with, read back from their CMakeCache (2026-10-10); the superbuild's
# defaults turn the first two ON on Linux. SuiteSparse off keeps GPL CHOLMOD out of the bundles, and the bundles' Ceres
# has always been CERES_NO_SUITESPARSE; pinning them makes a fresh superbuild (the jammy container) configure the same.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AV_DEV="$ROOT/third_party/aliceVision"
AV_INSTALL="${1:-/opt/AliceVision_deps}"
AV_BUILD="${2:-$HOME/av-deps-build}"
JOBS="${JOBS:-$(nproc)}"
# CMake 4 refuses projects declaring cmake_minimum_required < 3.5 (lz4, OpenMesh); this is
# honoured by every nested configure the superbuild runs.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
# GCC 13: assimp's bundled Draco lacks <cstdint> includes ("uint8_t was not declared")
ASSIMP_CMAKE="$AV_DEV/src/cmake/deps/assimp.cmake"
grep -q "include cstdint" "$ASSIMP_CMAKE" || sed -i 's|            -DASSIMP_BUILD_DRACO:BOOL=ON|            -DASSIMP_BUILD_DRACO:BOOL=ON\n            "-DCMAKE_CXX_FLAGS=-include cstdint"|' "$ASSIMP_CMAKE"
# Ceres without CUDA, as every shipped Linux Ceres (CERES_NO_CUDA): the noble deps were built before the box had a CUDA
# toolkit, so Ceres found none. Where one is installed (the jammy container), Ceres 2.2's USE_CUDA defaults ON, and its
# configure failed there identifying nvcc (2026-10-10).
CERES_CMAKE="$AV_DEV/src/cmake/deps/ceres.cmake"
grep -q "USE_CUDA" "$CERES_CMAKE" || sed -i 's|            -DMINIGLOG=ON|            -DMINIGLOG=ON\n            -DUSE_CUDA:BOOL=OFF|' "$CERES_CMAKE"
grep -q "USE_CUDA:BOOL=OFF" "$CERES_CMAKE" || { echo "could not pin USE_CUDA=OFF in $CERES_CMAKE" >&2; exit 1; }
mkdir -p "$AV_BUILD" "$AV_INSTALL/lib"
ln -sfn lib "$AV_INSTALL/lib64"
cd "$AV_BUILD"
cmake "$AV_DEV" \
  -DCMAKE_BUILD_TYPE=Release \
  -DALICEVISION_BUILD_DEPENDENCIES:BOOL=ON \
  -DAV_BUILD_ALICEVISION:BOOL=OFF \
  -DCMAKE_INSTALL_PREFIX="$AV_INSTALL" \
  -DAV_BUILD_ZLIB:BOOL=ON \
  -DAV_BUILD_CUDA:BOOL=OFF \
  -DAV_BUILD_POPSIFT:BOOL=OFF \
  -DAV_BUILD_CCTAG:BOOL=OFF \
  -DAV_BUILD_APRILTAG:BOOL=OFF \
  -DAV_BUILD_USD:BOOL=OFF \
  -DAV_BUILD_PCL:BOOL=OFF \
  -DAV_BUILD_E57FORMAT:BOOL=OFF \
  -DAV_BUILD_OPENCV:BOOL=OFF \
  -DAV_BUILD_ONNXRUNTIME:BOOL=OFF \
  -DAV_BUILD_FFMPEG:BOOL=OFF \
  -DAV_BUILD_VPX:BOOL=OFF \
  -DAV_BUILD_PYBIND11:BOOL=ON \
  -DAV_BUILD_SWIG:BOOL=OFF \
  -DAV_BUILD_SUITESPARSE:BOOL=OFF \
  -DAV_BUILD_LAPACK:BOOL=OFF \
  -DAV_BUILD_XERCESC:BOOL=OFF \
  -DAV_BUILD_OPENMP:BOOL=OFF \
  -DPython_EXECUTABLE="$(command -v python3)"
cmake --build . --parallel "$JOBS"
# the superbuild drops dependency tools into bin/; keep them out of the AliceVision bin
if [ -d "$AV_INSTALL/bin" ] && [ ! -d "$AV_INSTALL/bin-deps" ]; then mv "$AV_INSTALL/bin" "$AV_INSTALL/bin-deps"; fi
echo "DEPS OK -> $AV_INSTALL"
