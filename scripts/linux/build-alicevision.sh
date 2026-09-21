#!/usr/bin/env bash
# Cheshire Linux: build AliceVision with the HIP depth-map backend against the superbuild
# dependencies (scripts/linux/build-deps.sh) and ROCm in /opt/rocm.
# Usage: scripts/linux/build-alicevision.sh [configure|build|install|bundle] [gfx list]
#   gfx list default: RDNA1-RDNA4 discrete parts plus the APUs (gfx1035/1036 Rembrandt, gfx1103 Phoenix,
#   gfx1150/1151/1152/1153 Strix Point / Strix Halo / Krackan) and Vega (gfx900/906, untested: the
#   ROCm 7.2 runtime may refuse them the way it refuses Polaris)
#   RDNA2 cards other than gfx1030 (RX 6700 XT = gfx1031, 6600 = gfx1032 ...) run the gfx1030
#   code with HSA_OVERRIDE_GFX_VERSION=10.3.0 in the environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AV_DEV="$ROOT/third_party/aliceVision"
STEP="${1:-install}"
ARCHS="${2:-gfx900;gfx906;gfx1010;gfx1012;gfx1030;gfx1031;gfx1032;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201}"
ROCM="${ROCM_PATH:-/opt/rocm}"
# exported because apply_hip_patch.py's GPU SIFT step reads $ENV{ROCM_PATH} from CMake to put
# ROCm's headers and runtime on aliceVision_feature, which the popsift describer needs.
export ROCM_PATH="$ROCM"
AV_DEPS="${AV_DEPS:-/opt/AliceVision_deps}"
AV_BUILD="${AV_BUILD:-$HOME/av-hip-build}"
AV_INSTALL="${AV_INSTALL:-/opt/AliceVision_hip}"
AV_BUNDLE="${AV_BUNDLE:-$AV_INSTALL/bundle}"
# GPU SIFT: CHESHIRE_POPSIFT=ON with a HIP popsift build (scripts/linux/build-popsift.sh).
# Its lib directory also joins the bundle search path so libpopsift.so gets packaged.
CHESHIRE_POPSIFT="${CHESHIRE_POPSIFT:-OFF}"
POPSIFT_INSTALL="${CHESHIRE_POPSIFT_INSTALL:-$ROOT/build/popsift-linux-install}"
# A missing PopSift is a hard error, not a fallback. With CHESHIRE_POPSIFT=ON and this directory
# absent, -DPopSift_DIR points nowhere and find_package keeps searching - and found the CUDA
# PopSift cached from a previous configure of the same tree, so a HIP bundle linked
# aliceVision_feature against libpopsift.so.0.10.0 from /opt/popsift-cuda and said nothing
# (2026-09-20). On the WSL box the HIP PopSift is the Windows-side build, reached through /mnt/d;
# build/wsl_av_build.sh sets CHESHIRE_POPSIFT_INSTALL for exactly that reason.
if [ "$CHESHIRE_POPSIFT" = ON ] && [ ! -f "$POPSIFT_INSTALL/lib/cmake/PopSift/PopSiftConfig.cmake" ]; then
  echo "CHESHIRE_POPSIFT=ON but there is no PopSift at $POPSIFT_INSTALL" >&2
  echo "set CHESHIRE_POPSIFT_INSTALL to a HIP PopSift install (scripts/linux/build-popsift.sh)" >&2
  exit 1
fi
JOBS="${JOBS:-$(nproc)}"

# The submodule working tree has Windows line endings, so a diff taken here would rewrite the
# reviewable patch artifact as a 123 MB whole-tree diff. Leave it to the Windows side.
CHESHIRE_SKIP_PATCH_EXPORT=1 python3 "$ROOT/scripts/apply_hip_patch.py"

mkdir -p "$AV_BUILD"
cd "$AV_BUILD"
cmake "$AV_DEV" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$AV_DEPS;$ROCM" \
  -DCMAKE_INSTALL_PREFIX="$AV_INSTALL" \
  -DALICEVISION_BUNDLE_PREFIX="$AV_BUNDLE" \
  "-DALICEVISION_BUNDLE_SEARCH_LIBS_PATHS=$AV_DEPS/lib;$ROCM/lib;$ROCM/lib/llvm/lib;$POPSIFT_INSTALL/lib" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_ARCHITECTURES="$ARCHS" \
  -DALICEVISION_USE_CUDA=OFF -DALICEVISION_USE_HIP=ON -DALICEVISION_USE_SYCL=OFF \
  -DALICEVISION_USE_POPSIFT=$CHESHIRE_POPSIFT "-DPopSift_DIR=$POPSIFT_INSTALL/lib/cmake/PopSift" \
  -DALICEVISION_USE_CCTAG=OFF -DALICEVISION_USE_APRILTAG=OFF \
  -DALICEVISION_USE_OPENCV=OFF -DALICEVISION_USE_ONNX=OFF -DALICEVISION_USE_ONNX_GPU=OFF \
  -DALICEVISION_USE_USD=OFF -DALICEVISION_USE_ALEMBIC=ON -DALICEVISION_BUILD_LIDAR=OFF \
  -DALICEVISION_BUILD_TESTS=OFF -DALICEVISION_BUILD_DOC=OFF -DALICEVISION_BUILD_SWIG_BINDING=OFF \
  -DMINIGLOG=ON -DTARGET_ARCHITECTURE=core \
  ${CHESHIRE_CMAKE_EXTRA:-}
# Assert the cache took the PopSift we named, not one it found. A cache configured earlier for
# the CUDA backend keeps PopSift_DIR=/opt/popsift-cuda, and find_package keeps it when the named
# directory is empty - the configure step never clears a cache. Checked after every configure so
# the failure surfaces here, not as "cannot resolve item 'libpopsift.so.0.10.0'" at bundle time
# seven minutes later (2026-09-20).
#
# The cache type varies and must not be matched: a value CMake took from -D is recorded as
# PopSift_DIR:UNINITIALIZED=, one find_package had to search for as PopSift_DIR:PATH=. Matching
# only :PATH= would fail every correct configure and pass the stale one.
if [ "$CHESHIRE_POPSIFT" = ON ]; then
  got=$(sed -n 's/^PopSift_DIR:[A-Z]*=//p' CMakeCache.txt)
  case "$got" in
    "$POPSIFT_INSTALL"/*) ;;
    *) echo "PopSift_DIR is '$got', not under $POPSIFT_INSTALL: a stale cache. Delete $AV_BUILD and reconfigure." >&2
       exit 1 ;;
  esac
fi
[ "$STEP" = configure ] && exit 0
cmake --build . --parallel "$JOBS"
[ "$STEP" = build ] && exit 0
cmake --install .
[ "$STEP" = install ] && exit 0
cmake --build . --target bundle
# WSL build boxes carry the WSL flavour of the HSA runtime (probes /dev/dxg, never /dev/kfd);
# a bundle made there fails hsa_init with OUT_OF_RESOURCES on a real Linux box. Replace it
# with the standard runtime from AMD's apt repo (hsa-rocr) when it looks like the WSL one.
if grep -a -q "/dev/dxg" "$AV_BUNDLE/lib/libhsa-runtime64.so.1" && [ "$(stat -c %s "$AV_BUNDLE/lib/libhsa-runtime64.so.1")" -lt 3000000 ]; then
  echo "bundle has the WSL HSA runtime; swapping in hsa-rocr from the AMD apt repo"
  T=$(mktemp -d); ( cd "$T" && apt-get download hsa-rocr >/dev/null 2>&1 && dpkg-deb -x hsa-rocr_*.deb x )
  K=$(find "$T/x" -name "libhsa-runtime64.so.1.*" -type f | head -1)
  if [ -n "$K" ]; then
    rm -f "$AV_BUNDLE"/lib/libhsa-runtime64.so.1*
    cp "$K" "$AV_BUNDLE/lib/$(basename "$K")" && ln -sfn "$(basename "$K")" "$AV_BUNDLE/lib/libhsa-runtime64.so.1"
    echo "  -> $(basename "$K")"
  else
    echo "  WARNING: could not download hsa-rocr; bundle keeps the WSL runtime" >&2
  fi
  rm -rf "$T"
fi
# HIP dlopens the code-object manager at runtime; the bundler never sees it as a dependency.
if ! ls "$AV_BUNDLE"/lib/libamd_comgr.so.* >/dev/null 2>&1; then
  C=$(ls "$ROCM"/lib/libamd_comgr.so.*.*.* 2>/dev/null | head -1)
  [ -n "$C" ] && cp "$C" "$AV_BUNDLE/lib/" && ln -sfn "$(basename "$C")" "$AV_BUNDLE/lib/libamd_comgr.so.${C##*.so.}" 2>/dev/null
  ln -sfn "$(basename "$C")" "$AV_BUNDLE/lib/libamd_comgr.so.$(basename "$C" | sed 's/.*\.so\.\([0-9]*\).*/\1/')"
  echo "added $(basename "$C") to the bundle"
fi
echo "BUNDLE -> $AV_BUNDLE"
