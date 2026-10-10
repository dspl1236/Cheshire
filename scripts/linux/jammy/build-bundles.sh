#!/usr/bin/env bash
# Cheshire 0.5.0: both Linux bundles built in the jammy container (glibc 2.35, GCC 12), the same steps and order as the
# release chain's WSL side (build/wsl-0xx.lf.sh): the dependency superbuild once, the HIP PopSIFT, the HIP bundle, the
# CUDA PopSIFT (its reset to upstream and the fixes both backends take), the CUDA bundle. Packed into /out/<tag>/ with
# the newest glibc / libstdc++ symbol versions each bundle needs, which must not exceed jammy's (GLIBC_2.35,
# GLIBCXX_3.4.30).
#
#   scripts/linux/jammy/container.sh run 'bash scripts/linux/jammy/build-bundles.sh <tag> [hip|cuda|both]'
set -euo pipefail
TAG="${1:?tag}"
WHICH="${2:-both}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="/out/$TAG"
mkdir -p "$OUT"
say() { echo "=== $(date -Is) $*"; }
say "at $(git log --oneline -1); $(gcc --version | head -1); $(ldd --version | head -1)"

if ! ls /opt/AliceVision_deps/lib/libceres.so.* >/dev/null 2>&1; then
  say "dependency superbuild"
  bash scripts/linux/build-deps.sh /opt/AliceVision_deps /root/av-deps-build > /root/deps.log 2>&1 \
    || { tail -30 /root/deps.log; say "DEPS FAILED"; exit 1; }
fi

needs() {  # the newest GLIBC_ and GLIBCXX_ versions any ELF file under $1 needs; fails above jammy's
  local d="$1" g x
  g=$(find "$d" -type f -exec objdump -T {} \; 2>/dev/null | grep -o 'GLIBC_2\.[0-9]*' | sort -V -u | tail -1)
  x=$(find "$d" -type f -exec objdump -T {} \; 2>/dev/null | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -V -u | tail -1)
  echo "  needs $g and $x"
  [ "$(printf '%s\n' "$g" GLIBC_2.35 | sort -V | tail -1)" = GLIBC_2.35 ] || { say "NEEDS NEWER GLIBC THAN JAMMY"; return 1; }
  [ "$(printf '%s\n' "$x" GLIBCXX_3.4.30 | sort -V | tail -1)" = GLIBCXX_3.4.30 ] || { say "NEEDS NEWER LIBSTDC++ THAN JAMMY"; return 1; }
}

if [ "$WHICH" != cuda ]; then
  say "PopSIFT HIP"
  bash scripts/linux/build-popsift.sh jammy > /root/popsift-hip.log 2>&1 || { tail -20 /root/popsift-hip.log; say "POPSIFT HIP FAILED"; exit 1; }
  say "HIP bundle"
  rm -rf /root/av-hip-build /opt/AliceVision_hip
  CHESHIRE_POPSIFT=ON CHESHIRE_POPSIFT_INSTALL="$ROOT/build/popsift-jammy-install" AV_BUILD=/root/av-hip-build \
    AV_INSTALL=/opt/AliceVision_hip bash scripts/linux/build-alicevision.sh bundle > /root/hip.log 2>&1 \
    || { grep -n " error" /root/hip.log | head -20; tail -20 /root/hip.log; say "HIP BUNDLE FAILED"; exit 1; }
  bash scripts/linux/wsl-pack-bundle.sh /opt/AliceVision_hip "$OUT/cheshire-alicevision-hip-linux-x64-rocm7.2.tar.gz" > /root/hip-pack.log 2>&1 \
    || { tail -20 /root/hip-pack.log; say "HIP PACK FAILED"; exit 1; }
  tail -3 /root/hip-pack.log
  needs /opt/AliceVision_hip/bundle
fi

if [ "$WHICH" != hip ]; then
  say "PopSIFT CUDA"
  bash scripts/linux/build-popsift-cuda.sh > /root/popsift-cuda.log 2>&1 || { tail -20 /root/popsift-cuda.log; say "POPSIFT CUDA FAILED"; exit 1; }
  say "CUDA bundle"
  rm -rf /root/av-cuda-build /opt/AliceVision_cuda
  export AV_BUILD=/root/av-cuda-build AV_INSTALL=/opt/AliceVision_cuda
  bash scripts/linux/build-alicevision-cuda.sh install 61 > /root/cuda.log 2>&1 \
    || { grep -n " error" /root/cuda.log | head -20; tail -20 /root/cuda.log; say "CUDA INSTALL FAILED"; exit 1; }
  bash scripts/linux/build-alicevision-cuda.sh bundle 61 > /root/cuda-bundle.log 2>&1 || { tail -20 /root/cuda-bundle.log; say "CUDA BUNDLE FAILED"; exit 1; }
  bash scripts/linux/package-cuda.sh /opt/AliceVision_cuda/bundle "$OUT" > /root/cuda-pack.log 2>&1 || { tail -20 /root/cuda-pack.log; say "CUDA PACK FAILED"; exit 1; }
  tail -3 /root/cuda-pack.log
  needs /opt/AliceVision_cuda/bundle
fi
say "done -> $OUT"
