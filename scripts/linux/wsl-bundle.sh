#!/usr/bin/env bash
# Build the Linux HIP bundle on the WSL build box, cleanly, and say what came out.
#
#   scripts/linux/wsl-bundle.sh [suffix]      e.g. wsl-bundle.sh 031  ->  /root/av-hip-build-031,
#                                                  /opt/AliceVision_hip_031/bundle
#
# This is the recipe that produced every Linux bundle since v0.2.13, promoted from
# build/wsl_av_build.sh - a gitignored file, which meant the only record of how to build the
# Linux artifact correctly lived outside the repository. Three things it does deliberately:
#
#  * Fresh build directory and install prefix per run. The configure step never clears a cache,
#    and on 2026-09-20 a HIP rebuild over /root/av-hip-build - which had been configured for the
#    CUDA backend in between - linked aliceVision_feature against the CUDA PopSift cached there.
#    A directory that has been used for one backend is not reused for the other.
#  * CHESHIRE_POPSIFT_INSTALL set explicitly. The HIP PopSift is built on the Windows side
#    (all code objects verified there, docs/14) and reached through /mnt/d; the AliceVision build
#    here only links against it. build-alicevision.sh now refuses to run without it.
#  * The tree is reset to origin/main after a stash, so a build box never builds local edits by
#    accident - and never loses them either.
#
# Run it detached (setsid nohup ... &): a WSL process started from the session that launched it
# dies with that session (docs/06). Everything goes to /root/hip-rebuild-<suffix>.log.
set -u
SUFFIX="${1:-$(date +%m%d%H%M)}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
# A checkout under /mnt is the Windows checkout, shared with Windows git. This script stashes local
# changes and resets to origin/main, and Linux git rewrites every text file with LF as it does: on
# 2026-09-22 that turned meshroom-pair.cmd LF-only and broke Windows pairing, and it stashed an
# uncommitted patch export. Build from a WSL-side checkout (/root/cheshire) instead.
case "$ROOT" in
  /mnt/*)
    if [ "${CHESHIRE_ALLOW_SHARED_CHECKOUT:-0}" != 1 ]; then
      echo "refusing: $ROOT is the Windows checkout (shared with Windows git); run this from /root/cheshire" >&2
      echo "  (git -C /root/cheshire fetch && bash /root/cheshire/scripts/linux/wsl-bundle.sh <suffix>)" >&2
      exit 1
    fi ;;
esac

export CHESHIRE_POPSIFT=ON
export CHESHIRE_POPSIFT_INSTALL="${CHESHIRE_POPSIFT_INSTALL:-/mnt/d/MMI/cheshire/build/popsift-linux-install}"
export AV_BUILD="${AV_BUILD:-$HOME/av-hip-build-$SUFFIX}"
export AV_INSTALL="${AV_INSTALL:-/opt/AliceVision_hip_$SUFFIX}"
export AV_BUNDLE="$AV_INSTALL/bundle"
LOG="$HOME/hip-rebuild-$SUFFIX.log"

{
  echo "=== $(date -Is) start  build=$AV_BUILD  install=$AV_INSTALL"
  git stash push -q -m "wsl-bundle $SUFFIX $(date -Is)" 2>/dev/null && echo "=== local changes stashed" || true
  git fetch --quiet origin && git checkout --quiet main && git reset --hard --quiet origin/main
  git submodule update --init --recursive --quiet third_party/aliceVision
  echo "=== at $(git log --oneline -1)"
  ls -l "$CHESHIRE_POPSIFT_INSTALL/lib/libpopsift.so" | awk '{print "popsift: "$5" bytes"}'
  rm -rf "$AV_BUILD" "$AV_INSTALL"

  echo "=== configure (fresh cache)"
  bash scripts/linux/build-alicevision.sh configure || { echo "CONFIGURE FAILED"; exit 1; }
  echo "=== PopSift_DIR as configured:"
  grep '^PopSift_DIR' "$AV_BUILD/CMakeCache.txt"

  echo "=== build + install + bundle"
  bash scripts/linux/build-alicevision.sh bundle
  echo "=== exit $?"

  # What actually came out. A good depthMap library carries HIP code objects (amdhsa) for every
  # target and the bridge's VRAM-cap clamp; a good PopSift carries code objects and is the HIP
  # one, so featureExtraction must need libamdhip64 and never libcudart.
  echo "=== what the bundle carries:"
  for f in "$AV_BUNDLE/lib/libaliceVision_depthMap_cuda.so.3.4" "$AV_BUNDLE/lib/libpopsift.so"; do
    [ -f "$f" ] || { echo "MISSING $f"; continue; }
    printf '%-40s %10s bytes  amdhsa=%s  clamp=%s\n' "$(basename "$f")" "$(stat -c %s "$f")" \
      "$(grep -ac 'amdhsa--' "$f")" "$(grep -ac 'exceeds this device' "$f")"
  done
  echo "=== featureExtraction needs:"
  readelf -d "$AV_BUNDLE/bin/aliceVision_featureExtraction" 2>/dev/null | grep -E 'popsift|cudart|amdhip' || true
  echo "=== $(date -Is) done"
} > "$LOG" 2>&1
