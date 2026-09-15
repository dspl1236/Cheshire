#!/usr/bin/env bash
# Cheshire: one-shot setup + first HIP depth-map run on a Linux node with an AMD card
# (written for house-pc with an RX 6700 XT; works for any RDNA2/3/4 card).
#
#   sudo bash node-amd-setup.sh setup                 # groups, unpack, env file
#   bash node-amd-setup.sh check                      # driver, /dev/kfd, GPU seen by the bundle
#   bash node-amd-setup.sh run <scanName> [outDir]    # HIP DepthMap on /data/scans/<scan>/cache, compare with its CUDA DepthMap
#   bash node-amd-setup.sh pair [<Meshroom dir>]       # make Meshroom run its DepthMap node on the HIP build
#
# Assumes the tarball is in ~/apps/cheshire/ (scp'd there from the build box).
set -euo pipefail
# under sudo, use the invoking user's home, not root's
_HOME="$HOME"; [ -n "${SUDO_USER:-}" ] && _HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
APPS="${APPS:-$_HOME/apps/cheshire}"
BUNDLE="$APPS/bundle"
ENVF="$APPS/env.sh"
CMD="${1:-check}"

case "$CMD" in
setup)
  [ "$(id -u)" = 0 ] || { echo "run setup with sudo"; exit 1; }
  U="${SUDO_USER:-house}"
  usermod -aG render,video "$U"
  TB=$(ls -t "$APPS"/cheshire-alicevision-hip-linux-x64-*.tar.gz | head -1)
  echo "unpacking $TB"
  tar -C "$APPS" -xzf "$TB"
  chown -R "$U:$U" "$APPS"
  # No HSA_OVERRIDE_GFX_VERSION: the bundle carries native code objects for every RDNA part
  # (and the APUs / Vega), and an override left over from one card crashes the next one:
  # presenting an RX 5500 XT as gfx1030 dies with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
  # on the first kernel. Only set it by hand for a chip whose gfx id is missing from the bundle.
  cat > "$ENVF" <<EOF
export ALICEVISION_ROOT=$BUNDLE
export LD_LIBRARY_PATH=$BUNDLE/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export PATH=$BUNDLE/bin:\$PATH
EOF
  echo "wrote $ENVF ; log out and back in for the group change, then: bash $0 check"
  ;;
check)
  echo "--- kernel / driver ---"; uname -r; lsmod | grep -E "^amdgpu" || echo "amdgpu NOT loaded"
  ls -la /dev/kfd /dev/dri/renderD* 2>&1 | head -3
  echo "--- groups: $(id -nG)"
  echo "--- PCI ---"; lspci | grep -iE "vga|display|3d"
  echo "--- bundle GPU query ---"
  # shellcheck disable=SC1090
  source "$ENVF"
  "$BUNDLE/bin/aliceVision_hardwareResources" 2>&1 | grep -E "name:|total device memory|device memory available|No CUDA|Error|error" | head -6
  ;;
run)
  SCAN="${2:?scan name under /data/scans}"
  OUT="${3:-/data/scans/$SCAN/out/cheshire-hip}"
  # shellcheck disable=SC1090
  source "$ENVF"
  CACHE="/data/scans/$SCAN/cache"
  REF=$(ls -d "$CACHE"/DepthMap/*/ 2>/dev/null | head -1)
  mkdir -p "$OUT"
  echo "cache=$CACHE ref=$REF out=$OUT"
  START=$(date +%s)
  bash "$(dirname "$0")/run-depthmap.sh" "$BUNDLE" "$CACHE" "$OUT" "$REF" 2>&1 | tee "$OUT/run.log" | grep -E "Task done|Device memory|available:|RESULT|error|Error"
  echo "wall: $(( $(date +%s) - START )) s"
  echo "compare table: $OUT/compare_stats.md (python3 needs: pip install openexr numpy pillow)"
  ;;
pair)
  # swap Meshroom's DepthMap binary for the HIP build (scripts/linux/meshroom-pair.sh)
  bash "$(dirname "$0")/meshroom-pair.sh" "${2:-$_HOME/apps/Meshroom-2023.3.0}" "$BUNDLE"
  ;;
*) echo "usage: $0 setup|check|run <scan>|pair [<Meshroom dir>]"; exit 1;;
esac
