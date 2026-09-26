#!/usr/bin/env bash
# Cheshire: pair a Meshroom 2023.3 Linux bundle with the Cheshire HIP AliceVision bundle.
#
# Meshroom runs its nodes as aliceVision_* executables from <Meshroom>/aliceVision/bin. Seven of
# them get the Cheshire treatment when the bundle carries them: PrepareDenseScene,
# FeatureExtraction, FeatureMatching, DepthMap, DepthMapFilter, Meshing and Texturing. Each becomes
# a wrapper that execs the Cheshire build with the bundle's libraries in front, or Meshroom's own
# binary (kept as <name>.cuda) when nvidia-smi finds an NVIDIA card: the choice is made per run, so
# a node that swaps cards needs no re-pairing, and CHESHIRE_BACKEND forces it. The nodes Cheshire
# does not carry - CameraInit, ImageMatching, StructureFromMotion, MeshFiltering, Publish - keep
# running from the Meshroom bundle unchanged. `--unpair` restores.
#
# (Until v0.3.0 this header said two nodes and named meshing and texturing as staying on Meshroom's
# side; the body has paired seven since v0.2.9.)
#
#   meshroom-pair.sh <Meshroom dir> [<cheshire bundle dir>]      # default bundle: ~/apps/cheshire/bundle
#   meshroom-pair.sh <Meshroom dir> <cheshire bundle dir> --check  # only report compatibility, change nothing
#   meshroom-pair.sh <Meshroom dir> --unpair
#
# Before a node is paired, the options Meshroom's own binary takes are compared with the bundle's
# (compatible() below): a Meshroom newer than the bundle passes options the bundle does not know, and
# that node is then left to Meshroom instead of failing mid-job.
#
# The Meshroom 2023.3 DepthMap node's command line is accepted by the newer AliceVision the HIP
# build is based on except --sgmFilteringAxes, which upstream removed (YX is the only behaviour
# now, and the one docs/04-validation.md validates); the wrapper drops it. FeatureMatching's
# --rangeStart/--rangeSize are accepted by the Cheshire build directly.
set -euo pipefail
MESHROOM="${1:?Meshroom directory (e.g. ~/apps/Meshroom-2023.3.0)}"
# EXTERNAL CONTRACT - the photogrammetry node reads what this script writes.
# ~/bin/reconstruct and the job runner's app.py both resolve the live bundle from the wrapper
# below rather than guessing a path, because a stale bundle directory can exist beside the
# live one. Three things are load-bearing outside this repository, and none of them fail
# loudly if they change:
#   1. the wrapper contains a line matching ^export ALICEVISION_ROOT="..."$
#      -> change it and the bundle resolves empty: reconstruct refuses, the GUI says
#         "not paired"
#   2. pairing leaves the original binary as <name>.cuda
#      -> change it and paired nodes report themselves unpaired
#   3. libpopsift.so and libaliceVision_feature.so both sit in $BUNDLE/lib/
#      -> change it and feature extraction silently drops to the CPU
# If you restructure the wrapper or the bundle layout, say so in the same commit.
BIN="$MESHROOM/aliceVision/bin"
[ -d "$BIN" ] || { echo "$BIN not found: is $MESHROOM a Meshroom 2023.x Linux bundle?"; exit 1; }
if [ "${2:-}" = "--unpair" ]; then
  for name in aliceVision_depthMapEstimation aliceVision_featureMatching aliceVision_featureExtraction aliceVision_depthMapFiltering aliceVision_meshing aliceVision_texturing aliceVision_prepareDenseScene aliceVision_incrementalSfM aliceVision_imageMatching; do
    if [ -x "$BIN/$name.cuda" ]; then mv -f "$BIN/$name.cuda" "$BIN/$name"; echo "restored $name"; else echo "$name: not paired"; fi
  done
  NODE="$MESHROOM/lib/meshroom/nodes/aliceVision/DepthMap.py"
  if [ -f "$NODE" ] && grep -q "Cheshire" "$NODE"; then rm -f "$NODE"; echo "removed the DepthMap node override (block of 48)"; fi
  exit 0
fi
BUNDLE="${2:-$HOME/apps/cheshire/bundle}"
CHECK_ONLY=0
[ "${3:-}" = "--check" ] && CHECK_ONLY=1
[ -x "$BUNDLE/bin/aliceVision_depthMapEstimation" ] || { echo "no HIP aliceVision_depthMapEstimation in $BUNDLE/bin"; exit 1; }

# The options Meshroom's own binary takes must all be taken by the bundle's, except the one the
# wrapper drops (DepthMap's --sgmFilteringAxes): a Meshroom node passes options its node description
# knows, which are options Meshroom's binary takes, and a bundle older than that Meshroom would fail
# mid-job with "unrecognised option". Reads both --help texts, which needs no GPU. Returns 0 when
# compatible, or when it cannot tell (Meshroom's binary lists no options); 1 with the missing
# options printed. Options that kept their names but changed meaning are past what this can see.
compatible() {  # name  [option the wrapper drops]
  local name="$1" drop="${2:-}" own="$BIN/$1" mr ours missing
  [ -e "$BIN/$1.cuda" ] && own="$BIN/$1.cuda"   # once paired, Meshroom's own binary is <name>.cuda
  mr=$(ALICEVISION_ROOT="$MESHROOM/aliceVision" LD_LIBRARY_PATH="$MESHROOM/aliceVision/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$own" --help 2>&1 || true)
  ours=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/$1" --help 2>&1 || true)
  mr=$(grep -o -- ' --[A-Za-z][A-Za-z0-9_.]*' <<<"$mr" | LC_ALL=C sort -u || true)
  ours=$(grep -o -- ' --[A-Za-z][A-Za-z0-9_.]*' <<<"$ours" | LC_ALL=C sort -u || true)
  if [ -z "$mr" ]; then echo "$1: Meshroom's binary listed no options, compatibility not checked"; return 0; fi
  missing=$(LC_ALL=C comm -23 <(printf '%s\n' "$mr") <(printf '%s\n' "$ours") | grep -vxF -- " $drop" | tr '\n' ' ' || true)
  if [ -n "${missing// /}" ]; then
    echo "$1: Meshroom's binary takes options this bundle's does not:$missing(a newer Meshroom than the bundle was built for?)"
    return 1
  fi
  [ "$CHECK_ONLY" = 1 ] && echo "$1: compatible ($(wc -l <<<"$mr") options)"
  return 0
}

pair() {  # name  drop-option
  local name="$1" drop="${2:-}" target="$BIN/$1"
  if ! compatible "$name" "$drop"; then echo "$name: not paired, Meshroom keeps its own binary"; return 0; fi
  [ "$CHECK_ONLY" = 1 ] && return 0
  if [ ! -e "$target.cuda" ]; then mv "$target" "$target.cuda"; fi
  cat > "$target" <<EOF
#!/bin/bash
# Cheshire pairing (scripts/linux/meshroom-pair.sh): $name on whichever GPU is in the box.
# NVIDIA present (nvidia-smi answers): Meshroom's own binary, kept beside this file as .cuda.
# Otherwise the Cheshire build from the bundle. Decided per run, so swapping cards needs no
# re-pairing. CHESHIRE_BACKEND=auto|cheshire|meshroom forces one, for every paired node.
#
# auto hands the node back to Meshroom whenever an NVIDIA card is present. That was the right
# default while Cheshire was AMD-only and is exactly wrong when the paired bundle is itself a CUDA
# build, so a CUDA bundle has to be asked for. CHESHIRE_DEPTHMAP=cuda|hip is the older spelling,
# from when "hip" and "the Cheshire bundle" were the same thing, and is still honoured.
CHESHIRE_MODE="\${CHESHIRE_BACKEND:-}"
if [ -z "\$CHESHIRE_MODE" ]; then
  case "\${CHESHIRE_DEPTHMAP:-auto}" in
    cuda) CHESHIRE_MODE=meshroom ;;
    hip)  CHESHIRE_MODE=cheshire ;;
    *)    CHESHIRE_MODE="\${CHESHIRE_DEPTHMAP:-auto}" ;;
  esac
fi
if [ "\$CHESHIRE_MODE" = meshroom ] || { [ "\$CHESHIRE_MODE" = auto ] && nvidia-smi >/dev/null 2>&1; }; then
  echo "[cheshire] $name: Meshroom's own binary ($target.cuda)" >&2
  exec "$target.cuda" "\$@"
fi
# Meshroom's environment stays; the Cheshire bundle's libraries go first so the Cheshire build
# resolves its own OpenImageIO/Boost/ROCm sonames, not the Meshroom bundle's older ones.
export ALICEVISION_ROOT="$BUNDLE"
export LD_LIBRARY_PATH="$BUNDLE/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
# per-node knobs: CHESHIRE_BRIDGE_* for the memory bridge (docs/02-memory-bridge.md), CHESHIRE_GPU_MATCHER=0 for the CPU matcher
[ -f "$BUNDLE/../env.sh" ] && . "$BUNDLE/../env.sh"
ARGS=()
while [ \$# -gt 0 ]; do
  case "\$1" in
    ${drop:-__none__}) shift 2 ;;
    *) ARGS+=("\$1"); shift ;;
  esac
done
# Say which binary is about to run, the way the Windows launcher does. Without it a node's
# provenance can only be inferred from what its port prints, and that is not always decisive:
# Meshroom's own featureExtraction is a CUDA PopSIFT and prints the same "Choosing device 0" line
# as ours, so an unpaired node looked paired to the end-to-end gate.
echo "[cheshire] $name: Cheshire build ($BUNDLE/bin/$name)" >&2
exec "$BUNDLE/bin/$name" "\${ARGS[@]}"
EOF
  chmod +x "$target"
  echo "paired: $target -> $BUNDLE/bin/$name (Meshroom's binary kept as $target.cuda)"
}

pair aliceVision_depthMapEstimation --sgmFilteringAxes
# only a bundle with the GPU matcher understands Meshroom 2023.3's --rangeStart/--rangeSize; older
# bundles carry the plain CPU featureMatching, which must not be put in Meshroom's way
fm_help=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_featureMatching" --help 2>&1 || true)   # not piped: pipefail + grep -q would SIGPIPE the probe
if grep -q -- '--rangeStart' <<<"$fm_help"; then
  pair aliceVision_featureMatching
else
  echo "bundle's aliceVision_featureMatching has no GPU matcher (pre-v0.2.5): DepthMap paired only"
fi

# GPU SIFT (docs/14): only pair featureExtraction when the bundle's binary actually links
# popsift, otherwise the node would move CPU SIFT from one build to another for nothing.
# Note the describer falls back to CPU silently when no GPU is visible to HIP, so a paired
# node that still logs [cpu] means the runtime cannot see the card, not that pairing failed.
# The executable does not link popsift directly - libaliceVision_feature.so does, and the
# executable picks it up transitively - so test that library. Not with ldd: the bundle sets
# RUNPATH $ORIGIN/../lib, and ldd without LD_LIBRARY_PATH reports its siblings as not found,
# which greps to nothing and would silently refuse to pair a perfectly good bundle.
# Match libpopsift.so*, not libpopsift.so. A bundle may carry only the versioned soname - the
# v0.3.0 Linux CUDA bundle ships libpopsift.so.0.10.0 with no unversioned symlink, while the HIP
# one has the plain name - and testing the plain name silently refused to pair GPU SIFT on a
# package that has it. It failed quietly twice over: the node then runs Meshroom's own binary,
# which on an NVIDIA box is also a CUDA PopSIFT and prints the same "Choosing device 0" line, so
# the end-to-end gate scored it a pass.
if [ -x "$BUNDLE/bin/aliceVision_featureExtraction" ] \
   && compgen -G "$BUNDLE/lib/libpopsift.so*" > /dev/null \
   && grep -aq libpopsift.so "$BUNDLE/lib/libaliceVision_feature.so" 2>/dev/null; then
  pair aliceVision_featureExtraction
else
  echo "bundle's aliceVision_featureExtraction has no GPU SIFT: not paired"
fi
# DepthMapFilter (v0.2.6+): the bundle's depthMapFiltering carries the GPU vote pass (its --help says so)
df_help=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_depthMapFiltering" --help 2>&1 || true)
if grep -q 'CHESHIRE_GPU_FILTER' <<<"$df_help"; then
  pair aliceVision_depthMapFiltering
else
  echo "bundle's aliceVision_depthMapFiltering has no GPU pass (pre-v0.2.6): not paired"
fi
# Meshing (v0.2.7+): the bundle's meshing carries the GPU graph-weight votes (its --help says so)
ms_help=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_meshing" --help 2>&1 || true)
if grep -q 'CHESHIRE_GPU_VOTE' <<<"$ms_help"; then
  pair aliceVision_meshing
else
  echo "bundle's aliceVision_meshing has no GPU votes (pre-v0.2.7): not paired"
fi
# StructureFromMotion (v0.3.3+): the bundle's incrementalSfM finishes a resection pass with the bundle
# adjustment upstream skips - the "invalid map<K, T> key" crash of Meshroom #2344 on large sets (docs/04)
sf_help=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_incrementalSfM" --help 2>&1 || true)
if grep -q 'CHESHIRE_SFM_PENDING_BA' <<<"$sf_help"; then
  pair aliceVision_incrementalSfM
else
  echo "bundle's aliceVision_incrementalSfM is upstream's (pre-v0.3.3): not paired"
fi
# ImageMatching (v0.3.5+): the bundle's imageMatching can pair views by GPS distance
# (CHESHIRE_GPS_PAIRING_RADIUS, off unless set; its --help says so)
im_help=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_imageMatching" --help 2>&1 || true)
if grep -q 'CHESHIRE_GPS_PAIRING' <<<"$im_help"; then
  pair aliceVision_imageMatching
else
  echo "bundle's aliceVision_imageMatching is upstream's (pre-v0.3.5): not paired"
fi
# Texturing (v0.2.8+): the bundle's texturing carries the GPU pyramid + rasterisation (its --help says so)
tx_help=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_texturing" --help 2>&1 || true)
if grep -q 'CHESHIRE_GPU_TEX' <<<"$tx_help"; then
  pair aliceVision_texturing
else
  echo "bundle's aliceVision_texturing has no GPU pass (pre-v0.2.8): not paired"
fi
# PrepareDenseScene (v0.2.9+): the bundle's prepareDenseScene runs its image loop on every core (its --help says so)
pd_help=$(ALICEVISION_ROOT="$BUNDLE" LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_prepareDenseScene" --help 2>&1 || true)
if grep -q 'CHESHIRE_PDS_THREADS' <<<"$pd_help"; then
  pair aliceVision_prepareDenseScene
else
  echo "bundle's aliceVision_prepareDenseScene is upstream's (pre-v0.2.9): not paired"
fi
# The DepthMap node in blocks of 48 views instead of 12 (docs/04, 0.3.4 "the depth-map node was
# loading images"): each chunk is a process that loads the SfM data, probes the device and starts
# cold, 74 times at 884 views. The bundle carries meshroom-overrides/DepthMap.py, which loads
# Meshroom's compiled node and re-declares it with the larger block - same attributes, same UID,
# caches stay valid. Python prefers the .py beside the .pyc, so copying it in is the whole install.
NODES="$MESHROOM/lib/meshroom/nodes/aliceVision"
OVERRIDE="$BUNDLE/share/cheshire/meshroom-overrides/DepthMap.py"
if [ "$CHECK_ONLY" = 1 ]; then
  echo "--check: nothing changed"
elif [ -f "$OVERRIDE" ] && [ -f "$NODES/DepthMap.pyc" ]; then
  cp -f "$OVERRIDE" "$NODES/DepthMap.py" && echo "installed the DepthMap node override: blocks of 48 views (CHESHIRE_DEPTHMAP_BLOCK=0 for Meshroom's 12)"
elif [ -f "$OVERRIDE" ]; then
  echo "no compiled DepthMap node at $NODES: node override not installed"
else
  echo "bundle carries no meshroom-overrides (pre-0.3.4): DepthMap keeps Meshroom's block of 12"
fi
