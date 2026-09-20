#!/usr/bin/env bash
# Cheshire: pair a Meshroom 2023.3 Linux bundle with the Cheshire HIP AliceVision bundle.
#
# Meshroom runs its nodes as aliceVision_* executables from <Meshroom>/aliceVision/bin. Two of them
# get the GPU treatment: DepthMap (the HIP port) and FeatureMatching (the GPU descriptor matcher,
# bundles from v0.2.5 on). Each becomes a wrapper that execs the Cheshire build with the bundle's
# libraries in front, or Meshroom's own binary (kept as <name>.cuda) when nvidia-smi finds an NVIDIA
# card: the choice is made per run, so a node that swaps cards needs no re-pairing. Everything else
# (SfM, meshing, texturing) keeps running from the Meshroom bundle unchanged. `--unpair` restores.
#
#   meshroom-pair.sh <Meshroom dir> [<cheshire bundle dir>]      # default bundle: ~/apps/cheshire/bundle
#   meshroom-pair.sh <Meshroom dir> --unpair
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
  for name in aliceVision_depthMapEstimation aliceVision_featureMatching aliceVision_featureExtraction aliceVision_depthMapFiltering aliceVision_meshing aliceVision_texturing aliceVision_prepareDenseScene; do
    if [ -x "$BIN/$name.cuda" ]; then mv -f "$BIN/$name.cuda" "$BIN/$name"; echo "restored $name"; else echo "$name: not paired"; fi
  done
  exit 0
fi
BUNDLE="${2:-$HOME/apps/cheshire/bundle}"
[ -x "$BUNDLE/bin/aliceVision_depthMapEstimation" ] || { echo "no HIP aliceVision_depthMapEstimation in $BUNDLE/bin"; exit 1; }

pair() {  # name  drop-option
  local name="$1" drop="${2:-}" target="$BIN/$1"
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
if [ -x "$BUNDLE/bin/aliceVision_featureExtraction" ] && [ -f "$BUNDLE/lib/libpopsift.so" ] \
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
