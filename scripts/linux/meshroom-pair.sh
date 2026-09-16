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
BIN="$MESHROOM/aliceVision/bin"
[ -d "$BIN" ] || { echo "$BIN not found: is $MESHROOM a Meshroom 2023.x Linux bundle?"; exit 1; }
if [ "${2:-}" = "--unpair" ]; then
  for name in aliceVision_depthMapEstimation aliceVision_featureMatching; do
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
# re-pairing. CHESHIRE_DEPTHMAP=cuda|hip forces one (for every paired node).
if [ "\${CHESHIRE_DEPTHMAP:-auto}" = cuda ] || { [ "\${CHESHIRE_DEPTHMAP:-auto}" = auto ] && nvidia-smi >/dev/null 2>&1; }; then
  exec "$target.cuda" "\$@"
fi
# Meshroom's environment stays; the Cheshire bundle's libraries go first so the HIP build
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
if LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/aliceVision_featureMatching" --help 2>&1 | grep -q -- '--rangeStart'; then
  pair aliceVision_featureMatching
else
  echo "bundle's aliceVision_featureMatching has no GPU matcher (pre-v0.2.5): DepthMap paired only"
fi
