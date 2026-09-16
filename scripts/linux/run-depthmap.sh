#!/usr/bin/env bash
# Cheshire Linux: run the HIP aliceVision_depthMapEstimation on a Meshroom cache with the exact
# DepthMap parameters of the node's `standard` preset (Meshroom 2023.3, downscale 2), then
# compare against a reference DepthMap folder if given.
# Usage: run-depthmap.sh <AliceVision bundle or install dir> <cache dir with StructureFromMotion/ and PrepareDenseScene/> <out dir> [reference DepthMap dir]
#   CHESHIRE_DOWNSCALE: DepthMap downscale (default 2; 1 = full-resolution depth maps, 4x the memory per tile)
#   CHESHIRE_DEPTHMAP_EXTRA: extra depthMapEstimation arguments (e.g. "--rangeStart 0 --rangeSize 1")
set -euo pipefail
AV="$1"; CACHE="$2"; OUT="$3"; REF="${4:-}"
SFM=$(ls -d "$CACHE"/StructureFromMotion/*/sfm.abc | head -1)
IMGS=$(ls -d "$CACHE"/PrepareDenseScene/*/ | head -1)
mkdir -p "$OUT"
export ALICEVISION_ROOT="$AV"
export LD_LIBRARY_PATH="$AV/lib:$AV/aliceVision/lib:${LD_LIBRARY_PATH:-}"
"$AV/bin/aliceVision_hardwareResources" 2>&1 | grep -iE "name:|memory" || true
"$AV/bin/aliceVision_depthMapEstimation" --input "$SFM" --imagesFolder "$IMGS" \
  --downscale ${CHESHIRE_DOWNSCALE:-2} --minViewAngle 2.0 --maxViewAngle 70.0 --tileBufferWidth 1024 --tileBufferHeight 1024 --tilePadding 64 \
  --autoAdjustSmallImage True --chooseTCamsPerTile True --maxTCams 10 \
  --sgmScale 2 --sgmStepXY 2 --sgmStepZ -1 --sgmMaxTCamsPerTile 4 --sgmWSH 4 --sgmUseSfmSeeds True --sgmSeedsRangeInflate 0.2 \
  --sgmDepthThicknessInflate 0.0 --sgmMaxSimilarity 1.0 --sgmGammaC 5.5 --sgmGammaP 8.0 --sgmP1 10.0 --sgmP2Weighting 100.0 \
  --sgmMaxDepths 1500 --sgmDepthListPerTile True --sgmUseConsistentScale False \
  --refineEnabled True --refineScale 1 --refineStepXY 1 --refineMaxTCamsPerTile 4 --refineSubsampling 10 --refineHalfNbDepths 15 \
  --refineWSH 3 --refineSigma 15.0 --refineGammaC 15.5 --refineGammaP 8.0 --refineInterpolateMiddleDepth False --refineUseConsistentScale False \
  --colorOptimizationEnabled True --colorOptimizationNbIterations 100 --sgmUseCustomPatchPattern False --refineUseCustomPatchPattern False \
  --nbGPUs 0 --verboseLevel info --output "$OUT" ${CHESHIRE_DEPTHMAP_EXTRA:-}
if [ -n "$REF" ]; then
  python3 "$(dirname "$0")/../compare_depthmaps.py" "$REF" "$OUT" --png "$OUT/compare"
fi
