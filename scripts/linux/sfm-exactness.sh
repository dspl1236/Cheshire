#!/usr/bin/env bash
# Cheshire's own bundle-adjustment solver on Linux: the exactness checks (0.3.7).
#
#   sfm-exactness.sh <bundle> <root> [set ...]
#
# <root>/<set>/cache holds CameraInit/cameraInit.sfm, FeatureExtraction and FeatureMatching (the layout of
# scripts/sfmbench.py's caches, copied over); <root>/<set>/describer names the describer types (e.g. dspsift).
# Colours are not computed (the images stay behind), so the digests are this machine's own: the checks compare
# runs here with each other.
#   host      the host solver alone (CHESHIRE_BA_DEVICE=0)
#   host2     the same again (the run repeats itself)
#   device    the device on every solve it takes, from 1000 rows
#   devcheck  every such solve on the host from a snapshot, then on the device, compared (CHESHIRE_BA_DEVICE=check)
#   direct    the direct build against the Problem build, every solve (CHESHIRE_BA_DIRECT=check)
#   default   the defaults (the device from 200,000 rows), for the time
# One line per run: wall time, the sfm.abc digest, the checks and how many were not the same.
set -u
B=$1
R=$2
shift 2
export ALICEVISION_ROOT="$B" LD_LIBRARY_PATH="$B/lib"

run() {   # set tag [VAR=value ...]
    local s=$1 tag=$2
    shift 2
    local C=$R/$s/cache O=$R/$s/runs/$tag
    rm -rf "$O"
    mkdir -p "$O"
    local t0 t1 rc
    t0=$(date +%s.%N)
    env "$@" CHESHIRE_BA_PROFILE=1 "$B/bin/aliceVision_incrementalSfM" \
        --input "$C/CameraInit/cameraInit.sfm" --featuresFolders "$C/FeatureExtraction" --matchesFolders "$C/FeatureMatching" \
        --describerTypes "$(cat "$R/$s/describer")" --localizerEstimator acransac --observationConstraint Scale \
        --localizerEstimatorMaxIterations 4096 --localizerEstimatorError 0.0 --lockScenePreviouslyReconstructed False \
        --useLocalBA True --localBAGraphDistance 1 --nbFirstUnstableCameras 30 --maxImagesPerGroup 30 \
        --bundleAdjustmentMaxOutliers 50 --maxNumberOfMatches 0 --minNumberOfMatches 0 --minInputTrackLength 2 \
        --minNumberOfObservationsForTriangulation 2 --minAngleForTriangulation 3.0 --minAngleForLandmark 2.0 \
        --maxReprojectionError 4.0 --minAngleInitialPair 5.0 --maxAngleInitialPair 40.0 \
        --useOnlyMatchesFromInputFolder False --useRigConstraint True --rigMinNbCamerasForCalibration 20 \
        --lockAllIntrinsics False --minNbCamerasToRefinePrincipalPoint 3 --filterTrackForks False \
        --computeStructureColor False --useAutoTransform True --initialPairA "" --initialPairB "" \
        --interFileExtension .abc --logIntermediateSteps False --verboseLevel info \
        --output "$O/sfm.abc" --outputViewsAndPoses "$O/cameras.sfm" --extraInfoFolder "$O" > "$O/sfm.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    local abc="-" dev dir devbad dirbad devon
    [ -f "$O/sfm.abc" ] && abc=$(sha256sum "$O/sfm.abc" | cut -c1-16)
    dev=$(grep -c "BA device check" "$O/sfm.log")
    devbad=$(grep "BA device check" "$O/sfm.log" | grep -vc ": same")
    dir=$(grep -c "BA direct check" "$O/sfm.log")
    dirbad=$(grep "BA direct check" "$O/sfm.log" | grep -vc ": same")
    devon=$(grep "BA own profile" "$O/sfm.log" | grep -c ", device")
    printf '%-4s %-9s rc %s wall %7.1f s  abc %s  device solves %s  device check %s (not same %s)  direct check %s (not same %s)\n' \
        "$s" "$tag" "$rc" "$(echo "$t1 - $t0" | bc)" "$abc" "$devon" "$dev" "$devbad" "$dir" "$dirbad"
    grep -m1 "BA device:" "$O/sfm.log" | sed 's/^/     /'
    grep -m3 -i "device failed\|cheshire.*fallback" "$O/sfm.log" | sed 's/^/     /'
}

for s in "$@"; do
    run "$s" host CHESHIRE_BA_DEVICE=0
    run "$s" host2 CHESHIRE_BA_DEVICE=0
    run "$s" device CHESHIRE_BA_DEVICE_MIN_ROWS=1000
    run "$s" devcheck CHESHIRE_BA_DEVICE_MIN_ROWS=1000 CHESHIRE_BA_DEVICE=check
    run "$s" direct CHESHIRE_BA_DIRECT=check
    run "$s" default
done
