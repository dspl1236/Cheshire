#!/usr/bin/env python3
"""Run complete Meshroom pipelines on a Cheshire package and check every GPU port took part.

    verify_end_to_end.py <meshroom dir> <package dir> <photos> <out root> [config ...]

Runs on Windows and on Linux: both platforms pair the same way and take the same two arguments, so
only the pairing script's name, the directory it lives in here, and whether it needs a shell differ.

Until 2026-09-20 nothing had ever run a whole Meshroom graph on a Cheshire package. The stage gates
(verify_bundle_stages.py, verify-cuda-stages.ps1) drive one node at a time from a reference cache,
and the reference caches under build/meshroom were themselves made with DepthMap paired and nothing
else - every other node in them ran stock Meshroom, which is why they carry no [cheshire] lines at
all. So the GPU filter, meshing and texturing ports had never once run inside a pipeline; they had
only ever been driven by hand.

This pairs the Meshroom install with the package, runs meshroom_batch end to end under several
parameter sets, and for each run checks BOTH that a textured mesh came out AND that every paired
node logged its own port's line. That second half is the point: a run that silently fell back to
Meshroom's own binaries produces a perfectly good mesh, and file counts cannot tell the difference.

The last config inverts the test. It sets every CHESHIRE_GPU_* switch to 0 and requires each port
to log its DISABLED line instead, because a package whose CPU fallback is broken passes every other
check here and fails on the first machine without a supported card.

Markers follow the rule the stage gates arrived at the hard way: take the line from the success
branch of the port's available(), and include enough of it that the disabled branch cannot match.
Bare 'GPU brute-force' matches "GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0".
"""
import json
import os, re, shutil, subprocess, sys, time
from pathlib import Path

# The lines each node's ports print when their GPU paths are live - every one must appear. Absent
# = that node ran something else, or a port fell back without saying so. Meshing carries four
# ports (votes, max-flow, visibility knn, sim blur) and only the votes are provable on a default
# run: max-flow prints only under its verbose flag, sim blur prints only when it is NOT used, and
# the visibility knn's "GPU knn index: N points" is behind CHESHIRE_GPU_VIS_LOG / _CHECK - a first
# version of this table required it and would have failed every default run while the port was
# in fact running. Three ports silent on success is a source defect of the kind the marker rule
# forbids, found 2026-09-20 while inventorying the knobs; it is a rebuild to fix (every payload
# plus the Linux bundle), so it is scheduled rather than slipped in. Under `verify`, where the
# check flags are on, the knn port proves itself through its self-check verdict instead.
GPU_MARKERS = {
    "FeatureExtraction": [r"Choosing device \d+:",
                          r"GPU SIFT keypoints in stable order"],
    # 0.3.8: AC-RANSAC's bound (step 8d) announces itself at the first estimation it serves; CPU code, so it
    # is required in the fallback run too
    # after 0.4.0: the GPU search and the geometric filter at the same time (step 10a)
    "FeatureMatching":   [r"GPU brute-force L2 2-NN on",
                          r"7-point nullspace: (SVD \(default|Householder QR \(CHESHIRE_QR_NULLSPACE=1\))",
                          r"cheshire: AC-RANSAC skips the residual sort and the NFA scan of models that cannot beat the best so far",
                          r"cheshire: GPU search and geometric filter overlapped"],
    # after 0.4.0: the T cameras (step 10e) and the tiles' depth lists (10h) from each camera's own landmarks, the
    # tiles overlapping, no wait for the device per tile (10j), and the batches overlapping, with the images' headers
    # and the tiles' T cameras on every core (10k); host code, so required in the fallback run too
    "DepthMap":          [r"Number of GPU devices",
                          r"cheshire: depth map T cameras from each camera's own landmarks",
                          r"cheshire: depth map depth lists from each camera's own landmarks",
                          r"cheshire: depth map tiles overlap",
                          r"cheshire: depth map batches overlap",
                          r"cheshire: image headers read on every core",
                          r"cheshire: depth map tiles' T cameras for \d+ cameras in \S+ s on every core"],
    "DepthMapFilter":    [r"depth map filter: group votes on",
                          r"depth map filter cache: cap \d+ MB"],
    # 0.3.2: every port announces on both paths, so the three that were silent on success
    # (max-flow, sim blur, visibility knn) are required here like the rest.
    "Meshing":           [r"meshing votes: ray marching on",
                          r"max-flow: GPU push-relabel on",
                          r"sim blur: Gaussian on the GPU",
                          r"visibility knn on the GPU",
                          r"visibility votes on the GPU"],
    "Texturing":         [r"texturing: pyramid \+ rasterisation on"],
    # 0.3.7: Grin, the own bundle-adjustment solver, announced once per run from its success branch (a run
    # that fell back to Ceres for every solve would not print it). After 0.3.9: the 5-point solver's
    # nullspace (step 9j), which the initial pair's estimation runs, names its path once per process; CPU
    # code, so it is required in the fallback run too
    "StructureFromMotion": [r"cheshire: incremental SfM: a resection pass that ends without a bundle adjustment gets one",
                            r"cheshire: BA solver: Grin, Cheshire's own, for the Schur solves",
                            r"5-point nullspace: (SVD \(default|Householder QR \(CHESHIRE_QR_NULLSPACE5=1\))"],
    # 0.3.6: PrepareDenseScene has been paired since v0.2.9 and was checked by nothing, so a run
    # whose PrepareDenseScene was Meshroom's own binary passed. CPU only; the marker is the direct
    # 8-bit read (step 6n, hip/port/image_read/direct8.txt) announcing its first image. It is printed
    # only on that path's success branch (CHESHIRE_READ_DIRECT=0 prints nothing), and also when
    # CheshireJPG (6r) supplies the pixels, since the device decode feeds the same conversion.
    "PrepareDenseScene": [r"cheshire: 8-bit images read directly into float RGB\(A\)"],
}
# ... and the lines printed when switched off. DepthMap and PrepareDenseScene have no such switch,
# so they keep their markers in the fallback run. FeatureExtraction is forced onto the CPU there
# (forceCpuExtraction=True, see CPUFALLBACK below), so its GPU SIFT lines cannot appear and it is
# held only to being the paired binary, as in the defaults configuration. Sim blur announces its
# disabled state; max-flow and visibility do not, so with CHESHIRE_GPU_MAXFLOW=0 and
# CHESHIRE_GPU_VIS=0 set the fallback run exercises their CPU paths without being able to prove it.
CPU_MARKERS = {
    "FeatureExtraction": [],
    "DepthMap":          GPU_MARKERS["DepthMap"],
    "PrepareDenseScene": GPU_MARKERS["PrepareDenseScene"],
    "FeatureMatching":   [r"GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0",
                          r"cheshire: AC-RANSAC skips the residual sort and the NFA scan of models that cannot beat the best so far"],
    "DepthMapFilter":    [r"depth map filter: disabled by CHESHIRE_GPU_FILTER=0"],
    "Meshing":           [r"meshing votes: disabled by CHESHIRE_GPU_VOTE=0",
                          r"sim blur: disabled by CHESHIRE_GPU_BLUR=0",
                          r"max-flow: disabled by CHESHIRE_GPU_MAXFLOW=0",
                          r"visibility knn: disabled by CHESHIRE_GPU_VIS=0"],
    "Texturing":         [r"texturing: disabled by CHESHIRE_GPU_TEX=0"],
    "StructureFromMotion": GPU_MARKERS["StructureFromMotion"],
}
SILENT_PORTS = None  # 0.3.2: no port is silent on either path any more
# Meshroom's own defaults: FeatureExtraction runs DSP-SIFT on the CPU (forceCpuExtraction defaults to
# True), so its GPU lines cannot appear; the node must still be the paired binary. Every other port
# is as in a GPU run.
DEFAULT_MARKERS = dict(GPU_MARKERS, FeatureExtraction=[])
# Host votes: Meshing prints its disabled line instead of "visibility votes on the GPU" (asserted in the
# configuration's checks); every other port as in a GPU run. First run end to end in the 0.3.5 release
# gate, which it failed at 6/7 for requiring the GPU line.
HOSTVOTES_MARKERS = dict(GPU_MARKERS, Meshing=[m for m in GPU_MARKERS["Meshing"] if "visibility votes on the GPU" not in m])

# Lines that must NOT appear in any configuration. Step 5k (scripts/apply_hip_patch.py) makes the
# local-BA graph skip an edge to a posed view it was never handed, where upstream throws "invalid
# map<K, T> key" (Meshroom #2344), and warns when it does. The run then completes, so nothing else
# here would notice; a gate run that needed the workaround should say so.
# 0.3.8: the steps that keep a fallback say so when they take it, and their self-checks (under `verify`)
# print a count that must be N of N - the verdicts below find a passing line, these find a failing one in
# any chunk's log. The epipolar loop (8e) falls back when its self-test finds this build's Eigen and flags
# do not reproduce upstream's error(); the run is right, but slower, and the release should know.
EPIPOLAR_FALLBACK = r"vectorised epipolar distance does not reproduce FundamentalEpipolarDistanceError"
ACR_BOUND_WRONG = r"AC-RANSAC bound check: \d+ of \d+ models skipped the sort and the NFA scan, [1-9]\d* of them below minNFA"
TRACKS_DIFFER = r"tracks file check: [1-9]\d* of \d+ tracks"
# after 0.4.0 (steps 10c-10e): the tracks file's binary copy against the parsed file, SfM's statistics and filters
# on every core against upstream's loops, and the depth map's T cameras against upstream's walks
TRACKS_COPY_DIFFER = r"tracks copy check: [1-9]\d* of \d+ tracks differ"
SFM_FILTER_WRONG = [r"SfM filter check: .* differs from upstream's loop", r"SfM filter check: \b(\d+) of (?!\1 )\d+ comparisons"]
FORBIDDEN = {
    "StructureFromMotion": [r"local BA graph: \d+ edges to posed views the graph was never handed were skipped",
                            EPIPOLAR_FALLBACK] + SFM_FILTER_WRONG,
    "DepthMap": [r"depth map T cameras check: \b(\d+) of (?!\1 )\d+ tiles",
                 r"depth list check: a tile's depth list differs", r"depth list check: \b(\d+) of (?!\1 )\d+ tiles"],
    "TracksBuilding": [r"the tracks file's binary copy could not be written", TRACKS_DIFFER],
    "FeatureMatching": [EPIPOLAR_FALLBACK,
                        ACR_BOUND_WRONG,
                        r"epipolar residual check: \b(\d+) of (?!\1 )\d+ residuals"],
    "PrepareDenseScene": [r"EXR deflate write check: '.*' does not read back as written",
                          r"EXR deflate write check: \b(\d+) of (?!\1 )\d+ files",
                          r"undistort on the device failed",
                          r"undistort device check: \b(\d+) of (?!\1 )\d+ images"],
    # 0.4.0, the new SfM pipeline: its self-checks (under `verifyexp`) and the fallbacks its steps take. NACRANSAC
    # takes 8d's bound (9a) and runs in batches (9b), the tracks file is parsed into structs (9h), and ExportImages
    # reads, writes (9d, inside 8f's scope) and warps on the device (9i) as PrepareDenseScene does.
    "RelativePoseEstimating": [ACR_BOUND_WRONG, TRACKS_DIFFER, TRACKS_COPY_DIFFER],
    "SfMBootStrapping": [TRACKS_DIFFER, TRACKS_COPY_DIFFER],
    "SfMExpanding": [ACR_BOUND_WRONG, TRACKS_DIFFER, TRACKS_COPY_DIFFER,
                     r"NACRANSAC batch check: \b(\d+) of (?!\1 )\d+ calls",
                     r"NACRANSAC batch check: the batches' answer differs"] + SFM_FILTER_WRONG,
    "SfMColorizing": [r"direct 8-bit read check: \b(\d+) of (?!\1 )\d+ images"],
    "ExportImages": [r"direct 8-bit read check: \b(\d+) of (?!\1 )\d+ images",
                     r"EXR deflate write check: '.*' does not read back as written",
                     r"EXR deflate write check: \b(\d+) of (?!\1 )\d+ files",
                     r"the export warp on the device failed",
                     r"export warp device check: \b(\d+) of (?!\1 )\d+ images"],
}

# In-process self-checks: the port runs the CPU reference alongside itself and compares. These are
# the strongest correctness tests the project has, and until 2026-09-20 no gate switched them on.
# All fire inside Meshing. The tedge check's SUMS are not asserted - they differ by an ulp of
# summation order (docs/09) - only its cell counts, which must be equal.
SELF_CHECK_ENV = {
    "CHESHIRE_FILTER_CHECK": "1", "CHESHIRE_MAXFLOW_CHECK": "1", "CHESHIRE_GPU_VIS_CHECK": "1", "CHESHIRE_DENSE_SFM_CHECK": "1", "CHESHIRE_GPU_MATCHER_CHECK": "1", "CHESHIRE_GPU_FILTER_CHECK": "1",
    "CHESHIRE_SEGMENT_CHECK": "1", "CHESHIRE_GPU_TEDGE_CHECK": "1", "CHESHIRE_GPU_VOTE_LOG": "1",
    "CHESHIRE_MESHCLEAN_CHECK": "1", "CHESHIRE_READ_DIRECT_CHECK": "1",
    # 0.3.8: AC-RANSAC's bound (8d) and epipolar loop (8e), checked on every model and every residual;
    # PrepareDenseScene's libdeflate writer (8f) and device undistortion (8g), switched on here whatever
    # their platform defaults (8f is on only with OpenEXR before 3.2, 8g only with four hardware threads
    # or fewer), so every package checks them on its own card.
    "CHESHIRE_ACR_BOUND_CHECK": "1", "CHESHIRE_ACR_RESIDUALS_CHECK": "1",
    "CHESHIRE_EXR_DEFLATE_WRITE": "1", "CHESHIRE_EXR_DEFLATE_WRITE_CHECK": "1",
    "CHESHIRE_UNDISTORT_DEVICE": "1", "CHESHIRE_UNDISTORT_DEVICE_CHECK": "1",
    # after 0.4.0: SfM's statistics and filters on every core (10d), the depth map's T cameras (10e) and its tiles'
    # depth lists (10h)
    "CHESHIRE_SFM_FILTER_CHECK": "1", "CHESHIRE_DEPTHMAP_TCAMS_CHECK": "1", "CHESHIRE_DEPTHMAP_DEPTHLIST_CHECK": "1",
}
SFM_FILTER_SAME = r"SfM filter check: \b([1-9]\d*) of \1 comparisons identical to upstream's loops"
TCAMS_SAME = r"depth map T cameras check: \b([1-9]\d*) of \1 tiles identical to upstream's walks over every landmark"
DEPTHLIST_SAME = r"depth list check: \b([1-9]\d*) of \1 tiles identical to upstream's walks over every landmark"
SELF_CHECK_VERDICTS = {
    # The direct 8-bit read (6n) against OpenImageIO's path, every image, printed at process exit. Since
    # 0.3.8 also the device undistortion (8g) against the CPU loop, and the libdeflate writer's files (8f)
    # read back through OpenEXR.
    "PrepareDenseScene": [r"direct 8-bit read check: (\d+) of \1 images identical to OpenImageIO's path",
                          r"undistort device check: \b([1-9]\d*) of \1 images identical to the CPU loop",
                          r"EXR deflate write check: \b([1-9]\d*) of \1 files read back identical through OpenEXR"],
    # The colours' 8-bit read (8b) against OpenImageIO's path, every reconstructed view. (The legacy engine runs its
    # own passes after each adjustment, on every core since 7l - postAdjust.inc - so 10d's shared ones never run here.)
    "StructureFromMotion": [r"direct 8-bit read check: \b([1-9]\d*) of \1 images identical to OpenImageIO's path"],
    # after 0.4.0: every tile's T cameras (10e) and depth list (10h) from the per-camera lists against upstream's walks.
    "DepthMap": [TCAMS_SAME, DEPTHLIST_SAME],
    # The GPU matcher against upstream's brute force on a sample of every search (uint8 descriptors).
    # Since 0.3.8 also AC-RANSAC: every model the bound (8d) skipped sorted and scanned anyway, none below
    # minNFA, and every residual of the epipolar loop (8e) against upstream's error().
    "FeatureMatching": [r"GPU matcher check: ([1-9]\d*) of \1 sampled queries identical to upstream's brute force",
                        r"AC-RANSAC bound check: \d+ of [1-9]\d* models skipped the sort and the NFA scan, 0 of them below minNFA",
                        r"epipolar residual check: \b([1-9]\d*) of \1 residuals identical to FundamentalEpipolarDistanceError::error"],
    # The depth-map filter's vote pass against the CPU pass, every camera, printed at exit.
    "DepthMapFilter": [r"depth map filter check: ([1-9]\d*) of \1 cameras identical to the CPU vote pass"],
    "Meshing": [
        r"filterByPixSize check: identical to single-threaded upstream on all",
        # The float flow totals of the two algorithms are never equal and are documented as junk
        # (docs/12); the labelling is the verdict. The first version asserted "(identical)" on the
        # flows and failed a run whose labelling was 0 of 1,676,527 cells different.
        # Since 6p the verdict is the cut value: a minimum cut need not be unique, and on the 41-view
        # fold-in graph the two labellings differed in 2 cells Boykov-Kolmogorov left undetermined,
        # with the same value to the digit. Both labellings are evaluated on the adjacency-list graph.
        r"max-flow check: .*cut values on the adjacency-list graph \(double\): CSR labelling \S+, adjacency-list labelling \S+ \(equal",
        # The knn check: identical, vertex AND distance, in both visibility passes (its line names no
        # pass, so the pattern needs two). Until 2026-09-24 Linux builds differed in the last bits of
        # ~20 % of the distances: HIP's __dadd_rn/__dmul_rn live in a header included before the
        # file's fp-contract pragma and fused anyway (docs/04, step 6k). Since then: 0 on the
        # RX 6750 XT (Linux) and the RX 9070 (Windows).
        r"(?s)GPU knn check: identical to nanoflann on all.*GPU knn check: identical to nanoflann on all",
        # The visibility queries built on the device (6k), and the votes, per pass.
        r"GPU backprojection check \(pass 1\): identical to MultiViewParams on all",
        r"GPU backprojection check \(pass 2\): identical to MultiViewParams on all",
        # Since step 10 the votes run on the device by default: the verdict must name the GPU votes, so
        # a pass that silently fell back to host votes fails here (the hostvotes config gates that path).
        r"visibility votes check \(pass 1, GPU votes\): identical to the ordered host reference on all",
        r"visibility votes check \(pass 2, GPU votes\): identical to the ordered host reference on all",
        # The dense point cloud's SfMData built in parallel (6u) against upstream's.
        r"dense point cloud check: identical to upstream on all",
        # MeshClean's counting setup (6l) and pre-screened passes (6i) against upstream's.
        r"MeshClean setup check: identical to upstream's",
        r"cleanMesh check: .*every structure identical",
        r"segmentFullOrFree check: identical to upstream on all",
        r"tedge check: cells with on != 0: cpu (\d+), gpu \1;",
        r"facet weight check: .*differing from the sequential computation: 0\b",
    ],
}

# 0.4.0: the same on the new SfM pipeline (2025.1's Photogrammetry Experimental), whose nodes carry checks of their
# own: NACRANSAC's bound (9a) in RelativePoseEstimating and SfMExpanding, its batches (9b) against the loop, the
# tracks file's parse (9h) against Boost.JSON's DOM wherever it is loaded, the colours' 8-bit read (8b) in
# SfMColorizing, and in ExportImages the direct read (6n), the libdeflate writer (8f, inside 9d's scope) and the warp
# on the device (9i), both switched on whatever the platform's default. FeatureMatching, DepthMapFilter and Meshing
# are the legacy pipeline's nodes and keep their verdicts.
TRACKS_SAME = r"tracks file check: 0 of [1-9]\d* tracks \(\d+ observations\) differ between parse_into and the DOM"
ACR_BOUND_RIGHT = r"AC-RANSAC bound check: \d+ of [1-9]\d* models skipped the sort and the NFA scan, 0 of them below minNFA"
# After 0.4.0 the readers take TracksBuilding's binary copy (10c), which skips the parse; its check reads the copy AND
# parses the file, so 9h's parse check runs too, and both verdicts are asserted.
TRACKS_COPY_SAME = r"tracks copy check: 0 of [1-9]\d* tracks differ from the parsed file"
VERIFYEXP_ENV = dict(SELF_CHECK_ENV, CHESHIRE_ACR_BATCH_CHECK="1", CHESHIRE_TRACKS_PARSE_INTO_CHECK="1",
                     CHESHIRE_EXPORT_DEVICE="1", CHESHIRE_EXPORT_DEVICE_CHECK="1", CHESHIRE_TRACKS_SIDECAR_CHECK="1")
VERIFYEXP_VERDICTS = {n: SELF_CHECK_VERDICTS[n] for n in ("FeatureMatching", "DepthMap", "DepthMapFilter", "Meshing")}
VERIFYEXP_VERDICTS.update({
    "TracksBuilding": [TRACKS_SAME],
    "RelativePoseEstimating": [ACR_BOUND_RIGHT, TRACKS_SAME, TRACKS_COPY_SAME],
    "SfMBootStrapping": [TRACKS_SAME, TRACKS_COPY_SAME],
    "SfMExpanding": [ACR_BOUND_RIGHT, TRACKS_SAME, TRACKS_COPY_SAME, SFM_FILTER_SAME,
                     r"NACRANSAC batch check: \b([1-9]\d*) of \1 calls returned the loop's model, inliers, error and NFA, "
                     r"and left the generator where the loop leaves it"],
    "SfMColorizing": [r"direct 8-bit read check: \b([1-9]\d*) of \1 images identical to OpenImageIO's path"],
    "ExportImages": [r"direct 8-bit read check: \b([1-9]\d*) of \1 images identical to OpenImageIO's path",
                     r"EXR deflate write check: \b([1-9]\d*) of \1 files read back identical through OpenEXR",
                     r"export warp device check: \b([1-9]\d*) of \1 images identical to remapInter"],
})

# Meshroom's FeatureExtraction defaults to dspsift, which goes through vlfeat on the CPU whatever
# the flags - so every config asks for sift, or GPU SIFT is never exercised and the run says
# nothing about PopSIFT. That is the shape of the bug v0.2.17 shipped with.
#
# forceCpuExtraction defaults to TRUE in Meshroom 2023.3, and with it set the node logs [cpu] and
# never touches PopSIFT. meshroom-pair.cmd's own header warns about exactly this - and the first
# run of this script fell into it anyway, reporting a missing GPU SIFT marker as if the package
# were at fault.
SIFT = ["FeatureExtraction:describerTypes=sift", "FeatureExtraction:forceCpuExtraction=False"]
# The fallback run keeps the sift describer but on the CPU. Until 0.3.6 it used SIFT above, so every
# CHESHIRE_GPU_* switch was off while feature extraction still ran PopSIFT on the card, and the run
# said nothing about a machine without a supported GPU.
CPUFALLBACK = ["FeatureExtraction:describerTypes=sift", "FeatureExtraction:forceCpuExtraction=True"]

# The binary each Meshroom node runs, for the provenance check below.
BINARY = {
    "FeatureExtraction": "aliceVision_featureExtraction",
    "FeatureMatching":   "aliceVision_featureMatching",
    "DepthMap":          "aliceVision_depthMapEstimation",
    "DepthMapFilter":    "aliceVision_depthMapFiltering",
    "Meshing":           "aliceVision_meshing",
    "Texturing":         "aliceVision_texturing",
    # 0.3.3: the eighth paired node. CPU only; its marker is the fix announcing itself.
    "StructureFromMotion": "aliceVision_incrementalSfM",
    # 0.3.6: paired since v0.2.9, checked from here on (its launcher line and the 6n marker above).
    "PrepareDenseScene": "aliceVision_prepareDenseScene",
    # After 0.3.9: upstream's new SfM pipeline, paired where Meshroom ships it (2025.1's Photogrammetry Experimental).
    "TracksBuilding": "aliceVision_tracksBuilding",
    "RelativePoseEstimating": "aliceVision_relativePoseEstimating",
    "SfMBootStrapping": "aliceVision_sfmBootstrapping",
    "SfMExpanding": "aliceVision_sfmExpanding",
    "SfMTransform": "aliceVision_sfmTransform",
    "SfMColorizing": "aliceVision_sfmColorizing",
    "IntrinsicsTransforming": "aliceVision_intrinsicsTransforming",
    "ExportImages": "aliceVision_exportImages",
}

# Override paths are Meshroom ATTRIBUTE paths, not AliceVision command-line flags, and the two are
# not the same: some of a node's parameters sit in a group, so --sgmDepthListPerTile on the command
# line is sgm.sgmDepthListPerTile here and --tileBufferWidth is tiling.tileBufferWidth, while others
# are top level whatever they look like - Meshing's maxPoints is declared with advanced=True, which
# is a kwarg and not a group, so Meshing:advanced.maxPoints is a KeyError and Meshing:maxPoints is
# right. A wrong path is a KeyError before any node runs and shows up as a 0s FAIL with every port
# missing, so a config that fails instantly with ports=0/6 is a naming problem, not a GPU one; the
# failure prints the Overrides lines for that reason. The authority is the node definition in
# <Meshroom>/lib/meshroom/nodes/aliceVision/<Node>.pyc, not the AliceVision --help text.
# Meshroom 2025.1's RANSAC defaults, as each node logs its options. checks25 apply on 2025.1 only: 2023.3's
# defaults are the counts the Fast Ransac template sets.
RANSAC_2025 = {"FeatureMatching": [r"\* maxIteration = 50000\b"],
               "StructureFromMotion": [r"\* localizerEstimatorMaxIterations = 50000\b"]}
# After 0.3.9: upstream's new SfM pipeline, Meshroom 2025.1's Photogrammetry Experimental. SfMExpanding takes
# StructureFromMotion's place and IntrinsicsTransforming + ExportImages PrepareDenseScene's, so those two markers
# drop out. The pipeline's own nodes are paired too, each held to the launcher's provenance line and to what our
# build of it announces: NACRANSAC's bound (9a) and batches (9b), the 5-point nullspace (9j), the tracks file's
# parse (9k), the colours' direct 8-bit read (8b), ExportImages' threads and EXR compression (9d). After 0.4.0 the
# tracks file's readers take TracksBuilding's binary copy of it (10c), which TracksBuilding announces writing, and
# a reader announces either the copy or the parse. SfMTransform and IntrinsicsTransforming announce nothing of their
# own (IntrinsicsTransforming reads no tracks file in this template); the provenance line is their check.
# SfMExpanding's resection has the same 50000 default as StructureFromMotion's.
TRACKS_PARSE = r"cheshire: tracks (file parsed straight into structs|read from TracksBuilding's binary copy)"
NEWCHAIN_MARKERS = {n: p for n, p in GPU_MARKERS.items() if n not in ("StructureFromMotion", "PrepareDenseScene")}
NEWCHAIN_MARKERS.update({
    "TracksBuilding": [r"cheshire: tracks file's binary copy written beside it"],
    "RelativePoseEstimating": [r"cheshire: AC-RANSAC skips the residual sort and the NFA scan of models that cannot beat the best so far",
                               r"5-point nullspace: (SVD \(default|Householder QR \(CHESHIRE_QR_NULLSPACE5=1\))",
                               TRACKS_PARSE],
    "SfMBootStrapping": [TRACKS_PARSE],
    "SfMExpanding": [r"cheshire: NACRANSAC runs its iterations in batches on every core",
                     r"cheshire: BA solver: Grin, Cheshire's own, for the Schur solves",
                     r"cheshire: SfM residual statistics and outlier filters on \d+ threads",  # after 0.4.0 (step 10d)
                     r"cheshire: next-best views ranked by score, then by view id",  # after 0.4.0 (step 10i)
                     TRACKS_PARSE],
    "SfMTransform": [],
    "SfMColorizing": [r"cheshire: 8-bit images read as 8-bit RGB directly"],
    "IntrinsicsTransforming": [],
    "ExportImages": [r"cheshire: images exported \d+ at a time, one per thread",
                     r"cheshire: exported images written with EXR compression"],
})
RANSAC_2025_NEW = {"FeatureMatching": [r"\* maxIteration = 50000\b"],
                   "SfMExpanding": [r"\* localizerEstimatorMaxIterations = 50000\b"]}

CONFIGS = {
    # Stock settings, to establish that the paired pipeline works at all. On 2025.1 also that pairing left
    # Meshroom's own template and its RANSAC defaults alone (0.3.9: the faster counts are a template of
    # their own, below).
    "base": dict(overrides=SIFT, env={}, checks25=RANSAC_2025),
    # 0.3.9: the "Photogrammetry Fast Ransac" template the pairing writes on Meshroom 2025.1 (stock
    # Photogrammetry with 2023.3's FeatureMatching maxIteration 2048 and StructureFromMotion
    # localizerEstimatorMaxIterations 4096), by name, so the run also proves Meshroom lists it. Meshroom
    # 2023.3 gets no such template: its defaults are those counts, and the config is skipped there.
    "fastransac": dict(overrides=SIFT, env={}, pipeline="photogrammetryFastRansac", meshroom2025=True, checks={
        "FeatureMatching": [r"\* maxIteration = 2048\b"],
        "StructureFromMotion": [r"\* localizerEstimatorMaxIterations = 4096\b"]}),
    # After 0.3.9: Meshroom 2025.1's Photogrammetry Experimental (the new SfM pipeline) as shipped, and the Fast
    # Ransac copy of it the pairing writes, by name, so the run also proves Meshroom lists it.
    "experimental": dict(overrides=SIFT, env={}, pipeline="photogrammetryExperimental", meshroom2025=True,
                         markers="newchain", checks=RANSAC_2025_NEW),
    "fastransacexp": dict(overrides=SIFT, env={}, pipeline="photogrammetryExperimentalFastRansac", meshroom2025=True,
                          markers="newchain", checks={
                              "FeatureMatching": [r"\* maxIteration = 2048\b"],
                              "SfMExpanding": [r"\* localizerEstimatorMaxIterations = 4096\b"]}),
    # Meshroom's defaults with nothing overridden: what a user who only pairs the package runs.
    # DSP-SIFT on the CPU, then the GPU matcher on its descriptors and every later port. Also the
    # parameter set of the upstream reference meshes the mesh quality gate compares against.
    "defaults": dict(overrides=[], markers="defaults", env={}),
    # Many small tiles: the tiled path is where the memory bridge and the per-tile depth lists live,
    # and a 42-tile run behaves differently from a 1-tile one.
    "tiles": dict(overrides=SIFT + [
        "DepthMap:tiling.tileBufferWidth=512", "DepthMap:tiling.tileBufferHeight=512",
        "DepthMap:tiling.autoAdjustSmallImage=False", "DepthMap:downscale=2"], env={}),
    # The other end: coarse, one depth list for the whole image, a smaller point budget.
    "coarse": dict(overrides=SIFT + [
        "DepthMap:downscale=4", "DepthMap:sgm.sgmDepthListPerTile=False",
        "Meshing:maxPoints=300000"], env={}),
    # A bigger atlas at full resolution, which moves the texturing port's allocations.
    "texbig": dict(overrides=SIFT + [
        "Texturing:textureSide=8192", "Texturing:downscale=1"], env={}),
    # Every GPU port switched off. Must still produce a mesh, and must say it went to the CPU.
    # Until 2026-09-20 this set four of the seven switches and was described as "every port off".
    # 0.3.8: also PrepareDenseScene's device undistortion (8g), on by default on four-thread hosts.
    "cpufallback": dict(overrides=CPUFALLBACK, markers="cpu", note=SILENT_PORTS, env={
        "CHESHIRE_GPU_MATCHER": "0", "CHESHIRE_GPU_FILTER": "0", "CHESHIRE_GPU_VOTE": "0",
        "CHESHIRE_GPU_TEX": "0", "CHESHIRE_GPU_BLUR": "0", "CHESHIRE_GPU_MAXFLOW": "0",
        "CHESHIRE_GPU_VIS": "0", "CHESHIRE_UNDISTORT_DEVICE": "0"}),
    # Every in-process self-check on: each port must agree with its CPU reference, in its own words.
    "verify": dict(overrides=SIFT, env=dict(SELF_CHECK_ENV), checks=SELF_CHECK_VERDICTS),
    # 0.4.0: and the new SfM pipeline under its own checks, Meshroom 2025.1 only.
    "verifyexp": dict(overrides=SIFT, env=dict(VERIFYEXP_ENV), pipeline="photogrammetryExperimental", meshroom2025=True,
                      markers="newchain", checks=VERIFYEXP_VERDICTS),
    # The host votes (bucketed), which the device votes replaced as the default in step 10: still the
    # fallback, so still checked against the ordered reference in both passes.
    "hostvotes": dict(overrides=SIFT, markers="hostvotes", env={"CHESHIRE_GPU_VIS_VOTES": "0", "CHESHIRE_GPU_VIS_CHECK": "1"}, checks={
        "Meshing": [r"visibility votes: disabled by CHESHIRE_GPU_VIS_VOTES=0, host votes",
                    r"visibility votes check \(pass 1, bucketed host votes\): identical to the ordered host reference on all",
                    r"visibility votes check \(pass 2, bucketed host votes\): identical to the ordered host reference on all"]}),
    # The memory bridge under caps. Planner v2 absorbs 1.5 GB on the six-view set by planning
    # smaller tiles - budget 1200 MB, 0 full cameras + 2 tiles, no spill (docs/02) - so that run
    # asserts the re-plan, not a spill; the first version asserted "must spill", which was true of
    # planner v1 and encoded a fact three versions stale. 500 MB is below one tile and must spill.
    # Byte identity across caps is NOT asserted here: every config regenerates SfM from photos and
    # GPU SIFT is not bit-reproducible run to run, so no two runs share an input. That assertion
    # lives in the stage gate, on a fixed SfM, where it belongs.
    "bridgecap": dict(overrides=SIFT, env={"CHESHIRE_BRIDGE_VRAM_MB": "1500", "CHESHIRE_BRIDGE_LOG": "1"},
                      checks={"DepthMap": [r"bridge: vram cap 1500 MB", r"bridge planner 1: VRAM budget 1200"]}),
    "bridgespill": dict(overrides=SIFT, env={"CHESHIRE_BRIDGE_VRAM_MB": "500", "CHESHIRE_BRIDGE_LOG": "1"},
                        checks={"DepthMap": [r"bridge: vram cap 500 MB", r"bridge summary: [1-9]\d* spills"]}),
    # And the bridge switched off entirely: plain device allocation.
    "bridgeoff": dict(overrides=SIFT, env={"CHESHIRE_BRIDGE": "0"}),
    # Texturing's two 0.3.2 ports under their self-checks: the device edge padding against
    # upstream's sequential sweeps (must differ on 0 texels) and the direct OBJ writer with
    # Assimp's file written beside it (scripts/check_textured_obj.py compares them by content).
    # CheshireJPG in PrepareDenseScene's direct read (step 6r, off by default): every photo decoded on
    # the device and compared with libjpeg-turbo's pixels, none handed back to OpenImageIO.
    "gpujpeg": dict(overrides=SIFT, env={"CHESHIRE_GPU_JPEG": "1", "CHESHIRE_GPU_JPEG_CHECK": "1"}, checks={
        "PrepareDenseScene": [r"GPU JPEG check: (\d+) of \1 images identical to libjpeg-turbo",
                              r"GPU JPEG: [1-9]\d* decoded on the device \(CheshireJPG\), 0 read by OpenImageIO"]}),
    "texcheck": dict(overrides=SIFT, obj_check=True, env={"CHESHIRE_GPU_PAD_CHECK": "1", "CHESHIRE_OBJ_CHECK": "1", "CHESHIRE_GPU_RESIZE_CHECK": "1"}, checks={
        "Texturing": [r"GPU padding check: texels differing from the sequential sweeps: 0 of [1-9]",
                      r"GPU resize check: texel channels differing from OpenImageIO: 0 of [1-9]",
                      r"done on the GPU\.",
                      r"Edge padding \(\d+ pixels\) done on the GPU",
                      r"Saving obj mesh file \(cheshire direct writer\)"]}),
    # Full-resolution depth maps, uncapped: the card's own VRAM is the constraint. Per-view working
    # set is 4x the default, so on a large high-resolution set this is what makes the bridge spill
    # naturally rather than under an artificial cap - and on CUDA, where camera mipmaps never pass
    # through the bridge, it is where a small card may run out of what the bridge cannot see.
    "ds1": dict(overrides=SIFT + ["DepthMap:downscale=1"], env={"CHESHIRE_BRIDGE_LOG": "1"}),
    # ds1 plus every opt-in speed path and the profile logs, so the run reports where time went.
    # Most acceleration is already default-on; what this adds is the 7-point QR nullspace (held
    # opt-in, docs/15), the 5-point one (step 9j, docs/17) and a larger filter cache.
    "blast": dict(overrides=SIFT + ["DepthMap:downscale=1"], checks={
        "FeatureMatching": [r"7-point nullspace: Householder QR \(CHESHIRE_QR_NULLSPACE=1\)",
                            r"other +vram: [1-9]\d* allocs"],
        "StructureFromMotion": [r"5-point nullspace: Householder QR \(CHESHIRE_QR_NULLSPACE5=1\)"],
        # 0.3.2 items 4 and 5: the bridge summary (CHESHIRE_BRIDGE_LOG=1) must show the camera
        # mipmaps as "image" and the ports' buffers as "other" - on both backends.
        "DepthMap": [r"image +vram: [1-9]\d* allocs, peak [1-9]\d* MB"],
        "DepthMapFilter": [r"depth map filter cache: cap 8192 MB", r"other +vram: [1-9]\d* allocs"],
        "Meshing": [r"other +vram: [1-9]\d* allocs"],
        "Texturing": [r"other +vram: [1-9]\d* allocs"]}, env={
        "CHESHIRE_BRIDGE_LOG": "1", "CHESHIRE_QR_NULLSPACE": "1", "CHESHIRE_QR_NULLSPACE5": "1",
        "CHESHIRE_FILTER_CACHE_MB": "8192",
        "CHESHIRE_GPU_VOTE_LOG": "1", "CHESHIRE_GPU_VIS_LOG": "1", "CHESHIRE_GPU_TEX_LOG": "1",
        "CHESHIRE_GPU_MATCHER_LOG": "1", "CHESHIRE_FILTER_LOG": "1"}),
}


def node_logs(cache: Path, node: str) -> str:
    """Every log Meshroom wrote for a node, concatenated. Chunked nodes write 0.log, 1.log, ...;
    single-chunk nodes write a file plainly called 'log'."""
    d = cache / node
    if not d.is_dir():
        return ""
    text = []
    for uid in d.iterdir():
        if not uid.is_dir():
            continue
        for f in uid.iterdir():
            if f.is_file() and (f.name == "log" or f.suffix == ".log"):
                text.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(text)


def wrong_binary(cache: Path):
    """The first paired node whose log exists and names Meshroom's own binary, as a message; None
    while nothing has been decided yet or every decision so far was the Cheshire build. Read on a
    timer while meshroom_batch runs, so a mis-paired graph is stopped at its first node."""
    for node in BINARY:
        text = node_logs(cache, node)
        m = re.search(r"\[cheshire\] (\S+): Meshroom's own binary.*", text)
        if m:
            return f"{node} ran {m.group(0)[:120]}"
    return None


def meshroom_2025(meshroom: Path) -> bool:
    """Meshroom 2025.1 and later: the AliceVision node descriptions ship in the AliceVision tree
    (aliceVision/share/meshroom), where 2023.3 compiled them into lib/meshroom/nodes."""
    return (meshroom / "aliceVision" / "share" / "meshroom").is_dir()


def windows_children(pid: int) -> list:
    """Live processes whose parent is pid, on Windows: what an os.execv 're-launch' leaves running
    after the process that was started has ended."""
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]

    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    entry, found = Entry(), []
    entry.dwSize = ctypes.sizeof(Entry)
    ok = k32.Process32First(snap, ctypes.byref(entry))
    while ok:
        if entry.th32ParentProcessID == pid:
            found.append(entry.th32ProcessID)
        ok = k32.Process32Next(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return found


def windows_alive(pid: int) -> bool:
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not h:
        return False
    try:
        return k32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT: still running
    finally:
        k32.CloseHandle(h)


def run_one(name, cfg, meshroom: Path, photos: Path, outroot: Path) -> bool:
    out = outroot / name
    # Meshroom 2025.1's meshroom_batch takes --cache and then computes in <temp>/MeshroomCache anyway,
    # on Windows and Linux: executeGraph saves an unsaved graph to a temporary project file first, and
    # the cache moves next to it (Meshroom issue #2174 reports the symptom; docs/04, 0.3.6). --save
    # alone keeps the cache next to the project, and leaves the project beside the run to read later.
    new = meshroom_2025(meshroom)
    cache = out / ("MeshroomCache" if new else "cache")
    # CHESHIRE_E2E_RESUME=1 keeps an existing cache so Meshroom skips the chunks it already finished.
    # For the bench-pc BSOD of 2026-09-21 (bugcheck 0x1A, three hours and 48 full-resolution depth
    # maps into a run on a 4 GB card) a fresh start would have cost the same three hours again. The
    # classification at the end reads the same logs either way; a resumed node's log carries the
    # provenance line of the run that wrote it, so the pairing rule still holds per chunk.
    if os.environ.get("CHESHIRE_E2E_RESUME") == "1" and cache.is_dir():
        # Meshroom refuses a graph with any chunk still SUBMITTED or RUNNING ("Some nodes are
        # already submitted") - the statuses a dead run leaves behind. Drop those status files
        # (and the chunk's log) so those chunks are computed again; SUCCESS chunks are kept.
        cleared = 0
        for st in list(cache.glob("*/*/status")) + list(cache.glob("*/*/*.status")):
            try:
                state = json.loads(st.read_text(encoding="utf-8", errors="replace")).get("status")
            except ValueError:
                state = None
            if state in ("SUBMITTED", "RUNNING", "ERROR", "KILLED", "STOPPED"):
                st.unlink()
                lg = st.with_name(st.name.replace("status", "log"))
                if lg.exists():
                    lg.unlink()
                cleared += 1
        print(f"        resuming {name} from its existing cache (CHESHIRE_E2E_RESUME=1):"
              f" {cleared} unfinished chunk statuses cleared")
    else:
        shutil.rmtree(out, ignore_errors=True)
    (out / "out").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # Force the paired launcher to the Cheshire package. Its default is 'auto', which hands the node
    # back to Meshroom's own CUDA binary whenever an NVIDIA card is present - correct when Cheshire
    # was AMD-only, and precisely wrong when the package under test is itself the CUDA one.
    env["CHESHIRE_BACKEND"] = "cheshire"
    env.update(cfg["env"])
    if new and os.name == "nt":
        # Meshroom 2025.1's Windows start-up script re-launches the executable with os.execv until
        # ALICEVISION_LIBPATH lists its folder, and on Windows os.execv starts a new process and ends
        # the old one. meshroom_batch.exe exited in half a second while its copy computed the graph,
        # so this saw a 5 s run with nothing in it and unpaired Meshroom under the running node. Set
        # what the script would, so the process waited on here is the one that computes.
        d = str(meshroom)
        env["ALICEVISION_LIBPATH"] = os.pathsep.join(
            [os.path.join(d, "aliceVision", "bin"), os.path.join(d, "aliceVision", "lib"), os.path.join(d, "lib"), d]
            + ([env["ALICEVISION_LIBPATH"]] if env.get("ALICEVISION_LIBPATH") else []))
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(d, "aliceVision", "lib", "python"),
                                             os.path.join(d, "aliceVision", "lib", "python3.11", "site-packages")])

    batch = meshroom / ("meshroom_batch.exe" if os.name == "nt" else "meshroom_batch")
    cmd = [str(batch), "--input", str(photos), "--output", str(out / "out"), "--pipeline", cfg.get("pipeline", "photogrammetry")]
    cmd += ["--save", str(out / "project.mg")] if new else ["--cache", str(cache)]
    # CHESHIRE_E2E_OVERRIDES adds Meshroom parameter overrides to any config without editing the table
    # (space-separated, e.g. "StructureFromMotion:useLocalBA=False"), for one-off runs such as the
    # 884-view False Door where incremental SfM's local bundle adjustment crashes upstream.
    overrides = list(cfg["overrides"]) + os.environ.get("CHESHIRE_E2E_OVERRIDES", "").split()
    if overrides:
        cmd += ["--paramOverrides"] + overrides

    log = out / "meshroom_batch.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        aborted = None
        while proc.poll() is None:
            time.sleep(5)
            aborted = wrong_binary(cache)
            if aborted:
                proc.kill()
        rc = proc.wait()
        # Should the re-launch still happen (a Meshroom whose start-up script wants more than the
        # variables set above), wait for the copy that computes rather than report an empty run.
        relaunched = windows_children(proc.pid) if os.name == "nt" and not aborted else []
        if relaunched:
            print(f"        meshroom_batch re-launched itself (pid {', '.join(map(str, relaunched))}): waiting for it")
            while any(windows_alive(p) for p in relaunched):
                time.sleep(5)
                aborted = wrong_binary(cache)
                if aborted:
                    for p in relaunched:
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p)], capture_output=True)
                    break
    secs = round(time.time() - t0)
    if aborted:
        # Do not let a wrong pairing run for hours to be reported at the end. The 1050 Ti's
        # full-resolution run of 2026-09-20 was two hours and 24 depth maps in before anyone read
        # the first log line: a launcher predating CHESHIRE_BACKEND had handed every node to
        # Meshroom, and the harness would have said so only once the whole graph had finished.
        print(f"  FAIL  {name:<14} ABORTED after {secs}s: {aborted}")
        print("        the launcher paired here does not honour CHESHIRE_BACKEND=cheshire, or the"
              " package was never paired - nothing this run does tests the package")
        return False, None

    mesh = list((out / "out").glob("*.obj"))
    # Any extension: Meshroom's Texturing writes texture_1001.exr by default, not .png, so globbing
    # for .png reported a perfectly good textured mesh as having no textures.
    tex = list((out / "out").glob("texture_*.*"))
    markers = {"cpu": CPU_MARKERS, "defaults": DEFAULT_MARKERS, "hostvotes": HOSTVOTES_MARKERS,
               "newchain": NEWCHAIN_MARKERS}.get(cfg.get("markers"), GPU_MARKERS)

    # Provenance first, then the port. A port marker alone is not proof the Cheshire binary ran:
    # Meshroom's own featureExtraction is a CUDA PopSIFT and prints the very same
    # "Choosing device 0" line, so on an NVIDIA box an UNPAIRED node scored a pass here
    # (found 2026-09-20 on Linux, where the CUDA bundle's versioned libpopsift.so.0.10.0 made the
    # pairing script decline GPU SIFT silently). Both launchers announce the binary they run.
    logs = {n: node_logs(cache, n) for n in markers}
    # A node with no log at all never ran - an earlier node failed and Meshroom stopped there. That
    # is a different fact from "ran Meshroom's own binary", and the first version of this report
    # called it NOT PAIRED: when a comgr-less bundle made DepthMap exit 1, the three nodes behind
    # it were reported as having run the wrong binary, which sent the diagnosis the wrong way.
    never_ran = [n for n in markers if not logs[n].strip()]
    paired = {n: bool(re.search(rf"\[cheshire\] {BINARY[n]}: Cheshire build", logs[n]))
              for n in markers}
    missing = [n for n, pats in markers.items()
               if paired[n] and not all(re.search(p, logs[n]) for p in pats)]
    unpaired = [n for n in markers if not paired[n] and n not in never_ran]
    # Config-specific assertions on top of the port lines: self-check verdicts, bridge announcements, and
    # on Meshroom 2025.1 the checks25 ones.
    checks = {n: list(p) for n, p in cfg.get("checks", {}).items()}
    if new:
        for n, pats in cfg.get("checks25", {}).items():
            checks[n] = checks.get(n, []) + list(pats)
    unmet = [f"{n}: /{p}/" for n, pats in checks.items()
             for p in pats if not re.search(p, node_logs(cache, n))]
    for n, pats in FORBIDDEN.items():
        for p in pats:
            m = re.search(p, node_logs(cache, n))
            if m:
                unmet.append(f"{n}: must not log /{p}/ (logged: {m.group(0)[:100]})")
    if cfg.get("obj_check"):
        # The direct textured-OBJ writer against Assimp's file of the same mesh, by content
        # (CHESHIRE_OBJ_CHECK=1 makes Texturing write both; numbering differs by design).
        tdir = next(iter(sorted((cache / "Texturing").glob("*/"))), None)
        ours = tdir / "texturedMesh.obj" if tdir else None
        theirs = tdir / "texturedMesh.assimp.obj" if tdir else None
        if not (ours and ours.exists() and theirs.exists()):
            unmet.append("Texturing: texturedMesh.obj and texturedMesh.assimp.obj both present")
        elif not Path(__file__).with_name("check_textured_obj.py").exists():
            # it runs from beside this script; a copy of the harness alone crashed the 0.3.5 Linux gate here
            unmet.append("Texturing: check_textured_obj.py beside verify_end_to_end.py (not found)")
        else:
            r = subprocess.run([sys.executable, str(Path(__file__).with_name("check_textured_obj.py")), str(ours), str(theirs)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stdout + r.stderr).strip().splitlines()
                why = tail[-2] if len(tail) >= 2 else (tail[-1] if tail else f"the checker exited {r.returncode} with no output")
                unmet.append("Texturing: direct OBJ content == Assimp's (" + why[:100] + ")")
    # Depth and sim maps as bytes, for the cross-config identity check in main().
    dm_dir = next(iter(sorted((cache / "DepthMap").glob("*/"))), None)
    dm_digest = None
    if dm_dir is not None:
        import hashlib
        exrs = sorted(dm_dir.glob("*_depthMap.exr")) + sorted(dm_dir.glob("*_simMap.exr"))
        if exrs:   # otherwise the digest of nothing (e3b0c442...) masquerades as a result
            h = hashlib.sha256()
            for f in exrs:
                h.update(f.name.encode()); h.update(f.read_bytes())
            dm_digest = h.hexdigest()[:16]

    ok = rc == 0 and mesh and tex and not missing and not unpaired and not never_ran and not unmet
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<14} exit={rc:<4} mesh={len(mesh)} tex={len(tex)} "
          f"ports={len(markers) - len(missing) - len(unpaired) - len(never_ran)}/{len(markers)}"
          f"  dm={dm_digest or '-'}  {secs}s")
    if cfg.get("note"):
        print(f"        note: {cfg['note']}")
    if unmet:
        print(f"        assertions not met: {'; '.join(unmet)}")
    if never_ran:
        print(f"        did not run: {', '.join(never_ran)}"
              f" - an earlier node failed and the pipeline stopped before them")
    if unpaired:
        print(f"        NOT PAIRED: {', '.join(unpaired)}"
              f" - the node ran Meshroom's own binary, so nothing here tested the package")
    if missing:
        print(f"        paired but silent: {', '.join(missing)}"
              f" - the Cheshire binary ran and its GPU port did not announce itself")
    if rc != 0:
        # Name the node that failed and whose binary it was. The skull-turntable run of 2026-09-20
        # died in StructureFromMotion, a node the launcher did not pair before 0.3.3: Meshroom's own
        # aliceVision_incrementalSfM hit a Ceres CHECK (zero-rotation pose after resection), and the
        # report should say so rather than leave the reader to work out from the log that no
        # Cheshire code ran there.
        # A node of one chunk writes <uid>/status; a chunked node writes <uid>/<chunk>.status (Meshroom 2025.1's
        # FeatureMatching, DepthMap, RelativePoseEstimating, ExportImages, ...), which a bare */*/status missed.
        named = set()
        for st in sorted(cache.glob("*/*/status")) + sorted(cache.glob("*/*/*.status")):
            try:
                j = json.loads(st.read_text(encoding="utf-8", errors="replace"))
            except ValueError:
                continue
            if j.get("status") == "ERROR" and st.parent.parent.name not in named:
                node = st.parent.parent.name
                named.add(node)
                ours = bool(re.search(r"\[cheshire\] \S+: Cheshire build", node_logs(cache, node)))
                print(f"        failed node: {node} - "
                      + ("a Cheshire-paired binary" if ours else
                         "Meshroom's own binary (not a paired node, no Cheshire code ran there)"))
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]:
            print(f"        {line[:110]}")
    return bool(ok), dm_digest


def depthmap_identity(results: dict) -> bool:
    """Informational. The first version asserted byte-identical depth maps across configs with the
    same DepthMap parameters and failed every pair: each run regenerates features and SfM from the
    photographs, and GPU SIFT is not bit-reproducible run to run, so no two runs share an input.
    The bridge's byte-identity criterion is asserted by the stage gate on a fixed SfM instead. This
    reports the digests so a run-to-run coincidence is visible, and never fails the matrix."""
    rows = [(n, d) for n, (_, d) in results.items() if d]
    if len(rows) > 1:
        print("  depth-map digests (inputs differ per run - GPU SIFT is not bit-reproducible - so"
              " identity is asserted by the stage gate on a fixed SfM, not here):")
        for n, d in rows:
            print(f"    {n:<14} {d}")
    return True


def unpaired_by_script(pair_output: str) -> list:
    """The expected nodes the pairing script declined, each with the script's own line. Both scripts
    print a line containing "not paired" for a node they leave to Meshroom (the Linux script prints
    "... DepthMap paired only" when the bundle predates the GPU matcher), and name the binary in it:
    aliceVision_<name>, or bare <name> in the Windows script's GPU SIFT line. Nodes outside BINARY
    (ImageMatching) are not gated here."""
    out = []
    for line in pair_output.splitlines():
        if "not paired" not in line and "paired only" not in line:
            continue
        for node, binary in BINARY.items():
            if re.search(re.escape(binary.split("_", 1)[1]), line, re.IGNORECASE):
                out.append((node, line.strip()))
    return out


def unpair(pair_cmd, meshroom: Path, win: bool):
    """Put Meshroom back: pairing renames its binaries in place, and a half-paired install is a trap
    for whoever opens Meshroom next. CHESHIRE_E2E_KEEP_PAIRING=1 leaves it paired: for a node
    (house-pc) whose Meshroom is already paired with this same package by its owner, where unpairing
    would put the dashboard's jobs back on Meshroom's own binaries."""
    if os.environ.get("CHESHIRE_E2E_KEEP_PAIRING") == "1":
        print("\nleft Meshroom paired with this package (CHESHIRE_E2E_KEEP_PAIRING=1)")
        return
    u = subprocess.run(pair_cmd + [str(meshroom), "--unpair"],
                       capture_output=True, text=True, shell=win)
    print("\n" + (u.stdout.strip() or u.stderr.strip()))


def main(argv):
    if len(argv) < 5:
        print(__doc__)
        return 2
    meshroom, package, photos, outroot = (Path(a).resolve() for a in argv[1:5])
    names = argv[5:] or list(CONFIGS)
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        sys.exit(f"unknown config(s): {', '.join(bad)} - have {', '.join(CONFIGS)}")
    if not meshroom_2025(meshroom):
        skipped = [n for n in names if CONFIGS[n].get("meshroom2025")]
        if skipped:
            print(f"=== skipped on Meshroom 2023.3, which has none of 2025.1's templates these run (no Fast Ransac"
                  f" copies - its defaults are those counts - and no Photogrammetry Experimental): {', '.join(skipped)}")
            names = [n for n in names if n not in skipped]

    # Both platforms pair the same way and take the same two arguments; only the script's name,
    # its directory here, and whether it needs a shell differ.
    win = os.name == "nt"
    pair_name = "meshroom-pair.cmd" if win else "meshroom-pair.sh"
    pair = package / pair_name
    if not pair.exists():
        pair = Path(__file__).resolve().parent / ("windows" if win else "linux") / pair_name
    if not pair.exists():
        sys.exit(f"{pair_name} not found in the package or in scripts/{'windows' if win else 'linux'}")
    pair_cmd = [str(pair)] if win else ["bash", str(pair)]

    print(f"=== meshroom {meshroom}")
    print(f"=== package  {package}")
    print(f"=== photos   {photos} ({len(list(photos.glob('*.[jJ][pP][gG]')))} jpg)")
    p = subprocess.run(pair_cmd + [str(meshroom), str(package)],
                       capture_output=True, text=True, shell=win)
    print(p.stdout.strip() or p.stderr.strip())
    if p.returncode != 0:
        return 2
    # A node the script declined to pair runs Meshroom's own binary in every config, and until 0.3.6
    # that was noticed only once the node had run (wrong_binary), hours into a full-resolution run
    # when the node is late in the graph. The script has already said why, so stop here and repeat it.
    declined = unpaired_by_script(p.stdout)
    if declined:
        for node, line in declined:
            print(f"  FAIL  {node} was not paired, so no config can test it: {line}")
        unpair(pair_cmd, meshroom, win)
        return 2

    outroot.mkdir(parents=True, exist_ok=True)
    try:
        print()
        results = {n: run_one(n, CONFIGS[n], meshroom, photos, outroot) for n in names}
    finally:
        unpair(pair_cmd, meshroom, win)

    passed = sum(1 for ok, _ in results.values() if ok)
    print(f"\n{passed} of {len(results)} pipelines ran end to end on this package")
    identical = depthmap_identity(results)
    return 0 if passed == len(results) and identical else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
