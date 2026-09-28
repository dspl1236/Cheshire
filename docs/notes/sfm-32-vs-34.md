# Incremental SfM, AliceVision 3.2.0 against 2cb1a39: what can move the reconstruction

Status: reading, 2026-09-27. Nothing here was run. Task brief: `docs/tasks/036-sfm-32-vs-34.md`.
The measurements it starts from are the two "0.3.6" sections of docs/04.

Each claim carries its certainty: **[read]** means read in the code at the stated place, **[inferred]**
means reasoned from what was read and not checked by a run.

## The two trees

- **Meshroom 2023.3.0** (tag `v2023.3.0`, 1503056, 2023-12-11). Its `CHANGES.md` says "Based on
  AliceVision 3.2.0" and links the `v3.2.0` tree [read]. **AliceVision `v3.2.0`** is e8d3445
  (2023-12-11), version.hpp 3.2.0, sfmData file version 1.2.6 [read]. No 3.2.x patch tag exists; the
  next tags are v3.3.0 to v3.3.7, all ancestors of 2cb1a39 [read].
- **Cheshire's base** is 2cb1a39 (2026-08-28, PR #2172), version.hpp 3.4.0, sfmData 1.2.14 [read].
  2,950 commits separate the two [read].

Paths compared in full: `main_incrementalSfM.cpp`, `ReconstructionEngine_sequentialSfM.{hpp,cpp}`,
`BundleAdjustmentCeres.{hpp,cpp}` with its cost functions and manifolds, `SfMLocalizer.cpp`,
`sfmFilters.cpp`, `LocalBundleAdjustmentGraph.cpp`, `sfm/utils/alignment.{hpp,cpp}`,
`RelativePoseInfo.cpp`, `robustEstimation/`, `multiview/resection/`, the triangulation and essential
code, `track/TracksBuilder.cpp`, `regionsIO.cpp`, the radial distortion and pinhole camera code,
`IntrinsicBase.hpp`, `IntrinsicScaleOffset.cpp`, and the intrinsic part of `jsonIO.cpp`. Below,
"3.2:N" and "3.4:N" are line numbers at v3.2.0 and at 2cb1a39.

## First: what the gate's 1 degree is, and why `useAutoTransform` cannot be it

The mesh gate moves A into B's frame by the similarity that best maps A's camera centres onto B's
(`scripts/mesh_distance.py:332`, `:375`), then refines on the surfaces with a trimmed ICP
(`:243`). The "ICP rotation" in the 0.3.6 tables is the ICP's own rotation, so it is what is left
after the camera alignment [read].

`useAutoTransform` runs `computeNewCoordinateSystemAuto` and `computeNewCoordinateSystemGroundAuto`
and applies each with `applyTransform` (main_incrementalSfM.cpp 3.2:317-330, 3.4:299-312), which
moves every pose and every landmark by the same S, R, t (alignment.hpp 3.2:131, 3.4:190) [read]. A
transform of that kind is exactly what the camera alignment removes, so whatever frame the automatic
alignment picks, it cannot leave a rotation between the surfaces after the cameras are aligned
[inferred, from the two readings]. It also did not change: the auto, X-axis, ground and GPS helpers
are the same code in both trees apart from renames and accessors (alignment.cpp 3.2:490-618 and
3.2:949-1200 against 3.4:924-1052 and 3.4:1383-1640; `Pose3::transformSRt` identical) [read].

What can leave such a rotation is a change in where the surface sits relative to the camera centres.
One way is that every camera turns by the same small angle in its own frame while the centres barely
move. A shift of the principal point does that: d pixels turn each camera by atan(d / f), so 1 degree
is d of about 0.017 f, some tens of pixels at phone resolutions. The principal point is refined here
(`--minNbCamerasToRefinePrincipalPoint 3`) within +/-5 % of the image size (BundleAdjustmentCeres.cpp
3.2:747-754, 3.4:472-479) [read]. Focal length and distortion trade against the geometry in similar
ways [inferred]. So the suspects that matter are the ones that change what bundle adjustment
converges to, or which observations it is given.

## Ranked suspects

### 1. The Huber loss threshold went from 16 to 4

- 3.2: `CeresOptions` sets `lossFunction.reset(new ceres::HuberLoss(Square(4.0)))`
  (BundleAdjustmentCeres.hpp 3.2:43), so a = 16 [read].
- 3.4: `CeresOptions(..., double huberLoss = 4.0)` and `new ceres::HuberLoss(huberLoss)`
  (BundleAdjustmentCeres.hpp 3.4:38, :44), so a = 4 [read]. Commit b59096db5 "Move huberLoss as a
  parameter" (2026-07-27), merged with PR #2166 (dev/lidarPipeline, 2026-08-04) [read].
- The sequential engine constructs the default options (ReconstructionEngine_sequentialSfM.cpp
  3.2:740, 3.4:688), and the per-resection pose refinement builds a default `BundleAdjustmentCeres`
  (SfMLocalizer.cpp 3.2:214, 3.4:215). Both get the new value [read]. Only `aliceVision_sfmExpanding`
  was given a way to set it (`--lossParameter`, main_sfmExpanding.cpp 3.4:73, :137);
  `aliceVision_incrementalSfM` has no option for it [read]. `sfm/utils/poseFilter.cpp` 3.4:636 still
  says "delta = Square(4.0) taken from ... HuberLoss(Square(4.0))", which suggests the change of
  value for the other callers was a side effect of making it a parameter [inferred].

Ceres' `HuberLoss(a)` is quadratic up to a residual norm of a and linear beyond. With
`--observationConstraint Scale` the residual is the reprojection error divided by the feature's scale
(ResidualErrorFunctor.hpp 3.2:364-366; intrinsicsProject.hpp 3.4:54-59) [read]. In the pose
refinement after resection the observations carry scale 0, so there the residual is in pixels
(SfMLocalizer.cpp 3.4:204-212) [read].

Why it can move the result: this is the one change found that alters the function every bundle
adjustment minimises. Observations with a residual between 4 and 16 pull quadratically in 3.2 and
linearly in 3.4. Such observations exist at every solve that follows a resection or a triangulation,
before `removeOutliers` (threshold 4 in the same units) drops them, so the two versions converge to
different poses and intrinsics, and then keep a different set of observations [inferred]. Across 140
solves on the engine bay (docs/notes/ba-device-plan.md) the differences compound. Intrinsics that
land elsewhere, the principal point above all, give a camera-to-surface rotation of the kind the gate
measures, and a change of geometry of the order of 0.05 % [inferred]. The size cannot be read from
the code; it needs experiment 2.

### 2. Residuals are measured in undistorted pixels (outlier removal, resection inliers, RMSE)

- 3.2: `IntrinsicBase::residual` returns `x - project(pose, X)` (IntrinsicBase.hpp 3.2:175-179), and
  3.2's `Pinhole::project` always applies the distortion, whatever its flag says (Pinhole.cpp
  3.2:38-43). The residual is in distorted image pixels [read].
- 3.4: `residual` returns `getUndistortedPixel(x) - transformProject(pose, X, false)`
  (IntrinsicBase.hpp 3.4:177-182), i.e. in undistorted pixels [read]. Commit 34a784e85
  "Undistort observations in residual" (after 978dbac4c), PR #1729 (dev/fixSfm, 2024-07-30) [read].
- Callers on the incremental path [read]:
  - `removeOutliersWithPixelResidualError`, after every bundle adjustment (sfmFilters.cpp 3.2:38,
    3.4:38): the `--maxReprojectionError 4` test;
  - `updateScene`, which adds a resected view's 2D-3D inliers to the landmarks
    (ReconstructionEngine_sequentialSfM.cpp 3.2:1664, 3.4:1615). Its threshold `error_max` comes
    from AC-RANSAC run on undistorted points (SfMLocalizer.cpp 3.2:70-78, 3.4:70-104), so 3.2
    compared a distorted residual with an undistorted threshold;
  - the two-view check in the LO-RANSAC triangulation (3.2:1857, 3.4:1809-1810);
  - the final `residual RMSE` in the log (3.2:855, 3.4:804, through `RMSE()` in
    sfm/utils/statistics.cpp).
- The bundle adjustment itself still compares in distorted pixels in both trees
  (ResidualErrorFunctor.hpp 3.2:350-366; intrinsicsProject.hpp 3.4:53) [read]. So in 3.4 the
  filter and the optimiser use different spaces [read].

Why it can move the result: near the image border the two residuals differ by the local stretch of
the distortion model, a few percent for typical k1 [inferred]. Observations close to the 4-unit
threshold there are kept by one version and dropped by the other, which changes the landmark set and
takes away or adds exactly the observations that constrain distortion and principal point [inferred].
Probably smaller than suspect 1 but systematic, and it acts on the landmark set directly [inferred].
One consequence is certain: the RMSE that `sfmbench.py` parses and `quality_gate.py` compares is
not the same quantity in the two binaries, so an RMSE comparison of upstream's binary against
Cheshire's says nothing [read].

### 3. Poses are refined as rotation and centre instead of rotation and translation

The pose block holds [angle-axis, centre] instead of [angle-axis, t] (BundleAdjustmentCeres.cpp
3.2:530-544, 3.4:232-246; cost 3.2 `R X + t` in ResidualErrorFunctor.hpp, 3.4 `R (X - c)` in
costfunctions/projection.hpp:42-52; `REFINE_TRANSLATION` to `REFINE_CENTER` at
ReconstructionEngine_sequentialSfM.cpp 3.2:742, 3.4:690 and SfMLocalizer.cpp 3.2:218, 3.4:219).
Commit 02a3c68b8, PR #2124 (dev/bundleCenter, 2026-06-01) [read].

The objective is the same function of the same poses, so the minimum is the same [inferred]. The
Levenberg-Marquardt path is not: Ceres scales the Jacobian columns and the trust region differently
under the two parameterisations, solves stop at 50 iterations (`max_num_iterations`, 3.2:510,
3.4:211), and each solve feeds an outlier removal [read for the settings, inferred for the effect].
Expect run-to-run-sized differences that the incremental loop can carry forward, not a systematic
rotation by itself [inferred].

### 4. Same model, different solver set-up

All read, all changing rounding or the path, none the objective:

- Cost functions: one `AutoDiffCostFunction` per camera model in 3.2 (BundleAdjustmentCeres.cpp
  3.2:193-245) against a `DynamicAutoDiffCostFunction` around `CostIntrinsicsProject`, which calls the
  camera's own `project` and analytic derivatives, in 3.4 (projection.hpp:68-90,
  intrinsicsProject.hpp:41-92). PR #1778 (dev/mergeSfm, 2024-11-18), PR #1933 (cleanupBundle).
  Cheshire's analytic Jacobians replace these again, and `CHESHIRE_BA_JACOBIANS=autodiff` goes back
  to 3.4's autodiff, not 3.2's.
- Distortion is its own parameter block (3.4:494-517; 3.2 kept it in the intrinsic block,
  3.2:762-770). Commit 7318ef04e, PR #1819. The lock conditions are the same except that 3.4 adds
  `distortion->isLocked()`, false for a 1.2.6 input.
- Solver options added in 3.4: `update_state_every_iteration = true` (3.4:209, PR #2141) and
  `max_num_consecutive_invalid_steps = 10` (3.4:212, commit 124ef16bd, PR #1934; the Ceres default
  is 5).
- Focal bounds: +/-20 % of max(w, h) around the initial focal in both; 3.2 truncates the margin to
  `unsigned int` (3.2:707), 3.4 keeps a double and derives the x bound from the y bound
  (3.4:425-438, PR #1725). Only matters if the focal reaches a bound.
- `IntrinsicsManifold` with the focal ratio locked: in 3.4 `Plus` multiplies x[0] by the ratio
  (manifolds/intrinsics.hpp:65) while `PlusJacobian` puts the ratio on row 1 (:101-102). The two
  disagree unless the ratio is 1. With square pixels it is 1, so harmless here, but wrong for
  anamorphic intrinsics.

### 5. The libraries

2cb1a39 pins Eigen 3.4.0 and Ceres 2.2.0 (src/cmake/DependenciesVersions.cmake:27, :129) [read];
Cheshire's Windows packages use Ceres 2.2.0 built without SuiteSparse
(`scripts/windows/build-ceres-nosuitesparse.cmd`) [read]. v3.2.0 pins nothing (INSTALL.md: Ceres
>= 1.10, vcpkg `ceres[suitesparse,cxsparse]`) [read]; which Ceres, Eigen and SuiteSparse Meshroom
2023.3's own binaries carry was not determined. The sparse solver is used only above 100 poses
(3.2:753, 3.4:701) [read], i.e. for the last few solves of a 107-view run; without SuiteSparse,
Ceres takes `EIGEN_SPARSE` (3.4:54-84) [read]. A different sparse factorisation changes rounding,
not the solution [inferred]. Last in rank.

## Checked and ruled out

- **Option defaults.** Every option the Meshroom 2023.3 node writes is passed explicitly
  (`sfm_cmd`, `scripts/sfmbench.py:167-181`), so 3.4's defaults do not apply. The one changed default
  on this path, `localizerEstimatorMaxIterations` 4096 to 50000 (ReconstructionEngine_sequentialSfM.hpp
  3.2:69, 3.4:71; PR #1683), is overridden by `--localizerEstimatorMaxIterations 4096` [read]. The
  initial pair's relative pose uses a hard-coded 4096 in both [read].
- **What runs after reconstruction** is the same in both `main_incrementalSfM.cpp`: auto transform,
  colours, marker ids, report, save [read].
- **Initial pair, next best view, resection, triangulation.** The engine's diff is renames,
  accessors, `remapLandmarkIdsToTrackIds` moved out, and the two items above [read].
  `robustEstimation/` (AC-RANSAC, LO-RANSAC) and `RelativePoseInfo.cpp` are unchanged; P3P gains an
  overload only; `TriangulateDLT` and the N-view solver are unchanged, and the spherical variants
  that did change are not used by this path [read].
- **Tracks.** `TracksBuilder` gains statistics and optional 2D data, same tracks [read].
- **Local bundle adjustment.** The parameter states moved from the graph into SfMData (PR #1607,
  PR #1973); the distances, states and the 100-pose switch are the same logic [read].
- **Container order.** `HashMap` in 3.2 is `std::map` unless `ALICEVISION_UNORDERED_MAP` is defined,
  and nothing defines it (types.hpp 3.2:29-35), so the move to `std::map` (PR #1647) changes no
  iteration order [read].
- **Observation weights** (PR #2123): the weight is 1 unless `sfmExpanding`'s distance weighting sets
  it (DistanceWeighting.cpp:77, :110), so the `ScaledLoss` branch (3.4:631-635) never runs here [read].
- **Reading 3.2's cameraInit.sfm in 3.4.** For version 1.2.6 the 3.4 loader takes the compatibility
  branch (jsonIO.cpp 3.4:360-395); with pixel ratio 1 it gives the same pixel focal, initial focal
  and principal point as 3.2's loader (jsonIO.cpp 3.2:280-291, :324-331) [read].
- **Camera model.** Radial K3 projection and undistortion are unchanged; the K1 derivative and the
  pinhole Jacobians were rewritten, which changes derivatives, not values [read].

## A Cheshire change this comparison does not exclude

`CHESHIRE_SFM_PENDING_BA` (step 5k, scripts/apply_hip_patch.py:3414-3460) is on by default and runs a
triangulation and bundle adjustment that upstream never runs, when a resection pass ends with views
still pending [read]. It is not in the upstream-equivalent set of docs/04 (Jacobians, persistent
Problem, task seed) [read]. docs/04 notes the crash it fixes never showed on 107 photographs, but
that does not say whether the pending case itself occurred. `sfmbench.py` counts it per run
(`pending_ba_fired`, sfmbench.py:163) [read]. If it fired on the engine bay, part of the gap may be
Cheshire's, not the version's [inferred]. Step 5n's always-on parts (ordering group per block, a
total order in the next-best-views sort) change rounding and ties only [inferred].

## Experiments

Usage below is from `scripts/sfmbench.py` and `scripts/quality_gate.py` on origin/main. Two
constraints [read]: `sfmbench.py run` appends `-- <options>` after the fixed node command line
(sfmbench.py:181), and boost program_options refuses an option given twice [inferred], so the
extras cannot override an option already there (`--useAutoTransform`,
`--minNbCamerasToRefinePrincipalPoint`); change `sfm_cmd` for that. And `quality_gate.py sfm` has
no `--runner`, so legs from another binary go through `sfmbench.py run --runner` and then
`quality_gate.py report`.

**0. Rule out Cheshire's pending BA (no rebuild).** Look at `pending_ba_fired` in
`build/sfmbench/bench.jsonl` for the eb runs. If it is ever above 0:

    python scripts/quality_gate.py sfm eb --leg def --leg nopend:CHESHIRE_SFM_PENDING_BA=0 --repeat 5

**1. Place the difference inside SfM (no rebuild).** Run Meshroom 2023.3's own
`aliceVision_incrementalSfM` and Cheshire's on the same eb cache. This needs a launcher in `build/`
that puts Meshroom 2023.3's `aliceVision/bin` first on PATH and runs its arguments; none is in the
repository, so the name below is a placeholder.

    python scripts/sfmbench.py run eb --tag up --repeat 3 --json --runner mr2023-run.cmd
    python scripts/sfmbench.py run eb --tag ch --repeat 3 --json
    python scripts/quality_gate.py report eb --leg up=up --leg ch=ch --name eb-up-vs-ch

Read "rotation, median degrees": the median angle between each view's orientations after the
centres are aligned (quality_gate.py:146-168). If up-vs-ch is about 1 degree against a within-up
median near the mesh gate's 0.06, the cameras are turned relative to their centres, and the next look
is the intrinsics in the runs' `cameras.sfm` (`principalPoint`, `focalLength`, `distortionParams`).
A principal point that differs by about 0.017 f would account for the degree [inferred]. Compare
poses and landmarks, not RMSE (suspect 2).

**2. Put 3.2's Huber threshold back (one constant, rebuild).** In the experimental tree, set the
default at BundleAdjustmentCeres.hpp:38 to `double huberLoss = 16.0` (3.2's `Square(4.0)`). That
covers the engine's solves (ReconstructionEngine_sequentialSfM.cpp:688) and the pose refinement
(SfMLocalizer.cpp:215). If it should stay as a switch, it would need an environment variable in the
style of the other `CHESHIRE_*` switches; none exists. Then, with a launcher for that build:

    python scripts/sfmbench.py run eb --tag h16 --repeat 3 --json --runner h16-run.cmd
    python scripts/quality_gate.py report eb --leg up=up --leg h16=h16 --name eb-up-vs-h16

and the full Meshroom job with that build, measured as in 0.3.6:

    python scripts/quality_gate.py mesh --leg upstream=U1.obj@U1-cameras.sfm,U2.obj@U2-cameras.sfm --leg h16=H1.obj@H1-cameras.sfm,H2.obj@H2-cameras.sfm --name mesh-eb-upstream-vs-h16

If suspect 1 is the cause, the ICP rotation falls from about 1 degree towards 0.06 and the medians
towards 0.0017-0.0018 [inferred].

**3. Put 3.2's residual space back (one function, rebuild).** Replace the body of
`IntrinsicBase::residual` (IntrinsicBase.hpp:177-182) with
`return x - this->transformProject(pose, X, applyDistortion);`, the state after 978dbac4c and before
34a784e85. Callers that pass `false` already hand it undistorted observations, so they compute the
same thing as now [read]. Run it as a leg `res32`, and one build with both changes as `both`, the
same way as experiment 2. If `both` closes the gap and neither alone does, the two interact.

## Could not be determined

- The Ceres, Eigen and SuiteSparse versions inside Meshroom 2023.3's binaries.
- The size of any suspect's effect: nothing was run, and the ranking is by reading.
- Whether the engine-bay photos carry GPS. With 10 or more GPS views the automatic alignment also
  orients by GPS (alignment.cpp 3.4:1541-1639), but that is a similarity like the rest and is removed
  by the camera alignment [read].
- Whether `CHESHIRE_SFM_PENDING_BA` fired in the 0.3.6 gate runs.
