# Cheshire roadmap

Written 2026-09-22, the day v0.3.3 shipped. Sizes are S, M, L or XL.
Every item was checked against the tree and git history (159 candidate items from the README,
docs, validation log, memory notes and code; 32 turned out to be done already and are not listed).
Line references are to the tree at the commit that added this file. Upstream paths without a
prefix are under `third_party/aliceVision/src/aliceVision/`.

## 0.3.4

This release covers the CPU work left in the ported nodes and run-to-run determinism. It was
0.3.5 in the earlier proposal and moved up at the user's request.

The group is not thin. Several parts of the earlier plan have already shipped and are not
repeated here:
- dense point-cloud fusion, in part: the exact multi-threaded load and filter (4p, v0.2.12) and
  the gaussian on the GPU (4y, v0.2.16); what is left of fusion is the first Meshing item below
- the Meshing CPU fixes 4g-4i (v0.2.9) and 4n, 4o, 4q, 4s, 4t (v0.2.12), all exact, plus 4u
  (v0.2.12), which changes the mesh on purpose
- GPU edge padding and the direct textured-OBJ writer, 5d/5e (v0.3.2)
- the GPU texture downscale, 5j (v0.3.3)
- the PrepareDenseScene undistortion map, 5h (v0.3.3)
- the GPU SIFT keypoint sort, 5c (v0.3.2)
- the SfM pending-BA fix, 5k (v0.3.3)

What remains is 19 items plus one ride-along fix, four of them L. That is more than one release.
The fusion visibility passes, the S items and the SfM determinism work are the natural core; the
L items can slide.

### Determinism

- **Seed SfM per task, not per process** (L). A fixed `randomSeed` does not make incremental SfM
  reproducible, because resection and LORANSAC triangulation share one `std::mt19937` across OpenMP
  threads, so draw order follows thread scheduling (`sfm/pipeline/ReconstructionEngine.hpp:66,74`,
  `sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.cpp:627-667, :1796`). Start with a
  one-thread rerun (S). Then derive the generator from (seed, viewId) and (seed, trackId), decide
  what to do about Ceres' threaded Schur order, and gate later SfM changes on a byte-identical
  `sfm.abc`. Evidence of the cost (2026-09-23, docs/04 "0.3.4: bundle adjustment's Jacobians"):
  three replays of the False Door SfM cache did 0.57 to 1.54 billion residual-block-iterations of
  bundle adjustment and took 1302 s to 2368 s, so no BA change can be judged by wall clock at that
  size until this lands; per-work costs from `CHESHIRE_BA_PROFILE` are the workaround.
  **Done 2026-09-23 (step 5n, `hip/port/sfm_ba/deterministic.hpp`; docs/04 "0.3.4: incremental
  SfM, reproducible").** The one-thread rerun still differed: Ceres orders parameter blocks within
  a group by address. Now: a generator per (view or track, pass), landmark blocks in one array and
  an ordering group per block by key, a total order in the next-best-views sort - always on - and
  `CHESHIRE_SFM_DETERMINISTIC=1` for one Ceres thread, the only source left. Two deterministic runs
  are byte-identical to each other and to the single-threaded result on 6 and 41 views. The gate
  is `scripts/sfmbench.py run ... --json CHESHIRE_SFM_DETERMINISTIC=1` twice with equal digests.
- **GPU SIFT: one descriptor file differs at 884 views** (M). Across two False Door runs, 1,767 of
  1,768 feature files matched (`docs/04-validation.md:953-955`). Rerun view 216823232 on one package
  to separate run-to-run noise from a build difference. The float atomics in the orientation
  histogram are the first suspect (`third_party/popsift/src/popsift/s_orientation.cu:154`).
- **knn distance rounding on HIP is a missing pragma, not card noise** (S). `knnGPU.cu` has no
  `#pragma clang fp contract(off)`, and on HIP `__dadd_rn`/`__dmul_rn` are plain operators, so the
  "unfused" path is fused anyway (`hip/port/gpu_knn/knnGPU.cu:50-52`). That explains about 1.6 M
  last-ulp distance differences on every HIP build without `__FMA__`
  (`docs/04-validation.md:676` on Linux, `:1151` on the Windows hip6.2 payload). Add the pragma,
  rebuild the Linux bundle and the hip6.2 Windows family, rerun the check on the RX 6750 XT (Linux)
  and the RX 5500 XT (Windows), then tighten `scripts/verify_end_to_end.py:93-96`. **Refuted
  2026-09-22:** the rebuilt Linux bundle reports exactly the same counts on the RX 6750 XT (17,383,299
  and 18,404,248 of 87,354,192; docs/04, "The Linux knn distances are not a contraction"). The pragma
  stays (it is correct), the cause is unknown; next, dump differing pairs and recompute on the host.
- **ImagesCache eviction under the parallel prefetch** (S). Cheshire's own `refreshData` keeps
  upstream's `// TODO: oldCamId should be protected if already used`
  (`hip/port/sgm_fused/imagescache_refresh.cpp.txt:15`), and the prefetch refreshes a whole batch
  in parallel (`hip/port/sgm_fused/prefetch.cpp.txt:17`). Compare the batch camera count (R plus
  the SGM and refine T cameras) with the cache capacity; if a batch can exceed it, cap or chunk the
  prefetch in step 1h. File upstream only if upstream's own path can reach it.

### SfM and FeatureMatching CPU

- **RelativePoseKernel_K: compute F once per model** (S). It rebuilds F from E for every
  correspondence
  (`third_party/aliceVision/src/aliceVision/multiview/RelativePoseKernel.hpp:163-168`), and this is
  on SfM's relative-pose path. The change should be bit-identical. Profile the initial-pair phase
  first, since BA is 63 % of SfM.
- **Per-iteration allocations in the AC-RANSAC 7-point path** (M). `vec_sample`, `vec_models` and
  the dynamic subset matrices are allocated on every iteration
  (`robustEstimation/ACRansac.hpp:281,287`, `robustEstimation/PointFittingKernel.hpp:141-142`).
  Re-profile at Meshroom's `--maxIteration 2048` first. At real settings geometric filtering is
  about 8 s of a 24 s chunk (`docs/07-gpu-matcher.md:116-118`), so the payoff is small except on
  2-4-core hosts. The byte-identical match-file gate applies.
- **Analytic Jacobians for bundle adjustment** (M). Jacobians are 49 % of BA on the engine bay and
  421 s of 733 s of BA on the False Door (`docs/04-validation.md:1052`). Upstream already has
  analytic intrinsic and point derivatives. The autodiff cost comes from the pose and subpose Jet
  passes (`sfm/bundle/costfunctions/projection.hpp:156`). Analytic and automatic derivatives round
  differently, so this cannot be byte-identical: ship it off by default with a check mode against
  autodiff, and turn it on in 0.3.5 once the quality gate exists. The Lanczos downscale it was
  queued behind shipped in 0.3.3. It is also the prerequisite for bundle adjustment on the device
  (0.3.5): the derivatives written by hand here are what goes on the GPU.
  **Done 2026-09-23 (step 5m, `hip/port/sfm_ba/projectionCheshire.hpp`; docs/04 "0.3.4: bundle
  adjustment's Jacobians").** Upstream's inner cost was already analytic; the waste was Ceres'
  4-wide autodiff passes re-running it 3 to 5 times per block. `CHESHIRE_BA_JACOBIANS=stride` (one
  pass, bit-identical: 0 of 67,949,760 values differed) was the default for a day, 1.9x on the
  Jacobian phase; `analytic` (the chain rule by hand) is the default since 2026-09-23 at 3.0x with
  rounding-level differences (max 4.4e-10 relative), residuals identical - the user's call, on
  seven 41-view runs whose landmark counts and RMSE sit inside upstream's run-to-run spread. 41 views: SfM 67 s to 61 s and 57 s; engine bay: 69 s
  to 60 s and 59 s (Jacobians 18.7 s to 9.9 s and 6.8 s).
- **Bundle adjustment: the problem build** (M). **Done 2026-09-23 (steps 5o-5q; docs/04 "what a
  bundle adjustment costs around Ceres' Solve").** Measured first: at 41 views the build was
  2.1 us per residual block (9.2 s of a 27 s BA), Ceres' preprocessor 1.25 us, the destroy 0.5 us,
  and two log-only full residual evaluations per solve. The evaluations are skipped
  (`CHESHIRE_BA_LOG_COST=1` restores them); the build is 1.5 us (default) and 1.2 us (analytic)
  after per-view lookup caching, once-per-landmark ordering, Ceres' array overload and
  `disable_all_safety_checks`; the once-per-block ordering alone changed nothing.
- **Bundle adjustment: a Problem that lives across solves** (L). What is left of the build and
  destroy is Ceres' own per-residual-block bookkeeping; only keeping the Problem across solves
  removes it, and Ceres' preprocessor stays per `Solve`. Ceiling measured at 41 views: build +
  destroy is 1.9 of the 7.5 us every residual block costs per solve, a quarter of BA and a tenth
  of the node. It needs residual blocks added and removed as landmarks and observations change
  between solves and ignored blocks toggled constant, so it is an engine change, not a BA one.
  Judge it with the deterministic gate, not wall clock.
  **Done 2026-09-23 (step 5r, `hip/port/sfm_ba/persistent.inc`; docs/04 "a bundle-adjustment
  Problem that lives across solves").** Build 6.5 s to 2.4 s and destroy 1.8 s to 0.1 s at 41
  views, but Ceres' preprocessor 5.9 s to 7.8 s (its per-solve scans miss cache on residual blocks
  allocated over time), so net 6 % of the node, not a quarter of BA. Invariant-checked on every
  solve of 6, 41 and 884 views; deterministic gate byte-identical. Two upstream traps fixed on
  the way (the intrinsic block vector move-assigned under Ceres' pointer; kept observations of a
  view that lost its pose). Still open: the False Door rebuild-against-persistent pair without
  the check, and whether the preprocessor cost can be brought back down (it is Ceres' allocation
  order, not ours). **884-view pair measured (docs/04):** a wash under the local strategy (the
  frontier moves: 23.6 M residual blocks added, 18.4 M removed, against 70.8 M active; Ceres
  scans every block per Solve), so persistence is now an engine policy: on while every landmark
  is active, off under the local strategy.
- **DepthMap loading, done 2026-09-23** (docs/04 "the depth-map node was decoding"): the node
  was 70-85 % CPU-side image loading around 4.5 s/view of kernels. Tour order for a chunk's
  cameras (5u), load once per batch and only what the device lacks (5t), EXR straight through
  OpenEXR (5v), the host downscale after each read no longer twelve-times-twelve threads wide
  (5w), and that downscale - lanczos3, most of the node's CPU - computed directly with identical
  values at 18x less CPU (5x). Left: the per-chunk SfM load (a larger Meshroom block size), the
  downscale as a 2x2 average on the device (changes values, so a new reference), and the same
  reader for the texturing node's sRGB reads (a conversion after a direct read).
- **The per-solve walk over every landmark** (M). Both the rebuild and the sync visit all 1.35 M
  landmarks per solve at 884 views although the local strategy leaves most ignored (the rebuild's
  build is 2.2 us per active block there against 1.2 us at 41 views). An active-landmark list
  maintained by the local-BA graph, or a state change list, would make the build proportional to
  the active set. Judge with the deterministic gate. (A detour, 2026-09-23: the outlier and
  unstable-pose passes after each solve were restricted the same way, exactly - step 5s, opt-in
  `CHESHIRE_SFM_LOCAL_PASSES=1` - on a mis-measurement that put them at 605 s; they are about
  40 s at 884 views either way, docs/04 "the passes after every bundle adjustment". The
  BA-build walk, 177 s over the run there, is the real one and is still open.)
- **Eigen temporaries in the inner projection derivatives** (M). **Done 2026-09-23 (docs/04
  "the inner projection fused"):** one walk of the chain for pinhole + none / K1 / K3 / Brown,
  contraction off; Jacobians 5.5 s to 3.2-3.5 s at 41 views, the node 59 s to 50 s.
- **CPU share on weak hosts** (S; M if FeatureMatching has to be rerun for the split). On the
  Rubble box (i3-4330, 2 cores) SfM took 4291 s and FeatureMatching 2265 s
  (`docs/04-validation.md:1034`), but the matcher vs AC-RANSAC split there is a guess. Measure it,
  and run `CHESHIRE_BA_PROFILE` on the 1591-pose SfM before choosing between the two items above;
  that replay doubles as the CHOLMOD item under Packaging if it is run against both Ceres builds.
- **A current end-to-end engine-bay run** (S). One 107-photo run on the RX 9070 box with the 0.3.3
  package, all eight nodes paired, per-node times recorded in docs/04. The README's 39 to 30 minute
  figure (`README.md:44-47`) was measured at v0.2.5 (`docs/07-gpu-matcher.md:77`), and every
  per-node ranking since has come from other sets.

### Meshing CPU

- **Fusion: the visibility passes on the GPU** (L, the release's headline). Measured at 833 views
  (`docs/04-validation.md:1177-1199`): fusion is about 290 s of a 480 s Meshing, and its two
  visibility passes are 140-200 s of that, bound by HOST work - collecting votes (38-59 s) and
  backprojecting pixels (23-63 s), camera by camera - while the knn kernels overlap and the GPU
  waits (device wait 1.7 s of 72 s in pass 1). The work is to move backprojection and vote
  collection onto the device with each point's camera list in upstream's order, checked in-process.
  Designed 2026-09-22 (three designs judged; the plan is in the session record and docs/04). Done the
  same day: step 1, the knn contraction pragma (5d6045b); step 2, the per-pass digest, diagnostics
  and the votes check against a host-fed reference (7213bfa); step 3, bucketed host votes (ebc33f0):
  exact on 6, 41 and 833 views, visibility passes 110.5 -> 94.5 s and Meshing 357-439 -> 340 s at 833
  views, pass 1 now at the device's 62 ms per camera. Next: measure the kernel time on an idle box
  (it rose once the host stopped holding it up), the Linux rerun for step 1, then steps 4-7, the
  device votes behind CHESHIRE_GPU_VIS_VOTES with guards and recovery. The 133 s pass 2 was load.
- **Fusion: the rest** (M). filterByPixSize's CPU kd-tree is about 16 s at 833 views (the GPU
  radius filter that was briefed was never built), removeInvalidPoints plus margin setup about 9 s,
  and the depth-map load 43 s warm or 111 s cold, which is disk and EXR decode, not a GPU job.
- **MeshClean** (M). 12.9 s on the engine bay: 6.6 s of setup, then four single-threaded iterations
  (`docs/11-meshing-cpu.md:84-89`). At 833 views, post-cut processing plus cleaning is about 95 s
  (`docs/04-validation.md:1197`). Plan: a parallel read-only pre-screen, then a serial index-order
  pass checked byte for byte, with the setup built as a counting table like 4g.
- **Graph-cut post-processing flood fills** (M). Three serial flood fills still take about 4.2 s
  (`docs/11-meshing-cpu.md:191-196`). A parallel connected-components pass checked against
  `_cellIsFull` under `CHESHIRE_SEGMENT_CHECK` keeps the result exact. The solid-angle pass
  (2714 ms) is already OpenMP and must keep its summation order.
- **s-t graph build: measure the split first** (M). The 24.6-26.1 s bucket holds the GPU vote
  kernels (about 11-15 s), serial host packing (about 2.3 s), facet weights and edge recording. The
  serial CSR layout, about 3-4 s (`hip/port/meshing_csr/MaxFlow_CSR.hpp:192-233`), is billed to the
  cut. Add per-phase timestamps, then parallelise under `CHESHIRE_MAXFLOW_CHECK=1`.

### Texturing CPU

- **Re-profile the node, then pick** (L). What remains:
  - image loads, about 26 s of I/O
  - the rest of the UV unwrap: 8.56 s on the RX 9070 box, 17.3 s on the i3
  - the EXR write
  - Assimp's mesh load, about 9 s. It is kept on purpose for non-OBJ input, but an OBJ fast path
    is possible.

  Source: `docs/04-validation.md:783-784`. The "UV generation ~53 s" label at `:515` looks wrong,
  so the order of work is unknown until the profile.
- **computeTrisCamsFromPtsCams** (M). createCharts spends 2.44 s here serially, plus about 2.96 s
  in the projection loop (`docs/10-gpu-texturing.md:147-150`, `mesh/Mesh.cpp:2076-2097`). Filling
  the list by index keeps it exact. The gain is at most about 5 s on the engine bay.
- **Large sets: cost is atlas count times camera count** (L). False Door texturing took 5022 s: 13
  passes over 839 cameras for 51 atlases, where upstream's rule of 9 atlases per pass would have
  made 6 (`docs/04-validation.md:915-919`, `scripts/apply_hip_patch.py:884-904`). Profile with
  `CHESHIRE_GPU_TEX_LOG`, then choose between fewer passes and reusing decoded images.

### PrepareDenseScene and DepthMap host work

- **PrepareDenseScene I/O** (L). The node is bound by JPEG decode plus OCIO and by the EXR write.
  On house-pc the split is read 50.6, undistort 36.5 and write 94.7 thread-seconds
  (`docs/04-validation.md:853-855`), and Rubble spent 3363 s in this node. First decide whether
  this is Cheshire's work at all. If it is, output must stay at 107/107 byte-identical EXRs.
- **DepthMap host image loads** (M). About 30 % of the 41-view run is host time, and twelve
  parallel loads finish at roughly one per 0.5 s (`docs/05-performance.md:27-31`). Add a per-phase
  split in the style of `CHESHIRE_PDS_PROFILE` to tell a serial section from 6-core saturation.

### Rides the same rebuild

- **Bridge counts camera mipmaps at half size** (S). Step 5f divides by 2, but `CudaRGBA` is
  already the 8-byte half4 (`scripts/apply_hip_patch.py:3011-3016`). So CUDA and Windows-native
  runs read 372 MB where Linux emulation reads 744 MB. Drop the `/ 2` and fix the docs lines listed
  under Stale text.

Parked items from this group are listed under Parked and exploratory.

## 0.3.5

GPS image pairing, reconstruction quality, and the first SfM work on the GPU. This was 0.3.4 in
the earlier proposal.

- **Reconstruction-quality gate** (L). No gate measures quality today: `verify_end_to_end.py`
  checks outputs, provenance, port markers and self-check verdicts
  (`scripts/verify_end_to_end.py:300-330`). Build a script for landmark count, mean reprojection
  error, poses and mesh-to-reference distance on mini6 and the engine bay. Take the thresholds from
  measured run-to-run spread. docs/17 needed n=10 per leg because SfM varies
  (`docs/17-svd-nullspace.md:131-136`), so if 0.3.4 makes SfM reproducible, the repeats get cheaper.
  The same gate would serve an optional DepthMapFilter STRICT threshold study (M,
  `docs/08-gpu-depth-map-filter.md:54-56`).
- **QR nullspace default** (S). QR has shipped opt-in package-wide since v0.2.18 (f5bde79; first in
  the RDNA3/RDNA4 half of v0.2.17.1), after a ten-run study found -165 landmarks (p 0.0008) and
  +0.0024 RMSE (p 0.032) (`docs/17-svd-nullspace.md:126-127`). Flipping the default is the user's
  decision and waits for the quality gate. Needed either way:
  - fix `CHESHIRE_QR_NULLSPACE=0` turning QR on (`scripts/apply_hip_patch.py:2799`)
  - give the base e2e config a QR check if the default flips
  - record today's drone-set A/B (29 of 1051 pairs kept, matches within 0.3 %), which exists only
    in the session
- **"Nullspace SVD" is the same item, not separate work.** README roadmap item 6 holds the SVD
  "until the quality measurement exists" (`README.md:403-412`). That measurement was done, and the
  QR path it concerns is the one above. The over-determined branch and the spherical solver stay on
  the SVD (see Not planned).
- **Bundle adjustment on the device** (L; the first SfM work on the GPU). StructureFromMotion is
  the one heavy node that has never touched the GPU, and bundle adjustment is where its time goes:
  63 % of SfM on the engine bay (50.9 s of 92 s: Jacobians 25.1 s, linear solver 13.1 s, per-solve
  bookkeeping 11.0 s; `docs/04-validation.md:750-756`) and 733 s of the 1498 s False Door replay
  (Jacobians 421 s, linear solver 304 s; `docs/04-validation.md:1052-1053`). Ceres' own GPU
  solvers are CUDA-only, so this is our loop, not a Ceres option, and it serves AMD and NVIDIA
  alike. In order:
  1. Jacobians on the device, after the 0.3.4 analytic-Jacobian item, whose hand-written
     derivatives are exactly what runs there: residuals and derivative blocks per observation on
     the GPU, handed to the solver in Ceres' block structure, with a check mode against autodiff
     on the host as for the CPU item. This is the 49 %, and it is embarrassingly parallel.
  2. The Schur solve on the device, only if the profile after step 1 says it is the wall: a
     preconditioned CG on the reduced camera system. At 884 views that system is a few hundred
     poses, where Eigen's factorisation keeps up with CHOLMOD (`docs/04-validation.md:1055-1057`),
     so this may never be needed below thousands of poses.

  Not global SfM (it changes the geometry; every published pose comparison is against
  incremental). It cannot be byte-identical to autodiff, so it ships off by default behind the
  quality gate, like the CPU item. Incremental SfM stays sequential (resect, triangulate, BA); only
  the inner loops move, so the ceiling is the BA share of the node, not the node.
- **Env switches that `=0` turns on** (M). About 13 behaviour switches and about 15 check, log and
  profile flags test only whether the variable is set, and four parsing styles coexist (e.g.
  `scripts/apply_hip_patch.py:1017, :2799`). Route them all through one helper, then do a full
  rebuild and re-gate. Do this before flipping any default.
- **GPS-radius image pairing** (M). On a 444-photo DJI survey, the vocabulary tree proposed
  partners a median 214 m apart and SfM placed 49 of 444. Upstream has no GPS mode
  (`imageMatching/ImageMatching.cpp:31-43`), but it has `ImageInfo::getGpsPositionFromMetadata`.
  Measured on the exhaustive rerun (docs/04, "A 444-photo drone survey"): the real pairs sit a median
  29 m apart, 99 % within 56 m; an 80 m radius (5.4x the shot spacing) proposes 9,071 pairs, 9.2 % of
  exhaustive, and catches 2,376 of 2,377 verified pairs (the last is 302 m away, a false match).
  Steps:
  1. The exhaustive house-pc rerun is the baseline: 444/444 views, 2,377 verified pairs.
  2. Add an opt-in mode in `aliceVision_imageMatching` with a fallback for views without GPS, paired
     as the ninth binary.
  3. Validate on views placed and on pair count (about 5-8k against 98,346).
- **GPU matcher self-check** (S, or M with the filter check). The matcher is the only port with no
  `*_CHECK` (`hip/port/gpu_matcher/gpuMatcher.cu:265-370`). Add a sampled CPU comparison, which can
  be exact because uint8 distances are integers. The DepthMapFilter vote pass has no check either
  (`scripts/apply_hip_patch.py:637-655`).
- **Exact mode for compare_depthmaps.py** (S). It prints 0.0000 for anything under 5e-5 and still
  applies the strict 98 % bar (`scripts/compare_depthmaps.py:67,71`). As a result
  `scripts/bridge_matrix.py:50-59` can report a sub-5e-5 difference as bit-identical. Add
  decoded-pixel counts and make PASS/FAIL selectable.

## Cheshire Node (own milestone)

- **Cheshire Node** (XL). A static GitHub Pages UI that drives a localhost daemon with a CORS
  allowlist and a per-install token. The house-pc app becomes a Cheshire component, and
  `reconstruct` and `jobstatus` get ported so they run on Windows. Its "after 0.3.3" condition is
  now met, but nothing is built or scheduled.
- **Node glue left from NVIDIA** (M). The segmentation venv is still torch 2.5.1+cu121, and
  `genmasks` refuses to run without CUDA (`services/photogrammetry/bin/genmasks:100-103` in
  haus-infrastructure), so a masked job fails on house-pc. Also:
  - make `gpu-precheck.sh` work without amd-smi, or reword docs/06
  - commit the deployed app and scripts back into haus-infrastructure, whose copy is about
    440 lines behind
- **Cache size column shows 0 MB** (S). `du_mb` returns 0 on any exception
  (`services/photogrammetry/bin/jobstatus:84-89` in haus-infrastructure). A symlinked cache or the
  60 s timeout is the likely cause; which one is unverified.
- **Bridge budget split in job-settings.json** (S). The host cap is logged only under planner 2 or
  `CHESHIRE_BRIDGE_LOG=1` (`hip/port/bridge_v2/planner.cpp.txt:39,53`), and the web app writes the
  file before the run. Either parse the planner line in the stats endpoint or drop
  `docs/02-memory-bridge.md:70`.

## Hardware coverage

- **Eight Windows AMD targets never run** (L). Only gfx1012, gfx1031 and gfx12-generic are
  validated (`scripts/assemble_bundle.py:22-25`). The Meshroom launcher never reads
  `gpu/<family>/UNTESTED` (`scripts/windows/meshroom-pair-launcher.cpp:154-172`), so paired runs
  give no warning.
- **RDNA2 Windows payloads for 0.3.3** (S). gfx1031 has not run on Windows since v0.2.17
  (`docs/16-bundling.md:215`). Put the RX 6750 XT back in bench-pc and run the 12-config matrix.
- **RDNA3 discrete** (S, blocked on a tester). No run on any RX 7600-7900 (`README.md:328-329`).
- **APUs and the bridge on unified memory** (M). No APU has run, and what the bridge does when host
  RAM is the VRAM is unknown (`docs/16-bundling.md:153-155`).
- **Vega** (S). gfx900/906 are untested, and ROCm 7.2 may refuse them
  (`scripts/linux/build-alicevision.sh:6-7`). The ports have only ever run wave32.
- **Matcher fallback without dot4** (S). Never run on gfx1010 or gfx900, and the forced build was
  partly recompiled back into `v_dot4` (`docs/07-gpu-matcher.md:100-103`).
- **NVIDIA beyond Pascal** (M). Every CUDA build defaults to `CHESHIRE_CUDA_ARCHS=61`
  (`scripts/windows/build-alicevision-cuda.cmd:85`), so Turing and newer depend on untested PTX
  JIT. No card of compute 7.5 or higher is available here.
- **Low-VRAM strain runs** (S). The full-resolution engine-bay `blast` on the 4 GB GTX 1050 Ti
  died before the DIMM swap and was never rerun (`docs/04-validation.md:492-496`). The 8 GB
  RX 5500 XT full-resolution texturing run (`:527-528`) was never recorded.
- **Skull turntable on the 0.3.3 package** (S). docs/04 says its failure "was not ours" because
  incrementalSfM was not paired (`docs/04-validation.md:376, :385-388`); 0.3.3 pairs it, and the
  sorted keypoints now fix the initial pair. Run skull `base` once and record the pair it draws and
  whether the paired SfM passes.
- **Rubble re-score** (M). The run was scored on the b033 bundle at 6/6 with upstream's SfM and a
  one-camera texture caveat (`docs/04-validation.md:1032, :1040-1041`). Either rerun Texturing
  (about 2.8 h) or everything from SfM down (about 15 h).
- **Offered machines** (M). Linux on bench-pc's FX-8120 (useful once ISA tiers exist) and the
  DL380 Gen10 (AVX-512, dual-socket NUMA) for SfM variation across thread counts. Neither has run.
- **house-pc upkeep** (S). The documented udev rule lacks the `bind` action
  (`docs/06-linux-build.md:235`), and `node-amd-setup.sh` does not install it. The SATA timeout
  stays a watch item.

## Packaging and platforms

- **Linux generic code objects** (M). The Linux bundle lists 19 targets by hand
  (`scripts/linux/build-alicevision.sh:14`). Collapse them to six, keep Vega explicit (wave64), and
  keep PopSIFT per chip.
- **GPU SIFT for Linux APUs, and a real fallback for Vega** (S). `scripts/linux/build-popsift.sh:18`
  omits gfx1035/1036. On Vega, PopSIFT init most likely throws instead of falling back
  (`popsift/common/debug_macros.h:156-161`).
- **PopSIFT from generic code objects** (M). A gfx12-generic PopSIFT exits 0xC0000094
  (`docs/16-bundling.md:166-189`), so a future RDNA3/4 chip likely fails. Find the host-side divide
  and add a vlfeat fallback guard meanwhile.
- **`CHESHIRE_POPSIFT` defaults OFF in the HIP build scripts** (S). See
  `scripts/build-alicevision.cmd:16` and `scripts/linux/build-alicevision.sh:25`. The same trap
  shipped a CPU-SIFT CUDA zip in v0.3.2. Auto-detect PopSIFT and check the import at packaging time.
- **x86-64 ISA tiers** (L). Both Linux bundles and the Windows CUDA package are SSE2-only
  (`scripts/linux/build-alicevision.sh:60`). Build v1/avx/v3 tiers, choose between them by CPUID,
  and re-validate: a v3 gcc build contracts FMAs.
- **Older-glibc Linux bundles** (L). Both are built on glibc 2.39 (`docs/06-linux-build.md:22`).
  Needs a jammy or bookworm container and a test node. In the same pass, pin
  `AV_BUILD_SUITESPARSE=OFF` and add a GPL check to the Linux packagers.
- **Meshroom 2025.x** (M). Pairing, shims and gates target 2023.3 only; 2025.1.0 is declared
  untested (`docs/upstream/meshroom-issue-comments.md:20`).
- **`CHESHIRE_BACKEND=auto` on NVIDIA boxes** (S). `auto` returns every node to Meshroom whenever an
  NVIDIA card is present (`scripts/windows/meshroom-pair-launcher.cpp:131-135`). Record the
  package's backend at pair time and follow it. On Linux, `scripts/linux/meshroom-pair.sh` decides
  the backend at `:63-68` but sources the bundle's `env.sh` only at `:80`, so a `CHESHIRE_BACKEND`
  set there never takes effect: source it first, or document that it cannot.
- **Linux stage gate** (M). The stage gates need `cheshire-run.cmd` or PowerShell
  (`scripts/verify_bundle_stages.py:32-34`), so no Linux artifact gets per-stage checks.
- **Resume cannot detect a corrupt cached input** (M). SUCCESS chunks are kept without being
  decoded (`scripts/verify_end_to_end.py:238-255`), and one damaged EXR went through Texturing
  three times.
- **End-to-end gate gaps** (M). The gaps:
  - `cpufallback` never forces CPU SIFT
  - PrepareDenseScene is not in `BINARY`/`GPU_MARKERS` (`scripts/verify_end_to_end.py:114-123`)
  - no check that SfM's "edges skipped" warning is absent
  - a never-paired node is caught only at the end of the run
- **verify_packages.py stops at v0.2.16** (S). It has five markers
  (`scripts/verify_packages.py:21-32`). Revive it with one literal per step since 4d, or retire it.
- **Patch export** (S). Seven edited upstream files are missing from `TRACKED`
  (`scripts/apply_hip_patch.py:43-88`), and the port sources outside `depthMap/cuda/hip` are
  missing from `patches/0002`.
- **Guard against the two patch hazards** (S). A scan found no live case of either (a statement
  inserted after an unbraced `if`, or a ternary passed to a log macro). Add a check that fails the
  build if either comes back.
- **Defender real-time scanning of `build/`** (S, the user's call). Two release-build steps
  failed on scanner locks (docs/04, "0.3.3 release packages"); build_targets.py now retries. An
  exclusion is a machine security setting only the user can make, so it stays optional.
- **CHOLMOD at thousands of poses** (M). There was no penalty at 884 views (304 s vs 323 s,
  `docs/04-validation.md:1049-1054`). Replay Rubble's SfM against both builds. The result is
  informational only, because CHOLMOD cannot ship.

## Upstream reports to file

### Ready to file or open

- **ROCm: layered surfaces and the false 2048 limit** (S). The layered surface calls address a mip
  level instead of an array layer (`docs/14-gpu-sift.md:64-67, :105-106`). The reproducer is
  `hip/port/popsift/bugreport_layered_surface.hip`, and the earlier draft is lost. Rerun it on the
  RX 6750 XT and file only with the user's go-ahead.
- **ROCm: other defects** (M). The fp16 `surf2Dwrite` drop (`hip/tests/surf_probe.hip`) can go in
  the same report. The `__fmul_rn`/`__fadd_rn` contraction first needs a reproducer
  (`docs/11-meshing-cpu.md:281-285`). So does the 0xC0000005 exit when `amdhip64` loads a runtime
  the driver does not support in a non-interactive session, such as a scheduled task
  (`scripts/windows/cheshire-detect.cpp:10-12`).
- **ROCm/clr#285 label** (S). The second-card comment says gfx1010, but the card is gfx1012
  (`docs/upstream/rocm-issue-second-card-comment.md:3`).
- **PopSIFT** (S). Open the PR offered on popsift#193. Also report the `LinearTexture` type error
  (`scripts/apply_popsift_patch.py:42-54`).
- **AliceVision races and waste** (M). None is drafted:
  - filterByPixSize: survivors depend on thread timing (4p)
  - addGridHelperPoints: shared RNG across threads (4q)
  - the kd-tree reads vertices that other threads are moving (4l)
  - about 50 M discarded slots are left in the kd-tree
  - NaN residuals break `std::sort` in AC-RANSAC (4w)
- **AliceVision: silent CPU SIFT fallback** (S). No warning is printed when the GPU is denied
  (`feature/sift/ImageDescriber_SIFT.hpp:35`). Add a local warning step and file the same change
  upstream.
- **AliceVision: DepthMapFilter vote buffer** (S). Votes accumulate across neighbours
  (`hip/port/gpu_filter/depthMapFilterGPU.cu:263-270`). Frame it as a question, not a bug.
- **AliceVision: bundle-target PR** (S). Issue #2182 was filed on 2026-09-20, which a live GitHub
  search found. The drafted PR is not opened (`docs/upstream/alicevision-bundle-libs-paths.md`).
- **OpenMesh `/bigobj`** (S). Exported without a language guard
  (`docs/upstream/openmesh-bigobj-language-guard.md`).

### Waiting on maintainers

- AliceVision PR #2179 and PR #2181 have no reviews yet. Keep the fork until both close.
- The #2344 SfM fix was posted with a PR offer (`docs/drafts/meshroom-2344-comment.md:93`), and
  nobody has replied.
- The HIP backend and planner PRs (L) are blocked on discussion #2175 or #2116. The planner still
  depends on `cheshire::bridge` (`scripts/apply_hip_patch.py:228-240`).

### Corrections to posts already made

- Show-and-tell #2183: the "without checks" times for False Door (320 s, 5022 s) came from the
  earlier workaround run (`docs/drafts/alicevision-show-and-tell.md:53-54`; this run's Meshing
  without checks is 479-489 s), and the exactness row labelled "depth-map filter vote pass" is
  Meshing's filterByPixSize check (`:68`). Public edit: needs the user's OK.
- The #2116 reply (as corrected earlier today) says every CPU-stage port except the matcher has a
  self-check; the DepthMapFilter port has none either (`hip/port/gpu_filter/` has no check mode;
  `CHESHIRE_FILTER_CHECK` belongs to Meshing's filter). Public edit: needs the user's OK. Its
  "seven binaries" was true when posted.

### Outreach (the user's posts)

- r/photogrammetry, r/ROCm, r/LocalLLaMA, Level1Techs and Phoronix. Only r/LocalLLaMA has a draft
  (`docs/upstream/reddit-localllama.md`), and it needs a fact-check first.

## Parked and exploratory

### Parked from the 0.3.4 group

- **Linux capped-run 110-pixel difference** (M). One view differs by 1e-6 on Linux only
  (`docs/02-memory-bridge.md:306-310`). A few hours of triage would help: repeat the uncapped run
  as a run-to-run control, which Linux has never had.
- **geogram cell numbering** (M). Identical input produces different cells on every run (c23c1ae).
  One untested lead: `GEO::initialize()` runs before the single-thread switch
  (`fuseCut/Tetrahedralization.cpp:25, :36`). A report to geogram waits on that lead.
- **Fixed-point meshing votes** (M). The design is ready. The costs are 1.2 GB against 694 MB, and
  it gives no reproducible mesh while geogram renumbers cells.
- **Parallel Delaunay** (S-M, unverified). geogram 1.9.3 ships PDEL. Nobody has checked whether our
  builds enable it or whether it gives the same cell set.
- **Fusion re-decode** (S). Filtered maps are decoded three times inside Meshing, but only a few
  seconds are at stake.
- **Direct OBJ writer scope** (M). It handles OBJ only, without normal, bump or displacement maps
  (`scripts/apply_hip_patch.py:2908-2922`). Measuring the engine-bay save once would be S.

### DepthMap

- **Large-set per-pixel slowdown** (L). About 3x slower per pixel beyond the pixel ratio
  (`docs/04-validation.md:913-914`). Profile before any kernel work.
- **rocprofv3** (M, optional). Never run. Its stated reason (a gap to the 1080 Ti) no longer holds.
- **Texture-window caching and float-accumulator SGM streams** (L each). Both change the output,
  and the GPU is already saturated (`docs/05-performance.md:67`).
- **Hand-built hardware mip chain** (L). Worth 4-9 %, at the price of per-generation descriptor
  layouts.
- **Images to host below one tile** (M). Never tried at a 1 GB cap on Linux, and it keeps byte
  identity (`docs/02-memory-bridge.md:170`).
- **Bridge under genuine pressure, and vs unified memory** (M). No real data set has spilled so far
  (`docs/18-cuda-build.md:295-317`).
- **Re-tune defaults on large sets** (M). The max-flow constants were tuned on the engine bay only
  (`hip/port/gpu_maxflow/maxflowGPU.cu:307-317`).

### Extraction and matching

- **GPU DSP-SIFT** (XL). A cheaper option (S) is a GPU sift preset, plus a line in USING.md.
- **High keypoint caps** (S). Run 50k, 100k and 200k once now that the extrema counter is fixed.
- **GPU matcher throughput** (L). Register tiling is "a project, not a patch"
  (`docs/07-gpu-matcher.md:125-129`). Smaller steps first: run the shelved sliced variant
  (`CHESHIRE_MATCHER_SLICED=1`) on RDNA1/2, and batch per search instead of per pair.

### Other

- **Failed EXR write segfaults** (M). Reproduce it and add a free-space preflight
  (`docs/04-validation.md:940-945`).
- **HIP RDC** (S). Never built anywhere, and blocked on Windows.
- **ROCm 6.4 fallback bundle**. Only needed if the gfx1031/1032 SIGSEGV actually shows up.
- **Segmentation on AMD** (L). Only the CUDA provider is wired up (`segmentation.cpp:58-61`).
- **Intel spike on garage-pc** (L). Stays parked. Check the toolchain route first, since the box
  runs Windows.
- **CUDA Linux bundle under WSL2 on NVIDIA**. Untried and not ruled out.

## Not planned

- Automatic tile-size planner: it breaks byte identity across caps
  (`docs/04-validation.md:353-354`).
- WSL2 as a runtime: it has no textures (`docs/06-linux-build.md:30`).
- The rejected DepthMap tuning and chunk-level parallelism (`docs/05-performance.md:63-75, :84-85`).
- The 25-descriptor PopSIFT difference between runtimes (`docs/14-gpu-sift.md:356-362`).
- AC-RANSAC on the GPU: iterations are not independent (`docs/15-acransac-cpu.md:226-238`).
- The NFA tail-bound pruning, a recorded dead end (`docs/15-acransac-cpu.md:207-224`).
- QR for the over-determined branch (never reached) and the spherical solver (no fisheye data).
- Reporting FeatureMatching's doubled count lines: they are pre- and post-grid-filter, not a bug.
- Replacing geogram (XL).
- Float-atomic votes and max-flow flow totals: the labelling is the verdict.
- Bucketing visibility votes by owner: 17 % slower, reverted.
- DepthMapFilter's second pass on the GPU: it is a cheap threshold pass.
- Global SfM: it changes the geometry. (SfM on the GPU is no longer here: see 0.3.5, "Bundle
  adjustment on the device".)
- ImageMatching and MeshFiltering speed: 37 s of a 16 h run.
- Masking or a turntable-aware pipeline.
- Vulkan compute and ZLUDA: HIP proved sufficient.
- GCN4 / Polaris: ROCm 7.2 refuses the RX 570.
- Generic RDNA1/2 code objects on Windows: code-object v6 is not production-ready.
- RX 7000/9000 on pre-AVX2 Windows CPUs. Revisit only together with the ISA tiers.

## Stale text to fix

### README.md

- `:36`, `:373`: "What is left is the tetrahedralisation" and "every CPU change verified
  identical" are no longer accurate (4u changes faces on purpose).
- `:37`, `:382-384`: padding, save and downscale have all shipped, and the downscale is exact.
- `:38`: the 0.3.3 undistortion map is missing.
- `:42-43`, `:300-302`, `:474`: says seven paired nodes, and lists SfM as not carried.
- `:44-47`: the 39 → 30 min figure was measured at v0.2.5 (`docs/07-gpu-matcher.md:77`).
- `:189`: RDNA2 on Linux also lacks `hipMallocMipmappedArray`.
- `:238`, `:417`: future chips get no GPU SIFT.
- `:240`: the Vega/APU "fallback" claim is unverified.
- `:260`: the launcher warning claim is wrong - the launcher never reads `gpu/<family>/UNTESTED`.
  The same claim is in `scripts/assemble_bundle.py:10-11` and docs/16. Either fix the text or make
  the launcher read the marker, which makes every claim true.
- `:326-339`: points to the retired RDNA3+RDNA4 zip, uses a broken command, and "every package"
  does not hold for CUDA.
- `:357-358`: the cache shipped in v0.2.9.
- `:403-412`: QR has been opt-in since v0.2.18.
- `:421-423`: the collapse is to six targets, not four.
- `:425-429`: the CUDA work has shipped; only the older-glibc bundle remains.
- `:434-436`: SfM also blocks reproducibility, not only geogram.
- `:452-453`: Building covers RDNA3/4 only.
- `:474-494`: Status still reports the v0.3.0 gates, and cb2e496 is not mentioned.

### USING.md

- `:149`: "same poses" does not hold at 884 views.
- `:158-159`: the 13x image-spill figure is out of date, and `PLANNER=2` does not send images to
  host.
- `:167-168`: `verify` covers the Meshing checks only.
- `:201`: `=0` turns QR on.
- `:216`: lists 7 of the 12 configs.
- Missing: `CHESHIRE_GPU_RESIZE`, `CHESHIRE_GPU_RESIZE_CHECK`, `CHESHIRE_UNDISTORT_MAP`,
  `CHESHIRE_UNDISTORT_MAP_MB`, `CHESHIRE_PDS_PROFILE`, `CHESHIRE_OBJ_ASSIMP`,
  `CHESHIRE_INVERT_OLD`, and a note that WSL2 is not a supported runtime.

### docs/

- `docs/00-scope-and-findings.md:22` and `docs/03-port-status.md:21-23`: the option is
  `ALICEVISION_USE_HIP`. `docs/00:29-31`: DirectML segmentation still needs a port.
- `docs/01-toolchain-windows.md:79-81`: OpenMP comes from `third_party/llvm-openmp`. `:91-96`:
  superseded by `:98`.
- `docs/02-memory-bridge.md:75-76`: the bridge is not HIP-only. `:221-222`, `:326`: record that
  auto tile sizing was rejected. `:287-288` contradicts `docs/18-cuda-build.md:307-308`.
- `docs/04-validation.md`:
  - `:82-84`: "not yet profiled" is out of date.
  - `:172`: "CUDA-only" is present tense.
  - `:349-356`, `:509-510`: the sort did not make pipelines reproducible.
  - `:390`, `:563-565`: the SfM RNG is seeded but shared across threads.
  - `:441-443`: overstates the 8 GB risk.
  - `:515`: the label is wrong.
  - `:541-542`, `:553-556`: name what fusion work is actually left, and the iteration cap used.
  - `:559-561`: says Ceres links CHOLMOD.
  - `:647-649`, `:653`, `:694`, `:973`, `:1103`: the image peaks are half the true size.
  - `:680`, `:1152`: fix after the knn pragma run.
  - `:745-746`: omits the write phase.
  - `:763-765`: says the item is queued behind work that has shipped.
  - `:941-943`: crash attribution; "124 GB" was the whole cache.
  - `:944`: resume also clears ERROR, KILLED and STOPPED.
  - `:973-974`: raw log text.
  - `:1038-1039`: the CPU share omits PrepareDenseScene (the published v0.3.3 release notes,
    `docs/releases/0.3.3.md:53-54`, repeat it; a public edit, needs the user's OK). Also record
    the 30 GB Rubble trim.
  - `:376`, `:385-388`: the skull entry says incrementalSfM is not paired; 0.3.3 pairs it.
- `docs/05-performance.md:13-15`, `:33-37`, `:79`: the 1080 Ti gap was based on the 105.5 s
  figure. `:45-46`: name `CHESHIRE_SGM_LEGACY`.
- `docs/06-linux-build.md`:
  - `:53-55`, `:95-96`: the node does not use `gpu-precheck.sh`.
  - `:61-62`: RDC is untested, not "expected to work".
  - `:91-94`, `:132-133`: Meshroom 2023.3 does drive these binaries.
  - `:119`, `:126`, `:128-129`: the card is an RX 6750 XT, and gfx1031 was clean on 7.2.
  - `:166-170`: say Option 3 is parked.
  - `:213`: the zip is retired.
  - `:235`: the rule needs `bind`.
- `docs/07-gpu-matcher.md:111-112`: NVIDIA has had the GPU matcher since v0.3.0.
- `docs/09-gpu-meshing-votes.md:63-65` and `docs/11-meshing-cpu.md:51-56`: these shipped in
  v0.2.10 and v0.2.11.
- `docs/10-gpu-texturing.md:31-33`, `:52-53`: out of date after 0.3.2 and 0.3.3.
- `docs/11-meshing-cpu.md:149-156`: reproducibility claims. `:242`: `FUSION_PROFILE` was never
  committed.
- `docs/13-gpu-visibilities.md:60-63`, `:84-88`: says fusion is racy and left to do. `:66-68`: fix
  after the pragma run.
- `docs/14-gpu-sift.md`:
  - `:345-348`: says extraction order varies.
  - `:445-447`: name the two log blocks.
  - `:467-480`: RDNA1/2 have run, and the CUDA build exists.
  - `:474`: the AliceVision version disagrees with `docs/18-cuda-build.md:70, :130` and
    `docs/04-validation.md:48`.
- `docs/15-acransac-cpu.md:147-153`, `:168-171`: the RNG explanation. `:245-256`: point to docs/17.
- `docs/16-bundling.md:89-96`: generic PopSIFT does not work. `:146`: the gfx1010 matcher path
  differs.
- `docs/17-svd-nullspace.md`:
  - `:1`, `:8-9`: QR shipped in v0.2.18, not 0.3.0.
  - `:91`, `:159`: `CHESHIRE_SVD_NULLSPACE` does not exist.
  - `:93-96`: likely the 2048 vs 50000 iteration cap; confirm with one rerun.
- `docs/18-cuda-build.md:269-275`: CUDA camera images are counted since 5f. `:376`: filed as #2182.
  `:489-491`: settled by 6714ed7. `:517-519`: verified.
- `docs/upstream/alicevision-bundle-libs-paths.md:4`: filed as #2182.
  `rocm-issue-host-atomics.md:1, :14`: still labelled a draft, and the AtomicOps claim is wrong.
  `alicevision-discussion.md:13`: old title.

### Code comments

- `scripts/apply_hip_patch.py`:
  - `:113-115`: step 1a comment still describes float4.
  - `:263`: the Image class is no longer empty on CUDA.
  - `:445`: filed as #2182.
  - `:1695`: says score order; the code uses index order.
  - `:1839-1840`: says the RNG reset fixes insertion order.
  - `:2329`: `FUSION_PROFILE` does not exist.
  - `:3412-3413`: says every port file has the pragma.
- `scripts/apply_popsift_patch.py:56-59`, `:80-82`: the rationale is from before the fix.
- `scripts/verify_end_to_end.py:21`, `:33-42`, `:60-63`, `:161-163`, `:181-182`, `:185`,
  `:296-297`, `:390-394`.
- `scripts/linux/meshroom-pair.sh:4-14` and `scripts/windows/meshroom-pair.cmd:4-11`: say seven
  nodes and name the old switch.
- `hip/compat/include/cheshire/bridge.h:8`: should cite step 1j. `:15-16`, `:22-23`, `:144` and
  `hip/port/bridge_v2/planner.cpp.txt:2`: the 13x figure. `bridge.h:229`: the AtomicOps claim.
- `hip/port/gpu_knn/knnGPU.cu:10`, `hip/port/gpu_blur/simBlurGPU.hpp:9`,
  `scripts/linux/build-alicevision.sh:8-9`, `scripts/verify_bundle_stages.py:143`.
- `scripts/windows/build-ceres-nosuitesparse.cmd:2-3` and `scripts/windows/package-cuda.ps1:150`:
  control characters where backslashes were lost.

### Small decisions that ride the doc pass

- Keep or remove `CHESHIRE_TEXTURE_FLOAT4`, `CHESHIRE_SGM_LEGACY` and `CHESHIRE_MATCHER_NO_DOT4`,
  and document the ones kept (S).

### Published release pages

- GitHub v0.2.12-v0.3.2 still have hard wraps. Forgejo 102 and 103 still make the wrong gfx1012
  claim, and 100, 101, 102 and 105 need reflowing. These are public edits, so ask the user first.
