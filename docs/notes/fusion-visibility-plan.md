# Visibility passes on the device: verdict and implementation plan

> **Status, 2026-09-24.** Written 2026-09-22 from a three-design judge panel; kept here as the
> working plan. Done: steps 0-3 (5d6045b, 7213bfa, ebc33f0) and, as an intermediate not in the plan,
> step 6k (the backprojection on the device ahead of the unchanged knn, host votes kept; docs/04).
> Step 1's premise was incomplete: the pragma does not reach HIP's `__dadd_rn`/`__dmul_rn`, which are
> defined in a header included before the file, so they fused anyway; 6k replaced them. Open: steps
> 4-12 (device votes and the rest), and the house-pc measurements the plan asks for.

## 1. Verdict

All three designs reach the same floor, the unchanged knn kernel: 38.7 s in pass 1 and 42.9 s in pass 2 on the RX 9070, 46.5 ms per camera. Their gain estimates therefore agree within error, about 40-46 s on the 5600X. What separates them is exactness and risk. Scores run from 1 (poor) to 5 (best).

| | **A**: per-camera device pipeline, existing knn kernel kept | **B**: VisPass (fused kernel, batches of up to 32 cameras, lists and pixSize built on the device) | **C**: fused kernel, bands, bridge ladder, runtime form calibration |
|---|---|---|---|
| Exactness and how it is checked | **4**. Full-output check against the host path plus an always-on probe. Its reference, though, is fed the device's own knn answers. | **4**. Same reference, but more device surface to prove: lists, per-vertex pixSize, p recomputed in the fold. | **5**. The reference votes are fed by host nanoflann, so they do not depend on the device. Adds per-pixel vote comparison and a cross-run digest. |
| Gain, 5600X / house-pc | 4 / 4 | 4 / 4 | 4 / 4 |
| Memory safety on 4-8 GB cards | **4**. About 1.1 GB; a spill sends the pass to the old path. No bands, so no protection against the Windows 2 s watchdog (TDR). | **3**. Up to about 4.6 GB by design, and the fragment pool grows with the data. | **5**. About 0.94 GB, bands, a launch-time cap, atomics verified to be in VRAM. The ladder rungs are more than the realistic point counts need. |
| Implementation risk and size | **4**. About 1.2k lines. The proven kernel is untouched; one new dependency (hipCUB/CUB). | **2**. Batches, two compute streams, a fragment pool, a fused kernel, a device camera-list table. | **2**. Deep-stack re-walk kernel, mapped pinned ring, planner ladder, form selection by sample, host-mode K1. |

**Design A is the spine.** Grafted onto it:
- **From C:** the independent reference (host nanoflann feeding the ordered reference votes), the per-pass state digest, row bands with a cap on launch time, the rule that anything touched by atomics or CUB must be in VRAM, and the fold probe.
- **From B:** a larger reader pool and the decoded depth-map cache as a follow-up.
- **New in this plan:** restructure the host votes first. It is exact, small, and it is the fallback and reference path anyway. The arithmetic in section 8 shows it captures most of the 5600X gain on its own.

Rejected:
- **B's device-built lists and device pixSize:** more surface to prove, and no gain.
- **C's partial-spill rungs:** never reached at downscale 2, and spilled atomics are unsafe.
- **C's deep-stack re-walk:** a tree-depth guard makes overflow impossible.
- **C's mapped ring:** coarse-grained visibility risk on AMD.
- **Fusing backprojection into the knn kernel:** saves under 1 ms per camera and risks register pressure in the walk. Revisit only after measuring.

## 2. What the numbers say

Pass 1 per camera, measured:
- **Host: about 82 ms.** Maps 5.6, backproject 28.1, votes 46.1.
- **Device: 62 ms.** Upload 10.3, kernel 46.5, download 5.1.

The votes figure is the anomaly, and it comes from the port's own structure:
- Every OpenMP thread scans all 3.76M answers of the camera (`visibilitiesGPU.inc:337-341`).
- One camera's votes probably land in a narrow vertex-index range, because vertex order follows the camera that generated each vertex. So a few threads do most of the work.

Ordering the fixes:
1. **Restructure the host votes.** Stable bucketing by vertex, buckets processed in parallel. This makes the pass device-bound at 62 ms per camera.
2. **Move the votes to the device.** This removes 15.4 ms per camera of transfers (112 GB per pass) and the host backprojection. The pass then sits at the kernel floor of about 53 ms.
3. **Tune the knn kernel.** After step 2 it is the whole floor.

On a 4-thread host, backprojection (about 84 ms per camera) plus decoding keep the pass host-bound even after fix 1. That is where fix 2 pays most.

## 3. Chosen architecture

**The reference.** Upstream is timing-dependent:
- 3 cameras run at once with nested row threads.
- The tree reads live coordinates through `Kdtree.hpp:21,37` while other threads move vertices.

So the definition to reproduce is the 0.3.3 port's semantics:
- upstream's per-pixel arithmetic;
- one snapshot tree per pass (`visibilitiesGPU.inc:140-143`);
- each vertex's contributions applied in (camera ascending, pixel row-major) order;
- after each pass, each camera list is the incoming list followed by the new cameras in ascending order.

These semantics are deterministic run to run (`docs/11-meshing-cpu.md:149-152`).

**Per pass (host, then upload).**
1. Build the snapshot tree, flatten it and upload it as today (`visibilitiesGPU.inc:140-205`). New: record the tree's maximum inner depth.
2. Compute scoreV with the existing expression (`:320-322`).
3. Upload the per-vertex state:
   - `live` (double3): a device-to-device copy from `Index::_points`, which already holds the snapshot;
   - `nrc` (int32): gathered from `va.nrc`, so pass 2 continues pass 1;
   - `sim` (float) and `scoreV` (float);
   - a camera bitmap of N/8 bytes.
4. Save the list sizes at pass start (4 B per vertex) for recovery.

**Per camera c (loop position c, as upstream uses it), on one device stream, for each row band b in ascending order:**

| Kernel | Work |
|---|---|
| H2D | The band's float depth rows plus `rowStart` (the host counted valid pixels with `!(depth <= 0.0f)` while copying into pinned staging). |
| `K_bp<FMA>` | One block per row, in-row prefix, k = rowStart[y] + prefix (the host's row-major compaction). Writes `q[k] = C + normalize(iCam*(x,y))*depth` and `pixSize[k] = getCamPixelSize(q,c)`, replayed operation by operation with `_rn` intrinsics. |
| `knnKernel<FMA>` | Unchanged, through a new `Index::queryResident(nq, fma)`: no query upload, no answer download; the memset and the 4-byte overflow download stay. |
| `K_decide` | For `v = nn[k]`: skip 0xFFFFFFFF. `I = __double2float_rn(((double)sim[v]*ps)*ps)`, `m = (I < scoreV[v]) ? scoreV[v] : I`, `vote = dist < (double)__fmul_rn(vm, m)`, `contrib = vote && dist < (double)__fmul_rn(cm, scoreV[v])`. A vote does `atomicOr` on bit v of the bitmap. Writes `key[k] = contrib ? v : N`, `val[k] = k`. Vote and contribution counts are block-reduced. |
| Sort | hipCUB/CUB `DeviceRadixSort::SortPairs` over the band on bitlen(N) bits. It is stable. |
| `K_fold` | One thread per run head (`key != N`, first of its run). In order: `x = __ddiv_rn(__dadd_rn(__dmul_rn(x,(double)n), q.x), (double)(n+1))` per component, then `n += 1`. |
| After the last band | D2H of the bitmap and counters into pinned slot c%2, record an event, clear the bitmap. |

**Host loop.** For each c:
1. `stage(c)`: take the decoded map, run a parallel copy plus row count into pinned slot c%2. An empty map is skipped with upstream's warning.
2. `enqueue(c)`.
3. `wait(event c-1)`.
4. For each set bit v in c-1's bitmap, call `verticesAttrPrepare[v].cams.push_back_distinct(c-1)`, parallel over bitmap words. Each vertex occurs once per camera, so this is race-free.

At pass end:
- D2H `live` into `verticesCoordsPrepare` and `nrc` into `va.nrc`;
- run the unchanged host `getCamsMinPixelSize` loop (`:445-451`).

The coordinates and nrc on the host are untouched until this commit.

**Recovery.** A device error, a nonzero overflow, or the fault-injection knob triggers the same response:
1. Truncate every list to its saved pass-start size (`StaticVector::resize`, `StaticVector.hpp:68`).
2. Rerun the whole pass on the host-votes path.

This is exact, and costs one pass of time.

**Refusals: the pass goes to host votes, and the log names the reason.**
- Tree depth greater than 96 (`knnGPU.cu:23`).
- Any new buffer spilled (`bridge::tierOf`, `bridge.h:304`).
- The plan does not fit the budget (`bridge::budget()`, `bridge.h:329`).
- The first-camera probe mismatches.

**What moves and what stays.**

| Device | Host |
|---|---|
| Backprojection and query pixSize | Depth EXR decode on R readers, R = max(3, hw-2) |
| knn (existing kernel) | Pinned copy plus row count per camera |
| Vote and contribute decisions | Tree build, flatten, upload; scoreV |
| Voted-camera bitmap | Appending cameras from the bitmap (push_back_distinct) |
| Stable sort and ordered running-mean fold | Commit, then getCamsMinPixelSize (unchanged) |
| Live coordinates and nrc for the whole pass | Probe comparisons, recovery, the CHECK reference |

Everything outside the two passes stays on the CPU and unchanged:
- load loop;
- filterByPixSize;
- removeInvalidPoints;
- angle filter (acos cannot be replayed on the device).

## 4. How order and rounding are preserved

**Order.**
- Cameras run in ascending loop position on one in-order stream, bands in ascending rows, and k ascending within each band.
- The sort is stable over input already in k order, so each vertex's contributions stay in pixel order.
- One thread folds each run, and bands and cameras are serialized by the stream. Each vertex therefore sees its contributions in exactly (camera, pixel) order.
- Camera lists are appended camera by camera, in ascending order, through `push_back_distinct` itself. This holds even when pass 1 fell back to upstream and its lists are unsorted, so no sortedness is assumed.
- Every decision depends only on pass-start data (nn, dist, pixSize, sim[v], scoreV[v]), so evaluating decisions in parallel is exact.

**Rounding.**
- **Pragma everywhere.** Every new translation unit carries `#pragma clang fp contract(off)`. On HIP, `__dadd_rn`, `__dmul_rn`, `__ddiv_rn`, `__dsqrt_rn` and `__fmul_rn` are plain operators (`__clang_hip_math.h:1190,1212,1234,1262,271`), and HIP contracts device code by default. CUDA gets `--fmad=false` from step 5l, because the new step sits before `apply_hip_patch.py:3412`.
- **Fused forms, from step 0's disassembly.** `K_bp` uses `template<bool FMA>` chosen from `__FMA__`, overridable with `CHESHIRE_GPU_VIS_FMA=0|1`. Expected forms under clang-cl `/arch:AVX2`:
  - `iCam*pix` row: `fma(m1,x,m2*y)+m3`;
  - `camArr*X` row: `fma(m3,z,fma(m1,x,m2*y))+m4`;
  - `|v|^2`: `fma(z,z,fma(x,x,y*y))`;
  - cross product: `fma(a_i,b_j,-(a_j*b_i))`.
- **Always unfused:** C + n·depth, C − p, pix.x + 1.0, and all divisions.
- **Correctly rounded:** sqrt and division on both backends.
- **Hosts that cannot fuse:** the MSVC CUDA host and the generic Linux gcc build use the plain form.
- **Decision expressions:** upstream's float/double mix exactly (`PointCloud.cpp:217-221, 236`). They contain no additions, so contraction cannot touch them.
- **Fold:** three separate operations per component, unfused, as Point3d's separate operators (`Point3d.hpp:95-99`).
- **Proof layers:** the always-on probe on camera 1 (q, pixSize, plus 4096 fold triples); CHECK on every pixel and every vertex; the digest and the tetrahedralization input checksum across runs.

## 5. CHECK mode and the gate

`CHESHIRE_GPU_VIS_CHECK=1` is already in `SELF_CHECK_ENV` (`verify_end_to_end.py:82-85`). This plan extends it rather than adding a new switch. In the same process, per pass:

**Setup.** Shadow copies of the pass-start `verticesCoordsPrepare` and `verticesAttrPrepare` (deep copy, including the cams lists).

**Per camera and band:**
- the device's q, pixSize, nn and dist are downloaded into today's pinned slots;
- the host runs `prepare()` (mp.backproject, mp.getCamPixelSize) and `hostSearch` on the host's q;
- the verbatim ordered votes (today's `votes()`, kept as `cheshireVotesOrdered`) are applied to the shadows using the host's answers only.

The reference therefore never sees device output.

**Per pass**, every vertex is compared:
- coordinates, 24-byte memcmp;
- nrc;
- cams length and each element in order, with a separate "order only" count;
- pixSize float bits;
- total votes and contributions.

**Verdict lines.** A per-process call counter supplies the pass tag. The path name in the tag stops a silent fallback from passing the gate.
```
cheshire: GPU backprojection check (pass 1): identical to MultiViewParams on all Q queries
cheshire: GPU backprojection check (pass 1): N of Q queries differ (q A, pixSize B)          [WARNING]
cheshire: visibility votes check (pass 1, GPU votes): identical to the ordered host reference on all N vertices (coordinates, nrc, camera lists in order, pixSize); Q queries, V votes, K contributions
cheshire: visibility votes check (pass 1, GPU votes): X of N vertices differ: coordinates a, nrc b, cams c (order only d), pixSize e; votes V vs V', contributions K vs K'   [WARNING]
```
The existing `GPU knn check: identical to nanoflann on all` line is unchanged.

**Gate (step 10).** `SELF_CHECK_VERDICTS["Meshing"]` gains four regexes, one per line and per pass, because a single regex would be satisfied by pass 1 alone:
- `r"GPU backprojection check \(pass [1]\): identical to MultiViewParams on all"`
- the same with `\(pass 2\)`
- `r"visibility votes check \(pass 1, GPU votes\): identical to the ordered host reference on all"`
- the same with `pass 2`

Other gate entries:
- `GPU_MARKERS`: `"visibility votes on the GPU"`
- `CPU_MARKERS`: `"visibility votes: disabled by CHESHIRE_GPU_VIS_VOTES=0"`
- the `cpufallback` env gets `CHESHIRE_GPU_VIS_VOTES=0`
- `verify_packages.py`: `b'CHESHIRE_GPU_VIS_VOTES'` in aliceVision_fuseCut
- a new `visband` config: CHECK plus `CHESHIRE_GPU_VIS_BAND_PIXELS=65536`, so order preservation across bands stays proven.

**Out of process.** Under `CHESHIRE_GPU_VIS_LOG`, each pass prints an FNV-1a-64 digest of (coordinate bits, nrc, cams in order, pixSize bits) on every path. It must match between `VOTES=0` and the default, across bridge caps, and across band sizes. So must the tetrahedralization input checksum (False Door: `64b36ee445e30d38`, `meshing.log:1801`), the ray count (62,394,383, `:1805`) and the number of points removed by the angle filter (225,360, `:925`).

**Announce lines.** A `call_once` at dispatcher entry prints one of these on every path, including CHESHIRE_GPU_VIS=0:
- `visibility votes on the GPU (CHESHIRE_GPU_VIS_VOTES=0 for host votes)`
- `visibility votes: disabled by CHESHIRE_GPU_VIS_VOTES=0, host votes`
- `visibility votes: <reason>, host votes` (WARNING)

## 6. Memory plan

**Per point, resident for the pass: 75.3 B.**
- Index points 24, nodes 11.2 (3,478,063 × 32 B per 9.94M points, `meshing.log:87`), perm 4.
- live 24, nrc 4, sim 4, scoreV 4, bitmap 0.125.

**Per band pixel: 64 B.** depth 4, q 24, pixSize 8, nn 4, dist 8, key/val with sort alternates 16. CUB temp storage up to about 16 MB.

**False Door totals.**
- Pass 1: 748 MB + 324 MB (one whole 3000x1688 camera) ≈ **1.09 GB**.
- Pass 2: ≈ **0.69 GB**.
- Pinned host memory: 2 × (20.3 + 1.24) MB = 43 MB (today: 365 MB).

**Planner at pass start.**
- Read `bridge::budget()`.
- If 75.3·N + 64·minBand + temp exceeds the cap, use host votes. minBand = max(w, 262,144) pixels.
- Otherwise bandPixels = min(maxPix, room/64, launch cap). The launch cap is 250 ms divided by the knn ns per query measured on the first band.
- Allocate everything through `cheshire::devMalloc`: atomic targets and CUB temp first, then the rest.
- Verify every block with `tierOf == Vram`. Any spill releases the buffers and sends the pass to host votes. No partial-spill rungs.

| Card (bridge cap) | False Door pass 1 | Maximum N before host votes | Bands |
|---|---|---|---|
| 16 GB RX 9070 (about 13 GB) | 1.09 GB, 8 % | about 170M | 1 per camera |
| 12 GB RX 6750 XT (about 10 GB) | same | about 130M | 1 |
| 8 GB RX 5500 XT (about 6.5 GB) | same | about 85M | about 3, set by the launch cap (whole-camera knn launch there is about 0.6 s) |
| 4 GB GTX 1050 Ti (2417 MB measured, `docs/18-cuda-build.md:445`) | same, 45 % | about 31M | set by the launch cap |
| 2 GB | same | about 15-18M | set by the launch cap |

At downscale 2, pass 1 is bounded at about 12.5M points, so every card stays on the device path. At downscale 1 (up to about 50M points), a 4 GB card takes host votes. That is today's behaviour or better.

**Host RAM.** Unchanged in production. CHECK adds 0.3-1.5 GB of shadows plus the two 182 MB pinned slots. house-pc has 14.6 GB, and Rubble's Meshing peaked at 8 GB (`docs/04-validation.md:1036`), so run CHECK there at 41 views only.

## 7. Steps

Each step is validated on mini6 before moving on, then on 41 views, then on the 833-view False Door cache (`build/e2e-falsedoor-fix/base/cache`). "Digest" means the per-pass digest from step 2 plus the tetrahedralization input checksum.

**Step 0. Measure and prepare (no production change).**
- Generate the 41-view filtered maps once with the 0.3.3 `aliceVision_depthMapFiltering` on `data/ref/monstree-full`, using Meshroom 2023.3 defaults and the CLI shape at `verify_bundle_stages.py:93-95`. Keep them in `build/ref/monstree-41-dmf`, or copy house-pc's job.
- Iterate in `build/av-gfx1201-popsift`:
  - set `CHESHIRE_BUILD_SUFFIX=-popsift`, `CHESHIRE_POPSIFT=ON`, `CHESHIRE_HIP_ARCHS=gfx12-generic`;
  - confirm that a configure is a no-op;
  - do not use `build/av-gfx1201-install`, which is stale.
- On an idle 5600X, record mini6, 41 and 833 (warm twice, cold once) with `CHESHIRE_GPU_VIS_LOG=1`, plus mini6 and 41 with CHECK.
- Run the measurements listed in section 10.

**Step 1. Add `#pragma clang fp contract(off)` to `knnGPU.cu` (separate commit).**
- Why: `knnGPU.cu:1-17` has no pragma, unlike the other four float-exact ports. Its "unfused" metric (`:50-52`) and the initial-distance sums (`:89-95`) can be fused on HIP. That explains why about 20 % of Linux distances differ "with either form" (`docs/13-gpu-visibilities.md:64-68`).
- Validate on Windows: mini6, 41 and 833 knn check still identical, digest unchanged.
- Validate on house-pc: at 41 views the ~1.6M distance differences should go to 0. If they do, correct docs/13 and the comment at `verify_end_to_end.py:89-92`.

**Step 2. Reference hygiene and the harness (output unchanged).**
- Row count `!(depth <= 0.0f)` at `visibilitiesGPU.inc:278`, to match the fill at `:293`.
- LOG gets:
  - the NaN-depth count;
  - the tree depth;
  - the digest;
  - one line listing the mp cameras that the loop position c skips or visits outside `cams`.
- Rename `votes()` to `cheshireVotesOrdered`, verbatim.
- Add the CHECK shadow reference (section 5) with pass-tagged verdicts.
- Fix the stale comment at `:12`.
- Validate: the verdict is identical; the digest is stable over 3 runs and between `OMP_NUM_THREADS=1` and 12; the checksum equals step 0's.
- Detects: a harness bug before anything depends on it.

**Step 3. Restructure the host votes (exact, host only; can ship alone).**
1. Per camera, classify k in parallel over contiguous chunks, using the same expressions and skipping 0xFFFFFFFF.
2. Count per (chunk, bucket), with bucket = v>>12.
3. Exclusive scan, bucket-major.
4. Stable scatter of k plus a contribution bit.
5. `schedule(dynamic)` over buckets, applying `push_back_distinct` and the fold in k order.

- Validate: CHECK identical on all three sets; digest equals step 2's.
- **Decision point.** Proceed to step 4 if house-pc is still host-bound (its LOG shows host per camera greater than device per camera), or if the projected remaining 5600X gain is at least 10 s.
- If the device work is cancelled, the cheap substitute is chunked query transfers on a copy stream in `queryAsync`.

**Step 4. Scaffolding.**
- New `hip/port/gpu_knn/visVotesGPU.hpp/.cu`: CUDA dialect, pragma, a cub/hipcub alias as in `popsift_hip.h:189-194`, header exposing only `void*`.
- Step 4l copies the files and appends them to the three fuseCut CMake lines, before 5l.
- `knnGPU` gets `queryResident` plus `void*` accessors.
- `cuda_to_hip.h` gets `cudaStreamWaitEvent` and `cudaEventQuery` if used; neither is mapped today.
- Build HIP Windows, the CUDA Windows tree (`build/av-cuda`, compile only) and the Linux WSL bundle.
- Validate: the mini6 digest is unchanged.
- Detects: hipCUB/CUB toolchain breakage while no logic depends on it yet. The fallback is a hand-rolled stable 3×8-bit LSD sort, about 150 lines.

**Step 5. `K_bp` in shadow.**
- Under CHECK, the host path also runs `K_bp` per band and memcmps q and pixSize against `prepare()`.
- Negative control: `CHESHIRE_GPU_VIS_FMA=0` on Windows HIP must report differences.
- Validate: 0 differences on both passes at mini6, 41 and 833.
- Also time `K_bp` and one `SortPairs` per camera (rocPRIM's per-call host overhead on Windows).

**Step 6. Device votes, opt-in with `CHESHIRE_GPU_VIS_VOTES=1`.**
- The full per-camera pipeline, bands (default: whole camera), the host appends one camera behind, the pass-end commit, the always-on probe (q/pixSize on camera 1 plus fold triples), and the extended CHECK.
- Validate on mini6:
  - both verdicts identical on both passes;
  - digest and checksum equal the `VOTES=0` run;
  - `CHESHIRE_GPU_VIS_BAND_PIXELS=65536` and `=w` (one-row bands) also identical.
- Then validate on 41 views.

**Step 7. Guards and recovery.**
- Depth guard, planner, `tierOf` check, launch-time band cap, truncate-and-rerun recovery, `CHESHIRE_GPU_VIS_VOTES_FAIL_AT=c`.
- Validate on mini6:
  - an injected failure at camera 3 of each pass gives the identical digest and verdict;
  - `CHESHIRE_BRIDGE_VRAM_MB=300` announces the VRAM-short host path and gives an identical digest;
  - a forced depth-guard trip gives an identical digest.

**Step 8. Scale.**
- 41 views: CHECK with a matrix of band sizes (65536 / 1M / full) and bridge caps (uncapped / 1500 / 600 MB). All identical; `CHESHIRE_BRIDGE_LOG=1` shows no spill on the device path.
- 833: one CHECK run (slow), then timing runs on an idle box, warm twice and cold once.

**Step 9. Readers.**
- Add `CHESHIRE_GPU_VIS_READERS` (read order unchanged, so exact).
- Set the default from the maps-blocked time.

**Step 10. Default on.**
- Announce lines, gate entries, `visband` config, USING.md tables, a docs/13 section.
- Run the gate on mini6: base, verify, cpufallback, blast, bridgecap, visband.

**Step 11. Other hardware.**
- house-pc Linux, RX 6750 XT and RX 5500 XT: 41 with CHECK, then 833 timing. This exercises the unfused form on HIP.
- bench-pc Windows, RX 5500 XT: the launch cap.
- bench-pc CUDA, GTX 1080 Ti and 1050 Ti 4 GB: CHECK, plus a cap forcing host votes.

**Step 12. Follow-ups, each behind its own switch and proven by CHECK.**
- Copy stream to overlap the depth upload (about 2 s per pass).
- knn kernel layout: points stored in leaf order, 32-byte frames, stack sized to tree depth.
- Decoded depth cache, only if the maps-blocked time exceeds about 5 s. It does not fit in house-pc's 14.6 GB at 833 views (16.8 GB of maps).

## 8. Expected gains

**5600X + RX 9070, False Door, idle box.**

Baseline: pass 1 is 72.5 s (`meshing.log:919`). Pass 2 is 66 s, taken from the clean cold run; the 133 s run had other load on the box.

| Stage | Pass 1 | Pass 2 | Saved | Meshing |
|---|---|---|---|---|
| 0.3.3 | 72.5 s | 66 s | 0 | about 480 s |
| Step 3 | 56 s | 58 s | **about 24 s** (15-28) | about 456 s |
| Steps 6-9 | 48.7 s | 50.4 s | **about 40 s** (35-50) | about 440 s |
| Plus step 12 kernel layout (15-35 % of 81.6 s) | | | 12-28 s more | about 412-428 s |

**Step 3 arithmetic.**
- Host per camera: 5.6 + 28.1 + about 5-12 (estimated votes) = 39-46 ms, below the device's 62 ms. The pass becomes device-bound.
- Pass 1: 831 × 62 ms = 51.5 s, plus index 3.2 s, plus tail 1.4 s = **56 s**.
- Pass 2: (51.6 kernel + 15.4 transfer) ms × 831 = 55.7 s, plus 2.5 s = **58 s**.

**Steps 6-9 arithmetic.**
- Device per camera: depth H2D 2.3 + `K_bp` about 1.5 + knn 46.5 + decide/sort/fold about 2.5 + bitmap 0.15 ≈ 53 ms.
- Pass 1: 831 × 53 ms = 44 s, plus 3.2 s, plus about 1.5 s (setup, a 280 MB commit, the pixSize loop) = **48.7 s**.
- Pass 2: 831 × 57.6 ms = 47.9 s, plus 2.5 s = **50.4 s**.
- Condition: the readers keep up, i.e. decode time ≤ R × 53 ms.

**house-pc (i3-4330, 4 threads, RX 6750 XT), False Door. Extrapolated; step 0 measures it.**
- Today: about 3 × (28 + 46) ms of host work per camera, plus decoder contention ≈ 220 ms. With the index (about 10 s), that is about 190 s per pass and about 380 s for both.
- Device floor: 26 ns per query (from the 41-view kernel, 2.3 s over 88.3M queries) × 3.76M ≈ 98 ms, plus about 5 ms ≈ 103 ms per camera, i.e. about 86 s plus 13 s per pass.
- Decode floor: 831 × t_decode / 3 readers, which is under the device floor if t_decode ≤ 0.3 s.

| Stage | Saving |
|---|---|
| Step 3 alone | about 130-170 s (the pass stays host-bound at about 110-150 s) |
| Steps 6-9 on top | another 40-65 s, landing at about 100 s per pass |
| Total | **about 170-230 s of about 380 s** |

## 9. Risks and how each is detected

| Risk | Detected by | Response |
|---|---|---|
| The host's fused form is guessed wrong | Step 0 disassembly; step 5 check plus the negative control; always-on probe | The probe sends the pass to host votes and names the form that would match |
| A `-ffp-contract=fast` host build (e.g. gcc `-march=native`) fuses the fold | Fold probe (4096 triples, compiled in the same translation unit as the reference votes) | Host votes |
| HIP contracts `_rn` intrinsics | Pragma in every new translation unit; step 1's Linux distance count going to 0 | Re-establish the Linux baseline |
| hipCUB/CUB fails on a toolchain or generic target | Step 4 builds all 3 toolchains before any logic lands | Hand-rolled stable LSD sort |
| TDR on slow Windows cards | Launch time measured on the first band; step 11 on the RX 5500 XT and GTX 1050 Ti | Launch-time band cap |
| Spills; atomics in host memory | `tierOf` on every block; step 7/8 cap matrix | Host votes; never a partial spill |
| Recovery leaves lists half-appended | Fault injection in step 7 | Truncate to the saved sizes |
| Decoding becomes the floor (4 threads) | Maps-blocked time in LOG | Readers knob; the cache does not fit on house-pc |
| Gain smaller than modelled | Decision point after step 3; LOG split per stage | Stop, or take the chunked-transfer substitute |
| Pass-2 timing contaminated | Idle box, warm and cold runs | Discard loaded runs |
| Reference is not upstream (upstream is racy) | Stated in the verdict text and docs/13 | Same gap as 0.3.3, not widened |
| Pass-1 lists unsorted after an upstream fallback | Not an issue | `push_back_distinct` assumes nothing |

## 10. Measure first (step 0)

1. **Per-thread vote counts and times** in `votes()`: imbalance or memory latency? This sets step 3's payoff.
2. **`readMap(depthMapFiltered)` time at 3000x1688** on both boxes, idle and loaded, and whether OIIO parallelises internally (`exr_threads`). This sets R and the decode floor.
3. **house-pc False Door baseline** with LOG (copy about 23 GB of cache). The RAM is already known: 14.6 GB.
4. **Clean pass-2 time** on an idle 5600X (expected about 66 s).
5. **Object-code fused forms**, with `llvm-objdump`, for mvsUtils `getCamPixelSize` / `getPixelFor3DPoint`, mvsData `pointLineDistance3D`, and the `backproject` and fold inlined into fuseCut, on all three builds.
6. **Linux nanoflann version.** The superbuild pins commit `92911c0b…` (`DependenciesVersions.cmake:137-138`). Read `NANOFLANN_VERSION` and diff the leaf test and traversal against 1.9.0.
7. **hipCUB under house-pc's `/opt/rocm`.**
8. **Tree maximum depth** at 833 views, for the 96 guard.
9. **NaN count** in the filtered maps.
10. **Which 2 mp cameras** fall outside `cams`. Recorded, not changed.
11. **knn time per launch** on the RX 5500 XT and GTX 1050 Ti.

Two items become moot once the queries stay on the device: the 8.8 GB/s transfer rate and caching of CPU reads from pinned memory.

## 11. The kd-tree filter (about 16 s; 21 s measured)

Not in this release. The breakdown is:
- 12.2 s of nanoflann builds, which a device port cannot remove exactly;
- 7.4 s of rounds, which is the most a new radius walker plus a new CHECK could win;
- 1.5 s of classify and gather.

That is a poor ratio next to this work, and it would spend the same validation budget.

What is worth doing in the same release are two exact, host-only neighbours of the filter, each as its own commit after step 10:
- **Move instead of copy in `removeInvalidPoints`:** about −11 s (`PointCloud.cpp:128`).
- **Angle loop over j>i, with each camera vector normalized once:** about −4 s. The angle is bitwise symmetric, so this is exact.

Both are proven by the checksum and by the count of points removed by the angle filter. Reusing the filter tree across the margin loop (about −6 s) waits for the next release, gated by `CHESHIRE_FILTER_CHECK`.