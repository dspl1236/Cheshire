# Bundle adjustment on the device: plan (0.3.6)

Status: design, 2026-09-27. Nothing is built yet.

## Where the time is (step 0)

Today's defaults, `scripts/sfmbench.py`, RX 9070 box (Ryzen 5 5600X, 12 threads); runs p036-* in
`build/sfmbench/bench.jsonl`.

| | False Door, 884 views | engine bay, 107 views |
|---|---|---|
| SfM node | 1498 s | 46 s |
| bundle adjustment, by the log's clock | 1222 s (82 %) | about 21 s |
| Jacobians | 313 s | 4.4 s |
| linear solver | 207 s | 8.1 s |
| residuals | 55 s | 0.9 s |
| Problem build + Ceres preprocessor + teardown | 183 + 120 + 30 s | 2.7 + 4.6 + 0.3 s |
| solves | 953 | 140 |

Outside bundle adjustment at 884 views: resection 128 s, loading 60 s, triangulation 44 s.

The 0.3.4 roadmap figure, Jacobians at 49 % of bundle adjustment, predates the analytic Jacobians
(5m) and the fused projection (5q). Today the Jacobians are a fifth of a large set's SfM and a tenth
of a small one's. So this is a large-set feature.

## Part A: residuals and Jacobians on the device

**Mechanism: `ceres::EvaluationCallback`, with Ceres unchanged.** Ceres calls
`PrepareForEvaluation(evaluate_jacobians, new_evaluation_point)` before it evaluates the residual
blocks. By then the user's parameter memory holds the point being evaluated. The callback:

1. uploads the parameter values that changed (landmarks, poses, intrinsics);
2. evaluates every residual block of the Problem on the device, with residuals always and Jacobians
   when asked, in one launch;
3. downloads the result into a host buffer indexed by residual block.

Each residual block's `CostFunction::Evaluate` then copies its slice. Ceres' threads, its Jacobian
storage, the Schur elimination and the solver stay exactly as they are. The cost functions are the
5m/5q analytic ones: same parameterisation, same block layout, and the same fused projection for
pinhole with no, K1, K3 or Brown distortion. Other intrinsic models fall back to the host path.

**Exactness.** The device code is the host analytic code, operation by operation, in double, with
contraction off: the HIP pragma, and `--fmad=false` on CUDA as in 0.3.3. So it can be bit-identical
to the host analytic path, as the other ports are. `CHESHIRE_BA_CHECK` compares against autodiff
today; the device path gets its own check against the host analytic path, block by block, and it
must be exact.

**Transfer arithmetic, the risk.** Ceres needs the Jacobians on the host, so each one crosses the
bus once. Take the last local solve of the 884-view run as a sample: 476,332 residual blocks,
51 iterations, 4.2 s.

- Most blocks look at a constant pose (24 poses refined, 239 constant), so their Jacobian is the
  landmark's 2 x 3 doubles plus 2 residuals: about 64 bytes. That is about 30 MB per evaluation.
  - 12 GB/s (PCIe 3.0 x16, house-pc): about 2.5 ms.
  - 25 GB/s (PCIe 4.0 x16, RX 9070 box): about 1.2 ms.
- On the host, the Jacobian phase is about 313/808 of the solve time, roughly 30 ms per iteration
  for this solve on 12 threads.
- On the device: the evaluation is well under a millisecond, and the transfer is 1-3 ms. Copying the
  slices into Ceres' storage is about 30 MB of memcpy spread over its threads, a few ms.
- Estimate: roughly 5-10 ms against 30 ms per iteration, so the 313 s phase becomes something like
  60-100 s at 884 views. Residuals go the same way (55 s).

The parameter upload is small: 238k refined landmarks x 24 bytes, about 6 MB. Only the blocks that
changed need to go up; constant blocks are uploaded once per Problem.

What would sink it: many small solves where launch and transfer latency dominate (the 953 solves
include tiny ones), and the per-block copy into Ceres' storage. Step A1 measures both before
anything else is built.

**Steps.**

- **A1: measure first.** On the False Door replay, time a no-op `EvaluationCallback` that only
  copies precomputed buffers into the blocks, which is the host-side floor of the scheme. Also
  histogram the residual blocks per solve. If the floor is not well below today's Jacobian time, the
  plan stops here.
- **A2: the device kernel.** One thread per residual block, grouped by intrinsic model. The
  parameter-block layout is mirrored on the device when the Problem is built. Includes the check
  mode and a cutoff below which small solves stay on the host.
- **A3: gates.** The check is exact on 41, 107 and 884 views. `sfmbench` digests match the host
  analytic path; they should be byte-identical if A2 is exact. The quality gate is needed only if
  they are not.

## Part B: the per-solve bookkeeping on the host

Build, preprocessor and teardown are 333 s at 884 views, as much as the Jacobians. Step 5r keeps a
Ceres Problem alive across solves only while every landmark is active and there are at most 100
poses. A local-BA run at 884 views rarely qualifies. The local strategy's solves each build a new
Problem over a moving window, and Ceres re-derives the Schur ordering and the block structure every
time.

- **B1: find the split.** Instrument the build: how much is `AddResidualBlock` and parameter-block
  setup, how much is the ordering, and how much is `Problem` destruction (30 s of teardown is itself
  a hint).
- **B2: keep what repeats.** Candidates:
  - keep the landmark and pose parameter blocks registered across solves and toggle them
    constant or variable, instead of rebuilding;
  - reuse the ordering where the window changed little;
  - `Problem::Options::enable_fast_removal` if removal is the teardown cost.

  This is host code, so it helps every build and both backends.

## Part C: the linear solver (later, if the profile says so)

207 s at 884 views, 14 %. It is Schur complement plus a sparse or dense Cholesky on the reduced
camera system. Its time grows with poses, and the local windows keep that system small. So it
waits until A and B are done and the profile is taken again.

## Not in this plan

- Global SfM: it changes the geometry.
- Moving resection or triangulation: 128 s and 44 s at 884 views; small, and already parallel on
  the host.
- The new SfMExpanding pipeline needs nothing separate. It runs the same `BundleAdjustmentCeres`
  and cost functions (docs/roadmap.md, "Upstream's next pipeline"), so A carries over. B would need
  its own call site in SfmBundle.

## Step A1 result (2026-09-27): no-go for Part A as designed

`CHESHIRE_BA_EVAL_PROFILE=1` times the analytic `Evaluate` per thread and compares it with Ceres'
own phase times after each solve. Engine bay, 107 views:

| | our arithmetic inside `Evaluate` | Ceres' phase | arithmetic's share |
|---|---|---|---|
| Jacobians, 12 threads (thread-seconds = wall x 12) | 12.1 s (293 ns a call) | 57.2 s | 21 % |
| Jacobians, 1 Ceres thread (no idle time) | 10.0 s | 29.5 s | 34 % |
| residuals, 1 Ceres thread | 3.3 s | 8.5 s | 39 % |

Two thirds of the Jacobian phase is Ceres' per-block machinery, even on one thread. The rest of the
gap at 12 threads is parallel inefficiency (29.5 s on one thread becomes 57 thread-s on twelve).
Ceres 2.2's `ResidualBlock::Evaluate` (internal/ceres/residual_block.cc) does, unconditionally:

1. fill the outputs with a NaN sentinel (`InvalidateEvaluation`);
2. scan them afterwards (`IsEvaluationValid`);
3. apply manifold PlusJacobians through a generic dynamic-size `MatrixMatrixMultiply`;
4. apply the loss function's `Corrector`.

The evaluator then scatters the block into the block-sparse Jacobian. A device `Evaluate` would
remove at most the third that is arithmetic, a few percent of SfM, and pay transfers on top.

Options, in order of cost:

- **Profile the machinery first.** Run the Linux build of incrementalSfM under `perf` (WSL) on the
  engine-bay and False Door caches, with Ceres' symbols, to see which of the four pieces and the
  scatter take the time.
- **Trim it in the Ceres Cheshire builds**, which are ours on Windows and in the Linux deps:
  - the sentinel scan, where the cost function guarantees finite output (the analytic path can
    check its own inputs);
  - a fixed-size path for the common manifold sizes.

  Host-only, helps both backends. It has to stay exact: the same arithmetic, less bookkeeping.
- **Our own bundle-adjustment loop** that evaluates straight into the Schur structure, on the host
  or the device. XL, and the largest change to results risk; only if the first two leave the phase
  dominant.

Part B (the per-solve build, preprocessor and teardown, 22 % at 884 views) is unaffected by this and
comes next.

### Confirmed at 884 views, and one piece measured (2026-09-27)

False Door replay, 12 threads, 951 solves: the analytic arithmetic is 850 of 3,406 thread-seconds
of Ceres' Jacobian phase (25 %), over 3.36 billion evaluations at 253 ns each. Residuals: 184 of 656
(28 %). The same picture as the engine bay.

An experimental Ceres (the 2.2.0 source Cheshire builds, with `CERES_SKIP_EVAL_CHECK=1` guarding the
NaN-sentinel fill and scan; `build/ceres-exp.cmd`, not shipped) on the engine bay with one Ceres
thread:

| | Jacobian phase | residual phase | output |
|---|---|---|---|
| checks on | 23.0 s | 6.3 s | sfm `a76946037b3860e6`, cameras `38450e663421b3e4` |
| sentinel fill and scan skipped | 19.8 s (-14 %) | 6.1 s | the same, byte for byte |

The same run counted 36.2 million manifold Jacobian products, close to one per Jacobian evaluation.
AliceVision gives every intrinsics block a manifold (`IntrinsicsManifold`), and poses a
`SubsetManifold` when part of their extrinsics is constant (BundleAdjustmentCeres.cpp:290, 495).
Ceres applies both through the dynamic-size `MatrixMatrixMultiply`.

**Direction for 0.3.6.** Not device Jacobians inside Ceres. In order:

1. **Part B**, the per-solve host bookkeeping: 22 % of a large set's SfM, unaffected by any of this.
2. **Exact host trimming in the Ceres Cheshire builds.**
   - Replace the sentinel with the cost function's own finiteness check, returning false on a
     non-finite output, as `IsEvaluationValid` would: -14 % of the Jacobian phase, output
     byte-identical.
   - A fixed-size path for the manifold product sizes that occur.

   Both mean shipping a patched Ceres on Windows and in the Linux deps, a maintenance cost to weigh
   against a few percent of SfM.
3. **A bundle-adjustment loop of our own** that evaluates straight into the Schur structure. It
   removes Ceres' per-block machinery altogether, on the host or the device. XL, and not in 0.3.6.

## Step B1 result (2026-09-27): the build cost is the main solves' rebuilds

A local measurement build (stage timers in `createProblem`, and a 1-in-64 sample of cost-function
construction against `AddResidualBlock`; not in the generator). False Door replay, 929 solves,
1139 s. SfM wall time at this size varies run to run: 1139 to 1498 s over four runs, with 929-953
solves.

- **Only 36 solves rebuilt the Problem.** They are the main bundle adjustments under the local
  strategy, where 5r rebuilds on purpose: with persistence, Ceres' preprocessor scans every residual
  block of the Problem at each Solve. The other ~893 are small per-view refinements, about 8 s in all.
- **The 36 rebuilds took 144.8 s:**
  - landmarks 105.6 s, of which cost-function construction ~28.8 s and `AddResidualBlock` ~48.3 s,
    over 67.7 M observations;
  - 39.2 s outside the three stages, mostly dropping the previous persistent Problem;
  - plus 25.2 s of teardown after the solves, and most of the ~100 s preprocessor.

**Where 0.3.6 stands on bundle adjustment.** Both halves of its time sit in Ceres' generic machinery:
the per-block evaluation around our arithmetic, and the per-solve Problem build, preprocessing and
teardown. Neither a device `Evaluate` nor more persistence reaches it. The exact wins left are each
a few percent of a large set's SfM:

- keep cost-function objects across rebuilds (Problem option `cost_function_ownership =
  DO_NOT_TAKE_OWNERSHIP`, cache keyed by (landmark, view)): about 30-40 s at 884 views, no Ceres change;
- drop the NaN-sentinel scan (-14 % of the Jacobian phase) and give the manifold products a fixed
  size: both need a patched Ceres in the Windows and Linux dependency builds.

The structural answer is a bundle-adjustment loop of our own that evaluates straight into the Schur
structure, on the host or the device. That is an XL item, and a decision for the owner.

### Cost functions kept across rebuilds: measured and dropped (2026-09-27)

A local measurement build counted how many observations of each main-solve rebuild were already in
the previous one. On the False Door replay: 52.3 M of 72.7 M over 40 rebuilds (71.9 %; from 31 % to
100 % per rebuild). A cache holding one rebuild's cost functions would therefore save about 72 % of
the ~29 s of cost-function construction and part of the teardown, about 20-30 s, or 2 % of a
large set's SfM.

The same run took 1631 s, against 1139-1498 s for the four before it. At this size SfM's wall time
varies by more than 20 % run to run, so a 2 % gain cannot be shown without many repeats. It also
needs `DO_NOT_TAKE_OWNERSHIP` with separately owned cost functions for every other residual kind,
and holds a second copy of a rebuild's cost functions in memory. Dropped. The one remaining host
lever of any size is in Ceres itself (sentinel scan, fixed-size manifold products), and it waits for
0.3.7's decision on a bundle-adjustment loop of Cheshire's own.

