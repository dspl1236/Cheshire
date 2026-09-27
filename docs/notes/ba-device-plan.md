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
