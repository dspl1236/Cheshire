# A bundle-adjustment solver of Cheshire's own: design (0.3.7)

Status: 2026-09-29. Steps 0-3 done. The solver takes Ceres' iterates to rounding and solves in a third of Ceres' time. It is now the default in SfM (`CHESHIRE_BA_SOLVER=ceres` for Ceres), and passes the SfM quality gate on 41 views, the engine bay and the False Door (below). Background and the measurements
behind the decision are in docs/notes/ba-device-plan.md.

## Why

At 884 views, bundle adjustment is 82 % of SfM (1222 of 1498 s on the RX 9070 box). The analytic
arithmetic Cheshire already supplies is only a quarter of Ceres' Jacobian phase (A1). The rest of the
time is Ceres' generic machinery:
- **per-block evaluation:** the NaN sentinel, the manifold products through a dynamic-size multiply,
  the corrector and the scatter;
- **per-solve bookkeeping:** Problem build, preprocessor and teardown, 22 % of the node (B1).

No change inside Ceres reaches both halves. A solver written for this one problem shape does: it
evaluates straight into the structure the linear solve needs and keeps that structure between
iterations and, where the problem allows, between solves.

## Step 0: where the time is, per solve (2026-09-28)

`scripts/sfmbench.py run <set> CHESHIRE_BA_PROFILE=1`, today's defaults, RX 9070 box (Ryzen 5 5600X);
the per-solve lines are grouped by `build/ba037/solves.py`. "wall" = build + solve + update + destroy.

**False Door, 884 views**: SfM 1140 s, bundle adjustment 675 s in 924 solves.

| solver | residual blocks | solves | iterations | wall | share |
|---|---|---|---|---|---|
| DENSE_QR (pose refinements) | up to 10,000 | 860 | 4,725 | 16.9 s | 2.5 % |
| DENSE_SCHUR | 1,000 to 1,000,000 | 30 | 115 | 25.0 s | 3.7 % |
| SPARSE_SCHUR | 100,000 to 1,000,000 | 5 | 21 | 32.3 s | 4.8 % |
| **SPARSE_SCHUR** | **over 1,000,000** | **29** | **244** | **600.9 s** | **89.0 %** |

Inside those 29 solves:

| piece | time |
|---|---|
| Problem build | 156.1 s |
| Ceres preprocessor | 100.0 s |
| Jacobians | 129.1 s |
| linear solver | 105.6 s |
| residuals | 23.2 s |
| other minimizer work | 45.5 s |
| teardown | 27.6 s |

The largest took 108 s: 3.07 million residual blocks, 51 iterations, the 50-iteration cap.

**Engine bay, 107 views**: SfM 58 s, bundle adjustment 33.5 s in 141 solves. Dense Schur 29 solves
over 10,000 blocks, 75 %; sparse Schur 4 solves, 22 %; the pose refinements 105 solves, 2 %.

**What it decides.** The first target is the large sparse-Schur solves. Build, preprocessor and
teardown are 284 s, 47 % of their time: bookkeeping, not arithmetic. The Jacobian phase is 129 s
around roughly 35 s of analytic arithmetic (A1's quarter). The linear solve, 106 s, is mostly the
elimination. With three F blocks per residual of sizes the manifolds make vary, it runs on Ceres'
dynamic-size eliminator; Ceres specialises only one F block of size 6.
- **Shadow mode (steps 1-2)** can save the preprocessor, most of the Jacobian overhead, the generic
  elimination and the other minimizer work: roughly 250-300 s of the 601.
- **Building from the SfM data directly (step 3b)** also saves the build and teardown, 184 s.
- **The pose refinements** stay on Ceres (DENSE_QR, 2.5 %).

## Step 1: shadow mode, first results (2026-09-28)

`hip/port/sfm_ba/ownSolver.hpp`, generator step 7a. `CHESHIRE_BA_SHADOW=1` runs the solver in place
before Ceres on every DENSE_SCHUR / SPARSE_SCHUR solve, restores the starting values, and logs one line
comparing the two (`build/ba037/shadow.py` summarises a log).
- `=2` adds both trajectories whenever they part.
- `=3` is the control: Ceres itself on one thread as the shadow.

| set | solves | same iterations | parted | final cost, worst rel. diff | parameters, worst abs. diff |
|---|---|---|---|---|---|
| 41 views | 68 dense | 68 | 0 | 3.0e-14 | 7.1e-12 |
| engine bay | 137 dense, 3 sparse | 140 | 0 | 2.9e-13 | 1.2e-9 |
| False Door, first 715 solves | 691 dense, 24 sparse | 713 | 3 | 8.5e-3 | |

The False Door run was stopped at 715 solves, since step 2 replaced the code. Its one real parting,
a 2.5-million-row sparse solve that ended 0.85 % from Ceres, was the sparse ordering (step 2).

**What the first engine-bay run found:** four small early solves (24-42 columns) parted, from the
first or third iteration on, and one step was exactly 1000x Ceres'.
- **Not rounding.** The control run showed Ceres on one thread and on twelve agreeing on all 140
  solves, those four included.
- **The cause was bounds.** AliceVision bounds the focal length (with a prior) and the principal
  point (±5 % of the image). With any bound on a free block, Ceres projects every Plus onto the box
  and runs a projected Armijo line search, with cubic interpolation, along every trust-region step.
  - The line search shrinks a failing step by at most `max_line_search_step_contraction` = 1e-3:
    the 1000x.
- **The fix:** both, ported (the polynomial helpers from Ceres' polynomial.cc). All 140 solves then
  matched.

**A saving this exposes.** The line search's first trial evaluates the gradient at the trial point
(CUBIC interpolation), and it is used only if Armijo fails there. Ceres pays that gradient
evaluation on every iteration of every bounded solve. Computing it only when needed gives the same
iterates.

**Time so far:** the solver is 2.5-3.2x slower than Ceres on these sets. It is single-threaded, with
a dense S, map-based chunk buffers and a line search that always takes the gradient. That is step 2's
work, now that the iterates match.

## Step 2: the parallel, block-sparse solver (2026-09-28)

Same file, same shadow mode. What changed from step 1:

- **Ceres' reduced program, in Ceres' order.**
  - Parameter blocks no residual block uses are dropped, as Ceres' preprocessor drops them.
  - For `SPARSE_SCHUR` the camera blocks are reordered as `ReorderSchurComplementColumnsUsingEigen`
    reorders them: AMD on the block pattern F^T F − F^T E E^T F. `SimplicialLDLT` then keeps the
    natural order (`AreJacobianColumnsOrdered`) and analyses the pattern once per solve.
  - Step 1 had run scalar AMD, a different elimination order. That was its one real parting: the
    ill-conditioned 2.5-million-row False Door solve ended 0.85 % from Ceres (below).
- **Threads, without the thread count in the result.**
  - Evaluation runs over fixed partitions of whole chunks.
  - Elimination is fused with the assembly of S over fixed groups of chunks. Each group streams its
    landmarks once into its own copy of (S, rhs), and the copies are added in group order. The
    first version assembled row by row of camera blocks; it was bound by memory latency, and the
    groups are 1.8x faster.
  - Vector passes sum over fixed blocks.
  - All boundaries depend on the problem alone. On 41 views with `CHESHIRE_SFM_DETERMINISTIC=1`,
    `CHESHIRE_BA_OWN_THREADS=1` and `=12` give the same final-state digest on all 68 solves.
- **S as its upper block triangle**, with the pattern fixed per solve. Dense Schur fills the lower
  triangle for LLT. Sparse Schur keeps a CSC pattern whose values are refilled per iteration.
- **The line search takes a gradient only when a sample fails Armijo.** The cubic step is the only
  reader. The candidate's cost is the search's last sample, which is the point Ceres evaluates
  again. One deviation: a sample that passes is never asked for its Jacobian. A non-finite Jacobian
  there counts as valid here and as invalid in Ceres. On the False Door the search needed a gradient
  in 27 % of samples.
- **Fixed-size kernels** for the pose blocks, with the same summation order per entry (bit-identical).

**Diagnostics.**
- `CHESHIRE_BA_SHADOW=2` traces print our line search per iteration (samples, gradients, the step
  size, whether it failed). The shadow line carries Ceres' termination message.
- `CHESHIRE_BA_OWN_THREADS` sets our threads alone.
- `CHESHIRE_BA_OWN_EAGER=1` takes every sample with its gradient, as Ceres does.

| set | solves | same iterations | parted | final cost, worst rel. diff | ours | Ceres (solve) |
|---|---|---|---|---|---|---|
| 41 views | 68 dense | 68 | 0 | 1.4e-14 | 6.1 s | 21.1 s |
| engine bay | 137 dense, 3 sparse | 140 | 0 | 1.6e-11 | 7.7 s | 21.3 s |
| False Door | 886 dense, 34 sparse | 916 | 4 | 5.3e-2 (below) | 172 s | 487 s |

False Door split of our 161 s in the 34 sparse solves: setup 23 s, evaluations with Jacobians 28 s, costs 18 s, gradients 11 s, elimination and assembly 25 s, factorisation 12 s, back-substitution 4 s, and 39 s of vector passes, Plus and state copies. Ceres took 461 s for the same solves.

**Where they still differ, and why it is rounding.**
- **The line search at the floor.** In a 217-row solve, both reach iteration 5 identically.
  - Ceres: "Parameter tolerance reached. Relative step_norm: 1.24e-14", after 15 line-search steps.
  - Ours: the search shrinks to a = 1e-16 and fails on the minimum step size. We then go on to a
    lower cost (1,439,322 against 1,443,677).

  At a step of 1e-14 of the parameters, Armijo compares two costs that differ in the last bits,
  and the answer is decided by summation order. A 351-row solve does the same the other way round.
- **An ill-conditioned solve.** The 2.5-million-row solve starts at a cost of 1.4 million, and its
  first step has norm 960,000. After that step the two costs differ by 3.7e-10. That difference
  grows to 1e-7 to 5e-4 by the 50th iteration, depending on the run.
- **The control.** Ceres against itself (`CHESHIRE_BA_SHADOW=3`: one thread as the shadow, twelve
  driving, the same False Door run): 919 of 920 solves take the same iterations, 4 part, and the worst
  final difference is 3.0e-3. The worst is that same 217-row solve with the same two outcomes. One
  thread stops at iteration 4 on the parameter tolerance (1,443,676); twelve threads run all 50
  (1,439,322). The ill-conditioned large sparse solve parts at iteration 5 and ends 7.2e-5 apart. So
  Ceres disagrees with itself where the solver disagrees with Ceres, and by the same amounts. These are
  the places where summation order decides, not faults of either. The solver shows three more solves
  ending on a different iteration than the control does, all of them line searches at the floor.

**Time, next.**
- **Setup:** 23 s of the sparse solves. It is Ceres' `std::map` lookups per parameter
  block, and building from the SfM data (step 3b) removes it.
- **Factorisation:** Eigen's single-threaded simplicial LDLT, kept for parity. A supernodal or
  dense factorisation is a step-3 choice, judged by the quality gate.
- **Evaluations:** these are the analytic cost functions, and the device's (step 4).

## Step 3: in SfM, in place of Ceres, the default (2026-09-28)

Generator step 7b. Every `DENSE_SCHUR` / `SPARSE_SCHUR` solve goes to
`cheshire::own::solveInstead`. That solves it and fills Ceres' `Solver::Summary` with what
AliceVision's statistics and Cheshire's profile lines read: the costs, the iterations, the step
counts, the problem sizes and a time split. It is the default. `CHESHIRE_BA_SOLVER=ceres` gives
Ceres back, and one line per run says which solver is in use.

Everything else goes to `ceres::Solve` as before:
- **Options the solver does not implement:** a strategy other than Levenberg-Marquardt,
  nonmonotonic steps, inner iterations, a time limit, or line-search interpolation other than cubic.
- **Problems outside its shape:** a residual block on two eliminated blocks, an eliminated block not
  of size 3, a camera block over 32 parameters, or a `SPARSE_SCHUR` problem left with no eliminated
  block.

Each reason is logged the first time it sends a solve to Ceres. On the three sets, none did.

**Around the solve it follows Ceres:**
- The best point is kept on success, and the starting point on FAILURE (`Solver::Solve` restores it).
- Iteration 0 counts as a successful step.
- AliceVision's iteration callbacks see every iteration afterwards.
- **No eliminated block.** In the resection pose refinements (about 860 per False Door run) every
  landmark is constant. Ceres then gives up the Schur solver (`LinearSolverForZeroEBlocks`), keeps
  the problem's own order, and solves `DENSE_QR`: Eigen's HouseholderQR of [J; D]. The solver does
  the same. It had solved the normal equations, which square the condition number.
  - Every solve that ended on a different iteration count in step 2 was one of these ten-column
    problems.
  - They now agree to 1e-16.
- **Threads:** the solver's result does not depend on its thread count, so it takes the threads
  `BundleAdjustmentCeres` asked for even under `CHESHIRE_SFM_DETERMINISTIC=1`. That mode's single
  thread is Ceres' need; `CHESHIRE_BA_THREADS` still lowers it.

**The reverse shadow** (`CHESHIRE_BA_SHADOW=4`) keeps the solver's result, so its trajectory drives,
and solves every problem with Ceres beside it.
- **Why it was needed:** with the solver driving, the False Door's large solves took 666 iterations,
  against Ceres' 241 in its own runs. The iteration counts matched on the same problems, so the
  question was whether the solver's trajectory meets harder problems, or the solver is slow on them.
- **The answer is harder problems.** On the solver's trajectory, 922 of 925 solves take the same
  iterations under Ceres. The large ones take 350 iterations against Ceres' 336, the difference being
  one solve at the rounding floor.
- **The time on those same problems:** 174 s against 453 s.
- **Trajectories vary that much anyway.** Ceres' own runs reach the False Door's end through anything
  from 241 to 666 iterations of large solves.

**The SfM quality gate** (`scripts/quality_gate.py sfm`, today's Ceres as the baseline):

| set | solver runs | poses | landmarks | RMSE | camera centres / rotations vs baseline spread | verdict |
|---|---|---|---|---|---|---|
| 41 views | 10 | 41 = 41 | 80,802 vs 80,803 (p 0.52) | 1.23321 vs 1.23374, better (p 0.003) | 1.00x / 0.60x | PASS |
| engine bay | 10 | 107 = 107 | 140,471 vs 140,418 (p 0.09) | 1.68536 vs 1.68518 (p 0.72) | 1.13x / 0.97x | PASS |
| False Door | 3 | 830 vs 826 (817-831) | 1,354,019 vs 1,350,360 | 1.37719 vs 1.39055 (1.367-1.437) | 0.04x / 0.19x | PASS |

With the QR fix and as the default:
- **41 views:** 80,796 landmarks, RMSE 1.23379; one of Ceres' own ten runs has exactly 80,796. The
  verdict is WARN on landmarks: a leg with no spread makes a 7-landmark difference significant, but it
  is within the tolerance.
- **Engine bay:** PASS, the RMSE better (1.68404, p 0.046).
- **False Door:** PASS with step 7c (two runs): 830 and 831 poses against Ceres' 817-831, 1.355 M landmarks, RMSE 1.3790 against 1.3906, camera centres 1.03x and rotations 0.70x the baseline's own spread. Four step-7c runs in all: 830-831 poses, RMSE 1.3785-1.4269.

**Reproducible by default, with step 7c.** Every run with the solver gave the same output: 10 of 10
on 41 views, 10 of 10 on the engine bay. Ceres gave 10 different outputs on each; its multithreaded
Schur eliminator was the last source of run-to-run difference in SfM (0.3.4, step 5n).
- The formal check (`--json`): two default runs with the solver, and one with
  `CHESHIRE_SFM_DETERMINISTIC=1`, give the same `sfm.sfm` and `cameras.sfm`
  (`a6c6f270ccba9e1e` / `ab4e4706c5d87477`).
- The deterministic mode no longer costs anything. With Ceres it ran bundle adjustment on one
  thread, 72 s against 23 s on 41 views; with the solver it is 5.9 s.

On the False Door, two runs of the same build then ended apart (815 poses and 833) and one crashed.
The cause was in upstream's resection, fixed as step 7c:
- **What upstream does:** it resects a batch of views in parallel and applies each result at once
  (`updateScene`, in an omp critical section).
- **What races with that:** the other threads read the same scene without a lock. They read the poses
  map one thread is inserting into (`getValidViews`, `getReconstructedIntrinsics`) and the landmarks'
  observations. They also push to the HTML report stream.
- **A new camera's intrinsics:** on a camera's first use, `refinePose` writes its intrinsics back while
  the other threads clone them.
- **Why it shows here:** the False Door has five cameras (820, 27, 23, 13 and 1 views), so the new
  cameras' first views meet in one batch.
- **Why the solver made it more likely:** a faster solve shortens the gaps between the writes.

Step 7c:
1. It resects the views whose camera is already reconstructed in parallel, against the scene as it
   was, and applies them afterwards in view order.
2. It then resects a new camera's views one at a time: the first refines the intrinsics, the next
   ones use them, as upstream means it.
3. It puts the report under a lock.

On a one-camera set nothing changes: 41 views give the same bytes with and without it.

**Time:**

| set | bundle adjustment: Ceres | the solver | SfM wall: Ceres | the solver |
|---|---|---|---|---|
| 41 views | 21.4 s | 6.1 s | 51 s | 35 s |
| engine bay | 24.3 s | 8.5 s | 51 s | 35 s |
| False Door | 344-529 s (3 runs) | 225-527 s (4 runs) | 941-1214 s | 810-1144 s |

The False Door's time depends on the trajectory more than on the solver. On identical problems the
solver is 2.6x faster (the reverse shadow: 174 s against 453 s). A run, though, meets whatever solves
its trajectory brings, and the trajectories differ by hundreds of iterations of large solves, for
Ceres as much as for the solver.

**The False Door reproducible too (2026-09-29).** With step 7c, two runs still ended apart (830 and
831 poses). Two digest diagnostics found where:
- **Per solve:** `CHESHIRE_BA_DIGEST=1`, one line per solve with its starting and final states.
- **Per stage:** step 7d, `CHESHIRE_SFM_DIGEST=1`, the scene (intrinsics, poses, landmarks,
  observations) at every stage of the incremental loop.

The trail:
- Both runs had the same scene at a resection batch's start.
- They had the same RANSAC input, draw and result for a new camera's first view.
- Its pose-and-intrinsics refinement then started from the same values, in a different block order:
  pose, distortion, intrinsics in one run; intrinsics, distortion, pose in the other.

The cause was the solver's `DENSE_QR` path. It follows Ceres' program order (the order the blocks
were added), and took that order from `Problem::GetParameterBlocks`. That call walks a map keyed by
pointer, so it returns address order, which changes from run to run. Householder QR without pivoting
rounds differently when the columns are swapped.

The fix keeps the ordering groups' order. `BundleAdjustmentCeres` numbers the groups as it adds the
blocks, so for a one-camera refinement that is exactly Ceres' order, and elsewhere it is at least the
same every run.

Two False Door runs then gave the same bytes: 828 poses, 1,348,092 landmarks, RMSE 1.38079, and all
549 stage digests equal. So SfM is reproducible by default on every set so far, and with every thread.

## Step 3b: the solve built from the SfM data directly (2026-09-29)

Generator step 7e, `hip/port/sfm_ba/direct.inc`. Until now every solve still paid for the Problem
upstream builds: a cost-function object and a residual block per observation (about 5 M on the False
Door's large solves), read back by the solver and freed again. That was a third of the False Door's
bundle-adjustment time with persistence off, and still 158 s of build and 26 s of destruction with it
on.

**What it does** (`BundleAdjustmentCeres::cheshireDirectAdjust`, at the start of `adjust`):
- **The camera blocks go through upstream's own code.** `addExtrinsicsToProblem` and
  `addIntrinsicsToProblem` add them to a small Problem that holds only them (a few hundred blocks), so
  their constant flags, manifolds and bounds are upstream's.
- **The landmarks and observations become the solver's rows directly.** They are taken in the order
  `addLandmarksToProblem` adds them, with its statistics and error logs.
- **Each row evaluates the fused projection itself.** It runs the same code the cost functions run,
  now shared as `fusedProject` and `projectSimple` in `projectionCheshire.hpp`.
- **The solver reads blocks and rows through a `Source`.** A `ProblemSource` walks a `ceres::Problem`
  as before; the direct build is a second source. The solver itself did not change.

**The program is the one Ceres' preprocessor would build from `createProblem`'s Problem:** the same
blocks, rows, order and ordering groups. So the result is the same bytes.
- `CHESHIRE_BA_DIRECT=check` runs both on every solve and compares their digests: 68 of 68 solves the
  same on 41 views, and 141 of 141 on the engine bay.
- Whole runs give the same `sfm.abc` and `cameras.sfm` as the Problem path with persistence off
  (`CHESHIRE_BA_PERSIST=0`), on all three sets:
  - 41 views: `331c2705c4845713`;
  - engine bay: `3285152e129cf314`;
  - False Door: `d2e34fd8d903fb79` (828 poses, 1,351,227 landmarks, RMSE 1.37536).
- On the False Door all 939 solves took the direct path.

**Persistence (step 5r) is not used on this path,** because no Problem is left to keep. A kept
Problem's output is not byte for byte a fresh build's, so the default's digests change with 3b (41 views `feeaa42f96cc22fb` → `331c2705c4845713`, False Door
`14611aadf00ede49` → `d2e34fd8d903fb79`), to exactly the non-persistent path's.

**Scope.** It covers a solve when the solver is on and:
- the scene has no rigs, survey points, 2-D or point constraints, rotation priors, temporal
  smoothness, mesh-referenced landmarks or depth observations;
- the parameter ordering is on;
- the Jacobians are analytic, with no Jacobian check;
- every camera model is one the fused projection covers (pinhole, with no distortion or with radial
  K1, radial K3 or Brown).

Anything else takes the Problem path as before (and the solver or Ceres from there), with the reason
logged once. `CHESHIRE_BA_DIRECT=0` turns it off, and the shadow modes always use the Problem path.

**Time on the False Door** (same bytes, back to back):

| | SfM wall | bundle adjustment | build | destroy | preprocessor |
|---|---|---|---|---|---|
| Problem path, persistence off | 1340 s | 529 s | 218 s | 40 s | 37.7 s |
| direct | 787 s | 356 s | 11 s | 0 | 14.6 s |

Bundle adjustment itself also got faster: the Jacobians took 96 s against 148 s, and the residuals
35 s against 72 s. The rows are evaluated without a virtual call through a cost-function object, and
their data sit in one array instead of scattered over millions of small allocations.

The persistent default before 3b ran 1065-1138 s on a slightly different trajectory (948 solves, not
939): 158 s build, 26 s destroy, 423 s BA.

**On the small sets**, against the persistent default before 3b (`CHESHIRE_BA_DIRECT=0`, which still
gives its committed digests `feeaa42f96cc22fb` and `8fbb7e604ea6b79c`), one run each on the installed
build:

| set | SfM wall | bundle adjustment | build | preprocessor |
|---|---|---|---|---|
| 41 views | 33.3 s → 28.9 s | 5.7 s → 4.2 s | 1.71 s → 0.56 s | 1.46 s → 0.66 s |
| engine bay | 36.2 s → 33.0 s | 9.5 s → 9.3 s | 2.92 s → 0.63 s | 1.41 s → 0.77 s |

## Step 4a: where the host solver's time is (2026-09-29)

`CHESHIRE_BA_PROFILE=1` now also logs one line per solve with the solver's time per phase:
- the evaluations (Jacobian, cost, gradient);
- the elimination, the Schur sum, the factorisation and the back-substitution;
- the passes around them (column norms, Jacobi scaling, the model cost change, Plus, the state
  copies);
- the problem's sizes.

On the False Door (939 solves, 352.5 s of bundle adjustment, the output unchanged) the 33 solves of
over a million rows take 315 s. Split by phase:

| phase | time | |
|---|---|---|
| evaluations | 122 s | Jacobian 47 s (554), cost 32 s (1066), gradient 43 s (512) |
| elimination, Schur sum, back-substitution | 57 s | 589 linear solves |
| column norms, Jacobi scaling, model cost | 42 s | |
| Plus, state copies, vector passes | 42 s | |
| factorisation | 40 s | Eigen's `SimplicialLDLT`, one thread |
| setup | 12 s | the structure, once per solve |

- **Why so many cost and gradient evaluations:** these problems have bounds (the intrinsics), so
  every step runs Ceres' line search (the constrained case). Each iteration costs about two cost
  evaluations and one gradient evaluation besides the Jacobian.
- **The 868 pose refinements** (dense QR, 22-9,295 rows) take 7 s in all.
- **The problem sizes:** the largest solve has 3.9 M rows, 900 k landmarks, 2,358 camera columns, a
  57 M-value Jacobian and a 558 k-value reduced system.

So nine tenths of the time is work per row or per landmark, which a device does well. The
factorisation is a sparse Cholesky of the reduced camera system, and stays on the host.

**Host arithmetic with FMA.** The Windows build compiles with `/arch:AVX2`, and clang contracts
`a*b + c` within an expression unless told not to. The solver's own code and the fused projection
sit under `#pragma clang fp contract(off)`. The code they call from Ceres' and Eigen's headers, which
are parsed earlier, does not. The LLVM IR of `BundleAdjustmentCeres.cpp` (with line tables) names
every fused operation:
- **`AngleAxisRotatePoint<double>`** (rotation.h:822-849): the rotation in the cost-only path. Its
  Jet version, which the Jacobian path uses, is not contracted: the Jet operators go through Eigen.
- **`AngleAxisToRotationMatrix`** (rotation.h:472-480): the rotation matrix in the Jacobian path.
- **Eigen's 3x3 inverse** (`cofactor_3x3`, InverseImpl.h:135): the landmark blocks' inverses in the
  elimination.
- **`getPrincipalPoint`**, read once per camera.

The Linux build (GCC, generic x86-64) has no FMA, so the two hosts already round these differently.

## Step 4: the plan

**Same bytes as the host.** The device path must give the host solver's result, byte for byte, so
that validation stays a digest comparison. That requires:
- **The same arithmetic.** The device code is the host code, with no contraction: the clang pragma
  on HIP, and `--fmad=false` on CUDA (step 5l).
- **No libm.** The only transcendental functions are per pose (`hypot`, `sin`, `cos` in the
  angle-axis rotation). The host computes them once per pose and evaluation and hands them over; the
  device computes everything per row.
- **The same summation orders.** Every sum is over fixed partitions, groups, chunks or rows, in
  order, and can be run by one device thread per partial sum:
  - the cost and the model cost change: per row, then per partition in order;
  - the gradient and the column norms: per landmark for the E part, per partition and camera block
    for the F part (lists built once per solve);
  - the reduced system: one thread per (group, camera block, row of the block), walking the chunks
    that touch the block in chunk order. Each thread adds, per chunk, the rows' F^T F terms and then
    the chunk's Schur term, as the host does. The groups are then summed in order.

**The steps:**
- **4b: the host without FMA.** Cheshire's own FMA-free versions of the rotation (with the per-pose
  part computed once per pose) and of the 3x3 inverse, which the device will share. The Windows
  digests change once, to what the arithmetic says without contraction.
- **4c: the device.** Evaluation, the passes over the Jacobian, the elimination and the
  back-substitution on the device; the factorisation and the vectors on the host.
  - `CHESHIRE_BA_DEVICE=check` runs both on every solve and compares their digests.
  - The small solves stay on the host.
- **4d: the vectors on the device**, if the profile then says so; and the CUDA build (house-pc).

## Step 4b: the host without FMA (2026-09-29)

`hip/port/sfm_ba/baArith.hpp` holds the arithmetic the host solver and the device share, all under
`fp contract(off)`:
- the Jet<3> operators, restated operation by operation from Ceres' `jet.h`;
- the angle-axis rotation, split into its part per pose (`poseRotation`: `hypot`, `sin`, `cos`, the
  Jets of the angle, the rotation matrix; host only, it uses libm) and its part per point
  (`rotatePoint`, `rotateWithDerivative`);
- the fused projection over plain camera values (`fusedProjectValues`), the chain rule
  (`projectSimpleRot`), the Huber loss, Ceres' corrector;
- Eigen's 3x3 inverse (`inverse3`).

The host uses it everywhere the solver runs. The direct path also computes the part per pose once
per pose and evaluation (`Source::beforeEvaluation`) instead of once per row.

**Checked against the originals.** A standalone test (`hip/tests/sfm_ba/arith_test.cpp`, the build's
clang-cl with `/arch:AVX2`) compares 2 million random cases, including zero and tiny angles:
- **With Ceres' and Eigen's headers under the same pragma:** every value is the same bit for bit
  (the plain rotation, the Jet rotation, the rotation matrix, the inverse).
- **With the Windows build's contraction:** the plain rotation differs in 62 % of the cases, the
  rotation matrix in 67 %, the inverse in 95 %, the Jet rotation in none.

The IR of the rebuilt `BundleAdjustmentCeres.cpp` has no fused operation left on the solver's path
except `getPrincipalPoint`, which is read once per camera.

**The new default digests** (each set run twice, identical):

| set | sfm.abc | cameras.sfm | poses | landmarks | RMSE |
|---|---|---|---|---|---|
| 41 views | `801d1fb7b32bdd32` | `361ddd5cf663cfe6` | 41 | 80,799 (the same) | 1.23336 (the same) |
| engine bay | `814ae0d5b664d733` | `af98cde31a5f153c` | 107 | 140,394 (was 140,429) | 1.68300 (was 1.68441) |
| False Door | `a5cc84375086fdec` | `e05e2fa4df94376b` | 833 (was 828) | 1,355,323 | 1.36812 (was 1.37536) |

- **Quality gate** against step 3's ten Ceres runs (`scripts/quality_gate.py report`):
  - 41 views: WARN on landmarks, as in step 3 (4 fewer, significant only because a deterministic leg
    has no spread, inside the tolerance). RMSE better.
  - Engine bay: PASS, RMSE better.
- **The direct check** (`CHESHIRE_BA_DIRECT=check`): 68 of 68 solves the same. The Problem path
  computes the rotation per row, the direct path per pose.

**Time.** Per iteration of the large False Door solves, a Jacobian evaluation takes 55 ms against
84 ms, now that the part per pose is computed once. The False Door's run took 511 s against 787 s,
and bundle adjustment 123 s against 356 s. Most of that is the trajectory, not the arithmetic: the
large solves took 268 iterations against 562. Ceres' own runs span 241 to 666.

## Step 4c: the device (2026-09-29)

Generator step 7f, `hip/port/sfm_ba/baDevice.hpp` / `baDevice.cu` (CUDA dialect, HIP through the
compat header, `--fmad=false` on CUDA). It is the default, for the solves of at least 200,000 rows
(`CHESHIRE_BA_DEVICE_MIN_ROWS`; a million until the setup work below); `CHESHIRE_BA_DEVICE=0` keeps
everything on the host.

**What runs where.** The host keeps the trust-region loop, the vectors, Plus, the line search's
polynomials and the factorisation of the reduced camera system. The device holds the rows, the
Jacobian and the residuals, and runs:
- the evaluation: residuals, Jacobians, the manifolds' products, the loss and the corrector, the
  cost and the gradient;
- the column norms, the Jacobi scaling and the model cost change;
- the elimination per landmark (E^T E + D², its inverse, E^T b, E^T F, u = m^T inv), the assembly of
  the reduced system, and the back-substitution.

Per evaluation the host sends the landmarks' part of the state and three small tables: the poses'
rotations (their `hypot`, `sin`, `cos` are the host's), the cameras' values, and the manifolds'
Jacobians. It receives the partitions' costs and the gradient. `setState` writes only the camera blocks
while the device works; the landmarks go back at the end of the solve.

**The same bytes.** Every value comes from `baArith.hpp` or from ownSolver's loops restated operation
by operation, and every sum runs in the host's order:
- **Per row, per landmark, per partition:** one thread each, in order.
- **The F columns of the gradient and the norms:** one thread per (partition, camera block, column),
  over the partition's rows on the block.
- **The reduced system.** The host adds, per assembly group, landmark after landmark, each row's
  F^T F terms and then the landmark's Schur term. So an entry of a group's copy is a sum in a fixed
  order.
  - `kAssembleRec` follows that order, one thread per entry. It reads its (group, block)'s records,
    which the host builds once per solve.
  - The blocks inside one camera's intrinsics and distortion collect a term from every row of the
    camera: chains of up to a hundred thousand terms per group. A parallel pass writes their terms
    to a scratch array, one entry's terms contiguous, and a streaming pass adds them in order. A
    Schur term is stored negated, since x - y is x + (-y) exactly.
- **The groups' copies:** summed in group order.

`CHESHIRE_BA_DEVICE=check` solves every problem the device takes twice: first on the host from a copy
of every free block, then on the device from the same start. It compares the digests, iterations and
final costs. Results:
- 41 views: all 29 device solves the same, with the threshold at 1,000 rows. That held through
  every kernel revision.
- Engine bay: all 35 the same.
- The False Door's whole runs (device on its 29 solves of a million rows or more) give the host's
  digests: `a5cc84375086fdec` / `e05e2fa4df94376b`.

**Time on the False Door** (same build, back to back, the same output):

| | host | device (≥ 1 M rows) |
|---|---|---|
| SfM wall | 594.7 s | 543.6 s (-8.6 %) |
| bundle adjustment | 141.3 s | 93.0 s (-34 %) |

In the solves the device took (29, 73.5 s in all):

| phase | host (4b run) | device |
|---|---|---|
| evaluations | 29.7 s | 8.5 s |
| column norms, scaling, model cost | 18.3 s | 4.2 s |
| elimination | 19.1 s | 15.4 s |
| factorisation (host) | 11.2 s | 11.2 s |
| setup and the device's per-solve setup ("other") | 15.6 s | 27.9 s |

**What did not work**, on the way to `kAssembleRec`:
- **Walking each block pair's landmarks per entry:** latency chains of dependent loads (about 3 µs
  per row); 85-240 ms per elimination at 1-3 M rows.
- **Every term into a scratch array, then ordered folds:** far more memory traffic than the host.
  Scanning every row of a landmark for every block pair also cost k² R per landmark, which long tracks
  blow up.
- **One thread per row of a block, all columns in registers:** slower than one thread per entry.

**The device's FP64.** The RX 9070's double-precision rate is only two to three times this box's six
cores. What wins is memory bandwidth: the evaluation and the passes over the Jacobian are memory-bound.
The Schur terms are arithmetic, and the device is only level with the host there.

**Left for later:**
- **The per-solve setup:** done below (4c, the setup).
- **The vectors on the device** (4d).
- **The factorisation**, the largest host phase left.
- **The CUDA build** (house-pc), not compiled yet.

### 4c, the setup (2026-09-29)

Every solve builds its structure afresh: the solver's own setup (both paths), then, for the device,
the device's view, its tables and the uploads. On the False Door's largest solves (3.3-3.9 M rows),
the two together took about 1.2 s per solve, which is why the device started at a million rows. The
timers (`CHESHIRE_BA_PROFILE`: "BA own setup", "BA device setup"; `CHESHIRE_BA_DEVICE_PROFILE`:
"BA device create") found the time in serial loops, zero-filled arrays that are overwritten anyway,
hash lookups and repeated table lookups. Nothing it builds changed.

**The device's setup, 0.55-0.68 s → about 0.3 s:**
- the view of the rows and the row table in parallel, allocated without initialisation (the `Row`
  table had been constructing some 400 MB before the fill);
- the column lists by a counting sort per partition, placed in parallel;
- the records in 256 units (each group in eight ranges of landmarks), a pair's block looked up once
  and read back in the second pass;
- the rows' upload on a thread of its own, while the host builds the rest;
- the device memory kept across solves (`CHESHIRE_BA_DEVICE_POOL=0` to allocate per solve).

**The solver's own setup, 0.62-0.66 s → 0.33-0.36 s**, on both paths:
- the direct build gives each row's parameters as candidate positions, so no pointer hash table
  (0.14 s → 0.03 s);
- the per-parameter and per-row tables allocated without initialisation, the row table filled in
  parallel with its offsets in a prefix pass;
- the rows on constant blocks only evaluated in parallel, their costs added in row order. They are
  now evaluated after the direct source's per-pose and per-camera tables are filled: since 4b they
  would have read them empty. No set here has such rows, which is why no output changed.
- The groups' copies of the reduced system are no longer zero-filled at setup; each group zeroes its
  copy before use anyway.

The digests are the same on all three sets, the device check still 29 of 29, the direct check 68
of 68, and the Problem path without persistence still gives the direct path's bytes.

**The threshold.** Back-to-back False Door runs with the installed build on an idle box, all with
the same output (`a5cc84375086fdec`):

| device on solves of at least | SfM wall | bundle adjustment | solves ≥ 1 M rows | 100 k-1 M rows |
|---|---|---|---|---|
| never (host) | 541.4 s | 121.8 s | 105.1 s | 9.6 s |
| 1,000,000 rows | 483.6 s | 71.3 s | 55.2 s | 9.6 s |
| 200,000 rows (the default) | 483.1 s | 68.0 s | 54.4 s | 6.9 s |

- **Against the host:** bundle adjustment is 44 % shorter and SfM 11 %.
- **Against 4c before the setup work:** the large solves went from 73.5 s to 54.4 s.
- **The medium solves** now gain as well, 9.6 s → 6.9 s, so the default is 200,000 rows. The wall
  times of the last two runs are within this box's run-to-run drift.

A first set of these runs overlapped a video playing on the box. Their timings were discarded;
their output was the same.

## Step 4d: the vectors on the device (2026-09-29)

When the device takes a solve, the minimiser's vectors live there too: the state, the candidate,
the best point, the step, the LM diagonal, the Jacobi scaling, the gradient, and the line search's
sample. A state vector also keeps its camera part on the host, the "mirror": the host still reads
the camera blocks (their objects, the poses' rotations) and applies their manifolds. An iteration
now moves only the camera parts (a few thousand values), the reduced camera system and single
numbers between host and device. Before, it moved whole vectors of about 4 M values.

- **One minimiser for both.** `minimize()` and `lineSearch()` work on `Vec`, and every pass
  (`vop`, `vred`, `plus`, `setState`) runs on the host or on the device. The host path computes
  what it did before.
- **Plus.** A landmark's Plus is x + delta, on the device. Setup checks that no landmark has a
  manifold or bounds, and that its place is the same in the state and the tangent vector. The
  camera blocks are done on the host, from the state's mirror and the step's camera part, and
  their result goes up.
- **The elementwise passes** (`kVecOp`) are each the host loop's expression.
- **The sums keep the host's order.** The host's `psum` adds fixed blocks of 32,768 values, each in
  order from zero, then the blocks in order. The device takes the same blocks: one workgroup per
  block stages a tile in shared memory, and one thread adds it in order. A maximum does not depend
  on the order, so its blocks use a tree.
- **‖x‖ is kept across rejected steps,** since x does not change until a step is taken. Same value,
  one pass fewer.

**The first reduction was slow.** It used one thread per block, reading global memory. The loads
were not overlapped, so a reduction took about 10 ms, and the passes cost 7.8 s over the False Door.
Staged through shared memory, the same sums take 1.5 s.

**The factorisation and OpenMP's threads.** With the vectors gone from the host, the large solves'
factorisation took 15.7 s against 10.7 s. That is the same code on the same matrices. It showed
only on the device's solves, and only at the False Door's size (a memory-bound LDLT of up to about
0.3 s per call). What happened:
- Right before the single-threaded factorisation, a parallel pass copies S into the sparse matrix's
  order.
- After a parallel region, OpenMP's workers spin for `KMP_BLOCKTIME` (200 ms by default) before they
  sleep, so they were spinning beside the factorisation.
- The host path has the same pass, but its threads are busy all iteration.
- `KMP_BLOCKTIME=0` took the factorisation to 9.0 s. Which contention it is (SMT siblings, or
  cores unparked late) was not measured.

**The fix:** the device now hands S over already in the matrix's order (`kOrder`, `setOrder`), and
the host runs no parallel pass there. The environment variable is process-wide, so it is not used.
One observation for later: with `KMP_BLOCKTIME=0` the whole SfM run took 434 s, against 451 s for
the fix. The spinning may cost other stages of SfM as well.

**Exactness.** Nothing the solver computes changed:

| check | result |
|---|---|
| 41 views | `801d1fb7…` |
| engine bay | `814ae0d5…` |
| False Door | `a5cc8437…` |
| device check, 41 views | 29 / 29 |
| device check, engine bay | 35 / 35 (including line-search solves on the device) |
| direct check | 68 / 68 |
| host path alone (`CHESHIRE_BA_DEVICE=0`) | the same digests |

**Timing.** False Door, back to back on an idle box, both with the same output:

| | committed build (4c) | 4d |
|---|---|---|
| SfM wall | 467.2 s | 451.3 s |
| bundle adjustment | 67.5 s | 52.6 s |
| solves ≥ 1 M rows | 54.1 s | 39.8 s |
| of these: evaluations | 6.2 s | 4.3 s |
| of these: elimination | 9.8 s | 9.6 s |
| of these: factorisation | 10.7 s | 8.6 s |
| of these: back-substitution | 1.0 s | 0.4 s |
| of these: norms, scaling, model | 3.7 s | 2.0 s |
| of these: Plus | 4.6 s | 0.4 s |
| of these: vector passes | (in "other") | 1.5 s |
| of these: other (mostly the device's per-solve setup) | 12.0 s | 6.9 s |

Against the host alone (121.8 s, 4c's runs), bundle adjustment is now 57 % shorter.

**Next:**
- The factorisation (8.6 s) and the elimination (9.6 s) are the largest phases left.
- The per-solve setup is about 13 s: the solver's own 6.0 s, and most of "other".
- The CUDA build (house-pc) is still not compiled.

### The other builds and cards (2026-09-29)

The solver had only been built by the Windows HIP toolchain (clang-cl, ROCm 7.2) and run on the RX 9070.
Every other build now compiles it:
- **HIP 6.2 (RDNA1/2):** its older clang builds `baDevice.cu` unchanged.
- **Linux CUDA:** GCC builds the host code, and nvcc (sm_61) builds `baDevice.cu` with `--fmad=false`,
  step 7f.
- **Windows CUDA:** MSVC and nvcc build the solver and incrementalSfM.

The Linux build targets a generic x86-64 without AVX or FMA, so GCC cannot fuse a multiply-add there.
MSVC does not fuse without `/fp:contract`.

`scripts/windows/sfm-exactness.ps1` and `scripts/linux/sfm-exactness.sh` run the checks on a test box.
They use the 41-view and engine-bay inputs copied over, with colours off (the images stay behind).
Each set runs six ways:
- the host alone, twice;
- the device from 1,000 rows;
- the device check;
- the direct check;
- the defaults.

| build, card | 41 views | engine bay | device check | direct check |
|---|---|---|---|---|
| Windows ROCm 7.2 gfx12, RX 9070 | `12668473…` ×6 | `c5d68cd8…` ×6 | 29/29, 35/35 | 68/68, 140/140 |
| Windows HIP 6.2 gfx1012, RX 5500 XT | `e347b791…` ×6 | `5b6d80ba…` ×6 | 29/29, 34/34 | 68/68, 140/140 |
| Linux CUDA sm_61 (GCC), GTX 1080 Ti | `e7712237…` ×6 | `9668ad9a…` ×6 | 29/29, 36/36 | 68/68, 141/141 |
| Linux HIP (house-pc after a card swap), RX 6750 XT | `e7712237…` ×6 | `9668ad9a…` ×6 | 29/29, 36/36 | 68/68, 141/141 |
| Windows CUDA (MSVC + nvcc; bench-pc after a swap), GTX 1080 Ti | `7259fcf6…` ×6 | `81387335…` ×6 | 29/29, 34/34 | 68/68, 140/140 |

The last two rows ran the same evening, at 859d0a8, which also carries SfM steps 7i-7m (docs/04). The
Windows HIP 6.2, Linux CUDA and Linux HIP builds were run both before and after those steps. Each
gave the same digests both times, so steps 7i-7m are exact on those builds too.

**The two Linux builds agree byte for byte**, although one runs its device work on an NVIDIA Pascal
card and the other on an AMD RDNA2 card. Their host code is built the same way, so the device path
reproduces the host's arithmetic exactly on both vendors.

**Exactness.** Within each build every run gives the same bytes: the host twice, the device, both
checks and the defaults, and every checked solve is the same on the device as on the host. The builds
do not agree with each other. The HIP 6.2 family is compiled with `/arch:AVX` and the gfx12 one with
AVX2, the Linux one without AVX and with glibc's maths, so Eigen's code and the trigonometry round
differently all through SfM. That is also why the number of solves the device check sees differs
by one or two between them. Byte identity is a property of a build, and these checks confirm it.

**Timing.** The device from 1,000 rows against the host alone:

| box | 41 views | engine bay |
|---|---|---|
| house-pc: 4-core Haswell, GTX 1080 Ti (Pascal, FP64 at 1/32) | 57.8 → 51.4 s | 47.5 → 37.7 s |
| bench-pc, RX 5500 XT | 107.4 → 105.1 s | 120.3 → 116.8 s |

With the defaults (200,000 rows), house-pc took 54.3 s and 41.0 s. On a slow host the device pays on
far smaller solves, so the threshold may want to follow the machine. The RX 6750 XT in house-pc
gives the same picture: engine bay 47.5 s → 34.6 s from 1,000 rows, 40.1 s with the defaults.

**Packaging (02cb3a9).** `aliceVision_sfm_bundle.dll` carries the target's code objects since step 7f.
The bundled Windows package files a GPU-bearing DLL under its target only. `build_targets.py` harvests
it now as the seventh per-target DLL; without that, every target but the two base packages' would
have had no `sfm_bundle.dll` at all.

## Scope

The solver covers the problems incremental SfM builds most:
- **Residuals:** 2-D projection residuals of the common kind, `cheshire::createProjectionCost(false, ...)`,
  i.e. no rig, no mesh point.
- **Loss:** Huber (4 px) optionally scaled per observation (`ScaledLoss`, `observation.getWeight()`).
- **Parameter blocks:** intrinsics (with AliceVision's `IntrinsicsManifold`), distortion, pose
  (6 doubles, with a `SubsetManifold` when part of it is constant) and landmark (3). Any block may be
  constant.
- **Solvers:** both AliceVision configurations. `DENSE_SCHUR` up to 100 poses, or up to 20 cameras
  under the local strategy. `SPARSE_SCHUR` above that, which in Cheshire's GPL-free Ceres is
  `EIGEN_SPARSE`: Eigen's `SimplicialLDLT`, after Ceres' own block AMD of the camera blocks.

Anything else in a problem (rig sub-poses, mesh-point projections, depth residuals, 2-D constraints,
rotation priors, temporal constraints, the focal prior) sends that solve to Ceres, as today.
Step 0 says how much of the time that leaves to Ceres.

## Ceres-faithful

The target is Ceres' own algorithm, specialised, so that the new solver takes the same iterates up to
rounding. Validation is then a comparison run by run, not an argument about quality. From the Ceres
2.2.0 source Cheshire builds, and AliceVision's options (`BundleAdjustmentCeres::setSolverOptions`):

- **Minimizer:** trust region, Levenberg-Marquardt, `max_num_iterations` 50,
  `max_num_consecutive_invalid_steps` 10, monotonic steps.
- **Tolerances:**
  - function tolerance 1e-6: |Δcost| ≤ 1e-6 · cost;
  - gradient tolerance 1e-10, on max_i |x − Plus(x, −g)|;
  - parameter tolerance 1e-8: |step| ≤ 1e-8 (|x| + 1e-8).
- **Jacobi scaling:** column scale s_i = 1 / (1 + sqrt((JᵀJ)_ii)), computed once, at iteration 0, and
  applied to every later Jacobian (`trust_region_minimizer.cc:261-276`). Steps are unscaled by the
  same vector.
- **LM diagonal:** D = sqrt(clamp(diag(JᵀJ), 1e-6, 1e32) / radius) on the scaled Jacobian; the
  diagonal is reused after a rejected step (`levenberg_marquardt_strategy.cc`).
- **Radius:**
  - starts at 1e4, capped at 1e16, minimum 1e-32;
  - on acceptance, radius /= max(1/3, 1 − (2ρ − 1)³) and the decrease factor resets to 2;
  - on rejection, radius /= factor, then factor *= 2.
  - A step is accepted when ρ = (actual decrease) / (model decrease) > 1e-3.
- **Huber through the corrector:** Huber's ρ'' is never positive, so Ceres' corrector reduces to
  scaling the residual and the Jacobian rows by sqrt(ρ'(s)), s = |r|²:
  - ρ' = 1 within a² = 16;
  - ρ' = a / sqrt(s) outside.

  `ScaledLoss` multiplies ρ and its derivatives by the observation's weight. The cost is
  ½ Σ ρ(s), as Ceres reports it.
- **The evaluation callback:** `PrepareForEvaluation` pushes the intrinsics and distortion blocks
  into AliceVision's intrinsic objects before each new evaluation point
  (`update_state_every_iteration` is on). The analytic cost reads the blocks directly, so the new
  solver only has to leave those objects as Ceres would at the end.
- **Schur elimination:** landmarks form the first elimination group (step 5n orders the blocks by
  key), and the reduced camera system over pose, intrinsics and distortion columns is
  S = B − E C⁻¹ Eᵀ, with the LM diagonal added before elimination as Ceres does.
  - **Dense:** Cholesky of the dense S.
  - **Sparse:** for `EIGEN_SPARSE`, Ceres first reorders the camera blocks itself
    (`ReorderSchurComplementColumnsUsingEigen`: AMD on the block pattern F^T F − F^T E E^T F), then
    factors with `SimplicialLDLT` in the natural order (`AreJacobianColumnsOrdered`). The solver does
    the same. Step 1 ran scalar AMD instead, and that was its one real parting (below).

Where exact agreement is not reachable (summation order inside S, the threading of the
landmark-block loop), the comparison falls back to relative tolerances per iteration. It then stops
at the first iteration where the two diverge by more than rounding, which is where a mistake shows.

## Shape: the solver walks the Problem AliceVision builds

The first version takes the `ceres::Problem` AliceVision has already built and walks it through
Ceres' public API. From that API it gets:
- per residual block: the cost function, the loss and the parameter blocks;
- per parameter block: its manifold and whether it is constant;
- from `_linearSolverOrdering`, which blocks are eliminated first (the landmarks).

It evaluates each residual block by calling the block's own `CostFunction::Evaluate`, the analytic
code of 5m-5q, with plain arrays. It then does itself what Ceres wraps around that call:
- the manifold products, at fixed sizes;
- the loss correction;
- the scatter, straight into the Schur structures.

This has three consequences:
- **Every residual kind is covered**, rigs, depth and constraints included, as long as each residual
  block touches at most one eliminated block. Projection residuals touch one landmark; the others
  touch none.
- **The numbers are Ceres' numbers**, because the same cost-function objects compute them.
- **The problem build is not removed yet.** It is B1's 145 s of the False Door's rebuilds, and goes
  when the solver builds its structures from the SfM data directly (step 3b, after it has proved
  itself; done, above).

Validation moves into the run: **shadow mode**. `CHESHIRE_BA_SHADOW=1` runs the new solver on a copy
of the parameters beside Ceres, in every solve of a real SfM run, and logs both trajectories. Ceres
still drives the reconstruction, so a shadow run's output is today's.

## Layout

- **Observations** are sorted by landmark. Each carries the indices of its intrinsics, distortion,
  pose and landmark blocks, the 2-D measurement and the weight.
- **Per iteration, per landmark:** the 3×3 block C_l = Σ E_iᵀE_i + D_l², the landmark's rows of the
  gradient, and its coupling blocks E_iᵀF_i for the cameras that observe it. These are independent
  across landmarks, the parallel (and device) part.
- **S is accumulated** from those, per pair of camera blocks that share a landmark. The pattern is
  fixed for a solve and built once; under persistence (5r's conditions) it is kept across solves.
- **Nothing is allocated per iteration** after the first.

The analytic Jacobian is the one 5m-5q already use (`hip/port/sfm_ba/projectionCheshire.hpp`), with
the same operations in the same order, so the new solver and Ceres evaluate the same numbers.

## Validation

1. **Shadow mode (step 1).** On every solve of an SfM run, the new solver starts from the same
   parameters as Ceres. Both iteration traces are logged (cost, step norm, ρ and radius per
   iteration; Ceres' through an `IterationCallback`), with the first iteration where they part, and
   the final parameters' largest relative difference. It runs on 41 views, the engine bay and the
   False Door, dense and sparse, and with one Ceres thread where summation order must match.
2. **Iterate comparison (step 2).** The same, as the solver grows: dense Schur first, then sparse.
3. **In SfM (step 3).** The SfM quality gate (`scripts/quality_gate.py sfm`) on 41 views, the
   engine bay and the False Door against today's defaults, then time. With
   `CHESHIRE_SFM_DETERMINISTIC=1` both are reproducible, and the comparison can also be byte for
   byte, as far as step 2 reached.

## Steps and gates

| step | what | go if |
|---|---|---|
| 0 | per-solve profile of the engine bay and the False Door | the solves the scope covers hold most of the BA time |
| 1 | shadow mode: the solver beside Ceres on every solve, traces logged | the harness itself is exact: a copy of the parameters, Ceres' output unchanged |
| 2 | host solver, dense then sparse | iterates match Ceres to rounding, and time per solve beats Ceres |
| 3 | `CHESHIRE_BA_SOLVER=cheshire` in SfM | quality gate passes on 41 / engine bay / False Door; faster |
| 3b | the solve built from the SfM data, no Problem | same bytes as the Problem path on every solve; faster |
| 4 | residuals, Jacobians and landmark blocks on the device | faster than the host solver on the large solves |

## Risks

- **Faithfulness in the details:** Jacobi scaling's first-iteration vector, the reuse of the diagonal
  after a rejected step, and the gradient-tolerance test through the manifolds. Each is a place
  where "almost Ceres" diverges after a few iterations. The replay bench shows it at once.
- **The sparse ordering:** Ceres orders S by its own block order and AMD. Matching needs the same
  block order in the matrix handed to Eigen.
- **Scope creep:** rigs and constraints are real in some pipelines. They stay on Ceres until the
  common case has shipped.
