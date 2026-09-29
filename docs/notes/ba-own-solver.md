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
  itself).

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
| 4 | residuals, Jacobians and landmark blocks on the device | faster than the host solver on the large solves |

## Risks

- **Faithfulness in the details:** Jacobi scaling's first-iteration vector, the reuse of the diagonal
  after a rejected step, and the gradient-tolerance test through the manifolds. Each is a place
  where "almost Ceres" diverges after a few iterations. The replay bench shows it at once.
- **The sparse ordering:** Ceres orders S by its own block order and AMD. Matching needs the same
  block order in the matrix handed to Eigen.
- **Scope creep:** rigs and constraints are real in some pipelines. They stay on Ceres until the
  common case has shipped.
