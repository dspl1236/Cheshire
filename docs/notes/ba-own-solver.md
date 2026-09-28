# A bundle-adjustment solver of Cheshire's own: design (0.3.7)

Status: design, 2026-09-28. Step 0 done (below); nothing else is built. Background and the measurements
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
  `EIGEN_SPARSE`: Eigen's `SimplicialLDLT` with AMD ordering.

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
  - **Sparse:** `SimplicialLDLT` with `AMDOrdering`, the same Eigen class Ceres calls, on S
    assembled in Ceres' block order.

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
