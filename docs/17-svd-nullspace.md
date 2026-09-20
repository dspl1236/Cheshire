# The 7-point solver's nullspace (0.3.0)

docs/15 left `kernel.fit` as the largest single phase anywhere in the pipeline: **1589.4
thread-seconds, 39 % of FeatureMatching, 111.7 M calls**. It is `Nullspace2` in
[`numeric/algebra.hpp`](../third_party/aliceVision/src/aliceVision/numeric/algebra.hpp), running
Eigen's `JacobiSVD` with `ComputeFullV` on a 9x9.

This is the first change in this project that is **not bit-identical to upstream**, which is why it
is a minor bump rather than a patch release, and why most of this document is about correctness
rather than speed.

## The matrix is a 7x9 wearing a 9x9's clothes

`encodeEpipolarEquation` writes one row per correspondence, and the minimal case has seven:

```cpp
Mat9 A = Mat::Zero(9, 9);
encodeEpipolarEquation(x1, x2, &A);   // fills rows 0..6; rows 7 and 8 stay zero
Nullspace2(A, f1, f2);
```

So upstream runs a full iterative 9x9 SVD to find a two-dimensional nullspace in a 7x9 system.
Householder-QR the 9x7 transpose instead: the last two columns of the 9x9 `Q` are orthonormal and
orthogonal to every row of `A`, so they are a basis of exactly that nullspace, from one
non-iterative factorisation.

Two things that would have been bit-identical are already upstream, which is worth knowing before
looking for a free lunch: `Nullspace2` asks for `ComputeFullV` alone, so there is no `U` to skip,
and the minimal solver already hands it a fixed-size `Mat9` with a comment saying that is
deliberate.

## Why a different basis is not a different answer

The solver does not use `f1` and `f2` directly. It solves `det(F1 + a*F2) = 0` over the **pencil**
they span, and a pencil is a property of the subspace, not of the basis chosen for it. Any basis of
the same nullspace gives the same set of fundamental matrices, up to rounding.

That is the claim; the rest is measuring it.

## The control is what makes the numbers readable

Comparing QR against upstream on its own says nothing without a scale. The scale used throughout is
**`JacobiSVD` on the 7x9 block against `JacobiSVD` on the 9x9** - the same algorithm, on
mathematically identical input, differing only in that one is not fed two rows of zeros. Whatever
that control shows is the noise this computation already has.

Measured over synthetic systems with upstream's own cubic coefficients and root solver transcribed
from `Fundamental7PSolver.cpp` and `numeric/polynomial.hpp`:

| data | nullspace | whole fit | QR vs upstream | control vs upstream |
|---|---|---|---|---|
| general position | 7.6x | 6.75x | 3.96e-09, no root-count flips | 3.97e-09, none |
| near-degenerate | 7.5x | 6.66x | 1.183e-04 | 1.176e-04 |
| exactly rank deficient | 3.2x | 2.54x | arbitrary | equally arbitrary |

**QR is as close to upstream as upstream is to a trivial reformulation of itself.**

### The degenerate rows need reading carefully

On near-degenerate samples - seven nearly collinear points, clustered, small parallax, which is what
RANSAC actually draws - there are 275 root-count flips in 50,000 systems. That looks alarming until
the breakdown:

    NaN from some method (upstream included)   2
    root count flips: both QR and control      274
                      QR only                  1
                      control only             1
    stable systems                             49,722
      worst model difference, QR         1.183e-04
      worst model difference, control    1.176e-04

The same 274 systems flip for both, with one unique to each side. These sit near where the cubic's
discriminant changes sign, so the root count turns on rounding - the instability belongs to the
conditioning of the sample, not to the factorisation. Upstream produces NaN on two of them by
itself.

On **exactly** rank-deficient samples (three distinct points repeated to fill seven rows) the
subspace gap reaches 1.0 - completely different answers - and the worst model difference is 1.703
for QR **and 1.703 for the control**. With rank 3 the nullspace is six-dimensional, so "the last two
columns" is an arbitrary two-dimensional slice of it, for every method. Two different SVDs disagree
with each other just as violently. The model is meaningless whichever way it is computed, and
RANSAC discards it on inlier count.

This is the honest limit of the synthetic work: it shows QR is not worse, not that either is right.

## End to end, which is what actually decides it

The synthetic work shows QR is not worse than upstream. It cannot show that the pipeline is
unaffected, because AC-RANSAC drives 111.7 M of these solves and keeps whatever inliers they
produce. So: the real FeatureMatching stage, both legs from the **same binary**, differing only in
`CHESHIRE_SVD_NULLSPACE`.

**Geometric filtering: 19.9 s -> 10.7 s, 1.86x.** (That is the phase `Task done in` reports, which
is where the nullspace lives. It is far below the 351 s docs/15 records for AC-RANSAC on this data
set and the gap is not explained here, so treat the absolute figure as this configuration's, not
the stage's.)

The match file changes:

| | pairs | matches |
|---|---|---|
| upstream | 904 | 502,310 |
| QR | 910 | 502,523 |

657 of the 893 shared pairs differ, and **2.04 % of correspondences change**. That is a real
difference, not jitter: running upstream a second time reproduces its match file exactly, 0 of
502,310 correspondences different, so FeatureMatching is deterministic and the 2 % is attributable
to the change.

### Does it help or hurt the reconstruction?

SfM is not deterministic - `incrementalSfM` drifts because Ceres splits work across
`omp_get_max_threads()` - so a single comparison reads that jitter as an effect. **Ten**
reconstructions per leg, same match folder each time:

| | landmarks | RMSE | poses |
|---|---|---|---|
| upstream | 141,118 ± 131 | 1.68407 ± 0.00137 | 107 every run |
| QR | 140,953 ± 84 | 1.68650 ± 0.00331 | 107 every run |

Welch's t-test, which does not assume equal variance (QR's RMSE spread is more than twice
upstream's):

| | delta | t | p |
|---|---|---|---|
| landmarks | **-165** | -3.36 | **0.0008** |
| RMSE | **+0.0024** | +2.14 | **0.032** |

**Both are real.** QR loses about 0.12 % of landmarks and adds about 0.14 % to reprojection error.

Three runs per leg said the opposite - the landmark difference looked like it sat inside upstream's
own spread, and this document said so before the sweep was run. It did not: at n=3 the means were
141,081 and 140,975 with a 160-landmark range, which is exactly the situation where a spread and a
difference of similar size are indistinguishable without more samples. **A difference that is
"within the noise" at n=3 can be the strongest signal in the experiment at n=10**, and a 20-minute
sweep is cheap next to shipping the wrong conclusion.

## So it is off by default

1.86x on geometric filtering against 0.12 % of landmarks and 0.14 % of reprojection error is a
trade, not a free win, and this project's entire claim is that nothing it produces changes. Turning
that off silently for a speedup on one phase would be the wrong default, so upstream's `JacobiSVD`
stays the default and the QR path is opt-in:

    CHESHIRE_QR_NULLSPACE=1

For scale: the difference is far smaller than the gap between GPU and CPU SIFT (88,463 landmarks
against 78,372 on the 41-view set, 13 %), which is accepted as an improvement. Someone matching
thousands of pairs may well want the 1.86x and never notice 0.12 %. The point is that it should be
their decision, made against a measured number, rather than one taken on their behalf.

## What ships

`Nullspace2RankDeficient` in `algebra.hpp`, used by the **minimal branch only** of
`Fundamental7PSolver::solve`. The over-determined branch keeps the SVD, because there `Nullspace2`
is computing a least-squares fit that a QR nullspace does not reproduce, and the spherical solver is
left alone because there is no hardware here to validate it against.

`CHESHIRE_SVD_NULLSPACE=1` restores upstream's `JacobiSVD`. The flag is read through a
function-local static: `solve()` runs 111.7 M times and a `getenv` per call would cost more than
the change saves.
