# Geometric verification: profile first, and the GPU was the wrong answer

FeatureMatching on the 107-photo engine bay took 644 s. The GPU 2-NN matcher
([docs/07](07-gpu-matcher.md)) accounted for 78 s of that; the other 566 s was geometric
verification - AC-RANSAC fitting a fundamental matrix per image pair on the CPU. That made it the
largest single cost left in the pipeline outside Meshing, and the obvious next port.

It is now 1.6x faster and none of that came from the GPU. The match file is byte for byte the one
the unmodified build produces.

| | upstream | now |
|---|---|---|
| geometric filtering, wall clock | **565.8 s** | **351.0 s** |
| the same, with the instrumentation below compiled in | 597.3 s | 342.2 s |
| `std::sort` | 3437.5 s | 1034.6 s |
| `kernel.fit` (7-point solver) | 1839.0 s | 1589.4 s |
| `bestNFA` | 988.8 s | 900.0 s |
| `kernel.errors` | 594.4 s | 382.8 s |
| ErrorIndex fill | 64.2 s | 52.3 s |
| pair sort, rare path | - | 1.2 s |
| total CPU time, 12 threads | 7137.4 s | 4088.9 s |

Phase times are summed across 12 OpenMP threads and come from a temporary instrumentation pass, so
they are for attribution rather than headline timing; the second wall-clock row carries the same
instrumentation, which makes before and after comparable. The headline row is the shipping build.
Run-to-run spread is a few percent either way.

## What the profile said, and what it corrected

Reading the code first produced three plausible suspects, and the ranking was wrong on every one.

`FundamentalEpipolarDistanceError::error()` is virtual and `PointFittingKernel::errors()` calls it
once per correspondence, so per-element virtual dispatch on a ~15-flop functor looked like the
obvious culprit. It is 8.3 % of the stage. A perfect fix for it caps out at an 8 % win.

`bestNFA` evaluates `log10` per element - 13.9 %.

`std::sort`, which nobody suspected, is **48.2 %**.

Two further numbers came out of a second pass and both mattered more than they look:

- **2.49 sorts per iteration.** The 7-point solver returns up to three real roots and each model
  gets its own residuals, sort and NFA scan. 111,738,442 iterations produce 278,454,716 sorts.
- **The arrays are small.** `meanNData` is averaged over calls and reads 341, but weighted by
  iteration it is **246** - pairs with many correspondences find a model quickly and iterate less.
  The largest single pair is 5,707.

A 245-element introsort should take 2-4 us. It takes 12.3 us, which is about 125 cycles per
element for a sort that touches each element around eight times. That is branch misprediction in
the partition loop, on an unpredictable comparator over `std::pair<double,size_t>`.

## Sorting keys instead of pairs

The loop builds a `std::pair<double,size_t>` array per candidate model and sorts it. `bestNFA`
reads `e[k-1].first` - the values alone. The indices are needed only in the branch that fires when
a model improves on `minNFA`, to fill `vec_inliers`, and that branch is **72,381 of 278,454,716
sorts, 0.026 %**.

That ordering is not incidental and cannot simply be dropped: `vec_inliers` becomes the sampling
pool for later iterations and `uniformSample` draws *positions* out of it, so reordering it changes
which samples get picked and therefore changes the result. But it only has to be right 0.026 % of
the time, and on that path the original pair sort runs unchanged.

Everywhere else the sort is over bare 8-byte keys, and it is a stable LSD radix over the double's
bit pattern rather than a comparison sort. A non-negative double orders the same as its bit pattern
read as a `uint64_t`, and squared epipolar distances are non-negative. Byte positions where every
key agrees are skipped, which covers most of the exponent. A key with the sign bit set means a
negative value or `-0.0`, where the equivalence fails, and falls back to `std::sort`.

Sorting values alone gives the same value sequence as sorting pairs, because pair ordering breaks
ties by index and tied values are equal by definition. That is an argument, not a measurement, so
it was measured: 20,000 residual arrays were captured out of a real run
(`scratchpad/sortbench.cpp` benchmarks against them rather than against a guessed distribution),
4,724,846 values, no negatives, no NaN, no infinities, 3,038 exact zeros and 317 exact ties.

| | us per sort | | |
|---|---|---|---|
| `std::sort` on `pair<double,size_t>` - what ships | 8.89 | 1.00x | bit-identical |
| `std::sort` on `double` | 7.04 | 1.26x | bit-identical |
| stable LSD radix on the bit pattern | 2.55 | 3.49x | bit-identical |
| `bit_cast` to `uint64_t` keys, then radix - the form used | 2.76 | 3.22x | bit-identical |

In the pipeline the sort phase went 3437.5 s to 1034.6 s, a measured 3.32x against the 3.22x the
benchmark predicted. The rare pair sort costs 1.2 s in total.

One case is handed back to `std::sort`: a key above `0x7FF0000000000000`, which covers negative
values, `-0.0` and NaN (`+inf` is exactly equal to it and orders correctly). It is one comparison
folded into the histogram pass, and NaN is the reason it exists.

A NaN breaks the strict weak ordering `std::sort` requires, so upstream's order on such an array is
unspecified - but unspecified by upstream is still what upstream produces, and reproducing it is the
whole point. The radix would instead put NaN at the end, deterministically. That is arguably better
and definitely different.

This is not hypothetical, and it is why geometric filtering and SfM behaved differently. Geometric
filtering uses `RelativePoseKernel` and never produces one: 4.7 M residuals captured from a real
run contain no NaN, no infinity and no negative value, and the deferral counter stays at 0 across a
whole engine bay job. SfM's relative pose uses `RelativePoseKernel_K`, which builds F from E and can
divide by a zero `squaredNorm`; a 41-view reconstruction defers 24 to 44 arrays. Without the guard
those arrays came out ordered differently from upstream.

`CHESHIRE_ACR_PAIRSORT=1` restores upstream's per-model pair sort.

## The residual loop

`PointFittingKernel::errors()` loops calling the virtual `error()`, which stops the compiler
inlining the functor and stops it vectorising. `RelativePoseKernel` - the class the fundamental
matrix filter actually instantiates - does not override `error()`, so calling the estimator
directly is the same arithmetic in the same order. It is worth 558 s to 383 s, a 1.46x on that
phase and 8 % on the stage.

`RelativePoseKernel_K`, the essential-matrix sibling, *does* override it, and recomputes
`fundamentalFromEssential` per correspondence although the result is the same for every point in
the call. That is not the path FeatureMatching takes by default and is untouched here, but it is a
much larger factor for anyone who does take it.

## Evidence the search is unchanged

Three counters came out identical before and after: 111,738,442 iterations, 278,454,716 sorts, and
72,381 models improving on `minNFA`. The algorithm walked the same path and made the same decisions
at the same points, which is stronger than matching endpoints. The output file agrees byte for byte
with the reference produced by the unmodified build:

```
cmp build/bay-acrfinal/0.matches.txt build/bay-sift/match/0.matches.txt   # 16,730,688 bytes
```

The radix is integer code and the residual loop is ordinary floating point, so the risk is the
compiler and the instruction set rather than the GPU. Both builds were run against **one** feature
set on each host and the output diffed:

| host | compiler, instruction set | matching before | after | matches file |
|---|---|---|---|---|
| RX 9070, Ryzen 5600X, Windows | clang-cl, AVX2 | 565.8 s | 351.0 s | identical (107 photos and 41 views) |
| RX 5500 XT, FX-8120, Windows | clang-cl, AVX only | 356 s | 290 s | identical |
| RX 5500 XT, i3-4330, Linux | GCC 13.3 | 371 s | 280 s | identical |

The FX-8120 is the reason the `hip6.2` packages are built `/arch:AVX`: it is a Bulldozer part with
AVX and no AVX2, so it is the only host in the fleet that exercises that build. The gain is larger
on the newer processor, which is what a branch-misprediction story predicts - a deeper pipeline
pays more for each mispredicted comparison.

## Nothing downstream of matching can check this

Two stages either side of FeatureMatching are not reproducible, and both produced a false alarm
during this work before being pinned down.

**Feature extraction.** Two runs on one machine with one binary give identical descriptor counts and
41 of 41 differing `.feat` files. Sorting the lines of one file and comparing shows why: 23,781
lines against 23,781, same lines in a different order. PopSift allocates keypoint slots with
atomics, so the ordering follows warp scheduling. Any comparison that re-extracts features on each
side is therefore comparing two different inputs - which is what made an early bench-pc run look
like a regression.

**incrementalSfM**, measured below.

## incrementalSfM cannot validate this, and that is worth knowing on its own

The obvious end-to-end check is the landmark count, and it does not work. Running the **same binary
on the same inputs** four to five times:

| build | landmark counts | spread |
|---|---|---|
| pre-change (v0.2.13) | 88,463, 88,468, 88,469, 88,472 | 9 |
| radix, no NaN guard | 88,471, 88,478, 88,479, 88,479, 88,484 | 13 |
| radix + NaN guard - what ships | 88,465, 88,479, 88,473 | 14 |

`randomSeed` is fixed at 5489, so this is not the RNG.
[`BundleAdjustmentCeres.hpp:40`](../third_party/aliceVision/src/aliceVision/sfm/bundle/BundleAdjustmentCeres.hpp)
sets `nbThreads(multithreaded ? omp_get_max_threads() : 1)`, and multi-threaded Schur elimination
accumulates in a thread-dependent order. That is the probable mechanism; it has not been isolated by
rerunning with one thread.

The practical consequences:

- A landmark count cannot support an exactness claim at single-landmark resolution. The exactness
  evidence here is FeatureMatching's output, which **is** deterministic and **is** byte-identical on
  both the 107-photo engine bay and the 41-view set.
- The first comparison that looked like a regression - 88,478 against a remembered 88,463 - was
  noise. It took a control, the *pre-change* binary rerun four times, to see that it reaches 88,463
  and 88,472 on its own.
- Two of the three numbers here only separated once the control had four samples rather than three.
  An earlier reading of the same data, on fewer samples, said the ranges did not overlap and the
  shift was systematic. That was overstated.

## A dead end worth recording: pruning the sort away

The sort exists to feed `bestNFA`, which minimises

```
nfa(k) = loge0 + bracket(k)*(k - s) + logc_n[k] + logc_k[k]
bracket(k) = logalpha0 + dim*log10(sqrt(e(k)) + eps)
```

`logc_n[]` and `logc_k[]` are precomputed from `n` and `sizeSample` alone and do not depend on the
residuals, and `e(k)` is sorted ascending so `bracket(k)` is non-decreasing. For any cut `K` and any
`k >= K`, `bracket(k)*(k-s) >= bracket(K)*(k-s)` whatever the sign, so

```
nfa(k) >= L(k) := loge0 + bracket(K)*(k - s) + logc_n[k] + logc_k[k]
```

and `min L(k)` over the tail costs one multiply-add per element, no sorting and no `log10`. If that
bound beats the exact best over the prefix, the tail never needs sorting: `nth_element` to the K
smallest, sort only those, done - exactly.

It does not work. Evaluated against the captured corpus with the real parameters for this set
(4032x2268 images, so `N2(0,0) = 1/sqrt(w*h) = 1/3024` and `logalpha0 = 0.485664`, point-to-line so
`dim = 1`, 7-point solver so `s = 7` and at most 3 models):

```
argmin k: median 8, mean 13, p90 14, p99 74, max 211   (mean n = 236)

cut K   tail bound proves the prefix answer
    16    0.00 %
    64    0.00 %
   128    4.57 %
```

The answer is always in a tiny prefix - the NFA minimum sits at k of 8 to 14 in 90 % of cases - but
the bound cannot prove it. `bracket(K)*(k-s)` is linear in k and strongly negative once `e(K)` is
small, while `logc_n[k] + logc_k[k]` grows only sub-linearly, so the bound runs off to minus
infinity down the tail and certifies nothing. Tightening it needs a lower bound on `e(k)` for every
k past the cut, which is the sorted order - the thing being avoided.

## The question the caller asks (0.3.8, step 8d)

The dead end above tried to prove where the minimum is. `ACRANSAC` never asks that. It does one
thing with `bestNFA`'s answer:

```
if (best.first < minNFA)   // a better model: keep it and its inliers
```

A model that loses changes nothing, and almost every model loses. At Meshroom 2025.1's 50,000
iterations on 41 views, 17 of 47,310 sampled models beat the running `minNFA` (0.036 %). For the
rest it is enough to prove that no `k` goes below `minNFA`. That is a much weaker claim, and it does
not need the sorted order.

A histogram of the unsorted residuals gives it. Each squared residual falls in a bucket a quarter of
a binade wide over [2^-54, 2^36), with one bucket below it (lower edge 0) and one from 2^36 up; the
bucket is the key's top 14 bits, shifted and clamped. After counting, rank `k` of the sorted order
lies in a known bucket, so `e(k)` is at least that bucket's lower edge, and since `nfa(k)` rises
with `e(k)` (`dim > 0`, `k > s`):

```
nfa(k) >= lb(k) := loge0 + (logalpha0 + dim*G[b])*(k - s) + logc_n[k] + logc_k[k]
G[b] = log10(sqrt(lower edge of b) + FLT_EPSILON) - 1e-12      (a static table)
```

Unlike the tail bound, every rank gets its own floor, so the bound does not run off down the tail.
Within one bucket, `lb(k)` is `a*(k - s)` plus `log10 C(n,k) + log10 C(k,s)`, which is concave in
`k`. Its minimum over the bucket's ranks is therefore at the first or the last one: two evaluations
per bucket, with no `log10` and no sort. The float tables are not exactly concave. Once per
`ACRANSAC` call, the setup measures how far they are from the exact sums (at most 0.0073 on these
calls) and adds twice that to the slack, beside a margin for the rounding of both expressions.

If every bucket's bound clears `minNFA` plus the slack, `bestNFA` could not have returned a better
model, and the model skips the sort and the scan. Everything else takes the old path: a call with no
model yet (`minNFA` infinite), a negative or NaN residual, a kernel with `dim <= 0`, or a bound that
does not clear. So the only models that change path are ones `bestNFA` would have discarded.

The 47,310 models were captured from a real run (every 499th on the key path) and replayed by
`hip/tests/acrbound`:

| | per model |
|---|---|
| radix sort + `bestNFA` | 5.42 us |
| the bound, then the sort and the scan where it does not settle | 0.59 us |

The bound settles 99.67 % of the models (99.84 % of their residuals), and `bestNFA` puts none of
them below `minNFA`. Of the 0.59 us, the histogram pass is 0.25 us and the walk over the buckets
0.34 us; a branchless walk and split histograms were no faster. At Meshroom 2023.3's 2048
iterations the share is 99.0 %: more of those models come early, while `minNFA` is still high.

Over the whole run, `CHESHIRE_ACR_BOUND_CHECK=1` sorts and scans every skipped model as well:
23,526,805 of 23,607,436 skipped, none below `minNFA`. Geometric filtering goes from 36.8 s to
18.8 s, and the match file is byte for byte the same ([docs/04](04-validation.md), 0.3.8).

## The residual loop, vectorised (0.3.8, step 8e)

With the sort and the scan gone, `kernel.errors` was the second phase at 44.9 CPU-seconds. Step 4x
had already let the compiler inline `FundamentalEpipolarDistanceError::error`, but the loop still
did not vectorise. Each call builds two `Vec3` from the columns, multiplies by `F` through Eigen,
and takes a dot product and a norm: 5.7 ns a residual in the run.

Vectorising it must not change a bit, so the new loop repeats Eigen's arithmetic operation for
operation. Here is how Eigen 3.4 evaluates it for AliceVision's aligned fixed sizes, read off
clang-cl's code and checked below:

```
F x, rows 0-1 (one packet): F(.,0)*a0, then + F(.,1)*a1 as a pmadd (an FMA where Eigen uses FMA),
                            then + F(.,2)*1
F x, row 2 (not vectorised): F(2,0)*a0 + (F(2,1)*a1 + F(2,2)*1)
dot with (b0, b1, 1):        (fx0*b0 + fx1*b1) + fx2
squared norm of the head:    fx0*fx0 + fx1*fx1
```

The loop reads the points straight from the 2 x n matrices and gives every rounding its own
statement (clang contracts only within one), so the compiler fuses nothing that Eigen does not. Where
Eigen does fuse, the loop uses `std::fma` itself. `hip/tests/epipolar` compares the two bit for bit
on random, tiny and huge, zero and degenerate inputs:

| build | residuals | differ | per residual |
|---|---|---|---|
| clang-cl `/arch:AVX2` (the Windows packages; Eigen's FMA path) | 6,150,602 | 0 | 2.63 -> 0.70 ns |
| clang-cl `/arch:AVX` (the path without FMA) | 6,163,449 | 0 | 2.63 -> 1.19 ns |
| GCC 15 `-march=core2` (the Linux packages' `TARGET_ARCHITECTURE=core`) | 6,142,901 | 0 | 2.83 -> 1.50 ns |

Other flags, another Eigen, or another alignment choice could change Eigen's side. So on first use a
self-test runs the same comparison on fixed pseudo-random inputs. If one bit differs, the kernels
keep the per-point loop and the log says so. The loop covers every kernel that computes this
distance: the F and E geometric filters and SfM's initial pair. `CHESHIRE_ACR_RESIDUALS=0` turns it
off. `CHESHIRE_ACR_RESIDUALS_CHECK=1` computes both, keeps `error()`'s, and counts the residuals that
differ: on 41 views at 50,000 iterations, 7,861,787,651 of 7,861,787,651 are identical. Geometric
filtering goes from 18.9 s to 16.35 s, with the same match file and the same SfM digests
([docs/04](04-validation.md), 0.3.8).

## Why this did not go to the GPU

The work is per image pair and already parallel across pairs, so the shape would be one workgroup
per pair. What rules it out is that iterations are **not** independent: once a meaningful model
appears, `vec_index = vec_inliers` switches the sampling pool, so hypotheses cannot be batched.
The only parallelism is across the 5,671 pairs.

That sets a hard budget. 111,738,442 iterations, and to finish in 60 s with `B` workgroups in
flight each iteration must complete in `60*B / 111.7M`. Sorting 8-byte keys needs `8n` bytes of LDS,
so n=245 fits in 2 KB and a CU could hold many workgroups - but a 256-element bitonic sort is 36
barrier-separated compare-exchange stages, and `kernel.fit` is a `JacobiSVD` on a 9x9, which is the
least wavefront-friendly thing in the loop and is now 39 % of what is left. A serious attempt looks
like 2-3x for a large amount of device code, against the 1.6x that two functions bought.

The profile is what settled it, and it settled it twice: it killed the per-element-virtual-dispatch
theory that a code read had produced, and it killed the port that the stage's size had seemed to
justify. Same lesson as Meshing ([docs/11](11-meshing-cpu.md)) - measure the blocks, not the
algorithm's reputation.

## What is left

After steps 8d and 8e, on 41 views at Meshroom 2025.1's 50,000 iterations (CPU time summed over 12
threads, from a temporary instrumentation pass; 23,607,436 models, 333 residuals each):

| phase | after 8d | after 8e | share |
|---|---|---|---|
| `kernel.fit` (7-point solver) | 143.7 s | 146.5 s | 74.8 % |
| the bound (8d) | 23.2 s | 23.2 s | 11.9 % |
| `kernel.errors` | 44.9 s | 11.9 s | 6.1 % |
| sampling | 8.8 s | 8.5 s | 4.4 % |
| sort + `bestNFA`, 80,631 models | 0.6 s | 0.6 s | 0.3 % |
| the better-model branch | 0.5 s | 0.5 s | 0.2 % |
| each iteration, all told | 226.3 s | 196.0 s | |

What exact work can still reach is a quarter of the stage: the bound, the residuals and the
sampling.

On the engine bay before 0.3.8, `kernel.fit` was already the largest phase at 1589.4 s, 39 % of the
stage. It is
`Nullspace2` in [`numeric/algebra.hpp`](../third_party/aliceVision/src/aliceVision/numeric/algebra.hpp),
which runs `Eigen::JacobiSVD<Mat9>` with `ComputeFullV` - Eigen's slowest SVD - on a 9x9 matrix
whose bottom two rows are zero, 111.7 M times, to extract a two-dimensional nullspace from what is
really a 7x9 system. A QR or eigenvalue-based nullspace would be several times faster.

Every change above is bit-identical to upstream. That one cannot be: a different factorisation
gives a different basis for the same nullspace, the cubic `det(F1 + a*F2) = 0` is then solved in a
different parametrisation, and the fundamental matrices differ in their last bits. It belongs in a
different bucket from the rest and is not done here.
