# Draft: follow-up under our reply in alicevision discussion #2116 (the SPQR numbers)

*Posted 2026-09-22: https://github.com/alicevision/AliceVision/discussions/2116#discussioncomment-18557754
(a reply under ours, reflowed). Approved the same day ("post the follow-up"). The reply above it promised the numbers "if the numbers
are interesting"; figures from docs/04 (the Ceres A/B) and build/sfm-ba-old, build/sfm-ba-new.*

---

Following up on the SPQR paragraph above, with the numbers I said I would post.

v0.3.3, released today, builds the Windows packages against Ceres 2.2.0 without SuiteSparse: Eigen's
sparse backend with METIS ordering, LAPACK from the tree's OpenBLAS, and the same compiler, Eigen,
glog and gflags as the prebuilt vcpkg tree. `ceres.dll` now imports glog, METIS and LAPACK only, and
the packagers refuse a package that carries `libspqr.dll` or `libcholmod.dll`.

The cost, measured: two replays of incremental SfM on the 884-view False Door set, on the same cached
features and matches, with upstream's default local bundle adjustment (and the fix from
alicevision/Meshroom#2344, since the stock binary does not finish this set with local BA on), on an
otherwise idle Ryzen 5 5600X:

| Ceres | SfM wall | BA solves | BA total | Jacobians | linear solver | poses / landmarks |
|---|---|---|---|---|---|---|
| SuiteSparse (CHOLMOD/SPQR) | 1944 s | 938 | 1069 s | 597 s | 323 s | 815 / 1,342,489 |
| Eigen sparse + METIS | 1498 s | 904 | 733 s | 421 s | 304 s | 831 / 1,354,886 |

The linear solver, the only part the sparse backend touches, takes 304 s against 323 s: no penalty
at this size. The rest of the gap is the two runs taking different trajectories (incremental SfM's
initial pair and resection order vary run to run, hence 815 against 831 poses), not the solver.
`ALICEVISION_REQUIRE_CERES_WITH_SUITESPARSE` defaults to OFF, and the build took the new Ceres without
a change. At thousands of poses per bundle adjustment CHOLMOD should pull ahead; that is where I
would measure again.

Packages: https://github.com/dspl1236/Cheshire/releases/tag/v0.3.3 (the measurement is in docs/04).
