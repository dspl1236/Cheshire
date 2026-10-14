# The CPU matcher's `random_seed` is ignored on Windows (vcpkg FLANN)

**Filed 2026-10-14 as alicevision/AliceVision#2210.** The text below is the draft it was filed from.

**Title:** FeatureMatching (ANN_L2) differs run to run on Windows: vcpkg's FLANN ignores `random_seed`

`ArrayMatcher_kdtreeFlann::Build` sets `params["random_seed"] = 1` on its `KDTreeIndexParams`. AliceVision's FLANN fork
(used by the Linux superbuild) reads it in `KDTreeIndex::buildIndexImpl`:

```cpp
auto seed = get_param(index_params_, "random_seed", 0);
std::mt19937 g(seed);
```

and shuffles every randomized tree from that generator, so the CPU matcher repeats on Linux. The Windows dependencies come
from vcpkg's `flann` port (2022-10-28, FLANN 1.9.2), whose `kdtree_index.h` instead does, per tree:

```cpp
std::random_device rd;
std::mt19937 g(rd());
std::shuffle(ind.begin(), ind.end(), g);
```

so the trees, and the approximate neighbours, change on every run. Measured with Meshroom 2025.1's own Windows build:
`aliceVision_featureMatching` on the same six-view DSP-SIFT features (`--photometricMatchingMethod ANN_L2`, all pairs),
three runs, three different match files. Everything downstream then differs
too, which makes Windows CPU-only runs impossible to reproduce or compare.

**Fix:** build the Windows dependencies with the fork's `kdtree_index.h` (an overlay port, or the fork's commit as a patch
on the vcpkg port). In AliceVision itself, a configure-time check that the installed FLANN reads `random_seed` would catch
it; Cheshire does that and builds a seeded copy of the index when it does not (step 16b).
