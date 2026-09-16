# GPU descriptor matcher

FeatureMatching is the second GPU-shaped stage in the Meshroom pipeline, and upstream has no GPU
path for it on any vendor. On a 107-photo scan on a 12-thread desktop it was the longest node
after Meshing: 592 s, of which a measured 181.5 s of a 196 s chunk was "Regions Matching", the
2-nearest-neighbour search of every query descriptor in the other image's descriptors, with a
FLANN kd-tree (`ANN_L2`, approximate) one pair at a time. The rest (geometric filtering with a
fundamental-matrix RANSAC, grid filtering, writing) is 14 s per chunk.

## What it is

`hip/port/gpu_matcher/`: an exact brute-force 2-NN on the GPU, wired behind AliceVision's
`RegionsMatcher` so that a request for `ANN_L2` or `BRUTE_FORCE_L2` takes the GPU whenever a
device is present. Nothing changes in Meshroom's graph; `CHESHIRE_GPU_MATCHER=0` restores the CPU
path. `scripts/apply_hip_patch.py` (steps 4b/4c) copies the sources into
`src/aliceVision/matching/gpu/` and patches the factory, the CMake and the FeatureMatching main.

* `gpuMatcher.cu`, CUDA dialect: one thread per query descriptor, the database streamed through
  shared memory in tiles (256 rows of 128 B for uint8, 64 rows for float); every lane of a wave
  reads the same database row, so the tile reads are broadcasts and the loop is compute-bound.
  uint8 descriptors (SIFT, DSP-SIFT) stay packed four to a word and the distance is an exact
  integer (max 128 * 255^2 < 2^31), so the result is bit-exact against the CPU brute force; ties go
  to the lower row index. Float descriptors (64 or 128 dims) use FMA. Compiled as HIP through the
  same `cuda_to_hip.h` force-include as the DepthMap port, or as CUDA by nvcc on an NVIDIA build.
* `ArrayMatcher_gpuBruteForce.hpp`: the `ArrayMatcher` adapter (Build uploads the database,
  SearchNeighbours with NN = 2 returns `IndMatch` pairs and squared distances as `float`, which is
  what `L2_Vectorized<unsigned char>` reports too, so the ratio test sees the same numbers).
* The factory takes the GPU for `unsigned char` and `float` regions of 64 or 128 dimensions and
  falls through to upstream's matchers otherwise (`gpu::supportsDim`).

## Meshroom 2023.3 chunking

The AliceVision this port is based on chunks FeatureMatching over *pairs* with
`--rangeIteration/--rangeBlocksCount`; Meshroom 2023.3 chunks over *views* with
`--rangeStart/--rangeSize` and expects `<chunk>.matches.txt`. Step 4c adds the old pair of options:
the pairs whose first view's index falls in the range are kept, and the output prefix is
`rangeStart / rangeSize`, as the 2023.3 binary named it. With that the paired binary is a drop-in:
on the engine bay set it selected exactly Meshroom's 1930 pairs for chunk 0 and wrote
`0.matches.txt`.

## Measured (RX 9070, Windows, engine bay set, DSP-SIFT 128 x uint8)

| | Regions Matching | match lines | note |
|---|---|---|---|
| views 0-1 (190 pairs), CPU `BRUTE_FORCE_L2` | 446.6 s | 3345 | exact reference |
| views 0-1, GPU | 7.9 s | 3345 | match files byte-identical to the CPU brute force |
| views 0-1, CPU `ANN_L2` (kd-tree) | 9.9 s | 3358 | approximate: different neighbours on some queries |
| chunk 0 (20 views, 1930 pairs), kd-tree, Meshroom's own run | 181.5 s | 164461 | |
| chunk 0, GPU | 41.7 s / 42.0 s (two runs) | 164600 | 4.3x; exact search keeps a few more pairs |

The geometric filtering after it is unchanged (8 s per chunk).

## Pairing

`meshroom-pair.cmd` / `meshroom-pair.sh` now pair `aliceVision_featureMatching` next to
`aliceVision_depthMapEstimation`, gated on the package's binary understanding `--rangeStart` (a
package without the GPU matcher must not be put in Meshroom's way: its plain CPU featureMatching
would reject the 2023.3 options). The launcher takes its role from its own file name and applies
the `--sgmFilteringAxes` drop only for DepthMap. An NVIDIA box keeps Meshroom's own (CPU) matcher
until a CUDA build of the package exists; the source compiles as CUDA, the packaging does not yet.
