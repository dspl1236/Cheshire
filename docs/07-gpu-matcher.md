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

The geometric filtering after it is unchanged (8 s per chunk). The first kernel unpacked bytes
and took 42 s on chunk 0; the profile (`CHESHIRE_GPU_MATCHER_LOG=1`) put 41.2 of those seconds
inside the GPU search with 20,000 descriptors per image (51 billion multiply-adds per pair), so
the uint8 path became |q|^2 + |r|^2 - 2 q.r with the packed 4-byte dot instruction
(`v_dot4_u32_u8` on RDNA2+/Vega 20, `dp4a` on sm_61+, four multiply-adds elsewhere), same
integer, same match files: 13.0 s.

## End to end: "as good or better"

The engine bay job (107 phone photos) run twice through the paired Meshroom install with the same
CameraInit / FeatureExtraction / ImageMatching results, once with Meshroom's kd-tree matcher and
once with the GPU matcher, RX 9070:

| | kd-tree (Meshroom) | GPU matcher |
|---|---|---|
| FeatureMatching node (6 chunks, incl. geometric filtering) | 592 s | 75 s |
| matching stage alone, all chunks | 543 s | 40.7 s |
| match lines after geometric filtering | 506743 | 503214 |
| SfM: cameras registered | 107 / 107 | 107 / 107 |
| SfM: landmarks | 141963 | 141249 |
| SfM: final residual RMSE (px) | 1.712 | 1.694 |
| SfM: mean track length | 3.002 | 3.001 |
| textured mesh | 1,184,852 vertices, 2,357,143 faces | 1,180,762 vertices, 2,349,340 faces |

Same registration, a marginally lower reprojection error, 0.5 % fewer landmarks and a mesh within
0.3 %: the exact matcher and the approximate one land on the same reconstruction, and the node is
7.9x faster. The whole job went from 39 minutes to 30 on this desktop; the CPU share fell from 1750 s
to 1230 s and the matching stage no longer depends on the CPU at all.

## Linux, RDNA2, and the cross-platform check

The same chunk 0 through the v0.2.5 Linux bundle on the RX 6750 XT (house-pc): 29.2 s of matching
against 13.0 s on the RX 9070, GPU search 28.4 s of it. The 41 depth maps of that bundle are
bit-identical to v0.2.3's, so the matcher's presence changes nothing downstream.

The putative match sets (geometric filtering off) agree with the Windows RX 9070 run on 1923 of
1930 pairs; the other 7 differ by exactly one match each. Every one of those is a ratio-test
boundary case: the GPU distances are exact integers and identical on both platforms, and the
decision `d1 < 0.8^2 * d2` is made on the CPU in `float`, where MSVC and GCC do not round
`0.8f * 0.8f` and the product the same way. So the kernels are cross-platform exact; the last bit
of the ratio test is the compiler's. After geometric filtering (AC-RANSAC) the files differ more,
191 of 277 common pairs by a few matches each, which is that filter's own platform dependence
and not the matcher.

RDNA1 (RX 5500 XT, gfx1012, Linux, v0.2.5 bundle): chunk 0 in 55.5 s (GPU search 52.6 s), and
the putative matches byte-identical to the RX 6750 XT's. Three AMD generations and two operating
systems now agree on the matcher's output to the byte; the 41 depth maps of that run are also
bit-identical to the card's v0.2.3 output (396.4 s).

A build with the dot instruction compiled out (`CHESHIRE_MATCHER_NO_DOT4`) produces byte-identical
matches to the dot-instruction build on the RX 9070, 14.3 s against 13.0 s. Note the compiler
pattern-matched part of the manual multiply-add loop back into `v_dot4` (36 instructions against
64), so a card that genuinely lacks the instruction (gfx1010, gfx900) still awaits a hardware run.

## Pairing

`meshroom-pair.cmd` / `meshroom-pair.sh` now pair `aliceVision_featureMatching` next to
`aliceVision_depthMapEstimation`, gated on the package's binary understanding `--rangeStart` (a
package without the GPU matcher must not be put in Meshroom's way: its plain CPU featureMatching
would reject the 2023.3 options). The launcher takes its role from its own file name and applies
the `--sgmFilteringAxes` drop only for DepthMap. An NVIDIA box keeps Meshroom's own (CPU) matcher
until a CUDA build of the package exists; the source compiles as CUDA, the packaging does not yet.

## A sliced, register-tiled variant, measured and shelved (0.3.3)

At Meshroom's own settings (dspsift, 2048 AC-RANSAC iterations) FeatureMatching's chunk 0 of the
engine bay is 24 s on the RX 9070 + 5600X: 2 s of loading, 13 s of GPU search (1930 searches,
38.6 M query descriptors), 8 s of geometric filtering. The search is about a quarter of the card's
dot4 rate, so a second kernel was tried: Q queries per thread reading each database word once for
all Q (uint4 reads), and the database sliced along `blockIdx.y` so a search is hundreds of blocks
rather than tens, with a merge of the per-slice 2-NN by (distance, index). Byte-identical output
(165,162 match lines, verified against both the original kernel and the 0.3.2 bundle), and slower:
Q=4 23.0 s (192 VGPRs, 48 B/lane of scratch, occupancy 8), Q=2 13.8 s (119 VGPRs, no scratch,
occupancy 12) against the original's 12.6 s (52 VGPRs, occupancy 16). The original's tile reads are
wavefront broadcasts, as its header says, so it was never LDS-bound and register reuse buys
nothing; the remaining gap to the roofline is per-search transfer and launch (about 1 ms of 6.5)
and the scan itself, and closing it is 2-D register tiling with tuned occupancy - a project, not a
patch. The variant stays in the source behind `CHESHIRE_MATCHER_SLICED=1` so it can be measured on
cards with fewer CUs, where the block count may matter more.

The compiler remarks that settled it (`-Rpass-analysis=kernel-resource-usage` on a standalone
`clang++ -x hip` compile of the port) are the cheap way to ask this question before a rebuild.
