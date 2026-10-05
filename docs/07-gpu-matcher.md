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

## The 8-bit matrix instructions on RDNA4, a dp4a kernel for NVIDIA (after 0.4.0)

Two kernels for 128-byte uint8 descriptors, chosen once per process (the log line names it):
`knn2_wmma` on gfx120x and `knn2_dot4q` on NVIDIA sm_61+ and, since its measurement below, RDNA2
(gfx103x); every other card keeps `knn2_u8`.
`CHESHIRE_MATCHER_KERNEL=u8|sliced|dot4q|wmma` picks one (wmma only where the device runs gfx120x
code). Both were developed in `hip/tests/gpumatcher_wmma/` against a verbatim copy of `knn2_u8`.

**Exact by construction.** Both keep a query's two best as packed integer keys over 240-row
windows, 256 (2 dot - rowNorm) + f: 2 dot - rowNorm is queryNorm - distance, so it orders and ties
as the distance does, and fits an int32 at 128 x 255^2. The tie field f is 254 - rowInWindow for
the window's rows and 255 for the two carried in from earlier windows, so the lower row wins every
tie, as a sequential scan with a strict '<' keeps it. Integer sums in any grouping are the same
integers. Tested on every query of 8 engine-bay pairs of 20000 x 20000, and by `wmma_edge.hip`:
database sizes 0 to 7201 around every tile boundary, all-0 and all-255 descriptors, and heavy exact
ties (about 28 000 of 49 000 queries with equal nearest and second distances), three seeds.

**`knn2_wmma`** (RDNA4, `v_wmma_i32_16x16x16_iu8`): a wave takes two 16-query tiles and walks the
database 16 rows at a time; eight instructions cover the 128 bytes. The operand and result layout
was measured by `wmma_probe.hip`. The database goes through LDS transposed, filled with 16-byte
global reads written as two 8-byte stores that never share a bank (the first fill, one word per
thread, was a hidden 16-way conflict: fixing it took one variant from 41.7 to 24.0 ms). Per
candidate: one shift-add, a max and a min-max. On 8 pairs it takes 21.4 ms against `knn2_u8`'s
197 ms (9.2x). `wmma_peak.hip`: the card sustains 144 TMAC/s of WMMA alone and about 112 with three
VALU per value, so the kernel (77) is at about 70 % of what this design allows.

**`knn2_dot4q`** (two queries per thread, so every shared-memory read of a database word feeds two
dp4a): on the GTX 1080 Ti it is 2.03x `knn2_u8` (8 pairs: 133.7 ms against about 270, blocks of
128 threads; `cuda_matcher.cu`). There the card's dp4a peak is about 22 TMAC/s
(`cuda_matcher --peak`); `knn2_u8` runs at 27 % of it, `knn2_dot4q` at 55 %.

**dot4 on RDNA.** `dot4_peak.hip` shows `v_dot4_u32_u8` at full rate on the RX 9070 (39.4 TMAC/s with
eight independent chains, 23.7 with one), and `knn2_u8` at 8.3 TMAC/s. Its loop is 103 instructions
per row for 32 dot4 (one dependent chain, LDS waits, a branchy insertion), but the restructurings
tried stayed near 1.1-1.2x on this card: several rows per iteration 1.17x, scalar loads of the
wave-uniform row 1.23x, `knn2_dot4q` 1.13x. 20000 queries at one query per lane are about 5.6 waves
per SIMD on its 56 CUs, too few to hide latency. RDNA1/2 have fewer CUs and no matrix instructions, so
`knn2_dot4q` may well pay there as it does on Pascal. On RDNA2 it does (below); RDNA1 and RDNA3 are not
measured yet (`CHESHIRE_MATCHER_KERNEL=dot4q`), and `knn2_u8` stays their default.

**FeatureMatching end to end** (the node's own command lines, AC-RANSAC at 2048 iterations; the
search phase from `CHESHIRE_GPU_MATCHER_LOG=1`):

| card | set | `knn2_u8` (two runs) | new kernel (two runs) | search phase |
|---|---|---|---|---|
| RX 9070, Windows | 41 views | 9.9 / 8.8 s | 4.0 / 4.0 s (`knn2_wmma`) | 1.83 s -> 0.37 s per chunk |
| RX 9070, Windows | engine bay | 57.3 / 56.4 s | 25.6 / 24.9 s | 6.3 s -> 1.15 s per chunk |
| GTX 1080 Ti, Linux | 41 views | 13.1 / 13.0 s | 9.8 / 9.8 s (`knn2_dot4q`) | 7.42 s -> 4.27 s |

The matches are byte-identical to `knn2_u8`'s in every run, and the self-check
(`CHESHIRE_GPU_MATCHER_CHECK=1`) found 53,300 of 53,300 sampled queries identical to upstream's
brute force on each card. The Linux numbers are the 0.4.0 CUDA bundle with only
`libaliceVision_matching` rebuilt.

**RDNA2** (2026-10-05): the RX 6750 XT (gfx1031) in house-pc, Linux, the HIP bundle of 10a-10i
(`build/gate-10i`), the 41-view FeatureMatching of its gate run replayed three times with each kernel,
alternating (`build/gate-10i/fm_kernels.py`). The matches are byte-identical to the gate run's with both kernels.
That run used Meshroom 2025.1's 50,000 AC-RANSAC iterations, so on house-pc's four-thread i3 the
geometric filter takes about 131 s, and the search overlaps it (10a):

| per run, three runs each | `knn2_u8` | `knn2_dot4q` |
|---|---|---|
| kernels (`CHESHIRE_GPU_MATCHER_LOG=1`) | 13.45-13.48 s | 7.05-7.06 s |
| searches, uploads and downloads included | 14.13-14.17 s | 7.75 s |
| the search thread | 15.22-15.35 s | 8.81-8.83 s |
| the node, three chunks | 132.42-132.48 s | 130.58-130.87 s |

1.9x on the kernels, so `knn2_dot4q` is RDNA2's default from now on (the gfx103x family shares gfx1031's
ISA). On this machine the node gains only the 1.7 s the filter does not cover; with a CPU fast enough that
the search is the longer of the two, it gains up to the 6.4 s.

What is left: `CHESHIRE_GPU_MATCHER_LOG=1` now splits the searches into upload, kernels and download
(the kernels synchronized, only when profiling). On the RX 9070 an engine-bay chunk's 945 searches
are 0.26 s of query uploads, 0.69 s of kernels and 0.19 s of downloads, so the transfers are now
40 % of the search. One contiguous result buffer copied once into pinned memory was tried and was
slower (engine bay 26.0 s against 24.7: HIP's default pinned memory is coherent, slow for the CPU to
read back), so the two copies stay. RDNA3 has matrix instructions too, but other operand shapes
(`v4i`), and needs its own layout probe.
