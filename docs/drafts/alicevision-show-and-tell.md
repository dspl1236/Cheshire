# Draft: AliceVision Discussions, Show and tell - "Meshroom on AMD, and 1678 views through a 2013 dual-core"

*Posted 2026-09-22: https://github.com/orgs/alicevision/discussions/2183 (reflowed to one line per paragraph; corrected before posting against README, docs/04 and the disks: DepthMap vs CUDA is identical masks + median error 0, not bit-identical; set sizes; house-pc OS; downscale 2; cross-repo issue link).*

---

**TL;DR.** Cheshire (MPL-2.0, https://github.com/dspl1236/Cheshire) is AliceVision's GPU stages built
for AMD through HIP, plus several of the CPU-only stages moved to the GPU, packaged to run under a
stock Meshroom 2023.3 by swapping eight binaries. This week it ran two public sets that are larger
than anything we had tested on, on the hardware below. The numbers are the point of the post.

## The hardware

| box | CPU | RAM | GPU | OS |
|---|---|---|---|---|
| house-pc | Intel i3-4330, 2 cores / 4 threads (2013) | 14.6 GB | Radeon RX 6750 XT, 12 GB | Linux Mint 22.3 (Ubuntu 24.04 base), ROCm 7.2 |
| power-pc | AMD Ryzen 5 5600X, 6 cores / 12 threads | 64 GB | Radeon RX 9070, 16 GB | Windows 11 |

## Mill 19 "Rubble" (Mega-NeRF): 1678 drone photographs, 4608x3456, 9.8 GB of JPEG, on the i3

Meshroom 2023.3's `photogrammetry` pipeline at default parameters except `describerTypes=sift`;
depth maps at downscale 2, texture atlases of 8192^2.

| node | time | notes |
|---|---|---|
| FeatureExtraction | 756 s | 1678 views of GPU SIFT (HIP PopSift), 42 chunks |
| ImageMatching | 37 s | 44,377 pairs selected |
| FeatureMatching | 2265 s | exact GPU brute-force 2-NN, then AC-RANSAC on 2 cores |
| StructureFromMotion | 4291 s | 1591 poses, upstream's binary, CPU |
| PrepareDenseScene | 3364 s | 1590 undistorted EXR, 98 GB |
| DepthMap | 34,267 s | 1590 depth maps at downscale 2 (2304x1728), 140 chunks; bridge: 3 full R cameras + 5 tiles per view on 12 GB |
| DepthMapFilter | 1545 s | on the GPU |
| Meshing | 1020 s | GPU votes, GPU max-flow, GPU visibility; 8 GB RAM peak, no swap |
| MeshFiltering | 37 s | |
| Texturing | 10,013 s | 34 atlases of 8192^2, thirteen passes over 1590 cameras |
| total compute | 16.0 h | textured mesh 3,350,047 faces, 1,679,493 vertices; 164 GB of cache |

What that means in plain terms: stock Meshroom's CPU feature extraction alone would run for more
than a day on this box (it measured 49 s per view on the 12-thread Ryzen above; this CPU has four
threads), before the parts that need a CUDA card, which it does not have. Here the CPU is mostly
waiting on the card.

## British Museum "False Door of Ptahshepses": 884 photographs, 6000x3376, 5.5 GB, on the RX 9070

| node | time |
|---|---|
| FeatureExtraction | 560 s |
| FeatureMatching | 796 s |
| StructureFromMotion | 1636 s, 833 poses, 1,355,362 landmarks |
| PrepareDenseScene | 721 s |
| DepthMap | 12,010 s (833 maps, downscale 2) |
| DepthMapFilter | 521 s |
| Meshing | 852 s with five self-checks running alongside (479 s without them, rerun on the same cache) |
| Texturing | 5847 s, 52 atlases of 8192^2, with the padding and resize checks |
| total compute | 6.4 h; textured mesh 8,136,271 faces |

This set is also where we found and fixed an upstream crash in incremental SfM
(alicevision/Meshroom#2344: the resection loop can exit with views pending their bundle adjustment;
details there). Stock Meshroom died three of three times at view 830-834; with the fix it completes.

## Exactness, measured at that scale

The ports carry in-process self-checks that run upstream's CPU code on the same data and count
differences. On the 884-view set, with all of them on:

| check | result |
|---|---|
| Meshing's point filter by pixel size (parallel vs upstream single-threaded) | identical on all 52,305,736 slots |
| GPU nearest-neighbour vs nanoflann | identical on all 3,124,680,410 queries |
| GPU max-flow vs Boykov-Kolmogorov | 0 of 29,080,924 cells labelled differently |
| segment classification | identical on all 29,080,924 cells |
| GPU texture padding vs upstream's sweeps | 0 of 67,108,864 texels, on each of 52 atlases |
| GPU texture downscale vs OpenImageIO lanczos3 | 0 of 50,331,648 channels, on each of 52 atlases |

DepthMap itself, against the CUDA reference (GTX 1080 Ti, 41 views): identical validity masks and a
median relative depth error of 0 on every AMD card, with 97.5-98.9 % of pixels within 1 % on the
median view. RDNA1 and RDNA2 give bit-identical depth maps to each other, and the CUDA build of the
same tree is byte-identical to the CUDA 11.3 reference (82 of 82 files).

## What it runs on

Windows: one 184 MB package with eleven GPU payloads (RDNA1 through RDNA4, both HIP runtimes); a
probe picks the payload. Linux: a relocatable bundle on ROCm 7.2. NVIDIA: the same tree built as
CUDA 12.9 for both platforms. Validated on RX 5500 XT, RX 6750 XT, RX 9070, GTX 1050 Ti, GTX 1080 Ti.
Polaris (RX 400/500) is out: ROCm dropped it.

Packages: https://github.com/dspl1236/Cheshire/releases/tag/v0.3.3 (all four passed their gates
on their own hardware). Docs and the validation log are in the repo (docs/04 is the per-card
record); the earlier post, on the depth map alone, is #2175. Questions welcome.
