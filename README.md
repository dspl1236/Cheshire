# Cheshire

**Meshroom / AliceVision photogrammetry on AMD Radeon GPUs, via HIP.**

AliceVision's dense reconstruction (`DepthMap`, the stage Meshroom needs a GPU for) is
written in CUDA. An official SYCL backend was merged upstream in May 2026
([AliceVision#2077](https://github.com/alicevision/AliceVision/pull/2077), closing the
eight-year [#439](https://github.com/alicevision/AliceVision/issues/439)) but has not appeared
in any release, resolves to off when AdaptiveCpp is absent, and has no published results on
AMD hardware. Cheshire is a HIP port you can download and run today, on Windows and Linux:
the same CUDA kernels compiled through a small compatibility layer, plus a **memory bridge**
that lets the stage spill from VRAM into system RAM instead of failing, validated
per architecture against a CUDA node's output on RDNA1, RDNA2 and RDNA4 cards.

The cat that grins on any hardware.

Two things here are not off-the-shelf. The port itself is close to mechanical (a compat
header does what `hipify` would), but it is *validated*: every architecture is compared
against a CUDA reference with a stated metric, and the porting bugs found on the way are the
kind that produce wrong output rather than errors (HIP dropping `surf2Dwrite` into 16-bit
float arrays; GPU atomics into mapped host memory silently wrong on Linux unless the memory
is non-coherent). The memory bridge is the novel part: it decides *what* spills to system RAM
by buffer class, and the policy is set by measurement, not by intuition. With the default
fine-grained host mapping, 186 MB of texture-sampled camera images behind PCIe cost 22x while
6 GB of streamed volumes cost 4.7x; coarse-grained mapping turns the images to 1.0x. That
number is why the design is what it is.

## What it does today

| Meshroom node | upstream | Cheshire | status |
|---|---|---|---|
| DepthMap | CUDA only | HIP port + VRAM-to-RAM memory bridge | validated RDNA1/2/4, Windows + Linux; bit-identical to native across caps |
| FeatureMatching | CPU (kd-tree) on every vendor | exact GPU brute-force 2-NN (HIP; the source also builds as CUDA), plus exact CPU work in AC-RANSAC ([docs/15](docs/15-acransac-cpu.md)) | validated end to end (v0.2.5): 592 s -> 75 s on 107 photos, same reconstruction. Geometric verification followed in v0.2.15: 566 s -> 351 s, byte-identical match file, and the profile is why it stayed on the CPU |
| FeatureExtraction | CUDA (PopSift) or CPU | HIP PopSift port ([docs/14](docs/14-gpu-sift.md)) | done (v0.2.13): 2021 s -> 26 s on 41 views (RX 9070), and better than the CPU path rather than merely faster — 88,463 landmarks against 78,372 and RMSE 1.0377 against 1.03997 px. Validated on RDNA4, RDNA1 and RDNA2 across two runtimes and both systems, landmarks within 0.11 %; found and reported an upstream PopSift bug ([popsift#193](https://github.com/alicevision/popsift/issues/193)) |
| DepthMapFilter | CPU | GPU vote pass (HIP / CUDA source) + a shared decoded-map cache | validated (v0.2.6): 123.5 s -> 26.5 s on 107 photos (RX 9070), 317 s -> 26.9 s on 41 views on an i3 + RX 5500 XT, bit-identical; found and replicated an upstream vote-buffer quirk; v0.2.9 cache: 17.8 s -> 12.7 s, byte-identical |
| Meshing | CPU | GPU graph-weight votes + weakly-supported-surfaces pass, GPU min-cut (push-relabel, [docs/12](docs/12-gpu-maxflow.md)), GPU nearest-neighbour search for the visibility passes ([docs/13](docs/13-gpu-visibilities.md)), exact CPU fixes around them ([docs/11](docs/11-meshing-cpu.md)) | done (v0.2.7 through v0.2.12, v0.2.16): 510 s -> 101 s on 107 photos (RX 9070), 491 s -> 111 s on 41 views on an i3 + RX 5500 XT; the cut labelling identical to Boykov-Kolmogorov's and the nearest neighbours identical to nanoflann's on every query, every CPU change verified identical in-process; what is left is the tetrahedralisation |
| Texturing | CPU | GPU Laplacian pyramid + rasterisation, parallel camera selection (HIP / CUDA source), exact CPU fixes in the UV unwrap ([docs/10](docs/10-gpu-texturing.md)) | done (v0.2.8, v0.2.16): 165.8 s -> 79.3 s on 107 photos (RX 9070), 271.5 s -> 102.5 s on 41 views on an i3 + RX 5500 XT, textures inside the CPU's own run-to-run band; v0.2.16 took the UV unwrap 14.4 s -> 8.6 s with a byte-identical `texturedMesh.obj`, and gains most on the slowest hardware (51.9 s -> 16.8 s on a 2011 Bulldozer); what is left is mesh I/O and padding |
| PrepareDenseScene | CPU | every core instead of three threads | done (v0.2.9): 36 s -> 30 s on 107 photos, byte-identical; the loop is codec and disk work |
| SfM, ImageMatching, MeshFiltering | CPU | | stay on the CPU (sequential or tiny) |

Everything installs by pairing: one script swaps the node binaries (DepthMap, FeatureExtraction,
FeatureMatching, DepthMapFilter, Meshing, Texturing, PrepareDenseScene) in an existing Meshroom 2023.3 install and keeps Meshroom's own beside
them, choosing per run by the card present. On a
12-thread desktop the 107-photo engine bay job went from 39 minutes with the CUDA-era layout to 30
as measured at v0.2.13, before the geometric verification, point-cloud fusion and UV unwrap work in
v0.2.15 and v0.2.16 took another 230 s off the stages it covers.

## Results

Same photos, same SfM, same DepthMap parameters as the CUDA reference (Meshroom 2023.3 on a
GTX 1080 Ti); only the GPU stage differs. Metrics are per view over pixels valid in both maps.
Times are the DepthMap stage only. The CUDA reference ran as Meshroom runs it, in chunks of
12 views (four chunks for 41 views: 105.4 + 111.1 + 106.3 + 56.2 s); the HIP times are one
invocation over all views. The apples-to-apples numbers are the two cards run inside real
Meshroom jobs on the same node, also four chunks: RX 6750 XT 352 s, RX 5500 XT 594 s
(both jobs reconstructed the full 41-photo mesh, within 2 % of the CUDA job's vertex count).

| GPU | OS | 6 views | 41 views | per view (41) | valid masks | median rel. depth error | within 1 % (median view, 6 / 41) |
|---|---|---|---|---|---|---|---|
| GTX 1080 Ti (CUDA reference, 4 Meshroom chunks) | Linux | 31.8 s | 379.0 s | 9.2 s | | | |
| GTX 1080 Ti, CUDA 11.6 (Meshroom 2023.3 Windows build), same host as the RX 6750 XT Windows rows | Windows | 40.0 s | 287.1 s | 7.0 s | 37 / 41 identical to the reference | 0.0000 | 100 % / 100 % (worst view 92.9 %) |
| Radeon RX 9070, RDNA4 | Windows | 17.4 s | 124.4 s | 3.0 s | identical | 0.0000 | 98.9 % / 98.7 % |
| Radeon RX 6750 XT, RDNA2 | Linux | 31.0 s | 226.1 s (352 s in 4 Meshroom chunks) | 5.5 s | identical | 0.0000 | 97.5 % / 98.1 % |
| Radeon RX 6750 XT, RDNA2, HIP SDK 6.2 build | Windows | 28.1 s | 190.3 s | 4.6 s | identical | 0.0000 | 98.7 % / 98.7 % |
| Radeon RX 6750 XT, RDNA2, v0.2.3 bundle | Linux | 28.0 s | 205.7 s | 5.0 s | identical | 0.0000 | 97.5 % / 98.1 % |
| Radeon RX 5500 XT, RDNA1 | Linux | 60.7 s | 447.7 s (594 s in 4 Meshroom chunks) | 10.9 s | identical | 0.0000 | 97.5 % / 98.1 % |
| Radeon RX 5500 XT, RDNA1, v0.2.3 bundle | Linux | 52.4 s | 396.9 s | 9.7 s | identical | 0.0000 | 97.5 % / 98.1 % |
| Radeon RX 5500 XT, RDNA1, v0.2.5 bundle | Linux | | 396.4 s | 9.7 s | identical | 0.0000 | 97.5 % / 98.1 % |

Per-view cost on every HIP card is flat between 6 and 41 views (the port scales linearly);
the CUDA reference's per-view cost rises from 5.3 s to 9.2 s across the chunked run. Until
2026-09-15 this table listed the CUDA 41-view time as 105.5 s, which was the first chunk
alone; the corrected figure is the sum of all four (`data/ref/monstree-full/DepthMap/*/[0-3].log`).

### What "identical" means here, per comparison

Three different comparisons appear in this repo, and they have three different answers.
All metrics are per view over pixels valid in both maps (`scripts/compare_depthmaps.py`;
tables in [docs/validation](docs/validation/)).

| comparison | validity masks | median rel. depth error | pixels within 1 % | verdict |
|---|---|---|---|---|
| same card, any bridge configuration vs the uncapped run | identical | 0 | 100 % (every pixel equal) | bit-identical |
| RX 5500 XT vs RX 6750 XT (RDNA1 vs RDNA2), same HIP build | identical | 0 | 100 % (every pixel equal) | bit-identical |
| RX 9070 vs RX 6750 XT (RDNA4 vs RDNA2), same HIP build | identical | 0 | 97.0-98.5 % | not bit-identical: p95 error 0.07-0.24 % |
| RX 6750 XT under Windows / HIP 6.2 vs the same card under Linux / HIP 7.2 | identical | 0 | 97.1-98.7 % | not bit-identical: same silicon, different compiler and runtime |
| **GTX 1080 Ti, CUDA 11.6 on Windows vs GTX 1080 Ti, CUDA 11.3 on Linux** (the reference itself) | 41 / 41 | 0 | 100 % median view, 92.9 % worst | 37 / 41 depth maps bit-identical; the other four differ by up to 7.2 % of pixels. CUDA is not bit-identical to CUDA across builds either |
| any HIP card vs the CUDA GTX 1080 Ti | identical | 0 | 97.5-98.9 % | same noise floor as RDNA4 vs RDNA2 |

So the port is deterministic (same inputs, same build, same architecture family, same bits),
and the 1-3 % of pixels that differ by more than 1 % between RDNA4 and RDNA2, between two
toolchains on the same card, or between any AMD card and CUDA, is the algorithm's noise floor:
texture-filter and FMA rounding propagating through SGM's argmin, not a port defect. The same
1-3 % appears between two AMD generations running identical code, between two compilers on
identical silicon, and between two CUDA builds on the same NVIDIA card (the reference disagrees
with a Windows CUDA 11.6 run of itself on 4 of 41 views, one of them by 7.2 % of pixels), which
is what rules out the port as the cause.

## Memory bridge

Measured on the RX 9070 (`hip/tests/membridge_probe.hip`), the tiers the bridge chooses between:

| memory | kernel streaming read |
|---|---|
| VRAM (`hipMalloc`) | 585-619 GB/s |
| mapped system RAM (`hipHostMalloc`, PCIe 4.0 x16) | 24.6-26.6 GB/s |
| `hipMallocManaged` | 26.6 GB/s (no page migration on Windows) |
| pinned host copies | 28 GB/s each way |

On Windows the driver already lets `hipMalloc` run past VRAM into system RAM; on Linux it fails
hard. (Mapped host memory is real system memory on both: `hip/tests/host_backing.hip` reads it at
28 GB/s against 615 GB/s for VRAM and it leaves free VRAM untouched; Task Manager's "dedicated GPU
memory" for such a process is a commitment figure, not residency.) The bridge (`hip/compat/include/cheshire/bridge.h`) makes both behave the same, and v2
decides *what* spills by buffer class instead of by arrival order. Measured, one class at a
time behind PCIe, every run bit-identical:

| spilled to system RAM (6 views) | fine-grained host memory | coarse-grained (shipped) |
|---|---|---|
| nothing | 20.6 s | 20.2 s |
| depth/sim maps (879 MB) | 35.9 s (1.7x) | 25.5 s (1.3x) |
| similarity volumes (6.0 GB) | 96.4 s (4.7x) | 76.6 s (3.8x) |
| camera images (186 MB) | 460 s (22x) | 20.4 s (1.0x) |

The host tier is coarse-grained (`hipHostMallocNonCoherent`): the GPU caches it, atomics work,
and the texture-sampled camera images stop costing anything. Images stay resident regardless
(the planner reserves VRAM for them), volumes and maps spill, and the DepthMap planner sizes
its tile parallelism from the bridge's budget instead of `hipMemGetInfo`:

| VRAM available | v1 (arrival order, planner unaware) | v2 |
|---|---|---|
| 1.5 GB | 47.7 s, 188 spills | 20.0 s, 0 spills |
| 1 GB | | 20.3 s, 0 spills |
| 500 MB (below one tile) | fails upstream | 47.2 s, runs |

Tile parallelism costs nothing to give up on this GPU, so a card with 1 GB to spare runs the
stage at full speed. Design, knobs and every table:
[docs/02-memory-bridge.md](docs/02-memory-bridge.md).

### GPU descriptor matcher (2026-09-16)

FeatureMatching is the second GPU-shaped stage, and upstream has no GPU path for it on any vendor.
`hip/port/gpu_matcher/` is an exact brute-force 2-NN on the GPU behind AliceVision's
`RegionsMatcher` (taken for `ANN_L2` / `BRUTE_FORCE_L2` whenever a device is present; the same
source builds as HIP or CUDA), byte-identical to the CPU brute force. Engine bay set, 107 phone
photos, RX 9070: the FeatureMatching node from 592 s to 75 s, the matching stage from 543 s to
40.7 s, and the reconstruction unchanged (107 / 107 cameras, residual RMSE 1.694 px vs 1.712,
mesh within 0.3 %). The pairing scripts swap this binary in next to DepthMap. Details and the
Meshroom 2023.3 chunking compatibility in [docs/07-gpu-matcher.md](docs/07-gpu-matcher.md).

### Full-resolution depth maps under a cap (downscale 1, 41 views, 2026-09-16)

| card | uncapped | 2 GB cap | 1 GB cap |
|---|---|---|---|
| RX 9070, Windows | 499.6 s (30 tiles) | 558.7 s (1 tile, no spills) | 2871.6 s (volumes behind PCIe) |
| RX 6750 XT, Linux | 744.1 s (20 tiles) | 749.9 s (1 tile, no spills, after two bridge fixes) | 4035 s |

Capped runs are bit-identical to uncapped on Windows (41 / 41, twice); on Linux 40 / 41, the odd
view differing by 110 pixels at 1e-6. The series produced two bridge fixes (a 4 MB floor under which
nothing spills, and an image reserve net of the images already resident: the Linux 2 GB cap went
from 863 s with 6047 spills to 749.9 s with none) and a hard floor: at 1 GB the SGM volumes
themselves live behind the link and the job runs 4-5x slower, which is the price of running where
upstream refuses. Details in `docs/02-memory-bridge.md`.

## What was found on the way (reproducers in `hip/tests/`)

* **HIP drops `surf2Dwrite` stores into 16-bit float arrays**, on Windows and Linux alike;
  reads are fine. AliceVision's mip chain is built through a buffer copy instead.
* **Linux ROCm 7.2 has no `hipMallocMipmappedArray`** on RDNA1 (and WSL2 has no textures at
  all). Mipmaps are emulated in the compat layer as one texture per level, in array or pitched
  linear memory: bit-identical to native mipmaps on the RX 9070 and on the RX 6750 XT (41 / 41
  views), so the Windows build keeps native and the Linux bundle emulates on every
  architecture. Emulation used to cost 21 % (RX 9070, 6 views: 19.9 s vs 16.5 s) and 22 %
  (RX 6750 XT, 41 views: 232.3 s vs 190.3 s), which was the whole Linux-vs-Windows gap on
  that card. The disassembly showed why: the per-level texture handles lived in a device-side
  table, and the compiler could not prove the table index wave-uniform, so every sample
  fetched its descriptor with vector loads and ten `v_readfirstlane`s. The level descriptors
  are now packed into fixed-stride constant-memory slots addressed by arithmetic, with a
  run-time uniformity check that takes one scalar descriptor load on the common path (the
  per-pixel levels of `useConsistentScale` fall back to an out-of-line waterfall). Emulation
  now costs 4 % on the RX 9070 (17.2 s) and 9 % on the RX 6750 XT (208.1 s), still
  bit-identical; the v0.2.3 Linux bundle takes the RX 5500 XT from 447.7 s to 396.9 s and the
  RX 6750 XT from 226.1 s to 205.7 s on the 41-view set, byte-identical output. The
  bit-identity is expected rather than remarkable: AliceVision samples its mip chain only at
  integer levels (`level = log2(scale / minDownscale)`), where trilinear filtering reduces to
  bilinear on one level. Linear levels are what lets the bridge account for camera images.
* **GPU atomics into mapped host memory are silently wrong on Linux** unless the memory is
  allocated non-coherent: on an RX 6750 XT, `atomicMin` into default (fine-grained) host memory
  fails on every element while `atomicAdd` works, and both are right on
  `hipHostMallocNonCoherent` memory (`hip/tests/host_atomics.hip`; Windows passes all cases).
  Anything that spills a buffer kernels do atomics on, LLM runtimes included, needs to know:
  standalone write-up in [docs/notes/gpu-host-memory-on-rocm.md](docs/notes/gpu-host-memory-on-rocm.md).
* **Fine-grained host memory is the wrong tier for texture-sampled data**: with the default
  mapping, 186 MB of camera images behind PCIe cost 22x while 6 GB of streamed volumes cost
  4.7x; allocated coarse-grained, the device caches them and the images cost nothing.
* **ROCm for Windows installs as pip wheels**, no admin installer. CMake refuses to mix
  `cl.exe` with clang for HIP: use `clang-cl` for both. `-fgpu-rdc` is broken on Windows, so
  the device code is compiled as one unity translation unit.
* **A relocatable bundle needs things the dependency walkers never see**: HIP loads the
  code-object manager (`libamd_comgr` / `amd_comgr0702.dll`) at runtime on both OSes, and
  without it the GPU simply "does not exist". On Linux a WSL build box also ships the
  `/dev/dxg` flavour of `libhsa-runtime64`, which must be swapped for the KFD one. With those
  in place RDNA1 works natively on ROCm 7.2.

## Downloads

Binaries are on the [v0.2.16 release](https://github.com/dspl1236/Cheshire/releases/tag/v0.2.16)
(every package carries every stage; `scripts/verify_packages.py` checks an archive actually
contains the current work rather than an older build); the data sets
and references are on
[v0.1.0](https://github.com/dspl1236/Cheshire/releases/tag/v0.1.0) and unchanged, the depth
maps being bit-identical between the two:

| asset | size | contents |
|---|---|---|
| `cheshire-alicevision-hip-windows-x64-rocm7.2.1-gfx1201.zip` (v0.2.16) | 104 MB | self-contained AliceVision + HIP DepthMap + GPU SIFT, matcher, depth map filter, meshing votes and texturing for RDNA4 discrete on Windows (HIP 7.2 runtime, vcpkg and MSVC runtimes bundled), `meshroom-pair.cmd` + launcher in the root |
| `cheshire-alicevision-hip-windows-x64-rocm7.2.1-rdna3-rdna4.zip` (v0.2.16) | 108 MB | the same with gfx1100/1101/1102/1103, gfx1150/1151/1152/1153, gfx1200/1201 |
| `cheshire-alicevision-hip6.2-windows-x64-gfx1030-avx.zip`, `-gfx1031-avx.zip`, `-gfx1032-avx.zip`, `-gfx1012-avx.zip` (v0.2.16) | 147 MB each | RX 5000 / RX 6000 on Windows through AMD's HIP 6.2 runtime (the one the driver ships): one package per chip because the HIP SDK 6.2 toolchain cannot bundle several. gfx1012 = RX 5500/5500 XT, gfx1030 = RX 6800/6900/6950, gfx1031 = RX 6700/6750, gfx1032 = RX 6600/6650. Built `/arch:AVX`, so they also run on pre-Haswell CPUs. Validated on an RX 6750 XT and an RX 5500 XT |
| `cheshire-alicevision-hip-linux-x64-rocm7.2.tar.gz` (v0.2.16) | 123 MB | relocatable Linux bundle with the GPU matcher, depth map filter, meshing votes, texturing and the packed-slot mipmap sampler, depth-map code objects for RDNA1-RDNA4 discrete parts, the RDNA2/RDNA3 APUs (gfx1035/1036/1103/1150/1151/1152/1153) and Vega (gfx900/906, untested) - 19 in all; GPU SIFT covers 15 of those, not the RDNA2 APUs or Vega, which fall back to CPU SIFT; needs only `amdgpu` + `/dev/kfd`; validated on the RX 6750 XT and RX 5500 XT |
| `monstree-mini6-meshroom-cache.tar.gz` (v0.1.0) | 383 MB | 6-view Meshroom 2023.3 cache: CameraInit, SfM, PrepareDenseScene and the CUDA DepthMap reference |
| `monstree-full-cuda-reference.tar.gz` (v0.1.0) | 680 MB | 41-view SfM + CUDA DepthMap reference (GTX 1080 Ti) |
| `cheshire-hip-depthmap-outputs.tar.gz` (v0.1.0) | 966 MB | the HIP depth maps behind the table above (RX 9070 6 + 41 views, RX 5500 XT, RX 6750 XT) |

**Pick by `gfx` number, not by series.** RDNA1 and RDNA2 are each several architectures, and a
package built for one does not refuse to run on another - it enumerates the device, runs every
kernel and returns wrong data with no error at all. gfx1010 (RX 5600/5700), gfx1034 (RX 6500 XT /
6400) and the APUs have no Windows package yet; the Linux bundle covers them. Folding the lot into
one launcher-selected Windows package is the next packaging item.

Windows packages carry every DLL they need, the MSVC and OpenMP runtimes included (until
2026-09-15 they did not, and a machine without Visual Studio died with exit code 0xC0000135
and no message). The `rocm7.2.1` packages are compiled for AVX2, so a pre-2013 CPU dies with
0xC000001D (illegal instruction); the `hip6.2` packages are built `/arch:AVX` and run on
anything from Sandy Bridge and Bulldozer onwards (`CHESHIRE_ARCH_FLAG` sets this at build time).
The `rocm7.2.1` packages need **Adrenalin 26.2.2 or newer**: they carry the HIP 7.2 runtime, and
a 2025 driver (which ships the HIP 6 runtime) answers `hipErrorNoDevice`, shown by the tools as
"No CUDA-Enabled GPU" even though the card is fine. The `hip6.2` packages use the runtime the
driver ships and have no such floor.

Reproduce a row: unpack a cache under `data/ref/<dataset>/`, then `scripts\run-depthmap.cmd <dataset>`
(Windows, set `CHESHIRE_INSTALL` to the unzipped folder) or `scripts/linux/run-depthmap.sh` (Linux).
Both run the exact Meshroom 2023.3 DepthMap command line and finish with `scripts/compare_depthmaps.py`,
which prints the per-view table and writes the side-by-side panels. Photos are
[alicevision/dataset_monstree](https://github.com/alicevision/dataset_monstree).

### Using it from Meshroom on Windows

Meshroom runs its DepthMap node as `aliceVision_depthMapEstimation` from its own `aliceVision\bin`.
`meshroom-pair.cmd` (in every Windows zip, and in
[`cheshire-meshroom-pair-windows.zip`](https://github.com/dspl1236/Cheshire/releases/tag/v0.2.4) for
the older zips) replaces that one binary with a launcher and keeps the CUDA one beside it as
`.cuda.exe`:

```
meshroom-pair.cmd C:\Meshroom-2023.3.0 C:\cheshire-alicevision-hip-windows-x64-rocm7.2.1-gfx1201
meshroom-pair.cmd C:\Meshroom-2023.3.0 --unpair
```

The launcher decides per run: an NVIDIA card present (`nvidia-smi` answers) runs the CUDA binary,
otherwise the HIP build from the package, with the package's DLLs and `share/` in front so nothing
of Meshroom's older AliceVision leaks in. `CHESHIRE_DEPTHMAP=cuda|hip` forces one; the choice is
printed as the first line of the node's log. Every other node keeps running from Meshroom.

From v0.2.13 the pairing also covers **FeatureExtraction**, so GPU SIFT reaches the graph: leave
**forceCpuExtraction** unticked ([docs/14](docs/14-gpu-sift.md)). With an older package, or one
for a card the HIP runtime cannot see, the node falls back to CPU SIFT without saying so — a
paired node that still logs `[cpu]` means the runtime cannot see the card, not that pairing
failed. If you are on a pre-v0.2.13 package, keep the option ticked. The same pairing on Linux is
`scripts/linux/meshroom-pair.sh`.

## Help wanted: RDNA3

Every package carries RDNA3 code objects (gfx1100/1101/1102, plus the APUs gfx1103/1150/1151/1152/1153)
and none has run on RDNA3 hardware; there is no such chip here. Ten minutes on an RX 7600/7700/7800/7900,
or a Ryzen 7040/8040/AI 300 laptop, closes the gap (Windows: the RDNA3+RDNA4 zip; if it
exits without a message, the exit code says why: 0xC0000135 means a DLL is missing, 0xC000001D
means the CPU lacks AVX2; "No CUDA-Enabled GPU" with the card present means the Adrenalin
driver predates 26.2.2):

1. Download the RDNA3+RDNA4 Windows zip (or the Linux bundle) and
   `monstree-mini6-meshroom-cache.tar.gz` from the release pages above.
2. Unpack the cache under `data/ref/monstree-mini6/`, unzip the package anywhere, then
   `set CHESHIRE_INSTALL=<unzipped folder>` and `scripts\run-depthmap.cmd monstree-mini6`
   (Linux: `scripts/linux/run-depthmap.sh <bundle> <cache> <out> <cache>/DepthMap/<id>`).
3. It prints the card name, `Task done in (s)`, and a per-view table against the CUDA
   reference; `compare_stats.md` lands next to the depth maps. Open an issue with those two
   things and the card model. A `[cheshire] bridge summary` line appears with
   `CHESHIRE_BRIDGE_LOG=1` if anything spilled.

Expected: mask agreement 1.000, median relative depth error 0.0000, 97-99 % of pixels within
1 % on every view, and a time somewhere between the RX 6750 XT and the RX 9070 rows.

## Roadmap

The aim is a small PC with a big GPU that runs a Meshroom job as fast as a workstation and just
as precisely. In order, each timed on the 41-view and engine bay sets and checked against the CPU
output - bit-identity where the same computation moved to another device, and equivalence at the
reconstruction where it could not (SIFT is a different descriptor implementation, not the same
kernels elsewhere):

1. ~~**DepthMapFilter on the GPU.**~~ Done in v0.2.6 ([docs/08](docs/08-gpu-depth-map-filter.md)):
   123.5 s to 26.5 s on the engine bay job, what remains is EXR reading; a shared decoded-map cache
   is the follow-up.
2. ~~**Meshing's voting pass.**~~ Done in v0.2.7 ([docs/09](docs/09-gpu-meshing-votes.md)): both
   ray-marching passes on the GPU, 37 s to 18 s on the engine bay job. It was not the majority of
   Meshing: the profile said the Boykov-Kolmogorov max-flow (144 s) and building its graph (63 s)
   were, followed by the dense point cloud (74 s), the neighbour tables (47 s) and the
   tetrahedralisation (41 s). The graph build, the tables and the kd-tree were then fixed on the
   CPU, exactly ([docs/11](docs/11-meshing-cpu.md)), in v0.2.10 the max-flow itself moved to
   the GPU ([docs/12](docs/12-gpu-maxflow.md)): 109 s to 5.3 s, the same labelling, and the two
   visibility passes followed ([docs/13](docs/13-gpu-visibilities.md)): 56 s to 13 s, nanoflann's
   own tree walked on the device. The fusion went last, in v0.2.16, and again the name was
   misleading: "Load depth maps and add points" spent 60 % of its time in a gaussian blur, not
   loading anything - 108.7 of 180.8 thread-seconds against 65.5 for all three EXR reads. On the
   device it is 17.69 s to 3.06 s, reproducing OpenImageIO rather than approximating it, which
   needed the edge policy, the accumulation order and FMA contraction all matched (HIP defines
   `__fmul_rn` and `__fadd_rn` as the plain operators, so the compiler contracts them right back
   unless told not to). Meshing is 101 s; what is left is the tetrahedralisation.
3. ~~**Texturing.**~~ Done in v0.2.8 ([docs/10](docs/10-gpu-texturing.md)): the per-camera
   Laplacian pyramid and rasterisation on the GPU, the camera selection parallel, images read
   ahead; 165.8 s to 79.3 s on the engine bay job. The UV unwrap followed in v0.2.16, on the CPU
   and byte-identical: `ChartRect::insert` walked the whole tree for each of 69,565 charts
   because occupied leaves return `nullptr` but are still visited (6.28 s to 0.66 s, each node
   now carrying the largest free extent beneath it), and `packCharts` built 7,029,609
   single-element `std::vector`s where one contiguous array of 12-byte PODs does (3.31 s to
   1.90 s). Together 14.41 s to 8.56 s, and 51.9 s to 16.8 s on a 2011 Bulldozer - seven million
   allocations and an O(n^2) pointer chase punish a slow memory subsystem hardest. What is left
   is mesh load and save, padding and the Lanczos downscale (OpenImageIO's `sinf`-based filter,
   not reproducible bit for bit on a GPU).
4. ~~**FeatureExtraction (SIFT) on the GPU.**~~ Done in v0.2.13 ([docs/14](docs/14-gpu-sift.md)):
   PopSift built as HIP, 2021 s to 26 s on the 41-view set. This one is not bit-identical to the
   CPU and cannot be - it is a different descriptor implementation, not the same kernels on
   another device - so the test is the reconstruction: 88,463 landmarks against the CPU path's
   78,372 at a lower reprojection error, and within 0.11 % of each other across RDNA1, RDNA2 and
   RDNA4 on two runtimes. Found an upstream bug on the way
   ([popsift#193](https://github.com/alicevision/popsift/issues/193)).
5. ~~**Geometric verification on the GPU.**~~ Done in v0.2.15, on the CPU, and the profile is
   the whole story ([docs/15](docs/15-acransac-cpu.md)). AC-RANSAC was 566 s of
   FeatureMatching's 644 s, so it looked like the next port. Measuring first put 48 % of it in
   one `std::sort` - of `(residual, index)` pairs whose indices are read on 0.026 % of sorts -
   and only 8 % in the per-element virtual dispatch a code read had blamed. Sorting 8-byte keys
   with a radix on the double's bit pattern, and going straight to the error functor, gives
   566 s -> 351 s with a byte-identical match file and identical iteration, sort and
   improved-model counts. The port is not planned: iterations are not independent (the sampling
   pool becomes the current inliers once a meaningful model appears), so the only parallelism is
   across pairs, and 111.7 M sequential iterations against a per-workgroup budget of about a
   microsecond make 2-3x a lot of device code for less than two functions bought.
6. **The SVD in `Nullspace2`**, about 137 s of FeatureMatching and the largest single item left
   anywhere. Eigen's `JacobiSVD` is solving a 9x9 for the fundamental matrix's null space when
   only the last two singular vectors are wanted. There is no free lunch in it: the two changes
   that would be byte-identical are already upstream, since `Nullspace`/`Nullspace2` ask for
   `ComputeFullV` alone and the minimal solver already hands them a fixed-size `Mat9`. Every
   cheaper route left (a fixed-iteration Jacobi, `LDLT` on the normal equations, inverse
   iteration) gives a *different* vector within rounding, so unlike everything above this one
   cannot be byte-identical and has to be argued on reconstruction quality instead. That is why
   it is the first item that would be a minor bump rather than a patch, and it is held until the
   quality measurement exists.
7. **One Windows package** carrying the HIP 7.2 and HIP 6.2 builds with the launcher choosing by
   card, instead of one zip per toolchain and chip - which is also what would close the gfx1010,
   gfx1034 and APU gaps. Alongside it, x86-64 tiers (v1 / AVX / v3) for the Linux bundle, which is
   SSE2-only today.
8. **A CUDA build of the same tree** so NVIDIA users get the GPU matcher too (the source already
   compiles as CUDA; the packaging does not exist yet), and a Linux bundle built against an older
   glibc for Ubuntu 22.04 / Debian 12 nodes.
9. **Hardware still unrun:** RDNA3 discrete, the RDNA3.5 APUs, Vega, and a card without the 4-byte
   dot instruction (RX 5700, original Vega) for the matcher's fallback path.

Not planned: SfM on the GPU (incremental and sequential; Ceres's GPU solvers are CUDA-only and the
outer loop dominates anyway), and the tetrahedralisation, which would mean replacing geogram -
that is also what stands between this and a byte-reproducible mesh, since geogram renumbers its
cells from byte-identical input on every run.

## Layout

| Path | What |
|---|---|
| `hip/compat/include/cheshire/` | the CUDA -> HIP compatibility header, the memory bridge, the mipmap emulation |
| `hip/port/` | fused SGM aggregation and the other upstreamable source changes, applied by `scripts/apply_hip_patch.py` |
| `hip/tests/` | standalone HIP probes: textures, surfaces, mipmaps, bandwidth, allocation, launch overhead, the bridge |
| `scripts/` | Windows build (`build-alicevision.cmd`), Linux superbuild and bundle (`scripts/linux/`), validation runner, comparison tool |
| `docs/` | findings, toolchain notes, bridge design, port log, validation, performance, Linux build |
| `third_party/aliceVision` | upstream AliceVision, pinned as a submodule and left untouched; the HIP backend is a patch set |
| `third_party/popsift` | upstream PopSift, cloned rather than vendored and patched by `scripts/apply_popsift_patch.py`; built as HIP by `scripts/build-popsift.cmd` (`scripts/linux/build-popsift.sh`) |

## Building

* Windows (RDNA3/RDNA4): [docs/01-toolchain-windows.md](docs/01-toolchain-windows.md), then
  `scripts\build-alicevision.cmd gfx1201 install`.
* Linux (RDNA1-RDNA4): [docs/06-linux-build.md](docs/06-linux-build.md), then
  `scripts/linux/build-deps.sh` and `scripts/linux/build-alicevision.sh bundle`.
* Validate: `scripts\run-depthmap.cmd <dataset>` or `scripts/linux/run-depthmap.sh` against a
  Meshroom cache; `scripts/compare_depthmaps.py` produces the numbers and panels.

## Status

Works end to end on RDNA1, RDNA2 and RDNA4 (Windows and Linux); RDNA3 has its code object in
every bundle but no hardware run yet. GCN 4 (RX 400/500) is out: the ROCm 7.2 runtime refuses to
initialise on an RX 570 even though the kernel driver accepts it (docs/06). A production Meshroom
2023.3 node (house-pc, RX 5500 XT, Linux) runs its PrepareDenseScene, FeatureExtraction,
FeatureMatching, DepthMap, DepthMapFilter, Meshing and Texturing on the Cheshire bundle through the pairing script; the same works on Windows through
`meshroom-pair.cmd`. What
comes next is the roadmap above.

Primary repository: [git.hausofdub.com/dspl1236/Cheshire](https://git.hausofdub.com/dspl1236/Cheshire);
mirror: [github.com/dspl1236/Cheshire](https://github.com/dspl1236/Cheshire). Licensed MPL-2.0.
