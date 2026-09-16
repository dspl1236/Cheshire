# Memory bridge: measured facts and design

## Probe results (RX 9070 16 GB, PCIe 4.0 x16 on a Ryzen 5600X, Windows 11, ROCm 7.2.1)

`hip/tests/membridge_probe.hip`, 2026-09-03:

```
device: AMD Radeon RX 9070  managed=0 pageableMemoryAccess=0 concurrentManagedAccess=0 canMapHost=1

[bandwidth] kernel streaming read of 1 GB
  hipMalloc (VRAM)                585.4 GB/s
  hipMallocManaged                 26.6 GB/s
  hipHostMalloc mapped (PCIe)      24.6 GB/s
  hipHostMalloc coherent           24.6 GB/s
  hipHostMalloc noncoherent        26.6 GB/s

[copy] 1 GB host<->device
  H2D pageable 16.2 GB/s   D2H pageable 16.0 GB/s
  H2D pinned   28.3 GB/s   D2H pinned   28.1 GB/s

[oversubscribe] hipMalloc 512 MB chunks up to 24576 MB
  ... allocated 16384 MB, hipMemGetInfo free=0 MB
  ... allocated 24576 MB, hipMemGetInfo free=0 MB      <- keeps succeeding
  stopped at 24576 MB: reached limit
  last chunk (beyond VRAM)   25.9 GB/s
  first chunk (in VRAM)     619.3 GB/s
```

## What that means

1. **On Windows the driver already bridges.** WDDM backs `hipMalloc` with system RAM
   once VRAM is full (up to roughly half of system RAM). Kernels keep running; touching
   spilled memory costs ~24x in bandwidth. `hipMemGetInfo` reports 0 free from that point,
   so AliceVision's planner (`DepthMapEstimator::getNbSimultaneousTiles`, which takes
   `available * 0.8`) would refuse work that the card can actually run.
2. **Managed memory is not migrating memory here.** `managed=0`: `hipMallocManaged`
   succeeds but behaves like mapped host memory (26 GB/s, no page migration into VRAM).
   Do not build the bridge on managed memory.
3. **Host-resident memory is a 25 GB/s tier**, the same for mapped, coherent and
   non-coherent flavours. That is fast enough for read-mostly data that is touched
   sparsely (camera mipmaps sampled through textures: the depth-map kernels read a few
   texels per output pixel), and far too slow for data that is streamed every iteration
   (similarity volumes, which SGM sweeps repeatedly).
4. **On Linux (the production node) none of the WDDM behaviour exists.** ROCm `hipMalloc`
   fails hard at the VRAM limit. The bridge must place data explicitly there.

## Design

A small allocator policy layer, `bridge/`, used by the HIP depth-map backend:

* **Tiers**: `Vram` (`hipMalloc*`), `HostMapped` (`hipHostMalloc(hipHostMallocMapped)` +
  device pointer), with `hipMemcpy` staging where a kernel needs contiguous VRAM.
* **Placement by access pattern**, decided per buffer class, not per allocation:

  | Buffer class (AliceVision name) | Access pattern | Tier when VRAM is short |
  |---|---|---|
  | similarity volume (`CudaDeviceMemoryPitched<TSim,3>`) | streamed by SGM passes | always Vram; shrink tile instead |
  | depth/sim maps, normal maps (2D pitched) | streamed | Vram |
  | camera mipmap images (`DeviceMipmapImage`, T cameras) | sparse texture reads | spill to HostMapped, LRU by camera |
  | camera params (`__constant__`) | tiny | unchanged |
  | host staging (`CudaHostMemoryHeap`) | copy source | pinned (already) |

  Textures cannot be created over host memory on all drivers, so spilled mipmaps are
  kept as pitched host buffers and copied into a VRAM texture slot on demand (a second
  LRU of texture slots in front of the camera LRU). This turns "out of memory" into
  "more PCIe traffic", which at 25-28 GB/s is a few ms per 100 MB image.
* **Capacity planning**: replace the `hipMemGetInfo * 0.8` heuristic with a budget =
  VRAM budget + host budget (user-settable, default 25% of system RAM), and let the
  planner size `nbSimultaneousTiles` from the VRAM budget only while the T-camera cache
  is sized from the total. Log the split so a job's `job-settings.json` records it.
* **Hard rule**: never let a similarity volume land in host memory. If the volume for
  one tile does not fit in VRAM, reduce the tile size (AliceVision already tiles;
  `TileParams` is the knob), not the tier.

The bridge is a HIP-only addition inside the ported backend; the CUDA build path is
untouched, which keeps the port upstreamable.

## Implementation v1 (2026-09-03) — `hip/compat/include/cheshire/bridge.h`

Header-only, wired through the compat layer: AliceVision's `memory.hpp` calls
`cudaMalloc` / `cudaMallocPitch<T>` / `cudaMalloc3D` / `cudaFree`, and in a HIP build
those names resolve to `cheshire::bridge::{malloc, mallocPitch, malloc3D, free}`.
No upstream source change.

* VRAM first; on `hipErrorOutOfMemory` **or** when a soft cap would be exceeded, the block
  is allocated as pinned mapped host memory and the *device* pointer is returned, so
  kernels and texture objects use it unchanged.
* A registry (device ptr -> host ptr, bytes) makes `cudaFree` release the right thing and
  keeps live VRAM / host byte counts (`bridge::stats()`, `bridge::logSummary()`).
* Knobs: `CHESHIRE_BRIDGE=0` (off), `CHESHIRE_BRIDGE_VRAM_MB` (soft VRAM cap; the only way
  to trigger spilling on Windows, where WDDM never reports OOM), `CHESHIRE_BRIDGE_HOST_MB`
  (spill cap, default 25 % of RAM), `CHESHIRE_BRIDGE_LOG=1`.
* Host pitch = row bytes rounded up to the device texture pitch alignment (>= 256 B).

`hip/tests/bridge_test.hip` on the RX 9070 with a 1 GB cap: 3 x 700 MB pitched buffers ->
1 in VRAM, 2 spilled; kernel writes + `tex2D` reads over a spilled buffer verified
bit-exact; 3D volume path exercised; registry drains to zero after `cudaFree`.

Not yet done (v2): tier-aware placement per buffer class (keep similarity volumes in
VRAM by shrinking tiles instead of spilling them) and a host-resident camera-mipmap
tier in `DeviceCache`; the planner (`getNbSimultaneousTiles`) still sizes from
`hipMemGetInfo * 0.8`.

## First real-workload spill test (2026-09-03)

`CHESHIRE_BRIDGE_VRAM_MB=1500` on the 6-view set forces almost every depth-map buffer past the
soft cap (188 spills, 5.4 GB of host memory live at the peak):

| GPU | uncapped | 1.5 GB VRAM cap | output |
|---|---|---|---|
| RX 9070, Windows, PCIe 4.0 | 17.4 s | 47.7 s (2.7x) | bit-identical |
| RX 6750 XT, Linux (house-pc, 5 GT/s x16 link, i3) | 31.0 s | 161.9 s (5.2x) | bit-identical |

So the bridge does what v1 promised: a job that would not fit still finishes with exactly the
same result, at PCIe speed. On the node the first attempt stopped at 3.7 GB because the default
host budget is 25 % of RAM (14 GB box); `CHESHIRE_BRIDGE_HOST_MB=7000` lifted it. Two lessons
for v2: the slowdown is dominated by similarity volumes landing in host memory (v1 spills
whatever comes when the cap is hit - the tier rules in the design above are not implemented
yet), and the planner still sizes tile parallelism from `hipMemGetInfo`, so it never chooses
to spill on purpose; the cap only helps when the planner's estimate is wrong.

## v2 (2026-09-04): placement by class, a planner that knows about the bridge

v1 spilled whatever happened to be allocated after the cap was hit. v2 (`bridge.h`) knows what
it is placing, and the DepthMap planner sizes the work from the bridge's budgets. The policy
below is the one the measurements dictated, not the one the design above predicted: the
design had camera images as the cheapest class to spill; they turned out to be the worst by
far. Measure first.

* **Classes.** Every allocation is tagged: `Volume` (3D pitched: SGM / Refine similarity
  volumes), `Map` (2D pitched: depth / sim / normal maps, per-tile working buffers), `Image`
  (camera mip levels, tagged through a thread-local `ClassScope` by the mipmap emulation) and
  `Other`. Per-class live / peak bytes are kept and printed at exit with `CHESHIRE_BRIDGE_LOG=1`.
* **Cap by default.** The soft VRAM cap defaults to 90 % of the VRAM free when the bridge first
  runs (`CHESHIRE_BRIDGE_VRAM_FRACTION`, or `CHESHIRE_BRIDGE_VRAM_MB`; `0` = v1's no-cap). On
  Windows that keeps WDDM from paging behind our back; on Linux it keeps `hipMalloc` from failing.
* **Placement.** `Image` is VRAM-only. The planner reserves VRAM for the camera images of every
  simultaneous R camera (`bridge::setReserve`), and `Volume` / `Map` / `Other` take VRAM only
  under the cap minus that reserve, then mapped host memory. `CHESHIRE_BRIDGE_HOST_CLASSES=a,b`
  forces classes to host (the experiment knob); `CHESHIRE_BRIDGE_IMAGE_SPILL=1` lets images
  spill after all.
* **Planner.** `DepthMapEstimator::getNbSimultaneousTiles` (patched, `hip/port/bridge_v2/`)
  applies the upstream formula to `bridge::budget()` instead of `hipMemGetInfo * 0.8`, and
  never goes below one tile: when a single tile plus its images does not fit, it runs one tile
  and lets its volumes and maps spill rather than refusing the job (`CHESHIRE_BRIDGE_PLANNER=1`,
  default; `0` = upstream heuristic; `2` = the experiment where images go to host).
* **Linear mip levels.** Array memory cannot live in host RAM, so the mipmap emulation gained a
  second level storage: `CHESHIRE_MIPMAP_STORAGE=linear` puts each level in a pitched linear
  buffer sampled through a `hipResourceTypePitch2D` texture (the mip builder's per-level
  textures are redirected the same way). `hip/tests/linear_tex.hip`: same half4 data,
  normalized coordinates, bilinear filter: **bit-exact** against the array texture, in VRAM
  (578 vs 558 Gtexel/s on a 9x9 patch pass) and over mapped host memory (32 Gtexel/s).

### Measured: what each class costs behind PCIe

RX 9070, 6 views, `scripts/bridge_matrix.py`; every row bit-identical to the uncapped run and
PASS against the CUDA reference. Two allocations of the host tier are shown, because the
choice turned out to matter more than the class: the fine-grained (coherent) mapping the
bridge started with, and the coarse-grained (`hipHostMallocNonCoherent`) mapping it ships
with, which the GPU is allowed to cache. Full tables with per-class peaks:
`docs/validation/bridge-v2/rx9070-mini6-run2.md` (fine-grained) and `rx9070-mini6-run7-noncoherent.md`.

| build / placement | fine-grained host | coarse-grained host (shipped) |
|---|---|---|
| native mipmaps, everything in VRAM (the Windows release) | 17.9 s | 18.0 s |
| emulated mipmaps, array levels, everything in VRAM (what Linux runs) | 20.6 s | 20.4 s |
| emulated, linear levels, everything in VRAM | 20.8 s | 20.2 s |
| maps in host RAM (879 MB) | 35.9 s (1.7x) | 25.5 s (1.3x) |
| similarity volumes in host RAM (6.0 GB) | 96.4 s (4.7x) | 76.6 s (3.8x) |
| camera images in host RAM (186 MB) | 460 s (22x) | 20.4 s (1.0x) |
| everything in host RAM (7.0 GB) | 543 s (26x) | 84.2 s (4.2x) |

With fine-grained memory every texture fetch crosses PCIe: images, sampled as random texture
fetches by every similarity kernel, cost 22x for 186 MB while the streamed volumes cost 4.7x
for 6 GB. With coarse-grained memory the device caches host memory like its own, the camera
levels' working set lives in the RX 9070's 64 MB Infinity Cache, and images behind PCIe cost
nothing measurable; volumes and maps still pay for their streaming traffic. The emulation
costs 15 % against native mipmaps on this card, and linear levels cost nothing against array
levels, so linear is the Linux bundle's default storage (`CHESHIRE_MIPMAP_STORAGE=array` restores arrays).

The policy stays conservative: images resident, volumes then maps spill. The 22x number is
what any bridge gets by default on Linux (fine-grained is what `hipHostMalloc` gives), and
the RX 6750 XT results below show how far a 5 GT/s x16 link stretches the other classes.

### Measured: the planner is worth more than the spill

v1 capped VRAM at 1.5 GB with the planner still sizing 24 tiles from `hipMemGetInfo`:
188 spills, 47.7 s. Same cap, v2 planner:

| VRAM cap | planner | simultaneous tiles | spills | DepthMap |
|---|---|---|---|---|
| none | upstream heuristic | 24 | 0 | 20.2 s |
| 1.5 GB | 0 (v1 behaviour, images now spillable) | 24 | 504 (6 GB volumes + 879 MB maps in host RAM) | 83.5 s |
| 1.5 GB | 1 | 2 | 0 | 19.9 s |
| 4 GB | 1 | 8 | 0 | 19.9 s |
| 8 GB | 1 | 18 | 0 | 20.3 s |

Tile parallelism buys nothing on this GPU (2 tiles run as fast as 24), so keeping everything
resident and running fewer tiles at once is free. A card with 1.5 GB to spare runs the job at
full speed where v1 ran it 2.8x slower (4.2x with the final allocator, since it spills more).
The planner change is upstreamable on its own.

### Measured: the final policy below one tile

Same card, caps chosen so that the whole job, then one full R camera, then a single tile no
longer fit (one tile with its images needs 669 MB here). Final allocator;
`docs/validation/bridge-v2/rx9070-mini6-run8-final.md`.

| VRAM cap | what the planner did | spills | DepthMap | bit-identical |
|---|---|---|---|---|
| none (native mipmaps) | 24 tiles | 0 | 17.9 s | yes |
| none (emulated, linear) | 24 tiles | 0 | 20.2 s | yes |
| 1.5 GB | 2 tiles | 0 | 19.7 s | yes |
| 1 GB | 1 tile | 0 | 19.7 s | yes |
| 700 MB | 1 tile, 29 MB of maps spill | 191 | 22.3 s | yes |
| 500 MB | 1 tile, 155 MB of volumes + 45 MB of maps spill, images resident | 197 | 46.9 s (native build: 40.5 s) | yes |
| 500 MB, planner off | 24 tiles, 6 GB of volumes + 879 MB of maps spill | 504 | 83.7 s | yes |

Down to 1 GB the job runs at full speed. Below one tile it degrades gracefully instead of
failing (AliceVision's own planner throws "Not enough GPU memory to compute a single tile"),
and the images never leave VRAM. Next lever for that regime is smaller tiles
(`tileBufferWidth/Height` 512 quarters the volumes), which the planner could choose itself.

### Measured on Linux: RX 6750 XT, PCIe 2.0-speed link (house-pc), and a bug the matrix caught

Same matrix on the production node, first with the fine-grained host tier
(`docs/validation/bridge-v2/rx6750xt-mini6-run1.md`) and then with the shipped coarse-grained
one (`rx6750xt-mini6-run5-final.md`); 14 GB box, `CHESHIRE_BRIDGE_HOST_MB=7000` where 6 GB of
volumes have to fit:

| placement | fine-grained host | coarse-grained host (shipped) |
|---|---|---|
| array levels, everything in VRAM (v0.1.0 behaviour) | 31.1 s, identical | 31.1 s, identical |
| linear levels, everything in VRAM | 31.6 s, identical | 31.2 s, identical |
| maps in host RAM (879 MB) | 58.1 s, **wrong output** | 50.2 s (1.6x), identical |
| similarity volumes in host RAM (6.0 GB) | | 174 s (5.6x), identical |
| camera images in host RAM (186 MB) | 2529 s (81x), identical | 40.0 s (1.3x), identical |
| 4 GB / 1.5 GB VRAM cap, v2 planner | 30.9 s / 30.5 s, identical | 30.6 s / 30.5 s, identical |
| 700 MB cap (one tile, 29 MB of maps spill) | 30.6 s, **wrong output** | 31.1 s, identical |
| 500 MB cap (one tile, volumes + maps spill) | | 119 s (3.8x), identical |
| 1.5 GB cap, planner off (v1 behaviour) | 292 s, **wrong output** | 191 s (6.1x), identical |

Two things this said before the allocator changed. First, with fine-grained memory the node's
5 GT/s x16 link (PCIe 2.0 bandwidth; `docs/upstream/house-pc-pcie.txt`) made the image case 81x instead of 22x; coarse-grained memory turns that into 1.3x because
the device caches it. Second, every Linux run in which a *map* lived in fine-grained host
memory produced wrong depth maps (median depth error 17-29 %, extra "valid" pixels), while
the same cases were bit-identical on Windows, and volumes or images in host memory were fine
on both.

The first two suspects were ordering: ROCm may execute a copy or memset whose pointer is
host-resident on the CPU at the call instead of behind the stream's kernels, so the compat
layer now stream-orders any copy or memset that touches a spilled block
(`cheshire::bridge::inSpilled`, a range lookup). Correct in principle, no effect on this bug.
The actual cause is **GPU atomics on fine-grained host memory**. `hip/tests/host_atomics.hip`
on the RX 6750 XT under Linux:

| memory | `atomicMin` | `atomicAdd` | store / memset ordering |
|---|---|---|---|
| VRAM | ok | ok | ok |
| mapped host, default (fine-grained, coherent) | **wrong on every element** | ok | ok |
| mapped host, `hipHostMallocNonCoherent` (coarse-grained) | ok | ok | ok |

(All four placements pass on the RX 9070 under Windows.) The fused SGM aggregation
`atomicMin`s into a per-row accumulator, a Map-class buffer, and atomic min is not among the
operations PCIe carries as atomics, so on fine-grained system memory it is silently lost.
The bridge now allocates its host tier `hipHostMallocMapped | hipHostMallocNonCoherent`: the
device caches it, atomics resolve in L2, and the CPU only ever reads spilled blocks through
`hipMemcpy` after a synchronisation, which is where coarse-grained memory becomes visible.

### RDNA1 on the final build (RX 5500 XT, 2026-09-15)

First run of bridge v2, linear levels and the v0.2.1 bundle on RDNA1 (`docs/validation/bridge-v2/rx5500xt-mini6-final.md`);
every row bit-identical to the RDNA1/RDNA2 v0.1.0 output. The atomics probe reproduces the
fine-grained-memory failure on this card exactly as on the RX 6750 XT.

| placement | DepthMap | vs VRAM |
|---|---|---|
| everything in VRAM (v0.1.0: 60.7 s) | 61.1 s | |
| 4 GB / 1.5 GB / 700 MB cap, v2 planner | 60.1 / 59.9 / 61.0 s | 1.0x |
| camera images in host RAM (coarse-grained) | 110 s | 1.8x (no Infinity Cache on RDNA1; RDNA2 1.3x, RDNA4 1.0x) |
| maps in host RAM | 90 s | 1.5x |
| similarity volumes in host RAM | 347 s | 5.7x |
| 500 MB cap (one tile, volumes + maps spill) | 248 s | 4.1x |

### Full-resolution depth maps (downscale 1), 2026-09-16: two more fixes and a Windows check

Meshroom's "standard" preset runs DepthMap at downscale 2. At downscale 1 every tile carries four
times the volume memory, which is where a cap bites hardest, so the 41-view set was run at
`CHESHIRE_DOWNSCALE=1` on both cards, uncapped and under 2 GB and 1 GB caps.

| RX 9070, Windows (native mipmaps) | time | tiles in flight | spills |
|---|---|---|---|
| uncapped | 499.6 s | 30 | 0 |
| 2 GB cap | 558.7 s | 1 | 0 |
| 1 GB cap | 2871.6 s | 1 | 6061 (maps + 3 volumes) |

| RX 6750 XT, Linux (emulated mipmaps: images are bridge-tracked) | time | spills |
|---|---|---|
| uncapped | 744.1 s (20 tiles) | 0 |
| 2 GB cap, v0.2.3 bridge | 863.0 s | 6047 |
| 2 GB cap, + 4 MB spill floor | 854.7 s | 1052 |
| 2 GB cap, + reserve net of resident images | **749.9 s** | 0 |
| 1 GB cap, + 4 MB spill floor | 4035.1 s | 1061 |
| 1 GB cap, + reserve net of resident images | 4027.0 s | 1061 |

Every capped run is bit-identical to its uncapped run on Windows (41 / 41, and a second uncapped
run is bit-identical to the first). On Linux 40 / 41: one view differs in 110 pixels out of 12.2
million by a relative 1e-6 inside an 8 x 27 patch, masks unchanged; not a placement error (that
would be a tile-shaped block), and Windows never shows it. Left open as Linux-only float
accumulation-order noise.

What the series taught, and the two fixes it produced:

* **Sub-megabyte spills.** Under the Linux 2 GB cap the bridge pushed 3240 maps of under a
  megabyte to host memory, where every kernel read crosses the link, for no memory gain.
  `CHESHIRE_BRIDGE_MIN_SPILL_MB` (default 4): allocations under it always stay in VRAM. It cut the
  spill count six-fold but only 1 % of the time, which pointed at the real cost.
* **The image reserve counted resident images twice.** The planner reserves VRAM for the camera
  images it expects; on Linux those images are bridge-tracked, so once loaded they were counted
  both as live bytes and as the reserve, and 93 MB maps were spilled with 1.4 GB of VRAM empty.
  The reserve is now net of the Image class's live bytes. With it the 2 GB cap costs nothing on the
  RX 6750 XT (749.9 s vs 744.1 s uncapped, zero spills).
* **The 1 GB regime is the link.** At 1 GB the SGM volumes themselves live behind PCIe (host peak
  only 400 MB, but read at 28 GB/s instead of 615), and the run is 4-5x slower on both cards. That
  is the price of running at all where upstream refuses; the planner's one-tile floor is right,
  the spill placement is right, and nothing short of the tile size shrinking will change it.
* **Windows spills are real system memory.** Task Manager reports the capped process at 14 GB of
  "dedicated GPU memory" while the bridge holds 0 MB of VRAM. `hip/tests/host_backing.hip` settles
  it: 4 GB of `hipHostMallocMapped | NonCoherent` reads at 28 GB/s (VRAM: 615 GB/s) and leaves
  `hipMemGetInfo` unchanged. The Task Manager figure is a WDDM commitment number (it shows dwm.exe
  at 6.7 GB too), not residency.
* **Host RAM.** The depth map process reached 31 GB resident at downscale 1 with 400 MB of bridge
  spill: that is AliceVision's own full-resolution image cache, not the bridge. A 16 GB node should
  stay at downscale 2, whatever the GPU.
