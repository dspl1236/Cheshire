# Using Cheshire

Meshroom photogrammetry on an AMD Radeon card. This is the practical guide; the reasoning behind
any of it is in [`docs/`](docs/).

**On an NVIDIA card** there is a CUDA package from v0.3.0 -
`cheshire-alicevision-cuda-windows-x64-cuda12.9.zip` or
`cheshire-alicevision-cuda-linux-x64-cuda12.9.tar.gz`. It is worth having because Meshroom puts
only DepthMap on the GPU; the matcher, depth map filter, meshing votes, texturing and SIFT are on
the CPU upstream for everyone. Follow this guide with two changes: there is only one package, with
no payload to select, and the pairing needs `CHESHIRE_BACKEND=cheshire` - otherwise the launcher
sees an NVIDIA card and hands every node back to Meshroom.

## 1. Will it work on my card?

| card | supported |
|---|---|
| RX 9070 / 9070 XT (RDNA4) | yes |
| RX 7000 series, RDNA3/3.5 APUs | yes, not run on hardware here |
| RX 6000 series | yes - gfx1031 (6700/6750) run on hardware here |
| RX 6500 XT / 6400, RDNA2 APUs | yes, not run on hardware here |
| RX 5500 / 5500 XT | yes, run on hardware here |
| RX 5600 / 5700 | yes, not run on hardware here |
| Vega, Polaris (RX 500), older | Linux bundle only, Vega untested; Polaris not supported by ROCm |

"Not run on hardware here" means it compiled and is selected correctly, but nobody has put that
chip through a reconstruction. Those targets are listed in `gpu/<family>/UNTESTED` inside the
package and say so on every run. If you use one, a note about how it went is genuinely useful.

**You do not choose a package.** One Windows download carries every target and works out which one
your card needs.

## 2. Install

Download `cheshire-alicevision-windows-x64.zip` from the
[latest release](https://github.com/dspl1236/Cheshire/releases/latest) and unzip it anywhere -
`C:\cheshire-alicevision-windows-x64` is used below. Inside you will find:

```
cheshire-alicevision-windows-x64\
    cheshire-run.cmd              run a node, or ask which payload your card gets
    cheshire-detect.exe           the card probe, on its own
    meshroom-pair.cmd             pair it with Meshroom
    meshroom-pair-launcher.exe    installed by the pairing; not run directly
    common\  fam\  gpu\           the payloads - nothing to do in here
```

**Open a terminal in that folder.** In Explorer, shift-right-click an empty part of the window and
choose *Open in Terminal* (or *Open PowerShell window here*); or open Command Prompt and `cd` to it:

```
cd C:\cheshire-alicevision-windows-x64
```

Do not double-click the `.cmd` files - they print their answer and close before you can read it.

Check it sees your card:

```
cheshire-run.cmd --which
```

You should get something like `hip6.2 gfx1031` or `rocm7.2 gfx12-generic`. If not, see §4.

> In **PowerShell**, a script in the current folder needs the `.\` prefix:
> `.\cheshire-run.cmd --which`. Command Prompt does not care. Either way, calling it by its full
> path works from anywhere: `C:\cheshire-alicevision-windows-x64\cheshire-run.cmd --which`.

Then pair it with an existing Meshroom 2023.3 install - still from that folder, with your Meshroom
path and the package path:

```
meshroom-pair.cmd C:\Meshroom-2023.3.0 C:\cheshire-alicevision-windows-x64
```

That swaps the node binaries Cheshire accelerates and keeps Meshroom's originals beside them, so
`meshroom-pair.cmd C:\Meshroom-2023.3.0 --unpair` puts everything back. Meshroom itself is
unchanged; run it as you always do.

**Driver:** an RX 7000/9000 card needs **Adrenalin 26.2.2 or newer**, because that half of the
package carries the HIP 7.2 runtime and older drivers refuse it. RX 5000/6000 cards use the runtime
your driver already ships and have no such floor.

### Linux

Take `cheshire-alicevision-hip-linux-x64-rocm7.2.tar.gz`, unpack it, and point
`ALICEVISION_ROOT` at it with its `bin` on `PATH`. It needs only the `amdgpu` kernel driver and
`/dev/kfd` - no ROCm install. Your user must be in the `render` group:

```bash
sudo usermod -aG render $USER     # log out and back in
ls -l /dev/kfd                    # should exist
```

`scripts/linux/meshroom-pair.sh` does the same pairing as the Windows script.

## 3. Check it is actually using the GPU

Open a node's log in Meshroom. You are looking for lines starting `[cheshire]` or `cheshire:`:

```
[cheshire] aliceVision_meshing: bundle payload rocm7.2/gfx12-generic
[cheshire] matcher: GPU brute-force L2 2-NN on AMD Radeon RX 9070 (exact)
cheshire: 18907488 rays, 11676791 cells to the GPU (votes + weakly supported surfaces)
```

No `[cheshire]` lines at all means the pairing did not take, and Meshroom is running its own
binaries. Re-run `meshroom-pair.cmd`.

A line saying `Meshroom's own binary` means an NVIDIA card was detected and Meshroom's own CUDA
build was chosen deliberately. `CHESHIRE_BACKEND=cheshire` forces the Cheshire package instead -
which is what you want if the package you paired is itself a **Cheshire CUDA** one, since there the
automatic choice hands every node back to Meshroom and the package never runs.

## 4. When it goes wrong

| symptom | cause | fix |
|---|---|---|
| `no AMD GPU with a matching payload` | no Radeon, or no payload for this chip | run `cheshire-detect.exe -v .` - it names what each runtime answered |
| "No CUDA-Enabled GPU" / `hipErrorNoDevice` | driver too old for the HIP 7.2 runtime | update to Adrenalin 26.2.2+ (RX 7000/9000 only) |
| exits `0xC0000135`, no message | missing MSVC runtime on a machine without Visual Studio | current packages bundle it; if you are on an old one, take the latest release |
| exits `0xC000001D` (illegal instruction) | AVX2 build on a pre-2013 CPU | RX 5000/6000 half is `/arch:AVX` and works; an RX 7000/9000 card on such a CPU is not supported |
| exits `0xC0000005` when detecting | a HIP runtime crashing on load | expected and handled - each runtime is probed in its own process, so this is reported, not fatal |
| `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (Linux) | `HSA_OVERRIDE_GFX_VERSION` is forcing your card to run another chip's code | unset it. Older setup scripts set it because the bundle only had gfx1030; it now carries native code for every RDNA1-RDNA4 chip and the override does harm |
| runs out of VRAM | the memory bridge did not spill enough | `CHESHIRE_BRIDGE_VRAM_FRACTION=0.7`, or see §5 |
| results look wrong | narrow it down with the stage toggles in §5 | each falls back to the CPU path |

## 5. Knobs worth knowing

Three kinds, and the difference is the whole point. **Switches** are on by default - every GPU
path runs unless you set it to `0`. **Options** are off by default and take `1`. **Tuning values**
are numbers with a default you can see here. An earlier version of this section said everything
was off by default, which was true of the options and the opposite of the truth for the switches.

**How a value is read (since 0.3.5, the same for every `CHESHIRE_*` variable).** For a switch or
an option, `0`, `false`, `off`, `no` (any case) or an empty value means off, and anything else
means on - so `CHESHIRE_MAXFLOW_CHECK=0` really turns the check off. Up to 0.3.4 many options
only tested whether the variable existed, so `=0` turned them *on*, and a few took only a value
starting with `1`. A tuning value that is empty or not a number falls back to its default; to
remove the VRAM cap, write `CHESHIRE_BRIDGE_VRAM_MB=0`, not an empty value. The rules live in one
header, `hip/compat/include/cheshire/env.h`.

**If something looks wrong,** turn stages back to the CPU one at a time to find which. All of
these are *on* unless set to `0`:

| switch (default: on) | `=0` gives you |
|---|---|
| `CHESHIRE_GPU_MATCHER=0` | feature matching on the CPU |
| `CHESHIRE_GPU_FILTER=0` | depth map filtering on the CPU |
| `CHESHIRE_GPU_VOTE=0` | meshing votes on the CPU |
| `CHESHIRE_GPU_MAXFLOW=0` | graph cut on the CPU |
| `CHESHIRE_GPU_BLUR=0` | the similarity-map blur through OpenImageIO |
| `CHESHIRE_GPU_TEX=0` | texturing on the CPU |
| `CHESHIRE_BACKEND=auto\|cheshire\|meshroom` | force the launcher's choice (`CHESHIRE_DEPTHMAP=cuda\|hip` is the older spelling and still works) |
| `CHESHIRE_GPU_VIS=0` | the visibility passes' nearest-neighbour search through upstream's nanoflann on the CPU ([docs/13](docs/13-gpu-visibilities.md)) |
| `CHESHIRE_GPU_VIS_BUCKETS=0` | the visibility votes through the ordered loop (one vertex range per thread) instead of vertex buckets spread over all threads; same result, slower at scale |
| `CHESHIRE_SFM_PENDING_BA=0` | upstream's incremental SfM loop exit: a resection pass that ends because no candidate view reaches the score threshold leaves the views resected since the last bundle adjustment without one, and without a node in the local-BA graph; a later edge to one of them is the `[fatal] invalid map<K, T> key` of Meshroom #2344 (three of three runs on an 884-photo set, v0.3.3 finishes the pass with that bundle adjustment and the graph never throws) |
| `CHESHIRE_BA_JACOBIANS=stride` / `=autodiff` | bundle adjustment's Jacobians through upstream's autodiff functor in one 32-wide pass (bit-identical to upstream, 1.9x on that phase) or upstream's own 4-wide passes; the default since 2026-09-23 is `analytic` (below) |
| `CHESHIRE_SFM_TASK_SEED=0` | incremental SfM draws its RANSAC samples from upstream's one generator shared across threads again, so the result follows thread scheduling; the default gives every view's resection and every track's triangulation its own generator from (seed, view or track, pass), which is what makes `CHESHIRE_SFM_DETERMINISTIC=1` possible (docs/04, 0.3.4) |
| `CHESHIRE_DEPTHMAP_ORDER=0` | DepthMap: process the cameras in index order; the default orders them as a nearest-neighbour tour over their centres so a chunk's views share their neighbours and the once-per-batch loader has something to reuse (docs/04, 0.3.4 "the depth-map node was decoding") |
| `CHESHIRE_DEPTHMAP_BLOCK=12` | Meshroom 2023.3 only, read by the paired DepthMap node override: views per chunk (default 48; 0 or 12 for Meshroom's own). Each chunk is a process with a cold cache and its own SfM load (docs/04, 0.3.4 item 7) |
| `CHESHIRE_PDS_EXR_COMPRESSION=zips` | PrepareDenseScene's EXR compression as `method[:level]`; the default is now `zip:1` (16 scanlines per block at zlib level 1: same pixels, 1.4 % larger than level 4, 30 % less write CPU, and cheaper to read in every later node) where upstream writes `zips` at level 4; `none`, `rle`, `piz` and the lossy `dwaa`/`dwab`/`b44`/`pxr24` are accepted for experiments |
| `CHESHIRE_READ_DIRECT=0` | read 8-bit RGB JPEG and PNG files as float RGB/RGBA through OpenImageIO's whole-image passes (float read, colour conversion, alpha channel, copy); the default fills the same per-row scratch lines from the 8-bit pixels and applies the same colour processor, byte-identical and about 3x faster, which takes PrepareDenseScene's read from 139 to 67 thread-seconds on the engine bay (docs/04 6n) |
| `CHESHIRE_READ_DIRECT_CHECK=1` | with the direct read: read every such image both ways, keep upstream's result and print "direct 8-bit read check: N of N images identical" at exit |
| `CHESHIRE_GPU_JPEG=1` | (0.3.5, off by default) in the direct read, decode JPEG photos on the GPU with CheshireJPG, pixels identical to libjpeg-turbo; a file it does not take is read as before ([docs/04](docs/04-validation.md), 6r) |
| `CHESHIRE_GPU_JPEG_CHECK=1` | with it: decode each photo through OpenImageIO too, keep OpenImageIO's pixels on a difference, print "GPU JPEG check: N of N images identical to libjpeg-turbo" at exit |
| `CHESHIRE_GPU_VIS_BACKPROJECT=0` | Meshing's visibility passes: the host backprojects every depth-map pixel; the default stages the depth map and the device builds the queries with the host's arithmetic (bit-identical, checked under `CHESHIRE_GPU_VIS_CHECK=1`), which frees a four-thread host's largest per-camera cost (docs/04 6k) |
| `CHESHIRE_GPU_VIS_BP_FMA=0` / `=1` | with the device backprojection: the plain or the fused arithmetic forms instead of the ones this build's host compiler uses (a negative control for the check) |
| `CHESHIRE_GPU_VIS_VOTES=1` | (0.3.5, off by default) Meshing's visibility votes on the device: the decisions, the contributions folded in pixel order, and the voted vertices returned as a bitmap, so the host only appends cameras to the lists; identical to the host votes (checked under `CHESHIRE_GPU_VIS_CHECK=1`); meant for hosts with few threads (docs/04 6t) |
| `CHESHIRE_GPU_VIS_VOTES_FAIL_AT=c` | test knob: fail the device votes at camera c to exercise the recovery (the pass reruns with host votes) |
| `CHESHIRE_GPU_VIS_READERS=n` | the visibility passes read depth maps ahead on n threads (1-16, default 3); same result for any n. On a four-thread host with a fast disk 6 saved about 5 s per pass at 884 views; on a 3 Gbps disk it was slightly slower (docs/04, step 9) |
| `CHESHIRE_GPU_KNN_LAYOUT=1` | (off by default) the visibility knn kernel with the points in leaf order and the stack sized to the tree; same answers, about 1 % faster on the RX 9070 and about 50 % slower on the RX 6750 XT (docs/04 6s) |
| `CHESHIRE_MESHCLEAN_SETUP=0` | Meshing: mesh cleaning's setup with upstream's qsorts; the default builds the same lists and edge index by counting (docs/04 6l) |
| `CHESHIRE_MESHCLEAN_PRESCREEN=0` | Meshing: mesh cleaning runs upstream's pass over every point, one after another; the default screens the points in parallel, runs upstream's pass in index order only on the points that split or that an earlier split touched, and after the first pass considers only the points a split touched (the same mesh, docs/04 6i) |
| `CHESHIRE_MESHCLEAN_CHECK=1` | with the pre-screen: runs upstream's passes from the same state afterwards and compares every structure (points, triangles, colours, neighbour lists, boundary flags, edge index), printing the verdict |
| `CHESHIRE_FACET_PAIRS=0` | Meshing: binarize's facet weights computed entry by entry as upstream (each interior facet twice, 16 circumsphere centres per cell); the default computes each interior facet once from its lower-numbered cell, with the same values |
| `CHESHIRE_RESIZE_EXACT=0` | image downscales with the default filter through OpenImageIO's own resize; the default computes the same lanczos3 downscale directly with identical values, 18x less CPU (docs/04, 0.3.4 "the depth-map node was loading images") |
| `CHESHIRE_LOAD_PROFILE=1` | one line per image load (mvsUtils::loadImage) with the read, the metadata open, the exposure multiply, the colour conversion and the process downscale; what found texturing's read contention (docs/04, 0.3.4 "texturing was loading images too") |
| `CHESHIRE_TEX_READAHEAD=n` | texturing: cameras read ahead of the one on the GPU (default: hardware threads minus one, within the image cache's slots) |
| `CHESHIRE_DEPTHMAP_DEVICE_DOWNSCALE=0` | DepthMap (HIP builds): the process downscale on the host after each read; the default uploads full-resolution images and downscales on the device with the host's arithmetic (byte-identical maps) |
| `CHESHIRE_DEPTHMAP_DEVICE_DOWNSCALE_CHECK=1` | with the device downscale: compare every image with the host path (floats and texels) and print the counts |
| `CHESHIRE_EXR_PROFILE=1` | one line per EXR read through the direct reader: open time, read time, the thread's CPU time, threads used (what located the depth-map node's host downscale as its real cost, docs/04) |
| `CHESHIRE_EXR_DIRECT=0` | read EXR files through OpenImageIO; the default reads RGB/RGBA half or float EXRs straight through OpenEXR (identical pixels, about 6x faster on the 6000x3376 undistorted images) |
| `CHESHIRE_SFM_LOCAL_PASSES=1` | after every bundle adjustment under the local strategy, restrict upstream's outlier and unstable-pose passes to what the solve can have changed (exact: same digests). Off by default: at 884 views the full passes are about 40 s of the run and the restricted ones no cheaper; the case is thousands of views (docs/04, 0.3.4) |
| `CHESHIRE_BA_FUSED_PROJECTION=0` | bundle adjustment's inner projection through upstream's four-call chain instead of the fused one-walk version (pinhole with none / radial K1 / K3 / Brown distortion; other models use upstream's anyway); 1.7x on the Jacobian phase at 41 views (docs/04, 0.3.4) |
| `CHESHIRE_BA_PERSIST=0` | bundle adjustment builds its Ceres problem from scratch every solve, as upstream does; the default keeps one Problem across solves while every landmark is active (up to 100 poses) and rebuilds under the local strategy, where the moving frontier made persistence a wash (0.3.4; sets with rigs, survey points, constraints, rotation priors, temporal smoothness or depth observations use the rebuild anyway) |
| `CHESHIRE_SIFT_SORT=0` | keep GPU SIFT keypoints in the order the card finished them (v0.3.2 sorts them by position, scale and orientation, so `.feat` files and the matches are a function of the images alone; SfM itself still varies run to run within upstream's own band - same poses and landmark count, parameters differing in the fourth digit) |

The memory bridge has more knobs than the two above, and one of them has mattered on a real box
([docs/02](docs/02-memory-bridge.md) defines all of them):

| setting | effect |
|---|---|
| `CHESHIRE_BRIDGE_HOST_MB` | the host-RAM budget for spilled buffers; default 25 % of RAM, which stopped a 14 GB node until `=7000` lifted it |
| `CHESHIRE_BRIDGE_MIN_SPILL_MB` | allocations under this always stay in VRAM (default 4) |
| `CHESHIRE_BRIDGE_PLANNER=0\|1\|2` | how the tile planner fits the job under the cap; `1` (default) plans to fit and lets volumes and maps spill rather than refusing |
| `CHESHIRE_BRIDGE_IMAGE_SPILL=1` | allow camera images out to host memory; off by default because it costs about 13x behind PCIe |
| `CHESHIRE_BRIDGE_HOST_CLASSES=a,b` / `CHESHIRE_BRIDGE_VRAM_ONLY_CLASSES=a,b` | force buffer classes (`image`, `map`, `volume`, `other`) to host, or pin them to VRAM; experiment knobs |
| `CHESHIRE_FILTER_CACHE_MB` | DepthMapFilter's decoded-map cache, default 4096; `0` disables ([docs/08](docs/08-gpu-depth-map-filter.md)) |
| `CHESHIRE_PDS_THREADS`, `CHESHIRE_FUSION_THREADS` | thread counts for PrepareDenseScene's image loop and Meshing's point-cloud fusion; default one per core |
| `CHESHIRE_TEX_PARTS=1` | texturing's camera scoring done sequentially instead of in parallel parts; for comparing against the parallel order, not for speed |

**Verify rather than trust** - each of these runs the CPU reference alongside the GPU port, in
the same process, and prints its verdict. All off by default; `=1` enables. They cost time (the
CPU reference runs too) and change nothing in the output. The release gate runs all of them
together as its `verify` configuration ([docs/04](docs/04-validation.md)):

| option (default: off) | what `=1` prints in the Meshing log |
|---|---|
| `CHESHIRE_FILTER_CHECK=1` | `filterByPixSize check: identical to single-threaded upstream on all N slots` |
| `CHESHIRE_MAXFLOW_CHECK=1` | `max-flow check: ... cut values on the adjacency-list graph (double): CSR labelling X, adjacency-list labelling Y (equal, ...)` - equal cut values are the verdict: a minimum cut need not be unique, so the two labellings may differ in cells Boykov-Kolmogorov leaves undetermined (the line also counts them); the two float flow totals it prints are never equal and are not the test |
| `CHESHIRE_GPU_VIS_CHECK=1` | `GPU knn check: identical to nanoflann on all N queries`, and per pass `visibility votes check (pass N, ...): identical to the ordered host reference on all N vertices` (the votes redone from the host's own answers) |
| `CHESHIRE_SEGMENT_CHECK=1` | `segmentFullOrFree check: identical to upstream on all N cells` |
| `CHESHIRE_GPU_TEDGE_CHECK=1` | `tedge check: cells with on != 0: cpu N, gpu N` - the counts must match; the sums differ by an ulp of summation order |
| `CHESHIRE_GPU_VOTE_LOG=1` | includes `facet weight check: N facets, differing from the sequential computation: 0` |
| `CHESHIRE_GPU_BLUR_CHECK=1` | how many pixels differ from OpenImageIO, and the worst |
| `CHESHIRE_GPU_PAD_CHECK=1` | `GPU padding check: texels differing from the sequential sweeps: 0 of N` - texturing's edge padding on the device against upstream's two host sweeps (v0.3.2) |
| `CHESHIRE_SFM_DETERMINISTIC=1` | not a check but the condition for one: bundle adjustment on one Ceres thread (SfM's own loops keep every core), which with the per-task generators makes two runs of StructureFromMotion on the same input byte-identical (`--output x.sfm`; the Alembic file carries the date). `CHESHIRE_BA_THREADS=n` sets the Ceres thread count directly. Cost on 41 views in docs/04 |
| `CHESHIRE_BA_LOG_COST=1` | upstream's two `landmarksBlocks cost` log lines per bundle adjustment, each a full single-threaded residual evaluation; skipped by default since 0.3.4 (docs/04, "what a bundle adjustment costs around Ceres' Solve") |
| `CHESHIRE_BA_PERSIST_CHECK=1` | in the StructureFromMotion log, after every bundle adjustment: `BA persistent check: N residual blocks in the problem, N recorded, N expected from the scene; missing 0, extra 0, stale 0, landmarks whose Ceres residual ids differ from the record 0 - consistent` (and a `pre-sync check` line before each sync) |
| `CHESHIRE_BA_CHECK=1` | in the StructureFromMotion log, after every bundle adjustment: `BA check (stride against autodiff): ... residuals 0 of N values differ; Jacobians 0 of M values differ - identical`; with `CHESHIRE_BA_JACOBIANS=analytic` the Jacobians differ by rounding (max rel 4.4e-10 on 41 views) and the line says by how much |
| `CHESHIRE_OBJ_CHECK=1` | Meshing and Texturing also write Assimp's file beside the direct writer's (`mesh.assimp.obj`, `texturedMesh.assimp.obj`); `scripts/check_textured_obj.py` compares the textured pair by content |

**Memory.** Defaults fit the card you have; the bridge settings exist to give it *less* than it
has, never more. The bridge reads the card at every run - cap = total minus about 1.1 GB of
headroom, budget = 80 % of that - and plans full cameras, tiles and host spill from there: a 12 GB
RX 6750 XT and a 4 GB GTX 1050 Ti both ran 107 full-resolution photographs with nothing set and no
spill (docs/04). Reach for these to share the card with a display or another job, to bound the
host pool on a RAM-tight box, or to reproduce a smaller card. The bridge is a switch (on by
default); the rest are tuning values:

| setting | default | effect |
|---|---|---|
| `CHESHIRE_BRIDGE=0` | on | turn the VRAM-to-RAM bridge off entirely |
| `CHESHIRE_BRIDGE_VRAM_FRACTION=0.7` | `0.9` of free VRAM at start | use at most this fraction before spilling |
| `CHESHIRE_BRIDGE_VRAM_MB=4096` | unset | a hard cap instead of a fraction; a cap above the card's total memory is clamped (v0.3.1) |
| `CHESHIRE_BRIDGE_LOG=1` | off | show the cap, the plan, and what spilled and why |

**Speed, at a measured cost** - the only setting here that changes what is produced:

| | |
|---|---|
| `CHESHIRE_QR_NULLSPACE=1` | 1.86x on geometric filtering, for 0.12 % fewer landmarks and 0.14 % more reprojection error ([docs/17](docs/17-svd-nullspace.md)) |
| `CHESHIRE_BA_JACOBIANS=analytic` (the default) | bundle adjustment's Jacobians by the chain rule instead of autodiff: 3.0x on that phase (13.0 s to 4.3 s on 41 views, the SfM node 67 s to 57 s), for rounding-level differences from upstream (max 4.4e-10 relative) that move the final landmark count within its run-to-run spread ([docs/04](docs/04-validation.md), 0.3.4); `=stride` keeps upstream's numbers bit for bit |

Everything else produces the same output as the unmodified CPU build; that is checked per release
against reference values ([docs/04](docs/04-validation.md)). The exception is incremental SfM,
whose default output is not upstream's to the bit: the analytic Jacobians above, the fused
projection and the persistent Problem's summation order differ by rounding, and a random generator
per task draws different samples than upstream's shared one. `CHESHIRE_SFM_DETERMINISTIC=1` makes
it reproducible run to run. There are about fifty more
`CHESHIRE_*` variables in the code for profiling and debugging - they are in the docs for each
stage, and none of them are needed to use this.

### Where a setting lives, and how to find one that is not listed here

Three layers, set in three different places:

| layer | examples | where you set it |
|---|---|---|
| **Meshroom node parameters** | `downscale`, `tileBufferWidth`, `sgmDepthListPerTile`, `maxPoints`, `textureSide` | the node's attribute panel in Meshroom (some sit behind the *advanced* toggle), or `meshroom_batch --paramOverrides Node:attr=value`. The attribute path is not the command-line flag: `--sgmDepthListPerTile` is `DepthMap:sgm.sgmDepthListPerTile` and `--tileBufferWidth` is `DepthMap:tiling.tileBufferWidth`, but `Meshing:maxPoints` is top level. The authority is `<Meshroom>/lib/meshroom/nodes/aliceVision/<Node>.py` |
| **`CHESHIRE_*` environment variables** | every switch, option and tuning value in this section | the environment the node starts in. **Windows:** start Meshroom or `meshroom_batch` from a shell where the variable is set - the paired launcher inherits it and passes it through. **Linux:** the same, and the wrapper also sources `<bundle>/../env.sh`, so a file beside the bundle is the persistent place. Standalone: set them before `cheshire-run.cmd <node> …` |
| **Tested combinations** | `base`, `tiles`, `coarse`, `verify`, `bridgecap`, `ds1`, `blast` | `CONFIGS` in [`scripts/verify_end_to_end.py`](scripts/verify_end_to_end.py): each is a parameter set plus an environment that is known to run end to end, and a copyable example of the two layers above together. `verify_end_to_end.py … <name>` runs one |

To find any knob that is not on this page: every one is a `cheshire::env::` read in the source
(`flag`, `integer`, `real`, `text` or `isSet`), with its default beside it -

```
grep -rnE 'env::(flag|integer|real|text|isSet)\("CHESHIRE_' third_party/aliceVision/src hip/
```

- and the doc for the stage that owns it ([docs/07](docs/07-gpu-matcher.md) through
[docs/17](docs/17-svd-nullspace.md)) explains what it does and what was measured when it was added.

## 6. Getting help

Open an issue with the node's log (the `[cheshire]` lines especially), the output of
`cheshire-run.cmd --which`, your card, and your driver version. If the target you landed on is in
`gpu/<family>/UNTESTED`, say so - that is the most useful report this project can get.
