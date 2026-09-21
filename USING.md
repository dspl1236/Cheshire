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
| `CHESHIRE_SIFT_SORT=0` | keep GPU SIFT keypoints in the order the card finished them (v0.3.2 sorts them by position, scale and orientation, so `.feat` files - and everything downstream - are a function of the image alone) |

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
| `CHESHIRE_MAXFLOW_CHECK=1` | `max-flow check: ... cells labelled differently: 0 of N` - the labelling is the verdict; the two float flow totals it also prints are never equal and are not the test |
| `CHESHIRE_GPU_VIS_CHECK=1` | `GPU knn check: identical to nanoflann on all N queries` |
| `CHESHIRE_SEGMENT_CHECK=1` | `segmentFullOrFree check: identical to upstream on all N cells` |
| `CHESHIRE_GPU_TEDGE_CHECK=1` | `tedge check: cells with on != 0: cpu N, gpu N` - the counts must match; the sums differ by an ulp of summation order |
| `CHESHIRE_GPU_VOTE_LOG=1` | includes `facet weight check: N facets, differing from the sequential computation: 0` |
| `CHESHIRE_GPU_BLUR_CHECK=1` | how many pixels differ from OpenImageIO, and the worst |

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

Everything else produces the same output as the unmodified CPU build; that is checked per release
against reference values ([docs/04](docs/04-validation.md)). There are about fifty more
`CHESHIRE_*` variables in the code for profiling and debugging - they are in the docs for each
stage, and none of them are needed to use this.

### Where a setting lives, and how to find one that is not listed here

Three layers, set in three different places:

| layer | examples | where you set it |
|---|---|---|
| **Meshroom node parameters** | `downscale`, `tileBufferWidth`, `sgmDepthListPerTile`, `maxPoints`, `textureSide` | the node's attribute panel in Meshroom (some sit behind the *advanced* toggle), or `meshroom_batch --paramOverrides Node:attr=value`. The attribute path is not the command-line flag: `--sgmDepthListPerTile` is `DepthMap:sgm.sgmDepthListPerTile` and `--tileBufferWidth` is `DepthMap:tiling.tileBufferWidth`, but `Meshing:maxPoints` is top level. The authority is `<Meshroom>/lib/meshroom/nodes/aliceVision/<Node>.py` |
| **`CHESHIRE_*` environment variables** | every switch, option and tuning value in this section | the environment the node starts in. **Windows:** start Meshroom or `meshroom_batch` from a shell where the variable is set - the paired launcher inherits it and passes it through. **Linux:** the same, and the wrapper also sources `<bundle>/../env.sh`, so a file beside the bundle is the persistent place. Standalone: set them before `cheshire-run.cmd <node> …` |
| **Tested combinations** | `base`, `tiles`, `coarse`, `verify`, `bridgecap`, `ds1`, `blast` | `CONFIGS` in [`scripts/verify_end_to_end.py`](scripts/verify_end_to_end.py): each is a parameter set plus an environment that is known to run end to end, and a copyable example of the two layers above together. `verify_end_to_end.py … <name>` runs one |

To find any knob that is not on this page: every one is a `getenv("CHESHIRE_…")` in the source,
with its default beside it -

```
grep -rn 'getenv("CHESHIRE_' third_party/aliceVision/src hip/
```

- and the doc for the stage that owns it ([docs/07](docs/07-gpu-matcher.md) through
[docs/17](docs/17-svd-nullspace.md)) explains what it does and what was measured when it was added.

## 6. Getting help

Open an issue with the node's log (the `[cheshire]` lines especially), the output of
`cheshire-run.cmd --which`, your card, and your driver version. If the target you landed on is in
`gpu/<family>/UNTESTED`, say so - that is the most useful report this project can get.
