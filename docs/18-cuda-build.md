# A CUDA build of the same tree

Everything in this project except `DepthMap` is a stage upstream AliceVision runs on the CPU for
*every* vendor. The GPU matcher, GPU SIFT, the depth-map filter, the meshing votes and min-cut and
visibilities, the texturing - NVIDIA users do not get any of it from upstream either. The source
was written in CUDA dialect and compiled as HIP, so it should build as CUDA with no porting at all.

This is the work to find out, and to answer one open question properly: **does the memory bridge
earn its place on a vendor whose driver already migrates pages?**

## Pinned: CUDA 12.9.2

Not a preference - a ceiling. CUDA 13's release notes: *"Removed support for Maxwell, Pascal, and
Volta GPUs, corresponding to compute capabilities earlier than 7.5."* Both test cards are compute
6.1, and 12.9.2 is the last 12.x in NVIDIA's archive (the next entry is 13.0.0).

Verified rather than assumed - `nvcc -arch=sm_61` compiles on the installed toolkit:

    Cuda compilation tools, release 12.9, V12.9.86
    sm_61: OK

| | |
|---|---|
| test cards | GTX 1080 Ti (11 GB, house-pc, Linux), GTX 1050 Ti (4 GB, bench-pc, Windows) |
| both | Pascal, compute 6.1, so one `sm_61` target covers them |
| drivers | 580.173.02 and 581.57, both far above 12.9's floor |

**No driver rollback is needed.** CUDA's minor-version compatibility means a newer driver runs an
older runtime, and both boxes report a maximum of CUDA 13.0, so a 12.9 build runs as-is.

### Host compilers are the real constraint

| | Pascal `sm_61` | our host compiler |
|---|---|---|
| CUDA 13.x | removed | MSVC 14.50 / GCC 13 fine |
| CUDA 12.9.2 | supported | **Windows: MSVC 193x only** |

WSL Ubuntu 24.04 ships GCC 13.3 and 12.9 accepts GCC 6.x-14.x, so **the Linux side has no friction
at all**. Windows does: 12.9 tops out at MSVC 193x (Visual Studio 2022) and this machine builds with
MSVC 14.50 (Visual Studio 2026), so it needs VS 2022 Build Tools installed alongside and `nvcc
-ccbin` pointed at it. That is why Linux goes first.

A risk to smoke-test early on the Windows side rather than discover at final link: the prebuilt
vcpkg dependencies were built with MSVC 14.50, and linking them against `.cu` objects compiled
through an MSVC 193x host leans on Microsoft's binary-compatibility guarantee holding across that
boundary.

### Why not match the original benchmark's CUDA version

The reference runs used Meshroom 2023.3's own prebuilt binaries - CUDA 11.3 on Linux, 11.6 on
Windows. Matching those would mean installing GCC 10 and a VS 2019-era MSVC, and it still would not
give bit-identity, because Meshroom ships AliceVision ~3.0 and this tree is 3.4-dev plus the patches
in `patches/`. Different source.

It is also not the comparison worth having. `README.md` already records that CUDA is not
bit-identical to CUDA across versions: the same 1080 Ti under 11.6/Windows and 11.3/Linux differs on
4 of 41 views, one by 7.2 % of pixels.

The clean measurement is **this tree built as CUDA against this tree built as HIP** - same source,
same patches, same machine, only the backend differing. Nobody else can make that comparison. The
Meshroom reference stays for the user-facing "faster than what you have today" number, with the
version drift stated.

## The build should need no porting

The GPU sources ported in v0.2.5 through v0.2.16 are CUDA dialect throughout. The HIP build reaches
them by force-including the compat header, and that is applied by generator expression:

```cmake
add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:-include.../cheshire/cuda_to_hip.h>")
```

`COMPILE_LANGUAGE:HIP` - so a CUDA build never receives it and compiles the original dialect
directly. The references to `cuda_to_hip.h` inside `src/aliceVision/fuseCut/gpu/*` are comments
explaining this, not includes.

## The bridge is the open question

`hip/compat/include/cheshire/bridge.h` says of itself: *"Header-only, HIP-only."* It includes
`<hip/hip_runtime.h>` and calls `hipMalloc`, `hipMemGetInfo`, `hipMallocPitch`, `hipHostFree`
directly. Porting it is a thin shim over about ten entry points - a day, not a project.

Whether it is **needed** is the interesting part, and the answer is not obvious in our favour.
`docs/02` finding #2 is the bridge's whole reason for existing:

> `managed=0`: `hipMallocManaged` succeeds but behaves like mapped host memory (26 GB/s, no page
> migration into VRAM). Do not build the bridge on managed memory.

That is an AMD fact. CUDA's unified memory *does* migrate pages, with hardware fault handling, at
**page** granularity - finer than the bridge's whole-buffer placement by class. And a GTX 1050 Ti
already ran the 41-view set in 4 GB through upstream CUDA with no out-of-memory.

So the order is: build without the bridge, find where memory actually binds on 11 GB and on 4 GB,
and port the bridge only if something binds. Then the comparison is buffer-class placement against
page migration, which is a real question with a publishable answer either way - page migration can
thrash a volume that SGM sweeps repeatedly, and whole-buffer placement cannot.

Building it first and measuring afterwards would answer the wrong question.

## Result (2026-09-20): byte-identical to the reference

The build needed no porting, exactly as predicted: 674/674 steps, zero failures, all 14 `.cu`
files compiled unmodified. On house-pc's GTX 1080 Ti, 41 views, `--downscale 2`:

| | |
|---|---|
| depth maps vs `ref-cuda113` | **41 / 41 byte-identical** |
| sim maps vs `ref-cuda113` | **41 / 41 byte-identical** |
| wall time | **211 s vs the reference's 379 s (1.80x)** |

Byte-identical, not "within tolerance". This tree built with CUDA 12.9 for `sm_61` reproduces
Meshroom 2023.3 (AliceVision 3.1, CUDA 11.3) exactly, which means three things that were open
questions until now:

* every patch this project carries is numerically neutral on CUDA - provably, not plausibly;
* AliceVision 3.1 vs 3.4-dev contributes **zero** to depth-map numerics;
* CUDA 11.3 vs 12.9 contributes **zero** on Pascal.

The 1.80x is pure scheduling - one process with 16 simultaneous tiles against Meshroom's four
sequential chunks - not a numerical shortcut. The output is the same bytes.

### The clean backend comparison, at last

With the CUDA side pinned to the reference bit-for-bit, HIP-vs-CUDA is now unambiguous: same
source, same patches, same machine, only the backend differing.

| our CUDA vs our HIP, matching parameters | |
|---|---|
| mask agreement | 41 / 41 views >= 0.95 |
| median relative depth error | 0.0000 |
| median view within 1 % | 0.9806 |
| worst view | 0.9233 |
| max p95 | 6.2 % |

`docs/04` guessed this residual was AliceVision version drift plus FMA/texture-filter
differences. The version half is now disproven: it is entirely AMD hardware and compiler float
behaviour.

### The flag that cost four rebuilds

The first comparison scored 0.823 median-view within 1 %, worse than the HIP build managed
against the same reference - backwards, and it looked like a real defect in the CUDA port.
Four experiments chased it, each rebuilding and rerunning the 41 views:

| experiment | result |
|---|---|
| rerun the same build | 41 / 41 bit-identical - deterministic, not a race |
| `CHESHIRE_SGM_LEGACY` (upstream's three-kernel SGM) | 41 / 41 bit-identical to the fused kernel |
| buffer-copy mip levels instead of `surf2Dwrite` | 41 / 41 bit-identical |
| submodule + patch audit across v0.2.5..HEAD | depth-map estimation unchanged |

None of them moved a single byte. The actual cause was **`sgmDepthListPerTile`**: Meshroom's
node default is 1, the bare CLI default is 0, and the reference plus the HIP control were both
produced through Meshroom's full parameter set while the CUDA run was invoked with four
arguments. Comparing a defaults run against a Meshroom-parameters reference was never
apples-to-apples, and the flag changes how SGM samples depth.

The blunt lesson: **`scripts/linux/run-depthmap.sh` already encodes the node's `standard`
preset**, `--sgmDepthListPerTile True` included, and it is what produced every validated HIP
run in `docs/04`. The CUDA test hand-rolled its own four-argument invocation instead of using
it. Use the runner.

The recovery lesson, for when a comparison does look wrong: **diff the two runs' own parameter
dumps before diffing their output.** AliceVision prints every parameter with a `(default)`
marker; normalising that marker away and diffing the two logs found in one step what four
rebuilds did not. The rebuilds were not wasted - three toggles of our own patches producing
byte-identical output is a stronger correctness statement for the port than the comparison
being attempted - but they answered a question that did not need asking.

For the record, since it is the sort of thing that gets assumed: `surf2Dwrite` into 16-bit
float arrays works correctly on CUDA 12.9 / Pascal. The buffer-copy mip path is an AMD
workaround only, and carrying it costs nothing either way.

## The memory bridge on CUDA (2026-09-20)

Ported, and it answers the question above - though not in the shape the question assumed.

### How it is wired in

The bridge's logic has nothing backend-specific in it: ten entry points, all with direct CUDA
equivalents. Rather than rewrite `bridge.h` in CUDA dialect and risk changing behaviour that is
validated on five AMD cards, `cheshire/hip_to_cuda.h` maps those ten names onto CUDA - the exact
mirror of `cuda_to_hip.h`, and safe for the same reason: neither runtime's headers declare the
other's names.

Reaching AliceVision's allocations is the part that does not mirror. On HIP, `cuda_to_hip.h`
shadows `cudaMalloc` / `cudaMallocPitch` / `cudaMalloc3D` / `cudaFree` with inline functions,
which works only because the CUDA names are absent there. Two approaches that look like they
should work in a CUDA build, and do not:

* **a function-like macro** - `cudaMallocPitch<Type>(&buf, ...)` does not expand, because the
  token after the macro name is `<` and not `(`. AliceVision's main device allocation would have
  silently bypassed the bridge.
* **overloads inside `namespace aliceVision::depthMap`** - the idea being that unqualified lookup
  searches enclosing namespaces before the global one. It does, but **ADL adds the global
  namespace straight back in**: every argument type (`float2`, `__half`, `cudaPitchedPtr`,
  `cudaExtent`) lives in `::`, so all five call sites become ambiguous with the real CUDA
  declarations. The compiler said so twenty times.

So the five call sites in `memory.hpp` name the allocator through `CHESHIRE_DEV_MALLOC` and
friends (patch step 1j), which resolve to `cheshire::devmem` on CUDA, to `cheshire::bridge` on
HIP, and to plain CUDA if neither is compiled in. No CMake change is needed: `CHESHIRE_HIP` is
defined by `cuda_to_hip.h` itself, so `memory.hpp` can tell which backend it is in straight after
including `<cuda_runtime.h>`.

`cheshire/managed_cuda.h` adds the third arm, `CHESHIRE_CUDA_MANAGED=1`, routing the same five
sites to `cudaMallocManaged`. It emulates the pitched forms by hand, since unified memory has no
pitched allocator.

### What it costs when memory does not bind: nothing

41 views, downscale 2, GTX 1080 Ti. Every row byte-identical to `run-cuda-41-dlpt`, which is
itself byte-identical to the CUDA 11.3 reference.

| allocator | time | tiles | depth maps |
|---|---|---|---|
| plain `cudaMalloc` | 225 s | 16 | 41 / 41 identical |
| bridge | 226 s | 14 | 41 / 41 identical |
| unified memory | 219 s | 22 | 41 / 41 identical |

### The planner absorbs VRAM caps; it does not spill

Every natural cap spilled **zero bytes**. The planner cut tile parallelism to fit instead:

| VRAM cap | depth maps x tiles | volume peak | time | depth maps |
|---|---|---|---|---|
| none | 3 x 14 | 6118 MB | 225 s | 41 / 41 identical |
| 4000 MB | 1 x 5 | 2185 MB | 236 s | 41 / 41 identical |
| 2000 MB | 1 x 2 | 874 MB | 236 s | 41 / 41 identical |
| 1000 MB | 1 x 1 | 437 MB | 238 s | 41 / 41 identical |

**42 concurrent tiles down to 1 costs 6 %.** That is `docs/02`'s "tile parallelism costs nothing
to give up", now confirmed on NVIDIA, and it is the bridge's cheap lever: the same job in a
quarter of the VRAM for 6 %.

### What spilling costs when it is forced

Tile count pinned at 1 by a 1000 MB cap, so the delta is placement and nothing else:

| forced to host | time | vs VRAM | AMD equivalent (docs/02) |
|---|---|---|---|
| nothing (control) | 238 s | 1.00x | - |
| volume | 972 s | **4.08x** | 5.6x |
| map | 912 s | **3.83x** | 1.9x |
| volume + map | 1595 s | **6.70x** | - |

All byte-identical. Volumes are cheaper to spill on NVIDIA than on the RX 9070, maps dearer -
though the map row forces 1304 allocations totalling 58 MB, because naming a class bypasses the
4 MB minimum-spill threshold. It is a stress test, not a policy the defaults would produce.

### Camera images are outside the bridge on CUDA

On HIP-Windows the camera mipmaps are emulated with buffers (`mipmap_emu.h`), which is why the
bridge has an `Image` class at all. On CUDA they are real `cudaMipmappedArray`s and never pass
through `cudaMalloc`, so **the bridge cannot see or account for them and the image reserve is
inert**. The bridge manages volumes and maps - by `docs/02`'s measurements the cheap classes to
spill, not the expensive one.

This has one sharp edge, found by measurement. `CHESHIRE_BRIDGE_VRAM_MB` is read by both the
bridge's cap *and* the tile planner's budget. Set above the card's physical memory, the planner
commits more tiles than the card holds and the bridge fills VRAM to 10966 MB of 11165 trying to
honour the cap. Its own allocations spill correctly (117 spills, 0 spill failures) - and then the
camera arrays, which it does not own, hit a hard OOM with ~200 MB left. `bridge.h` now clamps a
cap that exceeds the device's **total** memory down to the default fraction of free VRAM. A cap
below total is a deliberate budget and is left exactly as asked, so the bridge-v2 matrices that
go down to 500 MB are untouched.

Verified on the configuration that failed:

```
[cheshire] bridge: requested vram cap 20000 MB exceeds this device (11165 MB total); clamping to 9911 MB
exit=0  225 s  41 depth maps  41/41 byte-identical  (0 spills, 0 spill failures)
```

Same command that previously stopped after 43 s with `CUDA Error: out of memory`.

### Bridge versus unified memory: the wrong question

The comparison cannot be run, and the reason is the answer. `CHESHIRE_BRIDGE_VRAM_MB` is both the
cap and the planner's budget, so there is no way to over-commit the tile count for the bridge
while leaving its cap honest. The two are not competing strategies for one situation:

* **the bridge prevents pressure** - it plans the job to fit the budget, and spills only when a
  single tile plus its images will not;
* **unified memory absorbs pressure** - it commits whatever is asked and migrates pages on fault.

Two attempts to force genuine oversubscription on an 11 GB card both failed, which is worth
recording so nobody repeats them. At downscale 2, 24 tiles x 437 MB came to ~10.5 GB, just under
the card, so unified memory never migrated a page. Downscale 1 does not help either:
`tileBufferWidth/Height` fix the tile size, so a finer downscale produces *more* tiles rather than
bigger ones, and per-tile cost stays ~437 MB. Forcing the issue needs a working set that genuinely
exceeds the card - many more views at downscale 1, or a raised `sgmMaxDepths` to inflate the
volume per tile.

So on this hardware and these datasets, **memory never binds**, exactly as this document
predicted before the port. What the bridge buys on CUDA is not survival, it is choice: the same
output in a quarter of the VRAM for 6 %, leaving the card free for something else. Unified memory
is a genuine alternative here in a way it never was on AMD - free when nothing binds, and it
tolerates an over-committed planner where the bridge needs its budget told the truth.
