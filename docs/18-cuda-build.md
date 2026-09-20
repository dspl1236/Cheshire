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
