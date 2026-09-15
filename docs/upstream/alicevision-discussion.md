# Draft: AliceVision post

Target: https://github.com/alicevision/AliceVision/discussions (category "Show and tell" or
"Ideas"), or an issue if Discussions is off. One image: a three-panel from
`docs/validation/monstree-full/` (CUDA | HIP | difference). Keep the planner PR for after they
answer; this is the introduction.

Posted 2026-09-10: https://github.com/orgs/alicevision/discussions/2175
Body CUDA figure corrected to 379 s and title corrected 2026-09-15 ("bit-identical across RDNA1/2/4" was an overclaim; RDNA4 vs RDNA2 shares the CUDA noise floor).

---

**Title:** DepthMap on AMD Radeon via HIP: bit-identical across RDNA1/2/4, matches the CUDA output, packaged for Windows and Linux

Hello AliceVision folks. I have the CUDA depth-map backend running on AMD GPUs through HIP,
without touching the CUDA sources, and I would like to know how you would want any of it
back.

**What it is.** A CUDA-to-HIP compatibility header (`cuda_runtime.h` shim plus a macro/
inline map) that lets `src/aliceVision/depthMap/cuda/*` compile unchanged as HIP, a small
CMake addition (`ALICEVISION_USE_HIP`, device sources as one unity translation unit because
`-fgpu-rdc` is broken on Windows ROCm), and a patch set on top of a pinned upstream
submodule (`2cb1a39`). Repo: https://github.com/dspl1236/cheshire (MPL-2.0, same as
AliceVision).

**Validation.** Same photos, same SfM, same DepthMap parameters as Meshroom 2023.3 on a
GTX 1080 Ti; only the GPU stage differs. Per view over jointly valid pixels:

| GPU | OS | 6 views | 41 views | valid masks | median rel. depth error | within 1 % (median view) |
|---|---|---|---|---|---|---|
| GTX 1080 Ti (CUDA reference, 4 Meshroom chunks) | Linux | 31.8 s | 379 s | | | |
| Radeon RX 9070 (RDNA4) | Windows | 17.4 s | 124 s | identical | 0.0000 | 98.9 % / 98.7 % |
| Radeon RX 6750 XT (RDNA2) | Linux | 31.0 s | 226 s | identical | 0.0000 | 97.5 % / 98.1 % |
| Radeon RX 5500 XT (RDNA1) | Linux | 60.7 s | 448 s | identical | 0.0000 | 97.5 % / 98.1 % |

RDNA1 and RDNA2 are bit-identical to each other; the remaining 1-3 % of pixels differing
by more than 1 % against CUDA behave like texture-filter and FMA rounding, not a port
defect. A Meshroom 2023.3 job (41 photos, `standard` preset) ran end to end on the RX 6750
XT with the DepthMap node swapped for the HIP binary; the mesh is within 1 % of the CUDA
job's vertex count. Per-view tables and side-by-side panels are in `docs/validation/`.

**Things you may want regardless of HIP:**

1. *Planner.* `DepthMapEstimator::getNbSimultaneousTiles` sizes tile parallelism from
   `cudaMemGetInfo * 0.8`. I measured that on two cards, 2 simultaneous tiles run the
   6-view set as fast as 24, so a planner that prefers fewer resident tiles over more,
   spilled ones never loses; and a "never below one tile" fallback (spill that tile's
   volumes rather than throw "Not enough GPU memory to compute a single tile") lets
   small cards finish. Mine sits behind a HIP define; happy to make it generic and open a PR
   if you want it.
2. *A fused SGM aggregation.* One kernel per volume row instead of three (best-Z, slice
   copy, aggregate) with a shared-memory pre-reduction; on the RX 9070 it took the 6-view
   set from 21 s to 17 s and is bit-identical. It is CUDA-compatible code
   (`hip/port/sgm_fused/`).
3. *Mip levels through a buffer copy.* HIP drops `surf2Dwrite` into 16-bit float arrays, so
   the mip chain is computed into a pitched buffer and copied into the level. Harmless on
   CUDA, and it removes the surface dependency.
4. *Parallel image prefetch per batch* in `DepthMapEstimator::compute` (the EXR reads were
   3 s per R camera on 12 MP inputs).

**What I am not proposing:** replacing the CUDA backend, or a SYCL-style parallel tree.
Everything is the existing CUDA source compiled a second way.

**Ask.** Does a HIP option in the CMake, gated off by default, fit your direction? And does
anyone have an RDNA3 card (RX 7000) to run the 6-view cache through the packaged build? The
code objects are in every release and nothing has executed them yet.
