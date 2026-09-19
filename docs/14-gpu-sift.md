# GPU SIFT for FeatureExtraction (PopSIFT on HIP)

FeatureExtraction is the largest pure-CPU node left in the pipeline: 166 s of the 107-photo engine
bay job on a 12-core host, 349 s on house-pc's i3, around 20 % of a full run. AliceVision already
has the GPU path wired, `ALICEVISION_USE_POPSIFT` with a complete describer in
`feature/sift/ImageDescriber_SIFT_popSIFT.cpp`, and the vcpkg dependencies even carry a built
PopSIFT. That library imports `nvcuda.dll`, so it needs an Nvidia driver, and this project's build
turns the option off.

## The port compiles

`hip/port/popsift/popsift_hip.h` is force-included in front of every PopSIFT translation unit, on
top of `cheshire/cuda_to_hip.h`. With it, and one patched line, all 29 sources of PopSIFT v0.10.0
compile to gfx1201 object code:

```
29 compiled, 0 failed
```

`hip/port/popsift/compile-probe.sh` is that check. Run `scripts/apply_popsift_patch.py` first; it
writes the config header upstream's CMake generates and applies the one source fix.

What the shim supplies, and why none of it needed the emulation the depth-map port did:

- **The pyramid keeps its shape.** HIP 7.2 has layered arrays, layered surfaces and 2D textures
  natively, so PopSIFT's Gaussian pyramid storage carries over unchanged. Only
  `surf2DLayeredwrite` differs: HIP's takes no boundary-mode argument, so an overload drops it.
- **The warp shuffles.** HIP's `*_sync` forms want a 64-bit lane mask because a wavefront can be 64
  wide; PopSIFT passes CUDA's 32-bit mask and HIP rejects it with a static assertion. The wrappers
  widen it. RDNA runs 32-wide wavefronts, so the mask values carry over as they are.
- **Round-toward-positive-infinity arithmetic.** PopSIFT's descriptor binning uses `__fmul_ru` and
  `__fmaf_ru`, which HIP does not have. A float product and a float fused multiply-add are both
  exact in double, so computing them there and rounding up gives exactly what CUDA returns.
- **`__constant__` symbols.** CUDA passes the symbol itself, HIP wants its address; template
  overloads take the reference and forward the address.
- **The thrust execution policy.** PopSIFT's grid filter uses `thrust::cuda::par.on(stream)` and
  rocThrust calls that namespace `thrust::hip`, so a namespace alias is enough.
- **Relocatable device code.** PopSIFT shares `__constant__` and `__device__` globals across
  translation units, which is why upstream sets separable compilation. The HIP build needs
  `-fgpu-rdc` and a device link step; without it the device linker reports the globals undefined.

One upstream bug turned up: `LinearTexture` holds a handle created by `cudaCreateTextureObject` and
read with `tex2D`, but declares it `cudaSurfaceObject_t`. CUDA makes both types `unsigned long long`
so it compiles there; HIP has them as distinct pointer types.

## It builds, links and runs

`hip/port/popsift/CMakeLists.txt` builds the library and installs a `PopSiftConfig.cmake` exporting
`PopSift::popsift`. `scripts/build-alicevision.cmd` takes `CHESHIRE_POPSIFT=ON` and points
`PopSift_DIR` at it, and AliceVision's configure reports `PopSIFT found`. The resulting
`popsift.dll` imports `amdhip64_7.dll` where the vcpkg one imports `nvcuda.dll`. Feature extraction
runs to completion on the 6-view set and writes its files.

Four things had to be solved to get there, none of them in the kernels:

- **No relocatable device code on Windows.** PopSIFT shares `__constant__` and `__device__` globals
  across translation units, so upstream switches on separable compilation. HIP cannot embed the
  device IR into a COFF object, and the failure surfaces as `llvm-objcopy: the file was not
  recognized as a valid object file`. `scripts/apply_popsift_patch.py` writes one translation unit
  that includes all 29 sources, which removes the need for the device link entirely.
- **clang-cl does not take `-include`.** It reads the header as a second source file and reports
  "cannot specify /Fo when compiling multiple source files". The MSVC spelling `/FI` works.
- **PopSIFT's own size check rejects the image.** `checkLimit_2DtexLinear` and its surface twin read
  `maxTexture2DLayered` and `maxSurface2DLayered`, which HIP reports as 2048 on RDNA, so `enqueue`
  returned nothing and AliceVision dereferenced the null job: that was the segmentation fault. The
  limit is not real. `hip/port/popsift/layered_test.cpp` allocates a layered array at 4032 x 3024
  and at 8192 x 8192 on the same device without error, so the shim raises the reported layered
  limits to the plain 2D limit the device itself reports.
- **The describer is host C++ that reaches CUDA headers.** PopSIFT's public headers include
  `<cuda_runtime.h>`, which resolves to this project's shim and needs the HIP headers, and the file
  calls `cudaDeviceReset()`. Apply step 4v hands `aliceVision_feature` the ROCm include directory
  and the HIP runtime import library.

## The bug that made it extract nothing: HIP's layered surface write

The port built, linked and ran while returning zero keypoints. Dumping the Gaussian pyramid
(`Config::All` writes it to `dir-octave`) located it exactly: octave 0 level 0 held the image, and
**every level above it was entirely zero**, in every octave.

All those levels are written by the same call, `surf2DLayeredwrite`, differing only in the layer
index. HIP's implementation is the reason:

```cpp
// hip/amd_detail/amd_surface_functions.h
static __device__ void surf2DLayeredwrite(T data, hipSurfaceObject_t surfObj, int x, int y, int layer) {
  ...
  __ockl_image_store_lod_2D(i, get_native_vector(coords), layer, tmp);   // a MIP LEVEL, not a layer
}
```

It calls the mip-level store on a plain 2D image and passes the array layer as the level of detail.
Layer 0 lands on the base level, which is why the input image survived, and every other layer is
written to a mip level that was never allocated. Nothing reports an error. Its read counterpart is
correct: `tex2DLayered` uses `__ockl_image_sample_2Da`, the 2D-array sampler. That asymmetry is the
whole bug.

The shim writes through the matching 2D-array store instead, with the layer in the coordinate
vector:

```cpp
int4 coords{px, y, layer, 0};
__ockl_image_store_2Da(i, get_native_vector(coords), payload);
```

This is a HIP defect rather than a PopSIFT one, and it will affect any CUDA code that writes layered
surfaces. It is worth reporting to ROCm.

## Measured

The 6-view set, 4032 x 3024, same describer settings, on an RX 9070:

| whole set, GPU | images | wall | keypoints |
|---|---|---|---|
| 6-view | 6 | 9 s | 316,304 |
| monstree, 41 views | 41 | 24 s | 2,100,099 |
| engine bay | 107 | 41 s | 5,947,356 |

Against the CPU path on the same images and settings, measured over the first four of each set so
the comparison is like for like:

| four images | CPU | GPU |
|---|---|---|
| monstree | 317 s | 10 s |
| engine bay | 121 s | 9 s |

At those CPU rates either full set takes about 54 minutes, against 24 s and 41 s, so between 80 and
135 times faster depending on the set. The GPU is also doing more work than the CPU in these runs:
the CPU path returns exactly 20,000 keypoints per image, a cap, and the GPU path is uncapped, which
is why its keypoint counts are two to three times higher. Matching the configurations, and then
judging the descriptors by match counts through FeatureMatching and the reconstruction that follows,
is the next step.

## Quality: it reconstructs, but sparsely, and the descriptors are the reason

The 41-view set through matching and incremental reconstruction, GPU features against CPU features,
same images and same settings:

| | GPU SIFT | CPU SIFT |
|---|---|---|
| extraction | 26 s | 2021 s |
| descriptors | 2,100,183 | 820,000 |
| image pairs matched | 436 | 532 |
| matches | 164,241 | 572,530 |
| mean matches per pair | 377 | 1,076 |
| cameras calibrated | 41 of 41 | 41 of 41 |
| landmarks | 28,471 | 78,372 |
| observations | 94,147 | 301,170 |
| reprojection RMSE | 1.087 px | 1.040 px |

So the port reconstructs the whole scene at a comparable reprojection error, and does the extraction
seventy times faster, but it recovers a third of the landmarks. That is not "as good or better", so
it is not ready to ship.

The cause is descriptor sparsity, not keypoint count. Comparing the descriptor bytes AliceVision
actually stores, for the same image:

| | mean byte | empty bins | max |
|---|---|---|---|
| GPU | 11.2 | 71.4 % | 254 |
| CPU | 36.5 | 3.3 % | 161 |

A descriptor whose bins are 71 % empty carries little of the information matching depends on, which
is exactly the collapse the match counts show. What has been ruled out, each by measurement rather
than reasoning:

- **The pyramid.** Dumped with `Config::All`: every level of every octave is fully populated, with
  maxima decreasing smoothly as the blur increases.
- **The keypoint budget.** Raising the cap from 20,000 to 50,000 extrema tripled the descriptors and
  moved matching from 377 to 397 per pair. More keypoints do not help.
- **The descriptor implementation.** PopSIFT ships six for the same algorithm. `loop` (the default),
  `iloop` and `grid` all give 71.4 % empty bins on real images.
- **The filter's sort order.** Keeping the smallest-scale extrema instead of the largest gives
  71.4 % against 71.3 %.
- **The normalisation mode.** RootSIFT and Classic both produce healthy descriptors on synthetic
  input (0 % empty bins, mean 43).

That last line turned out to be a red herring, and the reason is worth recording: Classic
normalisation was hiding the fault rather than escaping it. Feeding `smoke.cpp` a real photograph
(`CHESHIRE_SMOKE_IMAGE` takes the raw float buffer `scripts`' `to_raw.py` writes) located it in one
run.

## What is actually wrong

AliceVision's SIFT defaults to `_rootSift = true`, so the describer asks for RootSIFT and every
measurement above that used Classic was measuring the wrong path. On the same 1008 x 756 photograph:

| normalisation | descriptor bytes | dead descriptors |
|---|---|---|
| Classic | mean 44.1, 0.8 % zero | 0 of 70,946 |
| RootSIFT | mean 1.9, 94.9 % zero | 67,176 of 70,946 |

94.7 % of the raw descriptor floats are NaN, and none are infinite. RootSIFT computes
`sqrt(bin / sum)`, so a descriptor whose bins are all zero divides 0 by 0 and every bin becomes NaN,
which AliceVision's `static_cast<unsigned char>` turns into 0. Classic instead computes
`__frsqrt_rn(0)` = inf, multiplies to get NaN, and then `min(NaN, 0.2f)` returns 0.2 — so a dead
descriptor comes out as a uniform ~45 in every bin. It looks healthy in aggregate and matches
nothing. Both modes receive the same dead input; only RootSIFT admits it.

The dead descriptors are dead before normalisation. `ext_desc_loop_sub` returns without touching the
descriptor when `DESC_MAGNIFY * sigma` is zero, and 84 % of the keypoints have `sigma` exactly zero.
Sigma is initialised to 0 on entry to `find_extrema_in_dog_sub` and only set on the accept path, so
those are rejected extrema being read as accepted ones.

They arrive through the grid filter, which AliceVision always enables via
`setFilterMaxExtrema(_params._maxTotalKeypoints)`:

| cap | keypoints | sigma == 0 | live keypoints |
|---|---|---|---|
| 25000 (above the total, filter never runs) | 26,560 | 0 | 26,560 |
| 24000 | 24,000 | 23,183 | 817 |
| 22000 | 22,000 | 21,183 | 817 |
| 20000 | 20,000 | 19,183 | 817 |

Exactly 817 keypoints survive whatever the cap, and the rest is padding. 817 is the real number:
unfiltered, the same run reports 26,560 keypoints at **817 distinct positions**, and the run-length
histogram is six huge runs — 19,106 / 4,903 / 1,179 / 497 / 32 / 32, one per octave, sized like the
per-octave extrema counts — with a median run of 1. Each octave's surplus slots were never written,
so their `i_ext_off` entry is still 0 and they all alias that octave's extremum 0, which is a real
keypoint with a valid sigma. That is why the unfiltered path looks clean. Once the filter runs,
`copy_if` rewrites `i_ext_off` and those never-written slots become genuine survivors pointing at a
zeroed `InitialExtremum`, which is where `sigma == 0` comes from.

## The root cause: one uninitialised variable

`extrema_count()` in `s_extrema.cu` reserves output slots like this:

```c
int write_index;
if( threadIdx.x == 0 ) {
    write_index = atomicAdd( extrema_counter, ct );
}
write_index = popsift::shuffle( write_index, 0 );   // every lane reads it
```

Every lane except 0 reads `write_index` without it having been assigned. That is undefined
behaviour. nvcc leaves the guard alone and the code works; the AMDGPU backend takes the licence and
**every lane performs the atomic**, so the counter advances by `32 * ct` instead of `ct`.

Adding `= 0` to the declaration fixes it. Measured per octave on a 504 x 378 photograph, counting
actual writes with a device atomic rather than printf:

| octave | counter, before | counter, after | extrema actually written |
|---|---|---|---|
| 0 | 19,712 | 616 | 616 |
| 1 | 5,056 | 158 | 158 |
| 2 | 1,216 | 38 | 38 |
| 3 | 512 | 16 | 16 |
| 4 | 32 | 1 | 1 |
| 5 | 32 | 1 | 1 |

Exactly 32x in every octave, and exact agreement after. The whole chain follows from it: the surplus
slots are never written, so their `i_ext_off` entry stays 0 and they alias their octave's extremum 0
— which is why the unfiltered path looks clean, and why the run-length histogram is six spikes with
a median of 1. Once AliceVision's grid filter runs, `copy_if` promotes those never-written slots to
survivors pointing at a zeroed `InitialExtremum` whose sigma is 0, `ext_desc_loop_sub` returns
without touching the descriptor, and RootSIFT turns the all-zero descriptor into 128 NaNs.

RootSIFT on the same photograph, before and after, against AliceVision's CPU SIFT:

| | before | after | CPU SIFT |
|---|---|---|---|
| zero descriptor bytes | 94.9 % | 4.3 % | 3.3 % |
| mean descriptor byte | 1.9 | 35.1 | 36.5 |
| NaN raw floats | 94.7 % | 0 | — |
| orientations per feature | 3.55 | 1.18 | ~1.15 |
| duplicate keypoints | 32.5x | 1.0x | — |

### Ruled out, each by direct measurement

- **The warp shuffles.** `warpSize` is 32 on the RX 9070, host and device agree, and
  `hip/port/popsift/norm_test.cpp` replicates `normalize_histogram`'s (32,32) block and its
  width-less butterfly reduction: 0 of 64 rows wrong, identical to the same reduction pinned to
  width 32.
- **The ballot and popcount.** `hip/port/popsift/ballot_test.cpp` checks the shim's
  `__ballot_sync` and HIP's native `__ballot` against known lane patterns; both exact. In the real
  kernel the masks are single-bit and spread across lanes 0-25, and `ct` is only ever 1 or 2.
- **The atomicAdd leader election.** Instrumenting the counter showed all 447 sampled calls come
  from `lane=0`; the `if (threadIdx.x == 0)` guard holds.
- **Thread-index degeneracy.** `x = block_x + threadIdx.x + 1` is intact, and the median run length
  of 1 rules out whole warps landing on one pixel.
- **The layered pyramid reads.** `hip/port/popsift/layered_rw_test.cpp` writes a distinct value to
  every layer and reads it back through the exact texture descriptors `sift_octave.cu` builds:
  correct on all four layers, point and linear, with the layer as float or int.

### A second HIP defect

That same test found `surf2DLayeredread` broken the way `surf2DLayeredwrite` was: correct on layer 0
and zero on every layer above it, 1536 of 2048 cells wrong. PopSIFT does not use it, so it is not
this bug, but it is the same `__ockl_image_load_lod_2D` against `__ockl_image_load_2Da` mistake and
belongs in the same ROCm report.

### Do not run PopSIFT uncapped on a full-resolution photograph

Leaving `setFilterMaxExtrema` unset and handing PopSIFT a 4032 x 3024 photograph took the host
machine down hard on 2026-09-18 — an unexpected shutdown with no bugcheck, no crash dump and no
driver reset. A synthetic image at the same size is smooth and yields few extrema; real photographic
content does not, and with detection over-firing about 32x it is far worse. `smoke.cpp` now caps at
20,000 by default and only runs uncapped when a cap of 0 is passed explicitly.

## Validated: the 41-view set, end to end

Same images, same settings, GPU SIFT against AliceVision's CPU SIFT, through FeatureMatching and
incremental SfM. "before" is the same build with the uninitialised variable left alone.

| | GPU, before | GPU, after | CPU SIFT |
|---|---|---|---|
| extraction | 26 s | **26 s** | 2021 s |
| descriptors | 2,100,183 | 968,726 | 820,000 |
| image pairs matched | 436 | 538 | 532 |
| total matches | 164,241 | 615,742 | 572,530 |
| mean matches per pair | 377 | **1,144** | 1,076 |
| cameras calibrated | 41 of 41 | **41 of 41** | 41 of 41 |
| landmarks | 28,471 | **88,463** | 78,372 |
| residual RMSE | 1.087 px | **1.0377 px** | 1.03997 px |

GPU SIFT now recovers 13 % more landmarks than the CPU path at a marginally lower reprojection
error, with extraction 78x faster.

### The same set on every card here

Five combinations of architecture, runtime and operating system, all on the same 41 photographs
and the same `cameraInit`, so only the GPU path differs:

| | CPU SIFT | RDNA4 | RDNA1 | RDNA2 | RDNA2 | RDNA1 |
|---|---|---|---|---|---|---|
| card | - | RX 9070 | RX 5500 XT | RX 6750 XT | RX 6750 XT | RX 5500 XT |
| architecture | - | gfx1201 | gfx1012 | gfx1031 | gfx1031 | gfx1012 |
| runtime | - | ROCm 7.2.1 | ROCm 7.2 | ROCm 7.2 | HIP SDK 6.2 | HIP SDK 6.2 |
| system | - | Windows | Linux | Linux | Windows | Windows |
| descriptors | 820,000 | 968,726 | 968,726 | 968,726 | 968,701 | 968,701 |
| image pairs | 532 | 538 | 538 | 538 | 538 | 538 |
| mean matches per pair | 1,076 | 1,144 | 1,143 | 1,144 | 1,145 | 1,145 |
| cameras calibrated | 41 of 41 | 41 of 41 | 41 of 41 | 41 of 41 | 41 of 41 | 41 of 41 |
| landmarks | 78,372 | 88,463 | 88,562 | 88,517 | 88,537 | 88,513 |
| residual RMSE | 1.03997 px | 1.0377 px | 1.03903 px | 1.03744 px | 1.03758 px | 1.03727 px |

Landmarks span 88,463 to 88,562 - 0.11 % - across three architectures, two runtimes and two
operating systems, and every one recovers about 13 % more than the CPU path at a lower
reprojection error.

Part of that span is not the hardware. Two stages here are not reproducible
([docs/15](15-acransac-cpu.md) has both measurements):

- `incrementalSfM` gives 88,463, 88,468, 88,469 and 88,472 from the same binary on the same inputs.
  The likely cause is multi-threaded Ceres; `randomSeed` is fixed, so it is not the RNG.
- **Extraction itself** is only reproducible in its totals. Two runs on one machine agree exactly on
  the descriptor count and differ in all 41 `.feat` files - the same 23,781 lines per file in a
  different order, because PopSift allocates keypoint slots with atomics.

A spread of 9 on one machine against 99 across five configurations means the cross-configuration
differences are mostly real - the descriptor counts differ by toolchain, and the mean matches per
pair by one or two - but no single row here is reproducible to the landmark, and none of these
numbers should be read as one. Exactness claims elsewhere in this project are checked against
FeatureMatching's output, which is deterministic given fixed features.

The descriptor count tracks the **runtime, not the silicon**, and the fifth column is what
settles it. The three ROCm 7.2 builds agree exactly at 968,726 on three different
architectures; the two HIP SDK 6.2 builds agree exactly at 968,701 on two different
architectures. Grouping by card explains nothing - the same RX 5500 XT gives 968,726 under one
toolchain and 968,701 under the other, and the same figure as an RX 6750 XT when the toolchain
matches. A 25-descriptor difference in 968,700 is not worth chasing, but it is worth
attributing correctly.

Extraction times are not comparable between these rows - different hosts, operating systems and
storage - and were not set up as a timing comparison.

### RDNA1 runs on Windows too

Assumed not to, because the ROCm 7.2 runtime does not enumerate RX 5000 cards. The HIP SDK 6.2
runtime does: a gfx1012 build of `hip/port/popsift/bugreport_layered_surface.hip` on an RX 5500 XT
under Windows reads every texture layer correctly, exactly as on RDNA2 and RDNA4. So a
`gfx1012-avx` Windows package looks viable and would add RX 5500/5600/5700 support.

It does: the fifth column of the table above is that package, running the whole pipeline on an
RX 5500 XT under Windows. 968,701 descriptors in 60 s, 538 pairs matched in 357 s, and SfM
calibrated all 41 cameras to 88,513 landmarks at 1.03727 px - inside the same 0.11 % band as
every other card.

Note what the wrong package does: the gfx1031 binary on that gfx1012 card enumerated the device,
ran its host-side calls, and returned wrong data from every kernel **with no error at all** -
because nothing in that reproducer checks launch status. A single-architecture package handed to
the wrong card fails like corruption, not like a mismatch.

### A silent CPU fallback, and how to see it

The first attempt on house-pc produced a complete, plausible reconstruction: 41 of 41 cameras,
78,052 landmarks, RMSE 1.04082. It had run **CPU SIFT throughout**. The tells were `[cpu]` in the
extraction log and a descriptor count of exactly 820,000, which is the CPU cap of 20,000 x 41.

`ImageDescriber_SIFT`'s constructor does `setUseCuda(gpu::gpuSupportCUDA(3, 0))`, and our patch
defines `ALICEVISION_HAVE_CUDA` for HIP builds so that check runs through the shim to
`hipGetDeviceCount`. When that fails the describer falls back to VLFeat **without a warning that
GPU SIFT was asked for and denied**. Anyone packaging this can ship a "GPU" build that quietly runs
on the CPU.

The cause was the bundle: `scripts/linux/build-alicevision.sh` replaces the WSL HSA runtime with
`hsa-rocr` from AMD's repository *after* the `bundle` target, because the WSL one probes `/dev/dxg`
and reports no device on a real Linux box. Invoking `cmake --build . --target bundle` directly skips
that step. Always bundle through the script. To check a bundle before trusting a result:

```bash
python3 -c "import ctypes;h=ctypes.CDLL('bundle/lib/libamdhip64.so.7');n=ctypes.c_int(0);print(h.hipGetDeviceCount(ctypes.byref(n)),n.value)"
# 0 1  -> a device is visible;  100 0 -> no device, extraction will silently use the CPU
```

Note that the size is what distinguishes the two runtimes, not the `/dev/dxg` string: the standard
one contains it too, at 4,248,208 bytes against the WSL one's 1,439,536. The descriptor count falling from 2.1 M to 969 k is the 32x
duplication going away; 969 k against the CPU's 820 k is the expected spread, since PopSIFT and
VLFeat do not agree exactly by design.

## Builds

One library per runtime, not per card. `scripts/build-popsift.cmd` (Windows) and
`scripts/linux/build-popsift.sh` take the architecture set; the name selects the build and install
directories.

| artifact | code objects | covers |
|---|---|---|
| `popsift-rdna3-rdna4` | gfx1100/1101/1102/1103, gfx1150/1151/1152/1153, gfx1200/1201 | RDNA3, RDNA3.5 APUs, RDNA4 |
| `popsift-gfx1030`, `-gfx1031`, `-gfx1032` | one each | RDNA2, through the HIP SDK 6.2 toolchain |
| `popsift-linux` | gfx1010 through gfx1201, 15 objects | RDNA1 to RDNA4 in one file |

`rocm-sdk targets` lists only what the runtime wheel ships prebuilt; the compiler accepts more, so
gfx1103/1152/1153 and the RDNA1 and RDNA2 targets all build. RDNA2 on Windows still needs the
separate HIP SDK 6.2 toolchain, because the ROCm 7.2 runtime does not enumerate RX 6000 cards at
all, and that toolchain writes an offload bundle whose entries are all the same code object when
given several architectures — hence one build per chip. Linux has neither problem. Every bundle
here was checked by parsing it for `hipv4-amdgcn-amd-amdhsa--gfx*` rather than trusting the build.

### Three things the Linux build needed that Windows did not

- The shim was force-included with clang-cl's `/FI`, which clang++ reads as a filename. It is now
  the first line of the unity source instead, so no flag is needed on either driver.
- Forcing it in ahead of clang's HIP runtime wrapper broke glibc's feature detection.
- **PopSIFT ships a header called `features.h`.** With `third_party/popsift/src/popsift` on the
  include path, libstdc++'s `bits/os_defines.h` resolves its own `#include <features.h>` to that
  one, glibc's never loads, `__GLIBC_USE` is left undefined and every later system header fails.
  That directory is not needed - the sources use quoted includes - so it is gone. Windows never
  sees this because the MSVC STL has no `<features.h>`.

### Counting pairs

`aliceVision_featureMatching` logs each image pair's geometric match count **twice**, with the same
value both times. Counting log lines therefore doubles both the pair count and the match total; the
mean per pair is unaffected, since the doubling cancels. Count distinct pairs:

```bash
python3 -c "
import re,sys,collections
t=open(sys.argv[1],errors='ignore').read()
m=re.findall(r'image pair \((\d+), (\d+)\) contains (\d+) geometric matches',t)
best={frozenset((a,b)):int(c) for a,b,c in m}
print(len(best),'pairs,',sum(best.values()),'matches,',sum(best.values())//len(best),'per pair')
" match.log
```

538 of a possible 820 pairs on the 41-view set, 4,367 of a possible 5,671 on the engine bay.

## What is not done

The descriptors have not been judged, only counted. PopSIFT and CPU SIFT do not agree exactly by
design, so the acceptance test is match counts through FeatureMatching and the reconstruction that
follows, which is what the 41-view table above reports.

**Only the RDNA4 library has run.** The RDNA2 and Linux libraries have been built and their offload
bundles verified, which is not the same as working: no kernel from either has executed. house-pc
has the RX 5500 XT (RDNA1) for the Linux leg and bench-pc takes the RDNA2 cards for the Windows
leg. bench-pc is an FX-8120, so it needs the `/arch:AVX` packages - Bulldozer has AVX but not AVX2,
and an AVX2 build dies on an illegal instruction before any GPU code runs.

**The CUDA reference is two AliceVision versions behind.** Every "matches CUDA" comparison in this
repository comes from the Meshroom 2023.3.0 bundle, which ships AliceVision 3.2.0, while Cheshire
builds 3.4.0 from source. Meshroom 2025.1.0 ships 3.3.0, so neither release matches what we build.
A difference against that reference could be the version rather than the backend. Building a CUDA
AliceVision from our own 3.4.0 source would remove the confound - the vcpkg dependency set already
carries a CUDA PopSIFT and the CUDA OpenCV modules, so the only missing piece is a CUDA toolkit -
and it would give a matched reference for GPU SIFT as well as the depth map. Deferred deliberately,
2026-09-18, behind the hardware validation above.

`CHESHIRE_POPSIFT_DEBUG=1` prints the per-octave extrema and orientation counts, plus what the grid
filter kept, which is the fastest way to see whether a change broke detection.
