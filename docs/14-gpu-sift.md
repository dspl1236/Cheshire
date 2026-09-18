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

## What is not done

The descriptors have not been judged, only counted. PopSIFT and CPU SIFT do not agree exactly by
design, so the acceptance test is match counts through FeatureMatching and the reconstruction that
follows, with the keypoint caps matched first. Nothing has run on RDNA1, RDNA2 or Linux yet, and the
library is built for one architecture at a time. `CHESHIRE_POPSIFT_DEBUG=1` prints the per-octave
extrema and orientation counts, which is the fastest way to see whether a change broke detection.
