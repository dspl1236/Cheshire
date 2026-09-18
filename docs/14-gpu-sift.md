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

## What is not done

Running is not working. The pipeline completes and returns **zero keypoints**, on real photographs
and on synthetic input, at every image size. `hip/port/popsift/smoke.cpp` is the shortest way to see
it: it does exactly what the describer does, prints each stage, and ends with `0 features, 0
descriptors`. Nothing reports an error, including with `CHESHIRE_POPSIFT_ERRCHK=ON`, which checks
after every kernel launch.

So the next question is which stage produces nothing: the Gaussian pyramid, the difference of
Gaussians, extrema detection, or the descriptor pass. The leads worth taking first are the places
where the shim changed semantics rather than spelling. The warp intrinsics are the obvious suspects,
since `__ballot`, `__any` and `__all` return a 64-bit mask in HIP where PopSIFT's code expects 32
bits, and lane arithmetic derived from them would then be wrong. After that, the round-up
arithmetic, and the layered surface writes.

Once features appear, the acceptance test is not byte equality: PopSIFT and CPU SIFT do not agree
exactly by design. It is match counts through FeatureMatching and the reconstruction that follows.
