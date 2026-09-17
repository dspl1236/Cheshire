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

## What is not done

Compiling is not running. Still ahead: a library target with the device link step, building
AliceVision with `ALICEVISION_USE_POPSIFT=ON` against it, and then the part that decides whether the
port is correct, comparing descriptors against the CPU SIFT path on the same images. PopSIFT and
CPU SIFT do not agree exactly by design, so the acceptance test is match quality through
FeatureMatching and the resulting reconstruction, not byte equality.
