// Cheshire: what PopSIFT's own HIP port leaves to us (alicevision/popsift, pinned in
// scripts/apply_popsift_patch.py).
//
// Since 0.10.1, upstream PopSIFT builds for AMD itself: src/popsift/cuda_to_hip.h maps the runtime,
// and the sources take a HIP branch wherever the two platforms differ (software bilinear reads,
// readlane broadcasts, thrust::hip, a plain 3D array for the pyramid). This header is the first line
// of Cheshire's single translation unit and includes upstream's mapping, then changes three things:
//   - the layered surface write. Upstream gave up layered arrays on HIP because HIP's
//     surf2DLayeredwrite writes every layer to the same place, and keeps the pyramid in a plain 3D
//     array, which HIP on Windows will not allocate wider than 2048 (ROCm/rocm-systems#14652). apply_popsift_patch.py (3g) puts the
//     layered arrays back, and this writes them correctly (below).
//   - the round-toward-positive-infinity multiply and fused multiply-add. Upstream maps them to the
//     round-to-nearest forms; here both are computed exactly in double (a float product and a float
//     fma are exact in double) and then rounded up, so they are the values CUDA's intrinsics produce.
//   - the device limits. HIP on Windows reports maxTexture2DLayered as 2048, and PopSIFT reads that field
//     to decide whether an image fits (common/device_prop.cu), so it would refuse anything wider. The
//     limit is not real for layered arrays: hipMalloc3DArray with hipArrayLayered succeeds at
//     4032 x 3024 and at 8192 x 8192 on the RX 9070 (hip/port/popsift/layered_test.cpp).
// The warp shuffles need nothing: apply_popsift_patch.py sets PopSift_HAVE_SHFL_DOWN_SYNC to 0, so
// common/assist.h calls the non-sync intrinsics, which both toolchains have (the HIP SDK 6.2 one
// used for RDNA2 has no *_sync forms at all).
#pragma once
// windows.h defines max() and min() as macros, which eat std::numeric_limits<float>::max().
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef USE_HIP
#define USE_HIP 1
#endif
#include "popsift/cuda_to_hip.h"
#include <cmath>

// ---------------------------------------------------------------- layered surface write
// amd_surface_functions.h implements surf2DLayeredwrite as __ockl_image_store_lod_2D(i, coords, layer,
// data), the MIP-LEVEL store on a plain 2D image, passing the layer as a level of detail: layer 0 lands
// on the base level and every other layer on a mip level that does not exist, silently. The read side,
// tex2DLayered, is correct: it uses __ockl_image_sample_2Da, the 2D-array sampler. This writes through
// the matching 2D-array store, with the layer in the coordinate vector. The two toolchains spell the
// coordinate vector differently (ROCm 7.2's header passes it through get_native_vector(), the HIP SDK
// 6.2 one uses int2(x, y).data, and neither compiles on the other); both define
// HIP_vector_type::Native_vec_, so the native vector is built directly.
template<typename T>
__device__ inline void cheshirePopsiftSurf2DLayeredWrite(T data, hipSurfaceObject_t surf, int x, int y, int layer, int /*boundary*/)
{
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)surf;
    const int px = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2Da(i), __ockl_image_channel_order_2Da(i));
    auto payload = __hipMapTo<float4::Native_vec_>(data);
    typename int4::Native_vec_ coords = {px, y, layer, 0};
    __ockl_image_store_2Da(i, coords, payload);
}
#undef surf2DLayeredwrite
#define surf2DLayeredwrite(...) cheshirePopsiftSurf2DLayeredWrite(__VA_ARGS__)

// ---------------------------------------------------------------- round-up arithmetic
// a * b and fma(a, b, c) of floats are exact in double, so rounding that exact value up gives
// exactly what CUDA's round-toward-positive-infinity intrinsics return.
__device__ inline float cheshirePopsiftRoundUp(double exact)
{
    float r = (float)exact;
    if ((double)r < exact)
        r = nextafterf(r, INFINITY);
    return r;
}
__device__ inline float cheshirePopsiftFmulRu(float a, float b) { return cheshirePopsiftRoundUp((double)a * (double)b); }
__device__ inline float cheshirePopsiftFmafRu(float a, float b, float c)
{
    return cheshirePopsiftRoundUp(fma((double)a, (double)b, (double)c));
}
#undef __fmul_ru
#undef __fmaf_ru
#define __fmul_ru(a, b) cheshirePopsiftFmulRu((a), (b))
#define __fmaf_ru(a, b, c) cheshirePopsiftFmafRu((a), (b), (c))

// ---------------------------------------------------------------- device limits
// The reported layered limits raised to the plain 2D limit the same device reports, leaving the layer
// count alone, so PopSIFT's own check reflects what the hardware does.
inline hipError_t cheshirePopsiftGetDeviceProperties(hipDeviceProp_t* prop, int device)
{
    const hipError_t err = hipGetDeviceProperties(prop, device);
    if (err == hipSuccess && prop != nullptr)
    {
        for (int i = 0; i < 2; ++i)
        {
            if (prop->maxTexture2DLayered[i] < prop->maxTexture2D[i])
                prop->maxTexture2DLayered[i] = prop->maxTexture2D[i];
            // PopSIFT checks the layered SURFACE limit separately for the pyramid it writes into
            if (prop->maxSurface2DLayered[i] < prop->maxTexture2D[i])
                prop->maxSurface2DLayered[i] = prop->maxTexture2D[i];
        }
    }
    return err;
}
#ifdef cudaGetDeviceProperties
#undef cudaGetDeviceProperties
#endif
#define cudaGetDeviceProperties cheshirePopsiftGetDeviceProperties
