// Cheshire: the HIP compatibility shim for PopSIFT (alicevision/popsift v0.10.0).
//
// PopSIFT is CUDA-only; the library AliceVision links against for GPU SIFT imports nvcuda.dll. This
// header is force-included in front of every PopSIFT translation unit, on top of the project's
// cheshire/cuda_to_hip.h, and supplies the handful of things that header does not already cover.
// Unlike the depth-map port, no texture emulation is needed: HIP 7.2 has layered arrays, layered
// surfaces and 2D textures natively, so the pyramid keeps its original shape.
//
// What is here, and why:
//   - the warp shuffles. PopSIFT calls CUDA's *_sync forms with a 32-bit lane mask. HIP's *_sync
//     forms want a 64-bit mask (a wavefront can be 64 wide) and do not exist at all in the HIP SDK
//     6.2 toolchain used for RDNA2. The wrappers drop the mask and call the non-sync intrinsics,
//     which exist in both and which the *_sync forms resolve to anyway; PopSIFT only uses them on
//     converged warps. Measured identical on ROCm 7.2.1/gfx1201.
//   - surf2DLayeredwrite. HIP's takes no boundary-mode argument; PopSIFT passes cudaBoundaryModeZero.
//     The overload drops it, which is what HIP does anyway for an in-range write.
//   - the round-toward-positive-infinity multiply and fused multiply-add. HIP has only the
//     round-to-nearest forms. Both are computed exactly in double (a float product and a float fma
//     are exact in double) and then rounded up, so they are the values CUDA's intrinsics produce.
//   - a few runtime names cuda_to_hip.h has not needed before.
#pragma once
// windows.h defines max() and min() as macros, which eat std::numeric_limits<float>::max().
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cheshire/cuda_to_hip.h>
#include <hip/hip_runtime.h>
#include <cmath>

// ---------------------------------------------------------------- warp shuffles
// Defined before the macros below so the macros do not rewrite the calls inside them.
template<typename T>
__device__ inline T cheshirePopsiftShflDownSync(unsigned int /*mask*/, T var, unsigned int delta, int width = warpSize)
{
    return __shfl_down(var, delta, width);
}
template<typename T>
__device__ inline T cheshirePopsiftShflUpSync(unsigned int /*mask*/, T var, unsigned int delta, int width = warpSize)
{
    return __shfl_up(var, delta, width);
}
template<typename T>
__device__ inline T cheshirePopsiftShflXorSync(unsigned int /*mask*/, T var, int lane, int width = warpSize)
{
    return __shfl_xor(var, lane, width);
}
template<typename T>
__device__ inline T cheshirePopsiftShflSync(unsigned int /*mask*/, T var, int src, int width = warpSize)
{
    return __shfl(var, src, width);
}
__device__ inline unsigned long long cheshirePopsiftBallotSync(unsigned int /*mask*/, int pred)
{
    return __ballot(pred);
}
__device__ inline int cheshirePopsiftAnySync(unsigned int /*mask*/, int pred)
{
    return __any(pred);
}
__device__ inline int cheshirePopsiftAllSync(unsigned int /*mask*/, int pred)
{
    return __all(pred);
}

#define __shfl_down_sync(...) cheshirePopsiftShflDownSync(__VA_ARGS__)
#define __shfl_up_sync(...) cheshirePopsiftShflUpSync(__VA_ARGS__)
#define __shfl_xor_sync(...) cheshirePopsiftShflXorSync(__VA_ARGS__)
#define __shfl_sync(...) cheshirePopsiftShflSync(__VA_ARGS__)
#define __ballot_sync(...) cheshirePopsiftBallotSync(__VA_ARGS__)
#define __any_sync(...) cheshirePopsiftAnySync(__VA_ARGS__)
#define __all_sync(...) cheshirePopsiftAllSync(__VA_ARGS__)

// ---------------------------------------------------------------- layered surface write
// Two problems with HIP's surf2DLayeredwrite. It takes no boundary-mode parameter, which is
// cosmetic, and it writes to the wrong place, which is not: amd_surface_functions.h implements it
// as __ockl_image_store_lod_2D(i, coords, layer, data), the MIP-LEVEL store on a plain 2D image,
// passing the layer as a level of detail. Layer 0 therefore lands on the base level and every other
// layer is written to a mip level that does not exist, silently. Its read counterpart tex2DLayered
// is correct: it uses __ockl_image_sample_2Da, the 2D-array sampler. That asymmetry is what left
// PopSIFT's Gaussian pyramid with a correct level 0 and nothing above it.
// This writes through the matching 2D-array store, with the layer in the coordinate vector.
// The two toolchains spell the coordinate vector differently: ROCm 7.2's header passes it through
// get_native_vector(), the HIP SDK 6.2 one uses int2(x, y).data, and neither spelling compiles on
// the other. Both define HIP_vector_type::Native_vec_, so the native vector is built directly.
template<typename T>
__device__ inline void cheshirePopsiftSurf2DLayeredWrite(T data, hipSurfaceObject_t surf, int x, int y, int layer, int /*boundary*/)
{
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)surf;
    const int px = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2Da(i), __ockl_image_channel_order_2Da(i));
    auto payload = __hipMapTo<float4::Native_vec_>(data);
    typename int4::Native_vec_ coords = {px, y, layer, 0};
    __ockl_image_store_2Da(i, coords, payload);
}
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
#define __fmul_ru(a, b) cheshirePopsiftFmulRu((a), (b))
#define __fmaf_ru(a, b, c) cheshirePopsiftFmafRu((a), (b), (c))

// ---------------------------------------------------------------- runtime names
#ifndef cudaArrayLayered
#define cudaArrayLayered hipArrayLayered
#endif
#ifndef cudaArrayDefault
#define cudaArrayDefault hipArrayDefault
#endif
#ifndef cudaStreamWaitEvent
#define cudaStreamWaitEvent hipStreamWaitEvent
#endif
#ifndef cudaHostRegister
#define cudaHostRegister hipHostRegister
#endif
#ifndef cudaHostUnregister
#define cudaHostUnregister hipHostUnregister
#endif
#ifndef cudaHostRegisterDefault
#define cudaHostRegisterDefault hipHostRegisterDefault
#endif
#ifndef cudaHostRegisterPortable
#define cudaHostRegisterPortable hipHostRegisterPortable
#endif
#ifndef cudaMalloc3DArray
#define cudaMalloc3DArray hipMalloc3DArray
#endif
#ifndef cudaMemcpy3D
#define cudaMemcpy3D hipMemcpy3D
#endif
#ifndef cudaMemcpy3DParms
#define cudaMemcpy3DParms hipMemcpy3DParms
#endif
#ifndef cudaMemcpy2DAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#endif
#ifndef cudaPointerAttributes
#define cudaPointerAttributes hipPointerAttribute_t
#endif

// ---------------------------------------------------------------- __constant__ symbols
// CUDA takes the symbol itself; hipMemcpyToSymbol wants its address. cuda_to_hip.h already has the
// pointer form, so these overloads catch the calls that pass the variable.
template<typename T>
inline hipError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0,
                                     hipMemcpyKind kind = hipMemcpyHostToDevice)
{
    return cudaMemcpyToSymbol((const void*)&symbol, src, count, offset, kind);
}
template<typename T>
inline hipError_t cudaMemcpyToSymbolAsync(const T& symbol, const void* src, size_t count, size_t offset,
                                          hipMemcpyKind kind, hipStream_t stream)
{
    return cudaMemcpyToSymbolAsync((const void*)&symbol, src, count, offset, kind, stream);
}
// cuda_to_hip.h maps these straight to the HIP names, so the macro has to go before a wrapper of
// the same name can exist.
#ifdef cudaMemcpyFromSymbol
#undef cudaMemcpyFromSymbol
#endif
#ifdef cudaMemcpyFromSymbolAsync
#undef cudaMemcpyFromSymbolAsync
#endif
template<typename T>
inline hipError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0,
                                       hipMemcpyKind kind = hipMemcpyDeviceToHost)
{
    return hipMemcpyFromSymbol(dst, (const void*)&symbol, count, offset, kind);
}
template<typename T>
inline hipError_t cudaMemcpyFromSymbolAsync(void* dst, const T& symbol, size_t count, size_t offset,
                                            hipMemcpyKind kind, hipStream_t stream)
{
    return hipMemcpyFromSymbolAsync(dst, (const void*)&symbol, count, offset, kind, stream);
}

// ---------------------------------------------------------------- thrust execution policy
// PopSIFT's grid filter runs thrust algorithms on a stream with thrust::cuda::par.on(stream).
// rocThrust calls that namespace thrust::hip, so an alias is enough; the policy object and its
// .on(stream) are the same shape.
#include <thrust/system/hip/detail/par.h>
namespace thrust {
namespace cuda = ::thrust::hip;
}

// ---------------------------------------------------------------- device limits
// HIP reports maxTexture2DLayered as 2048 on RDNA, and PopSIFT reads that field to decide whether an
// image fits (common/device_prop.cu checkLimit_2DtexLinear), so it refuses anything wider than 2048
// and enqueue() returns nothing. The limit is not real: hipMalloc3DArray with hipArrayLayered
// succeeds at 4032 x 3024 and at 8192 x 8192 on this device (hip/port/popsift/layered_test.cpp).
// Raise the reported layered limit to the plain 2D limit the same device reports, leaving the layer
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

