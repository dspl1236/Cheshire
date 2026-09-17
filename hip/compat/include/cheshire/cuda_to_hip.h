// Cheshire "conversion API": compile AliceVision's CUDA depth-map sources as HIP
// without rewriting them. Force-included (clang -include / clang-cl /FI) or pulled in
// through the cuda_runtime.h shim next to this file.
//
// Style follows llama.cpp's ggml-cuda/vendors/hip.h: a flat macro map from the CUDA
// runtime names the sources use to their HIP equivalents. Only names that the
// AliceVision depthMap tree actually uses are mapped (grep-derived, see
// docs/00-scope-and-findings.md); add to it when a new one shows up in a build error.
#pragma once

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>

#define CHESHIRE_HIP 1

// ---- error handling -------------------------------------------------------
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError

// ---- device management ----------------------------------------------------
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaMemGetInfo hipMemGetInfo
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaFuncSetCacheConfig hipFuncSetCacheConfig
#define cudaFuncCachePreferL1 hipFuncCachePreferL1
#define cudaFuncCachePreferShared hipFuncCachePreferShared

// ---- streams --------------------------------------------------------------
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamDefault hipStreamDefault
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime

// ---- memory ---------------------------------------------------------------
// Device allocations go through the memory bridge (VRAM first, mapped host RAM when VRAM
// is short; see bridge.h). CUDA's cudaMalloc has a templated overload (T** devPtr).
#include <cheshire/bridge.h>
inline hipError_t cudaMalloc(void** devPtr, size_t bytes) { return cheshire::bridge::malloc(devPtr, bytes); }
template<class T>
inline hipError_t cudaMalloc(T** devPtr, size_t bytes) { return cheshire::bridge::malloc(reinterpret_cast<void**>(devPtr), bytes); }
inline hipError_t cudaFree(void* devPtr) { return cheshire::bridge::free(devPtr); }
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMallocManaged hipMallocManaged
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostGetDevicePointer hipHostGetDevicePointer
// cudaMallocPitch has a templated overload in cuda_runtime.h (AliceVision calls
// cudaMallocPitch<Type>(&buf, ...)); HIP only has the C function, so provide both forms.
inline hipError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{ return cheshire::bridge::mallocPitch(devPtr, pitch, width, height); }
template<class T>
inline hipError_t cudaMallocPitch(T** devPtr, size_t* pitch, size_t width, size_t height)
{ return cheshire::bridge::mallocPitch(reinterpret_cast<void**>(devPtr), pitch, width, height); }
inline hipError_t cudaMalloc3D(hipPitchedPtr* p, hipExtent extent) { return cheshire::bridge::malloc3D(p, extent); }
// memsets: same ordering rule as the copies below (the fused SGM aggregation clears a per-row
// accumulator with cudaMemsetAsync between kernels; spilled on ROCm-Linux that clear ran out of
// order and corrupted every depth map)
inline hipError_t cudaMemset(void* p, int v, size_t n);
inline hipError_t cudaMemsetAsync(void* p, int v, size_t n, hipStream_t s = 0);
inline hipError_t cudaMemset2D(void* p, size_t pitch, int v, size_t w, size_t h);
inline hipError_t cudaMemset2DAsync(void* p, size_t pitch, int v, size_t w, size_t h, hipStream_t s = 0);
// Copies that touch a spilled (mapped host) block are ordered by hand. On ROCm-Linux a copy
// whose source or destination is host-resident is executed by the CPU at the call, not queued
// behind the stream's kernels; AliceVision reads its finished tile maps exactly that way
// (cudaMemcpy2DAsync on the tile stream, cudaDeviceSynchronize afterwards) and got stale data
// when the map had been spilled (docs/02-memory-bridge.md, RX 6750 XT matrix). Windows (PAL)
// orders these itself. The cost is a stream sync per copy of a spilled buffer, i.e. only in
// the degraded mode.
namespace cheshire { namespace detail {
inline void orderCopy(const void* a, const void* b, hipStream_t s) {
    if (cheshire::bridge::inSpilled(a) || cheshire::bridge::inSpilled(b)) (void)hipStreamSynchronize(s);
}
inline void orderCopy(const void* a, const void* b) {
    if (cheshire::bridge::inSpilled(a) || cheshire::bridge::inSpilled(b)) (void)hipDeviceSynchronize();
}
}}  // namespace cheshire::detail
inline hipError_t cudaMemset(void* p, int v, size_t n)
{ cheshire::detail::orderCopy(p, nullptr); return hipMemset(p, v, n); }
inline hipError_t cudaMemsetAsync(void* p, int v, size_t n, hipStream_t s)
{ cheshire::detail::orderCopy(p, nullptr, s); return hipMemsetAsync(p, v, n, s); }
inline hipError_t cudaMemset2D(void* p, size_t pitch, int v, size_t w, size_t h)
{ cheshire::detail::orderCopy(p, nullptr); return hipMemset2D(p, pitch, v, w, h); }
inline hipError_t cudaMemset2DAsync(void* p, size_t pitch, int v, size_t w, size_t h, hipStream_t s)
{ cheshire::detail::orderCopy(p, nullptr, s); return hipMemset2DAsync(p, pitch, v, w, h, s); }
inline hipError_t cudaMemcpy(void* dst, const void* src, size_t n, hipMemcpyKind k)
{ cheshire::detail::orderCopy(dst, src); return hipMemcpy(dst, src, n, k); }
inline hipError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, hipMemcpyKind k, hipStream_t s = 0)
{ cheshire::detail::orderCopy(dst, src, s); return hipMemcpyAsync(dst, src, n, k, s); }
inline hipError_t cudaMemcpy2D(void* dst, size_t dp, const void* src, size_t sp, size_t w, size_t h, hipMemcpyKind k)
{ cheshire::detail::orderCopy(dst, src); return hipMemcpy2D(dst, dp, src, sp, w, h, k); }
inline hipError_t cudaMemcpy2DAsync(void* dst, size_t dp, const void* src, size_t sp, size_t w, size_t h, hipMemcpyKind k, hipStream_t s = 0)
{ cheshire::detail::orderCopy(dst, src, s); return hipMemcpy2DAsync(dst, dp, src, sp, w, h, k, s); }
#define cudaMemcpy3DAsync hipMemcpy3DAsync
#define cudaMemcpy3DParms hipMemcpy3DParms
// cudaMemcpyToSymbol: hipMemcpyToSymbol resolves the symbol on every call (tens of ms each on
// Windows: AliceVision spends ~70 ms per camera-parameter upload). Resolve the device address
// once per symbol and use a plain copy.
#include <unordered_map>
#include <mutex>
namespace cheshire { namespace detail {
inline void* symbolDevicePtr(const void* symbol) {
    static std::unordered_map<const void*, void*> cache; static std::mutex m;
    std::lock_guard<std::mutex> g(m);
    auto it = cache.find(symbol);
    if (it != cache.end()) return it->second;
    void* p = nullptr;
    if (hipGetSymbolAddress(&p, symbol) != hipSuccess) return nullptr;
    cache.emplace(symbol, p);
    return p;
}
}}  // namespace cheshire::detail
inline hipError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset = 0, hipMemcpyKind kind = hipMemcpyHostToDevice) {
    void* dst = cheshire::detail::symbolDevicePtr(symbol);
    if (!dst) return hipMemcpyToSymbol(symbol, src, count, offset, kind);
    return hipMemcpy(static_cast<char*>(dst) + offset, src, count, kind);
}
inline hipError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
    void* dst = cheshire::detail::symbolDevicePtr(symbol);
    if (!dst) return hipMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
    return hipMemcpyAsync(static_cast<char*>(dst) + offset, src, count, kind, stream);
}
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaPitchedPtr hipPitchedPtr
#define make_cudaPitchedPtr make_hipPitchedPtr
#define cudaExtent hipExtent
#define make_cudaExtent make_hipExtent
#define cudaPos hipPos
#define make_cudaPos make_hipPos

// ---- arrays / mipmaps -----------------------------------------------------
#define cudaArray hipArray
#define cudaArray_t hipArray_t
#define cudaMemcpy2DFromArray hipMemcpy2DFromArray
#define cudaMallocArray hipMallocArray
#define cudaFreeArray hipFreeArray
#define cudaArrayDefault hipArrayDefault
#define cudaArraySurfaceLoadStore hipArraySurfaceLoadStore
#define cudaMipmappedArray_t hipMipmappedArray_t
// Mipmapped arrays: emulated with one plain array + texture per level unless the platform's
// native support is requested (CHESHIRE_NATIVE_MIPMAP). Linux ROCm 7.2 has no
// hipMallocMipmappedArray on RDNA1 ("Mipmap not supported on one of the devices").
#if !defined(CHESHIRE_NATIVE_MIPMAP)
#define CHESHIRE_EMULATE_MIPMAP 1
#endif
#include <cheshire/mipmap_emu.h>
#if defined(CHESHIRE_EMULATE_MIPMAP)
inline hipError_t cudaMallocMipmappedArray(hipMipmappedArray_t* p, const hipChannelFormatDesc* d, hipExtent e, unsigned int levels, unsigned int flags = 0)
{ return cheshire::mip::mallocMipmappedArray(p, d, e, levels, flags); }
inline hipError_t cudaFreeMipmappedArray(hipMipmappedArray_t m) { return cheshire::mip::freeMipmappedArray(m); }
inline hipError_t cudaGetMipmappedArrayLevel(hipArray_t* a, hipMipmappedArray_t m, unsigned int level) { return cheshire::mip::getMipmappedArrayLevel(a, m, level); }
inline hipError_t cudaArrayGetInfo(hipChannelFormatDesc* d, hipExtent* e, unsigned int* f, hipArray_t a) { return cheshire::mip::arrayGetInfo(d, e, f, a); }
// copies into a level: linear (bridge-managed, spillable) levels are plain hipMemcpy2D targets
inline hipError_t cudaMemcpy2DToArray(hipArray_t dst, size_t wOff, size_t hOff, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)
{ return cheshire::mip::memcpy2DToArray(dst, wOff, hOff, src, spitch, width, height, kind); }
inline hipError_t cudaMemcpy3D(const hipMemcpy3DParms* p) { return cheshire::mip::memcpy3D(p); }
#else
#define cudaMallocMipmappedArray hipMallocMipmappedArray
#define cudaFreeMipmappedArray hipFreeMipmappedArray
#define cudaGetMipmappedArrayLevel hipGetMipmappedArrayLevel
#define cudaArrayGetInfo hipArrayGetInfo
#define cudaMemcpy2DToArray hipMemcpy2DToArray
#define cudaMemcpy3D hipMemcpy3D
#endif
#define cudaChannelFormatDesc hipChannelFormatDesc
#define cudaCreateChannelDesc hipCreateChannelDesc
#define cudaCreateChannelDescHalf hipCreateChannelDescHalf
#define cudaCreateChannelDescHalf1 hipCreateChannelDescHalf1
#define cudaCreateChannelDescHalf2 hipCreateChannelDescHalf2
#define cudaCreateChannelDescHalf4 hipCreateChannelDescHalf4
#define cudaChannelFormatKindFloat hipChannelFormatKindFloat
#define cudaChannelFormatKindUnsigned hipChannelFormatKindUnsigned
#define cudaChannelFormatKindSigned hipChannelFormatKindSigned

// ---- texture / surface objects -------------------------------------------
#define cudaTextureObject_t hipTextureObject_t
#define cudaSurfaceObject_t hipSurfaceObject_t
#if defined(CHESHIRE_EMULATE_MIPMAP)
inline hipError_t cudaCreateTextureObject(hipTextureObject_t* t, const hipResourceDesc* r, const hipTextureDesc* d, const hipResourceViewDesc* v)
{ return cheshire::mip::createTextureObject(t, r, d, v); }
inline hipError_t cudaDestroyTextureObject(hipTextureObject_t t) { return cheshire::mip::destroyTextureObject(t); }
// device side: level table lookup instead of a hardware mip fetch
#define tex2DLod cheshire_tex2DLod
#else
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject
#endif
#define cudaCreateSurfaceObject hipCreateSurfaceObject
#define cudaDestroySurfaceObject hipDestroySurfaceObject
#define cudaResourceDesc hipResourceDesc
#define cudaTextureDesc hipTextureDesc
#define cudaResourceViewDesc hipResourceViewDesc
#define cudaResourceTypeArray hipResourceTypeArray
#define cudaResourceTypeMipmappedArray hipResourceTypeMipmappedArray
#define cudaResourceTypeLinear hipResourceTypeLinear
#define cudaResourceTypePitch2D hipResourceTypePitch2D
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaAddressModeWrap hipAddressModeWrap
#define cudaAddressModeBorder hipAddressModeBorder
#define cudaAddressModeMirror hipAddressModeMirror
#define cudaFilterModePoint hipFilterModePoint
#define cudaFilterModeLinear hipFilterModeLinear
#define cudaReadModeElementType hipReadModeElementType
#define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat
// AliceVision uses these two spellings in places
#define cudaReadElementType hipReadModeElementType
#define cudaReadNormalizedFloat hipReadModeNormalizedFloat
#define cudaBoundaryModeZero hipBoundaryModeZero
#define cudaBoundaryModeClamp hipBoundaryModeClamp
#define cudaBoundaryModeTrap hipBoundaryModeTrap

// ---- half precision (names are identical in hip_fp16.h) -------------------
// __half, __half2, half, half2, __float2half, __half2float: no mapping needed.

// ---- math constants (math_constants.h) ------------------------------------
// hip_math_constants.h provides HIP_PI_F etc.; CUDA names are CUDART_*.
#ifndef CUDART_PI_F
#define CUDART_PI_F 3.14159265358979323846f
#define CUDART_PI 3.14159265358979323846
#endif
#ifndef CUDART_INF_F
#define CUDART_INF_F __builtin_inff()
#define CUDART_INF __builtin_inf()
#endif
#ifndef CUDART_NAN_F
#define CUDART_NAN_F __builtin_nanf("")
#define CUDART_NAN __builtin_nan("")
#endif
#ifndef CUDART_MAX_NORMAL_F
#define CUDART_MAX_NORMAL_F 3.402823466e38f
#endif
#ifndef CUDART_MIN_DENORM_F
#define CUDART_MIN_DENORM_F 1.401298464e-45f
#endif
#ifndef CUDART_VERSION
#define CUDART_VERSION HIP_VERSION
#endif

// ---- things that are the same in HIP and need no map ----------------------
// __global__ __device__ __host__ __constant__ __shared__ __forceinline__
// dim3, blockIdx/blockDim/threadIdx/gridDim, __syncthreads, make_float4 ...
// tex2D<T>, tex2DLod<T>, surf2Dwrite, atomicAdd ...
