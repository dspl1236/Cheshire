// Cheshire: the mirror of cuda_to_hip.h.
//
// bridge.h was written against the HIP runtime because that is where it was needed first. The
// CUDA build wants the same allocator, and the bridge's logic has nothing backend-specific in
// it - only the ten entry points below. Rather than rewrite bridge.h in CUDA dialect and risk
// changing behaviour that is validated on five AMD cards, this header maps the HIP names it
// uses onto their CUDA equivalents, so the bridge body compiles unmodified on both.
//
// Safe for the same reason cuda_to_hip.h is safe in the other direction: the CUDA headers never
// declare a hip* name, so nothing here can collide with the real runtime.
//
// Included by bridge.h when the translation unit is not HIP. Never include it in a HIP build.
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#  error "hip_to_cuda.h is for the CUDA backend; HIP builds include <hip/hip_runtime.h> directly"
#endif

#include <cuda_runtime.h>

// ---- types ----------------------------------------------------------------
using hipError_t = cudaError_t;
using hipPitchedPtr = cudaPitchedPtr;
using hipExtent = cudaExtent;
using hipDeviceProp_t = cudaDeviceProp;

// ---- status codes ---------------------------------------------------------
constexpr cudaError_t hipSuccess = cudaSuccess;
constexpr cudaError_t hipErrorOutOfMemory = cudaErrorMemoryAllocation;

// ---- host-allocation flags ------------------------------------------------
constexpr unsigned int hipHostMallocMapped = cudaHostAllocMapped;
// AMD cache-coherency hint with no CUDA equivalent. On NVIDIA the mapped allocation is already
// what the bridge wants (device-visible, host-written once, read over PCIe), so this is a no-op
// rather than a behaviour change: dropping it cannot make the CUDA path differ from plain
// cudaHostAlloc(..., cudaHostAllocMapped).
constexpr unsigned int hipHostMallocNonCoherent = 0u;

// ---- entry points ---------------------------------------------------------
inline cudaError_t hipMalloc(void** devPtr, size_t bytes) { return cudaMalloc(devPtr, bytes); }
inline cudaError_t hipFree(void* devPtr) { return cudaFree(devPtr); }

inline cudaError_t hipMallocPitch(void** devPtr, size_t* pitch, size_t widthBytes, size_t height)
{
    return cudaMallocPitch(devPtr, pitch, widthBytes, height);
}

inline cudaError_t hipMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent)
{
    return cudaMalloc3D(pitchedDevPtr, extent);
}

inline cudaError_t hipHostMalloc(void** ptr, size_t bytes, unsigned int flags)
{
    return cudaHostAlloc(ptr, bytes, flags);
}

inline cudaError_t hipHostFree(void* ptr) { return cudaFreeHost(ptr); }

inline cudaError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags)
{
    return cudaHostGetDevicePointer(devPtr, hostPtr, flags);
}

inline cudaError_t hipMemGetInfo(size_t* freeBytes, size_t* totalBytes)
{
    return cudaMemGetInfo(freeBytes, totalBytes);
}

inline cudaError_t hipGetDevice(int* device) { return cudaGetDevice(device); }

inline cudaError_t hipGetDeviceProperties(cudaDeviceProp* prop, int device)
{
    return cudaGetDeviceProperties(prop, device);
}

inline cudaError_t hipGetLastError() { return cudaGetLastError(); }
