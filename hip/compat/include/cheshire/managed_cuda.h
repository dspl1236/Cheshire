// Cheshire: the CUDA unified-memory arm of the device allocator, for comparison against the
// memory bridge.
//
// The bridge exists because of docs/02 finding #2: on AMD, hipMallocManaged succeeds but behaves
// like mapped host memory - 26 GB/s and no page migration into VRAM - so managed memory was
// never a candidate there. CUDA's unified memory genuinely does migrate pages, with hardware
// fault handling, at page granularity. That is finer than the bridge's whole-buffer placement by
// class, and it is the one thing the bridge cannot do.
//
// It can also lose badly: SGM sweeps the similarity volume repeatedly, and page migration can
// thrash a working set that whole-buffer placement simply pins. Which effect wins is a
// measurement, not a prediction, so this header exists to make the measurement possible.
//
//   CHESHIRE_CUDA_MANAGED=1   route device allocations to cudaMallocManaged instead of the bridge
//
// CUDA only: it is never compiled into a HIP build (see memory.hpp, which selects between this
// and the bridge).
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include "env.h"

namespace cheshire {
namespace managed {

inline bool enabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_CUDA_MANAGED");
    return on;
}

// Pitch for the emulated pitched allocations below. cudaMallocManaged has no pitched form, so we
// pick the alignment the device reports for pitched texture binding and lay the rows out by hand;
// 512 covers every NVIDIA part in practice and is what the driver returns for texturePitchAlignment
// on the cards this is measured on.
inline size_t pitchAlign()
{
    static const size_t a = [] {
        cudaDeviceProp p{};
        int d = 0;
        if (cudaGetDevice(&d) == cudaSuccess && cudaGetDeviceProperties(&p, d) == cudaSuccess && p.texturePitchAlignment > 0)
            return size_t(p.texturePitchAlignment);
        return size_t(512);
    }();
    return a;
}

inline size_t alignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

inline cudaError_t malloc(void** devPtr, size_t bytes)
{
    return cudaMallocManaged(devPtr, bytes);
}

inline cudaError_t mallocPitch(void** devPtr, size_t* pitch, size_t widthBytes, size_t height)
{
    const size_t p = alignUp(widthBytes, pitchAlign());
    const cudaError_t e = cudaMallocManaged(devPtr, p * height);
    if (e == cudaSuccess)
        *pitch = p;
    return e;
}

inline cudaError_t malloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent)
{
    const size_t p = alignUp(extent.width, pitchAlign());
    void* ptr = nullptr;
    const cudaError_t e = cudaMallocManaged(&ptr, p * extent.height * extent.depth);
    if (e != cudaSuccess)
        return e;
    pitchedDevPtr->ptr = ptr;
    pitchedDevPtr->pitch = p;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
    return cudaSuccess;
}

// cudaFree releases managed allocations too, so no registry is needed on this path.
inline cudaError_t free(void* devPtr) { return cudaFree(devPtr); }

}  // namespace managed
}  // namespace cheshire
