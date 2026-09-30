// CheshireRemap: the device backend and the public Remapper. CUDA dialect; the HIP build force-includes
// cheshire/cuda_to_hip.h, so device allocations go through the memory bridge like every other Cheshire
// port. Each Remapper has its own non-blocking stream and pinned staging area
// (cheshiregpu/asyncBackend.hpp); the staging area grows to the largest transfer, one RGBA float image
// (288 MB at 4896x3672), so the caller bounds how many remappers exist.
//
// CHESHIRE_REMAP=0 (or false, off, no) disables; read through cheshire/env.h like every CHESHIRE_* switch.
#include "asyncBackend.hpp"
#include "cudaRuntime.cuh"
#include "cheshireRemap.hpp"
#include "remapPipeline.hpp"

#include <cuda_runtime.h>

#include <cheshire/env.h>  // hip/compat/include; in AliceVision, next to bridge.h

#include <cstdio>
#include <mutex>

namespace cheshire {
namespace remap {

namespace {

using GpuBackend = gpu::AsyncBackend<gpu::CudaRuntime>;

std::once_flag g_once;
bool g_available = false;

void probe()
{
    if (!::cheshire::env::flag("CHESHIRE_REMAP", true))
    {
        std::fprintf(stderr, "[cheshire] CheshireRemap: disabled by CHESHIRE_REMAP\n");
        return;
    }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)
    {
        std::fprintf(stderr, "[cheshire] CheshireRemap: no GPU device\n");
        return;
    }
    g_available = true;
}

}  // namespace

const char* statusName(Status s)
{
    switch (s)
    {
        case Status::Ok:
            return "ok";
        case Status::NoDevice:
            return "no device";
        case Status::DeviceError:
            return "device error";
        case Status::InvalidArgument:
            return "invalid argument";
    }
    return "?";
}

struct Remapper::Impl
{
    GpuBackend backend;
    Pipeline<GpuBackend> pipeline{backend};
};

Remapper::Remapper()
  : impl_(new Impl)
{}

Remapper::~Remapper() = default;

bool Remapper::deviceAvailable()
{
    std::call_once(g_once, probe);
    return g_available;
}

size_t Remapper::deviceFreeBytes()
{
    if (!deviceAvailable())
        return 0;
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
        return 0;
    return freeBytes;
}

Status Remapper::setMap(const void* key, const float* coords, int width, int height)
{
    if (!deviceAvailable())
        return Status::NoDevice;
    return impl_->pipeline.setMap(key, coords, width, height);
}

Status Remapper::remap(const float* src, int srcWidth, int srcHeight, const float fill[4], float* dst)
{
    if (!deviceAvailable())
        return Status::NoDevice;
    return impl_->pipeline.remap(src, srcWidth, srcHeight, fill, dst);
}

}  // namespace remap
}  // namespace cheshire
