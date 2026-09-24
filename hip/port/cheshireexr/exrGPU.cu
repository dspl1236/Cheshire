// CheshireEXR: the device backend and the public Codec. CUDA dialect; the HIP build force-includes
// cheshire/cuda_to_hip.h (allocations go through the memory bridge). Each Codec has its own stream
// and pinned staging area (cheshiregpu/asyncBackend.hpp), as CheshireJPG's does.
// CHESHIRE_EXR_GPU=0 disables.
#include "asyncBackend.hpp"
#include "cudaRuntime.cuh"
#include "exrCodec.hpp"
#include "exrPipeline.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace cheshire {
namespace exr {

namespace {

using GpuBackend = gpu::AsyncBackend<gpu::CudaRuntime>;

std::once_flag g_once;
bool g_available = false;

void probe()
{
    if (const char* e = std::getenv("CHESHIRE_EXR_GPU"))
        if (e[0] == '0')
        {
            std::fprintf(stderr, "[cheshire] CheshireEXR: disabled by CHESHIRE_EXR_GPU=0\n");
            return;
        }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)
    {
        std::fprintf(stderr, "[cheshire] CheshireEXR: no GPU device\n");
        return;
    }
    g_available = true;
}

}  // namespace

struct Codec::Impl
{
    GpuBackend backend;
    Pipeline<GpuBackend> pipeline{backend};
};

Codec::Codec()
  : impl_(new Impl)
{}

Codec::~Codec() = default;

bool Codec::deviceAvailable()
{
    std::call_once(g_once, probe);
    return g_available;
}

Status Codec::decode(const uint8_t* file,
                     size_t size,
                     int nchannels,
                     const std::function<float*(int, int)>& allocate,
                     DecodeStats* stats)
{
    if (!file || !allocate)
        return Status::InvalidArgument;
    if (!deviceAvailable())
        return Status::NoDevice;
    return impl_->pipeline.decode(file, size, nchannels, allocate, stats);
}

Status Codec::decodeToDevice(const uint8_t* file, size_t size, int nchannels, DeviceImage& out, DecodeStats* stats)
{
    out = DeviceImage{};
    if (!file)
        return Status::InvalidArgument;
    if (!deviceAvailable())
        return Status::NoDevice;
    return impl_->pipeline.decodeToDevice(file, size, nchannels, out, stats);
}

Status Codec::download(const DeviceImage& img, float* dst) { return impl_->pipeline.download(img, dst); }

void* Codec::stream() const { return impl_->backend.runtime().stream(); }

void Codec::trim() { impl_->pipeline.trim(); }

}  // namespace exr
}  // namespace cheshire
