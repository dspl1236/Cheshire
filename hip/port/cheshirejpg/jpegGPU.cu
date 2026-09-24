// CheshireJPG: the device backend and the public Codec. CUDA dialect; the HIP build
// force-includes cheshire/cuda_to_hip.h, so device allocations go through the memory bridge like
// every other Cheshire port (a very large image spills to host RAM instead of failing).
//
// Each Codec has its own non-blocking stream and its own pinned staging area
// (cheshiregpu/asyncBackend.hpp). Every copy, memset and kernel of a decode or encode goes on that stream,
// and the waits inside a call - the flag read back after each synchronisation round, the final
// download - wait on that stream only. Codecs on different threads therefore queue their work
// independently. Before this, everything ran on the legacy default stream with synchronous copies,
// and on the RX 9070 box throughput stopped at 48 images/s from two threads on (docs/19).
//
// The staging area grows to the largest single transfer, typically the decoded image: about 36 MB
// of pinned memory per Codec for 4032x3024 RGB.
//
// CHESHIRE_JPG=0 disables.
#include "asyncBackend.hpp"
#include "cudaRuntime.cuh"
#include "jpegCodec.hpp"
#include "jpegPipeline.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace cheshire {
namespace jpeg {

namespace {

using GpuBackend = gpu::AsyncBackend<gpu::CudaRuntime>;

std::once_flag g_once;
bool g_available = false;

void probe()
{
    if (const char* e = std::getenv("CHESHIRE_JPG"))
        if (e[0] == '0')
        {
            std::fprintf(stderr, "[cheshire] CheshireJPG: disabled by CHESHIRE_JPG=0\n");
            return;
        }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)
    {
        std::fprintf(stderr, "[cheshire] CheshireJPG: no GPU device\n");
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

Status Codec::decode(const uint8_t* jpeg, size_t size, Image& out, DecodeStats* stats)
{
    if (!jpeg)
        return Status::InvalidArgument;
    if (!deviceAvailable())
        return Status::NoDevice;
    return impl_->pipeline.decode(jpeg, size, out, stats);
}

Status Codec::encode(const uint8_t* pixels,
                     int width,
                     int height,
                     int channels,
                     size_t rowBytes,
                     const EncodeOptions& options,
                     std::vector<uint8_t>& out)
{
    if (!deviceAvailable())
        return Status::NoDevice;
    return impl_->pipeline.encode(pixels, width, height, channels, rowBytes, options, out);
}

}  // namespace jpeg
}  // namespace cheshire
