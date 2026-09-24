// CheshireJPG: the device backend and the public Codec. CUDA dialect; the HIP build
// force-includes cheshire/cuda_to_hip.h, so device allocations go through the memory bridge like
// every other Cheshire port (a very large image spills to host RAM instead of failing).
//
// Each Codec has its own non-blocking stream and its own pinned staging area
// (jpegAsyncBackend.hpp). Every copy, memset and kernel of a decode or encode goes on that stream,
// and the waits inside a call - the flag read back after each synchronisation round, the final
// download - wait on that stream only. Codecs on different threads therefore queue their work
// independently. Before this, everything ran on the legacy default stream with synchronous copies,
// and on the RX 9070 box throughput stopped at 48 images/s from two threads on (docs/19).
//
// The staging area grows to the largest single transfer, typically the decoded image: about 36 MB
// of pinned memory per Codec for 4032x3024 RGB.
//
// CHESHIRE_JPG=0 disables.
#include "jpegAsyncBackend.hpp"
#include "jpegCodec.hpp"
#include "jpegPipeline.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace cheshire {
namespace jpeg {

namespace {

template <class F>
__global__ void forEachKernel(F f, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        f(i);
}

// The runtime AsyncBackend drives: one non-blocking stream, CUDA names (HIP through the compat
// header, whose copy and memset wrappers also order transfers of spilled buffers).
class CudaRuntime
{
  public:
    bool streamCreate() { return ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    void streamDestroy() { (void)cudaStreamDestroy(stream_); }
    bool streamSync() { return ok(cudaStreamSynchronize(stream_)); }
    bool deviceAlloc(void** p, size_t n) { return ok(cudaMalloc(p, n)); }
    void deviceFree(void* p) { (void)cudaFree(p); }
    bool hostAlloc(void** p, size_t n) { return ok(cudaHostAlloc(p, n, cudaHostAllocDefault)); }
    void hostFree(void* p) { (void)cudaFreeHost(p); }
    bool copyToDevice(void* d, const void* h, size_t n)
    {
        return ok(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, stream_));
    }
    bool copyToHost(void* h, const void* d, size_t n)
    {
        return ok(cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, stream_));
    }
    bool zero(void* d, size_t n) { return ok(cudaMemsetAsync(d, 0, n, stream_)); }
    template <class F>
    bool launch(size_t n, const F& f)
    {
        const unsigned block = 256;
        const size_t grid = (n + block - 1) / block;
        forEachKernel<F><<<(unsigned)grid, block, 0, stream_>>>(f, n);
        return ok(cudaGetLastError());
    }
    const char* error() const { return cudaGetErrorString(last_); }

  private:
    bool ok(cudaError_t e)
    {
        if (e != cudaSuccess)
            last_ = e;
        return e == cudaSuccess;
    }
    cudaStream_t stream_ = nullptr;
    cudaError_t last_ = cudaSuccess;
};

using GpuBackend = AsyncBackend<CudaRuntime>;

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
