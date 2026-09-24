// CheshireJPG: the device backend and the public Codec. CUDA dialect; the HIP build
// force-includes cheshire/cuda_to_hip.h, so device allocations go through the memory bridge like
// every other Cheshire port (a very large image spills to host RAM instead of failing).
//
// Everything runs on the legacy default stream: uploads and downloads are synchronous copies and
// kernels queue behind them, which keeps the ordering obvious. CHESHIRE_JPG=0 disables.
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

struct GpuBackend
{
    bool failed = false;

    void check(cudaError_t e, const char* what)
    {
        if (e != cudaSuccess && !failed)
        {
            std::fprintf(stderr, "[cheshire] CheshireJPG: %s failed: %s\n", what, cudaGetErrorString(e));
            failed = true;
        }
    }
    void* alloc(size_t bytes)
    {
        void* p = nullptr;
        check(cudaMalloc(&p, bytes), "cudaMalloc");
        return failed ? nullptr : p;
    }
    void release(void* p)
    {
        if (p)
            (void)cudaFree(p);
    }
    void upload(void* d, const void* h, size_t n)
    {
        if (!failed)
            check(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice), "upload");
    }
    void download(void* h, const void* d, size_t n)
    {
        if (!failed)
            check(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost), "download");
    }
    void zero(void* d, size_t n)
    {
        if (!failed)
            check(cudaMemset(d, 0, n), "memset");
    }
    template <class F>
    void forEach(size_t n, const F& f)
    {
        if (failed || n == 0)
            return;
        const unsigned block = 256;
        const size_t grid = (n + block - 1) / block;
        forEachKernel<F><<<(unsigned)grid, block>>>(f, n);
        check(cudaGetLastError(), "kernel launch");
    }
    bool ok() const { return !failed; }
};

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
    impl_->backend.failed = false;
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
    impl_->backend.failed = false;
    return impl_->pipeline.encode(pixels, width, height, channels, rowBytes, options, out);
}

}  // namespace jpeg
}  // namespace cheshire
