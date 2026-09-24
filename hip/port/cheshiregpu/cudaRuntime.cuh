// Cheshire GPU codecs: the CUDA/HIP runtime cheshire::gpu::AsyncBackend drives - one non-blocking
// stream, CUDA names (HIP through Cheshire's compat header, whose copy and memset wrappers also
// order transfers of spilled buffers), and the one kernel every codec stage runs as:
// forEachKernel<F> calls f(i) for i < n, one thread per index. Include only from a .cu file.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cheshire {
namespace gpu {

namespace {

template <class F>
__global__ void forEachKernel(F f, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        f(i);
}

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

}  // namespace

}  // namespace gpu
}  // namespace cheshire
