// Cheshire GPU codecs: a host backend for the codec pipelines. It runs every stage functor as a
// plain loop, in the order the device runs them, so the arithmetic and the control flow of the GPU
// path can be checked against the reference library (libjpeg-turbo, OpenEXR) on a machine without
// a GPU (hip/tests/cheshirejpg, hip/tests/cheshireexr). It is a verification tool, not a fast CPU
// codec.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace cheshire {
namespace gpu {

struct CpuBackend
{
    bool begin() { return true; }
    void* alloc(size_t bytes) { return std::malloc(bytes); }
    void release(void* p) { std::free(p); }
    void upload(void* d, const void* h, size_t n) { std::memcpy(d, h, n); }
    void download(void* h, const void* d, size_t n) { std::memcpy(h, d, n); }
    template <class F>
    void downloadVia(const void* d, size_t n, F&& consume)
    {
        if (n)
            consume(static_cast<const uint8_t*>(d));
    }
    void zero(void* d, size_t n) { std::memset(d, 0, n); }
    template <class F>
    void forEach(size_t n, const F& f)
    {
        for (size_t i = 0; i < n; ++i)
            f(i);
    }
    bool ok() const { return true; }
    bool finish() { return true; }
};

}  // namespace gpu
}  // namespace cheshire
