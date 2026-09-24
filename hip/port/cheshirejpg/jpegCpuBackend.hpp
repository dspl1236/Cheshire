// CheshireJPG: a host backend for jpegPipeline.hpp. It runs every stage functor as a plain
// loop, in the order the device runs them, so the arithmetic and the control flow of the GPU path
// can be checked against libjpeg-turbo on a machine without a GPU (hip/tests/cheshirejpg). It is a
// verification tool, not a fast CPU codec - libjpeg-turbo is that.
#pragma once

#include <cstdlib>
#include <cstring>

namespace cheshire {
namespace jpeg {

struct CpuBackend
{
    void* alloc(size_t bytes) { return std::malloc(bytes); }
    void release(void* p) { std::free(p); }
    void upload(void* d, const void* h, size_t n) { std::memcpy(d, h, n); }
    void download(void* h, const void* d, size_t n) { std::memcpy(h, d, n); }
    void zero(void* d, size_t n) { std::memset(d, 0, n); }
    template <class F>
    void forEach(size_t n, const F& f)
    {
        for (size_t i = 0; i < n; ++i)
            f(i);
    }
    bool ok() const { return true; }
};

}  // namespace jpeg
}  // namespace cheshire
