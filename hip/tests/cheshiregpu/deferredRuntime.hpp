// Cheshire GPU codec checks: a runtime for cheshire::gpu::AsyncBackend that makes asynchrony
// visible on the host. Used by hip/tests/cheshirejpg and hip/tests/cheshireexr.
//
// Every copy, memset and kernel is queued and runs only when the stream is waited on, in the order
// issued - the latest a real stream is allowed to run it. So a staging slice overwritten before its
// transfer ran, or host code that reads a result without a download, produces wrong output, and
// the comparison with the reference library in the host checks reports it. Fresh allocations are
// filled with garbage, as nothing promises that cudaMalloc returns zeros either.
#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

namespace check {

class DeferredRuntime
{
  public:
    bool streamCreate() { return true; }
    void streamDestroy() {}
    bool streamSync()
    {
        for (auto& op : queue_)
            op();
        queue_.clear();
        ++syncs_;
        return true;
    }
    bool deviceAlloc(void** p, size_t n) { return garbage(p, n); }
    void deviceFree(void* p) { std::free(p); }
    bool hostAlloc(void** p, size_t n) { return garbage(p, n); }
    void hostFree(void* p) { std::free(p); }
    bool copyToDevice(void* d, const void* h, size_t n)
    {
        queue_.push_back([=] { std::memcpy(d, h, n); });  // reads h when it runs, not now
        return true;
    }
    bool copyToHost(void* h, const void* d, size_t n)
    {
        queue_.push_back([=] { std::memcpy(h, d, n); });
        return true;
    }
    bool zero(void* d, size_t n)
    {
        queue_.push_back([=] { std::memset(d, 0, n); });
        return true;
    }
    template <class F>
    bool launch(size_t n, const F& f)
    {
        queue_.push_back([=] {
            for (size_t i = 0; i < n; ++i)
                f(i);
        });
        return true;
    }
    const char* error() const { return "deferred runtime"; }
    unsigned long syncs() const { return syncs_; }

  private:
    static bool garbage(void** p, size_t n)
    {
        *p = std::malloc(n ? n : 1);
        if (*p)
            std::memset(*p, 0xCD, n);
        return *p != nullptr;
    }
    std::vector<std::function<void()>> queue_;
    unsigned long syncs_ = 0;
};

}  // namespace check
