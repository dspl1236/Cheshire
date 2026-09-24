// Cheshire GPU codecs (CheshireJPG, CheshireEXR): the backend for a device that runs work
// asynchronously on a stream.
//
// The codec pipelines (cheshirejpg/jpegPipeline.hpp, cheshireexr/exrPipeline.hpp) need a backend whose upload() may return before the transfer has run, while
// the caller reuses its buffer at once, and whose download() returns with the data there. On a
// GPU both are asynchronous copies on the Codec's own stream, through a pinned staging area:
//
//   upload:   copy into a fresh slice of the staging area, queue the transfer to the device.
//   download: queue the transfer into a slice, wait for the stream, copy out.
//
// A slice is only handed out again after a wait on the stream: when the area is full, and after
// every download. So no queued transfer can read staging memory that has since been overwritten.
// Kernels, copies and memsets all go on the one stream, so they run in the order issued.
//
// The class is a template over the runtime so that exactly this code is also tested without a
// GPU. cudaRuntime.cuh is the CUDA/HIP runtime the codecs' .cu files instantiate it with. The host
// checks (hip/tests/cheshirejpg, hip/tests/cheshireexr) instantiate it with
// hip/tests/cheshiregpu/deferredRuntime.hpp, which queues every operation and runs the queue only
// when the stream is waited on. A read of the staging area before its transfer had run would then
// show up as output that differs from the reference library's.
//
// Runtime interface (each call returns false on failure, and error() then describes it):
//   bool streamCreate(); void streamDestroy(); bool streamSync();
//   bool deviceAlloc(void**, size_t); void deviceFree(void*);
//   bool hostAlloc(void**, size_t);   void hostFree(void*);      // pinned
//   bool copyToDevice(void* d, const void* h, size_t);            // async on the stream
//   bool copyToHost(void* h, const void* d, size_t);              // async on the stream
//   bool zero(void* d, size_t);                                   // async on the stream
//   template <class F> bool launch(size_t n, const F& f);         // f(i) for i < n, on the stream
//   const char* error();
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace cheshire {
namespace gpu {

template <class Runtime>
class AsyncBackend
{
  public:
    AsyncBackend() = default;
    AsyncBackend(const AsyncBackend&) = delete;
    AsyncBackend& operator=(const AsyncBackend&) = delete;

    ~AsyncBackend()
    {
        if (hasStream_)
            (void)rt_.streamSync();
        if (stage_)
            rt_.hostFree(stage_);
        if (hasStream_)
            rt_.streamDestroy();
    }

    Runtime& runtime() { return rt_; }
    const Runtime& runtime() const { return rt_; }

    // Start of a decode or encode: clear an earlier failure, create the stream on first use, and
    // make sure nothing from an earlier call (which may have stopped early) is still in flight.
    bool begin()
    {
        failed_ = false;
        if (!hasStream_)
        {
            hasStream_ = rt_.streamCreate();
            check(hasStream_, "stream create");
        }
        sync();
        used_ = 0;
        return !failed_;
    }

    void* alloc(size_t bytes)
    {
        void* p = nullptr;
        check(rt_.deviceAlloc(&p, bytes), "device alloc");
        return failed_ ? nullptr : p;
    }
    void release(void* p)
    {
        if (!p)
            return;
        sync();  // queued work may still use the buffer
        rt_.deviceFree(p);
    }
    void upload(void* d, const void* h, size_t n)
    {
        if (failed_ || n == 0)
            return;
        uint8_t* s = stage(n);
        if (!s)
            return;
        std::memcpy(s, h, n);
        check(rt_.copyToDevice(d, s, n), "upload");
    }
    void download(void* h, const void* d, size_t n)
    {
        if (failed_ || n == 0)
            return;
        uint8_t* s = stage(n);
        if (!s)
            return;
        check(rt_.copyToHost(s, d, n), "download");
        sync();
        if (!failed_)
            std::memcpy(h, s, n);
        used_ = 0;  // the stream is idle: every slice is free again
    }
    // Like download, but hands the staged bytes to consume(const uint8_t*) instead of copying
    // them out, so a caller that transforms the data (half to float, say) reads pinned memory once.
    template <class F>
    void downloadVia(const void* d, size_t n, F&& consume)
    {
        if (failed_ || n == 0)
            return;
        uint8_t* s = stage(n);
        if (!s)
            return;
        check(rt_.copyToHost(s, d, n), "download");
        sync();
        if (!failed_)
            consume(static_cast<const uint8_t*>(s));
        used_ = 0;
    }
    void zero(void* d, size_t n)
    {
        if (!failed_ && n)
            check(rt_.zero(d, n), "memset");
    }
    template <class F>
    void forEach(size_t n, const F& f)
    {
        if (!failed_ && n)
            check(rt_.launch(n, f), "kernel launch");
    }
    bool ok() const { return !failed_; }
    // Wait for everything queued so far; true if it all succeeded. For results that stay on the
    // device: once this returns they are complete, whatever stream reads them next.
    bool finish()
    {
        sync();
        used_ = 0;
        return !failed_;
    }

  private:
    void check(bool good, const char* what)
    {
        if (!good && !failed_)
        {
            std::fprintf(stderr, "[cheshire] GPU codec: %s failed: %s\n", what, rt_.error());
            failed_ = true;
        }
    }
    void sync()
    {
        if (hasStream_)
            check(rt_.streamSync(), "stream synchronize");
    }
    // A slice of n bytes of pinned memory that no queued transfer uses.
    uint8_t* stage(size_t n)
    {
        const size_t need = (n + 255) & ~(size_t)255;
        if (used_ + need > cap_)
        {
            sync();  // every transfer through the area has completed
            used_ = 0;
            if (need > cap_)
            {
                if (stage_)
                    rt_.hostFree(stage_);
                stage_ = nullptr;
                cap_ = 0;
                const size_t want = (std::max)(need, (size_t)8 << 20);
                void* p = nullptr;
                check(rt_.hostAlloc(&p, want), "pinned host alloc");
                if (failed_)
                    return nullptr;
                stage_ = static_cast<uint8_t*>(p);
                cap_ = want;
            }
        }
        if (failed_)
            return nullptr;
        uint8_t* p = stage_ + used_;
        used_ += need;
        return p;
    }

    Runtime rt_;
    bool hasStream_ = false;
    bool failed_ = false;
    uint8_t* stage_ = nullptr;  // pinned
    size_t cap_ = 0;
    size_t used_ = 0;
};

}  // namespace gpu
}  // namespace cheshire
