// CheshireRemap: the steps of a remap over a backend (cheshiregpu/asyncBackend.hpp on the device,
// cheshiregpu/cpuBackend.hpp in the host checks): the map uploaded once per key, then per image the
// source up, the functor over every output pixel, the result down.
#pragma once

#include "cheshireRemap.hpp"
#include "remapStages.hpp"

#include <cstddef>
#include <cstring>

namespace cheshire {
namespace remap {

template <class Backend>
class Pipeline
{
  public:
    explicit Pipeline(Backend& b)
      : b_(b)
    {}
    ~Pipeline()
    {
        b_.release(dMap_);
        b_.release(dSrc_);
        b_.release(dDst_);
    }
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Status setMap(const void* key, const float* coords, int width, int height)
    {
        if (!coords || width <= 0 || height <= 0)
            return Status::InvalidArgument;
        if (dMap_ && key == mapKey_ && width == width_ && height == height_)
            return Status::Ok;
        if (!b_.begin())
            return Status::DeviceError;
        const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 2 * sizeof(float);
        if (bytes > mapBytes_)
        {
            b_.release(dMap_);
            dMap_ = static_cast<float*>(b_.alloc(bytes));
            mapBytes_ = dMap_ ? bytes : 0;
        }
        mapKey_ = nullptr;
        if (!dMap_)
            return Status::DeviceError;
        b_.upload(dMap_, coords, bytes);
        if (!b_.finish())
            return Status::DeviceError;
        mapKey_ = key;
        width_ = width;
        height_ = height;
        return Status::Ok;
    }

    Status remap(const float* src, int srcWidth, int srcHeight, const float fill[4], float* dst)
    {
        if (!src || !dst || !fill || srcWidth <= 0 || srcHeight <= 0 || !dMap_)
            return Status::InvalidArgument;
        if (!b_.begin())
            return Status::DeviceError;
        const size_t srcBytes = static_cast<size_t>(srcWidth) * static_cast<size_t>(srcHeight) * 4 * sizeof(float);
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        const size_t dstBytes = n * 4 * sizeof(float);
        if (!reserve(dSrc_, srcCap_, srcBytes) || !reserve(dDst_, dstCap_, dstBytes))
            return Status::DeviceError;
        b_.upload(dSrc_, src, srcBytes);
        UndistortPixel f{dSrc_, srcWidth, srcHeight, dMap_, dDst_, {fill[0], fill[1], fill[2], fill[3]}};
        b_.forEach(n, f);
        b_.download(dst, dDst_, dstBytes);
        return b_.finish() ? Status::Ok : Status::DeviceError;
    }

  private:
    bool reserve(float*& p, size_t& cap, size_t bytes)
    {
        if (bytes <= cap && p)
            return true;
        b_.release(p);
        p = static_cast<float*>(b_.alloc(bytes));
        cap = p ? bytes : 0;
        return p != nullptr;
    }

    Backend& b_;
    const void* mapKey_ = nullptr;
    int width_ = 0, height_ = 0;
    float* dMap_ = nullptr;
    size_t mapBytes_ = 0;
    float* dSrc_ = nullptr;
    size_t srcCap_ = 0;
    float* dDst_ = nullptr;
    size_t dstCap_ = 0;
};

}  // namespace remap
}  // namespace cheshire
