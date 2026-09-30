// CheshireRemap: PrepareDenseScene's undistortion on the GPU, bit for bit AliceVision's.
//
// UndistortImage (camera/cameraUndistortImage.hpp) samples the source photo at each undistorted pixel's
// distorted position with Sampler2d<SamplerLinear>: the four neighbours weighted in double precision,
// summed per channel, divided by the weight when a border cut it short, rounded back to float. On a
// four-thread host that loop was a quarter of PrepareDenseScene (docs/04, 0.3.8), while the GPU sat idle.
// Here the same arithmetic runs on the device, one thread per output pixel, every product, sum and
// quotient a separate IEEE operation rounded to nearest (no fused multiply-add, as on the host), so the
// output is the host's to the bit. remapStages.hpp holds the per-pixel functor; the host checks
// (hip/tests/cheshireremap) run the same functor on the CPU backend against a copy of AliceVision's
// sampler.
//
// The caller turns AliceVision's double-precision coordinate map into what the sampler actually sees:
// per output pixel the two coordinates as floats (Sampler2d's parameters are floats), or NaN in x when
// Image::contains - which truncates the double coordinates to int - puts the pixel outside the source,
// where UndistortImage leaves the fill colour.
#pragma once

#include <cstddef>
#include <memory>

namespace cheshire {
namespace remap {

enum class Status
{
    Ok = 0,
    NoDevice,     // no GPU, or disabled by CHESHIRE_REMAP=0 (or false, off, no)
    DeviceError,  // a runtime call failed
    InvalidArgument,
};

const char* statusName(Status s);

// One remapper owns its device buffers (the map, a source and a destination image) and a stream, and
// reuses them across calls. Not thread-safe; one per thread at a time.
class Remapper
{
  public:
    Remapper();
    ~Remapper();
    Remapper(const Remapper&) = delete;
    Remapper& operator=(const Remapper&) = delete;

    // True if a device is present and CHESHIRE_REMAP does not switch it off. Checked once per process.
    static bool deviceAvailable();

    // The device memory free now, in bytes (0 without a device), for the caller's bound on remappers.
    static size_t deviceFreeBytes();

    // coords: 2 floats (x, y) per output pixel, width x height, row-major; x NaN for a pixel outside the
    // source. Uploaded unless key, width and height are the last map's.
    Status setMap(const void* key, const float* coords, int width, int height);

    // src: srcWidth x srcHeight RGBA floats, row-major. dst: the map's width x height RGBA floats.
    // fill: the colour of the pixels outside the source.
    Status remap(const float* src, int srcWidth, int srcHeight, const float fill[4], float* dst);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace remap
}  // namespace cheshire
