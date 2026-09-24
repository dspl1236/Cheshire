// CheshireJPG: a baseline JPEG decoder and encoder that run on any GPU HIP (or CUDA) runs
// on, with output identical to libjpeg-turbo's.
//
// rocJPEG, AMD's JPEG library, hands the bitstream to the VCN block's fixed-function JPEG engine
// through VA-API. That makes it Linux-only (there is no VA-API on Windows), limits it to the parts
// ROCm lists (gfx908 and newer; no RDNA1, which Cheshire validates on), and makes it decode-only.
// This codec is compute kernels instead: every stage, including the Huffman decode of streams
// without restart markers, runs as ordinary HIP/CUDA code, so it builds wherever the rest of
// Cheshire does - Windows and Linux, RDNA1 through RDNA4 - and on NVIDIA through the CUDA build.
//
// Scope, deliberately: baseline and extended-sequential Huffman, 8-bit, one scan holding every
// component (every camera JPEG Cheshire has seen), grayscale or three components, any sampling
// factors libjpeg accepts. Progressive, arithmetic-coded, 12-bit, CMYK/YCCK and multi-scan
// sequential files return Status::Unsupported and the caller keeps its CPU path; a stream the
// parallel decode cannot account for exactly returns Status::Corrupt, likewise.
//
// "Identical" means: decode gives the bytes libjpeg-turbo's jpeg_read_scanlines gives with its
// defaults (JDCT_ISLOW, fancy upsampling, RGB or grayscale out), and encode gives the file
// jpeg_write_scanlines writes with jpeg_set_defaults + jpeg_set_quality(q, TRUE) and the chosen
// sampling factors and restart interval. hip/tests/cheshirejpg checks both.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cheshire {
namespace jpeg {

enum class Status
{
    Ok = 0,
    NotJpeg,      // no SOI, or truncated before the first scan
    Unsupported,  // valid JPEG this codec does not handle (see above)
    Corrupt,      // damaged stream; libjpeg would decode it with warnings
    NoDevice,     // no GPU, or disabled by CHESHIRE_JPG=0
    DeviceError,  // a runtime call failed
    InvalidArgument,
};

const char* statusName(Status s);

struct Image
{
    int width = 0;
    int height = 0;
    int channels = 0;  // 1 (grayscale) or 3 (RGB)
    std::vector<uint8_t> pixels;  // row-major, width * channels bytes per row
};

enum class Subsampling
{
    S444,  // 1x1 all components
    S422,  // luma 2x1
    S420,  // luma 2x2 (libjpeg's default)
    S440,  // luma 1x2
    Gray,  // one component
};

struct EncodeOptions
{
    int quality = 75;                        // libjpeg's jpeg_set_quality scale, 1..100
    Subsampling subsampling = Subsampling::S420;
    int restartInterval = 0;                 // MCUs between RST markers, 0 = none
};

// What happened inside a decode, for logging and for the docs' measurements.
struct DecodeStats
{
    uint32_t subsequences = 0;
    uint32_t syncRounds = 0;       // Huffman synchronisation rounds after the initial pass, the
                                   // last one being the round that found nothing left to change
    uint32_t restartSegments = 0;
};

// One codec instance owns its device buffers and reuses them across calls. Not thread-safe; use
// one per thread.
class Codec
{
  public:
    Codec();
    ~Codec();
    Codec(const Codec&) = delete;
    Codec& operator=(const Codec&) = delete;

    // True if a device is present and CHESHIRE_JPG is not 0. Checked once per process.
    static bool deviceAvailable();

    Status decode(const uint8_t* jpeg, size_t size, Image& out, DecodeStats* stats = nullptr);

    // pixels: channels = 1 (gray) or 3 (RGB), rowBytes >= width * channels.
    Status encode(const uint8_t* pixels,
                  int width,
                  int height,
                  int channels,
                  size_t rowBytes,
                  const EncodeOptions& options,
                  std::vector<uint8_t>& out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace jpeg
}  // namespace cheshire
