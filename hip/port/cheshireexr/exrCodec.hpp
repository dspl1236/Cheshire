// CheshireEXR: an OpenEXR scanline decoder that runs on the GPU, with output identical to
// OpenEXR's.
//
// Cheshire's nodes read the EXRs PrepareDenseScene writes - 6000x3376 half RGBA, ZIP or ZIPS, 73 MB
// each - many times over: the depth-map node about ten per view, the texturing node every
// contributing camera per pass (about 60 GB per pass at 884 views, docs/04). On a machine with few
// cores the inflate is the CPU cost of each of those reads. This decoder inflates each chunk on the
// device, one thread per chunk (an EXR ZIP chunk is 16 scanlines compressed on its own, so a file is
// a few hundred independent jobs), and converts to float there.
//
// ZIP, ZIPS and RLE are lossless, so "identical" is not a tolerance: the floats are the ones
// OpenEXR's Imf::InputFile writes into a FLOAT frame buffer, bit for bit, or the decode returns a
// non-Ok status and the caller keeps its OpenEXR path. hip/tests/cheshireexr checks that.
//
// Scope, deliberately the same as Cheshire's direct EXR reader (image/cheshireExr.cpp, step 5v) so
// this can sit in front of it: single-part scanline files, compression NONE, RLE, ZIPS or ZIP,
// channels HALF or FLOAT without subsampling, the data window equal to the display window at the
// origin, and either one channel or exactly R, G, B (and optionally A). Everything else - tiled,
// multi-part, deep, PIZ, PXR24, B44, DWA, UINT channels - returns Status::Unsupported.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cheshire {
namespace exr {

enum class Status
{
    Ok = 0,
    NotExr,       // no EXR magic, or truncated before the chunk table
    Unsupported,  // a valid EXR this decoder does not handle (see above)
    Corrupt,      // a chunk OpenEXR would fail on; the caller's OpenEXR path decides
    NoDevice,     // no GPU, or disabled by CHESHIRE_EXR_GPU=0
    DeviceError,  // a runtime call failed
    InvalidArgument,
};

const char* statusName(Status s);

enum Compression : int
{
    kNone = 0,
    kRle = 1,
    kZips = 2,
    kZip = 3,
    kPiz = 4,
    kPxr24 = 5,
    kB44 = 6,
    kB44a = 7,
    kDwaa = 8,
    kDwab = 9,
};

enum PixelType : int
{
    kUint = 0,
    kHalf = 1,
    kFloat = 2,
};

struct Channel
{
    std::string name;
    int type = kHalf;
    int xSampling = 1, ySampling = 1;
};

// What a caller needs before deciding to decode: the size, the channels (sorted by name, as
// OpenEXR orders them) and the string attributes (AliceVision writes "AliceVision:ColorSpace").
struct Header
{
    int width = 0, height = 0;
    int dataMinX = 0, dataMinY = 0;
    bool displayEqualsData = false;
    int compression = kNone;
    int lineOrder = 0;
    bool tiled = false, multipart = false, deep = false;
    std::vector<Channel> channels;
    std::vector<std::pair<std::string, std::string>> strings;

    const std::string* findString(const std::string& name) const;
    // Chunks in the file: the height over the compression's lines per chunk (16 for ZIP, 1 for ZIPS,
    // RLE and NONE). The decoder inflates one chunk per GPU thread, so this is its parallelism.
    int chunkCount() const;
};

// Parses the header only (host, no device needed). NotExr for a file that is not an EXR.
Status readHeader(const uint8_t* file, size_t size, Header& header);

struct DecodeStats
{
    uint32_t chunks = 0;
    uint32_t rawChunks = 0;       // stored uncompressed (packed size == unpacked size)
    uint64_t packedBytes = 0;
    uint64_t unpackedBytes = 0;
};

// A decode left in device memory (Codec::decodeToDevice): row-major, channels floats per pixel
// (alpha last), rowBytes = width * channels * 4. The memory belongs to the Codec.
struct DeviceImage
{
    const float* data = nullptr;
    int width = 0, height = 0, channels = 0;
    size_t rowBytes = 0;
};

// What Codec::diagnose found about the device copy of the last file this Codec decoded.
struct Diagnosis
{
    bool ran = false;               // false: no file on the device, or a runtime call failed
    size_t bytes = 0;               // the file's size
    size_t copyMismatches = 0;      // bytes that differ in a plain download of the device copy
    size_t firstCopyMismatch = 0;
    size_t shaderMismatches = 0;    // bytes that differ after a kernel copied them to a new buffer
    size_t firstShaderMismatch = 0;
    bool chunkTableMatches = false;  // the chunk table on the device equals the host's
    std::string runtimeError;       // the runtime's last error, as the runtime names it
};

// One codec instance owns its stream and device buffers and reuses them. Not thread-safe; use one
// per thread.
class Codec
{
  public:
    Codec();
    ~Codec();
    Codec(const Codec&) = delete;
    Codec& operator=(const Codec&) = delete;

    // True if a device is present and CHESHIRE_EXR_GPU is not 0. Checked once per process.
    static bool deviceAvailable();

    // nchannels as Cheshire's direct reader takes it: 1 for a one-channel file (its only channel),
    // 3 for R, G, B, 4 for R, G, B, A. allocate(width, height) is called once the file has decoded
    // on the device and must return width * height * nchannels floats, row-major, alpha last; it is
    // not called when the decode fails.
    Status decode(const uint8_t* file,
                  size_t size,
                  int nchannels,
                  const std::function<float*(int, int)>& allocate,
                  DecodeStats* stats = nullptr);

    // The same decode with the floats left on the device, for a caller that uses them there (the
    // depth-map node uploads every image it reads). On Ok the pixels are complete, in memory owned by
    // this Codec that stays valid until its next decode, trim() or destruction.
    Status decodeToDevice(const uint8_t* file, size_t size, int nchannels, DeviceImage& out, DecodeStats* stats = nullptr);

    // Copies a decodeToDevice result to host memory: img.rowBytes * img.height bytes.
    Status download(const DeviceImage& img, float* dst);

    // This Codec's stream (a cudaStream_t / hipStream_t), for work that reads a DeviceImage in order
    // with the Codec's own. Null before the first decode.
    void* stream() const;

    // For a decode that came back Corrupt although the host parses the file: compares the device's
    // copy of file (the last one this Codec uploaded, decoded with nchannels) with the host bytes, once as the copy engine
    // wrote it and once as a kernel reads it, and the chunk table likewise. Slow; diagnostics only.
    Diagnosis diagnose(const uint8_t* file, size_t size, int nchannels);

    // Frees the device buffers, which are otherwise kept for the next decode (a 6000x3376 RGBA file
    // holds about 0.6 GB: the file, the inflated chunks and the floats).
    void trim();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace exr
}  // namespace cheshire
