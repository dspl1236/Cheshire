// CheshireEXR: the decode sequence, written once over a backend (cheshiregpu/asyncBackend.hpp on
// the device, cheshiregpu/cpuBackend.hpp in the host check).
#pragma once

#include "exrCodec.hpp"
#include "exrHost.hpp"
#include "exrStages.hpp"

#include <cstring>

namespace cheshire {
namespace exr {

template <class Backend>
class Pipeline
{
  public:
    explicit Pipeline(Backend& b)
      : b_(b)
    {}

    ~Pipeline()
    {
        for (Slot& s : slots_)
            b_.release(s.p);
    }

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Status decode(const uint8_t* file,
                  size_t size,
                  int nchannels,
                  const std::function<float*(int, int)>& allocate,
                  DecodeStats* stats)
    {
        if (!b_.begin())
            return Status::DeviceError;
        Plan p;
        Status st = makePlan(file, size, nchannels, p);
        if (st != Status::Ok)
            return st;
        const ExrGeom& g = p.geom;

        uint8_t* dFile = get<uint8_t>(kFile, size);
        ChunkInfo* dChunks = get<ChunkInfo>(kChunks, p.chunks.size());
        uint8_t* dScratch = get<uint8_t>(kScratch, p.scratchBytes);
        uint32_t* dFlag = get<uint32_t>(kFlag, 1);
        const size_t pixels = (size_t)g.width * g.height;
        const size_t outBytes = pixels * (size_t)g.nOut * sizeof(float);
        float* dOut = get<float>(kOut, pixels * (size_t)g.nOut);
        if (!b_.ok())
            return Status::DeviceError;

        uploadPieces(dFile, file, size);
        b_.upload(dChunks, p.chunks.data(), p.chunks.size() * sizeof(ChunkInfo));
        b_.zero(dFlag, sizeof(uint32_t));
        b_.forEach(p.chunks.size(), InflateChunks{g, dFile, dChunks, dScratch, dFlag});
        uint32_t bad = 0;
        b_.download(&bad, dFlag, sizeof(uint32_t));
        if (!b_.ok())
            return Status::DeviceError;
        if (bad)
            return Status::Corrupt;
        b_.forEach(pixels, ConvertPixels{g, dFile, dChunks, dScratch, dOut});

        float* dst = allocate(g.width, g.height);
        if (!dst)
            return Status::InvalidArgument;
        downloadPieces(dst, dOut, outBytes);
        if (!b_.ok())
            return Status::DeviceError;
        if (stats)
            *stats = p.stats;
        return Status::Ok;
    }

  private:
    // Large transfers go in pieces, so the pinned staging area stays at one piece (16 MB) per
    // Codec rather than the size of a decoded image (324 MB for 6000x3376 RGBA floats); a
    // dozen decoding threads would otherwise pin gigabytes.
    static constexpr size_t kPiece = (size_t)16 << 20;

    void uploadPieces(void* d, const void* h, size_t n)
    {
        for (size_t off = 0; off < n; off += kPiece)
            b_.upload(static_cast<uint8_t*>(d) + off, static_cast<const uint8_t*>(h) + off, n - off < kPiece ? n - off : kPiece);
    }
    void downloadPieces(void* h, const void* d, size_t n)
    {
        for (size_t off = 0; off < n; off += kPiece)
            b_.download(static_cast<uint8_t*>(h) + off, static_cast<const uint8_t*>(d) + off, n - off < kPiece ? n - off : kPiece);
    }

    enum SlotId
    {
        kFile,
        kChunks,
        kScratch,
        kFlag,
        kOut,
        kSlotCount
    };

    struct Slot
    {
        void* p = nullptr;
        size_t cap = 0;
    };

    template <class T>
    T* get(SlotId id, size_t count)
    {
        const size_t bytes = (count ? count : 1) * sizeof(T);
        Slot& s = slots_[id];
        if (s.cap < bytes)
        {
            b_.release(s.p);
            s.p = b_.alloc(bytes);
            s.cap = s.p ? bytes : 0;
        }
        return static_cast<T*>(s.p);
    }

    Backend& b_;
    Slot slots_[kSlotCount];
};

}  // namespace exr
}  // namespace cheshire
