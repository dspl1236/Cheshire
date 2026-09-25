// CheshireEXR: the decode sequence, written once over a backend (cheshiregpu/asyncBackend.hpp on
// the device, cheshiregpu/cpuBackend.hpp in the host check).
#pragma once

#include "exrCodec.hpp"
#include "exrHost.hpp"
#include "exrStages.hpp"

#include <cstring>
#include <vector>

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
        Plan p;
        const float* dOut = nullptr;
        Status st = run(file, size, nchannels, p, dOut);
        if (st != Status::Ok)
            return st;
        const ExrGeom& g = p.geom;
        float* dst = allocate(g.width, g.height);
        if (!dst)
            return Status::InvalidArgument;
        downloadPieces(dst, dOut, (size_t)g.width * g.height * (size_t)g.nOut * sizeof(float));
        if (!b_.ok())
            return Status::DeviceError;
        if (stats)
            *stats = p.stats;
        return Status::Ok;
    }

    // The same decode, with the floats left in device memory: out describes this pipeline's output
    // buffer, complete when this returns, and valid until the next decode or trim().
    Status decodeToDevice(const uint8_t* file, size_t size, int nchannels, DeviceImage& out, DecodeStats* stats)
    {
        out = DeviceImage{};
        Plan p;
        const float* dOut = nullptr;
        Status st = run(file, size, nchannels, p, dOut);
        if (st != Status::Ok)
            return st;
        if (!b_.finish())
            return Status::DeviceError;
        const ExrGeom& g = p.geom;
        out.data = dOut;
        out.width = g.width;
        out.height = g.height;
        out.channels = g.nOut;
        out.rowBytes = (size_t)g.width * (size_t)g.nOut * sizeof(float);
        if (stats)
            *stats = p.stats;
        return Status::Ok;
    }

    // A decodeToDevice result copied to host memory (rowBytes * height bytes).
    Status download(const DeviceImage& img, float* dst)
    {
        if (!img.data || !dst)
            return Status::InvalidArgument;
        downloadPieces(dst, img.data, img.rowBytes * (size_t)img.height);
        return b_.ok() ? Status::Ok : Status::DeviceError;
    }

    // Codec::diagnose: the device copy of file (this pipeline's last upload) against the host bytes,
    // by download and through a kernel, and the chunk table.
    Diagnosis diagnose(const uint8_t* file, size_t size, int nchannels)
    {
        Diagnosis d;
        d.bytes = size;
        Plan p;
        const Slot& fs = slots_[kFile];
        const Slot& cs = slots_[kChunks];
        if (!file || !fs.p || fs.cap < size || makePlan(file, size, nchannels, p) != Status::Ok || !b_.begin())
            return d;
        std::vector<uint8_t> back(size);
        downloadPieces(back.data(), fs.p, size);
        for (size_t i = 0; i < size; ++i)
            if (back[i] != file[i] && d.copyMismatches++ == 0)
                d.firstCopyMismatch = i;
        void* tmp = b_.alloc(size);
        if (tmp)
        {
            b_.forEach(size, CopyBytes{static_cast<const uint8_t*>(fs.p), static_cast<uint8_t*>(tmp)});
            downloadPieces(back.data(), tmp, size);
            b_.release(tmp);
            for (size_t i = 0; i < size; ++i)
                if (back[i] != file[i] && d.shaderMismatches++ == 0)
                    d.firstShaderMismatch = i;
        }
        const size_t tableBytes = p.chunks.size() * sizeof(ChunkInfo);
        if (cs.p && cs.cap >= tableBytes)
        {
            std::vector<uint8_t> table(tableBytes);
            b_.download(table.data(), cs.p, tableBytes);
            d.chunkTableMatches = std::memcmp(table.data(), p.chunks.data(), tableBytes) == 0;
        }
        d.ran = b_.ok() && tmp != nullptr;
        return d;
    }

    // Frees the device buffers (they are kept between decodes and only grow otherwise).
    void trim()
    {
        for (Slot& s : slots_)
        {
            b_.release(s.p);
            s = Slot{};
        }
    }

  private:
    // Everything up to and including ConvertPixels: on Ok, the floats are queued into dOut.
    Status run(const uint8_t* file, size_t size, int nchannels, Plan& p, const float*& dOutResult)
    {
        if (!b_.begin())
            return Status::DeviceError;
        Status st = makePlan(file, size, nchannels, p);
        if (st != Status::Ok)
            return st;
        const ExrGeom& g = p.geom;

        uint8_t* dFile = get<uint8_t>(kFile, size);
        ChunkInfo* dChunks = get<ChunkInfo>(kChunks, p.chunks.size());
        uint8_t* dScratch = get<uint8_t>(kScratch, p.scratchBytes);
        uint32_t* dFlag = get<uint32_t>(kFlag, 1);
        const size_t pixels = (size_t)g.width * g.height;
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
        if (!b_.ok())
            return Status::DeviceError;
        dOutResult = dOut;
        return Status::Ok;
    }

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
