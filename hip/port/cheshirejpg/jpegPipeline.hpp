// CheshireJPG: the decode and encode sequences, written once over a backend.
//
// A backend provides alloc/release, upload/download, zero, and forEach(n, functor). The GPU
// backend (jpegGPU.cu) launches one thread per index; the CPU backend (jpegCpuBackend.hpp) loops.
// Both run the functors of jpegStages.hpp in exactly this order, which is what lets the host check
// in hip/tests/cheshirejpg vouch for the device path's arithmetic and control flow without a GPU.
#pragma once

#include "jpegCodec.hpp"
#include "jpegHost.hpp"
#include "jpegStages.hpp"

#include <cstring>
#include <utility>
#include <vector>

namespace cheshire {
namespace jpeg {

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

    Status decode(const uint8_t* data, size_t size, Image& out, DecodeStats* stats)
    {
        ParsedJpeg p;
        Status st = parseJpeg(data, size, p);
        if (st != Status::Ok)
            return st;
        DecodeGeom g;
        st = makeDecodeGeom(p, g);
        if (st != Status::Ok)
            return st;
        Unstuffed u;
        st = unstuff(data, size, p.entropyBegin, u);
        if (st != Status::Ok)
            return st;
        const uint32_t nSeg = (uint32_t)u.segBegin.size();
        const uint32_t expectSeg = g.restartInterval ? (g.totalMcus + g.restartInterval - 1) / g.restartInterval : 1;
        if (nSeg != expectSeg)
            return Status::Corrupt;

        Tables T;
        std::memset(&T, 0, sizeof(T));
        buildZigzag(T.zz);
        std::memcpy(T.quant, p.quant, sizeof(T.quant));
        for (int t = 0; t < 4; ++t)
        {
            if (p.dc[t].present)
                buildHuffDecode(p.dc[t], T.dc[t]);
            if (p.ac[t].present)
                buildHuffDecode(p.ac[t], T.ac[t]);
        }

        std::vector<Subseq> subseqs;
        std::vector<uint32_t> segFirst;
        for (uint32_t s = 0; s < nSeg; ++s)
        {
            segFirst.push_back((uint32_t)subseqs.size());
            const uint32_t b0 = u.segBegin[s], e = u.segEnd[s];
            uint32_t b = b0;
            do
            {
                Subseq q;
                q.begin = b;
                q.end = e - b > kSubseqBits ? b + kSubseqBits : e;
                q.segEnd = e;
                q.head = b == b0;
                subseqs.push_back(q);
                b = q.end;
            } while (b < e);
        }
        segFirst.push_back((uint32_t)subseqs.size());
        const size_t nSub = subseqs.size();

        Tables* dT = get<Tables>(kTables, 1);
        uint32_t* dWords = get<uint32_t>(kWords, u.words.size());
        Subseq* dSub = get<Subseq>(kSubseq, nSub);
        uint32_t* dSegFirst = get<uint32_t>(kSegFirst, segFirst.size());
        SubseqRun* runA = get<SubseqRun>(kRunA, nSub);
        SubseqRun* runB = get<SubseqRun>(kRunB, nSub);
        uint32_t* dFlag = get<uint32_t>(kFlag, 1);
        if (!b_.ok())
            return Status::DeviceError;
        b_.upload(dT, &T, sizeof(T));
        b_.upload(dWords, u.words.data(), u.words.size() * sizeof(uint32_t));
        b_.upload(dSub, subseqs.data(), nSub * sizeof(Subseq));
        b_.upload(dSegFirst, segFirst.data(), segFirst.size() * sizeof(uint32_t));

        const size_t nChunks = (nSub + kScanChunk - 1) / kScanChunk;
        uint8_t* dFlags = get<uint8_t>(kSyncFlags, nSub);
        RunChunkAgg* dAgg = get<RunChunkAgg>(kAgg, nChunks);
        SubseqPrefix* dPrefix = get<SubseqPrefix>(kPrefix, nSub);
        if (!b_.ok())
            return Status::DeviceError;
        auto scanRuns = [&](const SubseqRun* runs) {
            b_.forEach(nChunks, RunScanChunk{g, dSub, runs, dAgg, nSub});
            b_.forEach(1, RunScanSerial{dAgg, nChunks});
            b_.forEach(nChunks, RunScanApply{g, dSub, runs, dAgg, dPrefix, nSub});
        };

        // Synchronisation rounds until one changes nothing. On the 88 camera JPEGs of the host
        // check that is 3 to 12 rounds (mean 5.1) and 2.23 decodes of each subsequence in all.
        b_.forEach(nSub, DecodeInitial{g, dT, dWords, dSub, runA});
        uint32_t rounds = 0;
        for (;;)
        {
            uint32_t changed = 0;
            b_.zero(dFlag, sizeof(uint32_t));
            b_.forEach(nSub, SyncPrepare{g, dSub, runA, runB, dFlags});
            b_.forEach(nSub, SyncWalk{g, dT, dWords, dSub, runA, dFlags, runB, dFlag, nSub});
            b_.download(&changed, dFlag, sizeof(uint32_t));
            std::swap(runA, runB);
            ++rounds;
            if (!b_.ok())
                return Status::DeviceError;
            if (!changed)
                break;
            if (rounds > nSub + 1)
                return Status::Corrupt;  // cannot happen: the chain from each head converges
        }
        scanRuns(runA);
        if (g.uniformTables && g.bpm > 1)
        {
            b_.forEach(nSub, ResolveBlockIndex{g, dPrefix, runA});
            scanRuns(runA);
        }

        uint32_t bad = 0;
        b_.zero(dFlag, sizeof(uint32_t));
        b_.forEach(nSeg, DecodeVerify{g, dSegFirst, runA, dPrefix, dFlag});
        b_.download(&bad, dFlag, sizeof(uint32_t));
        if (!b_.ok())
            return Status::DeviceError;
        if (bad)
            return Status::Corrupt;

        size_t planeBytes = 0;
        for (int c = 0; c < g.ncomp; ++c)
            planeBytes += (size_t)g.planeW[c] * g.planeH[c];
        const size_t outPitch = (size_t)g.width * g.outChannels;
        int16_t* dCoefs = get<int16_t>(kCoefs, (size_t)g.totalUnits * 64);
        uint8_t* dPlanes = get<uint8_t>(kPlanes, planeBytes);
        uint8_t* dOut = get<uint8_t>(kOut, outPitch * g.height);
        if (!b_.ok())
            return Status::DeviceError;
        b_.zero(dCoefs, (size_t)g.totalUnits * 64 * sizeof(int16_t));
        b_.zero(dFlag, sizeof(uint32_t));
        b_.forEach(nSub, DecodeWrite{g, dT, dWords, dSub, runA, dPrefix, dCoefs});
        b_.forEach(g.totalUnits, DecodeIdct{g, dT, dCoefs, dPlanes, dFlag});
        b_.download(&bad, dFlag, sizeof(uint32_t));
        if (!b_.ok())
            return Status::DeviceError;
        if (bad)
            return Status::Corrupt;
        b_.forEach((size_t)g.width * g.height, DecodeColor{g, dPlanes, dOut, outPitch});

        out.width = g.width;
        out.height = g.height;
        out.channels = g.outChannels;
        out.pixels.resize(outPitch * g.height);
        b_.download(out.pixels.data(), dOut, out.pixels.size());
        if (!b_.ok())
            return Status::DeviceError;
        if (stats)
        {
            stats->subsequences = (uint32_t)nSub;
            stats->syncRounds = rounds;
            stats->restartSegments = nSeg;
        }
        return Status::Ok;
    }

    // Coefficients as the decoder produced them, for the host check (totalUnits * 64, in MCU
    // order, natural order within a block). Valid after a successful decode().
    void downloadCoefs(std::vector<int16_t>& c, size_t count)
    {
        c.resize(count);
        b_.download(c.data(), slots_[kCoefs].p, count * sizeof(int16_t));
    }

    Status encode(const uint8_t* pixels,
                  int width,
                  int height,
                  int channels,
                  size_t rowBytes,
                  const EncodeOptions& o,
                  std::vector<uint8_t>& out)
    {
        if (!pixels)
            return Status::InvalidArgument;
        EncodeGeom g;
        Status st = makeEncodeGeom(width, height, channels, rowBytes, o, g);
        if (st != Status::Ok)
            return st;

        uint16_t quant[2][64];
        qualityTables(o.quality, quant[0], quant[1]);
        Tables T;
        std::memset(&T, 0, sizeof(T));
        buildZigzag(T.zz);
        for (int t = 0; t < 2; ++t)
        {
            buildQuantEncode(quant[t], T.eq[t]);
            buildHuffEncode(stdBits(false, t), stdVals(false, t), T.edc[t]);
            buildHuffEncode(stdBits(true, t), stdVals(true, t), T.eac[t]);
        }

        const size_t inBytes = (size_t)(height - 1) * rowBytes + (size_t)width * channels;
        const size_t n = g.totalUnits;
        Tables* dT = get<Tables>(kTables, 1);
        uint8_t* dIn = get<uint8_t>(kIn, inBytes);
        uint8_t* dPlanes = get<uint8_t>(kPlanes, g.planeTotal);
        int16_t* dCoefs = get<int16_t>(kCoefs, n * 64);
        uint32_t* dLen = get<uint32_t>(kLens, n + 1);
        uint32_t* dSegBytes = get<uint32_t>(kSegBytes, (size_t)g.numSegments + 1);
        uint32_t* dFlag = get<uint32_t>(kFlag, 1);
        if (!b_.ok())
            return Status::DeviceError;
        b_.upload(dT, &T, sizeof(T));
        b_.upload(dIn, pixels, inBytes);

        b_.forEach(g.planeTotal, EncodePlanes{g, dIn, dPlanes});
        b_.forEach(n, EncodeBlocks{g, dT, dPlanes, dCoefs});
        b_.forEach(n, EncodeDummies{g, dCoefs});
        b_.forEach(n, EncodeBitLengths{g, dT, dCoefs, dLen});
        if (!scan(dLen, n, dFlag))
            return b_.ok() ? Status::Unsupported : Status::DeviceError;  // over 4 Gbit
        b_.forEach(g.numSegments, EncodeSegBytes{g, dLen, dSegBytes});
        if (!scan(dSegBytes, g.numSegments, dFlag))
            return b_.ok() ? Status::Unsupported : Status::DeviceError;
        uint32_t totalBytes = 0;
        b_.download(&totalBytes, dSegBytes + g.numSegments, sizeof(uint32_t));
        if (!b_.ok())
            return Status::DeviceError;

        const size_t nWords = ((size_t)totalBytes + 3) / 4 + 2;
        uint32_t* dWords = get<uint32_t>(kWords, nWords);
        const size_t nStuff = ((size_t)totalBytes + kStuffChunk - 1) / kStuffChunk;
        uint32_t* dFF = get<uint32_t>(kFF, nStuff + 1);
        if (!b_.ok())
            return Status::DeviceError;
        b_.zero(dWords, nWords * sizeof(uint32_t));
        b_.forEach(n, EncodeEmit{g, dT, dCoefs, dLen, dSegBytes, dWords});
        b_.forEach(g.numSegments, EncodePad{g, dLen, dSegBytes, dWords});
        b_.forEach(nStuff, StuffCount{dWords, totalBytes, dFF});
        if (!scan(dFF, nStuff, dFlag))
            return b_.ok() ? Status::Unsupported : Status::DeviceError;
        uint32_t totalFF = 0;
        b_.download(&totalFF, dFF + nStuff, sizeof(uint32_t));
        const size_t dataBytes = (size_t)totalBytes + totalFF + 2 * ((size_t)g.numSegments - 1);
        uint8_t* dOut = get<uint8_t>(kOut, dataBytes);
        if (!b_.ok())
            return Status::DeviceError;
        b_.forEach(nStuff, StuffWrite{g, dWords, totalBytes, dFF, dSegBytes, dOut});

        out.clear();
        writeHeaders(out, g, quant);
        const size_t head = out.size();
        out.resize(head + dataBytes + 2);
        b_.download(out.data() + head, dOut, dataBytes);
        out[head + dataBytes] = 0xFF;
        out[head + dataBytes + 1] = 0xD9;
        return b_.ok() ? Status::Ok : Status::DeviceError;
    }

  private:
    enum SlotId
    {
        kTables,
        kWords,
        kSubseq,
        kSegFirst,
        kRunA,
        kRunB,
        kSyncFlags,
        kAgg,
        kPrefix,
        kFlag,
        kCoefs,
        kPlanes,
        kOut,
        kIn,
        kLens,
        kSegBytes,
        kFF,
        kScanChunks,
        kSlotCount
    };

    struct Slot
    {
        void* p = nullptr;
        size_t cap = 0;
    };

    // Grow-only buffers, reused across calls.
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

    // Exclusive scan of data[0..n) in place, total to data[n]. False on overflow of 32 bits.
    bool scan(uint32_t* data, size_t n, uint32_t* dFlag)
    {
        const size_t nChunks = (n + kScanChunk - 1) / kScanChunk;
        uint32_t* chunks = get<uint32_t>(kScanChunks, nChunks);
        if (!b_.ok())
            return false;
        b_.zero(dFlag, sizeof(uint32_t));
        b_.forEach(nChunks, ScanChunkSum{data, chunks, n});
        b_.forEach(1, ScanChunkSerial{chunks, nChunks, data + n, dFlag});
        b_.forEach(nChunks, ScanChunkApply{data, chunks, n});
        uint32_t overflow = 0;
        b_.download(&overflow, dFlag, sizeof(uint32_t));
        return b_.ok() && !overflow;
    }

    Backend& b_;
    Slot slots_[kSlotCount];
};

}  // namespace jpeg
}  // namespace cheshire
