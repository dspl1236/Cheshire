// CheshireEXR: the decode stages as functors over plain structs, run as kernels by the device
// backend and as loops by the host backend (cheshiregpu/), like CheshireJPG's.
//
//   InflateChunks  one thread per chunk: inflate (ZIP, ZIPS) or run-length decode (RLE) into the
//                  chunk's scratch slice, check it the way zlib and OpenEXR do, undo the predictor.
//                  Chunks stored raw (packed size == unpacked size) need nothing.
//   ConvertPixels  one thread per pixel: for each requested channel, gather its bytes from the
//                  chunk - through the byte interleave for compressed chunks - and write the float.
//
// The interleave (internal_zip.c interleave()) is never materialised: output byte k of a chunk is
// read straight from where the reordering would have put it.
#pragma once

#include "exrTypes.hpp"

namespace cheshire {
namespace exr {

constexpr int kMaxChannels = 4;

enum ChunkMode : uint32_t
{
    kModeRaw = 0,
    kModeZip = 1,
    kModeRle = 2,
};

struct ChunkInfo
{
    uint64_t fileOffset;     // first byte of the chunk's pixel data in the file
    uint64_t scratchOffset;  // where its decompressed bytes go (compressed chunks only)
    uint32_t packed, unpacked;
    uint32_t mode;
    uint32_t lines;
};

struct ExrGeom
{
    int width, height;
    int linesPerChunk;
    uint32_t numChunks;
    uint32_t lineBytes;  // one scanline of every channel
    int nOut;            // floats per output pixel
    int outChannel[kMaxChannels];  // index into the sorted channel list, per output channel
    uint32_t chanOffset[kMaxChannels];  // byte offset of each channel within a scanline
    int chanSize[kMaxChannels];         // 2 (half) or 4 (float)
};

struct InflateChunks
{
    ExrGeom g;
    const uint8_t* file;
    const ChunkInfo* chunks;
    uint8_t* scratch;
    uint32_t* error;  // set to 1 + the chunk index of a failing chunk (any one)

    CHESHIRE_EXR_HD void operator()(size_t i) const
    {
        const ChunkInfo c = chunks[i];
        if (c.mode == kModeRaw)
            return;
        uint8_t* dst = scratch + c.scratchOffset;
        bool ok;
        if (c.mode == kModeZip)
        {
            InflateState s;
            ok = zlibInflate(file + c.fileOffset, c.packed, dst, c.unpacked, s) == kInflateOk;
        }
        else
            ok = rleDecode(file + c.fileOffset, c.packed, dst, c.unpacked);
        if (!ok)
        {
            *error = (uint32_t)i + 1;
            return;
        }
        unpredict(dst, c.unpacked);
    }
};

struct ConvertPixels
{
    ExrGeom g;
    const uint8_t* file;
    const ChunkInfo* chunks;
    const uint8_t* scratch;
    float* out;

    CHESHIRE_EXR_HD void operator()(size_t i) const
    {
        const int x = (int)(i % (size_t)g.width);
        const int y = (int)(i / (size_t)g.width);
        const ChunkInfo& c = chunks[y / g.linesPerChunk];
        const uint32_t line = (uint32_t)(y % g.linesPerChunk);
        const bool interleaved = c.mode != kModeRaw;
        const uint8_t* base = interleaved ? scratch + c.scratchOffset : file + c.fileOffset;
        float* o = out + i * (size_t)g.nOut;
        for (int k = 0; k < g.nOut; ++k)
        {
            const int ch = g.outChannel[k];
            const int size = g.chanSize[ch];
            const uint32_t at = line * g.lineBytes + g.chanOffset[ch] + (uint32_t)x * (uint32_t)size;
            uint32_t v = 0;
            for (int b = 0; b < size; ++b)
            {
                const uint32_t src = interleaved ? interleavedSource(at + b, c.unpacked) : at + b;
                v |= (uint32_t)base[src] << (8 * b);  // little-endian in the file
            }
            const uint32_t bits = size == 2 ? halfToFloatBits((uint16_t)v) : v;
            float f;
            __builtin_memcpy(&f, &bits, 4);
            o[k] = f;
        }
    }
};

}  // namespace exr
}  // namespace cheshire
