// CheshireEXR: host-side parsing - the header, the chunk offset table and each chunk's leader - into
// the tables the stages read. Plain C++.
#pragma once

#include "exrCodec.hpp"
#include "exrStages.hpp"

#include <vector>

namespace cheshire {
namespace exr {

int linesPerChunk(int compression);

// The decode plan for one file and one request (nchannels as Codec::decode takes it): geometry,
// one ChunkInfo per chunk in scanline order, and the scratch the compressed chunks need.
struct Plan
{
    Header header;
    ExrGeom geom;
    std::vector<ChunkInfo> chunks;
    uint64_t scratchBytes = 0;
    DecodeStats stats;
};

Status makePlan(const uint8_t* file, size_t size, int nchannels, Plan& plan);

}  // namespace exr
}  // namespace cheshire
