// CheshireEXR: see exrHost.hpp. File layout per the OpenEXR file format specification: magic and
// version, attributes until an empty name, the chunk offset table (one uint64 per chunk), then per
// chunk an int32 scanline, an int32 packed size and the packed bytes.
#include "exrHost.hpp"

#include <algorithm>
#include <cstring>

namespace cheshire {
namespace exr {

namespace {

struct Reader
{
    const uint8_t* d;
    size_t n;
    size_t i = 0;
    bool fail = false;

    bool has(size_t k) const { return !fail && i + k <= n; }
    uint32_t u8()
    {
        if (!has(1))
        {
            fail = true;
            return 0;
        }
        return d[i++];
    }
    int32_t i32()
    {
        if (!has(4))
        {
            fail = true;
            return 0;
        }
        const uint32_t v = (uint32_t)d[i] | ((uint32_t)d[i + 1] << 8) | ((uint32_t)d[i + 2] << 16) | ((uint32_t)d[i + 3] << 24);
        i += 4;
        return (int32_t)v;
    }
    uint64_t u64()
    {
        const uint64_t lo = (uint32_t)i32();
        const uint64_t hi = (uint32_t)i32();
        return lo | (hi << 32);
    }
    // A NUL-terminated name of at most 255 bytes (long names allow 255, short 31).
    bool name(std::string& s)
    {
        s.clear();
        for (;;)
        {
            if (!has(1) || s.size() > 255)
            {
                fail = true;
                return false;
            }
            const char c = (char)d[i++];
            if (!c)
                return true;
            s.push_back(c);
        }
    }
};

}  // namespace

const char* statusName(Status s)
{
    switch (s)
    {
        case Status::Ok: return "ok";
        case Status::NotExr: return "not an EXR";
        case Status::Unsupported: return "unsupported EXR";
        case Status::Corrupt: return "corrupt EXR";
        case Status::NoDevice: return "no GPU device";
        case Status::DeviceError: return "GPU runtime error";
        case Status::InvalidArgument: return "invalid argument";
    }
    return "?";
}

const std::string* Header::findString(const std::string& name) const
{
    for (const auto& kv : strings)
        if (kv.first == name)
            return &kv.second;
    return nullptr;
}

int linesPerChunk(int compression)
{
    switch (compression)
    {
        case kNone:
        case kRle:
        case kZips: return 1;
        case kZip:
        case kPxr24: return 16;
        case kPiz:
        case kB44:
        case kB44a:
        case kDwaa: return 32;
        case kDwab: return 256;
    }
    return 0;
}

namespace {

// Reads everything up to the chunk offset table; r is left at the table.
Status parseHeader(Reader& r, Header& h)
{
    h = Header();
    if (r.n < 8 || r.d[0] != 0x76 || r.d[1] != 0x2f || r.d[2] != 0x31 || r.d[3] != 0x01)
        return Status::NotExr;
    r.i = 4;
    const uint32_t version = (uint32_t)r.i32();
    if ((version & 0xff) != 2)
        return Status::Unsupported;
    h.tiled = (version & 0x200) != 0;
    h.deep = (version & 0x800) != 0;
    h.multipart = (version & 0x1000) != 0;
    if (h.multipart)
        return Status::Unsupported;  // the header layout differs; nothing here writes one

    bool haveChannels = false, haveCompression = false, haveData = false, haveDisplay = false;
    int dw[4] = {0, 0, 0, 0}, dsp[4] = {0, 0, 0, 0};
    std::string name, type;
    for (;;)
    {
        if (!r.name(name))
            return Status::NotExr;
        if (name.empty())
            break;
        if (!r.name(type))
            return Status::NotExr;
        const int32_t len = r.i32();
        if (r.fail || len < 0 || !r.has((size_t)len))
            return Status::NotExr;
        const size_t end = r.i + (size_t)len;
        if (name == "channels" && type == "chlist")
        {
            haveChannels = true;
            while (r.i < end)
            {
                Channel c;
                if (!r.name(c.name))
                    return Status::Corrupt;
                if (c.name.empty())
                    break;
                c.type = r.i32();
                r.u8();  // pLinear
                r.u8();
                r.u8();
                r.u8();  // reserved
                c.xSampling = r.i32();
                c.ySampling = r.i32();
                if (r.fail || r.i > end)
                    return Status::Corrupt;
                h.channels.push_back(c);
            }
        }
        else if (name == "compression" && type == "compression" && len == 1)
        {
            haveCompression = true;
            h.compression = (int)r.u8();
        }
        else if (name == "dataWindow" && type == "box2i" && len == 16)
        {
            haveData = true;
            for (int k = 0; k < 4; ++k)
                dw[k] = r.i32();
        }
        else if (name == "displayWindow" && type == "box2i" && len == 16)
        {
            haveDisplay = true;
            for (int k = 0; k < 4; ++k)
                dsp[k] = r.i32();
        }
        else if (name == "lineOrder" && type == "lineOrder" && len == 1)
            h.lineOrder = (int)r.u8();
        else if (type == "string")
            h.strings.emplace_back(name, std::string((const char*)r.d + r.i, (size_t)len));
        r.i = end;
    }
    if (r.fail || !haveChannels || !haveCompression || !haveData || !haveDisplay)
        return Status::NotExr;
    // OpenEXR keeps channels in a map ordered by name; chunks store them in that order
    std::sort(h.channels.begin(), h.channels.end(),
              [](const Channel& a, const Channel& b) { return std::strcmp(a.name.c_str(), b.name.c_str()) < 0; });
    const int64_t w = (int64_t)dw[2] - dw[0] + 1, ht = (int64_t)dw[3] - dw[1] + 1;
    if (w < 1 || ht < 1 || w > (1 << 24) || ht > (1 << 24))
        return Status::Corrupt;
    h.width = (int)w;
    h.height = (int)ht;
    h.dataMinX = dw[0];
    h.dataMinY = dw[1];
    h.displayEqualsData = std::memcmp(dw, dsp, sizeof(dw)) == 0;
    return Status::Ok;
}

}  // namespace

int Header::chunkCount() const
{
    const int lines = linesPerChunk(compression);
    return lines > 0 ? (height + lines - 1) / lines : 0;
}

Status readHeader(const uint8_t* file, size_t size, Header& header)
{
    if (!file)
        return Status::InvalidArgument;
    Reader r{file, size};
    return parseHeader(r, header);
}

Status makePlan(const uint8_t* file, size_t size, int nchannels, Plan& p)
{
    if (!file || (nchannels != 1 && nchannels != 3 && nchannels != 4))
        return Status::InvalidArgument;
    p = Plan();
    Reader r{file, size};
    Status st = parseHeader(r, p.header);
    if (st != Status::Ok)
        return st;
    const Header& h = p.header;
    if (h.tiled || h.deep || !h.displayEqualsData || h.dataMinX != 0 || h.dataMinY != 0)
        return Status::Unsupported;
    if (h.compression != kNone && h.compression != kRle && h.compression != kZips && h.compression != kZip)
        return Status::Unsupported;
    const int nch = (int)h.channels.size();
    if (nch < 1 || nch > kMaxChannels)
        return Status::Unsupported;
    for (const Channel& c : h.channels)
        if ((c.type != kHalf && c.type != kFloat) || c.xSampling != 1 || c.ySampling != 1)
            return Status::Unsupported;

    // the channel selection Cheshire's direct reader makes
    ExrGeom& g = p.geom;
    std::memset(&g, 0, sizeof(g));
    g.nOut = nchannels;
    if (nchannels == 1)
    {
        if (nch != 1)
            return Status::Unsupported;
        g.outChannel[0] = 0;
    }
    else
    {
        const char* want[4] = {"R", "G", "B", "A"};
        int found[4] = {-1, -1, -1, -1};
        for (int k = 0; k < 4; ++k)
            for (int c = 0; c < nch; ++c)
                if (h.channels[c].name == want[k])
                    found[k] = c;
        const bool hasA = found[3] >= 0;
        if (found[0] < 0 || found[1] < 0 || found[2] < 0 || nch != (hasA ? 4 : 3) || (nchannels == 4 && !hasA))
            return Status::Unsupported;
        for (int k = 0; k < nchannels; ++k)
            g.outChannel[k] = found[k];
    }

    g.width = h.width;
    g.height = h.height;
    g.linesPerChunk = linesPerChunk(h.compression);
    uint32_t off = 0;
    for (int c = 0; c < nch; ++c)
    {
        g.chanSize[c] = h.channels[c].type == kHalf ? 2 : 4;
        g.chanOffset[c] = off;
        off += (uint32_t)g.chanSize[c] * (uint32_t)g.width;
    }
    g.lineBytes = off;
    g.numChunks = (uint32_t)((h.height + g.linesPerChunk - 1) / g.linesPerChunk);
    if ((uint64_t)g.lineBytes * g.linesPerChunk > 0x7fffffffu)
        return Status::Unsupported;

    // the offset table, then each chunk's leader
    if (!r.has((size_t)g.numChunks * 8))
        return Status::NotExr;
    std::vector<uint64_t> offsets(g.numChunks);
    for (uint32_t k = 0; k < g.numChunks; ++k)
        offsets[k] = r.u64();
    p.chunks.assign(g.numChunks, ChunkInfo{});
    std::vector<uint8_t> seen(g.numChunks, 0);
    uint64_t scratch = 0;
    for (uint32_t k = 0; k < g.numChunks; ++k)
    {
        const uint64_t at = offsets[k];
        if (at < r.i || at > size || size - at < 8)
            return Status::Corrupt;  // OpenEXR would try to reconstruct the table; leave that to it
        Reader cr{file, size, (size_t)at};
        const int32_t y = cr.i32();
        const int32_t packed = cr.i32();
        if (y < 0 || y >= h.height || y % g.linesPerChunk != 0)
            return Status::Corrupt;
        const uint32_t idx = (uint32_t)(y / g.linesPerChunk);
        const uint32_t lines = (uint32_t)std::min(g.linesPerChunk, h.height - y);
        const uint32_t unpacked = lines * g.lineBytes;
        if (seen[idx] || packed < 0 || (uint32_t)packed > unpacked || (uint64_t)packed > size - at - 8)
            return Status::Corrupt;
        seen[idx] = 1;
        ChunkInfo& c = p.chunks[idx];
        c.fileOffset = at + 8;
        c.packed = (uint32_t)packed;
        c.unpacked = unpacked;
        c.lines = lines;
        if ((uint32_t)packed == unpacked)
            c.mode = kModeRaw;
        else if (h.compression == kNone)
            return Status::Corrupt;
        else
        {
            c.mode = h.compression == kRle ? kModeRle : kModeZip;
            c.scratchOffset = scratch;
            scratch += unpacked;
        }
        p.stats.packedBytes += (uint32_t)packed;
        p.stats.unpackedBytes += unpacked;
        p.stats.rawChunks += c.mode == kModeRaw;
    }
    p.scratchBytes = scratch;
    p.stats.chunks = g.numChunks;
    return Status::Ok;
}

}  // namespace exr
}  // namespace cheshire
